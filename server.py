"""Чат-прототип: python server.py: http://localhost:8000

LLM настраивается как в курсе: OPENAI_API_KEY (+ OPENAI_BASE_URL, MODEL_NAME)
или LLM_MODEL='provider:model' для init_chat_model.
"""
import json
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import logging
import socket
import threading
import time
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from sommelier import catalog, memory, tracing
from sommelier.agent import build_graph, chat, chat_stream, get_topic_llm, load_env, mask_pii

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
STATIC = Path(__file__).parent / "static"
GRAPH = None


def get_graph():
    global GRAPH
    if GRAPH is None:
        GRAPH = build_graph(topic_llm=get_topic_llm())
        from sommelier.prefetch import _known_words
        _known_words()  # словарь для опечаток (~0.5 с) строится сразу, а не на первом сообщении клиента
    return GRAPH


MAX_BODY_BYTES = 64 * 1024
MAX_MESSAGE_CHARS = 1000      # как у виджета на vino-svoe.ru: длинные «прокладки» с инструкциями не принимаем
RATE_LIMIT = (15, 60.0)       # не больше 15 сообщений за 60 с с одного пользователя/IP — защита от злоупотреблений и счёта за LLM
_RECENT: dict = {}
_RATE_LOCK = threading.Lock()


def check_chat_request(message, client_key: str, now: float | None = None):
    """None, если запрос можно обрабатывать; иначе (HTTP-код, понятное пользователю сообщение)."""
    if not isinstance(message, str) or not message.strip():
        return 400, "Пустое сообщение."
    if len(message) > MAX_MESSAGE_CHARS:
        return 400, f"Сообщение слишком длинное: сократите его до {MAX_MESSAGE_CHARS} символов."
    now = time.monotonic() if now is None else now
    limit, window = RATE_LIMIT
    with _RATE_LOCK:
        recent = [t for t in _RECENT.get(client_key, []) if now - t < window]
        if len(recent) >= limit:
            _RECENT[client_key] = recent
            return 429, "Слишком много сообщений подряд. Подождите минуту и попробуйте снова."
        _RECENT[client_key] = recent + [now]
    return None


_THREAD_LOCKS: dict = {}
_THREAD_LOCKS_GUARD = threading.Lock()
BUSY_REPLY = "Сомелье ещё отвечает на предыдущее сообщение. Дождитесь ответа и отправьте снова."


def thread_lock(key: str) -> threading.Lock:
    """Один ход за раз в одном диалоге: двойной Enter или кнопка сценария во время ответа отправляли второй запрос,
    и два хода параллельно дописывали одну историю в MemorySaver (порядок сообщений перемешивался)."""
    with _THREAD_LOCKS_GUARD:
        if len(_THREAD_LOCKS) > 5000:  # не копим замки всех когда-либо открытых диалогов
            for k in [k for k, lk in _THREAD_LOCKS.items() if not lk.locked()]:
                del _THREAD_LOCKS[k]
        return _THREAD_LOCKS.setdefault(key, threading.Lock())


def user_error(e: Exception) -> str:
    """Клиенту — понятный текст без деталей исключения (они остаются в логе сервера)."""
    name = type(e).__name__
    if "RateLimit" in name or "429" in str(e):
        return "Сервис сейчас перегружен. Попробуйте ещё раз через минуту."
    if "Timeout" in name:
        return "Сомелье долго не отвечает. Попробуйте ещё раз."
    if "Authentication" in name or "api_key" in str(e).lower():
        return "Сомелье временно недоступен: не настроен доступ к модели."
    return "Не удалось получить ответ. Попробуйте ещё раз."


def cards_for(reply: str) -> list:
    ids = re.findall(r"vino-svoe\.ru/wines/([\w-]+)", reply)
    seen, out = set(), []
    for i in ids:
        w = catalog.get_wine(i)
        if w and i not in seen:
            seen.add(i)
            out.append(catalog.public_view(w))
    return out


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def do_HEAD(self):  # проверки живости (мониторинг, превью) шлют HEAD — раньше получали 501
        self.do_GET()

    def log_message(self, fmt, *args):
        logging.info(mask_pii(fmt % args))

    def do_GET(self):
        path = self.path.split("?", 1)[0]  # «/?» после случайной отправки формы — та же главная страница
        if path in ("/", "/index.html"):
            return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
        if path in ("/profile", "/favorites"):
            return self._send(200, (STATIC / "profile.html").read_bytes(), "text/html; charset=utf-8")
        if path == "/style.css":
            return self._send(200, (STATIC / "style.css").read_bytes(), "text/css; charset=utf-8")
        if path == "/api/metrics":  # p50/p95 задержки, срабатывания защит, медленные узлы — за сегодня или ?day=
            day = re.search(r"day=(\d{4}-\d{2}-\d{2})", self.path)
            summary = tracing.summarize(tracing.load(day.group(1) if day else None))
            # предохранитель основной модели прямо сейчас: open — провайдер сбоит, отвечает запасная
            summary["circuit"] = {k: b.state for k, b in getattr(GRAPH, "breakers", {}).items()}
            return self._send(200, summary)
        m = re.match(r"/api/profile/([\w-]+)$", path)
        if m:
            uid = m.group(1)
            return self._send(200, {"profile": memory.load_profile(uid), "favorites": memory.favorites(uid),
                                    "hidden": memory.hidden(uid)})
        self._send(404, {"error": "not found"})

    def _chat(self, body: dict, uid: str):
        """/api/chat — ответ целиком; /api/chat/stream — Server-Sent Events: этапы работы, затем проверенный ответ."""
        thread = body.get("thread_id", "web")
        if self.path == "/api/chat":
            try:
                reply = chat(get_graph(), body["message"], user_id=uid, thread_id=thread)
            except Exception as e:  # перегрузка провайдера, таймаут, нет ключа
                logging.exception("chat failed")
                return self._send(503, {"error": user_error(e)})
            return self._send(200, {"reply": reply, "wines": cards_for(reply)})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def emit(payload):
            self.wfile.write(("data: " + json.dumps(payload, ensure_ascii=False) + "\n\n").encode())
            self.wfile.flush()
        try:
            for kind, value in chat_stream(get_graph(), body["message"], user_id=uid, thread_id=thread):
                emit({"stage": value} if kind == "stage" else {"reply": value, "wines": cards_for(value)})
        except Exception as e:
            logging.exception("chat stream failed")
            emit({"error": user_error(e)})

    def do_POST(self):
        # битое тело (не UTF-8, не JSON, не объект) — понятный 400, а не оборванное соединение
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length < 0:  # rfile.read(-1) читал бы до закрытия соединения — поток бы завис
                raise ValueError("negative Content-Length")
            if length > MAX_BODY_BYTES:  # сообщения короче 1000 символов: огромное тело — злоупотребление
                return self._send(413, {"error": "Слишком большой запрос."})
            body = json.loads(self.rfile.read(length) or "{}")
            if not isinstance(body, dict):
                raise ValueError("body must be a JSON object")
        except (ValueError, UnicodeDecodeError):
            return self._send(400, {"error": "Некорректный запрос: ожидается JSON в UTF-8."})
        uid = re.sub(r"[^\w-]", "", body.get("user_id", "anon"))[:40] or "anon"
        if self.path in ("/api/chat", "/api/chat/stream"):
            problem = check_chat_request(body.get("message"), f"{uid}|{self.client_address[0]}")
            if problem:
                return self._send(problem[0], {"error": problem[1]})
            lock = thread_lock(f"{uid}:{body.get('thread_id', 'web')}")
            if not lock.acquire(blocking=False):
                return self._send(409, {"error": BUSY_REPLY})
            try:
                return self._chat(body, uid)
            finally:
                lock.release()
        if self.path == "/api/profile/delete":  # «Удалить мои данные» на странице профиля
            return self._send(200, {"deleted": memory.delete_profile(uid)})
        if self.path == "/api/profile/remove":  # страница «Профиль»: убрать предпочтение
            key, value = body.get("key"), body.get("value")
            if key not in memory.LIST_FIELDS - {"saved", "liked"} or not isinstance(value, str):
                return self._send(400, {"error": "bad key/value"})
            return self._send(200, {"profile": memory.remove_from_list(uid, key, value)})
        if self.path == "/api/favorites/remove":  # страница «Избранное»: убрать вино (и из избранного, и из понравившихся)
            wid = body.get("wine_id")
            for key in ("saved", "liked"):
                memory.remove_from_list(uid, key, wid)
            return self._send(200, {"favorites": memory.favorites(uid)})
        if self.path == "/api/feedback":  # кнопки в карточке вина — без LLM
            wid, verdict = body.get("wine_id"), body.get("verdict")
            if not catalog.get_wine(wid) or verdict not in ("liked", "disliked", "saved"):
                return self._send(400, {"error": "bad wine_id/verdict"})
            from sommelier.tools import record_feedback
            msg = record_feedback.invoke({"wine_id": wid, "verdict": verdict},
                                         config={"configurable": {"user_id": uid}})
            return self._send(200, {"ok": True, "message": msg})
        self._send(404, {"error": "not found"})


class ThreadingHTTPServerV6(ThreadingHTTPServer):
    address_family = socket.AF_INET6


def serve(port: int):
    """Только локальные адреса, но оба: браузер открывает http://localhost и на Windows сначала стучится в IPv6
    ::1. Без него каждое подключение ждало ~200 мс отказа и лишь потом шло на 127.0.0.1 (замер curl: 0.20 с
    против 0.002 с) — на каждом сообщении, странице и стилях."""
    # граф и словарь опечаток — до открытия сокетов: иначе первые запросы видели недостроенный граф,
    # а чат-запрос в эти секунды запускал бы вторую, параллельную сборку
    try:
        get_graph()
    except Exception as e:  # нет ключа: интерфейс всё равно открывается, чат объясняет, что модель не настроена
        print(f"Модель недоступна ({type(e).__name__}). Интерфейс откроется, но сомелье не ответит: "
              "заполните OPENAI_API_KEY в .env (см. .env.example). Офлайн-тесты ключа не требуют: python -m pytest",
              flush=True)  # без flush подсказка застревала в буфере, если вывод идёт не в терминал
    try:
        v6 = ThreadingHTTPServerV6(("::1", port), Handler)
        threading.Thread(target=v6.serve_forever, daemon=True).start()
    except OSError:  # IPv6 выключен в системе — работаем только по IPv4
        pass
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    import os
    load_env()  # PORT и ключ из .env — до чтения порта
    port = int(os.getenv("PORT", 8000))
    print(f"Своё Вино · ИИ-сомелье: http://localhost:{port}", flush=True)
    serve(port)
