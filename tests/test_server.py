"""HTTP-сервер чата: главная страница, профиль и защита от регрессии с «/?»."""
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest


@pytest.fixture
def base_url():
    import server
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def get(url):
    with urllib.request.urlopen(url) as r:
        return r.status, r.read().decode("utf-8")


def test_index_also_served_with_query_string(base_url):
    # регрессия: при сломанном JS форма уходила GET-ом на «/?», и сервер отвечал {"error": "not found"}
    for path in ("/", "/?", "/?q=test"):
        status, body = get(base_url + path)
        assert status == 200 and "Своё Вино" in body


def test_profile_endpoint(base_url):
    status, body = get(base_url + "/api/profile/test_user")
    assert status == 200 and json.loads(body) == {"profile": {}, "favorites": [], "hidden": []}


def post(url, data):
    req = urllib.request.Request(url, data=json.dumps(data).encode(), method="POST")
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read().decode("utf-8"))


def test_pages_and_styles_served(base_url):
    for path, marker in (("/profile", "Вкусовой профиль"), ("/favorites", "Избранное"), ("/style.css", "--wine")):
        status, body = get(base_url + path)
        assert status == 200 and marker in body


def test_favorites_dedup_and_remove_via_api(base_url):
    for verdict in ("saved", "liked", "saved"):
        post(base_url + "/api/feedback", {"user_id": "srv", "wine_id": "135", "verdict": verdict})
    favs = json.loads(get(base_url + "/api/profile/srv")[1])["favorites"]
    assert [w["id"] for w in favs] == ["135"] and favs[0]["saved"] and favs[0]["liked"]
    assert post(base_url + "/api/favorites/remove", {"user_id": "srv", "wine_id": "135"})["favorites"] == []


@pytest.mark.parametrize("page", ["index.html", "profile.html"])
def test_page_script_has_no_raw_newline_in_regex(page):
    # регрессия: перевод строки внутри /.../ в linkify ронял весь скрипт страницы
    html = (Path(__file__).resolve().parent.parent / "static" / page).read_text(encoding="utf-8")
    script = html.split("<script>")[1].split("</script>")[0]
    for line in script.splitlines():
        if ".replace(/" in line:
            assert line.count("/g") >= line.count(".replace(/"), f"незакрытый regex: {line[:80]}"


@pytest.mark.parametrize("page", ["index.html", "profile.html"])
def test_pages_have_no_emoji(page):
    from sommelier.agent import EMOJI
    html = (Path(__file__).resolve().parent.parent / "static" / page).read_text(encoding="utf-8")
    assert not EMOJI.search(html)


def test_hidden_wine_can_be_restored(base_url):
    post(base_url + "/api/feedback", {"user_id": "hid", "wine_id": "135", "verdict": "disliked"})
    assert [w["id"] for w in json.loads(get(base_url + "/api/profile/hid")[1])["hidden"]] == ["135"]
    post(base_url + "/api/profile/remove", {"user_id": "hid", "key": "disliked", "value": "135"})
    assert json.loads(get(base_url + "/api/profile/hid")[1])["hidden"] == []


def test_stream_endpoint_reports_error_as_event(base_url, monkeypatch):
    import server

    def boom():
        raise RuntimeError("нет ключа LLM")
    monkeypatch.setattr(server, "get_graph", boom)
    req = urllib.request.Request(base_url + "/api/chat/stream", data=json.dumps({"message": "вино"}).encode(), method="POST")
    with urllib.request.urlopen(req) as r:
        body = r.read().decode("utf-8")
    assert r.headers["Content-Type"].startswith("text/event-stream") and "Не удалось получить ответ" in body and "нет ключа" not in body


def test_message_length_and_rate_limit():
    import server
    assert server.check_chat_request("  ", "k")[0] == 400
    assert server.check_chat_request("в" * 1001, "k")[0] == 400
    for i in range(server.RATE_LIMIT[0]):
        assert server.check_chat_request("Хочу вино", "rl", now=100.0 + i) is None
    code, msg = server.check_chat_request("Хочу вино", "rl", now=110.0)
    assert code == 429 and "минуту" in msg
    assert server.check_chat_request("Хочу вино", "rl", now=200.0) is None  # окно прошло


def test_too_long_message_rejected_over_http(base_url):
    req = urllib.request.Request(base_url + "/api/chat", data=json.dumps({"message": "х" * 1500}).encode(), method="POST")
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req)
    assert e.value.code == 400 and "1000" in json.loads(e.value.read().decode("utf-8"))["error"]


def test_user_facing_errors_hide_exception_details():
    import server

    class RateLimitError(Exception):
        pass
    assert "перегружен" in server.user_error(RateLimitError("Error code: 429 - Provider returned error"))
    assert server.user_error(ValueError("secret stack trace")) == "Не удалось получить ответ. Попробуйте ещё раз."


def test_delete_my_data(base_url):
    post(base_url + "/api/feedback", {"user_id": "wipe", "wine_id": "135", "verdict": "saved"})
    assert post(base_url + "/api/profile/delete", {"user_id": "wipe"}) == {"deleted": True}
    assert json.loads(get(base_url + "/api/profile/wipe")[1]) == {"profile": {}, "favorites": [], "hidden": []}
    assert post(base_url + "/api/profile/delete", {"user_id": "wipe"}) == {"deleted": False}


@pytest.mark.parametrize("raw", ["Хочу вино".encode("cp1251"), b"{not json", b"[1, 2]"])
def test_malformed_body_gets_400_not_dropped_connection(base_url, raw):
    # регрессия: тело в cp1251 (curl из консоли Windows) роняло обработчик, клиент получал обрыв без ответа
    req = urllib.request.Request(base_url + "/api/chat", data=raw, method="POST")
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req)
    assert e.value.code == 400 and "JSON" in json.loads(e.value.read().decode("utf-8"))["error"]


def test_head_request_for_health_checks(base_url):
    req = urllib.request.Request(base_url + "/", method="HEAD")
    with urllib.request.urlopen(req) as r:
        assert r.status == 200 and r.read() == b""


def test_oversized_body_rejected_without_reading(base_url):
    import socket
    from urllib.parse import urlparse
    u = urlparse(base_url)
    with socket.create_connection((u.hostname, u.port)) as sock:  # заявляем 1 ГБ, но не шлём — сервер не должен ждать
        sock.sendall(b"POST /api/chat HTTP/1.1\r\nHost: x\r\nContent-Length: 1073741824\r\n\r\n")
        sock.settimeout(5)
        assert b" 413 " in sock.recv(200)


def test_negative_content_length_gets_400(base_url):
    import socket
    from urllib.parse import urlparse
    u = urlparse(base_url)
    with socket.create_connection((u.hostname, u.port)) as sock:
        sock.sendall(b"POST /api/chat HTTP/1.1\r\nHost: x\r\nContent-Length: -5\r\n\r\n")
        sock.settimeout(5)
        assert b" 400 " in sock.recv(200)


def test_second_message_while_answering_gets_409(base_url):
    # регрессия: двойной Enter отправлял второй запрос, и два хода параллельно писали одну историю диалога
    import server
    lock = server.thread_lock("busy_user:t1")
    lock.acquire()
    try:
        req = urllib.request.Request(base_url + "/api/chat", method="POST",
                                     data=json.dumps({"user_id": "busy_user", "thread_id": "t1", "message": "вино"}).encode())
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req)
        assert e.value.code == 409 and "ещё отвечает" in json.loads(e.value.read().decode("utf-8"))["error"]
    finally:
        lock.release()


def test_without_api_key_chat_explains_instead_of_crashing(base_url, monkeypatch):
    # проверяющий без ключа: сервер стартует, а чат отвечает понятным текстом, а не падает
    import server
    from openai import OpenAIError
    monkeypatch.setattr(server, "GRAPH", None)
    monkeypatch.setattr(server, "build_graph", lambda **kw: (_ for _ in ()).throw(OpenAIError("Missing credentials. Please pass an `api_key`")))
    req = urllib.request.Request(base_url + "/api/chat", method="POST",
                                 data=json.dumps({"user_id": "nokey", "message": "Хочу вино"}).encode())
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req)
    assert e.value.code == 503 and "не настроен доступ" in json.loads(e.value.read().decode("utf-8"))["error"]
