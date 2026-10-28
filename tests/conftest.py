import sys
from pathlib import Path

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def pytest_sessionstart(session):
    """Каталог не хранится в репозитории (тексты принадлежат vino-svoe.ru): без него — понятная подсказка,
    а не сотня одинаковых ошибок."""
    from sommelier.catalog import CATALOG_PATH
    if not CATALOG_PATH.exists():
        pytest.exit("Нет data/catalog.json. Скачайте каталог и статьи с vino-svoe.ru:\n"
                    "  python scripts/fetch_catalog.py\n  python scripts/fetch_articles.py\n"
                    "(несколько минут, публичный API сайта), затем снова python -m pytest", returncode=4)


class ScriptedLLM(BaseChatModel):
    """Фейковая LLM для офлайн-тестов графа: отдаёт заранее заданные AIMessage по очереди
    и запоминает, что ей прислали (чтобы проверить инъекцию профиля и фидбек critic)."""
    script: list = []
    seen: list = []

    @property
    def _llm_type(self):
        return "scripted"

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        self.seen.append(messages)
        msg = self.script.pop(0) if self.script else AIMessage(content="no")
        return ChatResult(generations=[ChatGeneration(message=msg)])


@pytest.fixture(autouse=True)
def tmp_profiles(tmp_path, monkeypatch):
    from sommelier import memory, tracing
    monkeypatch.setattr(memory, "PROFILES_DIR", tmp_path / "profiles")
    monkeypatch.setattr(tracing, "TRACES_DIR", tmp_path / "traces")
    monkeypatch.setenv("SOMMELIER_PREFETCH", "0")  # большинство тестов сценарно ведут модель сами; prefetch — в test_prefetch
    yield


@pytest.fixture
def scripted():
    def make(*msgs):
        return ScriptedLLM(script=list(msgs), seen=[])
    return make
