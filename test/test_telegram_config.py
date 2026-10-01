from __future__ import annotations

import pytest
from pydantic import ValidationError

from qseek.plugins.telegram import TelegramAlert


def test_credentials_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QSEEK_TELEGRAM_BOT_TOKEN", "token-from-env")
    monkeypatch.setenv("QSEEK_TELEGRAM_CHAT_ID", "1234")

    alert = TelegramAlert.model_validate({"callback": "TelegramAlert"})
    assert alert.bot_token.get_secret_value() == "token-from-env"
    assert alert.chat_id == "1234"

    # The token is not written to the configuration, but loads again from the env.
    dump = alert.model_dump_json()
    assert "token-from-env" not in dump
    reloaded = TelegramAlert.model_validate_json(dump)
    assert reloaded.bot_token.get_secret_value() == "token-from-env"


def test_config_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QSEEK_TELEGRAM_BOT_TOKEN", "token-from-env")
    monkeypatch.setenv("QSEEK_TELEGRAM_CHAT_ID", "1234")

    alert = TelegramAlert(bot_token="token-from-config", chat_id="42")
    assert alert.bot_token.get_secret_value() == "token-from-config"
    assert alert.chat_id == "42"


def test_missing_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QSEEK_TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("QSEEK_TELEGRAM_CHAT_ID", raising=False)

    with pytest.raises(ValidationError):
        TelegramAlert.model_validate({"callback": "TelegramAlert"})
