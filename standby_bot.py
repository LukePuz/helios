"""
Standby chat listener — the only thing running overnight/weekends once the
full pipeline (helios-local.service) has been stopped. Keeps the bot
reachable 24/7 without keeping IBKR, Celery, or the 72B model up:

  - Answers @mentions / DMs immediately via chat_assistant, using cached DB
    data (trade history, market regime) since IBKR isn't connected — those
    tools already degrade gracefully when IBKR is offline.
  - The moment a message arrives, triggers a full system boot
    (helios-local.service, which cascades ibc.service -> xvfb.service via
    their own Requires=) so live portfolio data and trading resume within
    about a minute. helios-local.service declares Conflicts=helios-standby
    .service, so systemd stops this process automatically the instant the
    full pipeline comes up — no self-stop logic needed here.
  - Also comes up naturally at market open: helios-start.timer starts
    helios-local.service on its own schedule, which stops this the same way.

Deliberately minimal: no IBKR, Celery, or dashboard import, so it can run
all night on a live-but-idle machine with negligible CPU/GPU load. Reuses
telegram_bot's token/allowlist/mention/polling primitives directly rather
than duplicating them, so standby behavior never drifts from the full
pipeline's behavior.

Requires a scoped sudoers rule (this process runs as a normal user and
can't start systemd units otherwise):
    YOUR_USER ALL=(root) NOPASSWD: /usr/bin/systemctl start helios-local.service
"""
import logging
import re
import subprocess
import time

import telegram_bot as tb

log = logging.getLogger(__name__)

_WAKE_UNIT = "helios-local.service"


def _system_awake() -> bool:
    """True if the full pipeline is already up or coming up — avoids firing
    a redundant `systemctl start` on every message once a wake is already
    in flight."""
    try:
        out = subprocess.run(
            ["systemctl", "is-active", _WAKE_UNIT],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        return out in ("active", "activating", "reloading")
    except Exception as e:
        log.warning(f"systemctl is-active check failed: {e}")
        return False


def _wake_system() -> None:
    """Boots the full pipeline via the scoped sudoers rule (see module
    docstring). -n (non-interactive) fails fast instead of hanging on a
    password prompt that will never come, since this runs unattended."""
    log.info("Wake triggered — starting helios-local.service.")
    try:
        result = subprocess.run(
            ["sudo", "-n", "systemctl", "start", _WAKE_UNIT],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            log.error(
                f"Failed to start {_WAKE_UNIT} (rc={result.returncode}): "
                f"{result.stderr.strip()} — check the NOPASSWD sudoers rule "
                f"documented in this module's docstring."
            )
    except Exception as e:
        log.error(f"Failed to trigger wake: {e}")


def _handle_message(update: dict) -> None:
    msg     = update.get("message", {})
    chat_id = msg.get("chat", {}).get("id")
    msg_id  = msg.get("message_id")
    from_   = msg.get("from", {})
    user_id = from_.get("id")
    display = from_.get("first_name") or from_.get("username") or str(user_id)
    text    = msg.get("text", "")

    if not chat_id or not text:
        return
    if tb._ALLOWED_USER_IDS and user_id not in tb._ALLOWED_USER_IDS:
        log.warning(f"[standby] Blocked message from unauthorized user_id={user_id} ({display})")
        return

    bot_mention = f"@{tb._BOT_USERNAME}".lower() if tb._BOT_USERNAME else None
    is_mention  = bool(bot_mention and bot_mention in text.lower())
    is_dm       = msg.get("chat", {}).get("type") == "private"
    if not (is_mention or is_dm):
        return

    clean = text
    if tb._BOT_USERNAME:
        clean = re.sub(rf"@{re.escape(tb._BOT_USERNAME)}", "", clean, flags=re.IGNORECASE).strip()
    if not clean:
        clean = "What's the latest on my portfolio?"

    log.info(f"[standby] msg from user_id={user_id} ({display}): {clean[:80]!r}")

    already_awake = _system_awake()
    tb._send_chat_action(chat_id)
    try:
        import chat_assistant
        reply = chat_assistant.respond(clean, display, user_id=user_id)
    except Exception as e:
        log.error(f"[standby] chat_assistant.respond failed: {e}")
        reply = "Sorry, I hit an error trying to answer that."

    if not already_awake:
        reply += (
            "\n\n💤 The full system was asleep for the night — that answer is "
            "from cached data. Waking everything up now (~1 min for live "
            "portfolio data and trading)."
        )
        _wake_system()

    tb._reply(chat_id, reply, msg_id)


def _poll_loop() -> None:
    log.info("Standby listener started.")
    tb._fetch_bot_username()
    offset = 0
    while True:
        try:
            updates = tb._get_updates(offset)
        except Exception as e:
            log.error(f"[standby] _get_updates failed: {e}")
            time.sleep(tb._POLL_INTERVAL)
            continue
        for update in updates:
            try:
                offset = update["update_id"] + 1
                if "message" in update:
                    _handle_message(update)
            except Exception as e:
                log.error(f"[standby] Failed to process update {update.get('update_id')}: {e}")
        if not updates:
            time.sleep(tb._POLL_INTERVAL)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not tb._TOKEN:
        log.error("TELEGRAM_BOT_TOKEN not set — standby listener cannot run.")
        return
    _poll_loop()


if __name__ == "__main__":
    main()
