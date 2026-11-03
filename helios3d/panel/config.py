"""Settings for the shadow analyst panel, read from the environment.

Nothing here has a default that calls a model. The compute budget
is a setting with three options and no default other than off:

HELIOS3D_PANEL_BUDGET
    off        (default) the runner calls no model; ``run`` only dry-runs.
    full       every decision record gets the full panel.
    shortlist  only decisions matching the shortlist rule below.

HELIOS3D_PANEL_SHORTLIST_ACTIONS     comma list of live actions (OPEN,ADD,...)
HELIOS3D_PANEL_SHORTLIST_MIN_CONVICTION  live conviction at or above this
    A decision is shortlisted when it matches either one that is set. With
    budget=shortlist and neither set, the settings are refused.

HELIOS3D_PANEL_MODEL                  model for every call (no default)
HELIOS3D_PANEL_MODEL_<ROLE>           per-role override (FUNDAMENTALS,
    TECHNICAL_FLOW, MACRO_REGIME, NEWS_SENTIMENT, PM, RED_TEAM): the
    "smaller model per role" option. Calling a model needs a model for
    every role that runs.

HELIOS3D_OLLAMA_URL                   Ollama base URL for the panel. No
    default, so the panel can never fall back to the live server's URL.
    Any value must be set on purpose.

Shared-server guard (Ollama reloads a model when a request changes the
model or num_ctx, which evicts the live 72B mid-session):
HELIOS3D_LIVE_OLLAMA_URL              the live server's URL (no default).
                                      Calling needs it set.
HELIOS3D_LIVE_MODEL                   the model live loads (ANALYST_MODEL).
    When HELIOS3D_OLLAMA_URL is the live server, calling is refused unless
    every role's model equals HELIOS3D_LIVE_MODEL and HELIOS3D_PANEL_NUM_CTX
    equals live's num_ctx (LIVE_NUM_CTX, 32768, agents/analyst.py _NUM_CTX).
    "The live server" (same_server): the same port, and host address sets
    that intersect. Both hosts are resolved with getaddrinfo; IPv4-mapped
    IPv6 addresses are unwrapped; every loopback address, the unspecified
    address and every address of this machine (getaddrinfo of its hostname
    and its interface addresses) fold into one "local" token, so
    127.0.0.2, 127.1, 0, 2130706433, "localhost.", [::ffff:127.0.0.1], the
    hostname and the LAN/Tailscale IP all count as localhost. A URL that
    does not parse or resolve, or a live URL with no explicit port, counts
    as shared (fail closed).

Call shape (engineering settings; the values are listed as open questions
in the Phase 2 report):
HELIOS3D_PANEL_TIMEOUT_S              per call, default 240; must be below
                                      live's 400 s (analyst._call_ollama)
HELIOS3D_PANEL_NUM_PREDICT_ANALYST    default 300 (compact roles)
HELIOS3D_PANEL_NUM_PREDICT_PM         default 700 (thesis plus plan block)
HELIOS3D_PANEL_NUM_PREDICT_RED_TEAM   default 500
HELIOS3D_PANEL_NUM_CTX                default 8192
HELIOS3D_PANEL_TEMPERATURE            default 0.0
HELIOS3D_PANEL_SEED                   default 0
HELIOS3D_PANEL_TRADING_ENVS           comma list; empty = every env

Live-lane protection:
HELIOS3D_PANEL_KILL_FILE              default <repo>/logs/.h3d_off
HELIOS3D_LIVE_REDIS_URL               the live broker, read only; falls
                                      back to REDIS_URL, then localhost
HELIOS3D_PANEL_GATE_WAIT_S            how long to wait for the live lane
                                      to clear before yielding (default 600)
HELIOS3D_PANEL_GATE_POLL_S            poll interval while waiting (15)
HELIOS3D_PANEL_LIVE_TASK_MAX_AGE_S    a STARTED live task older than this
                                      is taken as stale (default 1800)
HELIOS3D_CELERY_RESULT_EXPIRES_S      the live result TTL (Celery default
                                      86400), used to age STARTED entries
HELIOS3D_PANEL_GATE_MAX_KEYS          result keys / unacked entries the gate
                                      reads before failing closed (50000)
"""
from __future__ import annotations

import functools
import hashlib
import ipaddress
import json
import os
import socket
from dataclasses import asdict, dataclass, field

ANALYST_ROLES = ("fundamentals", "technical_flow", "macro_regime", "news_sentiment")
PM_ROLE = "pm"
RED_TEAM_ROLE = "red_team"
ALL_ROLES = ANALYST_ROLES + (PM_ROLE, RED_TEAM_ROLE)
BUDGETS = ("off", "full", "shortlist")
LIVE_TIMEOUT_S = 400  # agents.analyst._call_ollama
LIVE_NUM_CTX = 32768  # agents.analyst._NUM_CTX (a test reads it from the file)
LOCAL_TOKEN = "local"

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_KILL_FILE = os.path.join(_REPO, "logs", ".h3d_off")


class SettingsError(ValueError):
    """The panel settings are inconsistent or incomplete."""


def parse_host_port(url: str | None) -> tuple[str, int] | None:
    """(host, port) of a URL as requests would connect to it; None when
    unset or unparseable."""
    if not url:
        return None
    from urllib.parse import urlsplit
    try:
        u = urlsplit(url if "//" in url else f"http://{url}")
        host = u.hostname or ""
        port = u.port or (443 if u.scheme == "https" else 80)
    except ValueError:
        return None
    return (host, port) if host else None


def has_explicit_port(url: str | None) -> bool:
    """Whether a URL names its port (http://host:PORT), parsed as requests
    would; False when unset or unparseable."""
    if not url:
        return False
    from urllib.parse import urlsplit
    try:
        return urlsplit(url if "//" in url else f"http://{url}").port is not None
    except ValueError:
        return False


def _ip(addr: str):
    """An address string as an ipaddress object, IPv4-mapped IPv6 unwrapped
    and any %scope dropped."""
    ip = ipaddress.ip_address(str(addr).split("%", 1)[0])
    if ip.version == 6 and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip


@functools.lru_cache(maxsize=1)
def local_addresses() -> frozenset:
    """Every address of this machine: getaddrinfo of its hostname and, when
    psutil is installed, every interface address (LAN, Tailscale, ...)."""
    out = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None):
            out.add(_ip(info[4][0]))
    except (OSError, ValueError):
        pass
    try:
        import psutil
        for addrs in psutil.net_if_addrs().values():
            for a in addrs:
                if a.family in (socket.AF_INET, socket.AF_INET6):
                    try:
                        out.add(_ip(a.address))
                    except ValueError:
                        continue
    except Exception:
        pass
    return frozenset(out)


def host_tokens(host: str, port: int) -> frozenset | None:
    """The addresses a host resolves to, with every address of this machine
    (loopback, unspecified, own interfaces) folded into LOCAL_TOKEN. None
    when it does not resolve."""
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except (OSError, UnicodeError, ValueError):
        return None
    local = local_addresses()
    out = set()
    for info in infos:
        try:
            ip = _ip(info[4][0])
        except (ValueError, IndexError, TypeError):
            continue
        out.add(LOCAL_TOKEN if (ip.is_loopback or ip.is_unspecified or ip in local) else str(ip))
    return frozenset(out) or None


def same_server(a: str | None, b: str | None) -> bool:
    """Whether two Ollama URLs reach the same server: same port and host
    address sets that intersect. Fails closed: a URL that does not parse or
    resolve counts as the same server. Different ports are different
    servers without a lookup."""
    pa, pb = parse_host_port(a), parse_host_port(b)
    if pa is None or pb is None:
        return True
    if pa[1] != pb[1]:
        return False
    ta, tb = host_tokens(*pa), host_tokens(*pb)
    if ta is None or tb is None:
        return True
    return bool(ta & tb)


@dataclass(frozen=True)
class PanelSettings:
    budget: str = "off"
    shortlist_actions: tuple[str, ...] = ()
    shortlist_min_conviction: float | None = None
    model: str | None = None
    role_models: dict = field(default_factory=dict)
    ollama_url: str | None = None
    live_ollama_url: str | None = None
    live_model: str | None = None
    timeout_s: float = 240.0
    num_predict_analyst: int = 300
    num_predict_pm: int = 700
    num_predict_red_team: int = 500
    num_ctx: int = 8192
    temperature: float = 0.0
    seed: int = 0
    trading_envs: tuple[str, ...] = ()
    kill_file: str = DEFAULT_KILL_FILE
    live_redis_url: str = "redis://localhost:6379/0"
    gate_wait_s: float = 600.0
    gate_poll_s: float = 15.0
    live_task_max_age_s: float = 1800.0
    gate_max_keys: int = 50000
    result_expires_s: float = 86400.0

    # -- derived --------------------------------------------------------
    def model_for(self, role: str) -> str | None:
        return self.role_models.get(role) or self.model

    def num_predict_for(self, role: str) -> int:
        if role == PM_ROLE:
            return self.num_predict_pm
        if role == RED_TEAM_ROLE:
            return self.num_predict_red_team
        return self.num_predict_analyst

    def shortlisted(self, action: str | None, conviction) -> bool:
        """Whether a decision passes the budget rule. ``off`` passes nothing."""
        if self.budget == "full":
            return True
        if self.budget != "shortlist":
            return False
        if self.shortlist_actions and str(action or "").upper() in self.shortlist_actions:
            return True
        if self.shortlist_min_conviction is not None and conviction is not None:
            try:
                return float(conviction) >= self.shortlist_min_conviction
            except (TypeError, ValueError):
                return False
        return False

    def check(self, calling: bool) -> None:
        """Refuse inconsistent settings; with ``calling``, also refuse
        anything that would stop a model call from being well defined."""
        if self.budget not in BUDGETS:
            raise SettingsError(f"HELIOS3D_PANEL_BUDGET must be one of {BUDGETS}, not {self.budget!r}")
        if self.budget == "shortlist" and not (self.shortlist_actions
                                               or self.shortlist_min_conviction is not None):
            raise SettingsError("budget=shortlist needs HELIOS3D_PANEL_SHORTLIST_ACTIONS or "
                                "HELIOS3D_PANEL_SHORTLIST_MIN_CONVICTION")
        if not (0 < self.timeout_s < LIVE_TIMEOUT_S):
            raise SettingsError(f"HELIOS3D_PANEL_TIMEOUT_S must be in (0, {LIVE_TIMEOUT_S})")
        if not calling:
            return
        if self.budget == "off":
            raise SettingsError("HELIOS3D_PANEL_BUDGET is off: no model is called")
        if not self.ollama_url:
            raise SettingsError("HELIOS3D_OLLAMA_URL is not set (it has no default on purpose)")
        missing = [r for r in ALL_ROLES if not self.model_for(r)]
        if missing:
            raise SettingsError(f"no model for roles {missing}: set HELIOS3D_PANEL_MODEL "
                                "or HELIOS3D_PANEL_MODEL_<ROLE>")
        if not self.live_ollama_url:
            raise SettingsError("HELIOS3D_LIVE_OLLAMA_URL is not set: it is needed to tell "
                                "whether the panel shares the live server")
        if self.shares_live_server():
            if self.num_ctx != LIVE_NUM_CTX:
                raise SettingsError(f"the panel uses the live Ollama server: "
                                    f"HELIOS3D_PANEL_NUM_CTX must be {LIVE_NUM_CTX} (live's), "
                                    f"not {self.num_ctx}, or Ollama reloads the live model")
            if not self.live_model:
                raise SettingsError("the panel uses the live Ollama server: set "
                                    "HELIOS3D_LIVE_MODEL so the model match can be checked")
            other = sorted({self.model_for(r) for r in ALL_ROLES} - {self.live_model})
            if other:
                raise SettingsError(f"the panel uses the live Ollama server: models {other} "
                                    f"differ from the live model; Ollama would swap models")

    def shares_live_server(self) -> bool:
        """same_server(panel URL, live URL); fails closed. A live URL with no
        explicit port counts as shared: it is likely mis-set (Ollama listens
        on its own port, not on the scheme default)."""
        if not has_explicit_port(self.live_ollama_url):
            return True
        return same_server(self.ollama_url, self.live_ollama_url)

    def config_sha(self, templates_sha: str) -> str:
        """Hash of everything that shapes a panel output: templates, models,
        call options. Rows are unique per (decision, config)."""
        shape = {"templates": templates_sha,
                 "models": {r: self.model_for(r) for r in ALL_ROLES},
                 "num_predict": {r: self.num_predict_for(r) for r in ALL_ROLES},
                 "num_ctx": self.num_ctx, "temperature": self.temperature, "seed": self.seed}
        return hashlib.sha256(json.dumps(shape, sort_keys=True).encode()).hexdigest()[:16]

    def public_view(self) -> dict:
        """The settings for a report: the Ollama URL is reduced to whether it
        is set."""
        d = asdict(self)
        d["ollama_url"] = bool(self.ollama_url)
        d["live_ollama_url"] = bool(self.live_ollama_url)
        d["shares_live_server"] = (self.shares_live_server()
                                   if self.ollama_url and self.live_ollama_url else None)
        d["live_redis_url"] = bool(self.live_redis_url)
        return d


def _csv(v: str | None, upper: bool = True) -> tuple[str, ...]:
    if not v:
        return ()
    return tuple(x.strip().upper() if upper else x.strip() for x in v.split(",") if x.strip())


def _num(env, key, default, cast=float):
    v = env.get(key)
    if v is None or str(v).strip() == "":
        return default
    try:
        return cast(v)
    except ValueError:
        raise SettingsError(f"{key}={v!r} is not a number") from None


def from_env(env: dict | None = None) -> PanelSettings:
    env = os.environ if env is None else env
    role_models = {}
    for r in ALL_ROLES:
        m = env.get(f"HELIOS3D_PANEL_MODEL_{r.upper()}")
        if m and m.strip():
            role_models[r] = m.strip()
    min_conv = env.get("HELIOS3D_PANEL_SHORTLIST_MIN_CONVICTION")
    return PanelSettings(
        budget=(env.get("HELIOS3D_PANEL_BUDGET") or "off").strip().lower(),
        shortlist_actions=_csv(env.get("HELIOS3D_PANEL_SHORTLIST_ACTIONS")),
        shortlist_min_conviction=_num(env, "HELIOS3D_PANEL_SHORTLIST_MIN_CONVICTION", None)
        if min_conv else None,
        model=(env.get("HELIOS3D_PANEL_MODEL") or "").strip() or None,
        role_models=role_models,
        ollama_url=(env.get("HELIOS3D_OLLAMA_URL") or "").strip().rstrip("/") or None,
        live_ollama_url=(env.get("HELIOS3D_LIVE_OLLAMA_URL") or "").strip().rstrip("/") or None,
        live_model=(env.get("HELIOS3D_LIVE_MODEL") or "").strip() or None,
        timeout_s=_num(env, "HELIOS3D_PANEL_TIMEOUT_S", 240.0),
        num_predict_analyst=_num(env, "HELIOS3D_PANEL_NUM_PREDICT_ANALYST", 300, int),
        num_predict_pm=_num(env, "HELIOS3D_PANEL_NUM_PREDICT_PM", 700, int),
        num_predict_red_team=_num(env, "HELIOS3D_PANEL_NUM_PREDICT_RED_TEAM", 500, int),
        num_ctx=_num(env, "HELIOS3D_PANEL_NUM_CTX", 8192, int),
        temperature=_num(env, "HELIOS3D_PANEL_TEMPERATURE", 0.0),
        seed=_num(env, "HELIOS3D_PANEL_SEED", 0, int),
        trading_envs=_csv(env.get("HELIOS3D_PANEL_TRADING_ENVS"), upper=False),
        kill_file=env.get("HELIOS3D_PANEL_KILL_FILE") or DEFAULT_KILL_FILE,
        live_redis_url=(env.get("HELIOS3D_LIVE_REDIS_URL") or env.get("REDIS_URL")
                        or "redis://localhost:6379/0"),
        gate_wait_s=_num(env, "HELIOS3D_PANEL_GATE_WAIT_S", 600.0),
        gate_poll_s=_num(env, "HELIOS3D_PANEL_GATE_POLL_S", 15.0),
        live_task_max_age_s=_num(env, "HELIOS3D_PANEL_LIVE_TASK_MAX_AGE_S", 1800.0),
        gate_max_keys=_num(env, "HELIOS3D_PANEL_GATE_MAX_KEYS", 50000, int),
        result_expires_s=_num(env, "HELIOS3D_CELERY_RESULT_EXPIRES_S", 86400.0),
    )
