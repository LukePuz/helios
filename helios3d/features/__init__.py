"""The point-in-time feature store shared by every analyst: ``get(source,
ticker, as_of)`` with one reader per source (price bars, fundamentals, regime,
earnings calendar, news, portfolio snapshots, position stops), each refusing
rows recorded or published after ``as_of`` so no analyst sees look-ahead
data, and each flagging ``asof_safe`` with a reason (store.py). Also the
append-only price copy h3d.price_history_asof (capture.py) that closes the
price_history upsert gap.
"""
from helios3d.features.store import SOURCES, FeatureResult, get  # noqa: F401
