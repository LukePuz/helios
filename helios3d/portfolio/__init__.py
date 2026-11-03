"""The portfolio (Phase 3, shadow only): a live portfolio view (holdings,
cash, sector/theme exposure, correlations), a risk-budget sizer that sizes a
new idea by what the whole book can take, funding decisions (what to trim to
buy what, with logged reasons), and pure-function lifts of the existing hard
limits (risk per trade, heat, theme and bucket caps, the correlation cut) as
constraints the sizer must satisfy.
"""
