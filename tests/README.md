# Tests

Run the suite (Redis pointed at a dead port so no real Celery task is published):

```
REDIS_URL=redis://127.0.0.1:1/0 venv/bin/python -m pytest -q tests
```

## Two switch modes

`tests/conftest.py` has autouse fixtures that stub DB, network and broker reads,
and a second set (the ones calling `_switch_off`) that turns each newer
config switch off, so a test sees the old behaviour unless it opts in.

That default never runs the combination that deploys. To run every test with
config's real values instead, set `HELIOS_TEST_SHIPPED_DEFAULTS=1`:

```
HELIOS_TEST_SHIPPED_DEFAULTS=1 REDIS_URL=redis://127.0.0.1:1/0 venv/bin/python -m pytest -q tests
```

In that mode the switch-off fixtures leave config alone; the read stubs stay on.
The pytest header line `config switches: ...` (shown without `-q`) says which
mode ran. Both modes should pass.

A test that checks one switch at a given value says so with a marker, which
applies in both modes after the switch-off fixtures:

```python
@pytest.mark.switches(PT_STOP_QTY_FROM_BROKER=False)
def test_switch_off_sizes_from_the_cache(): ...
```

A name config does not have raises, so a renamed switch can't silently pass.
Tests that assert "no extra row or alert" should allow the shadow rows a shipped
switch writes (for example `ENTRY_QUALITY_SHADOW` or `late_fill_expect`) when
that switch is on, rather than pin it off.

## Two flake patterns, and their fixes

Both pass alone and fail only in some full runs, so a single green run doesn't
show they're fixed.

### The activity writer runs after the test

`db.log_activity_queued` puts a row on a queue. A daemon writer thread takes it
later and calls whatever `db.log_activity_direct` is *at write time*. A row one
test queued can therefore land in the next test's patched `log_activity_direct`,
so an `assert_not_called()` or a row count fails in an unrelated test. It shows
mostly in shipped mode (`HELIOS_TEST_SHIPPED_DEFAULTS=1`), where more shadow
switches queue rows (`add_stop_qty`, the stop guard).

- conftest's autouse `_isolated_activity_writer` settles the queue after each
  test (`_settle_activity_queue`: queued rows dropped, a row in flight waited for
  up to `_ACTIVITY_SETTLE_S`, a queue still busy detached). That covers rows
  crossing between tests.
- It can't help inside one test: the writer can still deliver a queued row while
  the test is asserting on its direct-row mock. When a test asserts on
  `log_activity_direct` (or counts rows) around code that also queues rows,
  patch the queued writer in the harness:

  ```python
  with patch("db.log_activity_queued"), \
       patch("db.log_activity_direct") as direct:
      ...
  ```

  Use `patch("db.log_activity_queued", side_effect=...)` when the test needs to see
  the queued rows. Examples: `TestAddStopsWholePosition._run` in
  `test_execution.py`, and `test_add_stale_record.py`.

### Whole-text comparisons and the clock line

Alert texts end with a clock line (`alerts.send_trade_alert` appends
`Time:    HH:MM AM ET`, minute resolution). A test that compares the texts of two
runs fails when the runs straddle a minute boundary. Drop the clock line before
comparing:

```python
def _unclocked(texts):
    return ["\n".join(l for l in t.split("\n") if not l.startswith("Time:")) for t in texts]

assert _unclocked(texts_a) == _unclocked(texts_b)
```

Example: `test_stop_record_write_failure.py`. Patching the clock also works, but
the texts get their time from `datetime.now(_ET)` inside `alerts`, so the strip is
simpler. Keep asserting on the other lines.
