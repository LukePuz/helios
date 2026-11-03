"""The replay harness: rerun any past day's decisions offline from stored
snapshots, with stub, cached or live LLM modes, and diff the result against
the stored record. Also the comparison of shadow 3D against the current
pipeline on the same days, including the hard-limit breach check.

Built so far: the stub mode (the stored model output stands in for the
model call), in helios3d.replay.harness, with the analyst's deterministic
gates lifted pure into helios3d.replay.gates. CLI:
python -m helios3d.replay --start YYYY-MM-DD --end YYYY-MM-DD
"""
