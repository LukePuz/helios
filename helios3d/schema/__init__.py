"""Storage for Helios 3D: its own migration runner and the separate Postgres
schema ``h3d``. Migrations are numbered, idempotent and tracked in
``h3d.schema_version``, so nothing here touches the live ``_MIGRATIONS`` list
in db.py or any public-schema table. Later waves add the decision record,
input snapshot, panel output and portfolio action tables here.
"""
