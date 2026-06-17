---
name: rerun-parquet
description: Ingest tabular Parquet files into Rerun chunk streams with rerun.experimental.ParquetReader. Read when converting trajectory or sensor tables (LeRobot-style parquet, exported logs) into entities and components — column grouping, timeline/index columns, static columns, and ColumnRules that assemble typed components (Transform3D, Scalars) from flat columns. Builds on rerun-chunk-processing and rerun-data-model.
user_invocable: true
allowed-tools: Read, Grep, Bash, WebFetch
---

# Rerun Parquet Ingestion

`ParquetReader` maps a flat table onto the Rerun model: column-name prefixes
become entities, suffixes become components, designated columns become
timelines. The whole job is configuration; fill in the `rerun-data-model`
mapping table first, then express it through the constructor. Stream mechanics
after `.stream()` are in `rerun-chunk-processing`.

Verified against `rerun-sdk 0.34.0a1`.

## The API

```python
from rerun.experimental import ColumnRule, ParquetReader

reader = ParquetReader(
    table_path,
    entity_path_prefix="/world",         # prepended to every entity path
    column_grouping="prefix",            # "prefix" | "individual" | "explicit_prefixes"
    delimiter="_",                       # split for column_grouping="prefix"
    prefixes=None,                       # required for "explicit_prefixes"
    use_structs=True,                    # pack grouped columns into one struct component
    static_columns=["robot_type"],       # constant-per-file values, logged static
    index_columns=[("timestamp", "timestamp", "us"), ("frame_index", "sequence")],
    column_rules=[...],                  # typed-component assembly, below
)
stream = reader.stream()
```

## Column grouping: which columns share an entity

- `"prefix"` (default): split each column name on `delimiter`, group by the
  first segment. `gripper_pos_x`, `gripper_pos_y` → entity `gripper`.
- `"explicit_prefixes"`: group by the strings in `prefixes`, tried
  longest-first; the prefix is stripped from the component name. Use this when
  names contain the delimiter ambiguously (`observation.state` vs
  `observation.images.top`: pass the full prefixes).
- `"individual"`: every column is its own chunk/entity. Rarely the model you
  want; reach for it only as a debugging baseline.

`use_structs=True` (default) packs a group's columns into a single Arrow
struct component; `False` emits one component per column (flat layout, what
queries see as separate columns).

## Timelines: `index_columns`

Each entry is `(name, type)` or `(name, type, unit)`:

- `type`: `"timestamp"` (since epoch), `"duration"` (elapsed), `"sequence"`
  (ordinal int).
- `unit` describes what the raw integers in the column *are* (`"ns"` default,
  `"us"`, `"ms"`, `"s"`); Rerun rescales to ns internally. Ignored for
  `"sequence"`.

**If omitted, a synthetic `row_index` sequence timeline is generated.** That
is almost never the timeline you want to query or align against; always name
the real time columns. Stamp both a timestamp and a sequence timeline when the
table has both (multi-rate alignment, see `rerun-data-model`).

## Typed components: `column_rules`

Without rules, grouped columns stay generic struct/scalar data. Rules combine
suffix-matched columns into real Rerun components so the viewer and transform
system understand them:

- `ColumnRule.translation3d([sx, sy, sz])` → `Translation3D`
- `ColumnRule.rotation_quat([sx, sy, sz, sw])` → `RotationQuat`
- `ColumnRule.rotation_axis_angle([ax, ay, az, angle])` → `RotationAxisAngle`
- `ColumnRule.scale3d([sx, sy, sz])` → `Scale3D`
- `ColumnRule.scalars(suffixes, names=[...])` → `Scalars` with named series
- `ColumnRule.transform(translation_suffixes, rotation_suffixes)` →
  `Transform3D` (3 + 4 columns; both suffix sets must match under the same
  sub-prefix)

```python
column_rules=[
    ColumnRule.translation3d(["_pos_x", "_pos_y", "_pos_z"], field_name_override="_pos"),
    ColumnRule.rotation_quat(["_quat_x", "_quat_y", "_quat_z", "_quat_w"], field_name_override="_quat"),
    ColumnRule.scalars(["_x", "_y", "_z"], names=["x", "y", "z"]),
]
```

Rules are tried **in list order; first match wins** — put specific rules
before broad catch-alls (a `scalars` rule on `["_x", "_y", "_z"]` placed first
would swallow the position columns meant for `translation3d`).

## Gotchas

1. No `index_columns` → synthetic `row_index` timeline only. Queries that
   expect a timestamp timeline find nothing.
2. The `unit` is the raw column's unit, not a desired output unit; a
   microsecond column declared `"ns"` lands 1000x in the past.
3. `static_columns` raises if a listed column actually varies; that error is a
   data-quality signal, not a reason to drop the static declaration.
4. Rule order: first matching rule wins.
5. Quaternion column order is x, y, z, w in `rotation_quat`; check the
   source's convention before wiring suffixes.
6. Anything the reader cannot express (per-row entity routing, derived
   values, unit conversion) belongs in lenses downstream, not in pre-pandas
   munging; keep the pipeline columnar (`rerun-chunk-processing`).

## References

- API source with full docstrings: `rerun/experimental/_parquet_reader.py` in
  the installed `rerun-sdk` package, or
  `python -c "from rerun.experimental import ParquetReader; help(ParquetReader)"`
- `rerun-lerobot` — LeRobot datasets store episodes as parquet; that skill
  covers the built-in importer route vs reading the parquet directly with
  this reader.
- `rerun-data-model` (mapping decisions), `rerun-chunk-processing` (stream
  mechanics after `.stream()`)
