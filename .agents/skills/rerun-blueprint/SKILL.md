---
name: rerun-blueprint
description: Design a Rerun blueprint from the data, then iterate on it from headless screenshots. Read this when laying out a recording or dataset in the viewer, designing a default blueprint, or deciding which views show which entities. Covers archetype-to-view mapping, layout reasoning, the rrb construction API, the contents grammar, and the screenshot loop.
user_invocable: true
allowed-tools: Read, Grep, Bash, WebFetch
---

# Rerun Blueprint

A blueprint decides how a recording is shown; the recording decides what exists.
Read the data, design a first layout, then **iterate from screenshots** until it
reads at a glance. The constructors are discoverable (`dir(rrb)`,
`help(rrb.Spatial3DView)`); this skill covers what you can't guess. Import as
`import rerun.blueprint as rrb`.

## 1. Read the data

Enumerate every `(entity_path, archetype)` pair first — the archetype picks the
view, the entity path scopes it. For a dataset (see `rerun-catalog-queries`):

```python
for c in dataset.schema().component_columns():
    print(c.entity_path, c.archetype, c.component_name)
```

For a local recording, stream it with `RrdReader` and read `entity_path` and
the archetype off each chunk (see `rerun-chunk-processing`).

## 2. Map archetype to view

| Archetype family | View |
| --- | --- |
| `Points2D`, `Image`, `EncodedImage`, `Boxes2D`, `LineStrips2D`, `Pinhole` projections | `Spatial2DView` |
| `Points3D`, `Mesh3D`, `Boxes3D`, `LineStrips3D`, `Transform3D`, `Asset3D` | `Spatial3DView` |
| `Scalars`, `SeriesLines`, `SeriesPoints` | `TimeSeriesView` |
| `TextLog` / `TextDocument` / `BarChart` | `TextLogView` / `TextDocumentView` / `BarChartView` |
| `Tensor`, `DepthImage` (heatmap) | `TensorView` |
| `GeoPoints`, `GeoLineStrings` | `MapView` |
| `GraphNodes`, `GraphEdges` | `GraphView` |
| tabular / catalog data | `DataframeView` |

## 3. Design the layout

A good layout reads like the recording: one subject, everything else as context.

- **Hero.** The entity with the richest spatial data (the 3D scene, the main
  camera) is the subject. Biggest pane, anchors left or center.
- **Place by role.** Extra cameras/scenes stack in a sidebar beside the hero.
  `TimeSeriesView`/`BarChartView` go in a wide, short band along the bottom (they
  read left-to-right against the time cursor). Logs, text, and dataframes go in a
  side column or tab.
- **Size to attention.** `column_shares`/`row_shares` are relative; give the hero
  2-3x its sidebar. Use equal shares only when views matter equally.
- **Mirror the entity tree, don't over-split.** Shared path prefix → group
  together. Dozens of raw message entities go in one `DataframeView` or get left
  out, never their own panes.
- **One spatial frame per spatial view.** A robot's whole `Transform3D` tree goes
  in one `Spatial3DView` at the common ancestor; a `Pinhole` camera gets its own
  `Spatial2DView` rooted at the camera so images inherit the projection.
- **Collapse scalars.** Many `Scalars` under a prefix → one `TimeSeriesView` over
  the prefix, not one each. Split only when value ranges clash.

## 4. Construct it

Containers (`Grid`, `Horizontal`, `Vertical`, `Tabs`) hold views. Default a flat
set to `Grid`; use the others for a deliberate split, `Tabs` for alternatives
competing for one slot (left/right/depth cameras). **Always set an explicit
`origin` and `name`** — `origin` defaults to `/`, which dumps the whole tree into
one view (the usual cause of an unreadable blob).

```python
blueprint = rrb.Blueprint(
    rrb.Horizontal(
        rrb.Spatial3DView(origin="/world", name="Scene"),
        rrb.Vertical(
            rrb.Spatial2DView(origin="/world/camera", name="Camera"),
            rrb.TimeSeriesView(origin="/sensors", name="Sensors"),
        ),
        column_shares=[3, 2],
    ),
    rrb.TextLogView(origin="/logs", name="Logs"),
    collapse_panels=True,
)
```

`contents` defaults to `"$origin/**"`. Scope a view with include/exclude rules,
e.g. `contents=["+ $origin/**", "- $origin/internal/**"]`. A bare line is an
include; `/**` is the only wildcard (matches a subtree). Most-specific rule wins,
ties go to the last, unmatched paths are excluded.

**Coordinate frames.** A spatial view only renders entities it can place relative
to its target frame. `Transform3D`/`Pinhole` logged on entities compose down the
tree, so `origin` is enough. But **named frames** (`CoordinateFrame(frame=...)`,
common in ROS/MCAP) live in a separate frame graph — point the view at a frame
the data occupies via `spatial_information=rrb.SpatialInformation(target_frame="<frame>")`.
The tell is an empty 3D view with "No transform path from `<frame>`..." errors:
`origin="/"` targets the root `tf#/`, which connects to nothing if the tf tree
was never materialized. Read the `CoordinateFrame:frame` values, target the one
the main 3D content sits in, and exclude entities in unconnected frames.

## 5. Iterate from screenshots

The gap between "correct" and "breathtaking" only shows in the picture, so render,
look, revise, repeat — plan on several passes. Spawn the viewer and load the
recording **once**, then re-send blueprints into it; each send + screenshot is one
cheap iteration.

```python
import time
import rerun as rr
import rerun.blueprint as rrb
from rerun.experimental import ViewerClient

port = 9879  # not 9876: spawn silently attaches to an existing viewer there
viewer = ViewerClient.spawn(headless=True, port=port)
time.sleep(2)
rr.init("blueprint_check")
rr.connect_grpc(f"rerun+http://127.0.0.1:{port}/proxy")
rr.log_file_from_path("segment.rrd")

def shot(blueprint, path):
    rr.send_blueprint(blueprint, make_active=True, make_default=True)
    time.sleep(3)  # let the frame render; bump if a view comes back blank
    viewer.save_screenshot(path)  # view_id=view.id for a single view

shot(blueprint_v1, "bp_v1.png")
# Read bp_v1.png; is the hero dominant, each view populated and scoped right,
# nothing cramped or crowding the stage? Each problem -> the next revision.
shot(blueprint_v2, "bp_v2.png")
viewer.close()
```

To bake a finished blueprint in instead, pass `default_blueprint=` to `rr.init` /
`spawn` / `connect_grpc` / `save`, or `blueprint=` to `notebook_show`.

## Gotchas

- A `DataframeView` shows "Unknown timeline" without a query:
  `query=rrb.archetypes.DataframeQuery(timeline="<timeline>", apply_latest_at=True)`.
- A view back blank? The importer may not have finished (bump the settle) or the
  cursor sits before the data (add `rrb.TimePanel(play_state=rrb.components.PlayState.Following)`).
- Headless needs a graphics stack (GPU or a software rasterizer like Mesa
  `lavapipe`); a bare container with no Vulkan adapter panics.
- `rrb` views and `ViewerClient` (`rerun.experimental`) are unstable — check
  `help()` if a constructor argument is rejected.

## See also

- `rerun-data-model` — entities, archetypes, timelines.
- `rerun-catalog-queries` — enumerate entities in a dataset.
