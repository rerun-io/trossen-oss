# trossen-oss

Local [Rerun](https://rerun.io) **ingest → register → query** pipeline for Trossen
bimanual robot episodes. Convert MCAP recordings into RRDs, register them into a
local Rerun data platform, and query/visualize them — entirely on your machine,
no cloud required.

![Episode 0 playing in the Rerun viewer](media/episode_0.gif)

> Episode 0 of the
> [`pablovela5620/trossen-mjwarp-episodes`](https://huggingface.co/datasets/pablovela5620/trossen-mjwarp-episodes)
> dataset, rendered with the project's default blueprint: a 3D scene, four camera
> feeds, the robot-state table, and per-joint / gripper signal graphs.
> ([full-resolution video](media/episode_0.mp4))

## What it does

Runs Rerun's experiment loop — **Collect → Refine → Train → Deploy** — locally.
Collect and Refine's *register* + *query* work today; the rest is on the
[roadmap](#roadmap).

- **Collect** (`src/trossen_oss/preprocessing.py`) — converts each
  `episode_<n>_proto.mcap` into layered RRDs (`episode_<NNN>_{data,urdf}.rrd`)
  plus a saved blueprint, using Rerun's chunk-processing API. Derives per-joint
  and gripper scalar signals and computes forward kinematics from the URDF.
- **Refine · register** (`src/trossen_oss/catalog.py`) — registers every episode
  as a catalog *segment* (a `base` + a `urdf` layer) in a local in-memory Rerun
  data platform, with the blueprint as the dataset default.
- **Refine · query** (`src/trossen_oss/query.py` + `notebooks/`) — cross-episode
  DataFusion queries: per-arm joint travel (which arm is scripted vs. task-driven)
  and per-joint velocity-limit violations, headless or in interactive notebooks.

## Quickstart

```bash
pixi install                # solve + install the environment

pixi run download           # fetch 10 sample episodes (or `download-all` for 100)
pixi run preprocess         # MCAP -> outputs/rrds/*.rrd (+ blueprint)

pixi run serve              # local Rerun data platform on :51234  (leave running)

# in another shell:
pixi run register           # register the RRDs as the `trossen_oss` dataset
pixi run query              # cross-episode queries (headless)
pixi run notebook           # interactive analytical notebooks
```

## Tasks

| Task | What it does |
| --- | --- |
| `download` / `download-all` | Fetch episode MCAPs from Hugging Face (10 / all 100) |
| `preprocess` | Convert MCAP episodes into RRDs + blueprint under `outputs/` |
| `serve` | Start the local in-memory Rerun catalog on `:51234` |
| `register` | Register the RRDs as the `trossen_oss` catalog dataset |
| `query` | Headless cross-episode DataFusion queries |
| `notebook` | JupyterLab analytical notebooks (welcome / SQL / DataFrame) |

Dev tasks live in the `dev` environment: `pixi run -e dev check-all`
(format-check + `pyrefly` typecheck + `pytest`).

## Data

Episodes come from the public Hugging Face dataset
[`pablovela5620/trossen-mjwarp-episodes`](https://huggingface.co/datasets/pablovela5620/trossen-mjwarp-episodes)
— 100 bimanual manipulation episodes (~188 MB each). The robot model (URDF +
meshes) is vendored under `assets/urdf/`.

## Roadmap

Collect and Refine's *register* + *query* are in place. Remaining work, grouped by
experiment-loop stage:

### Refine · enrich
- [ ] Attach derived signals as a separate post-registration **layer** (same
  `recording_id`, new layer name) instead of baking scalars into the base
  recording — and show fixing a bad pass by re-registering just that layer
- [ ] Compute at least one named enrichment signal after the fact (per-frame blur,
  keyframe, or quality verdict) as a derived layer
- [ ] Persist query results back to the catalog as **tables** that show up in the
  viewer, rather than only printing/displaying them

### Train — *deferred (observation-only dataset; task/model TBD)*
- [ ] Curate a training slice by filtering episodes on a derived quality signal
  ("the training set is a query")
- [ ] Map Rerun entity paths to a training schema via explicit column mapping
  (state / action / camera) with fixed-rate alignment
- [ ] Export the curated slice to **LeRobot** format
- [ ] Define the `[dataloader]` torch extra and log loss curves + sample batches as
  Rerun recordings during training

### Deploy — *not yet in scope*
- [ ] Record eval rollouts as episodes with provenance tags (checkpoint, scene,
  version, outcome)
- [ ] Attach domain metadata (robot / operator / task / model version / tags) at
  register time so segments are searchable by what matters
- [ ] Add a success-by-condition group-by over eval outcomes plus a failure-triage
  flow that opens the underlying recordings

## More

Built with [pixi](https://pixi.prefix.dev) and Rerun 0.33. See
[`docs/prd-local-ingestion.md`](docs/prd-local-ingestion.md) for the full design,
decisions, and scope.
