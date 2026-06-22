# trossen-oss

Local [Rerun](https://rerun.io) **collect → refine → train** pipeline for Trossen
bimanual robot episodes. Convert MCAP recordings into RRDs, register them into a
local Rerun catalog, and query, visualize, and train on them — entirely on
your machine, no cloud required.

![Episode 0 playing in the Rerun viewer](media/episode_0.gif)

> Episode 0 of the
> [`pablovela5620/trossen-mjwarp-episodes`](https://huggingface.co/datasets/pablovela5620/trossen-mjwarp-episodes)
> dataset, rendered with the project's default blueprint: a 3D scene, four camera
> feeds, the robot-state table, and per-joint / gripper signal graphs.
> ([full-resolution video](media/episode_0.mp4))

## What it does

Runs Rerun's experiment loop — **Collect → Refine → Train → Deploy** — locally.
Collect, Refine (*register* + *enrich* + *query*), and a toy **Train** work
end-to-end today; **Deploy** is future work.

- **Collect** (`src/trossen_oss/preprocessing.py`) — converts each
  `episode_<n>_proto.mcap` into layered RRDs (`episode_<NNN>_{data,urdf}.rrd`)
  plus a saved blueprint, using Rerun's chunk-processing API. In the same pass it
  fixes a recorded error (re-homes the camera videos onto their Pinhole
  image-plane frames so they show up in the 2D views), derives per-joint and
  gripper scalar signals, and computes forward kinematics from the URDF.
- **Refine · register** (`src/trossen_oss/catalog.py`) — registers every episode
  as a catalog *segment* (a `base` + a `urdf` layer) in a local in-memory Rerun
  catalog, with the blueprint as the dataset default.
- **Refine · enrich** (`src/trossen_oss/enrich.py`) — derives a per-episode
  *quality* verdict from the arm-activity query and re-registers it as a new
  `quality` layer on each segment — raw recordings untouched — flagging the
  low-motion episodes worth curating out of a training set.
- **Refine · query** (`src/trossen_oss/query.py` + `notebooks/`) — cross-episode
  DataFusion queries (DataFrame API **and** SQL): per-arm joint travel (which arm
  is scripted vs. task-driven) and per-joint velocity-limit violations, headless or
  in interactive notebooks. Results are persisted back as catalog *Table* entries,
  so analysis joins the data layer instead of living in stdout.
- **Train** (`src/trossen_oss/train.py`) — a toy *"the training set is a query"*:
  pick the most-active episodes with `arm_activity`, then **stream** their joint
  `Scalars` straight from the catalog through Rerun's
  [`RerunIterableDataset`](https://rerun.io/docs/howto/train/dataloader) wrapped in
  a PyTorch `DataLoader` (mirroring Rerun's
  [dataloader example](https://github.com/rerun-io/rerun/tree/main/examples/python/dataloader)),
  and fit a tiny MLP that predicts the next joint state. Loss curves are logged back
  to Rerun and each run is registered as a segment in a `trossen_oss_runs` dataset,
  so runs sit alongside the episodes in the catalog (CPU, ~3 min). This trains
  *directly from the catalog via the dataloader* rather than exporting a file-based
  LeRobot dataset first; a LeRobot export (the `rerun-lerobot` skill) is the
  alternative path and is future work here.

Everything here runs against the open-source, in-memory catalog (`rerun server`)
on a single machine — ideal for a project's worth of episodes. When you outgrow
that (far more recordings, object-store-backed storage, shared and hosted access
across a team), that's where [Rerun Hub](https://rerun.io) comes in: the managed,
hosted version of the same catalog — same API and workflow, at production scale.

## Quickstart

```bash
pixi install                # solve + install the environment

pixi run download           # fetch 10 sample episodes (or `download-all` for 100)
pixi run preprocess         # MCAP -> outputs/rrds/*.rrd (+ blueprint)

pixi run serve              # local Rerun catalog on :51234  (leave running)

# in another shell:
pixi run register           # register the RRDs as the `trossen_oss` dataset
pixi run enrich             # add a derived `quality` layer to each episode
pixi run query              # cross-episode queries (+ persist result tables)
pixi run notebook           # interactive analytical notebooks
pixi run train              # toy next-state training over a catalog query (CPU torch)
```

## Tasks

| Task | What it does |
| --- | --- |
| `download` / `download-all` | Fetch episode MCAPs from Hugging Face (10 / all 100) |
| `preprocess` | Convert MCAP episodes into RRDs + blueprint under `outputs/` |
| `serve` | Start the local in-memory Rerun catalog on `:51234` |
| `register` | Register the RRDs as the `trossen_oss` catalog dataset |
| `enrich` | Add a derived `quality` layer to the registered episodes (needs serve + register) |
| `query` | Headless cross-episode DataFusion queries; persist results as catalog tables |
| `notebook` | JupyterLab analytical notebooks (welcome / SQL / DataFrame) |
| `train` | Toy next-state training over a catalog query (CPU torch; needs serve + register) |

Dev tasks live in the `dev` environment: `pixi run -e dev check-all`
(format-check + `pyrefly` typecheck + `pytest`). The torch-dependent Train tests are
skipped there; run the full suite with `pixi run -e dataloader test`.

## Agent Skills

This repo vendors Rerun's [Agent Skills](https://github.com/rerun-io/rerun/tree/main/skills)
under `.agents/skills/` — focused playbooks that teach a coding agent the exact Rerun API
for each stage. Add them to your own project with `npx skills add rerun-io/rerun`.

| When you're… | Reach for |
| --- | --- |
| Deciding how your data maps onto Rerun (read first) | `rerun-data-model` |
| Converting MCAP (incl. custom protobuf) — `preprocessing.py` | `rerun-mcap`, `rerun-urdf`, `rerun-chunk-processing` |
| Registering / querying the catalog — `catalog.py`, `query.py` | `rerun-catalog-queries` |
| Designing the dataset's default layout | `rerun-blueprint` |
| Exporting a curated slice to LeRobot (alternative to the dataloader) | `rerun-lerobot` |

`rerun-catalog-queries` and `rerun-parquet` are vendored locally (not in the upstream set);
the dataloader Train path has no skill yet and follows the
[how-to](https://rerun.io/docs/howto/train/dataloader) directly.

## Data

Episodes come from the public Hugging Face dataset
[`pablovela5620/trossen-mjwarp-episodes`](https://huggingface.co/datasets/pablovela5620/trossen-mjwarp-episodes)
— 100 bimanual manipulation episodes (~188 MB each; budget ~35–40 GB of disk for
the full set of source MCAPs + converted RRDs). The robot model (URDF +
meshes) is vendored under `assets/urdf/`.

## Make it yours

The Trossen episodes are a stand-in for your data: run the local flow end-to-end against
the sample episodes, then point `preprocess` at your own MCAPs — almost everything (entity
paths, the blueprint, the query joints, the training fields) is configurable. Built with
[pixi](https://pixi.prefix.dev) and Rerun 0.33.
