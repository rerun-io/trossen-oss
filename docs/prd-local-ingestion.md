---
title: Local Ingestion Pipeline (trossen-oss)
status: ready-for-agent
---

# PRD: Local Ingestion Pipeline

> Triage label: `ready-for-agent`. Published as a repo file because `trossen-oss`
> has no issue tracker configured yet; migrate to an issue when one exists.

## Problem Statement

As a robotics demo user, I have Trossen bimanual episodes recorded as large MCAP
files, and I want to inspect, query, and eventually train on them entirely on my
own machine. The existing proof-of-concept (`trossen-poc-test`) only delivers the
full "ingest → register → query → train" story against the **hosted Rerun cloud**
(S3 for storage, Modal for compute, a remote catalog). Its "local" path stops at
writing RRD files and streaming them to a viewer — it never registers a catalog or
runs catalog queries locally. I cannot run the interesting capabilities (catalog
registration, DataFusion queries, a training dataloader) without cloud
credentials, an S3 bucket, and Modal. I want a version that does the whole loop
with nothing but a local Rerun OSS server and files on disk.

## Solution

As a user, I get a single local-only package (`trossen_oss`) that walks the full
pipeline against a local Rerun OSS catalog server:

1. **Acquire data** — on a dev machine, a `data/` directory points at
   already-downloaded episodes; for real users, an idempotent Hugging Face
   download task populates the same `data/` directory. The robot model
   (URDF + meshes) is vendored directly in the repo.
2. **Ingest** — convert each episode MCAP into a faithful base RRD plus a URDF
   model-layer RRD that attaches to it.
3. **Serve** — start a long-lived local Rerun OSS catalog server.
4. **Register** — register the converted RRDs into a dataset on that server,
   keeping the base recording and the URDF layer as a single layered segment.
5. **Query** — run DataFusion queries against the local catalog, both from a
   headless smoke script and from interactive notebooks that embed the Rerun
   viewer.
6. **Train** — *(deferred; see Out of Scope.)*

The cloud-only machinery (S3, Modal, the remote catalog) is removed. The runtime
stays observation-only and reproducible on a laptop/workstation.

## User Stories

1. As a new contributor, I want a one-command environment setup via pixi, so that
   I can run every step without manually installing dependencies.
2. As a developer, I want the package to use the same type-safety stack as
   `examples-monorepo` (pyrefly + beartype + jaxtyping), so that array shapes and
   types are validated statically and at runtime in dev.
3. As a developer, I want beartype runtime checking to activate only in the dev
   environment, so that production/default runs carry no type-checking overhead.
4. As a developer, I want a `typecheck` task that runs pyrefly with tensor-shape
   inference, so that I can catch shape and type errors before running code.
5. As a developer, I want a format/lint task (ruff), so that the codebase stays
   consistent with the wider rerun-projects style.
6. As a dev-machine user, I want a `data/` directory that points at my already
   downloaded episodes, so that I do not re-download ~19 GB to start working.
7. As a real user, I want an idempotent "download" task that fetches episodes from
   Hugging Face into `data/`, so that I can get started without access to the
   internal data mount.
8. As a user, I want the download to be skippable when the data already exists, so
   that re-running tasks does not re-fetch gigabytes.
9. As a user, I want a download task that fetches a small sample (10 episodes), so
   that I can iterate quickly without pulling the whole published dataset.
10. As a user, I want a download-all task that fetches all 100 published episodes,
    so that I can run the full OSS dataset.
11. As a user, I want the robot URDF and meshes vendored in the repo, so that the
    model layer renders without any external download.
12. As a user, I want a `convert`/ingest task that turns one or more episode MCAPs
    into RRDs, so that I can produce catalog-ready artifacts locally.
13. As a user, I want each episode converted into a faithful base recording, so
    that joint signals, forward-kinematics transforms, camera intrinsics, video,
    and the task description all appear in Rerun.
14. As a user, I want a URDF model layer that adds link meshes and the static
    scene on top of the base recording, so that the arms and table render with
    geometry, not just frames.
15. As a user, I want the base recording and its URDF layer to share a recording
    identity, so that they register as one layered segment rather than two
    unrelated ones.
16. As a user, I want generated RRDs written to a dedicated, git-ignored output
    location separate from the read-only source data, so that I never corrupt the
    source episodes.
17. As a user, I want conversion to skip episodes whose outputs already exist
    (unless I force overwrite), so that re-runs are fast and idempotent.
18. As a user, I want to convert a filtered or limited subset of episodes, so that
    I can smoke-test on a single episode before a larger run.
19. As a user, I want a task that starts a local Rerun OSS catalog server, so that
    I can register and query data without the cloud.
20. As a user, I want the server URL to be discoverable/overridable, so that
    client steps can connect to it consistently.
21. As a user, I want a `register` step that creates a dataset and registers the
    base segments plus the URDF layer, so that the catalog reflects my converted
    episodes.
22. As a user, I want re-registration to replace existing segments cleanly, so
    that re-running after a re-convert does not duplicate or error.
23. As a user, I understand the local server is in-memory, so I expect to re-run
    the register step after restarting the server.
24. As a data engineer, I want a headless query smoke script, so that I can verify
    the register→query path in CI without a browser.
25. As a data engineer, I want to list segment metadata (segment ids, layer
    names) from the local catalog, so that I can confirm what was registered.
26. As a data engineer, I want to run a content-level DataFusion query over a real
    entity/timeline, so that I can confirm the data is queryable, not just
    registered.
27. As an analyst, I want interactive notebooks that connect to the local catalog,
    so that I can explore the data with the DataFrame API and SQL.
28. As an analyst, I want the notebooks pre-filled with the real Trossen schema
    (real timelines, entity paths, components), so that I am not editing
    placeholders before anything runs.
29. As an analyst, I want the notebooks to embed the Rerun viewer pointed at the
    local server, so that querying and visual inspection happen in one place.
30. As a user, I want a default blueprint registered on the dataset, so that the
    viewer opens with a sensible layout instead of an unstructured entity dump.
31. As a maintainer, I want conversion correctness covered by tests, so that
    porting the MCAP/URDF logic does not silently regress.
32. As a maintainer, I want the episode discovery/planning logic covered by tests,
    so that path resolution, filtering, and skip behavior stay correct.
33. As a maintainer, I want one end-to-end test that converts, registers to an
    in-process server, and queries, so that the whole local loop is guarded by a
    single high-level test.
34. As a maintainer, I want data-dependent tests to skip gracefully when the data
    mount is absent, so that contributors without the mount can still run the rest
    of the suite.
35. As a user, I want a README documenting the local flow end-to-end, so that I
    can follow the steps without reading the source.
36. As a maintainer, I want the cloud-only modules (S3, Modal, remote catalog
    re-registration) removed, so that the local package has no dead cloud paths.

## Implementation Decisions

### Packaging & toolchain
- The package is `trossen_oss` using a `src/` layout, Python 3.12.
- pixi configuration lives in `pyproject.toml` (not a standalone `pixi.toml`),
  with a `common` feature (runtime deps) and a `dev` feature (tooling), and a
  `dev` environment that composes both.
- The `dev` feature sets `PIXI_DEV_MODE=1` via its activation env. The package
  `__init__` activates beartype's import-time checker (`beartype_this_package()`)
  only when `PIXI_DEV_MODE == "1"`, matching the `examples-monorepo` convention.
- Static checking uses a root `pyrefly.toml` with tensor-shape inference enabled
  and search/site-package paths pointing at the `dev` environment; exposed as a
  `typecheck` task. Linting/formatting via ruff.
- Arrays are annotated with jaxtyping (`<0.3`) at assignment time; ported
  dataclasses use pyserde `@serde`.

### Dependencies
- Runtime: `rerun-sdk[catalog,datafusion]==0.33.0` (from PyPI), `numpy`,
  `pyarrow`, `jaxtyping`, `pyserde`, `huggingface_hub` (Xet transfer; `hf-transfer`
  was dropped), `tyro`, `tqdm`, plus video decode support (av/ffmpeg).
  conda-forge channel. `jupyterlab`/`dataloader` are added at the notebook/training steps.
- Removed relative to the proof-of-concept: `boto3`/S3, `modal`, and the remote
  catalog re-registration path.

### Data acquisition
- A `data/` directory is the single read location for source episodes. On a dev
  machine it is a git-ignored symlink to the existing episode directory; for real
  users an idempotent Hugging Face download task populates the same directory
  (`--local-dir data/`, guarded by an existence check). A placeholder dataset
  `repo_id` is used until the real Hugging Face dataset identity is decided.
- Episode location is overridable via the `--data-dir` CLI flag (default `data/`),
  so nothing is hard-bound to the symlink. Paths are plain `PreprocessingConfig`
  fields (tyro flags) rather than environment variables.
- HF-resolved paths must not be canonicalized to the cache blob; the
  extension-bearing path is preserved so Rerun's loader accepts the file.
- The published OSS dataset is capped at 100 episodes (the internal corpus has
  1024, but only 100 are published). The `download` task fetches 10 sample
  episodes; `download-all` fetches all 100.
- The robot model (URDF files + meshes) is vendored in the repo under an assets
  directory and is never downloaded.
- Generated RRDs are written to a dedicated git-ignored output directory, always
  distinct from `data/`.

### Conversion (ingest)
- Ported from the proof-of-concept largely unchanged: a chunk-stream conversion
  produces a base recording, and a separate stream produces a URDF model layer.
- Each episode yields two RRDs (base + URDF layer) that share one `recording_id`
  so the layer attaches to the base segment. Streams are collected with the
  object-store optimization profile before writing.
- The base recording's data contract (Rerun entities/components):
  - `/world` — root transform bridging the named world frame to the viewer root
    (child frame only, no parent).
  - `/transforms_static` — world→arm-base offsets (static).
  - `/{arm}/joints/{joint_name}` — per-joint scalar series on a `timestamp`
    timeline (one entity per joint; 8 joints per arm, both arms).
  - `/{arm}/gripper/{position,current,claw_state}` — gripper scalar/enum signals.
  - camera entities (high/low external + per-wrist) carry a static pinhole
    (intrinsics + resolution); a child `/video` entity carries an identity
    transform plus an H.264 video stream.
  - `/task_description` — the per-episode language instruction as text, also
    surfaced on the recording properties.
- Two MCAP readers are used deliberately: a foxglove+protobuf reader for the
  forward-kinematics transforms, and a protobuf-only reader for signals and
  video (the foxglove decoder otherwise drops compressed video). Constant URDF
  bookkeeping transform rows (visual/collision/inertial prefixes) are thinned out.
- The URDF layer adds arm link meshes (driven by the base recording's temporal
  transforms, not static rest-pose edges) and the static scene (table, cell
  frame, fixed external camera poses); collision geometries are dropped. The
  layer verifies every frame reaches the world root before emitting.

### Local server, registration, and dataset identity
- Step 3 and step 4 are separate. A long-lived local Rerun OSS catalog server is
  started (CLI `rerun server`), default URL `rerun+http://127.0.0.1:51234`,
  in-memory (no persistence).
- A registration module connects with `CatalogClient(url)`, creates the dataset,
  registers the base RRDs as segments, and registers the URDF RRDs as a named
  layer on those segments, with a replace-on-duplicate policy. Dataset name and
  application id are package constants.
- Because the server is in-memory, registration is expected to re-run after each
  server restart.
- **Open risk to resolve during build (R1):** whether the local OSS server's
  registration accepts a `layer_name`. The catalog round-trip test settles this.
  Fallback if unsupported: merge the URDF content into the base recording at
  conversion time, or register it as a sibling segment.

### Query
- A headless smoke script connects to the local catalog, reads segment metadata,
  and runs one content-level DataFusion query, printing results — runnable
  without a browser.
- Three notebooks are ported from the proof-of-concept and filled in with the
  real schema (timeline `timestamp`, real entity paths and components): an
  overview/welcome notebook, a DataFrame-API analysis notebook, and a SQL
  analysis notebook. Each connects via `CatalogClient` and embeds the Rerun
  viewer pointed at the local server. SQL queries register views on the client's
  DataFusion context; DataFrame queries use the dataset's filtered reader.
- Server URL and dataset name are configuration fields with sensible defaults
  (`CatalogConfig.catalog_url` / `dataset_name`, exposed as `--catalog-url` /
  `--dataset-name`), so the same notebooks work against any catalog URL.

### Blueprint
- The proof-of-concept's default blueprint is ported and registered on the
  dataset so the embedded viewer opens with a sensible layout.

## Testing Decisions

- A good test asserts **external, observable behavior** — which Rerun
  entities/components a conversion emits, what the local catalog contains after
  registration, and whether a query returns rows — never internal implementation
  details (private helpers, intermediate data structures, exact chunk counts).
- **Catalog round-trip seam (new, highest):** one end-to-end test converts an
  episode, registers the base + URDF layer into an in-process local server, and
  queries it, asserting the dataset exists, the expected segment(s) are present,
  the schema exposes the expected entity/component columns, and a DataFusion
  query returns rows. This single seam covers ingest → serve → register → query
  and is where the layer-registration risk (R1) is resolved.
- **Conversion seam (ported):** assertions over the base chunk stream and the
  URDF-layer chunk stream — e.g. the world frame connects to the viewer root,
  each video entity inherits its camera transform, and the arm layers do not log
  static joint transforms while the scene layer does.
- **Planning/discovery seam (ported):** pure functions for target/path
  resolution, episode discovery (natural sort + filter + limit), and skip/
  overwrite planning, exercised with temporary directories and environment mocks.
- Prior art: the proof-of-concept's conversion and local-ingestion tests provide
  the templates for seams 2 and 3; the `examples-monorepo` conftest provides the
  pattern for skipping data-dependent tests when fixtures are absent.
- Tests that need a real episode MCAP (the conversion seam and the round-trip
  seam) skip gracefully when the data mount/episode is unavailable, so the suite
  is runnable without the internal data.

## Out of Scope

- **Training (step 6)** — deferred. The dataloader integration
  (`rerun.experimental.dataloader`, torch) is understood but the task formulation
  (cross-arm vs next-state vs other), inputs (scalars vs video), and model are
  not yet decided. This dataset is observation-only (no commanded action
  channel), so any training task is self-supervised or cross-signal.
- **Cloud paths** — S3 storage, Modal jobs, the hosted catalog, and remote
  dataset re-registration are removed, not ported.
- **The final Hugging Face dataset** — the real `repo_id`/owner and the dataset
  upload itself are TBD; a clearly-marked placeholder is used.
- **Persistent server storage** — the local OSS server is in-memory by design.
- **The full 1024-episode internal corpus** — the published OSS dataset is capped
  at 100 episodes; the larger internal set is not published or processed here.
- **Model quality** — when training lands, the bar is proving the
  register→dataloader→train loop, not accuracy.

## Further Notes

- Sizing: each episode MCAP is ~188 MB (camera H.264 dominates), so the full
  100-episode OSS dataset is ~19 GB of source and the 10-episode `download` sample
  is ~1.9 GB. Base RRDs are the same order (video passes through largely
  unchanged), so budget ~35–40 GB on disk for source + RRDs at the full 100.
- In-memory server implies a predictable workflow: start server → register →
  query, and re-register after any restart.
- When training is built, the dataloader requires
  `torch.multiprocessing.set_start_method("spawn", force=True)` because Rerun's
  runtime is not fork-safe; the `[dataloader]` extra pulls torch/torchvision/av/
  pillow.
- pyrefly's site-package paths resolve only after the dev environment is
  installed, so `pixi install` (dev) precedes the first `typecheck`.
- Build order for piece-by-piece delivery: (0) toolchain scaffold → (1) data +
  vendored assets + download task → (2) conversion + conversion/planning tests →
  (3) server + register + round-trip test (resolve R1) → (4) query smoke +
  notebooks + blueprint → (5) scale to ~100 episodes → (6) training (TBD).
