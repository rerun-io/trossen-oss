---
name: rerun-mcap
description: Ingest MCAP files into Rerun chunk streams with rerun.experimental.McapReader. Read when converting an MCAP recording, selecting topics or decoders, fixing protobuf schemas that ship without compiled descriptors, or when an MCAP-derived stream comes out empty. Builds on rerun-chunk-processing (stream mechanics) and rerun-data-model (what the topics should become).
user_invocable: true
allowed-tools: Read, Grep, Bash, WebFetch
---

# Rerun MCAP Ingestion

`McapReader` turns an MCAP file into a lazy chunk stream: one entity per topic
at the topic's path, message payloads decoded by pluggable decoders. This
skill is the reader's options and the one failure mode that produces no error:
protobuf schemas without compiled descriptors. Stream mechanics (filter, drop,
lenses, merge, write) are in `rerun-chunk-processing`.

Verified against `rerun-sdk 0.34.0a1`.

## The API

```python
from rerun.experimental import McapReader

reader = McapReader(
    mcap_path,
    timeline_type="timestamp",      # or "duration" (ns offsets instead of wall-clock)
    timestamp_offset_ns=None,       # added to every timestamp column
    decoders=None,                  # None = all; or a subset of available_decoders()
    include_topic_regex=None,       # RE2 patterns; NOT implicitly anchored
    exclude_topic_regex=None,       # applied after includes
)
stream = reader.stream()
```

`McapReader.available_decoders()` as of `rerun-sdk 0.34.0a1`:
`attachments`, `foxglove`, `metadata`, `protobuf`, `raw`, `recording_info`,
`ros2_reflection`, `ros2msg`, `schema`, `stats`, `urdf`.

The interesting ones: `foxglove` and `ros2msg`/`ros2_reflection` decode
well-known message types into Rerun archetypes; `protobuf` reflection-decodes
custom protobuf messages into struct components; `raw` passes payloads through
undecoded; `urdf` ingests a URDF embedded in the MCAP (then see `rerun-urdf`);
the rest emit MCAP metadata (schemas, stats, attachments) as entities.

## What a topic becomes

With the default decoders (`decoders=None`), the message **schema name** decides
what a topic becomes — **pass archetypes through, lens only the raw `:message`
topics**:

| MCAP schema name | decodes to | what to do |
| --- | --- | --- |
| `foxglove.FrameTransforms` | `Transform3D` | pass through; do **not** hand-build |
| `foxglove.CameraCalibration` | `Pinhole` | pass through |
| `foxglove.CompressedVideo` | `VideoStream` (real sample bytes) + `CoordinateFrame` | pass through |
| other `foxglove.*` well-known types | the matching archetype | pass through |
| ros2 well-known types (`ros2msg` / `ros2_reflection`) | archetype | pass through |
| your own `schemas.proto.*` / custom protobuf | one `<schema.name>:message` struct | attach semantics with a `DeriveLens` + `Selector` |

So a camera topic already arrives as `Pinhole`, its video as `VideoStream`, and
a `frame_transforms` topic as `Transform3D` — only custom signal messages (joint
states, gripper status) come through raw and need lenses.

The `foxglove` decoder does the schema→archetype mapping; because foxglove
messages are protobuf-*encoded* it rides on the `protobuf` decoder, so keep
`decoders=None` (verified: `decoders=["protobuf"]` alone leaves
`foxglove.CameraCalibration` a raw `:message`; adding `foxglove` makes it a
`Pinhole`). Confirm on your file: `McapReader(path).stream()`, then read
`McapSchema:name` and a few `Chunk.format()` before deciding anything is missing
or needs rebuilding.

- Entity path = topic name (`/sensors/joint_states` stays
  `/sensors/joint_states`). Filter early:
  `McapReader(path).stream().filter(content="/sensors/**")`.
- A reflection-decoded message lands as one struct component named
  `<fully.qualified.MessageName>:message`. Navigate it with `Selector`
  (`Selector(".joint_positions")`) inside lenses; this is how custom messages
  get Rerun semantics attached (see the DeriveLens patterns in
  `rerun-chunk-processing`).
- Topic regexes use RE2 syntax and are **not anchored**: `cam` matches
  `/external/cam_low` and `/camera_info`. Anchor explicitly (`^/external/cam`)
  when it matters. Prefer reader-level topic filtering over `.filter(...)`
  when you can, so excluded topics are never decoded at all.

## Protobuf descriptors: the silent empty-stream failure

A protobuf channel only decodes when its MCAP schema record carries a compiled
`FileDescriptorSet`. Raw robot MCAPs often ship the schema `name` with empty
`data`; the reader then yields **zero rows for that topic with no error**, and
everything downstream (FK layers, derived scalars) comes out empty.

Fix it by patching the descriptor into the schema before any reader touches
the file. Compile the matching `.proto`:

```bash
protoc --proto_path=<proto_dir> --include_imports \
  --descriptor_set_out=schema.desc <message>.proto
```

Then rewrite the MCAP with the `mcap` pip package: re-register every schema,
channel, and message verbatim, replacing only the target schema's `data` with
the `FileDescriptorSet` bytes:

```python
from mcap.reader import make_reader
from mcap.writer import Writer

descriptor_data = Path("schema.desc").read_bytes()
with input_path.open("rb") as inf, output_path.open("wb") as outf:
    reader = make_reader(inf)
    summary = reader.get_summary()          # requires a summarized MCAP
    writer = Writer(outf)
    writer.start(profile=reader.get_header().profile)

    schema_ids, channel_ids = {}, {}
    for old_id, schema in sorted(summary.schemas.items()):
        data = descriptor_data if schema.name == TARGET_SCHEMA_NAME else schema.data
        schema_ids[old_id] = writer.register_schema(name=schema.name, encoding=schema.encoding, data=data)
    for old_id, channel in sorted(summary.channels.items()):
        channel_ids[old_id] = writer.register_channel(
            topic=channel.topic,
            message_encoding=channel.message_encoding,
            schema_id=0 if channel.schema_id == 0 else schema_ids[channel.schema_id],
            metadata=channel.metadata,
        )
    for _, _, message in reader.iter_messages(log_time_order=False):
        writer.add_message(
            channel_id=channel_ids[message.channel_id],
            log_time=message.log_time,
            publish_time=message.publish_time,
            sequence=message.sequence,
            data=message.data,
        )
    writer.finish()
```

The descriptor's fully-qualified message name must equal the schema `name` in
the MCAP (`TARGET_SCHEMA_NAME` above), or the reader still cannot bind it.
ROS2/CDR and already-descriptored sources skip all of this.

## When to use the low-level `mcap` package instead

`McapReader` keeps payloads in columnar chunk streams; that is almost always
what you want. Drop to `mcap.reader.make_reader` only when you need raw record
metadata and no payloads (e.g. reading `log_time` of empty marker messages
used as frame pointers), or when rewriting the container itself as in the
descriptor patch above.

## Gotchas

1. Empty stream, no error: missing protobuf descriptor (above), or a topic
   regex that matched nothing. Check `Chunk.format()` on a few chunks of
   `reader.stream().to_chunks()` against a tiny test file, or compare topic
   names with the `mcap` CLI / package first.
2. Topic regexes are unanchored RE2; excludes run after includes.
3. `timeline_type="timestamp"` interprets MCAP log times as wall-clock ns
   since epoch. If the recording's clock is wrong, fix it at the reader with
   `timestamp_offset_ns` rather than mutating timestamps downstream.
4. Decoder subsets silently skip topics no decoder claims; when a topic is
   missing, retry with `decoders=None` to rule out decoder selection.
5. Example fix-lenses are dataset-specific. Before copying a `MutateLens` like
   the `Pinhole:resolution` swap from the `robot_data_preprocessing` example,
   read the raw component from `McapReader(path).stream()` and confirm the defect
   exists in *your* data — applied blindly it corrupts correct calibration (a
   correct 648×480 flipped to 480×648).
6. `foxglove` derives both the camera's `Pinhole:child_frame` and the video's
   `CoordinateFrame:frame` from each message's `.frame_id` (plus an image-plane
   suffix), so they **match** when the calibration and video topics share a
   `frame_id`. Only when those topics carry *different* `frame_id`s does the
   video frame diverge and orphan the video from its image plane — re-home it
   then with a per-camera `MutateLens("CoordinateFrame:frame", ...)`.

## References

- End-to-end MCAP pipeline:
  `https://github.com/rerun-io/rerun/tree/main/examples/python/robot_data_preprocessing`
- `rerun-chunk-processing` (stream/lens mechanics), `rerun-urdf` (FK from
  joint-state topics), `rerun-data-model` (modeling decisions)
