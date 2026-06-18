"""
Demonstrates how to use Rerun's chunk processing API to assemble a robot recording
from multiple file sources (MCAP, custom data, URDF, …):

- fix recording errors
- add external static data
- compute joint transforms using URDF
- insert URDF assets
- …

The resulting merged stream is saved to an RRD file, which can be
opened in the Rerun viewer or registered to a dataset catalog.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import rerun as rr
import rerun.blueprint as rrb
from rerun.experimental import (
    Chunk,
    DeriveLens,
    LazyChunkStream,
    McapReader,
    MutateLens,
    OptimizationProfile,
    Selector,
)
from rerun.urdf import UrdfTree


def json_transforms_stream(json_path: Path) -> LazyChunkStream:
    """Loads transform data saved in JSON as a chunk stream of static Transform3D."""
    with json_path.open() as f:
        transforms = json.load(f)["transforms"]

    chunk = Chunk.from_columns(
        "/tf_static/robot_offsets",
        indexes=[],
        columns=rr.Transform3D.columns(
            translation=[transform["translation"] for transform in transforms],
            quaternion=[transform["quaternion_xyzw"] for transform in transforms],
            parent_frame=[transform["parent"] for transform in transforms],
            child_frame=[transform["child"] for transform in transforms],
        ),
    )
    return LazyChunkStream.from_iter([chunk])


def change_albedo_factor_lens(new_albedo: rr.components.AlbedoFactor) -> MutateLens:
    """Replaces Asset3D albedo factors with a fixed color."""

    return MutateLens(
        "Asset3D:albedo_factor",
        Selector(".").pipe(lambda old_albedo: pa.array([new_albedo] * len(old_albedo), type=old_albedo.type)),
    )


# The MCAP importer tags each camera video with a foxglove frame_id that does NOT
# match the camera Pinhole's image-plane frame, so the video lands in an orphan
# frame and never appears in the 2D camera view. Re-home each video onto its
# Pinhole's image-plane frame (the Pinhole's `child_frame`, "<optical>_image_plane").
CAMERA_IMAGE_PLANE_FRAMES: dict[str, str] = {
    "/external/cam_high/video_compressed": "cam_high_optical_image_plane",
    "/external/cam_low/video_compressed": "cam_low_optical_image_plane",
    "/robot_left/wrist_camera/video_compressed": "cam_link_optical_left_image_plane",
    "/robot_right/wrist_camera/video_compressed": "cam_link_optical_right_image_plane",
}


def place_video_in_frame_lens(frame: str) -> MutateLens:
    """Overwrite a camera video's coordinate frame so it renders in the 2D camera view."""
    return MutateLens(
        "CoordinateFrame:frame",
        Selector(".").pipe(lambda frames: pa.array([frame] * len(frames), type=frames.type)),
    )


def joints_batch_lens(robot_urdf: UrdfTree, to_entity: str = "/tmp") -> DeriveLens:
    """Computes intermediate transform batches from each joint state message using the URDF."""
    joint_names: list[str] = [joint.name for joint in robot_urdf.joints() if joint.joint_type != "fixed"]
    return DeriveLens("schemas.proto.JointState:message", output_entity=to_entity).to_component(
        "rerun.urdf.JointTransformBatch",
        Selector(".").pipe(
            lambda joint_state_messages: robot_urdf.compute_joint_transform_batches(
                names=pa.array([joint_names] * len(joint_state_messages), type=pa.list_(pa.string())),
                values=Selector(".joint_positions").execute(joint_state_messages),
            )
        ),
    )


def output_transforms_lens() -> DeriveLens:
    """Scatters transform batches into final Transform3D rows per joint."""
    return (
        DeriveLens("rerun.urdf.JointTransformBatch", output_entity="/tf", scatter=True)
        .to_component(
            rr.Transform3D.descriptor_translation(),
            Selector(".[].translation"),
        )
        .to_component(
            rr.Transform3D.descriptor_quaternion(),
            Selector(".[].quaternion"),
        )
        .to_component(
            rr.Transform3D.descriptor_parent_frame(),
            Selector(".[].parent_frame"),
        )
        .to_component(
            rr.Transform3D.descriptor_child_frame(),
            Selector(".[].child_frame"),
        )
    )


def _nth_position(index: int) -> Callable[[pa.Array], pa.Array]:
    """Select the index-th element of each per-row joint-position list."""

    def extract(positions: pa.Array) -> pa.Array:
        # pyarrow.compute generates list_element at runtime; its type stub omits it.
        return pc.list_element(positions, index)  # type: ignore[missing-attribute]

    return extract


def _as_f64(values: pa.Array) -> pa.Array:
    """Cast a per-row scalar column to float64 for the Scalars component."""
    return pc.cast(values, pa.float64())


def joint_scalar_lenses(robot_urdf: UrdfTree, arm: str) -> list[DeriveLens]:
    """Derive one plottable Scalars series per joint from each JointState message.

    The reference example only turns joint states into FK transforms; here we
    additionally split the N-wide ``joint_positions`` vector into one ``Scalars``
    entity per joint (``/{arm}/joints/{name}``) so a TimeSeriesView can plot them.
    A lens emits one instance per row, so per-joint entities (not a single packed
    entity) are what give one line — and a legend label — per joint.
    """
    joint_names: list[str] = [joint.name for joint in robot_urdf.joints() if joint.joint_type != "fixed"]
    return [
        DeriveLens("schemas.proto.JointState:message", output_entity=f"/{arm}/joints/{name}").to_component(
            rr.Scalars.descriptor_scalars(),
            Selector(".joint_positions").pipe(_nth_position(index)),
        )
        for index, name in enumerate(joint_names)
    ]


def gripper_scalar_lenses(arm: str) -> list[DeriveLens]:
    """Derive the gripper position/current Scalars from each GripperStatus message."""
    entity: str = f"/{arm}/gripper"
    return [
        DeriveLens("schemas.proto.GripperStatus:message", output_entity=f"{entity}/position").to_component(
            rr.Scalars.descriptor_scalars(),
            Selector(".position").pipe(_as_f64),
        ),
        DeriveLens("schemas.proto.GripperStatus:message", output_entity=f"{entity}/current").to_component(
            rr.Scalars.descriptor_scalars(),
            Selector(".current").pipe(_as_f64),
        ),
    ]


def robot_data_blueprint() -> rrb.Blueprint:
    """Build a default viewer layout for the robot preprocessing recording."""
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial3DView(
                    origin="/",
                    name="Scene",
                    contents=[
                        "+ /tf/**",
                        "+ /tf_static/**",
                        "+ /robot_left/**",
                        "+ /robot_right/**",
                        "+ /external/**",
                        "+ /transforms_static/**",
                        "+ /trossen_ai_scene/**",
                    ],
                    spatial_information=rrb.SpatialInformation(target_frame="world"),
                ),
                rrb.Vertical(
                    rrb.Grid(
                        rrb.Spatial2DView(origin="/external/cam_high", name="High"),
                        rrb.Spatial2DView(origin="/external/cam_low", name="Low"),
                        rrb.Spatial2DView(origin="/robot_left/wrist_camera", name="Left wrist"),
                        rrb.Spatial2DView(origin="/robot_right/wrist_camera", name="Right wrist"),
                        grid_columns=2,
                        name="Cameras",
                    ),
                    rrb.DataframeView(
                        origin="/",
                        contents=[
                            "+ /robot_left/joint_states",
                            "+ /robot_right/joint_states",
                            "+ /robot_left/gripper_status",
                            "+ /robot_right/gripper_status",
                        ],
                        name="Robot state",
                        query=rrb.archetypes.DataframeQuery(timeline="message_log_time", apply_latest_at=True),
                    ),
                    row_shares=[3, 2],
                ),
                column_shares=[3, 2],
            ),
            rrb.Horizontal(
                rrb.TimeSeriesView(origin="/robot_left/joints", name="Left joints"),
                rrb.TimeSeriesView(origin="/robot_right/joints", name="Right joints"),
                rrb.TimeSeriesView(
                    origin="/",
                    name="Grippers",
                    contents=["+ /robot_left/gripper/**", "+ /robot_right/gripper/**"],
                ),
                name="Signals",
            ),
            row_shares=[7, 3],
        ),
        rrb.TimePanel(timeline="message_log_time", play_state=rrb.components.PlayState.Following),
        collapse_panels=True,
    )


@dataclass
class PreprocessingConfig:
    """Configuration for the robot-data chunk-processing pipeline."""

    data_dir: Path = Path("data")
    """Directory containing the input data files (MCAP, …)."""
    urdf_dir: Path = Path("assets/urdf")
    """Directory containing the robot/scene URDFs and offsets.json."""
    output_dir: Path = Path("outputs")
    """Directory where the processed output files will be saved."""
    num_rrd_to_process: int | None = None
    """Optionally limit the number of RRD files to process (for testing)."""


def main(cfg: PreprocessingConfig) -> None:
    """Run the main chunk-processing pipeline for this example."""
    cfg.output_dir.mkdir(exist_ok=True)

    # Create a chunk stream from the MCAP file.
    # The reader uses Rerun's MCAP importer (like the viewer or `rerun mcap convert` CLI),
    # so we get Rerun components that we can process in-stream.
    mcap_stream = McapReader(cfg.data_dir / "episode_0_proto.mcap").stream()

    # The world-to-base transform offsets of the two robots are stored in a separate JSON file.
    robot_offsets_stream = json_transforms_stream(cfg.urdf_dir / "offsets.json")

    # Load the same robot URDF twice, with distinct entity path and frame name prefixes for each robot.
    robot_urdf_left = UrdfTree.from_file_path(
        cfg.urdf_dir / "robot.urdf",
        entity_path_prefix="robot_left",
        frame_prefix="left_",
        static_transform_entity_path="/tf_static/left_robot",
    )
    robot_urdf_right = UrdfTree.from_file_path(
        cfg.urdf_dir / "robot.urdf",
        entity_path_prefix="robot_right",
        frame_prefix="right_",
        static_transform_entity_path="/tf_static/right_robot",
    )
    # Load the scene URDF (table & external cameras).
    scene_urdf = UrdfTree.from_file_path(cfg.urdf_dir / "scene.urdf", static_transform_entity_path="/tf_static/scene")

    # NB: the upstream robot_data_preprocessing example swaps Pinhole:resolution for
    # cam_high/cam_low because *its* example MCAP has a swapped-resolution bug. This
    # dataset's calibration is already correct (648x480, matching the video), so we
    # intentionally do NOT apply that swap — doing so would reverse width/height.

    # Re-home each camera video onto its Pinhole's image-plane frame so it renders
    # in the 2D camera view (see CAMERA_IMAGE_PLANE_FRAMES for why this is needed).
    for video_entity, image_plane_frame in CAMERA_IMAGE_PLANE_FRAMES.items():
        mcap_stream = mcap_stream.lenses(
            place_video_in_frame_lens(image_plane_frame),
            content=video_entity,
            output_mode="forward_unmatched",
        )

    # For each robot, compute the joint transforms in batches and convert to the final Transform3D chunks.
    # We keep the original joint states in the stream ("forward_all") while dropping the temporary batch values.
    mcap_stream = (
        mcap_stream.lenses(
            joints_batch_lens(robot_urdf_left), content="/robot_left/joint_states", output_mode="forward_all"
        )
        .lenses(output_transforms_lens(), content="/tmp", output_mode="drop_unmatched")
        .lenses(joints_batch_lens(robot_urdf_right), content="/robot_right/joint_states", output_mode="forward_all")
        .lenses(output_transforms_lens(), content="/tmp", output_mode="drop_unmatched")
    )

    # Beyond the reference example: also split each JointState/GripperStatus message
    # into per-signal Scalars so the blueprint can show joint/gripper line graphs.
    # "forward_all" keeps the original messages in the stream (for the dataframe view).
    mcap_stream = (
        mcap_stream.lenses(
            joint_scalar_lenses(robot_urdf_left, "robot_left"),
            content="/robot_left/joint_states",
            output_mode="forward_all",
        )
        .lenses(
            joint_scalar_lenses(robot_urdf_right, "robot_right"),
            content="/robot_right/joint_states",
            output_mode="forward_all",
        )
        .lenses(gripper_scalar_lenses("robot_left"), content="/robot_left/gripper_status", output_mode="forward_all")
        .lenses(gripper_scalar_lenses("robot_right"), content="/robot_right/gripper_status", output_mode="forward_all")
    )

    # We also modify each robot's visual meshes to have custom colors / transparency by mutating the albedo factor.
    robot_urdf_left_stream = robot_urdf_left.stream().lenses(
        change_albedo_factor_lens(rr.components.AlbedoFactor([80, 120, 175, 125])),
        content="/robot_left/wxai/visual_geometries/**",
        output_mode="forward_unmatched",
    )
    robot_urdf_right_stream = robot_urdf_right.stream().lenses(
        change_albedo_factor_lens(rr.components.AlbedoFactor([200, 120, 90, 125])),
        content="/robot_right/wxai/visual_geometries/**",
        output_mode="forward_unmatched",
    )

    # Drop the collision meshes from each URDF.
    # (you can also disable them in the viewer, but here we demonstrate how to drop them entirely)
    robot_urdf_left_stream = robot_urdf_left_stream.drop(content="/robot_left/wxai/collision_geometries/**")
    robot_urdf_right_stream = robot_urdf_right_stream.drop(content="/robot_right/wxai/collision_geometries/**")

    # Merge the streams in logical groups (base recording and URDF data).
    # (alternatively we could also merge everything in one stream here, if desired)
    data_stream = LazyChunkStream.merge(
        mcap_stream,
        robot_offsets_stream,
    )
    urdf_stream = LazyChunkStream.merge(
        robot_urdf_left_stream,
        robot_urdf_right_stream,
        scene_urdf.stream(),
    )

    # Run the pipeline, materialize into a ChunkStore and optimize it before writing to an RRD.
    # Here we use an optimization profile suited for object-store (query & stream applications).
    data_stream.collect(optimize=OptimizationProfile.OBJECT_STORE).write_rrd(
        cfg.output_dir / "data.rrd",
        application_id="rerun_example_robot_data_preprocessing",
        recording_id="episode",
    )
    # Write also the URDF streams to an RRD.
    # Note how we use the same `recording_id` here to group the two RRD layers into the same logical recording.
    # https://rerun.io/docs/concepts/logging-and-ingestion/recordings#logical-vs-physical-recordings
    urdf_stream.collect(optimize=OptimizationProfile.OBJECT_STORE).write_rrd(
        cfg.output_dir / "urdf.rrd",
        application_id="rerun_example_robot_data_preprocessing",
        recording_id="episode",
    )
    blueprint = robot_data_blueprint()
    blueprint.save(
        "rerun_example_robot_data_preprocessing",
        cfg.output_dir / "robot_data_preprocessing.rbl",
    )

    print(f"\nWrote output RRDs to: {cfg.output_dir}")
    print(f"Wrote blueprint to: {cfg.output_dir / 'robot_data_preprocessing.rbl'}")
