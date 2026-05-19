from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, AssetBaseCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.managers import TerminationTermCfg as DoneTerm

from isaac_so_arm101.tasks.pick_place.pick_place_env_cfg import PickPlaceEnvCfg
import isaac_so_arm101.tasks.cube_replay.success as replay_success
from isaac_so_arm101.tasks.camera_config.so101_pick_place_cube_same_place import (
            make_so101_up_camera_cfg,
            make_so101_wrist_camera_cfg,
        )

@configclass
class SoArm101CubeReplayEnvCfg(PickPlaceEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5

        self.observations.policy.enable_corruption = False
        self.commands.object_pose.debug_vis = False

        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.2275, 0.015, 0.0],
                rot=[1, 0, 0, 0],
            ),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.45, 0.45, 0.45),
                mass_props=sim_utils.MassPropertiesCfg(
                    mass=0.08,
                ),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 1.0, 1.0)
                ),
            ),
        )

        # self.scene.object = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/Object",
        #     init_state=RigidObjectCfg.InitialStateCfg(
        #         pos=[0.2275, 0.015, 0.015],
        #         rot=[1, 0, 0, 0],
        #     ),
        #     spawn=sim_utils.CuboidCfg(
        #         size=(0.03, 0.03, 0.03),
        #         mass_props=sim_utils.MassPropertiesCfg(
        #             mass=0.08,
        #         ),
        #         rigid_props=RigidBodyPropertiesCfg(
        #             solver_position_iteration_count=64,
        #             solver_velocity_iteration_count=8,
        #             max_angular_velocity=1000.0,
        #             max_linear_velocity=1000.0,
        #             max_depenetration_velocity=0.5,
        #             disable_gravity=False,
        #         ),
        #         collision_props=sim_utils.CollisionPropertiesCfg(
        #             contact_offset=0.005,
        #             rest_offset=0.0,
        #         ),
        #         visual_material=sim_utils.PreviewSurfaceCfg(
        #             diffuse_color=(1.0, 1.0, 1.0),
        #         ),
        #     ),
        # )

        # ---------------------------------------------------------------------
        # Camera sensors for LeRobot / VLA Foundry export
        # ---------------------------------------------------------------------
        # Match the original LeRobot seed dataset:
        #   observation.images.wrist
        #   observation.images.up
        #
        # Internal HDF5 names should be:
        #   camera_obs/wrist
        #   camera_obs/up
        #
        # Use 480x640 @ 30 Hz to match the seed LeRobot dataset.
        if getattr(self, "enable_camera_sensors", False):
            self.scene.up_camera = make_so101_up_camera_cfg()
            self.scene.wrist_camera = make_so101_wrist_camera_cfg()

        # ---------------------------------------------------------------------
        # Goal Region (visual marker + logical bounds for filtering)
        # ---------------------------------------------------------------------
        goal_center = [0.15, 0.25, 0.0005]
        goal_size = [0.15, 0.15, 0.15]   
        goal_x, goal_y, goal_z = goal_center
        goal_l, goal_w, goal_h = goal_size

        # Visual border settings
        line_thickness = 0.005
        visual_height = 0.005

        # Store logical goal bounds on the config so scripts can read them if needed
        self.goal_region = {
            "x_min": goal_x - goal_l / 2.0,
            "x_max": goal_x + goal_l / 2.0,
            "y_min": goal_y - goal_w / 2.0,
            "y_max": goal_y + goal_w / 2.0,
            "z_min": 0.0,
            "z_max": goal_h,
            "max_lin_vel": 0.15,
        }

        # Visual-only red border on the table
        self.scene.goal_zone_top = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/GoalZoneTop",
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=[goal_x, goal_y + goal_l / 2.0, goal_z],
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
            spawn=sim_utils.CuboidCfg(
                size=(goal_l, line_thickness, visual_height),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.0, 0.0),
                ),
            ),
        )

        self.scene.goal_zone_bottom = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/GoalZoneBottom",
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=[goal_x, goal_y - goal_l / 2.0, goal_z],
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
            spawn=sim_utils.CuboidCfg(
                size=(goal_l, line_thickness, visual_height),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.0, 0.0),
                ),
            ),
        )

        self.scene.goal_zone_left = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/GoalZoneLeft",
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=[goal_x - goal_l / 2.0, goal_y, goal_z],
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
            spawn=sim_utils.CuboidCfg(
                size=(line_thickness, goal_l, visual_height),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.0, 0.0),
                ),
            ),
        )

        self.scene.goal_zone_right = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/GoalZoneRight",
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=[goal_x + goal_l / 2.0, goal_y, goal_z],
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
            spawn=sim_utils.CuboidCfg(
                size=(line_thickness, goal_l, visual_height),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.0, 0.0),
                ),
            ),
        )

        self.terminations.success = DoneTerm(
            func=replay_success.object_in_aabb_success,
            params={
                "asset_name": "object",
                **self.goal_region,
            },
        )

        self.events.reset_object_position = None

        self.rewards = {}
        self.curriculum = {}
        # self.terminations = {}
        self.recorders = {}