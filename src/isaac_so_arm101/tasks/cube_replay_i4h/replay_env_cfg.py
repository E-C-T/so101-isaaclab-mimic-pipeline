from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils import configclass

from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaac_so_arm101.tasks.pick_place.pick_place_env_cfg import PickPlaceEnvCfg
import isaac_so_arm101.tasks.cube_replay_i4h.success as replay_success


@configclass
class SoArm101CubeReplayI4HEnvCfg(PickPlaceEnvCfg):
    """Base SO-101 cube replay scene using the I4H SO-ARM USD backend.

    This cfg defines the cube, goal region, optional cameras, and success
    condition. The robot asset, action interface, and EE frame are added in
    cube_replay_i4h/joint_pos_env_cfg.py.
    """

    def __post_init__(self):
        super().__post_init__()

        # ---------------------------------------------------------------------
        # Scene/global settings
        # ---------------------------------------------------------------------
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5

        self.observations.policy.enable_corruption = False
        self.commands.object_pose.debug_vis = False

        # ---------------------------------------------------------------------
        # Object: keep same cube task, but use clean cuboid collision for contact
        # ---------------------------------------------------------------------
        # For size=(0.03, 0.03, 0.03), center z should be 0.015.
        # self.scene.object = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/Object",
        #     init_state=RigidObjectCfg.InitialStateCfg(
        #         pos=[0.1873, 0.015, 0.0127],
        #         rot=[1.0, 0.0, 0.0, 0.0],
        #     ),
        #     spawn=sim_utils.CuboidCfg(
        #         size=(0.0254, 0.0254, 0.0254),
        #         mass_props=sim_utils.MassPropertiesCfg(
        #             mass=0.05,
        #         ),
        #         rigid_props=RigidBodyPropertiesCfg(
        #             solver_position_iteration_count=16,
        #             solver_velocity_iteration_count=1,
        #             max_angular_velocity=1000.0,
        #             max_linear_velocity=1000.0,
        #             max_depenetration_velocity=1.0,
        #             disable_gravity=False,
        #         ),
        #         collision_props=sim_utils.CollisionPropertiesCfg(
        #             contact_offset=0.001,
        #             rest_offset=0.0,
        #         ),
        #         visual_material=sim_utils.PreviewSurfaceCfg(
        #             diffuse_color=(1.0, 0.0, 0.0),
        #         ),
        #     ),
        # )

        # Rigid body properties based on Franka cube-stack reference,
        # but adapted to the SO101 pick-place cube scene.
        cube_properties = RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        )

        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.1873, 0.015, 0.0203],
                rot=[1.0, 0.0, 0.0, 0.0],
            ),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd",
                scale=(0.65, 0.65, 0.65),
                rigid_props=cube_properties,
            ),
        )



        # ---------------------------------------------------------------------
        # Goal Region: same as current URDF cube task
        # ---------------------------------------------------------------------
        goal_center = [0.10, 0.20, 0.0005]
        goal_size = [0.15, 0.15, 0.04]
        goal_x, goal_y, goal_z = goal_center
        goal_l, goal_w, goal_h = goal_size

        line_thickness = 0.005
        visual_height = 0.005

        self.goal_region = {
            "x_min": goal_x - goal_l / 2.0,
            "x_max": goal_x + goal_l / 2.0,
            "y_min": goal_y - goal_w / 2.0,
            "y_max": goal_y + goal_w / 2.0,
            "z_min": 0.0,
            "z_max": goal_h,
            "max_lin_vel": 0.15,
        }

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

        # ---------------------------------------------------------------------
        # Success condition
        # ---------------------------------------------------------------------
        self.terminations.success = DoneTerm(
            func=replay_success.object_in_aabb_success,
            params={
                "asset_name": "object",
                **self.goal_region,
            },
        )

        # Keep deterministic replay/generation behavior.
        self.events.reset_object_position = None

        # This replay/debug cfg does not need rewards/curriculum.
        self.rewards = {}
        self.curriculum = {}

        # Recorder is configured by scripts such as generate_dataset_so101.py.
        self.recorders = {}