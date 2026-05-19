from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

SOARM_USD = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/Healthcare/0.5.0/132c82d/Robots/SO-ARM/SO-ARMDualCamera.usd"
)

SO_ARM101_I4H_USD_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=SOARM_USD,
        visible=True,
        copy_from_source=True,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.005,
            rest_offset=0.001,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0), #rot=(0.707, 0.0, 0.0, 0.707),
        joint_pos={
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
            "gripper": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            effort_limit_sim=5.2,
            velocity_limit_sim=6.28,
            stiffness=80.0,
            damping=20.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper"],
            effort_limit_sim=12.0, 
            velocity_limit_sim=31.4,
            stiffness=80.0,
            damping=10.0,
        ),
    },
)

