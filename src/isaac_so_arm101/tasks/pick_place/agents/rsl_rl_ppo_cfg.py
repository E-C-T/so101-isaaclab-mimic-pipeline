# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class PickPlacePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 5000
    save_interval = 50
    experiment_name = "pick_place"
    run_name = ""
    resume = False
    empirical_normalization = False

    actor = RslRlMLPModelCfg(
        class_name="MLPModel",
        hidden_dims=[256, 128, 64],
        activation="elu",
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            class_name="GaussianDistribution",
            init_std=1.0,
            std_type="scalar",
        ),
    )

    critic = RslRlMLPModelCfg(
        class_name="MLPModel",
        hidden_dims=[256, 128, 64],
        activation="elu",
        distribution_cfg=None,
    )

    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )