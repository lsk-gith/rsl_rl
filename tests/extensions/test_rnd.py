# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Random Network Distillation."""

from __future__ import annotations

import torch
import torch.optim as optim
from tensordict import TensorDict

from rsl_rl.extensions.rnd import RandomNetworkDistillation
from tests.conftest import make_obs

NUM_ENVS = 8
OBS_DIM = 6


def _make_rnd(weight: float = 1.0, weight_schedule: dict | None = None) -> tuple[RandomNetworkDistillation, TensorDict]:
    """Create an RND module and matching observations."""
    obs = make_obs(NUM_ENVS, OBS_DIM)
    obs_groups = {"rnd_state": ["policy"]}
    rnd = RandomNetworkDistillation(
        num_states=OBS_DIM,
        obs_groups=obs_groups,
        num_outputs=4,
        predictor_hidden_dims=[32],
        target_hidden_dims=[32],
        weight=weight,
        weight_schedule=weight_schedule,
    )
    return rnd, obs


class TestIntrinsicReward:
    """
    Tests for intrinsic reward computation.
    测试内在奖励的核心功能
    """

    def test_intrinsic_reward_is_non_negative(self) -> None:
        """
        L2 norm distance should always be >= 0.
        目的：验证内在奖励（L2 范数平方）始终 ≥ 0。
        方法：调用 get_intrinsic_reward，断言所有元素 >= 0。
        意义：MSE 天然非负，确保实现没有错误取负值。
        """
        rnd, obs = _make_rnd(weight=1.0)
        rnd.train()

        reward = rnd.get_intrinsic_reward(obs)
        assert torch.all(reward >= 0), "Intrinsic reward (L2 norm) must be non-negative"

    def test_intrinsic_reward_is_finite(self) -> None:
        """
        Intrinsic reward should not contain NaN or Inf.
        目的：确保奖励值不包含 NaN 或 Inf。
        方法：调用 torch.isfinite 检查。
        意义：避免数值不稳定导致训练崩溃。
        """
        rnd, obs = _make_rnd(weight=1.0)
        rnd.train()

        reward = rnd.get_intrinsic_reward(obs)
        assert torch.all(torch.isfinite(reward)), "Intrinsic reward should be finite"

    def test_predictor_training_reduces_reward(self) -> None:
        """
        Training the predictor to match the target should reduce intrinsic reward.
        目的：验证预测网络训练后，内在奖励下降。
        方法：
            记录初始平均内在奖励。
            用 MSE 损失训练预测网络 200 步（目标网络冻结）。
            再次计算内在奖励，断言 final_reward < initial_reward。
            注意：测试中重置了 update_counter，防止权重调度干扰。
            意义：确保学习机制正确：预测网络逐渐匹配目标，误差降低。
        """
        rnd, obs = _make_rnd(weight=1.0)
        rnd.train()

        # Measure initial intrinsic reward
        with torch.no_grad():
            initial_reward = rnd.get_intrinsic_reward(obs).mean().item()

        # Train predictor to match target
        optimizer = optim.Adam(rnd.predictor.parameters(), lr=1e-3)
        rnd_state = rnd.get_rnd_state(obs)

        for _ in range(200):
            optimizer.zero_grad()
            pred = rnd.predictor(rnd_state)
            target = rnd.target(rnd_state).detach()
            loss = torch.nn.functional.mse_loss(pred, target)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            # Reset counter to avoid schedule interference
            rnd.update_counter = 0
            final_reward = rnd.get_intrinsic_reward(obs).mean().item()

        assert final_reward < initial_reward, (
            f"Intrinsic reward should decrease after predictor training: {initial_reward:.4f} -> {final_reward:.4f}"
        )

    def test_zero_weight_produces_zero_reward(self) -> None:
        """
            With weight=0, intrinsic reward should be zero.
            目的：当权重系数为 0 时，最终内在奖励应为 0。
            方法：weight=0.0 构造 RND，断言奖励全零。
            意义：验证 weight 参数正确缩放内在奖励（可用于消融或禁用探索）。
        """
        rnd, obs = _make_rnd(weight=0.0)
        rnd.train()

        reward = rnd.get_intrinsic_reward(obs)
        assert torch.allclose(reward, torch.zeros_like(reward))

    def test_update_counter_increments(self) -> None:
        """
            Each call to get_intrinsic_reward should increment update_counter.
            目的：检查每次调用 get_intrinsic_reward 都会增加内部计数器。
            方法：调用前后比较 update_counter。
            意义：计数器用于支持权重调度（按步数衰减），必须准确累加。
        """
        rnd, obs = _make_rnd()
        assert rnd.update_counter == 0

        rnd.get_intrinsic_reward(obs)
        assert rnd.update_counter == 1

        rnd.get_intrinsic_reward(obs)
        assert rnd.update_counter == 2


class TestWeightSchedules:
    """
    Tests for RND weight scheduling.
    测试权重调度（衰减策略）
    """

    def test_constant_schedule(self) -> None:
        """
        Constant schedule should return the initial weight regardless of step.
        调度模式："constant"
        预期：权重始终等于初始值，不随步数变化。
        验证：调用 10 次 get_intrinsic_reward 后，rnd.weight 仍为 0.5。
        """
        rnd, obs = _make_rnd(
            weight=0.5,
            weight_schedule={"mode": "constant"},
        )
        rnd.train()

        for _ in range(10):
            rnd.get_intrinsic_reward(obs)

        assert rnd.weight == 0.5

    def test_step_schedule_before_final(self) -> None:
        """
        Before final_step, step schedule should return initial weight.
        调度模式："step"，final_step=5，final_value=0.0
        预期：步数 < final_step 时，权重保持初始值 1.0。
        验证：前 4 步后权重仍为 1.0。
        """
        rnd, obs = _make_rnd(
            weight=1.0,
            weight_schedule={"mode": "step", "final_step": 5, "final_value": 0.0},
        )
        rnd.train()

        # Steps 1-4 should use initial weight
        for _ in range(4):
            rnd.get_intrinsic_reward(obs)

        assert rnd.weight == 1.0

    def test_step_schedule_at_final(self) -> None:
        """
        At final_step, step schedule should switch to final_value.
        预期：当步数 等于 final_step 时，权重立即切换为 final_value=0.1。
        验证：执行 3 步后，权重变为 0.1。
        """
        rnd, obs = _make_rnd(
            weight=1.0,
            weight_schedule={"mode": "step", "final_step": 3, "final_value": 0.1},
        )
        rnd.train()

        for _ in range(3):
            rnd.get_intrinsic_reward(obs)

        assert rnd.weight == 0.1

    def test_linear_schedule_interpolation(self) -> None:
        """
        Linear schedule should interpolate between initial and final values.
        调度模式："linear"，initial_step=2，final_step=6，初始权重 1.0 → 最终值 0.0。
        预期：
            步数 < 2 → 权重 = 初始值
            步数 ∈ [2,6] → 线性插值
            步数 ≥ 6 → 权重 = 最终值
        验证：
            第 1 步后权重 = 1.0
            第 2 步后权重 = 1.0（起点）
            第 4 步后：progress = (4-2)/(6-2)=0.5，权重 = 1.0 + 0.5*(0-1)=0.5
            第 6 步后：权重 = 0.0
        """
        initial_weight = 1.0
        final_value = 0.0
        initial_step = 2
        final_step = 6

        rnd, obs = _make_rnd(
            weight=initial_weight,
            weight_schedule={
                "mode": "linear",
                "initial_step": initial_step,
                "final_step": final_step,
                "final_value": final_value,
            },
        )
        rnd.train()

        # Before initial_step
        rnd.get_intrinsic_reward(obs)  # step=1
        assert rnd.weight == initial_weight

        # At initial_step
        rnd.get_intrinsic_reward(obs)  # step=2
        assert rnd.weight == initial_weight

        # Midpoint: step=4, progress = (4-2)/(6-2) = 0.5
        rnd.get_intrinsic_reward(obs)  # step=3
        rnd.get_intrinsic_reward(obs)  # step=4
        progress = (4 - initial_step) / (final_step - initial_step)
        expected_mid = initial_weight + (final_value - initial_weight) * progress
        assert abs(rnd.weight - expected_mid) < 1e-6, f"Expected {expected_mid}, got {rnd.weight}"

        # At final_step
        rnd.get_intrinsic_reward(obs)  # step=5
        rnd.get_intrinsic_reward(obs)  # step=6
        assert abs(rnd.weight - final_value) < 1e-6

    def test_linear_schedule_past_final_step(self) -> None:
        """
        After final_step, linear schedule should return final_value.
        预期：超过 final_step 后，权重保持最终值 0.0。
        验证：执行 10 步后，权重仍为 0.0。
        """
        rnd, obs = _make_rnd(
            weight=1.0,
            weight_schedule={
                "mode": "linear",
                "initial_step": 1,
                "final_step": 3,
                "final_value": 0.0,
            },
        )
        rnd.train()

        for _ in range(10):
            rnd.get_intrinsic_reward(obs)

        assert rnd.weight == 0.0
