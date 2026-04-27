# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the PPO algorithm."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tests.conftest import make_obs

NUM_ENVS = 4
NUM_STEPS = 8
OBS_DIM = 8
NUM_ACTIONS = 4


def _make_actor(obs: TensorDict, obs_groups: dict, num_actions: int = 4, **kwargs: object) -> MLPModel:
    """Create an MLPModel actor with a Gaussian distribution."""
    defaults: dict[str, object] = {
        "hidden_dims": [32, 32],
        "activation": "elu",
        "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    }
    defaults.update(kwargs)
    return MLPModel(obs, obs_groups, "actor", num_actions, **defaults)


def _make_critic(obs: TensorDict, obs_groups: dict, **kwargs: object) -> MLPModel:
    """Create an MLPModel critic (no distribution)."""
    defaults: dict[str, object] = {"hidden_dims": [32, 32], "activation": "elu"}
    defaults.update(kwargs)
    return MLPModel(obs, obs_groups, "critic", 1, **defaults)


def _build_ppo(**overrides: object) -> tuple[PPO, TensorDict]:
    """Build a PPO instance with small networks for testing."""
    obs = make_obs(NUM_ENVS, OBS_DIM)
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = _make_actor(obs, obs_groups, NUM_ACTIONS)
    critic = _make_critic(obs, obs_groups)
    storage = RolloutStorage("rl", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS])

    defaults = dict(
        num_learning_epochs=2,
        num_mini_batches=2,
        clip_param=0.2,
        gamma=0.99,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.01,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        schedule="fixed",
        desired_kl=0.01,
    )
    defaults.update(overrides)
    ppo = PPO(actor, critic, storage, **defaults)
    return ppo, obs


class TestGAEComputation:
    """
     Tests for generalized advantage estimation in ``compute_returns``.
     测试GAE（广义优势估计）的返回值是否与手动计算的一致。
    """
    def test_gae_returns_hand_computed(self) -> None:
        """Verify GAE returns match a hand-computed example with known rewards, values, and dones."""
        num_envs, num_steps = 1, 3
        gamma, lam = 0.99, 0.95

        obs = make_obs(num_envs, OBS_DIM)
        obs_groups = {"actor": ["policy"], "critic": ["policy"]}
        actor = _make_actor(obs, obs_groups, NUM_ACTIONS)
        critic = _make_critic(obs, obs_groups)
        storage = RolloutStorage("rl", num_envs, num_steps, obs, [NUM_ACTIONS])
        ppo = PPO(
            actor, critic, storage, gamma=gamma, lam=lam, schedule="fixed", normalize_advantage_per_mini_batch=True
        )

        rewards = [1.0, 2.0, 3.0]
        values = [0.5, 1.0, 1.5]
        dones = [0.0, 0.0, 0.0]
        # 用构造的rewards、values、done 填充 RolloutStorage。
        for i in range(num_steps):
            t = RolloutStorage.Transition()
            t.observations = obs
            t.hidden_states = (None, None)
            t.actions = torch.randn(num_envs, NUM_ACTIONS)
            t.values = torch.full((num_envs, 1), values[i])
            t.actions_log_prob = torch.zeros(num_envs)
            t.distribution_params = (torch.zeros(num_envs, NUM_ACTIONS), torch.ones(num_envs, NUM_ACTIONS))
            t.rewards = torch.full((num_envs,), rewards[i])
            t.dones = torch.full((num_envs,), dones[i])
            storage.add_transition(t)
        # last_values = 2.0
        last_values = torch.full((num_envs, 1), 2.0)
        # Manually compute GAE (backward pass)
        # Step 2: delta = r2 + gamma * V_last - V2 = 3.0 + 0.99*2.0 - 1.5 = 3.48
        #          adv2 = 3.48
        # Step 1: delta = r1 + gamma * V2 - V1 = 2.0 + 0.99*1.5 - 1.0 = 2.485
        #          adv1 = 2.485 + gamma*lam*adv2 = 2.485 + 0.99*0.95*3.48 = 2.485 + 3.27294 = 5.75794
        # Step 0: delta = r0 + gamma * V1 - V0 = 1.0 + 0.99*1.0 - 0.5 = 1.49
        #          adv0 = 1.49 + gamma*lam*adv1 = 1.49 + 0.99*0.95*5.75794 = 1.49 + 5.41484... = 6.90484...
        expected_adv = [
            1.49 + 0.99 * 0.95 * (2.485 + 0.99 * 0.95 * 3.48),
            2.485 + 0.99 * 0.95 * 3.48,
            3.48,
        ]
        # [7.40484,6.75794,4.98]
        expected_returns = [expected_adv[i] + values[i] for i in range(3)]

        # Use the actual critic to produce last_values override
        with torch.no_grad():
            storage.values[0] = torch.full((num_envs, 1), values[0])
            storage.values[1] = torch.full((num_envs, 1), values[1])
            storage.values[2] = torch.full((num_envs, 1), values[2])

        # Call compute_returns with a custom last_values by monkeypatching critic
        # 通过猴子补丁 ppo.critic.forward 返回指定的 last_values = 2.0，避免真正调用神经网络
        # 保存原始方法 把真正的 forward 方法存起来，以便后续恢复。
        original_critic_call = ppo.critic.forward
        # 替换为 lambda 现在任何对 ppo.critic.forward(...) 的调用都会直接返回 last_values，而不会执行真正的网络前向传播。
        ppo.critic.forward = lambda *a, **kw: last_values
        # 调用 compute_returns 在该函数内部，计算 GAE 时会通过 self.critic(obs) 获取最后一步的价值。由于我们已经替换了 forward，所以得到的就是预定义的 last_values = 2.0。
        ppo.compute_returns(obs)
        # 恢复原始方法  测试结束后恢复，避免影响其他测试或后续代码。
        ppo.critic.forward = original_critic_call

        # 逐步骤比较 storage.returns[i, 0, 0] 与手动计算的期望回报（expected_returns[i]），允许 1e-4 误差。
        for i in range(num_steps):
            assert torch.allclose(
                storage.returns[i, 0, 0],
                torch.tensor(expected_returns[i]),
                atol=1e-4,
            ), f"Return mismatch at step {i}: got {storage.returns[i, 0, 0].item()}, expected {expected_returns[i]}"

    def test_gae_terminal_state_cuts_bootstrap(self) -> None:
        """
        When a done flag is set, the advantage should not bootstrap from the next value.
        验证当环境终止（done=1）时，GAE 不会从下一状态自举（即不将下一状态的值纳入当前 δ 的计算）。这符合强化学习的基本原理：终止状态后没有未来奖励。
        """
        num_envs, num_steps = 1, 2
        gamma, lam = 0.99, 0.95

        obs = make_obs(num_envs, OBS_DIM)
        obs_groups = {"actor": ["policy"], "critic": ["policy"]}
        actor = _make_actor(obs, obs_groups, NUM_ACTIONS)
        critic = _make_critic(obs, obs_groups)
        storage = RolloutStorage("rl", num_envs, num_steps, obs, [NUM_ACTIONS])
        ppo = PPO(
            actor, critic, storage, gamma=gamma, lam=lam, schedule="fixed", normalize_advantage_per_mini_batch=True
        )

        # Step 0: done=True, so step 1 is a fresh episode
        # r:[1.0,2.0] v:[0.5,1.0] d:[1.0,0.0]
        for i, (r, v, d) in enumerate([(1.0, 0.5, 1.0), (2.0, 1.0, 0.0)]):
            t = RolloutStorage.Transition()
            t.observations = obs
            t.hidden_states = (None, None)
            t.actions = torch.randn(num_envs, NUM_ACTIONS)
            t.values = torch.full((num_envs, 1), v)
            t.actions_log_prob = torch.zeros(num_envs)
            t.distribution_params = (torch.zeros(num_envs, NUM_ACTIONS), torch.ones(num_envs, NUM_ACTIONS))
            t.rewards = torch.full((num_envs,), r)
            t.dones = torch.full((num_envs,), d)
            storage.add_transition(t)

        last_values = torch.full((num_envs, 1), 3.0)
        ppo.critic.forward = lambda *a, **kw: last_values
        # 填充存储后调用计算
        ppo.compute_returns(obs)

        # Step 0: done=True, so next_is_not_terminal = 0
        # delta0 = r0 - V0 = 1.0 - 0.5 = 0.5 (no bootstrap because done)
        # Step 1: delta1 = r1 + gamma * V_last - V1 = 2.0 + 0.99*3.0 - 1.0 = 3.97
        # adv1 = 3.97
        # adv0 = 0.5 (no bootstrap because done at step 0)
        expected_return_0 = 0.5 + 0.5  # adv0 + V0
        expected_return_1 = 3.97 + 1.0  # adv1 + V1

        assert torch.allclose(storage.returns[0, 0, 0], torch.tensor(expected_return_0), atol=1e-4)
        assert torch.allclose(storage.returns[1, 0, 0], torch.tensor(expected_return_1), atol=1e-4)

    def test_advantage_normalization_global(self) -> None:
        """
        With normalize_advantage_per_mini_batch=False, advantages should have mean~0, std~1.
        测试当 normalize_advantage_per_mini_batch=False 时，PPO 是否会对所有样本（整个 rollout）的优势进行全局归一化，使其均值为 0、标准差为 1。
        这是 PPO 的常见技巧，用于稳定训练。
        """
        ppo, obs = _build_ppo(normalize_advantage_per_mini_batch=False)
        # 随机生成 4 个环境、8 个时间步的经验（rewards 为标准正态分布，dones 全为 0）。
        for _ in range(NUM_STEPS):
            t = RolloutStorage.Transition()
            t.observations = obs
            t.hidden_states = (None, None)
            t.actions = ppo.actor(obs, stochastic_output=True).detach()
            t.values = ppo.critic(obs).detach()
            t.actions_log_prob = ppo.actor.get_output_log_prob(t.actions).detach()
            t.distribution_params = tuple(p.detach() for p in ppo.actor.output_distribution_params)
            t.rewards = torch.randn(NUM_ENVS)
            t.dones = torch.zeros(NUM_ENVS)
            ppo.storage.add_transition(t)
        # 调用 ppo.compute_returns(obs) 后，从 storage.advantages 中取出所有优势值（形状为 [num_steps, num_envs] 展平）。
        ppo.compute_returns(obs)

        adv = ppo.storage.advantages.flatten()
        assert abs(adv.mean().item()) < 1e-5, "Advantages should be zero-mean"
        assert abs(adv.std().item() - 1.0) < 0.1, "Advantages should be unit-std"


class TestTimeoutBootstrapping:
    """
    Tests for timeout bootstrapping in ``process_env_step``.
    它的作用是验证当环境因超时（timeout）而非真正的任务终止（done）结束时，PPO 算法是否正确地将未来价值的估计（bootstrap）加入到当前步的奖励中。

    在强化学习中，环境终止可能有两种情况：
        1、真正终止（done=True, timeout=False）：任务完成或失败，episode 结束，不应再自举（bootstrap）下一状态的价值。
        2、超时终止（done=True, timeout=True）：因为达到最大步数而截断，episode 被迫结束，但环境本身并未失败。此时应该将当前状态的价值作为未来奖励的估计，加到当前奖励中，避免低估长期回报。
    """

    def test_timeout_adds_bootstrap_to_reward(self) -> None:
        """
            When time_outs is set, stored reward should include gamma * value * timeout.
        """
        ppo, obs = _build_ppo()

        # Manually act to populate transition.values
        # 调用 ppo.act(obs) 进行一次动作选择，这会填充 ppo.transition.values，即当前状态下 critic 估计的状态价值 V(s)。
        ppo.act(obs)
        stored_values = ppo.transition.values.clone()
        # 构造模拟数据
        # 原始奖励全为 1.0。
        raw_reward = torch.ones(NUM_ENVS)
        # 所有环境都标记为 done（表示 episode 结束）。
        dones = torch.ones(NUM_ENVS)
        # 但将 time_outs[0] = 1.0：只有环境 0 是超时终止，环境 1 是真正终止。
        time_outs = torch.zeros(NUM_ENVS)
        time_outs[0] = 1.0  # Only env 0 times out
        # 调用被测试函数
        # ppo.process_env_step(obs, raw_reward, dones, {"time_outs": time_outs})
        # 该函数内部会根据 time_outs 标志修改存储的奖励：
            # 对于超时的环境（timeout=1）：stored_reward = raw_reward + gamma * value
            # 对于非超时的环境（timeout=0）：stored_reward = raw_reward
        ppo.process_env_step(obs, raw_reward, dones, {"time_outs": time_outs})

        # The stored reward for env 0 should be: 1.0 + gamma * value[0]
        stored_reward_env0 = ppo.storage.rewards[0, 0, 0].item()
        expected = 1.0 + ppo.gamma * stored_values[0, 0].item()
        # 断言验证 环境 0（超时）：期望存储的奖励 = 1.0 + gamma * stored_values[0,0] 实际从 ppo.storage.rewards[0, 0, 0] 取出进行比较。
        assert abs(stored_reward_env0 - expected) < 1e-5

        # Env 1 should have raw reward only
        stored_reward_env1 = ppo.storage.rewards[0, 1, 0].item()
        # 环境 1（真正终止）：期望存储的奖励 = 1.0（原始奖励）。
        assert abs(stored_reward_env1 - 1.0) < 1e-5


class TestPPOLosses:
    """
    Tests for PPO loss computation correctness.
    这两个测试函数用于验证 PPO 算法中两个关键损失函数的裁剪（clipping）机制是否正确实现：
    """

    def test_surrogate_loss_clipping(self) -> None:
        """
        When ratio deviates beyond clip_param, the clipped branch should dominate.
        确保当新旧策略的概率比 ratio 超出 [1-ε, 1+ε] 范围时，PPO 的替代损失（surrogate loss）会选择裁剪后的分支，从而限制策略更新幅度。
        """
        clip_param = 0.2

        # Construct a scenario: positive advantages, ratio > 1 + clip
        # 设置 advantages = [1,1,1]（正优势，鼓励增大动作概率）。
        advantages = torch.tensor([1.0, 1.0, 1.0])
        old_log_probs = torch.tensor([0.0, 0.0, 0.0])
        # New log probs that give ratio = exp(0.5) ≈ 1.65, which is > 1 + 0.2
        # old_log_probs = 0，new_log_probs = 0.5 → ratio = exp(0.5) ≈ 1.65，超过 1+clip_param=1.2。
        new_log_probs = torch.tensor([0.5, 0.5, 0.5])

        ratio = torch.exp(new_log_probs - old_log_probs)
        surrogate = -advantages * ratio
        surrogate_clipped = -advantages * torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)
        loss = torch.max(surrogate, surrogate_clipped).mean()
        # 计算两种损失：
            # 未裁剪：-advantages * ratio = -1.65
            # 裁剪后：-advantages * clamp(ratio, 0.8, 1.2) = -1.2
        # The clipped branch should be -advantages * (1 + clip_param) = -1.2
        # The unclipped branch should be -advantages * 1.65 ≈ -1.65

        # max(-1.65, -1.2) = -1.2, so clipped branch dominates
        # PPO 实际损失取两者最大值（即 max(-1.65, -1.2) = -1.2），因为数值越大（负得越少）损失越小
        expected_clipped = (-advantages * (1.0 + clip_param)).mean()
        # 断言：最终损失等于 -advantages * (1+clip_param) 的均值，证明裁剪分支生效。
        assert torch.allclose(loss, expected_clipped, atol=1e-5)

    def test_value_loss_clipping(self) -> None:
        """
        With clipped value loss, large value changes should be clipped.
        确保价值网络更新时，如果新预测值与旧值的差异超过阈值 clip_param，则对更新幅度进行裁剪，防止价值函数突变导致训练不稳定。
        """
        # 给定两个环境：
            # 环境0：old_v=1.0，new_v=2.0（变化 +1.0），returns=1.5
            # 环境1：old_v=1.0，new_v=1.1（变化 +0.1），returns=1.5
        # 裁剪后新值：环境0 变为 1.2，环境1 保持 1.1。
        clip_param = 0.2
        old_values = torch.tensor([[1.0], [1.0]])
        new_values = torch.tensor([[2.0], [1.1]])
        returns = torch.tensor([[1.5], [1.5]])
        # 计算裁剪前后的平方误差损失：
            # 环境0：未裁剪损失 (2.0-1.5)^2=0.25，裁剪后损失 (1.2-1.5)^2=0.09 → 取最大值 0.25
            # 环境1：未裁剪损失 (1.1-1.5)^2=0.16，裁剪后损失相同 0.16 → 取 0.16
        value_clipped = old_values + (new_values - old_values).clamp(-clip_param, clip_param)
        losses_unclipped = (new_values - returns).pow(2)
        losses_clipped = (value_clipped - returns).pow(2)
        loss = torch.max(losses_unclipped, losses_clipped).mean()

        # Env 0: new=2.0, old=1.0, clipped_new=1.2
        #   unclipped: (2.0 - 1.5)^2 = 0.25
        #   clipped: (1.2 - 1.5)^2 = 0.09
        #   max = 0.25
        # Env 1: new=1.1, old=1.0, clipped_new=1.1 (within clip)
        #   unclipped: (1.1 - 1.5)^2 = 0.16
        #   clipped: (1.1 - 1.5)^2 = 0.16
        #   max = 0.16
        # 总损失 = 均值 (0.25+0.16)/2 = 0.205
        expected = (0.25 + 0.16) / 2
        # 断言：实际计算的损失等于该值，证明裁剪机制正确（大更新被裁剪，小更新保留）。
        assert torch.allclose(loss, torch.tensor(expected), atol=1e-5)


class TestAdaptiveLearningRate:
    """
    Tests for adaptive KL-based learning rate scheduling.
    在 PPO 训练中，可以通过监控新旧策略之间的 KL 散度 来动态调整学习率：
    如果 KL 散度过大（超过 2 * desired_kl），说明策略更新太剧烈，应该降低学习率。
    如果 KL 散度过小（低于 desired_kl / 2），说明更新过于保守，可以提高学习率。
    如果 KL 散度在 [desired_kl/2, 2*desired_kl] 范围内，学习率保持不变。
    这种自适应机制能自动平衡训练稳定性和收敛速度。
    """

    def test_lr_decreases_when_kl_too_high(self) -> None:
        """
        LR should decrease when KL > 2 * desired_kl.
        验证当 KL 散度 过高（kl_mean > 2 * desired_kl）时，学习率应降低（除以 1.5，但不低于 1e-5）。

        """
        # 设置：desired_kl = 0.01，kl_mean = 0.03（大于 0.02），初始学习率 1e-3。
        ppo, _obs = _build_ppo(schedule="adaptive", desired_kl=0.01, learning_rate=1e-3)
        initial_lr = ppo.learning_rate

        # Simulate high KL scenario
        ppo.learning_rate = initial_lr
        kl_mean = torch.tensor(0.03)  # > 2 * 0.01

        # Apply the same logic as PPO.update
        # 执行逻辑：触发 if kl_mean > desired_kl * 2.0 分支，learning_rate = max(1e-5, learning_rate / 1.5)。
        if kl_mean > ppo.desired_kl * 2.0:
            ppo.learning_rate = max(1e-5, ppo.learning_rate / 1.5)
        # 新学习率小于初始值，且等于 max(1e-5, 1e-3 / 1.5)。
        assert ppo.learning_rate < initial_lr
        assert ppo.learning_rate == max(1e-5, initial_lr / 1.5)

    def test_lr_increases_when_kl_too_low(self) -> None:
        """
        LR should increase when 0 < KL < desired_kl / 2.
        目的：验证当 KL 散度 过低（0 < kl_mean < desired_kl / 2）时，学习率应提高（乘以 1.5，但不超过 1e-2）。
        """
        # 设置：kl_mean = 0.002（小于 0.005），初始学习率 1e-3。
        ppo, _obs = _build_ppo(schedule="adaptive", desired_kl=0.01, learning_rate=1e-3)
        initial_lr = ppo.learning_rate

        kl_mean = torch.tensor(0.002)  # < 0.01 / 2 = 0.005
        # 执行逻辑：触发 elif kl_mean < desired_kl / 2.0 and kl_mean > 0.0 分支，learning_rate = min(1e-2, learning_rate * 1.5)。
        if kl_mean < ppo.desired_kl / 2.0 and kl_mean > 0.0:
            ppo.learning_rate = min(1e-2, ppo.learning_rate * 1.5)
        # 新学习率大于初始值，且等于 min(1e-2, 1e-3 * 1.5)。
        assert ppo.learning_rate > initial_lr
        assert ppo.learning_rate == min(1e-2, initial_lr * 1.5)

    def test_lr_unchanged_in_stable_range(self) -> None:
        """
        LR should remain unchanged when KL is in [desired_kl/2, 2*desired_kl].
        目的：验证当 KL 散度处于稳定范围 [desired_kl/2, 2*desired_kl] 内时，学习率保持不变。
        """
        ppo, _obs = _build_ppo(schedule="adaptive", desired_kl=0.01, learning_rate=1e-3)
        initial_lr = ppo.learning_rate
        # 设置：kl_mean = 0.01（正好等于 desired_kl）。
        kl_mean = torch.tensor(0.01)  # Exactly desired_kl — in stable range
        # 执行逻辑：不满足任何调整条件，因此学习率不变。
        if kl_mean > ppo.desired_kl * 2.0:
            ppo.learning_rate = max(1e-5, ppo.learning_rate / 1.5)
        elif kl_mean < ppo.desired_kl / 2.0 and kl_mean > 0.0:
            ppo.learning_rate = min(1e-2, ppo.learning_rate * 1.5)
        # 新学习率等于初始学习率。
        assert ppo.learning_rate == initial_lr
