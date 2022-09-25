import copy
from calendar import c
from random import sample
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

import wandb
from agent.common.agent import MultiTaskAgent
from agent.common.policy import GaussianPolicy, GMMPolicy
from agent.common.teacher import AttitudePID, PositionPID
from agent.common.util import (
    assert_shape,
    check_dim,
    get_sa_pairs,
    grad_false,
    hard_update,
    linear_schedule,
    np2ts,
    soft_update,
    ts2np,
    update_learning_rate,
    update_params,
)
from agent.common.value_function import TwinnedSFNetwork, MultiheadSFNetwork

# disable api to speed up
api = False
torch.autograd.set_detect_anomaly(api)  # detect NaN
torch.autograd.profiler.profile(api)
torch.autograd.profiler.emit_nvtx(api)

Observation = Union[np.ndarray, float]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPS = 1e-6


class SFGPIAgent(MultiTaskAgent):
    """GPI
    Andre Barreto, Transfer in Deep Reinforcement Learning Using
        Successor Features and Generalised Policy Improvement
    """

    @classmethod
    def default_config(cls):
        return dict(
            env="InvertedDoublePendulum-v4",
            env_kwargs={},
            total_timesteps=int(1e6),
            reward_scale=1,
            alpha_lr=3e-4,
            gamma=0.99,
            n_particles=128,
            multi_step=1,
            updates_per_step=1,
            replay_buffer_size=1e6,
            prioritized_memory=True,
            mini_batch_size=256,
            min_n_experience=int(2048),
            grad_clip=None,
            tau=5e-3,
            alpha=0.2,
            lr=1e-4,
            lr_schedule=np.array([1e-4, 1e-5]),
            td_target_update_interval=1,
            compute_target_by_sfv=True,
            weight_decay=1e-5,
            policy_class="GaussianMixture",
            n_gauss=4,
            policy_regularization=1e-3,
            policy_lr=5e-4,
            policy_lr_schedule=np.array([2e-4, 2e-5]),
            delayed_policy_update=1,
            calc_pol_loss_by_advantage=True,
            entropy_tuning=True,
            alpha_bnd=np.array([0.5, 10]),
            net_kwargs={"value_sizes": [64, 64], "policy_sizes": [32, 32]},
            explore_with_gpe=True,
            explore_with_gpe_after_n_round=0,
            task_schedule=None,
            train_nround=10,
            enable_curriculum_learning=False,
            eval=True,
            eval_interval=20,  # episodes
            evaluate_episodes=3,  # 10
            log_interval=50,  # learning_steps
            seed=0,
            render=False,
        )

    def __init__(
        self,
        config: dict = {},
    ) -> None:
        self.config = config
        super().__init__(**config)

        self.explore_with_gpe_after_n_round = config.get(
            "explore_with_gpe_after_n_round", 3
        )
        if self.explore_with_gpe_after_n_round > self.n_round:
            self.explore_with_gpe = False
        else:
            self.explore_with_gpe = config.get("explore_with_gpe", True)

        self.lr = config.get("lr", 1e-3)
        self.lr_schedule = np.array(config.get("lr_schedule", None))
        self.gamma = config.get("gamma", 0.99)
        self.tau = config.get("tau", 5e-3)
        self.td_target_update_interval = int(config.get("td_target_update_interval", 1))
        self.n_particles = config.get("n_particles", 20)
        self.calc_target_by_sfv = config.get("compute_target_by_sfv", True)
        self.weight_decay = config.get("weight_decay", 1e-5)

        self.policy_class = config.get("policy_class", "GaussianMixture")
        self.pol_lr = config.get("policy_lr", 1e-3)
        self.pol_lr_schedule = np.array(config.get("policy_lr_schedule", None))
        self.pol_reg = config.get("policy_regularization", 1e-3)
        self.n_gauss = config.get("n_gauss", 5)
        self.delayed_policy_update = config.get("delayed_policy_update", 1)
        self.calc_pol_loss_by_advantage = config.get("calc_pol_loss_by_advantage", True)

        self.net_kwargs = config["net_kwargs"]
        self.grad_clip = config.get("grad_clip", None)
        self.entropy_tuning = config.get("entropy_tuning", True)

        self.create_policy = False
        self.sfs, self.sf_tars, self.sf_optims = [], [], []
        self.pols, self.pol_optims = [], []
        if self.task_schedule is not None:
            for _ in range(len(self.task_schedule)):
                self.create_sf_policy()
        else:
            self.create_sf_policy()
        print(f"{len(self.pols)} policies are created")

        if self.entropy_tuning:
            self.alpha_lr = config.get("alpha_lr", 3e-4)
            self.target_entropy = -torch.prod(
                torch.Tensor(self.env.action_space.shape).to(device)
            ).item()  # target entropy = -|A|
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_bnd = np.array(config.get("alpha_bnd", np.array([1e-1, 10])))
            self.alpha = self.log_alpha.exp()
            self.alpha = torch.clip(self.alpha, *self.alpha_bnd)
            self.alpha_optimizer = Adam([self.log_alpha], lr=self.alpha_lr)
        else:
            self.alpha = torch.tensor(config.get("alpha", 0.2)).to(device)

        wandb.watch(self.sfs[0])
        wandb.watch(self.pols[0])

    def create_sf_policy(self):
        """create new set of policies and successor features"""
        sf, sf_target, sf_optimizer = self.create_sf()

        policy, policy_optimizer = self.create_pol()
        self.create_policy = True

        self.sfs.append(sf)
        self.sf_tars.append(sf_target)
        self.sf_optims.append(sf_optimizer)
        self.pols.append(policy)
        self.pol_optims.append(policy_optimizer)

    def create_sf(self):
        sf = TwinnedSFNetwork(
            observation_dim=self.observation_dim,
            feature_dim=self.feature_dim,
            action_dim=self.action_dim,
            sizes=self.net_kwargs.get("value_sizes", [64, 64]),
        ).to(device)
        sf_target = (
            TwinnedSFNetwork(
                observation_dim=self.observation_dim,
                feature_dim=self.feature_dim,
                action_dim=self.action_dim,
                sizes=self.net_kwargs.get("value_sizes", [64, 64]),
            )
            .to(device)
            .eval()
        )
        hard_update(sf_target, sf)
        grad_false(sf_target)
        sf_optimizer = (
            Adam(sf.SF0.parameters(), lr=self.lr, weight_decay=self.weight_decay),
            Adam(sf.SF1.parameters(), lr=self.lr, weight_decay=self.weight_decay),
        )
        return sf, sf_target, sf_optimizer

    def create_pol(self):
        if self.policy_class == "GaussianMixture":
            policy = GMMPolicy(
                observation_dim=self.observation_dim,
                action_dim=self.action_dim,
                sizes=self.net_kwargs.get("policy_sizes", [64, 64]),
                n_gauss=self.n_gauss,
                reg=self.pol_reg,
            ).to(device)
        elif self.policy_class == "Gaussian":
            policy = GaussianPolicy(
                observation_dim=self.observation_dim,
                action_dim=self.action_dim,
                sizes=self.net_kwargs.get("policy_sizes", [64, 64]),
            ).to(device)
        policy_optimizer = Adam(policy.parameters(), lr=self.pol_lr)
        return policy, policy_optimizer

    def run(self):
        while True:
            self.train_episode()
            if self.learn_all_tasks:
                break
            if self.steps > self.total_timesteps:
                break

    def post_episode_process(self):
        self.update_lr()
        return super().post_episode_process()

    def post_all_tasks_process(self):
        if self.n_round > self.explore_with_gpe_after_n_round:
            self.explore_with_gpe = True

        return super().post_all_tasks_process()

    def curriculum(self):
        self.reward_scale *= 1.1

        if self.entropy_tuning:
            self.alpha_bnd *= 0.9

        if self.lr_schedule is not None:
            self.lr_schedule *= 0.95

        if self.pol_lr_schedule is not None:
            self.pol_lr_schedule *= 0.95

        return super().curriculum()

    def update_lr(self):
        if self.task_schedule is not None:
            budget_per_task = self.budget_per_task
        else:
            budget_per_task = self.total_timesteps

        steps = self.steps % budget_per_task
        if self.lr_schedule is not None:
            self.lr = linear_schedule(steps, budget_per_task, self.lr_schedule)
            for optim in self.sf_optims:
                update_learning_rate(optim[0], self.lr)
                update_learning_rate(optim[1], self.lr)

        if self.pol_lr_schedule is not None:
            self.pol_lr = linear_schedule(steps, budget_per_task, self.pol_lr_schedule)
            for optim in self.pol_optims:
                update_learning_rate(optim, self.pol_lr)

    def explore(self, obs, w):
        if self.explore_with_gpe:
            acts, qs = self.gpe(obs, w, "explore")
            act = self.gpi(acts, qs)
        else:
            act, _, _ = self.pols[self.task_idx](obs)
        return act

    def exploit(self, obs, w):
        acts, qs = self.gpe(obs, w, "exploit")
        act = self.gpi(acts, qs)
        return act

    def gpe(self, obs, w, mode):
        acts, qs = [], []
        for i in range(len(self.sfs)):
            if mode == "explore":
                act, _, _ = self.pols[i](obs)
            if mode == "exploit":
                _, _, act = self.pols[i](obs)

            q = self.calc_q_from_sf(obs, act, self.sfs[i], w)

            acts.append(act)
            qs.append(q)
        return acts, qs

    def gpi(self, acts, qs):
        qs = torch.tensor(qs)
        pol_idx = torch.argmax(qs)
        act = acts[pol_idx].squeeze()
        return act

    def learn(self):
        self.learn_steps += 1

        i = self.task_idx
        sf_optim, sf, sf_tar = self.sf_optims[i], self.sfs[i], self.sf_tars[i]
        pol_optim, pol = self.pol_optims[i], self.pols[i]

        if self.learn_steps % self.td_target_update_interval == 0:
            soft_update(sf_tar, sf, self.tau)

        if self.prioritized_memory:
            batch, indices, weights = self.replay_buffer.sample(self.mini_batch_size)
        else:
            batch = self.replay_buffer.sample(self.mini_batch_size)
            weights = 1

        sf_loss, errors, mean_sf, target_sf = self.calc_sf_loss(
            batch, sf, sf_tar, pol, weights
        )
        update_params(sf_optim[0], sf.SF0, sf_loss[0], self.grad_clip)
        update_params(sf_optim[1], sf.SF1, sf_loss[1], self.grad_clip)

        if self.learn_steps % self.delayed_policy_update == 0:
            pol_loss, entropy = self.calc_pol_loss(batch, sf_tar, pol, weights)
            update_params(pol_optim, pol, pol_loss, self.grad_clip)

            if self.entropy_tuning:
                entropy_loss = self.update_alpha(weights, entropy)

        if self.prioritized_memory:
            self.replay_buffer.update_priority(indices, errors.detach().cpu().numpy())

        if (
            self.learn_steps % self.log_interval == 0
            and self.learn_steps % self.delayed_policy_update == 0
        ):
            metrics = {
                "loss/SF0": sf_loss[0].detach().item(),
                "loss/SF1": sf_loss[1].detach().item(),
                "loss/policy": pol_loss.detach().item(),
                "state/mean_SF0": mean_sf[0],
                "state/mean_SF1": mean_sf[1],
                "state/target_sf": target_sf.detach().mean().item(),
                "state/lr": self.lr,
                "state/entropy": entropy.detach().mean().item(),
                "task/task_idx": self.task_idx,
                "task/step": self.steps,
            }
            if self.entropy_tuning:
                metrics.update(
                    {
                        "loss/alpha": entropy_loss.detach().item(),
                        "state/alpha": self.alpha.detach().item(),
                    }
                )
            wandb.log(metrics)

    def update_alpha(self, weights, entropy):
        entropy_loss = self.calc_entropy_loss(entropy, weights)
        update_params(self.alpha_optimizer, None, entropy_loss)
        self.alpha = torch.clip(self.log_alpha.exp(), *self.alpha_bnd)
        return entropy_loss

    def calc_sf_loss(self, batch, sf_net, sf_target_net, pol_net, weights):
        (obs, features, actions, _, next_obs, dones) = batch

        cur_sf0, cur_sf1 = self.get_cur_sf(sf_net, obs, actions)
        target_sf, _ = self.get_target_sf(
            sf_target_net, pol_net, features, next_obs, dones
        )

        sf0_loss = torch.mean((cur_sf0 - target_sf).pow(2) * weights)
        sf1_loss = torch.mean((cur_sf1 - target_sf).pow(2) * weights)

        errors = torch.mean((cur_sf0 - target_sf).pow(2), 1)

        mean_sf0 = cur_sf0.detach().mean().item()
        mean_sf1 = cur_sf1.detach().mean().item()

        return (sf0_loss, sf1_loss), errors, (mean_sf0, mean_sf1), target_sf

    def calc_pol_loss(self, batch, sf_tar_net, pol_net, weights, task=None):
        if task is None:
            task = self.w

        (obs, _, _, _, _, _) = batch

        act, entropy, _ = pol_net(obs)
        q = self.calc_q_from_sf(obs, act, sf_tar_net, task)

        if self.calc_pol_loss_by_advantage:
            v = self.calc_v_from_sf(obs, sf_tar_net, task)
            q = q - v

        loss = torch.mean((-q - self.alpha * entropy) * weights) + self.reg_loss(
            pol_net
        )
        return loss, entropy

    def calc_entropy_loss(self, entropy, weights):
        entropy_loss = -torch.mean(
            self.log_alpha * (self.target_entropy - entropy).detach() * weights
        )
        return entropy_loss

    def get_cur_sf(self, sf, obs, actions):
        cur_sf0, cur_sf1 = sf(obs, actions)
        assert_shape(cur_sf0, [None, self.feature_dim])
        return cur_sf0, cur_sf1

    def get_target_sf(self, sf_target, pol, features, next_observations, dones):
        features = check_dim(features, self.feature_dim)

        if self.calc_target_by_sfv:
            target_sf, next_sf = self.calc_target_sfv(
                next_observations, features, dones, sf_target
            )
        else:
            target_sf, next_sf = self.calc_target_sf(
                next_observations, features, dones, sf_target, pol
            )
        assert_shape(target_sf, [None, self.feature_dim])
        return target_sf, next_sf

    def calc_target_sf(self, next_obs, features, dones, sf_target, pol):
        with torch.no_grad():
            _, _, next_act = pol(next_obs)
            next_sf0, next_sf1 = sf_target(next_obs, next_act)
            next_sf = torch.min(next_sf0, next_sf1)
            assert_shape(next_sf, [None, self.feature_dim])

            target_sf = features + (1 - dones) * self.gamma * next_sf
        return target_sf, next_sf

    def calc_target_sfv(self, next_obs, features, dones, sf_target):
        with torch.no_grad():
            next_sf0, next_sf1, _, _ = self.calc_sf_value_by_random_actions(
                next_obs, sf_target
            )
            next_sf = torch.min(next_sf0, next_sf1)
            next_sf = next_sf.view(-1, self.n_particles, self.feature_dim)
            next_sfv = self.log_mean_exp(next_sf)
            assert_shape(next_sfv, [None, self.feature_dim])

            target_sf = (features + (1 - dones) * self.gamma * next_sfv).squeeze(1)
        return target_sf, next_sfv

    def calc_q_from_sf(self, obs, act, sf_net, w):
        sf0, sf1 = sf_net(obs, act)
        assert_shape(sf0, [None, self.feature_dim])

        q0 = self.get_q_from_sf_w(sf0, w)
        q1 = self.get_q_from_sf_w(sf1, w)
        q = torch.min(q0, q1)
        q = q[:, None]
        assert_shape(q, [None, 1])
        return q

    def calc_v_from_sf(self, obs, sf_tar_net, w):
        with torch.no_grad():
            sf0, sf1, _, _ = self.calc_sf_value_by_random_actions(obs, sf_tar_net)
            q0 = self.get_q_from_sf_w(sf0, w)
            q1 = self.get_q_from_sf_w(sf1, w)
            q = torch.min(q0, q1)
            q = q.view(-1, self.n_particles)
            assert_shape(q, [None, self.n_particles])

            v = self.log_mean_exp(q)
            v = v[:, None]
            assert_shape(v, [None, 1])
        return v

    def get_q_from_sf_w(self, sf, w):
        q = torch.matmul(sf, w)
        q = self.reward_scale * q
        return q

    def log_mean_exp(self, val):
        v = torch.logsumexp(val / self.alpha, 1)
        v -= np.log(self.n_particles)
        v += self.action_dim * np.log(2)
        v = v * self.alpha
        return v

    def calc_sf_value_by_random_actions(self, obs, sf_tar_net):
        sample_act = (
            torch.distributions.uniform.Uniform(-1, 1)
            .sample((self.n_particles, self.action_dim))
            .to(device)
        )
        sample_obs, sample_act = get_sa_pairs(obs, sample_act)
        sf0, sf1 = sf_tar_net(sample_obs, sample_act)
        assert_shape(sf0, [None, self.feature_dim])
        return sf0, sf1, sample_obs, sample_act

    def reg_loss(self, pol):
        reg = getattr(pol, "reg_loss", None)
        loss = pol.reg_loss() if callable(reg) else 0
        return loss

    def calc_priority_error(self, batch):
        (obs, features, act, _, next_obs, dones) = batch

        i = self.task_idx
        with torch.no_grad():
            cur_sf0, cur_sf1 = self.get_cur_sf(self.sfs[i], obs, act)
            target_sf, _ = self.get_target_sf(
                self.sf_tars[i], self.pols[i], features, next_obs, dones
            )
            error = torch.sum(torch.abs(cur_sf0 - target_sf), 1).item()
        return error


class TEACHER_SFGPI(SFGPIAgent):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "enable_tsf": True,
                "enable_reset_controller": True,
                "expert_supervise_loss": True,
                "normalize_critic": True,
                "n_teacher_steps": int(5e4),
                "pretrain_sf_epochs": 1e4,
            }
        )
        return config

    def __init__(self, config: Optional[Dict[Any, Any]] = None) -> None:
        super().__init__(config)

        self.n_teacher_steps = config.get("n_teacher_steps", 10000)
        self.pretrain_sf_epochs = config.get("pretrain_sf_epochs", 1e4)

        self.enable_tsf = config.get("enable_tsf", True)
        self.teachers = [AttitudePID(), PositionPID()]

        self.enable_reset_ctrl = config.get("enable_reset_controller", True)
        self.reset_ctrl = PositionPID()

        self.train_tsf = self.enable_tsf
        self.expert_supervise_loss = (
            config.get("expert_supervise_loss", True) and self.enable_tsf
        )
        self.normalize_critic = config.get("normalize_critic", True) and self.enable_tsf

        self.tsfs, self.tsf_tars, self.tsf_optims = [], [], []
        for _ in range(len(self.teachers)):
            self.create_teacher_sfs()

        self.reset_teachers()
        self.stu_act_cnt = 0
        self.tea_act_cnt = 0
        self.sq, self.tq0, self.tq1 = [], [], []
        self.tidx = 0

        self.origin_goal = []
        self.safe_goal_isset = False

        self.beta = 1 / self.mini_batch_size

    def create_teacher_sfs(self):
        self.tsf, self.tsf_tar, self.tsf_optim = self.create_sf()

        self.tsfs.append(self.tsf)
        self.tsf_tars.append(self.tsf_tar)
        self.tsf_optims.append(self.tsf_optim)

    def explore(self, obs, w):
        if self.explore_with_gpe:
            acts, qs = self.gpe(obs, w, "explore")

            if self.enable_tsf:
                tacts, tqs = self.tgpe(obs, w)
                acts.extend(tacts)
                qs.extend(tqs)

            act = self.gpi(acts, qs)

        else:
            act, _, _ = self.pols[self.task_idx](obs)
        return act

    def exploit(self, obs, w):
        acts, qs = self.gpe(obs, w, "exploit")

        if self.enable_tsf:
            tacts, tqs = self.tgpe(obs, w)

            # log qs in wandb
            self.sq.append(qs)
            self.tq0.append(tqs[0])
            self.tq1.append(tqs[1])

            acts.extend(tacts)
            qs.extend(tqs)

        act = self.gpi(acts, qs)

        if self.enable_tsf:
            if act in torch.stack(tacts):
                self.tea_act_cnt += 1
            else:
                self.stu_act_cnt += 1

        return act

    def tgpe(self, obs, w):
        tacts, tqs = [], []
        tacts = [tea.act(obs) for tea in self.teachers]
        tqs = [
            self.calc_q_from_sf(obs, tacts[i], self.tsf_tars[i], w)
            for i in range(len(self.tsf_tars))
        ]
        return tacts, tqs

    def act(self, obs, mode="explore"):
        if self.trigger_reset(obs):  # overwrite action by reset controller
            if not self.safe_goal_isset:
                self.set_safe_goal()
            action = ts2np(self.reset_ctrl.act(np2ts(obs)))
            return action
        else:
            self.set_origin_goal()

        if self.train_tsf:
            action = ts2np(self.teachers[self.tidx].act(np2ts(obs)))
        else:
            action = super().act(obs, mode)
        return action

    def run(self):
        if self.train_tsf:
            for i in range(len(self.teachers)):
                print(f"========= collect data with teacher {i} =========")
                self.tidx = i
                step = 0
                while True:
                    self.train_episode()
                    step += 1

                    if step > self.n_teacher_steps:
                        break

            for i in range(len(self.pols)):
                print(f"========= pretrain sf and student {i}  =========")
                self.pretrain_student(i)

        self.train_tsf = False
        return super().run()

    def pretrain_student(self, idx=0):
        self.expert_supervise_loss = True
        self.normalize_critic = True
        self.update_task_by_task_index(idx)

        for epoch in range(int(self.pretrain_sf_epochs)):
            self.learn()

            if epoch % self.log_interval == 0:
                if self.prioritized_memory:
                    batch, _, _ = self.replay_buffer.sample(self.mini_batch_size)
                else:
                    batch = self.replay_buffer.sample(self.mini_batch_size)

                with torch.no_grad():
                    (obs, _, _, _, _, _) = batch
                    _, _, act = self.pols[idx](obs)
                    tact0 = self.teachers[0].act(obs)
                    tact1 = self.teachers[1].act(obs)
                    q = self.calc_q_from_sf(obs, act, self.sf_tars[idx], self.w)

                metrics = {
                    f"pretrain/q_value{idx}": q.mean().item(),
                    f"pretrain/act{idx}": act,
                    f"pretrain/tact0": tact0,
                    f"pretrain/tact1": tact1,
                }
                wandb.log(metrics)

        self.expert_supervise_loss = False
        self.normalize_critic = False
        self.update_task_by_task_index(0)  # reset task

    def calc_sf_loss(self, batch, sf_net, sf_target_net, pol_net, weights):
        tsf_loss, errors, mean_tsf, target_tsf = super().calc_sf_loss(
            batch, sf_net, sf_target_net, pol_net, weights
        )

        (obs, features, act, _, _, dones) = batch
        if self.expert_supervise_loss:
            tsf_loss = self.calc_expert_supervise_loss(
                sf_net, sf_target_net, tsf_loss, obs, act
            )

        if self.normalize_critic:
            tsf_loss = self.calc_normalization_loss(
                sf_net,
                sf_target_net,
                target_tsf,
                obs,
                features,
                act,
                dones,
                tsf_loss,
            )

        return tsf_loss, errors, mean_tsf, target_tsf

    def calc_expert_supervise_loss(self, sf_net, sf_target_net, tsf_loss, obs, act):
        tiled_obs = torch.tile(obs, (1, self.n_particles)).reshape(
            -1, self.observation_dim
        )
        tiled_act = torch.tile(act, (1, self.n_particles)).reshape(-1, self.action_dim)

        sf0, sf1 = self.get_cur_sf(sf_target_net, tiled_obs, tiled_act)
        expert_sf = torch.min(sf0, sf1)

        (
            sampled_sf0,
            sampled_sf1,
            _,
            sampled_act,
        ) = self.calc_sf_value_by_random_actions(obs, sf_net)
        margin = self.calc_margin(tiled_act, sampled_act)

        sup_loss0 = sampled_sf0 + margin - expert_sf
        sup_loss1 = sampled_sf1 + margin - expert_sf

        sup_loss0 = torch.masked_select(sup_loss0, sup_loss0 > 0).mean()
        sup_loss1 = torch.masked_select(sup_loss1, sup_loss1 > 0).mean()

        tsf_loss0 = tsf_loss[0]
        tsf_loss1 = tsf_loss[1]

        tsf_loss0 = tsf_loss0 + sup_loss0
        tsf_loss1 = tsf_loss1 + sup_loss1

        tsf_loss = (tsf_loss0, tsf_loss1)

        if self.learn_steps % self.log_interval == 0:
            supervise_loss = sup_loss0.detach().mean().item()
            wandb.log({"pretrain/supervise_loss": supervise_loss})
        return tsf_loss

    def calc_normalization_loss(
        self, sf_net, sf_target_net, tar_sf, obs, features, act, dones, tsf_loss
    ):
        cur_sf0, cur_sf1 = self.get_cur_sf(sf_net, obs, act)
        _, cur_sfv = self.calc_target_sfv(obs, features, dones, sf_target_net)
        mean_sfv = cur_sfv.detach().mean().item()

        norm_loss0 = self.calc_sf_normalize_loss(sf_net.SF0, cur_sf0, tar_sf, cur_sfv)
        norm_loss1 = self.calc_sf_normalize_loss(sf_net.SF1, cur_sf1, tar_sf, cur_sfv)

        tsf_loss0 = tsf_loss[0]
        tsf_loss1 = tsf_loss[1]

        tsf_loss0 = tsf_loss0 + norm_loss0
        tsf_loss1 = tsf_loss1 + norm_loss1

        tsf_loss = (tsf_loss0, tsf_loss1)

        if self.learn_steps % self.log_interval == 0:
            normalization_loss = norm_loss0.detach().mean().item()
            wandb.log({"pretrain/normalization_loss": normalization_loss})
            wandb.log({"pretrain/mean_sfv": mean_sfv})
        return tsf_loss

    def calc_sf_normalize_loss(self, net, curr_q, target_q, curr_v):
        gradient = self.calc_normalized_grad(net, curr_q, target_q, curr_v)
        pg_loss = self.calc_surrogate_loss(net, gradient)
        pg_loss = self.beta * pg_loss
        return pg_loss

    def calc_normalized_grad(self, net, q, target_q, v):
        # apply critic normalization trick,
        # ref: Yang-Gao, Reinforcement Learning from Imperfect Demonstrations
        gradient = torch.autograd.grad(
            (q - v),
            net.parameters(),
            grad_outputs=(q - target_q),
        )
        return gradient

    def calc_surrogate_loss(self, net, gradient):
        # multiply weights and differentiate later to apply gradient
        sur_loss = torch.sum(
            torch.stack(
                [torch.sum(w * g.detach()) for w, g in zip(net.parameters(), gradient)],
                dim=0,
            )
        )
        assert sur_loss.requires_grad == True
        return sur_loss

    def calc_margin(self, expert_action, sampled_action, scale=1000):
        norm = torch.norm(expert_action - sampled_action, dim=1)
        norm = norm.unsqueeze(1)
        return scale * norm

    def set_safe_goal(self):
        print("set safe goal")
        self.origin_goal = [
            self.env.target_type.pos_cmd_data,
            self.env.target_type.vel_cmd_data,
            self.env.target_type.ang_cmd_data,
            self.env.target_type.ori_cmd_data,
            self.env.target_type.angvel_cmd_data,
        ]
        self.env.target_type.pos_cmd_data = np.array([0, 0, -25])
        self.reset_teachers()
        self.reset_ctrl.reset()
        self.safe_goal_isset = True

    def set_origin_goal(self):
        self.safe_goal_isset = False
        try:
            self.env.target_type.pos_cmd_data = self.origin_goal[0]
            self.env.target_type.vel_cmd_data = self.origin_goal[1]
            self.env.target_type.ang_cmd_data = self.origin_goal[2]
            self.env.target_type.ori_cmd_data = self.origin_goal[3]
            self.env.target_type.angvel_cmd_data = self.origin_goal[4]
        except:
            pass

    def trigger_reset(self, obs):
        if self.enable_reset_ctrl:
            if self.hard_reset_condition(obs):
                return True
        return False

    def hard_reset_condition(self, obs):
        obs_info = self.env.obs_info
        # TODO: get rid of these magic numbers
        pos_lbnd = self.env.pos_lbnd + 5
        pos_ubnd = self.env.pos_ubnd - 5
        vel_ubnd = np.array([2.5, 2.5, 2.5])
        ang_bnd = 0.015

        if (obs_info["obs_dict"]["position"] <= pos_lbnd).any():
            return True
        if (obs_info["obs_dict"]["position"] >= pos_ubnd).any():
            return True

        if (np.abs(obs_info["obs_dict"]["velocity"]) >= vel_ubnd).any():
            return True

        roll = obs_info["obs_dict"]["angle"][0]
        if (roll > ang_bnd) or (roll < -ang_bnd):
            return True
        pitch = obs_info["obs_dict"]["angle"][1]
        if (pitch > ang_bnd) or (pitch < -ang_bnd):
            return True

        return False

    def reset_env(self):
        self.reset_teachers()
        self.origin_goal = []
        self.safe_goal_set = False
        self.stu_act_cnt = 0
        self.tea_act_cnt = 0
        return super().reset_env()

    def reset_teachers(self):
        if self.enable_tsf:
            for pol in self.teachers:
                pol.reset()

    def log_evaluation(self, episodes, returns, success):
        if self.enable_tsf:
            st_act_ratio = self.stu_act_cnt / (self.stu_act_cnt + self.tea_act_cnt + 1)
            metrics = {
                "evaluate/teacher_q0": torch.mean(torch.tensor(self.tq0)),
                "evaluate/teacher_q1": torch.mean(torch.tensor(self.tq1)),
                "evaluate/student_q0": torch.mean(torch.tensor(self.sq)),
                "evaluate/student_teacher_act_ratio": st_act_ratio,
            }
            self.sq, self.tq0, self.tq1 = [], [], []
            wandb.log(metrics)
        return super().log_evaluation(episodes, returns, success)


if __name__ == "__main__":
    # ============== profile ==============#
    # pip install snakeviz
    # python -m cProfile -o sfgpi.profile rl/rl/agent/sfgpi.py -s time
    # snakeviz sfgpi.profile

    # ============== sweep ==============#
    # wandb sweep rl/rl/sweep_sfgpi.yaml
    import benchmark_env
    import drone_env
    import gym
    from drone_env.envs.script import close_simulation

    Agent = TEACHER_SFGPI  # SFGPIAgent, TEACHER_SFGPI

    env = "multitask-v0"  # "multitask-v0", "myInvertedDoublePendulum-v4"
    render = True
    auto_start_simulation = False
    if auto_start_simulation:
        close_simulation()

    if env == "multitask-v0":
        from agent.task_config import get_task_schedule

        env_config = {
            "simulation": {
                "gui": render,
                "enable_meshes": True,
                "auto_start_simulation": auto_start_simulation,
                "position": (0, 0, 25),  # initial spawned position
            },
            "observation": {
                "noise_stdv": 0.0,
            },
            "action": {
                "act_noise_stdv": 0.0,
                "thrust_range": [-0.2, 0.2],
            },
            "target": {"type": "FixedGoal"},
        }
        env_kwargs = {"config": env_config}
        task_schedule = get_task_schedule(
            # ["roll", "pitch", "yaw", "att", "z", "xz", "yz", "xyz"]
            ["att", "xyz"]
        )
    else:
        env_kwargs = {}
        task_schedule = None

    default_dict = Agent.default_config()
    enable_tsf = False
    min_n_experience = 0 if enable_tsf else 2048
    default_dict.update(
        dict(
            env=env,
            env_kwargs=env_kwargs,
            render=render,
            task_schedule=task_schedule,
            evaluate_episodes=3,
            updates_per_step=2,
            multi_step=2,
            min_batch_size=int(256),  # 256
            min_n_experience=int(min_n_experience),  # 0 if enable_tsf else 2048
            total_timesteps=1e6,  # 1e6
            train_nround=5,  # 5
            pretrain_sf_epochs=1e4,  # 1e4
            n_teacher_steps=int(5e3),  # 5e3
            entropy_tuning=False,
            enable_reset_controller=False,
            policy_class="Gaussian",
            net_kwargs={"value_sizes": [256, 256], "policy_sizes": [32, 32]},
            enable_tsf=enable_tsf,
            normalize_critic=False,
            expert_supervise_loss=False,
            enable_curriculum_learning=False,
            explore_with_gpe_after_n_round=1,
        )
    )

    wandb.init(config=default_dict)
    print(wandb.config)

    agent = Agent(wandb.config)
    agent.run()