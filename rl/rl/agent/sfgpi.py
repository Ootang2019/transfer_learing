from calendar import c
from random import sample
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

import wandb
from agent.common.teacher import PositionPID, AttitudePID
from agent.common.agent import MultiTaskAgent
from agent.common.policy import GaussianPolicy, GMMPolicy
from agent.common.util import (
    linear_schedule,
    update_learning_rate,
    get_sa_pairs,
    grad_false,
    hard_update,
    soft_update,
    ts2np,
    np2ts,
    update_params,
    assert_shape,
)
from agent.common.value_function import TwinnedSFNetwork

# disable api to speed up
torch.autograd.set_detect_anomaly(False)  # detect NaN
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

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
            tau=5e-3,
            alpha=0.2,
            lr=1e-4,
            lr_schedule=[5e-4, 1e-5],
            policy_lr=5e-4,
            policy_lr_schedule=[5e-3, 1e-4],
            alpha_lr=3e-4,
            gamma=0.99,
            policy_regularization=1e-3,
            n_particles=32,
            n_gauss=4,
            multi_step=1,
            updates_per_step=1,
            replay_buffer_size=1e6,
            prioritized_memory=True,
            mini_batch_size=256,
            min_n_experience=int(2048),
            td_target_update_interval=1,
            grad_clip=None,
            entropy_tuning=True,
            alpha_bnd=10,
            compute_target_by_sfv=True,
            normalize_critic=True,
            share_successor_feature=True,
            policy_class="GaussianMixture",
            net_kwargs={"value_sizes": [64, 64], "policy_sizes": [32, 32]},
            task_schedule=None,
            eval=True,
            eval_interval=100,
            log_interval=5,
            seed=0,
            render=False,
        )

    def __init__(
        self,
        config: dict = {},
    ) -> None:
        self.config = config
        super().__init__(**config)

        self.lr = config.get("lr", 1e-3)
        self.lr_schedule = config.get("lr_schedule", None)
        self.gamma = config.get("gamma", 0.99)
        self.tau = config.get("tau", 5e-3)
        self.td_target_update_interval = int(config.get("td_target_update_interval", 1))
        self.n_particles = config.get("n_particles", 20)
        self.calc_target_by_sfv = config.get("compute_target_by_sfv", True)
        self.normalize_critic = config.get("normalize_critic", True)

        self.policy_class = config.get("policy_class", "GaussianMixture")
        self.pol_lr = config.get("policy_lr", 1e-3)
        self.pol_lr_schedule = config.get("policy_lr_schedule", None)
        self.pol_reg = config.get("policy_regularization", 1e-3)
        self.n_gauss = config.get("n_gauss", 5)

        self.net_kwargs = config["net_kwargs"]
        self.grad_clip = config.get("grad_clip", None)
        self.entropy_tuning = config.get("entropy_tuning", True)

        self.create_policy = False
        self.share_sf = config.get("share_successor_feature", False)
        self.sfs, self.sf_tars, self.sf_optims = [], [], []
        self.pols, self.pol_optims = [], []
        if self.task_schedule is not None:
            for _ in range(len(self.task_schedule)):
                self.create_sf_policy()
        else:
            self.create_sf_policy()

        if self.entropy_tuning:
            self.alpha_lr = config.get("alpha_lr", 3e-4)
            self.target_entropy = -torch.prod(
                torch.Tensor(self.env.action_space.shape).to(device)
            ).item()  # target entropy = -|A|
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_bnd = config.get("alpha_bnd", True)
            self.alpha = self.log_alpha.exp()
            self.alpha = torch.clip(self.alpha, -self.alpha_bnd, self.alpha_bnd)
            self.alpha_optimizer = Adam([self.log_alpha], lr=self.alpha_lr)
        else:
            self.alpha = torch.tensor(config.get("alpha", 0.2)).to(device)

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

    def update_lr(self):
        if self.task_schedule is not None:
            budget_per_task = self.budget_per_task
        else:
            budget_per_task = self.total_timesteps

        if self.lr_schedule is not None:
            self.lr = linear_schedule(self.steps, budget_per_task, self.lr_schedule)
            for optim in self.sf_optims:
                update_learning_rate(optim[0], self.lr)
                update_learning_rate(optim[1], self.lr)

        if self.pol_lr_schedule is not None:
            self.pol_lr = linear_schedule(
                self.steps, budget_per_task, self.pol_lr_schedule
            )
            for optim in self.pol_optims:
                update_learning_rate(optim, self.pol_lr)

    def explore(self, obs, w):
        acts, qs = self.gpe(obs, w, "explore")
        act = self.gpi(acts, qs)
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
        pol_loss, entropy = self.calc_pol_loss(batch, sf_tar, pol, weights)

        update_params(sf_optim[0], sf.SF0, sf_loss[0], self.grad_clip)
        update_params(sf_optim[1], sf.SF1, sf_loss[1], self.grad_clip)
        update_params(pol_optim, pol, pol_loss, self.grad_clip)

        if self.entropy_tuning:
            entropy_loss = self.calc_entropy_loss(entropy, weights)
            update_params(self.alpha_optimizer, None, entropy_loss)
            self.alpha = torch.clip(
                self.log_alpha.exp(), -self.alpha_bnd, self.alpha_bnd
            )
            wandb.log(
                {
                    "loss/alpha": entropy_loss.detach().item(),
                    "state/alpha": self.alpha.detach().item(),
                }
            )

        if self.prioritized_memory:
            self.replay_buffer.update_priority(indices, errors.detach().cpu().numpy())

        if self.learn_steps % self.log_interval * 100 == 0:
            metrics = {
                "loss/SF0": sf_loss[0].detach().item(),
                "loss/SF1": sf_loss[1].detach().item(),
                "loss/policy": pol_loss.detach().item(),
                "state/mean_SF0": mean_sf[0],
                "state/mean_SF1": mean_sf[1],
                "state/target_sf": target_sf.detach().mean().item(),
                "state/entropy": entropy.detach().mean().item(),
                "state/lr": self.lr,
                "task/task_idx": self.task_idx,
            }
            w = ts2np(self.w)
            for i in range(w.shape[0]):
                metrics[f"task/{i}"] = w[i]
            wandb.log(metrics)

    def calc_sf_loss(self, batch, sf, sf_target, pol, weights):
        (obs, features, actions, rewards, next_obs, dones) = batch

        cur_sf0, cur_sf1 = self.get_cur_sf(sf, obs, actions)
        target_sf, _ = self.get_target_sf(sf_target, pol, features, next_obs, dones)

        errors = torch.sum(torch.abs(cur_sf0 - target_sf), 1)

        mean_sf0 = cur_sf0.detach().mean().item()
        mean_sf1 = cur_sf1.detach().mean().item()

        if self.normalize_critic:
            _, sfv = self.calc_target_sfv(obs, features, dones, sf_target, pol)
            grad_sf0 = self.calc_normalized_grad(
                sf.SF0, weights, cur_sf0, target_sf, sfv
            )
            grad_sf1 = self.calc_normalized_grad(
                sf.SF1, weights, cur_sf1, target_sf, sfv
            )
            sf0_loss = self.calc_surrogate_loss(sf.SF0, grad_sf0)
            sf1_loss = self.calc_surrogate_loss(sf.SF1, grad_sf1)
        else:
            sf0_loss = torch.mean((cur_sf0 - target_sf).pow(2) * weights)
            sf1_loss = torch.mean((cur_sf1 - target_sf).pow(2) * weights)

        return (sf0_loss, sf1_loss), errors, (mean_sf0, mean_sf1), target_sf

    def get_cur_sf(self, sf, obs, actions):
        cur_sf0, cur_sf1 = sf(obs, actions)
        assert_shape(cur_sf0, [None, self.feature_dim])
        return cur_sf0, cur_sf1

    def get_target_sf(self, sf_target, pol, features, next_observations, dones):
        if self.calc_target_by_sfv:
            target_sf, next_sf = self.calc_target_sfv(
                next_observations, features, dones, sf_target, pol
            )
        else:
            target_sf, next_sf = self.calc_target_sf(
                next_observations, features, dones, sf_target, pol
            )
        assert_shape(target_sf, [None, self.feature_dim])
        return target_sf, next_sf

    def calc_pol_loss(self, batch, sf_tar, pol, weights):
        (obs, _, _, _, _, _) = batch

        act, entropy, _ = pol(obs)
        q = self.calc_q_from_sf(obs, act, sf_tar, self.w)

        if self.normalize_critic:
            with torch.no_grad():
                v = self.calc_v_from_sf(obs, sf_tar, self.w)
            q -= v

        loss = torch.mean((-self.alpha * entropy - q) * weights) + self.reg_loss(pol)
        return loss, entropy

    def calc_entropy_loss(self, entropy, weights):
        entropy_loss = -torch.mean(
            self.log_alpha * (self.target_entropy - entropy).detach() * weights
        )
        return entropy_loss

    def calc_target_sf(self, next_obs, features, dones, sf_target, pol):
        with torch.no_grad():
            _, _, next_act = pol(next_obs)

            next_sf0, next_sf1 = sf_target(next_obs, next_act)
            next_sf = torch.min(next_sf0, next_sf1)
            assert_shape(next_sf, [None, self.feature_dim])
            assert_shape(features, [None, self.feature_dim])

            target_sf = features + (1 - dones) * self.gamma * next_sf
            assert_shape(target_sf, [None, self.feature_dim])

        return target_sf, next_sf

    def calc_target_sfv(self, next_obs, features, dones, sf_target, pol):
        with torch.no_grad():
            next_sf = self.calc_sf_value_by_random_actions(next_obs, sf_target)
            next_sf = next_sf.view(-1, self.n_particles, self.feature_dim)

            next_sfv = torch.logsumexp(next_sf, 1)
            next_sfv -= np.log(self.n_particles)
            next_sfv += self.action_dim * np.log(2)
            assert_shape(next_sfv, [None, self.feature_dim])

            target_sf = (features + (1 - dones) * self.gamma * next_sfv).squeeze(1)

        return target_sf, next_sfv

    def calc_normalized_grad(self, sf, weights, cur_sf, target_sf, sfv):
        # apply critic normalization trick,
        # ref: Yang-Gao, Reinforcement Learning from Imperfect Demonstrations
        grad_sf = torch.autograd.grad(
            (cur_sf - sfv),
            sf.parameters(),
            grad_outputs=(cur_sf - target_sf) * weights,
        )
        return grad_sf

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

    def calc_q_from_sf(self, obs, act, sf, w):
        sf0, sf1 = sf(obs, act)
        assert_shape(sf0, [None, self.feature_dim])

        sf = torch.min(sf0, sf1)
        q = torch.matmul(sf, w)
        q *= self.reward_scale
        q = q[:, None]
        assert_shape(q, [None, 1])
        return q

    def calc_v_from_sf(self, obs, sf_tar, w):
        sf = self.calc_sf_value_by_random_actions(obs, sf_tar)
        q = torch.matmul(sf, w)
        q *= self.reward_scale
        q = q.view(-1, self.n_particles)
        assert_shape(q, [None, self.n_particles])

        v = torch.logsumexp(q, 1)
        v = v[:, None]
        assert_shape(v, [None, 1])
        return v

    def calc_sf_value_by_random_actions(self, obs, sf_target):
        sample_act = (
            torch.distributions.uniform.Uniform(-1, 1)
            .sample((self.n_particles, self.action_dim))
            .to(device)
        )
        sample_obs, sample_act = get_sa_pairs(obs, sample_act)
        sf0, sf1 = sf_target(sample_obs, sample_act)
        sf = torch.min(sf0, sf1)
        assert_shape(sf, [None, self.feature_dim])
        return sf

    def reg_loss(self, pol):
        reg = getattr(pol, "reg_loss", None)
        loss = pol.reg_loss() if callable(reg) else 0
        return loss

    def calc_priority_error(self, batch):
        (obs, features, act, rewards, next_obs, dones) = batch
        i = self.task_idx
        with torch.no_grad():
            cur_sf0, cur_sf1 = self.get_cur_sf(self.sfs[i], obs, act)
            target_sf, _ = self.get_target_sf(
                self.sf_tars[i], self.pols[i], features, next_obs, dones
            )
            error = torch.sum(torch.abs(cur_sf0 - target_sf), 1).item()
        return error

    def create_sf_policy(self):
        """create new set of policies and successor features"""
        if self.share_sf and self.create_policy:  # reference to the same sf
            sf = self.sfs[0]
            sf_target = self.sf_tars[0]
            sf_optimizer = self.sf_optims[0]
        else:
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
            Adam(sf.SF0.parameters(), lr=self.lr),
            Adam(sf.SF1.parameters(), lr=self.lr),
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

    def update_task(self, w):
        super().update_task(w)
        if self.create_new_pol(w):
            self.create_sf_policy()

    def create_new_pol(self, w):
        """should we create new set of policies"""
        b = False

        if not self.create_policy:
            b = True

        if w not in self.prev_ws:
            b = True

        return b


class TEACHER_SFGPI(SFGPIAgent):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "train_tsf": True,
                "n_teacher_episodes": 60,
                "pretrain_sf_epochs": 1e4,
            }
        )
        return config

    def __init__(self, config: Optional[Dict[Any, Any]] = None) -> None:
        super().__init__(config)

        self.n_teacher_episodes = config.get("n_teacher_episodes", 60)
        self.pretrain_sf_epochs = config.get("pretrain_sf_epochs", 1e4)

        self.teachers = [AttitudePID(), PositionPID()]
        self.train_tsf = config.get("train_tsf", True)

        self.tsfs, self.tsf_tars, self.tsf_optims = [], [], []
        for _ in range(len(self.teachers)):
            self.create_teacher_sfs()

        self.reset_teachers()
        self.stu_act_cnt = 0
        self.tea_act_cnt = 0
        self.teacher_idx = 0

    def run(self):
        if self.train_tsf:
            for i in range(len(self.teachers)):
                self.teacher_idx = i
                for j in range(self.n_teacher_episodes):
                    print(f"========= teacher{i} sf episode {j} =========")
                    self.train_episode()
            self.pretrain_student()
        self.train_tsf = False
        return super().run()

    def pretrain_student(self):
        hard_update(self.sfs[0], self.tsfs[0])
        hard_update(self.sf_tars[0], self.tsf_tars[0])
        self.pretrain_pol(0, self.pretrain_sf_epochs)

    def pretrain_pol(self, pol_idx=0, pretrain_epochs=1e3):
        print(f"start pretraing policy{pol_idx}")
        i = pol_idx
        for epoch in range(int(pretrain_epochs)):
            if self.prioritized_memory:
                batch, indices, weights = self.replay_buffer.sample(
                    self.mini_batch_size
                )
            else:
                batch = self.replay_buffer.sample(self.mini_batch_size)
                weights = 1

            pol_loss, entropy = self.calc_pol_loss(
                batch, self.sf_tars[i], self.pols[i], weights
            )
            update_params(self.pol_optims[i], self.pols[i], pol_loss, self.grad_clip)

            if self.entropy_tuning:
                entropy_loss = self.calc_entropy_loss(entropy, weights)
                update_params(self.alpha_optimizer, None, entropy_loss)
                self.alpha = torch.clip(
                    self.log_alpha.exp(), -self.alpha_bnd, self.alpha_bnd
                )

            if epoch % self.log_interval == 0:
                with torch.no_grad():
                    (obs, _, _, _, _, _) = batch
                    _, _, act = self.pols[i](obs)
                    _, _, tact = self.teachers[i](obs)

                    q = self.calc_q_from_sf(obs, act, self.sf_tars[i], self.w)
                    tq = self.calc_q_from_sf(obs, tact, self.sf_tars[i], self.w)

                metrics = {
                    "pretrain/loss": pol_loss,
                    "pretrain/q_value": q.mean().item(),
                    "pretrain/tq_value": tq.mean().item(),
                    "pretrain/act": act,
                    "pretrain/tact": tact,
                }
                wandb.log(metrics)

    def explore(self, obs, w):
        acts, qs = self.gpe(obs, w, "explore")
        tacts, tqs = self.tgpe(obs, w)
        acts.extend(tacts)
        qs.extend(tqs)
        act = self.gpi(acts, qs)
        return act

    def exploit(self, obs, w):
        acts, qs = self.gpe(obs, w, "exploit")
        tacts, tqs = self.tgpe(obs, w)
        acts.extend(tacts)
        qs.extend(tqs)
        act = self.gpi(acts, qs)

        if act in tacts:  # TODO:
            self.tea_act_cnt += 1
        else:
            self.stu_act_cnt += 1

        st_act_ratio = self.stu_act_cnt / (self.stu_act_cnt + self.tea_act_cnt)
        metrics = {
            "evaluate/teacher_q0": tqs[0],  # TODO: consider more teachers case
            "evaluate/teacher_q1": tqs[1],  # TODO: consider more teachers case
            "evaluate/student_q0": qs[0],  # TODO: consider more students case
            "evaluate/student_teacher_act_ratio": st_act_ratio,
        }
        wandb.log(metrics)

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
        if self.train_tsf:
            action = ts2np(self.teachers[self.teacher_idx].act(np2ts(obs)))
        else:
            action = super().act(obs, mode)
        return action

    def learn(self):
        if self.train_tsf:
            self.learn_tsf()
        else:
            super().learn()

    def learn_tsf(self):
        self.learn_steps += 1

        if self.learn_steps % self.td_target_update_interval == 0:
            soft_update(self.tsf_tar, self.tsf, self.tau)

        if self.prioritized_memory:
            batch, indices, weights = self.replay_buffer.sample(self.mini_batch_size)
        else:
            batch = self.replay_buffer.sample(self.mini_batch_size)
            weights = 1

        i = self.teacher_idx
        (tsf_loss, errors, mean_tsf, target_tsf,) = self.calc_sf_loss(
            batch, self.tsfs[i], self.tsf_tars[i], self.teachers[i], weights
        )

        update_params(
            self.tsf_optims[i][0], self.tsfs[i].SF0, tsf_loss[0], self.grad_clip
        )
        update_params(
            self.tsf_optims[i][1], self.tsfs[i].SF1, tsf_loss[1], self.grad_clip
        )

        if self.prioritized_memory:
            self.replay_buffer.update_priority(indices, errors.detach().cpu().numpy())

        if self.learn_steps % self.log_interval == 0:
            metrics = {
                "pretrain/tSF0_loss": tsf_loss[0].detach().item(),
                "pretrain/tSF1_loss": tsf_loss[1].detach().item(),
                "pretrain/mean_tSF0": mean_tsf[0],
                "pretrain/mean_tSF1": mean_tsf[1],
                "pretrain/target_tsf": target_tsf.detach().mean().item(),
            }
            w = ts2np(self.w)
            for i in range(w.shape[0]):
                metrics[f"task/{i}"] = w[i]
            wandb.log(metrics)

    def create_teacher_sfs(self):
        if self.share_sf:
            self.tsf = self.sfs[0]
            self.tsf_tar = self.sf_tars[0]
            self.tsf_optim = self.sf_optims[0]
        else:
            self.tsf, self.tsf_tar, self.tsf_optim = self.create_sf()

        self.tsfs.append(self.tsf)
        self.tsf_tars.append(self.tsf_tar)
        self.tsf_optims.append(self.tsf_optim)

    def reset_env(self):
        self.reset_teachers()
        self.stu_act_cnt = 0
        self.tea_act_cnt = 0
        return super().reset_env()

    def reset_teachers(self):
        for pol in self.teachers:
            pol.reset()


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

    env_kwargs = {}
    if env == "multitask-v0":
        env_config = {
            "DBG": False,
            "duration": 1000,
            "simulation": {
                "gui": render,
                "enable_meshes": True,
                "auto_start_simulation": auto_start_simulation,
                "position": (0, 0, 25),  # initial spawned position
            },
            "observation": {
                "DBG_ROS": False,
                "DBG_OBS": False,
                "noise_stdv": 0.0,
                "scale_obs": True,
                "include_raw_state": True,
                "bnd_constraint": True,
            },
            "action": {
                "DBG_ACT": False,
                "act_noise_stdv": 0.01,
                "thrust_range": [-0.25, 0.25],
            },
            "target": {
                "type": "FixedGoal",
                "DBG_ROS": False,
            },
            "tasks": {
                "tracking": {
                    "ori_diff": np.array([0.0, 0.0, 0.0, 0.0]),
                    "ang_diff": np.array([1.0, 1.0, 1.0]),
                    "angvel_diff": np.array([0.0, 0.0, 0.0]),
                    "pos_diff": np.array([0.0, 0.0, 1.0]),
                    "vel_diff": np.array([0.0, 0.0, 1.0]),
                    "vel_norm_diff": np.array([0.0]),
                },
                "constraint": {
                    "survive": np.array([1]),
                    "action_cost": np.array([0.1]),
                    "pos_ubnd_cost": np.array([0.0, 0.0, 0.5]),
                    "pos_lbnd_cost": np.array([0.0, 0.0, 0.5]),
                },
                "success": {
                    "pos": np.array([0.0, 0.0, 10.0]),
                    "fail": np.array([-10.0]),
                },
            },
        }
        env_kwargs = {"config": env_config}

    default_dict = Agent.default_config()
    default_dict.update(
        dict(
            env=env,
            env_kwargs=env_kwargs,
            render=render,
            task_schedule=None,
            n_teacher_episodes=1,  # 10
            min_batch_size=int(2),  # 256
            min_n_experience=int(10),  # 2048
            multi_step=1,
            updates_per_step=2,
            n_particles=64,
            n_gauss=4,
            policy_class="GaussianMixture",
            net_kwargs={"value_sizes": [128, 128], "policy_sizes": [32, 32]},
            pretrain_sf_epochs=10,  # 5e4
            entropy_tuning=True,
            normalize_critic=True,
            compute_target_by_sfv=True,
        )
    )

    wandb.init(config=default_dict)
    print(wandb.config)

    agent = Agent(wandb.config)
    agent.run()
