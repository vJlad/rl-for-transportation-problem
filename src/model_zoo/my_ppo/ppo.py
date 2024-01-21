import torch
from torch.optim import Adam

import time
import wandb
import logging
from tqdm.auto import tqdm
from typing import Callable

from model_zoo.my_ppo.metrics import EpochMetricsAggregator, FullAggregator
from src.batched_env.env_wrapper import BatchedEnvWrapper
from src.model_zoo.my_ppo.buffering import ReplayBuffer
from src.model_zoo.my_ppo.actor_critic_models.base import ActorCriticFabric


def ppo(
        env_fn: Callable[[], BatchedEnvWrapper],
        actor_critic: ActorCriticFabric,
        seed: int = 0,
        steps_per_epoch: int = 4000,
        epochs: int = 50,
        gamma: float = 0.99,
        clip_ratio: float = 0.2,
        pi_lr: float = 3e-4,
        vf_lr: float = 1e-3,
        train_pi_iters: int = 80,
        train_v_iters: int = 80,
        lam: float = 0.97,
        max_ep_len: int = 1000,
        target_kl: float = 0.01,
        device: str = 'cuda'
):
    train_steps_per_epoch = max(train_v_iters, train_pi_iters)
    local_steps_per_epoch = int(steps_per_epoch)

    # Set up logger
    epoch_aggregator = EpochMetricsAggregator({
        'VVals': FullAggregator.clone().to(device=device),
        'EpRet': FullAggregator.clone().to(device=device),
        'EpLen': FullAggregator.clone().to(device=device),
    })

    # Random seed
    torch.manual_seed(seed)
    # np.random.seed(seed)

    # Instantiate environment
    env = env_fn()

    # Create actor-critic module
    ac = actor_critic(env).to(device)

    # Count variables
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    var_counts = tuple(count_parameters(module) for module in [ac.pi, ac.v])
    logging.info('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)

    # Set up experience buffer
    buf = ReplayBuffer(env, local_steps_per_epoch, gamma, lam, epoch_aggregator, max_ep_len, device)

    # Set up function for computing PPO policy loss
    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        pi, logp = ac.pi(obs, act)
        # print(list(zip(logp, logp_old))[:10])
        # raise "KEK"
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        return ((ac.v(obs) - ret) ** 2).mean()

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    # Wandb logger config # TODO: rewrite logger
    config = dict(
        seed=seed,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        gamma=gamma,
        clip_ratio=clip_ratio,
        pi_lr=pi_lr,
        vf_lr=vf_lr,
        train_pi_iters=train_pi_iters,
        train_v_iters=train_v_iters,
        lam=lam,
        max_ep_len=max_ep_len,
        target_kl=target_kl,
        actor_critic_classname=actor_critic.__class__.__name__,
        env_dstrategy=env.d_strategy.__class__.__name__,
    )
    with wandb.init(project='transp-batched_env-ppo', config=config) as _, \
            tqdm(total=train_pi_iters, desc='pi_iter') as tqdm_train_pi, \
            tqdm(total=train_v_iters, desc='v_iter') as tqdm_train_v, \
            tqdm(total=local_steps_per_epoch, desc='playing') as tqdm_playing:
        log_mem = None

        def update():
            ac.train()
            data = buf.get()

            # Train policy with multiple steps of gradient descent
            tqdm_train_pi.reset()
            for i in range(train_pi_iters):
                pi_optimizer.zero_grad()
                loss_pi, pi_info = compute_loss_pi(data)
                # kl = mpi_avg(pi_info['kl'])
                kl = pi_info['kl']
                log_mem[i].update(dict(
                    LossPi=loss_pi.item(),
                    KL=pi_info['kl'],
                    Entropy=pi_info['ent'],
                    ClipFrac=pi_info['cf'],
                    TargetKL=target_kl,
                    MaxKL=1.5 * target_kl,
                ))
                tqdm_train_pi.set_postfix(dict(
                    kl=kl, target_kl=target_kl, kl_rel=kl / target_kl
                ))
                if kl > 1.5 * target_kl:
                    logging.debug(f'Early stopping at step {i} due to reaching max kl.'
                                  f'({kl=}, {target_kl=}, {kl / target_kl=})')
                    break
                loss_pi.backward()
                # mpi_avg_grads(ac.pi)  # average grads across MPI processes
                pi_optimizer.step()
                tqdm_train_pi.update()

            # noinspection PyUnboundLocalVariable
            log_mem[train_steps_per_epoch - 1].update(dict(StopIter=i))

            # Value function learning
            tqdm_train_v.reset()
            for i in range(train_v_iters):
                vf_optimizer.zero_grad()
                loss_v = compute_loss_v(data)
                loss_v.backward()
                # mpi_avg_grads(ac.v)  # average grads across MPI processes
                vf_optimizer.step()
                log_mem[i].update(dict(
                    LossV=loss_v.item(),
                ))
                tqdm_train_v.update()

        # @torch.no_grad()
        # def greedy_validation():
        #     o = batched_env.reset()
        #     r_mem = []
        #     num_actions = batched_env.supplier_network.number_of_actions()
        #     ac.eval()
        #     while True:
        #         dist, _ = ac.pi(o)
        #         a = dist.log_prob(torch.arange(num_actions, device=device)).argmax().item()
        #         o, r, done = batched_env.step(a)
        #         r_mem.append(r)
        #         if done:
        #             break
        #     return sum(r_mem)

        # Prepare for interaction with environment
        start_time = time.time()

        # Main loop: collect experience in batched_env and update/log each epoch
        for epoch in tqdm(range(epochs), 'epoch'):
            log_mem = [{} for _ in range(train_steps_per_epoch)]

            # Fill buffer with current policy actions
            buf.sample_new_trajectories(ac, tqdm_playing)

            # Perform PPO update!
            update()

            # Play greedy strategy with current policy for validation purposes
            # greedy_reward = greedy_validation()

            # Log info about epoch
            epoch_aggregator.store(Epoch=epoch)
            # epoch_aggregator.store(GreedyReward=greedy_reward)
            epoch_aggregator.store(TotalEnvInteracts=(epoch + 1) * steps_per_epoch)
            epoch_aggregator.store(Time=time.time() - start_time)
            log_mem[train_steps_per_epoch - 1].update(epoch_aggregator.close_epoch())
            for i, metrics in enumerate(log_mem):
                wandb.log(metrics, step=epoch * train_steps_per_epoch + i)

    return ac
