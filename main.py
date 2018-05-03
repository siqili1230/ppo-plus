import copy
import glob
import os
import time
from collections import deque
from plot import plot_line
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gc
import sys
from arguments import get_args
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
from envs import make_env
from model import CNNPolicy, MLPPolicy,History,Future,HistoryCell, FutureCell
from storage import RolloutStorage
from visualize import visdom_plot

import algo

args = get_args()

assert args.algo in ['a2c', 'ppo', 'acktr']
if args.recurrent_policy:
    assert args.algo in ['a2c', 'ppo'], \
        'Recurrent policy is not implemented for ACKTR'

num_updates = int(args.num_frames) // args.num_steps // args.num_processes

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

try:
    os.makedirs(args.log_dir)
except OSError:
    files = glob.glob(os.path.join(args.log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)

def safe(xs):
    if len(xs) == 0:
        return np.nan,np.nan,np.nan,np.nan
    else:
        return np.mean(xs),np.median(xs),np.min(xs),np.max(xs)

def main():
    print("#######")
    print("WARNING: All rewards are clipped or normalized so you need to use a monitor (see envs.py) or visdom plot to get true rewards")
    print("#######")

    os.environ['OMP_NUM_THREADS'] = '1'
    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #os.environ['CUDA_VISIBLE_DEVICES'] = "9"
    if args.vis:
        from visdom import Visdom
        viz = Visdom(port=args.port)
        win = None

    envs = [make_env(args.env_name, args.seed, i, args.log_dir, args.add_timestep)
                for i in range(args.num_processes)]

    if args.num_processes > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    if len(envs.observation_space.shape) == 1:
        envs = VecNormalize(envs)

    obs_shape = envs.observation_space.shape
    obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])

    if len(envs.observation_space.shape) == 3:
        actor_critic = CNNPolicy(obs_shape[0], envs.action_space,args.hid_size, args.feat_size,args.recurrent_policy)
    else:
        assert not args.recurrent_policy, \
            "Recurrent policy is not implemented for the MLP controller"
        actor_critic = MLPPolicy(obs_shape[0], envs.action_space)

    if envs.action_space.__class__.__name__ == "Discrete":
        action_shape = 1
    else:
        action_shape = envs.action_space.shape[0]
    if args.use_cell:
        hs = HistoryCell(obs_shape[0], actor_critic.feat_size, 2*actor_critic.hidden_size, 1)
        ft = FutureCell(obs_shape[0], actor_critic.feat_size, 2 * actor_critic.hidden_size, 1)
    else:
        hs = History(obs_shape[0], actor_critic.feat_size, actor_critic.hidden_size, 2, 1)
        ft = Future(obs_shape[0], actor_critic.feat_size, actor_critic.hidden_size, 2, 1)

    if args.cuda:
        actor_critic=actor_critic.cuda()
        hs = hs.cuda()
        ft = ft.cuda()
    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, lr=args.lr,
                               eps=args.eps, alpha=args.alpha,
                               max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(actor_critic, hs,ft,args.clip_param, args.ppo_epoch, args.num_mini_batch,
                         args.value_loss_coef, args.entropy_coef, args.hf_loss_coef,ac_lr=args.lr,hs_lr=args.lr,ft_lr=args.lr,
                                eps=args.eps,
                                max_grad_norm=args.max_grad_norm,
                                num_processes=args.num_processes,
                                num_steps=args.num_steps,
                                use_cell=args.use_cell,
                                lenhs=args.lenhs,lenft=args.lenft,
                                plan=args.plan,
                                ac_intv=args.ac_interval,
                                hs_intv=args.hs_interval,
                                ft_intv=args.ft_interval
                                )
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, acktr=True)

    rollouts = RolloutStorage(args.num_steps, args.num_processes, obs_shape, envs.action_space, actor_critic.state_size,
                              feat_size=512)
    current_obs = torch.zeros(args.num_processes, *obs_shape)

    def update_current_obs(obs):
        shape_dim0 = envs.observation_space.shape[0]
        obs = torch.from_numpy(obs).float()
        if args.num_stack > 1:
            current_obs[:, :-shape_dim0] = current_obs[:, shape_dim0:]
        current_obs[:, -shape_dim0:] = obs

    obs = envs.reset()
    update_current_obs(obs)

    rollouts.observations[0].copy_(current_obs)


    if args.cuda:
        current_obs = current_obs.cuda()
        rollouts.cuda()

    rec_x = []
    rec_y = []
    file = open('./rec/' + args.env_name + '_' + args.method_name + '.txt', 'w')

    hs_info = torch.zeros(args.num_processes, 2 * actor_critic.hidden_size).cuda()
    hs_ind = torch.IntTensor(args.num_processes, 1).zero_()

    epinfobuf = deque(maxlen=100)
    start_time = time.time()
    for j in range(num_updates):
        print('begin sample, time  {}'.format(time.strftime("%Hh %Mm %Ss",
                                                                time.gmtime(time.time() - start_time))))
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                rollouts.feat[step]=actor_critic.get_feat(rollouts.observations[step])

                if args.use_cell:
                    for i in range(args.num_processes):
                        h = torch.zeros(1, 2 * actor_critic.hid_size).cuda()
                        c = torch.zeros(1, 2 * actor_critic.hid_size).cuda()
                        start_ind = max(hs_ind[i],step+1-args.lenhs)
                        for ind in range(start_ind,step+1):
                            h,c=hs(rollouts.feat[ind,i].unsqueeze(0),h,c)
                        hs_info[i,:]=h.view(1,2*actor_critic.hid_size)
                        del h,c
                        gc.collect()
                else:
                    for i in range(args.num_processes):
                        start_ind = max(hs_ind[i], step + 1 - args.lenhs)
                        hs_info[i,:]=hs(rollouts.feat[start_ind:step+1,i])

                hidden_feat=actor_critic.cat(rollouts.feat[step],hs_info)
                value, action, action_log_prob, states = actor_critic.act(
                        hidden_feat,
                        rollouts.states[step])
            cpu_actions = action.data.squeeze(1).cpu().numpy()

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(cpu_actions)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo:
                    epinfobuf.extend([maybeepinfo['r']])
            reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            hs_ind = ((1-masks)*(step+1)+masks*hs_ind.float()).int()

            if args.cuda:
                masks = masks.cuda()

            if current_obs.dim() == 4:
                current_obs *= masks.unsqueeze(2).unsqueeze(2)
            else:
                current_obs *= masks

            update_current_obs(obs)
            rollouts.insert(current_obs, hs_ind,states.data, action.data, action_log_prob.data, value.data, reward, masks)
        with torch.no_grad():
            rollouts.feat[-1] = actor_critic.get_feat(rollouts.observations[-1])
            if args.use_cell:
                for i in range(args.num_processes):
                    h = torch.zeros(1, 2 * actor_critic.hid_size).cuda()
                    c = torch.zeros(1, 2 * actor_critic.hid_size).cuda()
                    start = max(hs_ind[i], step + 1 - args.lenhs)
                    for ind in range(start, step + 1):
                        h, c = hs(rollouts.feat[ind, i].unsqueeze(0), h, c)
                    hs_info[i, :] = h.view(1, 2 * actor_critic.hid_size)
                    del h,c
            else:
                for i in range(args.num_processes):
                    start_ind = max(hs_ind[i], step + 1 - args.lenhs)
                    hs_info[i, :] = hs(rollouts.feat[start_ind:step + 1, i])
            hidden_feat = actor_critic.cat(rollouts.feat[-1],hs_info)
            next_value = actor_critic.get_value(hidden_feat).detach()
        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)
        rollouts.compute_ft_ind()

        print('begin update, time  {}'.format(time.strftime("%Hh %Mm %Ss",
                                     time.gmtime(time.time() - start_time))))
        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        print('end update, time  {}'.format(time.strftime("%Hh %Mm %Ss",
                                                            time.gmtime(time.time() - start_time))))
        rollouts.after_update()

        if j % args.save_interval == 0 and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            # A really ugly way to save a model to CPU
            save_model = actor_critic
            if args.cuda:
                save_model = copy.deepcopy(actor_critic).cpu()

            save_model = [save_model,
                            hasattr(envs, 'ob_rms') and envs.ob_rms or None]

            torch.save(save_model, os.path.join(save_path, args.env_name + ".pt"))

        if j % args.log_interval == 0:
            end = time.time()
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            v_mean,v_median,v_min,v_max = safe(epinfobuf)
            print("Updates {}, num timesteps {},time {}, FPS {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, entropy {:.5f}, value loss {:.5f}, policy loss {:.5f}".
                format(j, total_num_steps,
                       time.strftime("%Hh %Mm %Ss",
                                     time.gmtime(time.time() - start_time)),
                       int(total_num_steps / (end - start_time)),
                       v_mean, v_median, v_min, v_max,
                       dist_entropy,
                       value_loss, action_loss))

            if not (v_mean==np.nan):
                rec_x.append(total_num_steps)
                rec_y.append(v_mean)
                file.write(str(total_num_steps))
                file.write(' ')
                file.writelines(str(v_mean))
                file.write('\n')

        if args.vis and j % args.vis_interval == 0:
            try:
                # Sometimes monitor doesn't properly flush the outputs
                win = visdom_plot(viz, win, args.log_dir, args.env_name,
                                  args.algo, args.num_frames)
            except IOError:
                pass
    plot_line(rec_x, rec_y, './imgs/' + args.env_name + '_' + args.method_name + '.png', args.method_name,
              args.env_name, args.num_frames)
    file.close()

if __name__ == "__main__":
    main()
