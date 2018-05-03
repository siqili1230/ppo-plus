import torch
import torch.nn as nn
import torch.optim as optim
import gc
from .kfac import KFACOptimizer


class PPO(object):
    def __init__(self,
                 actor_critic,
                 hs,
                 ft,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 hf_loss_coef,
                 ac_lr=None,
                 hs_lr=None,
                 ft_lr=None,
                 eps=None,
                 max_grad_norm=None,
                 num_processes=0,
                 num_steps=0,
                 use_cell=True,lenhs=128,lenft=128,
                 plan=0,
                 ac_intv=0,
                 hs_intv=0,
                 ft_intv=0,):
        self.plan=plan
        self.ac_intv = ac_intv
        self.hs_intv = hs_intv
        self.ft_intv = ft_intv
        self.actor_critic = actor_critic
        self.hs = hs
        self.ft = ft
        self.lenhs = lenhs
        self.lenft = lenft
        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch
        self.num_processes = num_processes
        self.num_steps = num_steps
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.hidden_size = actor_critic.hidden_size
        self.max_grad_norm = max_grad_norm
        self.hf_loss_coef = hf_loss_coef
        self.ac_optimizer = optim.Adam(actor_critic.parameters(), lr=ac_lr, eps=eps)
        self.hs_optimizer = optim.Adam(hs.parameters(), lr=hs_lr, eps=eps)
        self.ft_optimizer = optim.Adam(ft.parameters(), lr=ft_lr, eps=eps)
        self.use_cell=use_cell

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)
        mini_batch_size = self.num_processes * self.num_steps // self.num_mini_batch
        count=0

        for e in range(self.ppo_epoch):
            tem=self.actor_critic.get_feat(rollouts.observations[:-1].view(-1,
                                        *rollouts.observations.size()[2:]))
            if hasattr(self.actor_critic, 'gru'):
                data_generator = rollouts.recurrent_generator(
                    advantages, tem, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, tem, self.num_mini_batch)


            for sample in data_generator:
                print(e,count)
                feat_batch, pro_ind,hs_ind,now_ind,ft_ind,states_batch, actions_batch, \
                   return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ = sample
                hs_info_batch = torch.zeros(mini_batch_size, 2 * self.hidden_size).cuda()
                ft_info_batch = torch.zeros(mini_batch_size, 2 * self.hidden_size).cuda()
                if self.use_cell:
                    for i in range(mini_batch_size):
                        h = torch.zeros(1, 2 * self.hidden_size).cuda()
                        c = torch.zeros(1, 2 * self.hidden_size).cuda()
                        start_ind=max(hs_ind[i],int(now_ind[i]) + 1-self.lenhs)
                        for ind in range(start_ind,(now_ind[i]+1)):
                            h,c=self.hs(rollouts.feat[ind,pro_ind[i]].unsqueeze(0),h,c)
                        hs_info_batch[i] = h.view(1,2*self.hidden_size)
                        del h,c
                        gc.collect()
                        h = torch.zeros(1, 2 * self.hidden_size).cuda()
                        c = torch.zeros(1, 2 * self.hidden_size).cuda()
                        end_ind=min(ft_ind[i],int(now_ind[i]) + 1 + self.lenft)
                        for ind in range((now_ind[i] + 1),end_ind):
                            h,c=self.ft(rollouts.feat[ind,pro_ind[i]].unsqueeze(0),h,c)
                        ft_info_batch[i] = h.view(1,2*self.hidden_size)
                        del h,c
                        gc.collect()
                else:
                    for i in range(mini_batch_size):
                        start_ind = max(hs_ind[i], int(now_ind[i]) + 1 - self.lenhs)
                        hs_info_batch[i] = self.hs(rollouts.feat[start_ind:(now_ind[i]+1),pro_ind[i]])
                        end_ind = min(ft_ind[i], int(now_ind[i]) + 1 + self.lenft)
                        ft_info_batch[i] = self.ft(rollouts.feat[(now_ind[i] + 1):end_ind, pro_ind[i]])

                # Reshape to do in a single forward pass for all steps
                if self.plan==0:
                    feat_batch = self.actor_critic.cat(feat_batch,hs_info_batch)

                elif self.plan==1:
                    feat_batch = self.actor_critic.cat(feat_batch, ft_info_batch)

                values, action_log_probs, dist_entropy, states = self.actor_critic.evaluate_actions(
                        feat_batch, states_batch, actions_batch)
                #print(feat_batch.shape)

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                           1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                value_loss = (return_batch - values).pow(2).mean()
                hf_loss = (hs_info_batch-ft_info_batch).pow(2).mean()
                loss = value_loss * self.value_loss_coef + action_loss -\
                 dist_entropy * self.entropy_coef + hf_loss * self.hf_loss_coef
                self.ac_optimizer.zero_grad()
                self.hs_optimizer.zero_grad()
                self.ft_optimizer.zero_grad()
                loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.hs.parameters(),
                                         self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.ft.parameters(),
                                         self.max_grad_norm)
                if count % self.ac_intv == 0:
                    self.ac_optimizer.step()
                if count % self.hs_intv == 0:
                    self.hs_optimizer.step()
                if count % self.ft_intv == 0:
                    self.ft_optimizer.step()
                count+=1

        gc.collect()

        return value_loss, action_loss, dist_entropy
