import torch
import torch.nn as nn
import torch.nn.functional as F
from distributions import get_distribution


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class HistoryCell(nn.Module):
    def __init__(self,num_inputs,input_size,hid_size,batch_size=1):
        super(HistoryCell, self).__init__()
        '''
        self.hsc_main = nn.Sequential(
            nn.Conv2d(num_inputs, 16, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 16, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(16 * 7 * 7,input_size),
            nn.ReLU()
        )
        '''
        self.batch_size=batch_size
        self.hsc_lstm=nn.LSTMCell(input_size,hid_size)
        self.hid_size=hid_size

    def forward(self, input,h,c):
        #x=self.hsc_main(input)
        h,c=self.hsc_lstm(input,(h,c))
        return h,c

class FutureCell(nn.Module):
    def __init__(self,num_inputs,input_size,hid_size,batch_size=1):
        super(FutureCell, self).__init__()
        '''
        self.ftc_main = nn.Sequential(
            nn.Conv2d(num_inputs, 16, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 16, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(16 * 7 * 7,input_size),
            nn.ReLU()
        )
        '''
        self.batch_size=batch_size
        self.ftc_lstm=nn.LSTMCell(input_size,hid_size)
        self.hid_size=hid_size

    def forward(self, input,h,c):
        #x=self.ftc_main(input)
        h,c=self.ftc_lstm(input,(h,c))
        return h,c

class History(nn.Module):
    def __init__(self,num_inputs,input_size,hid_size,num_layers,batch_size):
        super(History, self).__init__()
        '''
        self.hs_main = nn.Sequential(
            nn.Conv2d(num_inputs, 16, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 16, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(16 * 7 * 7,input_size),
            nn.ReLU()
        )
        '''
        self.hs_lstm=nn.LSTM(input_size,hid_size,num_layers)
        self.hs_h0=torch.zeros(num_layers,batch_size,hid_size).cuda()
        self.hs_c0=torch.zeros(num_layers,batch_size,hid_size).cuda()
        self.hid_size=hid_size

    def forward(self, input):
        #x=self.hs_main(input)
        x=input.unsqueeze(1)
        out,(h,c)=self.hs_lstm(x,(self.hs_h0,self.hs_c0))
        return h.view(1,2*self.hid_size)

class Future(nn.Module):
    def __init__(self,num_inputs,input_size,hid_size,num_layers,batch_size):
        super(Future, self).__init__()
        '''
        self.ft_main = nn.Sequential(
            nn.Conv2d(num_inputs,16, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 16, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(16 * 7 * 7,input_size),
            nn.ReLU()
        )
        '''
        self.ft_lstm=nn.LSTM(input_size,hid_size,num_layers)
        self.ft_h0=torch.zeros(num_layers,batch_size,hid_size).cuda()
        self.ft_c0=torch.zeros(num_layers,batch_size,hid_size).cuda()
        self.hid_size=hid_size

    def forward(self, input):
        #x=self.ft_main(input)
        x=input.unsqueeze(1)
        out,(h,c)=self.ft_lstm(x,(self.ft_h0,self.ft_c0))
        return h.view(1,2*self.hid_size)


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        """
        All classes that inheret from Policy are expected to have
        a feature exctractor for actor and critic (see examples below)
        and modules called linear_critic and dist. Where linear_critic
        takes critic features and maps them to value and dist
        represents a distribution of actions.        
        """
        
    def forward(self, inputs, states, masks):
        raise NotImplementedError

    def get_feat(self,inputs):
        hidden_feat= self(inputs)
        return hidden_feat

    def cat(self,hidden_feat,hs_info):
        #hidden_feat = torch.cat((hidden_feat, hs_info), 1)
        return hidden_feat

    def act(self, hidden_feat, states, deterministic=False):
        #hidden_critic, hidden_actor, states = self(inputs,hs_info, states, masks)
        
        action = self.dist.sample(hidden_feat, deterministic=deterministic)

        action_log_probs, dist_entropy = self.dist.logprobs_and_entropy(hidden_feat, action)
        value = self.critic_linear(hidden_feat)
        
        return value, action, action_log_probs, states

    def get_value(self, hidden_feat):
        #hidden_critic, _, states = self(hidden_feat, hs_info,states, masks)
        value = self.critic_linear(hidden_feat)
        return value
    
    def evaluate_actions(self, hidden_feat, states,actions):
        #hidden_critic, hidden_actor, states = self(inputs, hs_info,states, masks)

        action_log_probs, dist_entropy = self.dist.logprobs_and_entropy(hidden_feat, actions)
        value = self.critic_linear(hidden_feat)
        
        return value, action_log_probs, dist_entropy, states


class CNNPolicy(Policy):
    def __init__(self, num_inputs, action_space, hid_size,feat_size,use_gru):
        super(CNNPolicy, self).__init__()
        self.hid_size=hid_size
        self.feat_size=feat_size
        self.main = nn.Sequential(
            nn.Conv2d(num_inputs, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(32 * 7 * 7, self.feat_size),
            nn.ReLU()
        )
        
        if use_gru:
            self.gru = nn.GRUCell(self.feat_size+self.hid_size*2, self.feat_size+self.hid_size*2)

        #self.critic_linear = nn.Linear(self.feat_size+self.hid_size*2, 1)
        self.critic_linear = nn.Linear(self.feat_size, 1)
        #self.dist = get_distribution(self.feat_size+self.hid_size*2, action_space)
        self.dist = get_distribution(self.feat_size, action_space)
        self.train()
        self.reset_parameters()

    @property
    def state_size(self):
        if hasattr(self, 'gru'):
            return self.feat_size
        else:
            return 1

    @property
    def hidden_size(self):
        return self.hid_size


    def reset_parameters(self):
        self.apply(weights_init)

        def mult_gain(m):
            relu_gain = nn.init.calculate_gain('relu')
            classname = m.__class__.__name__
            if classname.find('Conv') != -1 or classname.find('Linear') != -1:
                m.weight.data.mul_(relu_gain)
    
        self.main.apply(mult_gain)

        if hasattr(self, 'gru'):
            nn.init.orthogonal_(self.gru.weight_ih.data)
            nn.init.orthogonal_(self.gru.weight_hh.data)
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)

        if self.dist.__class__.__name__ == "DiagGaussian":
            self.dist.fc_mean.weight.data.mul_(0.01)

    def forward(self, inputs):
        x = self.main(inputs / 255.0)

        '''
        if hasattr(self, 'gru'):
            if inputs.size(0) == states.size(0):
                x = states = self.gru(x, states * masks)
            else:
                x = x.view(-1, states.size(0), x.size(1))
                masks = masks.view(-1, states.size(0), 1)
                outputs = []
                for i in range(x.size(0)):
                    hx = states = self.gru(x[i], states * masks[i])
                    outputs.append(hx)
                x = torch.cat(outputs, 0)
        '''
        #print(x.shape)
        #print(hs_info.shape)
        #print(hs_info.type())

        #print(x.shape)
        return x


def weights_init_mlp(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class MLPPolicy(Policy):
    def __init__(self, num_inputs, action_space):
        super(MLPPolicy, self).__init__()

        self.action_space = action_space

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )

        self.critic_linear = nn.Linear(64, 1)
        self.dist = get_distribution(64, action_space)

        self.train()
        self.reset_parameters()

    @property
    def state_size(self):
        return 1

    def reset_parameters(self):
        self.apply(weights_init_mlp)
        if self.dist.__class__.__name__ == "DiagGaussian":
            self.dist.fc_mean.weight.data.mul_(0.01)
    
    def forward(self, inputs, states, masks):
        hidden_critic = self.critic(inputs)
        hidden_actor = self.actor(inputs)

        return hidden_critic, hidden_actor, states
