from copy import deepcopy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

from utils import ExperienceReplay, slice_tensor_tensor, flatten
from utils import save_checkpoint
from model import Network

floatX = 'float32'

class AI(object):
    def __init__(self, state_shape, nb_actions, action_dim, reward_dim,
                 history_len=1, gamma=.99, learning_rate=0.00025, epsilon=0.05,
                 final_epsilon=0.05, test_epsilon=0.0, minibatch_size=32,
                 replay_max_size=100, update_freq=50, learning_frequency=1,
                 num_units=250, remove_features=False, use_mean=False,
                 use_hra=True, rng=None, outf="outputs", cuda=True):
        self.rng = rng
        self.history_len = history_len
        self.state_shape = (1,) + tuple(state_shape)
        self.nb_actions = nb_actions
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.learning_rate_start = learning_rate
        self.epsilon = epsilon
        self.start_epsilon = epsilon
        self.test_epsilon = test_epsilon
        self.final_epsilon = final_epsilon
        self.minibatch_size = minibatch_size
        self.update_freq = update_freq
        self.update_counter = 0
        self.nb_units = num_units
        self.use_mean = use_mean
        self.use_hra = use_hra
        self.remove_features = remove_features
        self.learning_frequency = learning_frequency
        self.replay_max_size = replay_max_size
        self.mse_crit = nn.MSELoss()
        self.cuda = cuda
        self.outf = outf

        self.transitions = ExperienceReplay(max_size=self.replay_max_size,
                                            history_len=history_len,
                                            rng=self.rng,
                                            state_shape=state_shape,
                                            action_dim=action_dim,
                                            reward_dim=reward_dim)

        self.networks = [self._build_network() for _ in range(self.reward_dim)]
        self.optimizers = [optim.RMSprop(self.networks[i].parameters(),
                                         lr=learning_rate, alpha=.95, eps=1e-7)
                                         for i in range(len(self.networks))]

        self.target_networks = [self._build_network() for _ in range(self.reward_dim)]
        self.weight_transfer(from_model=self.networks, to_model=self.target_networks)
        self._compile_learning()

        if self.cuda:
            self.networks = [net_.cuda() for net_ in self.networks]
            self.target_networks = [tnet_.cuda() for tnet_ in self.networks]

        print('Compiled Model and Learning.')

    def _build_network(self):
        return Network(self.state_shape, int(self.nb_units / self.reward_dim),
                       self.nb_actions, self.reward_dim, self.remove_features)

    def _remove_features(self, s, i):
        return torch.cat((s[:, :, :, : -self.reward_dim],
                          torch.unsqueeze(s[:, :, :, self.state_shape[-1]
                                          - self.reward_dim + i], dim=-1)), dim=-1)

    def _compute_cost(self, q, a, r, t, q2):
        zero_target = torch.FloatTensor(self.minibatch_size, 1)
        if self.cuda:
            zero_target = zero_target.cuda()
        zero_target = Variable(zero_target)
        preds = slice_tensor_tensor(q, a, self.cuda)
        if self.cuda:
            preds = preds.cuda()
        preds = Variable(preds)
        bootstrap = torch.max if not self.use_mean else torch.mean
        targets = r + (1 - t) * self.gamma * bootstrap(q2, dim=1)[0]
        zero_target.data.resize_as_(preds.data).fill_(0)
        cost = self.mse_crit(preds-targets, zero_target)
        return cost

    def _compile_learning(self):
        self.s_val  = torch.FloatTensor(self.replay_max_size, self.history_len, *self.state_shape)
        self.s2_val = torch.FloatTensor(self.replay_max_size, self.history_len, *self.state_shape)
        if self.action_dim == 1:
            self.a_val = torch.LongTensor(self.replay_max_size)
        else:
            self.a_val = torch.LongTensor(self.replay_max_size, self.action_dim)
        if self.reward_dim == 1:
            self.r_val = torch.FloatTensor(self.replay_max_size)
        else:
            self.r_val = torch.FloatTensor(self.replay_max_size, self.reward_dim)
        self.t_val  = torch.FloatTensor(self.replay_max_size)
        if self.cuda:
            self.s_val = self.s_val.cuda()
            self.s2_val = self.s2_val.cuda()
            self.a_val = self.a_val.cuda()
            self.r_val = self.r_val.cuda()
            self.t_val = self.t_val.cuda()
        self.s = Variable(self.s_val)
        self.s2 = Variable(self.s2_val)
        self.a = Variable(self.a_val)
        self.r = Variable(self.r_val)
        self.t = Variable(self.t_val)

    def _train_on_batch(self, s_, a_, r_, s2_, t_, _updates=None):
        t_ = t_.astype('int')
        s = torch.from_numpy(s_)
        a = torch.from_numpy(a_)
        r = torch.from_numpy(r_)
        t = torch.from_numpy(t_)
        s2 = torch.from_numpy(s2_)
        a = a.type(torch.LongTensor)
        t = t.type(torch.FloatTensor)
        s = s.type(torch.FloatTensor)
        s2 = s2.type(torch.FloatTensor)
        if self.cuda:
            s, s2 = s.cuda(), s2.cuda()
            a, r, t = a.cuda(), r.cuda(), t.cuda()
        updates = []
        if _updates is not None:
            updates = _updates
        costs = 0
        qs = []
        q2s = []
        self.a.data.resize_as_(a).copy_(a)
        self.t.data.resize_as_(t).copy_(t)
        self.r.data.resize_as_(r).copy_(r)
        self.s.data.resize_as_(s).copy_(s)
        self.s2.data.resize_as_(s2).copy_(s2)
        for i in range(len(self.networks)):
            local_s = s
            local_s2 = s2
            if self.remove_features:
                local_s = self._remove_features(local_s, i)
                local_s2 = self._remove_features(local_s2, i)
                self.s.data.resize_as_(local_s).copy_(local_s)
                self.s2.data.resize_as_(local_s2).copy_(local_s2)
            qs.append(self.networks[i](self.s))
            q2s.append(self.target_networks[i](self.s2))
            if self.use_hra:
                cost = self._compute_cost(qs[-1], self.a, self.r[:, i], self.t, q2s[-1])
                cost.backward()
                costs += cost
            self.optimizers[i].step()
        if not self.use_hra:
            qs = [torch.unsqueeze(qs_, 0) for qs_ in qs]
            q2s = [torch.unsqueeze(q2s_, 0) for q2s_ in q2s]
            q = torch.sum(torch.cat(qs), dim=0)
            q2 = torch.sum(torch.cat(q2s), dim=0)
            summed_reward = torch.sum(self.r, dim=-1)
            cost = self._compute_cost(q, self.a, summed_reward, self.t, q2)
            cost.backward()
            costs += cost
            self.optimizers[i].step()
        return costs

    def predict_network(self, s_):
        s = torch.from_numpy(s_)
        s = s.type(torch.FloatTensor)
        if self.cuda:
            s = s.cuda()
        qs = []
        self.s.data.resize_as_(s).copy_(s)
        for i in range(len(self.networks)):
            local_s = s
            if self.remove_features:
                local_s = self._remove_features(local_s, i)
                self.s.data.resize_as_(local_s).copy_(local_s)
            qs.append(self.networks[i](self.s))
        return qs

    def update_weights(self):
        target_updates = []
        for network, target_network in zip(self.networks, self.target_networks):
            for name, param in target_network.state_dict().items():
                param = network.state_dict()[name].clone()

    def update_lr(self, cur_step, total_steps):
        self.learning_rate = ((total_steps - cur_step - 1) / total_steps) * self.learning_rate_start
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate

    def get_max_action(self, states):
        states = self._reshape(states)
        x = self.predict_network(states)
        x = [torch.unsqueeze(x_, 0) for x_ in x]
        x = torch.cat(x, dim=0)
        q = x.cpu().data.numpy()
        q = np.sum(q, axis=0)
        return np.argmax(q, axis=1)

    def get_action(self, states, evaluate):
        eps = self.epsilon if not evaluate else self.test_epsilon
        if self.rng.binomial(1, eps):
            return self.rng.randint(self.nb_actions)
        else:
            return self.get_max_action(states=states)

    def train_on_batch(self, s, a, r, s2, t):
        s = self._reshape(s)
        s2 = self._reshape(s2)
        if len(r.shape) == 1:
            r = np.expand_dims(r, axis=-1)
        return self._train_on_batch(s, a, r, s2, t)

    def learn(self):
        assert self.minibatch_size <= self.transitions.size, 'not enough data in the pool'
        s, a, r, s2, term = self.transitions.sample(self.minibatch_size)
        objective = self.train_on_batch(s, a, r, s2, term)
        if self.update_counter == self.update_freq:
            self.update_weights()
            self.update_counter = 0
        else:
            self.update_counter += 1
        return objective

    def dump_network(self):
        save_dict = {}
        for i, network in enumerate(self.networks):
            save_dict["net_"+str(i)] = network.state_dict()
            save_dict["opt_"+str(i)] = self.optimizers[i].state_dict()
        save_checkpoint(save_dict, is_best=True, path=self.outf)

    def load_weights(self, checkpoint=None):
        print("=> loading checkpoint '{}'".format(checkpoint))
        checkpoint = torch.load(checkpoint)
        for i, network in enumerate(self.networks):
            network.load_state_dict(checkpoint["net_"+str(i)])
            self.optimizers[i].load_state_dict(checkpoint["opt_"+str(i)])
        self.update_weights()

    @staticmethod
    def _reshape(states):
        if len(states.shape) == 2:
            states = np.expand_dims(states, axis=0)
        if len(states.shape) == 3:
            states = np.expand_dims(states, axis=1)
        return states

    @staticmethod
    def weight_transfer(from_model, to_model):
        for f_model, t_model in zip(from_model, to_model):
            for name, param in t_model.state_dict().items():
                param = f_model.state_dict()[name].clone()
