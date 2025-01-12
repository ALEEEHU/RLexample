# Pytorch implementation of PG with baseline
#
# Bolei Zhou, 22 Feb 2019
# PENG Zhenghao, updated 23 October 2021
import argparse
import os
from itertools import count

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

is_cuda = torch.cuda.is_available()

parser = argparse.ArgumentParser(description='PyTorch PG with baseline example at openai-gym pong')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99')
parser.add_argument('--decay_rate', type=float, default=0.99, metavar='G',
                    help='decay rate for RMSprop (default: 0.99)')
parser.add_argument('--learning_rate', type=float, default=3e-4, metavar='G',
                    help='learning rate (default: 1e-4)')
parser.add_argument('--batch_size', type=int, default=20, metavar='G',
                    help='Every how many episodes to da a param update')
parser.add_argument('--seed', type=int, default=87, metavar='N',
                    help='random seed (default: 87)')
parser.add_argument('--test', action='store_true',
                    help='whether to test the trained model or keep training')
parser.add_argument('--use_value_gradient', action='store_true',
                    help='whether to pass gradient to value network when computing policy loss. Test purpose only!')
parser.add_argument('--disable_model_loading', action='store_true',
                    help='whether to skip the model loading process. Test purpose only!')

args = parser.parse_args()

env = gym.make('Pong-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)

D = 80 * 80
test = args.test
if test == True:
    render = True
else:
    render = False


def prepro(I):
    """ prepro 210x160x3 into 6400 """
    I = I[35:195]
    I = I[::2, ::2, 0]
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0] = 1
    return I.astype(np.float).ravel()


class PGbaseline(nn.Module):
    def __init__(self, num_actions=2):
        super(PGbaseline, self).__init__()
        self.affine1 = nn.Linear(6400, 200)
        self.action_head = nn.Linear(200, num_actions)  # action 1: static, action 2: move up, action 3: move down
        self.value_head = nn.Linear(200, 1)

        self.num_actions = num_actions
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x)) # feature是share的
        action_scores = self.action_head(x) #输出head1——应该采取的策略
        state_values = self.value_head(x)# 输出head2——实际价值
        return F.softmax(action_scores, dim=-1), state_values #softmax输出的为概率

    def select_action(self, x):#得到stochastic policy过后会对其进行采样
        x = Variable(torch.from_numpy(x).float().unsqueeze(0))
        if is_cuda: x = x.cuda()
        probs, state_value = self.forward(x)
        m = Categorical(probs)
        action = m.sample()#采样来得到action

        self.saved_log_probs.append((m.log_prob(action), state_value))
        return action


# built policy network
policy = PGbaseline()
if is_cuda:
    policy.cuda()

# check & load pretrain model
if os.path.isfile('pgb_params.pkl') and (not args.disable_model_loading):
    print('Load PGbaseline Network parametets ...')
    if is_cuda:
        policy.load_state_dict(torch.load('pgb_params.pkl'))
    else:
        policy.load_state_dict(torch.load('pgb_params.pkl', map_location=lambda storage, loc: storage))

# construct a optimal function
optimizer = optim.RMSprop(policy.parameters(), lr=args.learning_rate, weight_decay=args.decay_rate)


def finish_episode():
    R = 0
    policy_loss = []
    value_loss = []
    rewards = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    # turn rewards to pytorch tensor and standardize
    rewards = torch.Tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
    if is_cuda:
        rewards = rewards.cuda()
    for (log_prob, value), reward in zip(policy.saved_log_probs, rewards):
        advantage = reward - value
        if args.use_value_gradient:
            pass
        else:
            advantage = advantage.detach()
        policy_loss.append(- log_prob * advantage)  # policy gradient (policy loss)
        value_loss.append(F.smooth_l1_loss(value, reward))  # value function approximation (value loss)
    optimizer.zero_grad()
    policy_loss = torch.stack(policy_loss).sum()
    value_loss = torch.stack(value_loss).sum()
    loss = policy_loss + value_loss
    if is_cuda:
        loss.cuda()
    loss.backward()
    optimizer.step()

    # clean rewards and saved_actions
    del policy.rewards[:]
    del policy.saved_log_probs[:]


# Main loop
running_reward = None
reward_sum = 0
for i_episode in count(1):
    state = env.reset()
    prev_x = None
    for t in range(10000):
        if render: env.render()
        cur_x = prepro(state)
        x = cur_x - prev_x if prev_x is not None else np.zeros(D)
        prev_x = cur_x
        action = policy.select_action(x)
        action_env = action + 2
        state, reward, done, _ = env.step(action_env)
        reward_sum += reward

        policy.rewards.append(reward)
        if done:
            # tracking log
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            print('Policy Gradient with Baseline ep %03d done. reward: %f. reward running mean: %f' % (
                i_episode, reward_sum, running_reward))
            reward_sum = 0
            break

    # use policy gradient update model weights
    if i_episode % args.batch_size == 0 and test == False:
        finish_episode()

    # Save model in every 50 episode
    if (i_episode % 50 == 0) and (test == False) and (not args.disable_model_loading):
        print('ep %d: model saving...' % (i_episode))
        torch.save(policy.state_dict(), 'pgb_params.pkl')
