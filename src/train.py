from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

"""
ReplayBuffer
"""
class ReplayBuffer:

    def __init__(self, capacity):
        self.capacity = capacity # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled

    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[int(self.index)] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)), list(zip(*batch))))

    def __len__(self):
        return len(self.data)

"""
DQN and initialization of hyperparameters
"""
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()

        self.layer1 = nn.Linear(n_observations, 200)
        self.layer2 = nn.Linear(200, 200)
        self.layer3 = nn.Linear(200, 200)
        self.layer4 = nn.Linear(200, 200)
        self.layer5 = nn.Linear(200, 200)
        self.layer6 = nn.Linear(200, 200)
        self.layer7 = nn.Linear(200, 200)
        self.layer8 = nn.Linear(200, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = F.relu(self.layer5(x))
        x = F.relu(self.layer6(x))
        x = F.relu(self.layer7(x))

        return self.layer8(x)


# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor 
# EPS_MAX is the maximum value of epsilon
# EPS_END is the minimum value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# EPS_DELAY is the delay on which we keep the same value of epsilon
# LR is the learning rate of the ``Adam`` optimizer
BUFFER_SIZE = 1000000
BATCH_SIZE = 1000
GAMMA = 0.95
EPS_MAX = 1.
EPS_MIN = 0.01
EPS_DECAY = 20000
EPS_DELAY = 2000
LR = 0.0001
TAU = 0.005
NB_GRADIENT_STEPS = 10


n_actions = env.action_space.n
state, _ = env.reset()
n_observations = len(state)
    
"""
ProjectAgent 
"""
class ProjectAgent:

    def __init__(self):
        self.batch_size = BATCH_SIZE
        self.gamma = GAMMA
        self.eps_max = EPS_MAX
        self.eps_min = EPS_MIN
        self.eps_decay = EPS_DECAY
        self.eps_delay = EPS_DELAY
        self.lr = LR
        self.tau = TAU
        self.nb_gradient_steps = NB_GRADIENT_STEPS

        self.memory = ReplayBuffer(BUFFER_SIZE)
        self.model = DQN(n_observations, n_actions)
        self.target_model = DQN(n_observations, n_actions)
        self.target_model.load_state_dict(self.model.state_dict())
        self.criterion = nn.SmoothL1Loss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def act(self, observation, use_random=False):
        with torch.no_grad():
            Q = self.model(torch.Tensor(observation).unsqueeze(0))
            return torch.argmax(Q).item()

    def train(self, env, max_episode):
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.eps_max
        step = 0

        while episode < max_episode:
            # update epsilon
            if step > self.eps_delay:
                epsilon = max(self.eps_min, epsilon - ((self.eps_max-self.eps_min)/self.eps_decay))

            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = self.act(state)

            # step
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward

            # train
            for _ in range(self.nb_gradient_steps): 

                if len(self.memory) > self.batch_size:
                    X, A, R, Y, D = self.memory.sample(self.batch_size)
                    QYmax = self.target_model(Y).max(1)[0].detach()
                    update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
                    QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
                    loss = self.criterion(QXA, update.unsqueeze(1))
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            # update target network if needed (using exponential moving average)
            target_state_dict = self.target_model.state_dict()
            model_state_dict = self.model.state_dict()
            tau = self.tau
            
            for key in model_state_dict:
                target_state_dict[key] = tau*model_state_dict[key] + (1-tau)*target_state_dict[key]
                
            self.target_model.load_state_dict(target_state_dict)

            # next transition
            step += 1
            if done or trunc:
                episode += 1
                print("Episode ", '{:3d}'.format(episode), 
                      ", epsilon ", '{:6.2f}'.format(epsilon), 
                      ", batch size ", '{:5d}'.format(len(self.memory)), 
                      ", episode return ", '{:4.1f}'.format(episode_cum_reward),
                      sep='')
                state, _ = env.reset()
                episode_return.append(episode_cum_reward)
                episode_cum_reward = 0
            else:
                state = next_state

        return episode_return

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self):
        self.model.load_state_dict(torch.load('src/model.pth'))


"""
Training
"""
agent = ProjectAgent()
if False:
    episode_return = agent.train(env, 500)
    print("Best episode return: ",max(episode_return))
    agent.save('src/model.pth')
else:
    print("Pas de train")
