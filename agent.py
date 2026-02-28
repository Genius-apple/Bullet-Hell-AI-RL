import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size, seed=42):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        # 超参数
        self.batch_size = 64
        self.gamma = 0.99
        self.tau = 1e-3
        self.lr = 5e-4
        self.update_every = 4
        
        # 设备
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size).to(self.device)
        self.qnetwork_target = QNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)
        
        # Replay Buffer
        self.memory = deque(maxlen=100000)
        self.t_step = 0
        
    def step(self, state, action, reward, next_state, done):
        # 存储经验
        self.memory.append((state, action, reward, next_state, done))
        
        # 每隔几步学习一次
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.sample()
                self.learn(experiences, self.gamma)

    def act(self, state, eps=0.0):
        """返回给定状态下的动作 (Epsilon-Greedy)"""
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        
        # Epsilon-greedy selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def sample(self):
        """随机采样一批经验"""
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        
        return (states, actions, rewards, next_states, dones)

    def learn(self, experiences, gamma):
        """更新 Q 网络参数"""
        states, actions, rewards, next_states, dones = experiences

        # 获取预期的 Q 值 (Target Network)
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # 获取当前的 Q 值 (Local Network)
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # 计算 Loss
        loss = torch.nn.functional.mse_loss(Q_expected, Q_targets)
        
        # 最小化 Loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新 Target Network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def save(self, filename):
        torch.save(self.qnetwork_local.state_dict(), filename)

    def load(self, filename):
        self.qnetwork_local.load_state_dict(torch.load(filename))
        self.qnetwork_target.load_state_dict(torch.load(filename))
