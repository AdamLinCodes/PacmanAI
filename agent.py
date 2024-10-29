import torch
import random
import numpy as np
from collections import deque
from network import Network
from utils import preprocess_frame
import torch.optim as optim
import torch.nn.functional as F

class Agent():
    def __init__(self, action_size, learning_rate=5e-4, device=None):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.action_size = action_size
        self.local_qnetwork = Network(action_size).to(self.device)
        self.target_qnetwork = Network(action_size).to(self.device)
        self.optimizer = optim.Adam(self.local_qnetwork.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=10000)

    def step(self, state, action, reward, next_state, done, discount_factor=0.99, minibatch_size=64):
        state = preprocess_frame(state)
        next_state = preprocess_frame(next_state)
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > minibatch_size:
            experiences = random.sample(self.memory, k=minibatch_size)
            self.learn(experiences, discount_factor)

    def act(self, state, epsilon=0.):
        state = preprocess_frame(state).to(self.device)
        self.local_qnetwork.eval()
        with torch.no_grad():
            action_values = self.local_qnetwork(state)
        self.local_qnetwork.train()
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, discount_factor):
        states, actions, rewards, next_states, dones = zip(*experiences)
        states = torch.stack(states).float().to(self.device)
        actions = torch.tensor(actions).long().to(self.device)
        rewards = torch.tensor(rewards).float().to(self.device)
        next_states = torch.stack(next_states).float().to(self.device)
        dones = torch.tensor(dones).float().to(self.device)
        
        next_q_targets = self.target_qnetwork(next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + discount_factor * next_q_targets * (1 - dones)
        q_expected = self.local_qnetwork(states).gather(1, actions.unsqueeze(1))
        
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
