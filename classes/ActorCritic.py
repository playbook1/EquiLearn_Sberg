import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal

class Actor(nn.Module):
    def __init__(self, input_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean, std = self.fc3(x)

        return mean, std

class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class ActorCritic:
    def __init__(self, actor, critic, discount, actor_lr=1e-3, critic_lr=1e-3):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.discount = discount
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
    
    def select_action(self, state):
        state = state.to(self.device)
        mean, f = self.actor(state)
        std = torch.exp(f)
  
        dist = Normal(mean,std)
        action = dist.sample()
        return action, dist.log_prob(action)
    
    def update(self, state, reward, log_prob, next_state):
        state = state.to(self.device)
        next_state = next_state.to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
        
        # Critic update
        value = self.critic(state)
        next_value = self.critic(next_state)
        target = reward + self.discount * next_value

        critic_loss = (target - value) ** 2
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        

        advantage = (target - value).detach()
        actor_loss = -log_prob * advantage
        print(actor_loss)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()