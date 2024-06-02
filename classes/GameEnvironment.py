#######################
# Game Meta Game
# Authors: Francisco and Dhivya
######################

import gymnasium as gym
import torch

class GameEnvironment(gym.Env):


    def __init__(self, alpha, N, T):

        self.N = N
        self.T = T
        self.quantity = torch.zeros((self.N,self.T))
        self.profit = torch.zeros((self.N,self.T))
        self.alpha = alpha
        self.t = 0


    def reset(self):
        pass
    

    def step(self):
        pass 
        
        

class ModelGameEnvironment(GameEnvironment):

    def __init__(self, alpha, N, gamma, T):
        super().__init__(alpha, N, T)
        self.gamma = gamma

    def step(self, action_profile):

        #update info of the game for every agent
        for agent in range(0,self.N):
            self.quantity[agent,self.t] = action_profile[agent]

        total_demand = action_profile.sum()
        for agent in range(0,self.N):
            self.profit[agent,self.t] = action_profile[agent]*(self.alpha-total_demand)
        
        # compute next state, reaward profile, 
        new_state = self.quantity[:,self.t]
        reward = self.profit[:,self.t]
        self.t += 1

        return new_state, reward, (self.t == self.T)



    def reset(self):
        self.t = 0
        self.quantity = torch.zeros((self.N,self.T))
        self.profit = torch.zeros((self.N,self.T))        
        return init_state, None, False



