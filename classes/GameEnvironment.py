#######################
# Game Meta Game
# Authors: Francisco and Dhivya
######################

import gymnasium as gym
import torch

class GameEnvironment(gym.Env):


    def __init__(self, alpha, N, T):
        self.N: int = N
        self.T: int = T
        self.price = torch.zeros((self.N,self.T), dtype = torch.float32)
        self.profit = torch.zeros((self.N,self.T), dtype = torch.float32)
        self.alpha = alpha
        self.t = 0


    def reset(self):
        pass
    

    def step(self):
        pass 
        
        

class ModelGameEnvironment(GameEnvironment):

    def __init__(self, alpha, N, T, demand = 100):
        super().__init__(alpha, N, T)
        self.init_state = torch.Tensor([0,0])
        self.demand = demand
        self.alpha = alpha

    def step(self, action_profile):

        #update info of the game for every agent
        for agent in range(0,self.N):
            self.price[agent,self.t] = action_profile[agent]

        for agent in range(0,self.N):
            self.profit[agent,self.t] = action_profile[agent]*(self.demand + self.alpha*action_profile[-1*(self.N -(agent+1) )] - action_profile[agent])
        
        # compute next state, reaward profile, 
        new_state = self.price[:,self.t]
        reward = self.profit[:,self.t]
        self.t += 1

        return new_state, reward, (self.t == self.T)



    def reset(self):
        self.t = 0
        self.price = torch.zeros((self.N,self.T), dtype = torch.float32)
        self.profit = torch.zeros((self.N,self.T), dtype = torch.float32)        
        return self.init_state, None, False



