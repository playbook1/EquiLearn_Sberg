#######################
# Game environment based on Stackelberg. 
# Authors: Francisco and Dhivya
######################

import gymnasium as gym
import torch
class GameEnvironment(gym.Env):


    def __init__(self):

        self.N 
        self.total_demand 
        self.T = None
        self.quantity = torch.zero((self.N,self.T))
        self.profit = torch.zero((self.N,self.T))
        self.t = 0


    def reset(self):
        pass
    

    def step(self):
        pass 
        
        

class ModelGameEnvironment(GameEnvironment):

    def __init__(self):
        super.__init__(gamma)
        self.gamma = gamma

    def step(self, action)-> (new state, reward, done):
        #update info of the game
        # compute next state

        # compute reward
        # finished or not

    def reset(self):
        self.t = 0
        self.quantity = torch.zero((self.N,self.T))
        self.profit = torch.zero((self.N,self.T))        
        return init_state, None, False



    
