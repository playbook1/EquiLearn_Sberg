# Demand Inertia definition
# Created by faristireina in 2022 updated in 2024



import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import sys
import numpy as np # numerical python
# printoptions: output limited to 2 digits after decimal point
np.set_printoptions(precision=2, suppress=False)



class DemandInertiaGame():
    """
        Fully defines demand Potential Game. It contains game rules, memory and agents strategies.
    """
    
    def __init__(self, N, total_demand, vector_costs, total_stages) -> None:
        self.N = N
        self.total_demand = total_demand
        self.costs = vector_costs
        self.T = total_stages
        # first index is always player
        self.demand_potential = None # two lists for the two players
        self.prices = None # prices over T rounds
        self.profit = None  # profit in each of T rounds
        self.t = None


    def resetGame(self):
        self.demand_potential = torch.zeros((self.N,self.T), dtype = torch.float32) # two lists for the two players
        self.prices = torch.zeros((self.N,self.T), dtype = torch.float32) # prices over T rounds
        self.profit = torch.zeros((self.N,self.T), dtype = torch.float32)  # profit in each of T rounds
        self.demand_potential[0][0] = self.total_demand/2 # initialize first round 0
        self.demand_potential[1][0] = self.total_demand/2


    def profits(self):
        return self.profit[:,self.t]

    def updatePricesProfitDemand(self, pricepair):
        # pricepair = list of prices for players 0,1 in current round t
        for player in range(0, self.N):
            self.profit[player][self.t] = (self.demand_potential[player][self.t] - pricepair[player])*(pricepair[player] - self.costs[player])
            if self.t < self.T-1 :
                self.demand_potential[player][ self.t + 1] = \
                    self.demand_potential[player][self.t] + (pricepair[1-player] - pricepair[player])/2

  


class ModelGameEnvironment(DemandInertiaGame):
    """
        Defines the Problem's Model. It is assumed a Markov Decision Process is defined.
    """
    def __init__(self, N, total_demand,vector_costs,total_stages) -> None:
        super().__init__(N, total_demand,vector_costs,total_stages)

        self.reward_function = self.profits
        self.init_state = torch.Tensor([57,71,200,200])
        #self.episodesMemory = list()

    def reset(self):
        self.t = 0
        self.resetGame()        
        return self.init_state, None, False



    def step(self, action_profile):

        #update info of the game for every agent
        for agent in range(0,self.N):
            self.prices[agent,self.t] = action_profile[agent]


        self.updatePricesProfitDemand(action_profile)
        new_state = torch.cat((self.prices[:,self.t], self.demand_potential[:,self.t]))

        reward = self.profit[:,self.t]
        self.t += 1

        
        return new_state, reward, (self.t == self.T)



