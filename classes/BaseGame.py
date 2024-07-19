
import gymnasium as gym
import numpy as np 
import config as cf
from gymnasium import spaces
from numpy.random import choice


class MemorylessPricingGame(gym.env):
    # The base game has 2 players playing a pricing game
    # U1 = x(100+ay-x)
    # U2 = y(100+ax-y)
    # Best response equilibrium : x = y = 100/(2-a)
    # Total utility = x^2 + y^2
    # Collusive strategy equilibrium : x = y = 50/(1-a)
    # Higher total utility, higher individual utility

    # An episode - repeated base game - say m rounds
    # there's a discount factor - a probability 'd' with which each of the m rounds might be the end of an episode.
    # State of the base game : Accumulated utility, who's the opponent?, how many stages so far in the episode?, is this the end? 
    # Action: best strategy against the opponent 
    # Step: update utility based on action, move to next stage/ the end. 

    
    def __init__(self,tuple_costs, adversary_mixed_strategy, memory):
        super().__init__()
        # gl.initialize()
        
        self.action_step=None

        self.total_demand = cf.TOTAL_DEMAND
        self.alpha = cf.ALPHA
        #self.T = 0 total number of stages 
        self.prices = None # prices over rounds
        self.total_utility = None  # profit in each round
        self.stage = None
        self.done = False
        
        self.adversary_mixed_strategy = adversary_mixed_strategy
        #memory of both players
        self.memory=memory

        self.action_space = spaces.Box(low=0, high=cf.CON_ACTIONS_RANGE, shape=(1,))
        
        # State space
        self.observation_space = spaces.Box(
            low=0, high=self.total_demand, shape=(2+2*memory,))


    def reset(self, seed = None, options = None):
        super().reset(seed=seed)

        self.resetGame()
        self.adversary_strategy = self.adversary_mixed_strategy.choose_strategy()
        # [stage, agent_ demand, agent_last_price, adversary_price_history]
        observation = self.get_state(stage=0)
        return observation, {}# reward, done, info can't be included

 

    def resetGame(self):
        """
        Method resets game memory: Demand Potential, prices, profits
        """
        self.episodesMemory = list()
        self.stage = 0
        self.done = False
        self.prices = [[0]*self.T, [0]*self.T]  # prices over T rounds
        self.total_utility = [[0]*self.T, [0]*self.T]  # utility in each of T rounds
        # initialize first round 0
        self.actions=[0]*self.T
    
    def get_state(self, stage, player=0, memory=None):
        # [stage, our demand, our price memory, adv price memory]

        mem_len = memory if (
            memory is not None) else self.memory
        

        stage_part = [stage]
        self_mem=[]
        adv_mem=[]
        
        if stage == 0:
            if (mem_len > 0):
                adv_mem = [0]*mem_len
                self_mem = [0]*mem_len
            observation = stage_part+[self.demand_potential[player][self.stage]] + self_mem+ adv_mem
        else:
            if (mem_len > 0):
                adv_mem = [0]*mem_len
                self_mem = [0]*mem_len
                j = mem_len-1
                for i in range(stage-1, max(-1, stage-1-mem_len), -1):
                    adv_mem[j] = self.prices[1-player][i]
                    self_mem[j] = self.prices[player][i]
                    j -= 1

            observation = stage_part+ [self.demand_potential[player][self.stage]]+ self_mem+ adv_mem

        return np.array(observation)
        
        
    

    def step(self,action):
        
        self.actions[self.stage]=action[0]
        adversary_action  = self.adversary_strategy.play(
            env=self, player=1)
        self.update_game_variables( [self.myopic() - action[0], adversary_action] ) 

        done = choice([True, False], 1, [1-cf.DELTA, cf.DELTA])

        reward = self.total_utility[0][self.stage]
        self.stage += 1

        info = {}

        return self.get_state(stage=self.stage), reward, done,False, info



    def update_game_variables(self, price_pair):
        """
        Updates Prices, Profit and Demand Potential Memory.
        Parameters. 
        price_pair: Pair of prices from the learning agent and adversary.
        """

        for player in [0,1]:
            price = price_pair[player]
         
            self.prices[player][self.stage] = price
            self.total_utility[player][self.stage] = self.total_utility[player][self.stage] + self.getUtilityGained(price_pair, player)


    def getUtilityGained(price_pair, player):
        if player == 0:
            opponent = 1
        else:
            opponent = 0
        return price_pair[player] * (cf.TOTAL_DEMAND + cf.ALPHA * price_pair[opponent] - price_pair[player])
    

    # def myopic(self, player = 0): 
    #     """
    #         Adversary follows Myopic strategy
    #     """
    #     return (self.demand_potential[player][self.stage]+self.costs[player])/2
    #     # return self.monopoly_price(player)    

       
    
    def render(self):
        pass

    def close(self):
        pass

