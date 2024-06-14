from enum import Enum
import numpy as np
import globals as gl
# import torch
# from torch.distributions import Categorical
# from openpyxl import load_workbook
from fractions import Fraction
import time
import os
import sqlite3 as sql
from collections import namedtuple
from stable_baselines3 import SAC, PPO
from copy import copy


class DataBase():

    AgentRow = namedtuple(
        "AgentRow", "name, base_agent,  num_ep, cost, mixed_adv_txt, expected_payoff, payoff_treshhold, alg, lr, memory, added, action_step, seed, num_process, running_time")

    AGENTS_TABLE = "trained_agents"
    ITERS_TABLE = "agents_iters"
    AGENTS_COLS = "id integer PRIMARY  key AUTOINCREMENT,name text NOT NULL,base_agent text DEFAULT NULL,n_ep integer NOT NULL,cost integer NOT NULL,mixed_adv text NOT NULL,expected_payoff real,payoff_treshhold real,alg text NOT NULL,lr real NOT NULL,memory integer NOT NULL, action_step integer DEFAULT NULL,seed integer,num_procs integer DEFAULT 1,running_time  integer, added integer,time text"
    ITERS_COLS = "id integer PRIMARY key AUTOINCREMENT,agent_id integer NOT NULL,adv text  NOT NULL,agent_return text,adv_return text,agent_rewards text,adv_rewards text,actions text,agent_prices text,adv_prices text, agent_demands text,adv_demands text"

    def __init__(self, name="data.db") -> None:
        self.db_name = name
        self.reset()

    def reset(self):
        self.connection = sql.connect(self.db_name)
        self.cursor = self.connection.cursor()
        self.cursor.execute(
            f'CREATE TABLE IF NOT EXISTS {self.AGENTS_TABLE}({self.AGENTS_COLS});')
        self.cursor.execute(
            f'CREATE TABLE IF NOT EXISTS {self.ITERS_TABLE}({self.ITERS_COLS});')

    # def insert_new_agent(self, name, base_agent,  num_ep, cost, mixed_adv_txt, expected_payoff, payoff_treshhold, lr, memory, added, action_step=None, seed=0, num_process=1, running_time=0):
    def insert_new_agent(self, row):
        """
         adds a new agent to db and returns the id
         row: AgentRow named tuple
        """
        query = f'INSERT INTO {self.AGENTS_TABLE} VALUES (NULL,\'{row.name}\',' + ('NULL' if (row.base_agent is None) else f'\'{row.base_agent}\'') + \
            f',{row.num_ep},{row.cost},\'{row.mixed_adv_txt}\',{row.expected_payoff},{row.payoff_treshhold},\"{row.alg}\",{row.lr},{row.memory},{("NULL" if row.action_step is None else row.action_step)},{row.seed},{row.num_process},{row.running_time},{int(row.added)},\'{ time.ctime(time.time())}\')'
        # print(query)
        self.cursor.execute(query)
        self.connection.commit()
        return self.cursor.lastrowid

    def insert_new_iteration(self, agent_id, adv_txt, agent_return, adv_return, agent_rewards_txt, adv_rewards_txt, actions_txt, agent_prices_txt, adv_prices_txt,
                             agent_demands_txt, adv_demands_txt):
        """
        adds a new iteration to db and returns the id
        """
        query = f'INSERT INTO {self.ITERS_TABLE} VALUES (NULL,{agent_id},\'{adv_txt}\',{agent_return},{adv_return},\'{agent_rewards_txt}\',\'{adv_rewards_txt}\',\
            \'{actions_txt}\',\'{agent_prices_txt}\',\'{adv_prices_txt}\',\'{agent_demands_txt}\',\'{adv_demands_txt}\')'
        # print(query)
        self.cursor.execute(query)
        self.connection.commit()
        return self.cursor.lastrowid

    def get_list_of_added_strategies(self):
        """ returns two lists of low_cost and high_cost strategies """
        low_q= f"SELECT name, alg, memory, action_step FROM {self.AGENTS_TABLE} WHERE (added=1 and cost={gl.LOW_COST})"
        high_q= f"SELECT name, alg, memory, action_step FROM {self.AGENTS_TABLE} WHERE (added=1 and cost={gl.HIGH_COST})"
        low_lst=[]
        high_lst=[]
        self.cursor.execute(low_q)
        low_all=self.cursor.fetchall()
        for tup in low_all:
            if tup[1]==str(SAC):
                model=SAC
            elif tup[1]== str(PPO):
                model=PPO
            else:
                print("ERROR in loading strategies from db: model not recognised!")
                return [],[]

            low_lst.append(Strategy(strategy_type=StrategyType.sb3_model,model_or_func=model,name=tup[0],memory=tup[2],action_step=tup[3]))
            
        self.cursor.execute(high_q)
        high_all=self.cursor.fetchall()
        for tup in high_all:
            if tup[1]==str(SAC):
                model=SAC
            elif tup[1]== str(PPO):
                model=PPO
            else:
                print("ERROR in loading strategies from db: model not recognised!")
                return [],[]

            high_lst.append(Strategy(strategy_type=StrategyType.sb3_model,model_or_func=model,name=tup[0],memory=tup[2],action_step=tup[3]))
        return low_lst, high_lst
        

class Strategy():
    """
    strategies can be static or they can be models trained with sb3.
    """
    type = None
    env = None
    name = None
    memory = None
    policy = None
    model = None

    def __init__(self, strategy_type, model_or_func, name, first_price=132, memory=0, action_step=None) -> None:
        """
        model_or_func: for static strategy is the function, for sb3 is the optimizer class
        """
        self.type = strategy_type
        self.name = name
        # self._env = environment
        self.memory = memory

        self.action_step = action_step

        if strategy_type == StrategyType.sb3_model:
            self.dir = f"{gl.MODELS_DIR}/{name}"
            self.model = model_or_func
            # self.policy = self.model.predict

        else:
            self.policy = model_or_func
            self.first_price = first_price

    def reset(self):
        pass

    def play(self, env, player=1):
        """
            Computes the price to be played in the environment, nn.step_action is the step size for pricing less than myopic
        """

        if self.type == StrategyType.sb3_model:
            if self.policy is None:
                if env.memory != self.memory:
                    env_adv=(env.__class__)(tuple_costs=env.costs, adversary_mixed_strategy=env.adversary_mixed_strategy, memory=self.memory)
                    self.policy = (self.model.load(self.dir, env=env_adv)).predict
                else:
                    self.policy = (self.model.load(self.dir, env=env)).predict
            state = env.get_state(
                stage=env.stage, player=player, memory=self.memory)
            action, _ = self.policy(state)
            # compute price for co model and disc model
            price = (env.myopic(player)-action[0]) if (self.action_step is None) else (
                env.myopic(player)-(self.action_step*action))

            if player == 0:
                env.actions[env.stage] = (action[0] if (
                    self.action_step is None) else (self.action_step*action))

            return price
        else:
            return self.policy(env, player, self.first_price)

    def play_against(self, env, adversary):
        """ 
        self is player 0 and adversary is layer 1. The environment should be specified. action_step for the neural netwroks should be set.
        output: tuple (payoff of low cost, payoff of high cost)
        """
        # self.env = env
        env.adversary_mixed_strategy = adversary.to_mixed_strategy()
        
        state, _ = env.reset()
        while env.stage < (env.T):
            prices = [0, 0]
            prices[0], prices[1] = self.play(env, 0), adversary.play(env, 1)
            env.update_game_variables(prices)
            env.stage += 1

        return [sum(env.profit[0]), sum(env.profit[1])]

    def to_mixed_strategy(self):
        """
        Returns a MixedStrategy, Pr(self)=1
        """
        mix = MixedStrategy(probablities_lst=[1],
                            strategies_lst=[self])

        return mix


class MixedStrategy():
    strategies = []
    strategy_probs = None

    def __init__(self, strategies_lst, probablities_lst) -> None:
        self.strategies = strategies_lst
        self.strategy_probs = probablities_lst
        self.support_size = support_count(probablities_lst)

    def choose_strategy(self):
        if len(self.strategies) > 0:
            # adversaryDist = Categorical(torch.tensor(self._strategyProbs))
            # if not torch.is_tensor(self._strategyProbs):
            #     self._strategyProbs = torch.tensor(self._strategyProbs)
            # adversaryDist = Categorical(self._strategyProbs)
            # strategyInd = (adversaryDist.sample()).item()
            strategy_ind = np.random.choice(
                len(self.strategies), size=1, p=self.strategy_probs)
            return self.strategies[strategy_ind[0]]
        else:
            print("adversary's strategy can not be set!")
            return None

    def play_against(self, env, adversary):
        pass

    def __str__(self) -> str:
        s = ""
        for i in range(len(self.strategies)):
            if self.strategy_probs[i] > 0:
                s += f"{self.strategies[i].name}-{self.strategy_probs[i]:.2f},"
        return s


class StrategyType(Enum):
    static = 0
    neural_net = 1
    sb3_model = 2


def myopic(env, player, firstprice=0):
    """
        Adversary follows Myopic strategy
    """
    return env.myopic(player)


def const(env, player, firstprice):  # constant price strategy
    """
        Adversary follows Constant strategy
    """
    if env.stage == env.T-1:
        return env.myopic(player)
    return firstprice


def imit(env, player, firstprice):  # price imitator strategy
    if env.stage == 0:
        return firstprice
    if env.stage == env.T-1:
        return env.myopic(player)
    return env.prices[1-player][env.stage-1]


def fight(env, player, firstprice):  # simplified fighting strategy
    if env.stage == 0:
        return firstprice
    if env.stage == env.T-1:
        return env.myopic(player)
    # aspire = [ 207, 193 ] # aspiration level for demand potential
    aspire = [0, 0]
    for i in range(2):
        aspire[i] = (env.total_demand-env.costs[player] +
                     env.costs[1-player])/2

    D = env.demand_potential[player][env.stage]
    Asp = aspire[player]
    if D >= Asp:  # keep price; DANGER: price will never rise
        return env.prices[player][env.stage-1]
    # adjust to get to aspiration level using previous
    # opponent price; own price has to be reduced by twice
    # the negative amount D - Asp to getenv.demandPotential to Asp
    P = env.prices[1-player][env.stage-1] + 2*(D - Asp)
    # never price to high because even 125 gives good profits
    # P = min(P, 125)
    aspire_price = (env.total_demand+env.costs[0]+env.costs[1])/4
    P = min(P, int(0.95*aspire_price))

    return P


def fight_lb(env, player, firstprice):
    P = env.fight(player, firstprice)
    # never price less than production cost
    P = max(P, env.costs[player])
    return P

# sophisticated fighting strategy, compare fight()
# estimate *sales* of opponent as their target


def guess(env, player, firstprice):  # predictive fighting strategy
    if env.stage == 0:
        env.aspireDemand = [(env.total_demand/2 + env.costs[1]-env.costs[0]),
                            (env.total_demand/2 + env.costs[0]-env.costs[1])]  # aspiration level
        env.aspirePrice = (env.total_demand+env.costs[0]+env.costs[1])/4
        # first guess opponent sales as in monopoly ( sale= demand-price)
        env.saleGuess = [env.aspireDemand[0]-env.aspirePrice,
                         env.aspireDemand[1]-env.aspirePrice]

        return firstprice

    if env.stage == env.T-1:
        return env.myopic(player)

    D = env.demand_potential[player][env.stage]
    Asp = env.aspireDemand[player]

    if D >= Asp:  # keep price, but go slightly towards monopoly if good
        pmono = env.myopic(player)
        pcurrent = env.prices[player][env.stage-1]
        if pcurrent > pmono:  # shouldn't happen
            return pmono
        elif pcurrent > pmono-7:  # no change
            return pcurrent
        # current low price at 60%, be accommodating towards "collusion"
        return .6 * pcurrent + .4 * (pmono-7)

    # guess current *opponent price* from previous sales
    prevsales = env.demand_potential[1 -
                                     player][env.stage-1] - env.prices[1-player][env.stage-1]
    # adjust with weight alpha from previous guess
    alpha = .5
    newsalesguess = alpha * env.saleGuess[player] + (1-alpha)*prevsales
    # update
    env.saleGuess[player] = newsalesguess
    guessoppPrice = env.total_demand - D - newsalesguess
    P = guessoppPrice + 2*(D - Asp)

    if player == 0:
        P = min(P, 125)
    if player == 1:
        P = min(P, 130)
    return P


def spe(env, player, firstprice=0):
    """
    returns the subgame perfect equilibrium price
    """
    t = env.stage
    P = gl.SPE_a[t]*(env.demand_potential[player][t]-200) + gl.SPE_b[t] + gl.SPE_k[t]*(env.costs[player]-64)
    return P


def monopolyPrice(demand, cost):  # myopic monopoly price
    """
        Computes Monopoly prices.
    """
    return (demand + cost) / 2
    # return (self.demandPotential[player][self.stage] + self.costs[player])/2


def prt(string):
    """
    writing the progres into a file instead of print
    """
    global job_name
    with open(f'progress_{job_name}.txt', 'a') as file:
        file.write("\n"+string)


# def write_to_excel(file_name, new_row):
#     """
#     row includes:  name	ep	costs	adversary	agent_return	adv_return	agent_rewards	actions	agent_prices	adv_prices	agent_demands	adv_demands	lr	hist	total_stages	action_step	num_actions	gamma	stae_onehot	seed	num_procs	running_time
#     """

#     path = f'results_{job_name}.xlsx' if (file_name is None) else file_name

#     wb = load_workbook(path)
#     sheet = wb.active
#     row = 2
#     col = 1
#     sheet.insert_rows(idx=row)

#     for i in range(len(new_row)):
#         sheet.cell(row=row, column=col+i).value = new_row[i]
#     wb.save(path)


# def write_results(new_row):
#     write_to_excel(f'results_{job_name}.xlsx', new_row)


# def write_agents(new_row):
#     # name	ep	costs	adversary	expected_payoff	payoff_treshhold	lr	hist	total_stages	action_step	num_actions\
#     # gamma	seed	num_procs	running_time	date

#     write_to_excel(f'trained_agents_{job_name}.xlsx', new_row)


def support_count(list):
    """
    gets a list and returns the number of elements that are greater than zero
    """
    counter = 0
    for item in list:
        if item > 0:
            counter += 1
    return counter


def recover_probs(test):
    low_cost_probs, high_cost_probs, rest = test.split(")")
    low_cost_probs = low_cost_probs.split("(")[1]
    _, high_cost_probs = high_cost_probs.split("(")
    high_cost_probs = [float(Fraction(s)) for s in high_cost_probs.split(',')]
    low_cost_probs = [float(Fraction(s)) for s in low_cost_probs.split(',')]
    _, low_cost_support, high_cost_support = rest.split('[')
    high_cost_support, _ = high_cost_support.split(']')
    high_cost_support = [int(s) for s in high_cost_support.split(',')]
    low_cost_support, _ = low_cost_support.split(']')
    low_cost_support = [int(s) for s in low_cost_support.split(',')]
    return low_cost_probs, high_cost_probs, low_cost_support, high_cost_support


def return_distribution(number_players, cost_probs, cost_support):
    player_probabilities = [0] * number_players
    for index, support in enumerate(cost_support):
        player_probabilities[support] = cost_probs[support]
    return player_probabilities


def create_directories():
    if not os.path.exists(gl.MODELS_DIR):
        os.makedirs(gl.MODELS_DIR)
    if not os.path.exists(gl.LOG_DIR):
        os.makedirs(gl.LOG_DIR)
    if not os.path.exists(gl.GAMES_DIR):
        os.makedirs(gl.GAMES_DIR)


def set_job_name(name):
    global job_name
    job_name = name