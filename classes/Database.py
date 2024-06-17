import config as cf
from collections import namedtuple
import sqlite3 as sql
import time
import sqlite3 as sql
from stable_baselines3 import SAC, PPO
from Strategy import Strategy, StrategyType


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
        low_q= f"SELECT name, alg, memory, action_step FROM {self.AGENTS_TABLE} WHERE (added=1 and cost={cf.LOW_COST})"
        high_q= f"SELECT name, alg, memory, action_step FROM {self.AGENTS_TABLE} WHERE (added=1 and cost={cf.HIGH_COST})"
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
        
