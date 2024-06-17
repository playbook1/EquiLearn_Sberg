import numpy as np
from enum import Enum
import config as cf


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
            self.dir = f"{cf.MODELS_DIR}/{name}"
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


def support_count(list):
    """
    gets a list and returns the number of elements that are greater than zero
    """
    counter = 0
    for item in list:
        if item > 0:
            counter += 1
    return counter

class StrategyType(Enum):
    static = 0
    neural_net = 1
    sb3_model = 2