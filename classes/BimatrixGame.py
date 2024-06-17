import sys
import numpy as np
sys.path.append("../Utils/")
import config as cf
import Bimatrix as Bimatrix
import time
from fractions import Fraction


class BimatrixGame():
    """
    strategies play against each other and fill the matrix of payoff, then the equilibria would be computed using Lemke algorithm
    """

    def __init__(self, low_cost_strategies, high_cost_strategies, env_class) -> None:
        # globals.initialize()
        self.low_strategies = low_cost_strategies
        self.high_strategies = high_cost_strategies
        self.env_class = env_class

    def reset_matrix(self):
        self.matrix_A = np.zeros(
            (len(self.low_strategies), len(self.high_strategies)))
        self.matrix_B = np.zeros(
            (len(self.low_strategies), len(self.high_strategies)))

    def fill_matrix(self):
        self.reset_matrix()

        for low in range(len(self.low_strategies)):
            for high in range(len(self.high_strategies)):
                self.update_matrix_entry(low, high)

    def update_matrix_entry(self, low_index, high_index):
        strt_L = self.low_strategies[low_index]
        strt_H = self.high_strategies[high_index]
        strt_L.reset()
        strt_H.reset()

        env = self.env_class(tuple_costs=(
            cf.LOW_COST, cf.HIGH_COST), adversary_mixed_strategy=strt_H.to_mixed_strategy(), memory=strt_L.memory)
        payoffs = [strt_L.play_against(env, strt_H)
                   for _ in range(cf.NUM_MATRIX_ITER)]

        mean_payoffs = (np.mean(np.array(payoffs), axis=0))

        self.matrix_A[low_index][high_index], self.matrix_B[low_index][high_index] = mean_payoffs[0], mean_payoffs[1]

    def write_all_matrix(self):
        # print("A: \n", self._matrix_A)
        # print("B: \n", self._matrix_B)

        output = f"{len(self.matrix_A)} {len(self.matrix_A[0])}\n\n"

        for matrix in [self.matrix_A, self.matrix_B]:
            for i in range(len(self.matrix_A)):
                for j in range(len(self.matrix_A[0])):
                    output += f"{matrix[i][j]:7.0f} "
                output += "\n"
            output += "\n"

        with open(f"game_{job_name}.txt", "w") as out:
            out.write(output)

        output += "\nlow-cost strategies: \n"
        for strt in self.low_strategies:
            output += f" {strt.name} "
        output += "\nhigh-cost strategies: \n"
        for strt in self.high_strategies:
            output += f" {strt.name} "

        with open(f"games/game{int(time.time())}.txt", "w") as out:
            out.write(output)

    def add_low_cost_row(self, row_A, row_B):
        self.matrix_A = np.append(self.matrix_A, [row_A], axis=0)
        self.matrix_B = np.append(self.matrix_B, [row_B], axis=0)

    def add_high_cost_col(self, colA, colB):
        self.matrix_A = np.hstack((self.matrix_A, np.atleast_2d(colA).T))
        self.matrix_B = np.hstack((self.matrix_B, np.atleast_2d(colB).T))
        # for j in range(len(self._matrix_A)):
        #     self._matrix_A[j].append(colA[j])
        #     self._matrix_B[j].append(colB[j])

    
    def compute_equilibria(self):
        self.write_all_matrix()
        game = Bimatrix.Bimatrix(f"game_{job_name}.txt")
        equilibria_traces = game.tracing(100, cf.NUM_TRACE_EQUILIBRIA)
        equilibria = []
        for equilibrium in equilibria_traces:
            low_cost_probs, high_cost_probs, low_cost_support, high_cost_support = recover_probs(
                equilibrium)
            low_cost_probabilities = return_distribution(
                len(self.low_strategies), low_cost_probs, low_cost_support)
            high_cost_probabilities = return_distribution(
                len(self.high_strategies), high_cost_probs, high_cost_support)
            low_cost_payoff = np.matmul(low_cost_probabilities, np.matmul(
                self.matrix_A, np.transpose(high_cost_probabilities)))
            high_cost_payoff = np.matmul(low_cost_probabilities, np.matmul(
                self.matrix_B, np.transpose(high_cost_probabilities)))

            result = {"low_cost_probs": low_cost_probabilities,
                      "high_cost_probs": high_cost_probabilities,
                      "low_cost_payoff": low_cost_payoff,
                      "high_cost_payoff": high_cost_payoff}
            equilibria.append(result)
        return equilibria


def set_job_name(name):
    global job_name
    job_name = name

def return_distribution(number_players, cost_probs, cost_support):
        player_probabilities = [0] * number_players
        for index, support in enumerate(cost_support):
            player_probabilities[support] = cost_probs[support]
        return player_probabilities

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