import numpy as np
import config as cf
# import torch
# from torch.distributions import Categorical
# from openpyxl import load_workbook
import os


# DAK's comment : sorted all classes into separate files. The functions below - not sure where they're called. 
#Leaving them here for now, will sort them out and delete this file at a later stage.

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
    P = cf.SPE_a[t]*(env.demand_potential[player][t]-200) + cf.SPE_b[t] + cf.SPE_k[t]*(env.costs[player]-64)
    return P


def monopolyPrice(demand, cost):  # myopic monopoly price
    """
        Computes Monopoly prices.
    """
    return (demand + cost) / 2
    # return (self.demandPotential[player][self.stage] + self.costs[player])/2


def create_directories():
    if not os.path.exists(cf.MODELS_DIR):
        os.makedirs(cf.MODELS_DIR)
    if not os.path.exists(cf.LOG_DIR):
        os.makedirs(cf.LOG_DIR)
    if not os.path.exists(cf.GAMES_DIR):
        os.makedirs(cf.GAMES_DIR)

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
