#!/usr/bin/python3

import numpy as np # numerical python
# printoptions: output limited to 2 digits after decimal point
np.set_printoptions(precision=2, suppress=False)
import re # re: regular expressions, used in cleanbrackets 

T=25 # number of rounds
# player 0 = low cost
# player 1 = high cost
cost = [57, 71] # cost
# first index is always player
demandpotential = [[0]*T,[0]*T] # two lists for the two players
demandpotential[0][0]=200 # initialize first round 0
demandpotential[1][0]=200
prices = [[0]*T,[0]*T]  # prices over T rounds
profit = [[0]*T,[0]*T]  # profit in each of T rounds

def monopolyprice(player, t): # myopic monopoly price 
    return (demandpotential[player][t] + cost[player])/2

def updatePricesProfitDemand(pricepair, t):
    # pricepair = list of prices for players 0,1 in current round t
    for player in [0,1]:
        price = pricepair[player]
        prices[player][t] = price
        profit[player][t] = \
            (demandpotential[player][t] - price)*(price - cost[player])
        if t<T-1 :
            demandpotential[player][t+1] = \
                demandpotential[player][t] + (pricepair[1-player] - price)/2
    return

def totalprofit(): # gives pair of total profits over T periods
    return sum(profit[0]), sum(profit[1])

def avgprofit(): # gives pair of average profits per round
    return sum(profit[0])/T, sum(profit[1])/T

def match (stra0, stra1):
    # matches two strategies against each other over T rounds
    # each strategy is a function giving price in round t
    # assume demandpotentials in round 0 are untouched, rest
    # will be overwritten
    for t in range(T):
        pricepair = [ stra0(t), stra1(t) ]
        # no dumping
        pricepair[0] = max (pricepair[0], cost[0])
        pricepair[1] = max (pricepair[1], cost[1])
        updatePricesProfitDemand(pricepair, t)
    return avgprofit()

def tournament(strats0, strats1):
    # strats0,strats1 are lists of strategies for players 0,1
    # all matched against each other
    # returns resulting pair A,B of payoff matrices 
    m = len(strats0)
    n = len(strats1)
    # A = np.array([[0.0]*n]*m) # first index=row, second=col
    # B = np.array([[0.0]*n]*m)
    A = np.zeros((m,n))
    B = np.zeros((m,n))
    for i in range (m):
        for j in range (n):
            A[i][j], B[i][j] = match (strats0[i], strats1[j])
    return A,B 

def cleanbrackets(astring): 
    # formats matrix string from np.array_str(A) for lrsnash
    astring = re.sub('[\[\]]', '', astring)
    astring = re.sub('\n ', '\n', astring)
    astring = re.sub('\.', ' ', astring)
    return astring 

def outgame (A,B,divideby):
    # to stdout: A,B payoff matrices for use with lrsnash
    # divides entries by divideby (e.g. 10) to get fewer digits
    # all payoffs output as rounded integers
    # also gnuplot output in files, REQUIRES ./PLOT to exist

    m = len(A)
    n = len(A[0])
    print ("A =")
    A = A / divideby
    np.set_printoptions(precision=0)
    print (cleanbrackets(np.array_str(A)))
    print ("\nB =")
    B = B / divideby
    print (cleanbrackets(np.array_str(B)))
    # create gnuplot files in ./PLOT/
    for i in range (m):
        out = open("PLOT/"+str(i)+stratsinfo0[i],'w')
        for j in range (n):
            out.write(str(A[i][j])+" "+str(B[i][j])+"\n")
    out = open("PLOT/gplot",'w')
    out.write('set terminal postscript eps color\n')
    out.write('set output "Pareto.eps"\n')
    out.write("plot")
    for i in range (m):
        out.write(' "'+str(i)+stratsinfo0[i]+'" with lines lw 3,')
    out.write("\n")
    return

# strategies with varying parameters
def myopic(player, t): 
    return monopolyprice(player, t)    

def const(player, price, t): # constant price strategy
    if t == T-1:
        return monopolyprice(player, t)
    return price

def imit(player, firstprice, t): # price imitator strategy
    if t == 0:
        return firstprice
    if t == T-1:
        return monopolyprice(player, t)
    return prices[1-player][t-1] 

def fight(player, firstprice, t): # simplified fighting strategy
    if t == 0:
        return firstprice
    if t == T-1:
        return monopolyprice(player, t)
    aspire = [ 207, 193 ] # aspiration level for demand potential
    D = demandpotential[player][t] 
    Asp = aspire [player]
    if D >= Asp: # keep price; DANGER: price will never rise
        return prices[player][t-1] 
    # adjust to get to aspiration level using previous
    # opponent price; own price has to be reduced by twice
    # the negative amount D - Asp to get demandpotential to Asp 
    P = prices[1-player][t-1] + 2*(D - Asp) 
    # never price to high because even 125 gives good profits
    P = min(P, 125)
    return P

# sophisticated fighting strategy, compare fight()
# estimate *sales* of opponent as their target, kept between
# calls in global variable oppsaleguess[]. Assumed behavior
# of opponent is similar to this strategy itself.
oppsaleguess = [61, 75] # first guess opponent sales as in monopoly
def guess(player, firstprice, t): # predictive fighting strategy
    if t == 0:
        oppsaleguess[0] = 61 # always same start 
        oppsaleguess[1] = 75 # always same start 
        return firstprice
    if t == T-1:
        return monopolyprice(player, t)
    aspire = [ 207, 193 ] # aspiration level
    D = demandpotential[player][t] 
    Asp = aspire [player]
    if D >= Asp: # keep price, but go slightly towards monopoly if good
        pmono = monopolyprice(player, t)
        pcurrent = prices[player][t-1] 
        if pcurrent > pmono: # shouldn't happen
            return pmono
        if pcurrent > pmono-7: # no change
            return pcurrent
        # current low price at 60%, be accommodating towards "collusion"
        return .6 * pcurrent + .4 * (pmono-7)
    # guess current *opponent price* from previous sales
    prevsales = demandpotential[1-player][t-1] - prices[1-player][t-1] 
    # adjust with weight alpha from previous guess
    alpha = .5
    newsalesguess = alpha * oppsaleguess[player] + (1-alpha)*prevsales
    # update
    oppsaleguess[player] = newsalesguess 
    guessoppPrice = 400 - D - newsalesguess 
    P = guessoppPrice + 2*(D - Asp) 
    if player == 0:
        P = min(P, 125)
    if player == 1:
        P = min(P, 130)
    return P

strats0 = [ # use lambda to get function with single argument t
    lambda t : myopic(0,t)        # 0 = myopic
    , lambda t : guess(0,125,t)   # 1 = clever guess strategy
    , lambda t : const(0,125,t)
    , lambda t : const(0,117,t) 
    , lambda t : const(0,114.2,t)
    , lambda t : const(0,105,t) 
    , lambda t : const(0,100,t)   # suppressed for easier plot
    , lambda t : const(0,95,t) 
    , lambda t : imit(0,120,t)
    , lambda t : imit(0,110,t)
    , lambda t : fight(0,125,t)
    ]

# description of above strategies, please maintain *manually*
stratsinfo0 = [
    "myopic"
    , "guess125"
    , "const125"
    , "const117"
    , "const114.2"
    , "const105"
    , "const100"
    , "const95"
    , "imit120"
    , "imit110"
    , "fight125"
    ]

strats1 = [
    lambda t : myopic(1,t)        # 0 = myopic
    , lambda t : guess(1,130,t)   # 1 = clever guess strategy
    , lambda t : imit(1,131,t)    # 2 = imit starting nice
    , lambda t : imit(1,114.2,t)  # 3 = imit starting competitive
    , lambda t : fight(1,130,t)   # 4 = aggressive fight
    ]
stratsinfo1 = [
    "myopic"
    , "guess130"
    , "imit131"
    , "imit114.2"
    , "fight130"
    ]

# sample detailed output of two strategies
i = 1  # clever guess for player 0
# i = 2  # const125 for player 0
j = 1  # clever guess for player 1
print ("matching", i, stratsinfo0[i],"to", j, stratsinfo1[j])
match (strats0[i], strats1[j])
print (np.array(demandpotential))
print (np.array(prices))
print (np.array(profit))
avgprof = avgprofit()
print (np.array([avgprof[0], avgprof[1]]))
print ()

# tournament
# test of single-item list in tournament
# A,B = tournament ([ lambda t : myopic(0,t) ], [lambda t : myopic(1,t) ] )
A,B = tournament (strats0, strats1)
# outgame(A,B,10) # output divided by 10
outgame(A,B,1) # output divided by 1 (full 4 digits)
# reset to 2 decimal points
# np.set_printoptions(precision=2, suppress=False)
# print (A) # print with brackets
# print (B) # print with brackets
print (len(strats0),len(strats1)," should be",
    len(stratsinfo0),"x",len(stratsinfo1))
    # check that strats0 and stratsinfo0 match in length
print (stratsinfo0) # information about used strategies
print (stratsinfo1)


