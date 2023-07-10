import BimatrixGame
import globals as gl
import environmentModelBase as em
from environmentModelBase import Model
from learningBase import ReinforceAlgorithm, MixedStrategy, Strategy, StrategyType
from neuralNetworkSimple import NNBase
import numpy as np

def create_nn_strategy(name):
    nn=NNBase(num_input=gl.total_stages+2+gl.num_adv_history,lr=gl.lr, num_actions=gl.num_actions)
    nn.load(name)
    return Strategy(StrategyType.neural_net,NNorFunc=nn, name=name)

# const132=Strategy(StrategyType.static,name="const132",staticIndex=1)
# const95=Strategy(StrategyType.static,"const95",staticIndex=2)  

# mainGame._strategies.append(const132)
# mainGame._strategies.append(const95)
# mainGame._strategies.append(myopic)
# mainGame.fill_matrix()
# mainGame.write_all_matrix()

# nnMyopic=NNBase(num_input=27, num_actions=50, adv_hist=0)
# nnMyopic.reset()
# nnMyopic.load("0,[1e-05,1][1, 10000, 1, 1],1682423487")
# nn1st=Strategy(StrategyType.neural_net,nnMyopic,"nnMyopic" )
# mainGame._strategies.append(nn1st)


# nnConst=NNBase(num_input=27, num_actions=50, adv_hist=0)
# nnConst.reset()
# nnConst.load("0,[1e-05,1][1, 10000, 1, 1],1682506150")
# mainGame._strategies.append(Strategy(StrategyType.neural_net,nnConst,"nnConst132" ))

# mainGame._strategies.append(Strategy(StrategyType.static,NNorFunc=em.const,name="staticConst132",firstPrice=132))
# mainGame._strategies.append(Strategy(StrategyType.static,NNorFunc=em.myopic,name="staticMyopic"))


# mainGame._strategies.append(Strategy(StrategyType.static,NNorFunc=em.guess,name="staticGuess132",firstPrice=132))
# mainGame.fill_matrix()

# mainGame.write_all_matrix()
if __name__ == '__main__':
    np.random.seed(0)
    gl.initialize()
    low_nn_names=["low,1684386202","low,1684484716","low,1684557152","low,1684856358","low,1685028503"]
    low_strategies =[Strategy(StrategyType.static, NNorFunc=em.myopic, name="myopic"), 
                    Strategy(StrategyType.static, NNorFunc=em.const, name="const", firstPrice=132), 
                    Strategy(StrategyType.static, NNorFunc=em.guess, name="guess", firstPrice=132)]+ [create_nn_strategy(name) for name in low_nn_names]
    high_nn_names=["high,1684424924","high,1684821735"]
    high_strategies =[Strategy(StrategyType.static, NNorFunc=em.myopic, name="myopic"), 
                    Strategy(StrategyType.static, NNorFunc=em.const, name="const", firstPrice=132), 
                    Strategy(StrategyType.static, NNorFunc=em.guess, name="guess", firstPrice=132)]+ [create_nn_strategy(name) for name in high_nn_names]

    # low_strategies=[Strategy(StrategyType.static,NNorFunc=em.guess,name="staticGuess132",firstPrice=132)]
    # high_strategies=[Strategy(StrategyType.static,NNorFunc=em.guess,name="staticGuess132",firstPrice=132)]

    bimatrix_game = BimatrixGame.BimatrixGame(low_strategies, high_strategies)
    equilibria=BimatrixGame.run_tournament(bimatrix_game=bimatrix_game,number_rounds= 4)
    print(equilibria)