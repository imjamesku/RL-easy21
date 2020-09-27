from easy21 import Easy21
import numpy as np
import dill as pickle
import utils


env = Easy21()
actions = [0, 1]
N0 = 100

n_episodes = int(10000)


def init():
    # action value function
    Q = np.zeros((22, 11, len(actions)))
    # Number of times an action has been chosen in state s
    nsa = np.zeros((22, 11, len(actions)))
    wins = 0
    return Q, nsa, wins


Q, nsa, wins = init()
trueQ = pickle.load(open('Q.dill', 'rb'))

# number of times state s has been visited


def ns(player, deader): return np.sum(nsa[player, deader])
# step size
def alpha(p, d, a): return 1/nsa[p, d, a]
# # exploration probability
def epsilon(player, deader): return N0 / (N0 + ns(player, deader))


def epsilonGreedy(player, dealer, Q):
    if np.random.random() < epsilon(player, dealer):
        # explore
        action = np.random.choice(actions)
    else:
        # explot
        action = np.argmax(Q[player, dealer])
    return action


lambdas = list(np.arange(0, 1.1, 0.1))
mseLambdas = np.zeros((len(lambdas), n_episodes))
finalMSE = np.zeros(len(lambdas))

for li, lmd in enumerate(lambdas):
    Q, nsa, wins = init()
    for episode in range(n_episodes):
        E = np.zeros((22, 11, len(actions)))
        playerSum, dealerSum = env.startGame()
        action = epsilonGreedy(playerSum, dealerSum, Q)
        sa = []  # state action path
        terminal = False
        while not terminal:
            # take action A, observe R, S'
            pPrime, dPrime, terminal, r = env.step(
                playerSum, dealerSum, action)
            if not terminal:
                # print(pPrime, dPrime)
                aPrime = epsilonGreedy(pPrime, dPrime, Q)
                tdError = r + Q[pPrime, dPrime, aPrime] - \
                    Q[playerSum, dealerSum, action]
            else:
                tdError = r - Q[playerSum, dealerSum, action]
            E[playerSum, dealerSum, action] += 1
            nsa[playerSum, dealerSum, action] += 1
            sa.append((playerSum, dealerSum, action))

            for p, d, a in sa:
                Q[p, d, a] += alpha(p, d, a) * tdError * E[p, d, a]
                E[p, d, a] *= lmd
            if not terminal:
                playerSum, dealerSum, action = pPrime, dPrime, aPrime
        # bookkeeping
        if r == 1:
            wins += 1
        mse = np.sum(np.square(Q-trueQ)) / (21*10*2)
        mseLambdas[li, episode] = mse
        if episode % 1000 == 0:
            print("Lambda=%.1f Episode %06d, MSE %5.3f, Wins %.3f" %
                  (lmd, episode, mse, wins/(episode+1)))
    finalMSE[li] = mse
    print("Lambda=%.1f Episode %06d, MSE %5.3f, Wins %.3f" %
          (lmd, episode, mse, wins/(episode+1)))
    print("--------")


utils.plotMseLambdas(finalMSE, lambdas)
utils.plotMseEpisodesLambdas(mseLambdas)
