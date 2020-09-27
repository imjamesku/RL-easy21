from easy21 import Easy21
import numpy as np
import dill as pickle
import utils
env = Easy21()
actions = [0, 1]
printEvery = 100000
N0 = 100
# action value function
Q = np.zeros((22, 11, len(actions)))
# Number of times an action has been chosen in state s
nsa = np.zeros((22, 11, len(actions)))
def ns(player, deader): return np.sum(nsa[player, deader])
# step size
def alpha(p, d, a): return 1/nsa[p, d, a]
def epsilon(player, deader): return N0 / (N0 + ns(player, deader))


n_episodes = int(1000000)


def epsilonGreedy(player, dealer):
    if np.random.random() < epsilon(player, dealer):
        # explore
        action = np.random.choice(actions)
    else:
        # explot
        action = np.argmax(Q[player, dealer])
    return action


meanReturn = 0
wins = 0
for i in range(n_episodes):
    sarPath = []  # state, action, reward
    playerSum, dealerSum = env.startGame()
    print(playerSum, dealerSum)
    terminal = False
    while not terminal:
        # take an action based on Q
        action = epsilonGreedy(playerSum, dealerSum)
        nsa[playerSum, dealerSum, action] += 1
        prevPlayerSum, prevDealerSum = playerSum, dealerSum
        playerSum, dealerSum, terminal, reward = env.step(
            playerSum, dealerSum, action)
        sarPath.append((prevPlayerSum, prevDealerSum, action, reward))
    # update Q
    totalReturn = sum(r for _, _, _, r in sarPath)  # sum all rewards
    print(sarPath)
    for p, d, a, _ in sarPath:
        Q[p, d, a] += alpha(p, d, a) * (totalReturn - Q[p, d, a])
    # bookkeeping
    meanReturn = meanReturn + 1/(i+1) * (totalReturn - meanReturn)
    if reward == 1:
        wins += 1
    if i % printEvery == 0:
        print("Episode %i, Mean-Return %.3f, Wins %.2f" %
              (i, meanReturn, wins/(i+1)))

pickle.dump(Q, open('Q.dill', 'wb'))
_ = pickle.load(open('Q.dill', 'rb'))  # sanity check

utils.plot(Q, [0, 1])
