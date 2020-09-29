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
    theta = np.random.randn(3*6*2, 1)
    wins = 0
    return theta, wins


theta, wins = init()
trueQ = pickle.load(open('Q.dill', 'rb'))

# number of times state s has been visited


# step size
def alpha(p, d, a): return 0.01
# # exploration probability
def epsilon(player, deader): return 0.05


def epsilonGreedy(player, dealer, Q):
    if np.random.random() < epsilon(player, dealer):
        # explore
        action = np.random.choice(actions)
    else:
        # explot
        action = np.argmax([Q(player, dealer, a, theta) for a in actions])
    return action


def getFeatures(p, d, a):
    features = np.zeros((3, 6, 2))
    x, y = [], []
    for i, (lower, upper) in enumerate([[1, 4], [4, 7], [7, 10]]):
        if lower <= d <= upper:
            x.append(i)
    for j, (lower, upper) in enumerate([[1, 6], [4, 9], [7, 12], [10, 15], [13, 18], [16, 21]]):
        if lower <= p <= upper:
            y.append(j)
    for i in x:
        for j in y:
            features[i, j, a] = 1
    return features.reshape(1, -1)


def Q(p, d, a, theta):
    return np.dot(getFeatures(p, d, a), theta)


allFeatures = np.zeros((22, 11, 2, 3*6*2))
for p in range(1, 22):
    for d in range(1, 11):
        for a in range(0, 2):
            allFeatures[p-1, d-1, a] = getFeatures(p, d, a)


def allQ(allFeatures, theta):
    return np.dot(allFeatures.reshape(-1, 3*6*2), theta).reshape(-1)


lambdas = [0, 1]
mseLambdas = np.zeros((len(lambdas), n_episodes))
finalMSE = np.zeros(len(lambdas))

for li, lmd in enumerate(lambdas):
    theta, wins = init()
    for episode in range(n_episodes):
        terminal = False
        E = np.zeros_like(theta)  # Eligibility Trace
        # init state
        playerSum, dealerSum = env.startGame()
        action = epsilonGreedy(playerSum, dealerSum, Q)
        while not terminal:
            # take action A, observe R, S'
            pPrime, dPrime, terminal, r = env.step(
                playerSum, dealerSum, action)
            if not terminal:
                # print(pPrime, dPrime)
                aPrime = epsilonGreedy(pPrime, dPrime, Q)
                tdError = r + Q(pPrime, dPrime, aPrime, theta) - \
                    Q(playerSum, dealerSum, action, theta)
            else:
                tdError = r - Q(playerSum, dealerSum, action, theta)
            # E[playerSum, dealerSum, action] += 1
            E = lmd * E + getFeatures(playerSum,
                                      dealerSum, action).reshape(-1, 1)
            gradient = alpha(playerSum, dealerSum, action) * tdError * E
            theta += gradient

            if not terminal:
                playerSum, dealerSum, action = pPrime, dPrime, aPrime
        # bookkeeping
        if r == 1:
            wins += 1
        mse = np.sum(np.square(allQ(allFeatures, theta) -
                               trueQ.ravel())) / (21*10*2)
        mseLambdas[li, episode] = mse
        if episode % 1000 == 0 or episode == n_episodes - 1:
            print("Lambda=%.1f Episode %06d, MSE %5.3f, Wins %.3f" %
                  (lmd, episode, mse, wins/(episode+1)))
    finalMSE[li] = mse
    print("Lambda=%.1f Episode %06d, MSE %5.3f, Wins %.3f" %
          (lmd, episode, mse, wins/(episode+1)))
    print("--------")


utils.plotMseLambdas(finalMSE, lambdas)
utils.plotMseEpisodesLambdas(mseLambdas)
