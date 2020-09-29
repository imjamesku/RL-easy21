# RL Course by David Silver - Easy21 assignment

## Monte-Carlo Control in Easy21

Algorithm:

![](https://github.com/imjamesku/RL-easy21/blob/master/figs/MC.png?raw=true)

1 million episodes of the game has been evaluated to obtain the following value function:
![](https://github.com/imjamesku/RL-easy21/blob/master/figs/MC_1e6.png?raw=true)

## TD Learning

![](https://github.com/imjamesku/RL-easy21/blob/master/figs/Sarsa_lambda.png?raw=true)

MSE of the state-action function from Monte-Carlo with different lambdas. For each lambda, 10000 episodes have been evaluated.

![](https://github.com/imjamesku/RL-easy21/blob/master/figs/MSE_lambda.png?raw=true)

MSE iteration with different lambdas.

![](https://github.com/imjamesku/RL-easy21/blob/master/figs/MSE_episode.png?raw=true)

## Linear Function Approximation

MSE per lambda
![](https://github.com/imjamesku/RL-easy21/blob/master/figs/lfa_MSE_per_lambda.png?raw=true)

MSE evolution with lambda=0 and 1
![](https://github.com/imjamesku/RL-easy21/blob/master/figs/lfa_MSE_evolution.png?raw=true)
