# SAC_TF2
A simple Tensorflow 2.0 implementation of OpenAI's spinning up Soft Actor Critic. 

The spinning up implementation SAC is very compact and easily extensible for quick experiments on other ideas, so I brought it over to tf2.0 for some experiments I want to do. It doesn't have the full logging print outs, instead it has some printing and logging to tensorboard. There are two variants here:
1. SAC, which is a direct analog to OpenAI's implementation.
2. Modular, which is a slightly more modular implementation.

TODO
2. Collect train/test rollouts into a single 'collect episode' function similar to TF agent's drivers for the modular version.

With the same inputs and weight initializations I've checked that up to 1000 gradient steps result in identical end weights to 6 decimal places, and this is confirmed by effectively identical performance on Cartpole and Reacher2D. 

Also see this link for a colab version of the modular version. https://colab.research.google.com/drive/1QwIThAaK5F-DtV5o36XXP2-_rxWWsv8S

## Cartpole-V0 Return vs Environment Steps
![Cartpole-V0 Return vs Environment Steps](https://github.com/sholtodouglas/SAC_TF2/blob/master/images/SAC%20performance.png)
