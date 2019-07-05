# SAC_TF2
A simple TF2.0 implementation of OpenAI's spinning up SAC. 

The spinning up implementation SAC is very compact and easily extensible for quick experiments on other ideas, so I brought it over to tf2.0 for some experiments I want to do. It doesn't have the full logging print outs, instead logging to tensorboard. There are two variants here:
1. SAC, which is a direct analog to OpenAI's implementation.
2. Modular, which is a slightly more modular implementation.

Todo
2. Collect train/test rollouts into a single 'collect episode' function similar to TF agent's drivers for the modular version.
