# In this file we will do HER, but with a curriculum as our start states.
# How best to implement? Every second run - init somewhere along your demo states?
# in that case each of our RL envs will needs a function, init with observation vector, init with goal.
# first thing to attack is 'set init function, making that general'.