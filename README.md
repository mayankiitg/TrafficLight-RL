# TrafficLight-RL

This is Reinforcement Learning Project to Build RL agent to control Traffic light in a 4 way intersection to optimize traffic flow. Main aim is to minimize average waiting time of vehicles and increase average velocities. This is course project for CS221 course of Stanford University. 

## Simulator Used and Environement Setup:

We used Flow project by OpenAI and SUMO simulator. Flow us wrapper on top of SUMO, which provides RL wrappre and provide wasy ways to integrate with RL algorithms. Basically it creates a gym environment for us, that we can use with various out of box RL algorithms directly.

We had just 1 intersection. and TrafficLightFlow.py is the file where all the parameters of simulations are defined. Here are state,action and rewards used by us:
1. States: k-nearest vehicles position & Velocity from interseciton. we tried with k=2,5,10
2. Actions: 0,1. 1 when signal needs to be changed, else 0
3. Reward: - avg waiting time + penality for standstill vehicles + penality for switching lights. (penalities are -ve)

We had to dig further and by spending lot of time, we were able to add left turn in to the simulator. FLOW didn;t have any documentation for it. FLOW seems to be mroe focused on creating AI agents for Cars and not traffic lights. LeftTurnWithTwoLanes directory contains, custom network, environment and parameter file to create flow environment with 2 lanes in each edge and adding left turn. 

## RL Algorithms
We first tried Out of Box, OpenAI Baselines algorithms DQN with replay buffer and a target network. It uses TensorFlow with Keras Layer abstraction. It worked pretty well. Then we spent most of our time in writing algorithms from scratch in pytorch and performing experiments and generating SUMO emission file from predictions and generating inference from this emission results. we tried these 3 algorithms varient of Deep Q Learning.

1. Simple DQN: simple 2 layer MLP network for Q function approximator. without any replay experience and no target netowork.
2. DQN with Replay buffer: DQN + a replay buffer. Replay buffer improves sample efficiency of training. and you can get most out of your training data with few iterations.
3. DDQN with priortity buffer: Then we added one mpre Neural Net, which is called target netowork, which stores the weights for target Q Values. and after each n_updates, we copy parameters from Q network to Target Q network. Q network is updated once for each sample.


## We also used Checkpointing to keep storing models after each 100 episodes of training. you can find code in prediction files for storing and loading checkpoints in pytorch.
