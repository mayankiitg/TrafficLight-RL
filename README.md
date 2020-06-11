# TrafficLight-RL

This is Reinforcement Learning Project to Build RL agent to control Traffic light in a 4 way intersection to optimize traffic flow. Main aim is to minimize average waiting time of vehicles and increase average velocities.

##Simulator Used and Environement Setup:

We used Flow project by OpenAI and SUMO simulator. Flow us wrapper on top of SUMO, which provides RL wrappre and provide wasy ways to integrate with RL algorithms. Basically it creates a gym environment for us, that we can use with various out of box RL algorithms directly.

We had just 1 intersection. and TrafficLightFlow.py is the file where all the parameters of simulations are defined. Here are state,action and rewards used by us:
1. States:
2. Actoins:
3. Reward:

We had to dig further and by spending lot of time, we were able to add left turn in to the simulator. FLOW didn;t have any documentation for it. FLOW seems to be mroe focused on creating AI agents for Cars and not traffic lights. LeftTurnWithTwoLanes directory contains, custom network, environement and parameter file to create flow environment 
