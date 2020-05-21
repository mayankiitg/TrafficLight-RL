import gym

from baselines import deepq
from TrafficLightFlow import GetTrafficLightEnv, getFlowParamsForTls
from flow.core.experiment import Experiment 

t = 0
def static_rl_actions(state):
    global t
    t += 1
    return t%20 == 0 


def main():
    env = GetTrafficLightEnv()
    act = deepq.learn(env, network='mlp', total_timesteps=0, load_path="tls_model.pkl")
    reward = 0
    iterations = 1
    exp = Experiment(getFlowParamsForTls())

    ## This is the RL agent that is using the trained model that we saved from train_tls file
    rl_agent = lambda state: act(state[None])[0]

    ## This is the static agent that switches the light every 20s
    static_agent = static_rl_actions

    # Passing the appropriate lambda among static and rl, you can perform the experiment
    exp.run(10, rl_agent, convert_to_csv=True)


if __name__ == '__main__':
    main()
