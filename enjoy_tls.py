## This file is used to run the trained model

import gym

from baselines import deepq
from TrafficLightFlow import GetTrafficLightEnv


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

    for i in range(iterations):
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            env.render()
            #for RL use: act(obs[None])[0]
            action = static_rl_actions(obs)

            print(f'computed action: {action}')

            obs, rew, done, _ = env.step(action)
            episode_rew += rew
        print("Episode reward", episode_rew)


if __name__ == '__main__':
    main()
