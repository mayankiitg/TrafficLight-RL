import gym

from baselines import deepq
from TrafficLightFlow import GetTrafficLightEnv


def callback(lcl, _glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved


def main():
    #env = gym.make("TrafficLightGridPOEnv-v0")
    
    env = GetTrafficLightEnv()

    act = deepq.learn(
        env,
        network='mlp',
        lr=1e-3,
        total_timesteps=20000,
        buffer_size=5000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10,
        callback=callback
    )
    print("Saving model to cartpole_model.pkl")
    act.save("tls_model.pkl")


if __name__ == '__main__':
    main()
