from os.path import exists
from pathlib import Path
import uuid

from stream_agent_wrapper import StreamWrapper
from tensorboard_callback import TensorboardCallback
from mathieu_gym_env import PokeRedEnv
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, ProgressBarCallback
from datetime import datetime
def make_env(rank, env_config, seed=0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = PokeRedEnv(env_config['gb_path'], env_config['init_state'], headless=env_config['headless'])
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init

if __name__ == '__main__':


    ep_length = 2048 * 8 # 16384
    sess_path = Path(f'session_{datetime.now().strftime("%Y%m%d-%H%M")}')

    env_config = {
                'headless': True, 'save_final_state': True, 'early_stop': False,
                'action_freq': 24, 'init_state': '../has_pokedex_nballs.state', 'max_steps': ep_length, 
                'print_rewards': True, 'save_video': True, 'fast_video': True, 'session_path': sess_path,
                'gb_path': '../PokemonRed.gb', 'debug': False, 'sim_frame_dist': 2_000_000.0, 
                'use_screen_explore': True, 'extra_buttons': False
            }
    
    
    num_cpu = 4 #64 #46  # Also sets the number of episodes per training iteration
    policy = 'MultiInputPolicy'

    env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])

    checkpoint_callback = CheckpointCallback(save_freq=ep_length, save_path=sess_path, name_prefix='poke')
    callbacks = [
        checkpoint_callback
        ,TensorboardCallback(sess_path)
        ,ProgressBarCallback()
    ]

    # put a checkpoint here you want to start from
    file_name = 'session_20260131-0921/poke_1310720_steps.zip'
    use_checkpoint = False
    if (use_checkpoint & exists(file_name)):
        print('\nloading checkpoint')
        model = PPO.load(file_name, env=env)
        model.n_steps = ep_length
        model.n_envs = num_cpu
        model.rollout_buffer.buffer_size = ep_length
        model.rollout_buffer.n_envs = num_cpu
        model.rollout_buffer.reset()
    else:
        #n_steps: The number of steps to run for each environment per update
        # (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        # NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        model = PPO(policy, env, verbose=1, n_steps=ep_length, batch_size=512, n_epochs=1, gamma=0.999, tensorboard_log=sess_path)

    # total_timesteps – The total number of samples (env steps) to train o
    total_timesteps = ep_length * num_cpu * 10
    print('total timesteps:', total_timesteps)
    print('expected n_updates', total_timesteps // (ep_length * num_cpu))
    model.learn(total_timesteps=total_timesteps, callback=CallbackList(callbacks))
