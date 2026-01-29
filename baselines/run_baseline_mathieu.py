from os.path import exists
from pathlib import Path
import uuid

from stream_agent_wrapper import StreamWrapper
from tensorboard_callback import TensorboardCallback
from mathieu_gym_env import PokeRedEnv
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
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


    ep_length = 2048 * 2
    sess_path = Path(f'session_{datetime.now().strftime("%Y%m%d-%H%M%S")}_{str(uuid.uuid4())[:8]}')
    #sess_path = Path(f'session_{str(uuid.uuid4())[:8]}')

    env_config = {
                'headless': False, 'save_final_state': True, 'early_stop': False,
                'action_freq': 24, 'init_state': '../has_pokedex_nballs.state', 'max_steps': ep_length, 
                'print_rewards': True, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
                'gb_path': '../PokemonRed.gb', 'debug': False, 'sim_frame_dist': 2_000_000.0, 
                'use_screen_explore': True, 'extra_buttons': False
            }
    
    
    num_cpu = 1 #64 #46  # Also sets the number of episodes per training iteration

    if (num_cpu > 1):
       env_config['headless'] = True
       policy = 'MultiInputPolicy'
    else:
       env_config['headless'] = False
       policy = 'MultiInputPolicy'

    env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])

    checkpoint_callback = CheckpointCallback(save_freq=ep_length, save_path=sess_path, name_prefix='poke')
    callbacks = [checkpoint_callback, TensorboardCallback(sess_path)]
    learn_steps = 40

    model = PPO(policy, env, verbose=1, n_steps=ep_length, batch_size=32, n_epochs=1, gamma=0.999, tensorboard_log=sess_path)

    for i in range(learn_steps):
        model.learn(total_timesteps=(ep_length)*num_cpu*1000, callback=CallbackList(callbacks))
