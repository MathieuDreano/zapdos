from pathlib import Path
from datetime import datetime
import uuid
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from tensorboard_callback import TensorboardCallback
from baselines_utils import load_or_create_model

if __name__ == '__main__':

    ep_length = 2048 * 1
    sess_path = Path(f'sessions/session_{datetime.now().strftime("%Y%m%d_%H%M")}')
    pretrained_model = 'session_4da05e87_main_good/poke_439746560_steps'
    model_i_like = 'sessions/session_20240303_2026/poke_6553600_steps'
    model_to_load_path = model_i_like
    env_config = {
                'headless': True, 'save_final_state': True, 'early_stop': False,
                'action_freq': 24, 'init_state': '../has_pokedex_nballs.state', 'max_steps': ep_length,
                'print_rewards': True, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
                'gb_path': '../PokemonRed.gb', 'debug': False, 'sim_frame_dist': 2_000_000.0,
                'use_screen_explore': True, 'extra_buttons': False, 'stream': True, 'instance_id': str(uuid.uuid4())[:8]
            }
    n_envs = 4 #64 #46  # Also sets the number of episodes per training iteration
    model = load_or_create_model(model_to_load_path, env_config, ep_length, n_envs=n_envs)

    save_freq = max(ep_length // n_envs, 1)  # From CheckpointCallback documentation
    print(f"save_freq: {save_freq}")
    checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path=sess_path, name_prefix='poke')
    callbacks = [checkpoint_callback, TensorboardCallback()]
    learn_steps = 5

    model.learn(total_timesteps=ep_length * n_envs * learn_steps, callback=CallbackList(callbacks), progress_bar=True)
