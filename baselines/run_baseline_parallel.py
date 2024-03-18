from pathlib import Path
from datetime import datetime
import uuid
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from tensorboard_callback import TensorboardCallback
from baselines_utils import load_or_create_model

if __name__ == '__main__':

    ep_length = 2048 * 16
    n_envs = 10 #64 #46
    sess_path = Path(f'sessions/session_{datetime.now().strftime("%Y%m%d_%H%M")}')
    env_config = {
                'headless': True, 'save_final_state': True, 'early_stop': False,
                'action_freq': 24, 'init_state': '../has_pokedex_nballs.state', 'max_steps': ep_length,
                'print_rewards': True, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
                'gb_path': '../PokemonRed.gb', 'debug': False, 'sim_frame_dist': 2_000_000.0,
                'use_screen_explore': True, 'extra_buttons': False, 'stream': False, 'instance_id': str(uuid.uuid4())[:8]
            }

    pretrained_model = 'session_4da05e87_main_good/poke_439746560_steps'
    model_fresh_start = 'sessions/session_20240318_1426/poke_3276000_steps'
    model_to_load_path = model_fresh_start
    model = load_or_create_model(model_to_load_path, env_config, ep_length, n_envs=n_envs)

    callbacks = CallbackList([
        CheckpointCallback(save_freq=max(ep_length // n_envs, 1), save_path=sess_path, name_prefix='poke'),
        TensorboardCallback()
    ])

    # LEARN
    learn_steps = 10
    model.learn(total_timesteps=ep_length * n_envs * learn_steps, callback=callbacks, progress_bar=True)
