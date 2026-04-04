"""Train Medium environment. Run: python src/train_medium.py"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_wrapper import make_env
from config import get_config, get_ppo_kwargs
from callbacks import TrainingLogCallback, WinRateStoppingCallback, EvalCallback
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

env_name = 'medium'
config = get_config(env_name)
total_timesteps = 2_000_000

model_dir = f'models/{env_name}'
log_dir = f'results/{env_name}'

env = make_env(env_name, time_scale=20.0, no_graphics=True)
env = Monitor(env)

ppo_kwargs = get_ppo_kwargs(config)
model = PPO('MlpPolicy', env, verbose=0, tensorboard_log=log_dir, **ppo_kwargs)

callbacks = [
    CheckpointCallback(save_freq=50_000, save_path=model_dir, name_prefix='checkpoint'),
    TrainingLogCallback(log_dir=log_dir, config=config, env_name=env_name, print_freq=200),
    EvalCallback(env_name=env_name, eval_freq=50_000, n_eval_episodes=50,
                 log_dir=log_dir, worker_id=100),
    WinRateStoppingCallback(win_rate_threshold=0.92, min_episodes=300),
]

print(f'Training {env_name} for {total_timesteps:,} steps...', flush=True)
start = time.time()
try:
    model.learn(total_timesteps=total_timesteps, callback=callbacks)
except Exception as e:
    print(f'Error: {e}', flush=True)
elapsed = time.time() - start
print(f'Training completed in {elapsed/60:.1f} minutes', flush=True)

model.save(f'{model_dir}/final')
print(f'Model saved to {model_dir}/final.zip', flush=True)
env.close()
