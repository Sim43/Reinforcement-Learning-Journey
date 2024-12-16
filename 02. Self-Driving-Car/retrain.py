# ----- Import Dependencies ----- #
import os 
import gym 
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# ----- Load Model ----- #
env = DummyVecEnv([lambda: gym.make('CarRacing-v2', render_mode='human')])
log_path = os.path.join('Training', 'Logs')

ppo_path = os.path.join('Training', 'Saved_Models', 'PPO_Driving_Model')
model = PPO.load(ppo_path, env)

# ----- Further Train Model ----- # 
model.learn(total_timesteps=100000)

# ----- Save Model ----- #

ppo_path = os.path.join('Training', 'Saved_Models', 'PPO_Driving_Model')
model.save(ppo_path)