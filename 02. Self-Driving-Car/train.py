# ----- Import Dependencies ----- #
import os 
import gym 
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


# ----- Load and Understand Environment ----- #

env = gym.make('CarRacing-v2', render_mode='human')
print(f'Action Space: {env.action_space}')
print(f'Observation Space: {env.observation_space}')

for ep in range(1, 3):
    env.reset()
    done = False
    score = 0
    while not done:
        env.render()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        score += reward
    print(f"Episode: {ep} Score: {score}")
env.close()

# ----- Train RL Model ----- #

train_env = DummyVecEnv([lambda: gym.make('CarRacing-v2', render_mode='human')])
log_path = os.path.join('Training', 'Logs')
model = PPO('CnnPolicy', train_env, verbose=1, tensorboard_log=log_path)
model.learn(total_timesteps=100000)

# ----- Save Model ----- #

ppo_path = os.path.join('Training', 'Saved_Models', 'PPO_Driving_Model')
model.save(ppo_path)

