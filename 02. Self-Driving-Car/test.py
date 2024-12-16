# ----- Import Dependencies ----- #
import os 
import gym 
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# ----- Load Model ----- #
env = DummyVecEnv([lambda: gym.make('CarRacing-v2', render_mode='human')])
log_path = os.path.join('Training', 'Logs')

ppo_path = os.path.join('Training', 'Saved_Models', 'PPO_Driving_Model')
model = PPO.load(ppo_path, env)

# ----- Evaluate Model ----- #
# evaluate_policy(model, env, n_eval_episodes=1, render=True)
# env.close()

# ----- Test Model ----- #

env = gym.make('CarRacing-v2', render_mode='human')

env = gym.make('CarRacing-v2', render_mode='human')
for ep in range(1, 3):
    obs, _ = env.reset()
    done = False
    score = 0
    while not done:
        env.render()
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        score += reward
    print(f"Episode: {ep} Score: {score}")
env.close()

# ----- View Logs ----- #
training_log_path = os.path.join(log_path, 'PPO_1')
print(training_log_path)
# !tensorboard --logdir={training_log_path}   (Run in terminal)