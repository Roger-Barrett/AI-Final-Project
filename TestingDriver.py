from Agent02 import ApproximateQAgent
import math
import random
import gymnasium as gym
import ale_py
import cv2
from PIL import Image

my_agent = ApproximateQAgent()


#gym.register_envs(ale_py)
#env = gym.make("ALE/SpaceInvaders-v5", render_mode = "human")
#obs, info = env.reset()
#episode_over = False
#count = 0
gym.register_envs(ale_py)
for _ in range(50):
    env = gym.make("ALE/SpaceInvaders-v5")
    obs, info = env.reset()
    episode_over = False
    print("Current Weights" ,my_agent.weights)
    while not episode_over:
        action = my_agent.getAction(obs)
        previous_obs = obs.copy()
        obs, reward, terminated, truncated, info = env.step(action)
        episode_over = terminated or truncated
        my_agent.update(previous_obs,action,obs,reward)
    my_agent.alpha = my_agent.alpha*0.70

gym.register_envs(ale_py)
env = gym.make("ALE/SpaceInvaders-v5", render_mode = "human")
obs, info = env.reset()
episode_over = False
count = 0
while not episode_over:
    action = my_agent.getAction(obs)
    previous_obs = obs.copy()
    obs, reward, terminated, truncated, info = env.step(action)
    episode_over = terminated or truncated
    my_agent.update(previous_obs, action, obs, reward)
    image = Image.fromarray(obs)
    image = image.resize((image.width*3, image.height*3))
    count = count + 1
    image.save(f"Agent02_Trained/image{count}.png")
