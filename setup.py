from PIL import Image
import gymnasium as gym
import ale_py


gym.register_envs(ale_py)
env = gym.make("ALE/SpaceInvaders-v5", render_mode = "human")
obs, info = env.reset()
episode_over = False
count = 0
while not episode_over:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    episode_over = terminated or truncated
    #image = Image.fromarray(obs)
    #image = image.resize((image.width*3, image.height*3))
    #count = count + 1
    #image.save(f"Images/image{count}.png")
env.close()

print(env.action_space)