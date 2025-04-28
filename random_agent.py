import random
import gymnasium as gym
import ale_py


class QAgent(object):
    def __init__(self):
        self.q_table = {}
        self.epsilon = 0.2
        self.alpha = 0.5
        self.discount = 0.5
    def getQValue(self,state,action):
        if state not in self.q_table:
            self.q_table[state] = {}
            self.q_table[state][action] = 0
            q_value = 0.0
        if action not in self.q_table[state]:
            q_value = 0.0
        else:
            q_value = self.q_table[state][action]

        return q_value

    def computeValuefromQValue(self,state):
        actions = [0,1,2,3,4]
        best_action = None
        max_value = -1000
        for action in actions:
            value = self.getQValue(state,action)
            if value > max_value:
                max_value = value
                best_action = action
        return max_value

    def computeActionfromQValue(self,state):
        actions = [0,1,2,3,4]
        best_action = None
        max_value = -1000
        for action in actions:
            value = self.getQValue(state,action)
            if value > max_value:
                max_value = value
                best_action = action
        return best_action

    def getAction(self,state):
        actions = [0,1,2,3,4]
        action = None
        if random.random() > self.epsilon:
            action = random.choice(actions)
        else:
            action = self.computeActionfromQValue(state)
        if action == None:
            action = random.choice(actions)
        return action

    def update(self,state,action,next_state,reward):
        current_q_value = self.getQValue(state,action)
        value = self.alpha * (reward + self.discount*self.computeValuefromQValue(next_state) - current_q_value)
        self.q_table[state][action] = (1-self.alpha)*current_q_value + value
        return

gym.register_envs(ale_py)
env = gym.make("ALE/SpaceInvaders-v5", render_mode = "human")
obs, info = env.reset()
episode_over = False
my_agent = QAgent()
for _ in range(10):
    while not episode_over:
        action = my_agent.getAction(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_over = terminated or truncated

gym.register_envs(ale_py)
env = gym.make("ALE/SpaceInvaders-v5", render_mode = "human")
obs, info = env.reset()
episode_over = False


action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)


env.close()

print(env.action_space.list())