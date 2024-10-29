import gymnasium as gym
import numpy as np
from collections import deque
from agent import Agent
from utils import show_video_of_model

env = gym.make('MsPacmanDeterministic-v0', full_action_space=False)
number_actions = env.action_space.n

agent = Agent(number_actions)

number_episodes = 2000
maximum_timesteps = 10000
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
scores = deque(maxlen=100)

for episode in range(1, number_episodes + 1):
    state, _ = env.reset()
    score = 0
    
    for t in range(maximum_timesteps):
        action = agent.act(state, epsilon)
        next_state, reward, done, _, _ = env.step(action)
        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward
        if done:
            break

    scores.append(score)
    epsilon = max(epsilon_min, epsilon_decay * epsilon)

    print(f'\rEpisode {episode}\tAverage Score: {np.mean(scores):.2f}', end="")
    if episode % 100 == 0:
        print(f'\rEpisode {episode}\tAverage Score: {np.mean(scores):.2f}')
    if np.mean(scores) >= 500.0:
        print(f'\nEnvironment solved in {episode - 100} episodes!\tAverage Score: {np.mean(scores):.2f}')
        torch.save(agent.local_qnetwork.state_dict(), 'checkpoint.pth')
        break

show_video_of_model(agent, 'MsPacmanDeterministic-v0')
