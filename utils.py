import numpy as np
from PIL import Image
from torchvision import transforms

def preprocess_frame(frame):
    frame = Image.fromarray(frame)
    preprocess = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    return preprocess(frame).unsqueeze(0)

def show_video_of_model(agent, env_name):
    import gymnasium as gym
    import imageio

    env = gym.make(env_name, render_mode='rgb_array')
    state, _ = env.reset()
    frames = []
    done = False

    while not done:
        frames.append(env.render())
        action = agent.act(state)
        state, reward, done, _, _ = env.step(action)
    
    env.close()
    imageio.mimsave('video.mp4', frames, fps=30)
