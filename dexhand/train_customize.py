import dexhand
import torch

env_name = 'BotyardHandPick'
# env_name = 'ShadowHandBlockStack'
algo = "ppo"
env = dexhand.make(env_name, algo)

obs = env.reset()
terminated = False

while not terminated:
    act = torch.tensor(env.action_space.sample()).repeat((env.num_envs, 1))
    obs, reward, done, info = env.step(act)
