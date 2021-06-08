import gym
import torch
import numpy as np
class MetaEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self,env_name,agent_list):
        self.env = gym.make(env_name)
        self.observation_space = self.env.observation_space
        self.action_space = gym.spaces.Discrete(len(agent_list))
        self.agents = [torch.load(fpath) for fpath in agent_list]

    def seed(self, seed=None):
        self.np_random, seed = random.seeding.np_random(seed)
        return [seed]
    

    def step(self, action, state):
        # action在此处是agent的索引
        agent = self.agents[action]
        with torch.no_grad():
            state = torch.as_tensor(state, dtype=torch.float32)
            prob_logit = agent.pi.logits_net(state)
            a = np.argmax(prob_logit)
            a = a.numpy().item()
        next_state, r, is_terminal,_ = self.env.step(a)
        return next_state, r, is_terminal,_

    def reset(self):
        self.state = self.env.reset()
        return self.state