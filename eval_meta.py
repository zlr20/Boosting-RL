import time
import gym
import os
import os.path as osp
import numpy as np
import torch
import argparse
from utils.logx import EpochLogger
import random

def load_pytorch_policy(fpath, agent_list=['./pretrained/ppo_2*256_200.pt','./pretrained/ppo_4*128_200.pt']):
    """ Load a pytorch policy saved with Spinning Up Logger."""

    model = torch.load(fpath)

    agents = [torch.load(f) for f in agent_list]

    # make function for producing an action given a single state
    def get_action(x):
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32)
            prob_logit = model.pi.logits_net(x)
            action = np.argmax(prob_logit)
            agent = agents[action]
            real_prob_logit = agent.pi.logits_net(x)
            real_action = np.argmax(real_prob_logit)
        return real_action.numpy().item()

    return get_action



def run_policy(env, get_action, max_ep_len, num_episodes, render=False, collect_data=False):

    logger = EpochLogger(output_dir='.', output_fname='eval.txt')

    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
    action1_count = 0
    while n < num_episodes:
        if render:
            env.render()
            #time.sleep(1e-3)
        a = get_action(o)
        if a == 1:
            action1_count += 1
        o_, r, d, _ = env.step(a)
        o = o_
        ep_ret += r
        ep_len += 1

        if d or (ep_len == max_ep_len):
            action1_rate = 1. * action1_count/ep_len
            logger.store(EpRet=ep_ret, EpLen=ep_len, Action1Rate=action1_rate)
            print('Episode %d \t EpRet %.3f \t EpLen %d ' % (n, ep_ret, ep_len))
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            diff_count = 0
            action1_count = 0
            n += 1

    
    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.log_tabular('Action1Rate', average_only=True)
    logger.dump_tabular()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='LunarLander-v2')  #CartPole-v0
    parser.add_argument('--fpath', '-f', type=str, default='log/meta/seed0/pyt_save/model.pt')
    parser.add_argument('--maxlen', '-l', type=int, default=400)
    parser.add_argument('--episodes', '-n', type=int, default=20)
    parser.add_argument('--render', '-r',action='store_true',default=0)
    args = parser.parse_args()
    env = gym.make(args.env)
    get_action = load_pytorch_policy(args.fpath)
    run_policy(env, get_action, args.maxlen, args.episodes, args.render)
    os.remove('eval.txt')
