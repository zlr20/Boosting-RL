import argparse
import gym
import os
from utils.mpi_tools import mpi_fork
from algo.ppo.ppo import ppo_train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='LunarLander-v2')
    parser.add_argument('--hid', type=int, default=128)
    parser.add_argument('--l', type=int, default=4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--log_dir', type=str, default='./log')
    parser.add_argument('--exp_name', type=str, default='ppo')
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    logger_kwargs = dict(
        output_dir=os.path.join(args.log_dir,args.exp_name,'seed'+str(args.seed)),
        exp_name=args.exp_name,
    )

    ppo_train(  lambda : gym.make(args.env),
            ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
            gamma=args.gamma, 
            seed=args.seed, 
            steps_per_epoch=args.steps,
            epochs=args.epochs,
            max_ep_len=400,
            logger_kwargs=logger_kwargs)