import torch.multiprocessing as mp 
from optimizer import SharedAdam
from agent import Agent
from model import ActorCritic
from utils import LearningReport

import gym

def main(bruh):
    lr = 1e-3
    # env_id = 'LunarLander-v2'
    # n_actions = 4 
    # input_dims = [8]
    env_id = 'CartPole-v0'
    n_actions = 2 
    input_dims = [4]

    global_actor_critic = ActorCritic(input_dims, n_actions)
    global_actor_critic.share_memory()

    report = LearningReport()

    optim = SharedAdam(global_actor_critic.parameters(),
                       lr=lr, betas=(0.9, 0.999), weight_decay=1e-7)

    workers = [Agent(global_actor_critic,
                     optim,
                     input_dims,
                     n_actions,
                     gamma=0.99,
                     lr=lr,
                     report=report,
                     name=i,
                     env_id=env_id) for i in range(mp.cpu_count())]
    [w.start() for w in workers]
    [w.join() for w in workers]




if __name__ == '__main__':
    main('moment')
