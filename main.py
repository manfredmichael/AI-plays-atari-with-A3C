import torch.multiprocessing as mp 
from optimizer import SharedAdam
from agent import Agent
from model import ActorCritic

import gym

def main(bruh):
    lr = 1e-3
    env_id = 'LunarLander-v2'
    n_actions = 4 
    input_dims = [8]
    global_actor_critic = ActorCritic(input_dims, n_actions)
    global_actor_critic.share_memory()
    optim = SharedAdam(global_actor_critic.parameters(),
                       lr=lr, betas=(0.92, 0.9999), weight_decay=1e-6)
    global_ep = mp.Value('i', 0)

    workers = [Agent(global_actor_critic,
                     optim,
                     input_dims,
                     n_actions,
                     gamma=0.99,
                     lr=lr,
                     name=i,
                     global_ep_idx=global_ep,
                     env_id=env_id) for i in range(mp.cpu_count())]
    [w.start() for w in workers]
    [w.join() for w in workers]




if __name__ == '__main__':
    main('moment')
