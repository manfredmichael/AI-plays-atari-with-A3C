import tensorflow as tf
from collections import deque
import torch.multiprocessing as mp 
import numpy as np
import datetime


class LearningReport:
    def __init__(self, filename='agent', text_space=18, average_reward_episode=100):
        self.text_space = text_space
        self.episode_rewards = deque(maxlen=average_reward_episode)
        self.episode_reward = 0
        self.episode_counter = mp.Value('i', 0)
        
        filename = filename + datetime.datetime.now().strftime(' (%d-%m-%y %H:%M:%S)')
        self.writer = tf.summary.create_file_writer(logdir='logs/{}'.format(filename))

    def format_report(self, text):
        white_space = ''.join([' ' for i in range(self.text_space - len(text))])
        return text + white_space

    def add_reward(self, reward):
        self.episode_reward += reward

    def report_episode(self, epsilon=None, count_episode=True):
        if count_episode:
            with self.episode_counter.get_lock():
                self.episode_counter.value += 1

        self.episode_rewards.append(self.episode_reward)
        self.write_episode_report(epsilon)

        episode    = self.format_report('episode: ' + str(self.episode_counter.value))
        reward     = self.format_report('reward: ' + '{:.2f}'.format(np.mean(self.episode_reward)))
        reward_avg = self.format_report('reward avg: ' + '{:.2f}'.format(np.mean(self.episode_rewards)))

        if epsilon:
            epsilon = self.format_report('epsilon: ' + '{:.2f}'.format(epsilon))
        else:
            epsilon = ""

        report = episode + reward + reward_avg + epsilon
        self.episode_reward = 0
        
        return report
        
    def write_episode_report(self, epsilon):
        with self.writer.as_default():
            tf.summary.scalar('Episode_reward', self.episode_reward, step=self.episode_counter.value)
            tf.summary.scalar('Running_avg_reward', np.mean(self.episode_rewards), step=self.episode_counter.value)
            if epsilon:
                tf.summary.scalar('Epsilon', epsilon, step=self.episode_counter.value)
