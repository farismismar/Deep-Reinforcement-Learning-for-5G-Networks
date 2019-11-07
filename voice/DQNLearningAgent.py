#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 19:02:40 2019

@author: farismismar
"""

# Used from: https://keon.io/deep-q-learning/
# https://github.com/keon/deep-q-learning/blob/master/dqn.py
# https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/reinforcement_learning/deep_q_network.py

# Check some more here: https://github.com/leimao/OpenAI_Gym_AI/tree/master/CartPole-v0/Deep_Q-Learning
# This adds a means to compute AverageQ as a sign of experience.

# https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
# https://github.com/dennybritz/reinforcement-learning/blob/master/DQN/dqn.py

# Deep Q-learning Agent
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.callbacks import History 
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf

class DQNLearningAgent:
    def __init__(self, seed,
                 discount_factor=0.995,
                 exploration_rate=1.0,
                 exploration_decay_rate=0.9995):
                               
        self.memory = deque(maxlen=2000)
        self.gamma = discount_factor    # discount rate
        self.exploration_rate = exploration_rate / exploration_decay_rate # exploration rate
        self.exploration_rate_min = 0.15
        self.exploration_rate_decay = exploration_decay_rate
        self.learning_rate = 0.01 # this is eta for SGD

        self._state_size = 6 
        self._action_size = 16 # 16 mmWave and 16 for voice.  check the actions in the environment
                  
        # Add a few lines to caputre the seed for reproducibility.
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        self.model = self._build_model()
                
    def begin_episode(self, observation):
        # Reduce exploration over time.
        self.exploration_rate *= self.exploration_rate_decay
        if (self.exploration_rate < self.exploration_rate_min):
            self.exploration_rate = self.exploration_rate_min
            
        # return an action at random
        action = random.randrange(self._action_size)

        return action

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        # This is a state-to-largest Q converter to find best action, basically
        model = Sequential()
        model.add(Dense(2, input_dim=self._state_size, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(self._action_size, activation='relu'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        
        return model
    
    def _construct_training_set(self, replay):
        # Select states and next states from replay memory
        states = np.array([a[0] for a in replay])
        new_states = np.array([a[3] for a in replay])

        # Predict the expected Q of current state and new state using DQN
        with tf.device('/gpu:0'):
            Q = self.model.predict(states)
            Q_new = self.model.predict(new_states)

        replay_size = len(replay)
        X = np.empty((replay_size, self._state_size))
        y = np.empty((replay_size, self._action_size))
        
        # Construct training set
        for i in range(replay_size):
            state_r, action_r, reward_r, new_state_r, done_r = replay[i]

            target = Q[i]
            target[action_r] = reward_r
            # If we're done the utility is simply the reward of executing action a in
            # state s, otherwise we add the expected maximum future reward as well
            if not done_r:
                target[action_r] += self.gamma * np.amax(Q_new[i])

            X[i] = state_r
            y[i] = target

        return X, y
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        # Make sure we restrict memory size to specified limit
        if len(self.memory) > 2000:
            self.memory.pop(0)
        
    def act(self, state):
        # Exploration/exploitation: choose a random action or select the best one.
        if np.random.uniform(0, 1) <= self.exploration_rate:
            return random.randrange(self._action_size)
        
        state = np.reshape(state, [1, self._state_size])
        with tf.device('/gpu:0'):
            act_values = self.model.predict(state)
            
        return np.argmax(act_values[0])  # returns action
    
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        X, y = self._construct_training_set(minibatch)
        with tf.device('/gpu:0'):
            loss = self.model.train_on_batch(X, y)
        
        _q = np.mean(y)
        return loss, _q
                
    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())
        return

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)