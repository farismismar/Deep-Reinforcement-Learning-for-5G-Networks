#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 20:46:19 2019

@author: farismismar
"""
import random
import numpy as np

# Following from
# https://github.com/YuriyGuts/cartpole-q-learning/blob/master/cartpole.py

# Action size and bin sizes both need to be reflected to the environment settings.

class QLearningAgent:
    def __init__(self, seed,
                 learning_rate=0.2,
                 discount_factor=0.995,
                 exploration_rate=1.0,
                 exploration_decay_rate=0.9995):

        self.learning_rate = learning_rate          # alpha
        self.discount_factor = discount_factor      # gamma
        self.exploration_rate = exploration_rate / exploration_decay_rate   # epsilon
        self.exploration_rate_min = 0.15
        self.exploration_decay_rate = exploration_decay_rate # d
        self.state = None
        self.action = None
        
        self._state_size = 6 # discretized from observation
        self._action_size = 16 # check the actions in the environment
        
        self.oversampling = 1 # for discretization.  Increasing may cause memory exhaust.
        
        # Add a few lines to caputre the seed for reproducibility.
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # Discretize the continuous state space for each of the features.
        num_discretization_bins = self._state_size * self.oversampling
        
        # check the site distance configuration in the environment
        self._state_bins = [
            # User X - serv
            self._discretize_range(-350, 350, num_discretization_bins),
            # User Y - serv
            self._discretize_range(-350, 350, num_discretization_bins),
            # User X - interf
            self._discretize_range(175, 875, num_discretization_bins),
            # User Y - interf
            self._discretize_range(-350, 350, num_discretization_bins),
            # Serving BS power.
            self._discretize_range(0, 40, num_discretization_bins),
            # Interfering BS power.
            self._discretize_range(0, 40, num_discretization_bins),
        ]
        
        # Create a clean Q-Table.
        self._max_bins = max(len(bin) for bin in self._state_bins)
        num_states = (self._max_bins + 1) ** len(self._state_bins)
        self.q = np.zeros(shape=(num_states, self._action_size))
        
    def begin_episode(self, observation):
        # Reduce exploration over time.
        self.exploration_rate *= self.exploration_decay_rate
        if (self.exploration_rate < self.exploration_rate_min):
            self.exploration_rate = self.exploration_rate_min
    
        self.state = self._build_state(observation)
        
        return np.argmax(self.q[self.state, :]) # returns the action with largest Q
    
    def act(self, observation, reward):
        next_state =  self._build_state(observation)
        state = self.state

        # Exploration/exploitation: choose a random action or select the best one.
        enable_exploration = np.random.uniform(0, 1) <= self.exploration_rate
        if enable_exploration:
            next_action = np.random.randint(0, self._action_size)
        else:
            next_action = np.argmax(self.q[next_state])
        
        # Learn: update Q-Table based on current reward and future action.
        self.q[state, self.action] += self.learning_rate * \
            (reward + self.discount_factor * max(self.q[next_state, :]) - self.q[state, self.action])
    
        self.state = next_state
        self.action = next_action
        return next_action

    def get_performance(self):
        return self.q.mean()

    # Private members:
    def _build_state(self, observation):
        # Discretize the observation features and reduce them to a single integer.
        state = sum(
            self._discretize_value(feature, self._state_bins[i]) * ((self._max_bins + 1) ** i)
            for i, feature in enumerate(observation)
        )
        return state
    
    def _discretize_value(self, value, bins):
        return np.digitize(x=value, bins=bins)
    
    def _discretize_range(self, lower_bound, upper_bound, num_bins):
        return np.linspace(lower_bound, upper_bound, num_bins + 1)[1:-1]
