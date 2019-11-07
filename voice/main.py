#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  28 09:36:05 2019

@author: farismismar
"""

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
 
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="0";   # My NVIDIA GTX 1080 Ti FE GPU

import random
import numpy as np
from colorama import Fore, Back, Style

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as tick

from environment import radio_environment
from DQNLearningAgent import DQNLearningAgent as QLearner # Deep with GPU and CPU fallback
#from QLearningAgent import QLearningAgent as QLearner

MAX_EPISODES = 5000
MAX_EPISODES_DEEP = 5000

os.chdir('/Users/farismismar/Desktop/voice')

##    ##    ##    ##    ##    ##    ##    ##    ##    ##    ##    ##    ##    ##    ##    ##    ##    ##    ##    ##    ##


def run_agent_tabular(env, plotting=True):
    max_episodes_to_run = MAX_EPISODES # needed to ensure epsilon decays to min
    max_timesteps_per_episode = radio_frame
    successful = False
    episode_successful = [] # a list to save the good episodes
    Q_values = []   

    max_episode = -1
    max_reward = -np.inf
    
    print('Ep.         | TS | Recv. SINR (srv) | Recv. SINR (int) | Serv. Tx Pwr | Int. Tx Pwr | Reward ')
    print('--'*54)
    
    # Implement the Q-learning algorithm
    for episode_index in 1 + np.arange(max_episodes_to_run):
        observation = env.reset()
      #  observation = np.reshape(observation, [1, agent._state_size])
        (_, _, _, _, pt_serving, pt_interferer) = observation
        
        action = agent.begin_episode(observation)
        # Let us know how we did.
        print('{}/{} | {:.2f} | {} | {:.2f} dB | {:.2f} dB | {:.2f} W | {:.2f} W | {:.2f} | {} '.format(episode_index, max_episodes_to_run, 
                                                                                      agent.exploration_rate,
                                                                                      0, 
                                                                                      np.nan,
                                                                                      np.nan,
                                                                                      pt_serving, pt_interferer, 
                                                                                      0, action))   
        
        action = agent.begin_episode(observation)
        total_reward = 0
        done = False
        actions = []
        
        sinr_progress = [] # needed for the SINR based on the episode.
        sinr_ue2_progress = [] # needed for the SINR based on the episode.
        serving_tx_power_progress = []
        interfering_tx_power_progress = []
        
        episode_q = []

        for timestep_index in 1 + np.arange(max_timesteps_per_episode):
            # Take a step
            next_observation, reward, done, abort = env.step(action)
            (_, _, _, _, pt_serving, pt_interferer) = next_observation

            action = agent.act(observation, reward)
            
            received_sinr = env.received_sinr_dB
            received_ue2_sinr = env.received_ue2_sinr_dB
            
            # Learn control policy
            q = agent.get_performance()

            episode_q.append(q)
            
            # If the episode has ended prematurely, penalize the agent by taking away the winning reward.
  #          if done and timestep_index < max_timesteps_per_episode - 1:
  #              reward -= env.reward_max
 #               abort = True

            # make next_state the new current state for the next frame.
            observation = next_observation
            total_reward += reward            
                            
            successful = (total_reward > 0) and (abort == False)
            
            # Let us know how we did.
            print('{}/{} | {:.2f} | {} | {:.2f} dB | {:.2f} dB | {:.2f} W | {:.2f} W | {:.2f} | {} '.format(episode_index, max_episodes_to_run, 
                                                                                          agent.exploration_rate,
                                                                                          timestep_index, 
                                                                                          received_sinr,
                                                                                          received_ue2_sinr,
                                                                                          pt_serving, pt_interferer, 
                                                                                          total_reward, action), end='')     
    
            actions.append(action)
            sinr_progress.append(env.received_sinr_dB)
            sinr_ue2_progress.append(env.received_ue2_sinr_dB)
            serving_tx_power_progress.append(env.serving_transmit_power_dBm)
            interfering_tx_power_progress.append(env.interfering_transmit_power_dBm)
            
            if abort == True:
                print('ABORTED.')
                break
            else:
                print()            
                        
        # at the level of the episode end
        q_z = np.mean(episode_q)
        
        if (successful == True) and (abort == False):
            #reward = env.reward_max
            #total_reward += reward
            print(Fore.GREEN + 'SUCCESS.  Total reward = {}.'.format(total_reward))
            print(Style.RESET_ALL)
            
            if (total_reward > max_reward):
                max_reward, max_episode = total_reward, episode_index
        else:
            reward = 0
            print(Fore.RED + 'FAILED TO REACH TARGET.')
            print(Style.RESET_ALL)

        Q_values.append(q_z)
        
        if (successful):
            episode_successful.append(episode_index)
            
        #        print('SINR progress')
        #        print(sinr_progress)
        #        print('Serving BS transmit power progress')
        #        print(serving_tx_power_progress)
        #        print('Interfering BS transmit power progress')
        #        print(interfering_tx_power_progress)
             
# Note..  remove the break on line 181
# for looping against all episodes
           
            # Plot the episode...
            if (plotting and successful): 
                plot_measurements(sinr_progress, sinr_ue2_progress, serving_tx_power_progress, interfering_tx_power_progress, max_timesteps_per_episode, episode_index, episode_index)
                plot_actions(actions, max_timesteps_per_episode, episode_index, episode_index)

            plot_performance_function_deep(Q_values, episode_index, is_loss=False)
                             
            print('First successful episode:')
            print(episode_successful)
            optimal = 'Episode {}/{} generated the highest reward {}.'.format(max_episode, MAX_EPISODES, max_reward)
            print(optimal)
            
            # Store all values in files
            filename = 'figures/optimal.txt'
            file = open(filename, 'w')
            file.write(optimal)    
            file.close()
            
            #break

    if (len(episode_successful) == 0):
        print("Goal cannot be reached after {} episodes.  Try to increase maximum episodes.".format(max_episodes_to_run))

def run_agent_deep(env, plotting=True):
    max_episodes_to_run = MAX_EPISODES_DEEP # needed to ensure epsilon decays to min
    max_timesteps_per_episode = radio_frame
    successful = False
    episode_successful = [] # a list to save the good episodes
    Q_values = []   
    losses = []
    
    batch_size = 32
    
    max_episode = -1
    max_reward = -np.inf
    
    print('Ep.         | TS | Recv. SINR (srv) | Recv. SINR (int) | Serv. Tx Pwr | Int. Tx Pwr | Reward ')
    print('--'*54)
    
    # Implement the Q-learning algorithm
    for episode_index in 1 + np.arange(max_episodes_to_run):
        observation = env.reset()
      #  observation = np.reshape(observation, [1, agent._state_size])
      
        (_, _, _, _, pt_serving, pt_interferer) = observation
        
        action = agent.begin_episode(observation)
        # Let us know how we did.
        print('{}/{} | {:.2f} | {} | {:.2f} dB | {:.2f} dB | {:.2f} W | {:.2f} W | {:.2f} | {} '.format(episode_index, max_episodes_to_run, 
                                                                                      agent.exploration_rate,
                                                                                      0, 
                                                                                      np.nan,
                                                                                      np.nan,
                                                                                      pt_serving, pt_interferer, 
                                                                                      0, action))   

        action = agent.begin_episode(observation)
        total_reward = 0
        done = False
        actions = []
        
        sinr_progress = [] # needed for the SINR based on the episode.
        sinr_ue2_progress = [] # needed for the SINR based on the episode.
        serving_tx_power_progress = []
        interfering_tx_power_progress = []
        
        episode_loss = []
        episode_q = []

        for timestep_index in 1 + np.arange(max_timesteps_per_episode):
            # Take a step
            next_observation, reward, done, abort = env.step(action)
            (_, _, _, _, pt_serving, pt_interferer) = next_observation
                        
            received_sinr = env.received_sinr_dB
            received_ue2_sinr = env.received_ue2_sinr_dB
            
          #  next_observation = np.reshape(next_observation, [1, agent._state_size])
            
            # Remember the previous state, action, reward, and done
            agent.remember(observation, action, reward, next_observation, done)
                           
            # Sample replay batch from memory
            sample_size = min(len(agent.memory), batch_size)
            
            # Learn control policy
            loss, q = agent.replay(sample_size)
                      
            episode_loss.append(loss)
            episode_q.append(q)
            
            # If the episode has ended prematurely, penalize the agent by taking away the winning reward.
  #          if done and timestep_index < max_timesteps_per_episode - 1:
  #              reward -= env.reward_max
 #               abort = True

            # make next_state the new current state for the next frame.
            observation = next_observation
            total_reward += reward            
                            
            successful = (total_reward > 0) and (abort == False)
            
            # Let us know how we did.
            print('{}/{} | {:.2f} | {} | {:.2f} dB | {:.2f} dB | {:.2f} W | {:.2f} W | {:.2f} | {} | '.format(episode_index, max_episodes_to_run, 
                                                                                          agent.exploration_rate,
                                                                                          timestep_index, 
                                                                                          received_sinr,
                                                                                          received_ue2_sinr,
                                                                                          pt_serving, pt_interferer, 
                                                                                          total_reward, action), end='')     
    
            actions.append(action)
            sinr_progress.append(env.received_sinr_dB)
            sinr_ue2_progress.append(env.received_ue2_sinr_dB)
            serving_tx_power_progress.append(env.serving_transmit_power_dBm)
            interfering_tx_power_progress.append(env.interfering_transmit_power_dBm)
            
            if abort == True:
                print('ABORTED.')
                break
            else:
                print()            
            
            # Update for the next episode
            action = agent.act(observation)
            
        # at the level of the episode end
        loss_z = np.mean(episode_loss)
        q_z = np.mean(episode_q)
        
        if (successful == True) and (abort == False):
            #reward = env.reward_max
            #total_reward += reward
            print(Fore.GREEN + 'SUCCESS.  Total reward = {}.  Loss = {}.'.format(total_reward, loss_z))
            print(Style.RESET_ALL)
            
            if (total_reward > max_reward):
                max_reward, max_episode = total_reward, episode_index
        else:
            reward = 0
            print(Fore.RED + 'FAILED TO REACH TARGET.')
            print(Style.RESET_ALL)

        
        losses.append(loss_z)
        Q_values.append(q_z)
        
        if (successful):
            episode_successful.append(episode_index)
            
        #        print('SINR progress')
        #        print(sinr_progress)
        #        print('Serving BS transmit power progress')
        #        print(serving_tx_power_progress)
        #        print('Interfering BS transmit power progress')
        #        print(interfering_tx_power_progress)
                        
# Note.. remove the break on line 342
# for looping against all episodes
        
            # Plot the episode...
            if (plotting and successful): 
                plot_measurements(sinr_progress, sinr_ue2_progress, serving_tx_power_progress, interfering_tx_power_progress, max_timesteps_per_episode, episode_index, episode_index)
                plot_actions(actions, max_timesteps_per_episode, episode_index, episode_index)
          
            plot_performance_function_deep(losses, episode_index, is_loss=True)
            plot_performance_function_deep(Q_values, episode_index, is_loss=False)
                             
            print('First successful episode:')
            print(episode_successful)
            optimal = 'Episode {}/{} generated the highest reward {}.'.format(max_episode, MAX_EPISODES_DEEP, max_reward)
            print(optimal)
            
            # Store all values in files
            filename = 'figures/optimal.txt'
            file = open(filename, 'w')
            file.write(optimal)    
            file.close()
            
            agent.save('deep_rl.model')
            #break

    if (len(episode_successful) == 0):
        print("Goal cannot be reached after {} episodes.  Try to increase maximum episodes.".format(max_episodes_to_run))


def run_agent_fpa(env, plotting=True):
    max_episodes_to_run = MAX_EPISODES # needed to ensure epsilon decays to min
    max_timesteps_per_episode = radio_frame
    successful = False
    episode_successful = [] # a list to save the good episodes

    max_episode = -1
    max_reward = -np.inf
    
    print('Ep.         | TS | Recv. SINR (srv) | Recv. SINR (int) | Serv. Tx Pwr | Int. Tx Pwr | Reward ')
    print('--'*54)
    
    # Implement the Q-learning algorithm
    for episode_index in 1 + np.arange(max_episodes_to_run):
        observation = env.reset()
      #  observation = np.reshape(observation, [1, agent._state_size])
        (_, _, _, _, pt_serving, pt_interferer) = observation
        
        # Let us know how we did.
        print('{}/{} | {:.2f} | {} | {:.2f} dB | {:.2f} dB | {:.2f} W | {:.2f} W | {:.2f} | '.format(episode_index, max_episodes_to_run, 
                                                                                      np.nan,
                                                                                      0, 
                                                                                      np.nan,
                                                                                      np.nan,
                                                                                      pt_serving, pt_interferer, 
                                                                                      0))   
        
        action = -1
        total_reward = 1
        done = False
        actions = []
        
        sinr_progress = [] # needed for the SINR based on the episode.
        sinr_ue2_progress = [] # needed for the SINR based on the episode.
        serving_tx_power_progress = []
        interfering_tx_power_progress = []

        for timestep_index in 1 + np.arange(max_timesteps_per_episode):
            # Take a step
            next_observation, reward, done, abort = env.step(action)
            (_, _, _, _, pt_serving, pt_interferer) = next_observation

            
            received_sinr = env.received_sinr_dB
            received_ue2_sinr = env.received_ue2_sinr_dB
            
            # make next_state the new current state for the next frame.
            observation = next_observation
            total_reward += reward        
                            
            successful = (total_reward > 0) and (abort == False)
            
            # Let us know how we did.
            print('{}/{} | {:.2f} | {} | {:.2f} dB | {:.2f} dB | {:.2f} W | {:.2f} W | {:.2f} | '.format(episode_index, max_episodes_to_run, 
                                                                                          np.nan,
                                                                                          timestep_index, 
                                                                                          received_sinr,
                                                                                          received_ue2_sinr,
                                                                                          pt_serving, pt_interferer, 
                                                                                          total_reward), end='')     
    
            actions.append(action)
            sinr_progress.append(env.received_sinr_dB)
            sinr_ue2_progress.append(env.received_ue2_sinr_dB)
            serving_tx_power_progress.append(env.serving_transmit_power_dBm)
            interfering_tx_power_progress.append(env.interfering_transmit_power_dBm)
            
            if abort == True:
                print('ABORTED.')
                break
            else:
                print()            
                        
        if (successful == True) and (abort == False):
            print(Fore.GREEN + 'SUCCESS.')
            print(Style.RESET_ALL)
            
        else:
            print(Fore.RED + 'FAILED TO REACH TARGET.')
            print(Style.RESET_ALL)
            
        if (successful):
            episode_successful.append(episode_index)
            
            if (plotting and successful): 
                plot_measurements(sinr_progress, sinr_ue2_progress, serving_tx_power_progress, interfering_tx_power_progress, max_timesteps_per_episode, episode_index, episode_index)
                plot_actions(actions, max_timesteps_per_episode, episode_index, episode_index)

                             
            print('First successful episode:')
            print(episode_successful)
            optimal = 'Episode {}/{} generated the highest reward {}.'.format(max_episode, MAX_EPISODES, max_reward)
            print(optimal)
            
            # Store all values in files
            filename = 'figures/optimal.txt'
            file = open(filename, 'w')
            file.write(optimal)    
            file.close()
            
            #break

    if (len(episode_successful) == 0):
        print("Goal cannot be reached after {} episodes.  Try to increase maximum episodes.".format(max_episodes_to_run))

    
def plot_performance_function_deep(values, episode_count, is_loss=False):
    return
    print(values)
    title = r'\bf Average $Q$' if not is_loss else r'\bf Episode Loss' 
    y_label = 'Expected Action-Value $Q$' if not is_loss else r'\bf Expected Loss' 
    filename = 'q_function' if not is_loss else 'losses'
    fig = plt.figure(figsize=(8,5))
        
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.size'] = 16
    matplotlib.rcParams['text.latex.preamble'] = [
        r'\usepackage{amsmath}',
        r'\usepackage{amssymb}']    

    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel(y_label)
   # plt.step(episodes, values, color='k')
   
    plt.plot(1 + np.arange(episode_count), values, linestyle='-', color='k')
    # plt.plot(values, linestyle='-', color='k')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('figures/{}_{}.pdf'.format(filename, episode_count), format="pdf")
    
    # Store all values in files
    file = open('figures/{}_{}.txt'.format(filename, episode_count), 'w')
    file.write('values: {}'.format(values))
    file.close()
    
    plt.show(block=True)
    plt.close(fig)
    
def plot_measurements(sinr_progress, sinr_ue2_progress, serving_tx_power_progress, interfering_tx_power_progress, max_timesteps_per_episode, episode_index, max_episodes_to_run):
    # Do some nice plotting here
#    fig = plt.figure(figsize=(8,5))
#    
#    plt.rc('text', usetex=True)
#    plt.rc('font', family='serif')
#    matplotlib.rcParams['text.usetex'] = True
#    matplotlib.rcParams['font.size'] = 16
#    matplotlib.rcParams['text.latex.preamble'] = [
#        r'\usepackage{amsmath}',
#        r'\usepackage{amssymb}']
#    
#    plt.xlabel('Transmit Time Interval (1 ms)')
#    
#    # Only integers                                
#    ax = fig.gca()
#    ax.xaxis.set_major_formatter(tick.FormatStrFormatter('%0g'))
#    ax.xaxis.set_ticks(np.arange(0, max_timesteps_per_episode))
#    
#    ax_sec = ax.twinx()
#    
#    ax.set_autoscaley_on(False)
#    ax_sec.set_autoscaley_on(False)
#    
#    ax.plot(sinr_progress, marker='o', color='b', label='SINR for UE 0')
#    ax.plot(sinr_ue2_progress, marker='*', color='m', label='SINR for UE 2')
#    ax_sec.plot(serving_tx_power_progress, linestyle='--', color='k', label='Serving BS')
#    ax_sec.plot(interfering_tx_power_progress, linestyle='--', color='c', label='Interfering BS')
#    
#    ax.set_xlim(xmin=0, xmax=max_timesteps_per_episode - 1)
#    
#    ax.axhline(y=env.min_sinr, xmin=0, color="red", linewidth=1.5, label='SINR min')
#    ax.axhline(y=env.sinr_target, xmin=0, color="green",  linewidth=1.5, label='SINR target')
#    ax.set_ylabel('DL Received SINR (dB)')
#    ax_sec.set_ylabel('BS Transmit Power (dBm)')
#    
#    max_sinr = max(max(sinr_progress), max(sinr_ue2_progress))
#    ax.set_ylim(env.min_sinr - 1, max_sinr + 1)
#    ax_sec.set_ylim(0, np.ceil(10*np.log10(1e3 * max(env.max_tx_power_interference, env.max_tx_power))))
#    
#    ax.legend(loc="lower left")
#    ax_sec.legend(loc='upper right')
#    
#   # plt.title('Episode {0} / {1} ($\epsilon = {2:0.3}$)'.format(episode_index, max_episodes_to_run, agent.exploration_rate))
#    plt.grid(True)
#    
    # Store all values in files
    filename = 'figures/measurements_{}_seed{}.txt'.format(episode_index, seed)
    file = open(filename, 'w')
    file.write('sinr_progress: {}'.format(sinr_progress))
    file.write('sinr_ue2_progress: {}'.format(sinr_ue2_progress))
    file.write('serving_tx_power: {}'.format(serving_tx_power_progress))
    file.write('interf_tx_power: {}'.format(interfering_tx_power_progress))
    file.close()
    
#    plt.tight_layout()
#    plt.savefig('figures/measurements_episode_{}.pdf'.format(episode_index), format="pdf")
##    plt.show(block=True)
#    plt.close(fig)
#    
def plot_actions(actions, max_timesteps_per_episode, episode_index, max_episodes_to_run):
    return
    fig = plt.figure(figsize=(8,5))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.xlabel('Transmit Time Interval (1 ms)')
    plt.ylabel('Action Transition')
    
     # Only integers                                
    ax = fig.gca()
    ax.xaxis.set_major_formatter(tick.FormatStrFormatter('%0g'))
    ax.xaxis.set_ticks(np.arange(0, max_timesteps_per_episode))
    
    ax.set_xlim(xmin=0, xmax=max_timesteps_per_episode-1)
    
    ax.set_autoscaley_on(False)
    ax.set_ylim(-1,env.num_actions)
    
    #plt.title('Episode {0} / {1} ($\epsilon = {2:0.3}$)'.format(episode_index, max_episodes_to_run, agent.exploration_rate))
    plt.grid(True)
    
    # Store all values in files
    filename = 'figures/actions_{}.txt'.format(episode_index)
    file = open(filename, 'w')
    file.write('actions: {}'.format(actions))    
    file.close()
    
    plt.step(np.arange(len(actions)), actions, color='b', label='Actions')
    plt.tight_layout()
    plt.savefig('figures/actions_episode_{}.pdf'.format(episode_index), format="pdf")
#    plt.show(block=True)
    plt.close(fig)
    return

########################################################################################
    
radio_frame = 20
seeds = np.arange(100).tolist()

for seed in seeds:

    random.seed(seed)
    np.random.seed(seed)
 
    env = radio_environment(seed=seed)
    agent = QLearner(seed=seed)

#    run_agent_fpa(env)
#    run_agent_tabular(env)
    run_agent_deep(env)

########################################################################################
