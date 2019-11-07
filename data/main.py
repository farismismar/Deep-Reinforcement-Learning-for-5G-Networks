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
from keras import backend as K

import random
import numpy as np
from colorama import Fore, Back, Style

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as tick

from environment import radio_environment
from DQNLearningAgent import DQNLearningAgent as QLearner # Deep with GPU and CPU fallback

MAX_EPISODES_DEEP = 2500
MAX_EPISODES_OPTIMAL = 2500

# Succ: 

os.chdir('/Users/farismismar/Desktop/deep')

    
##    ##    ##    ##    ##    ##    ##    ##    ##    ##    ##    ##    ##    ##    ##    ##    ##    ##    ##    ##    ##

def run_agent_optimal(env, plotting=True):
    max_episodes_to_run = MAX_EPISODES_OPTIMAL
    max_timesteps_per_episode = radio_frame
                    
    print('Ep.         | TS | Recv. SINR (srv) | Recv. SINR (int) | Serv. Tx Pwr | Int. Tx Pwr')
    print('--'*54)
    
    # Implement the Q-learning algorithm
    for episode_index in 1 + np.arange(max_episodes_to_run):
        observation = env.reset()
      #  observation = np.reshape(observation, [1, agent._state_size])
        
        (_, _, _, _, pt_serving, pt_interferer, _, _) = observation
        action = -1
                
        sinr_progress = [] # needed for the SINR based on the episode.
        sinr_ue2_progress = [] # needed for the SINR based on the episode.
        serving_tx_power_progress = []
        interfering_tx_power_progress = []

        # Let us know how we did.
        for timestep_index in 1 + np.arange(max_timesteps_per_episode):
            # Take a step
            next_observation, reward, done, abort = env.step(action)
            (x_ue_1, y_ue_1, x_ue_2, y_ue_2, pt_serving, pt_interferer, _, _) = next_observation            
            
            # This environment step above will not do anything, besides moving the UEs around
            # so here is the work:
            
            # Exhaustive search across all sets...
#            pt_serving_orig = pt_serving
 #           pt_interferer_orig = pt_interferer
            max_sinr = -np.inf
            max_sinr_ue1_sinr = -np.inf
            max_sinr_ue2_sinr = -np.inf
            
            for bs1_index in np.arange(env.M_ULA*env.k_oversample):
                env.f_n_bs1 = bs1_index
                for bs2_index in np.arange(env.M_ULA*env.k_oversample):
                    env.f_n_bs2 = bs2_index
                    for pc_1 in [-1, 1]:
                        pt_serving_candidate = pt_serving * 10 ** (pc_1/10.)
                        for pc_2 in [-1, 1]:
                            pt_int_candidate = pt_interferer * 10 ** (pc_2/10.)
                            received_power, interference_power, received_sinr = env._compute_rf(x_ue_1, y_ue_1, pt_serving_candidate, pt_int_candidate, is_ue_2=False)
                            received_power_ue2, interference_power_ue2, received_ue2_sinr = env._compute_rf(x_ue_2, y_ue_2, pt_serving_candidate, pt_int_candidate, is_ue_2=True)
                            if (received_sinr + received_ue2_sinr > max_sinr): # dB
                                max_sinr = received_sinr + received_ue2_sinr
                                max_sinr_ue1_sinr = received_sinr
                                max_sinr_ue2_sinr = received_ue2_sinr
#                                print('Current max SINR product is: {:.2f} dB'.format(max_sinr))
                          #      env.received_sinr_dB = received_sinr
                          #      env.received_ue2_sinr_dB = received_ue2_sinr
                                pt_serving = pt_serving_candidate
                                pt_interferer = pt_int_candidate

            # solution is found.  Use it
            received_sinr = max_sinr_ue1_sinr
            received_ue2_sinr = max_sinr_ue2_sinr
            
            # Let us know how we did.
            print('{}/{} | {} | {:.2f} dB | {:.2f} dB | {:.2f} W | {:.2f} W'.format(episode_index, max_episodes_to_run, 
                                                                                      timestep_index,
                                                                                      received_sinr, received_ue2_sinr,
                                                                                      pt_serving, pt_interferer))

            abort = (pt_serving > env.max_tx_power) or (pt_interferer > env.max_tx_power_interference) or (received_sinr < env.min_sinr) or (received_ue2_sinr < env.min_sinr) \
                or (received_sinr > 70) or (received_ue2_sinr > 70) or (max_sinr < 0) #or (received_sinr < 10) or (received_ue2_sinr < 10) # an extra condition

            if abort:
                break

            # Record all in dB/dBm
            sinr_progress.append(received_sinr)
            sinr_ue2_progress.append(received_ue2_sinr)
            serving_tx_power_progress.append(10*np.log10(pt_serving*1e3))
            interfering_tx_power_progress.append(10*np.log10(pt_interferer*1e3))

            done = (pt_serving <= env.max_tx_power) and (pt_serving >= 0) and (pt_interferer <= env.max_tx_power_interference) and (pt_interferer >= 0) and \
                (received_sinr >= env.min_sinr) and (received_ue2_sinr >= env.min_sinr) and (received_sinr >= env.sinr_target) and (received_ue2_sinr >= env.sinr_target)

        successful = (done == True) and (timestep_index > max_timesteps_per_episode - 1)
        if successful:
            print(Fore.GREEN + 'SUCCESS.')
            print(Style.RESET_ALL)
        else:
            print(Fore.RED + 'FAILED TO REACH TARGET.')
            print(Style.RESET_ALL)
        
        # Plot the episode...
        if (plotting and successful): 
            plot_measurements(sinr_progress, sinr_ue2_progress, serving_tx_power_progress, interfering_tx_power_progress, max_timesteps_per_episode, episode_index, episode_index)

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
      
        (_, _, _, _, pt_serving, pt_interferer, _, _) = observation

        action = agent.begin_episode(observation)        
        # Let us know how we did.
        print('{}/{} | {:.2f} | {} | {:.2f} dB | {:.2f} dB | {:.2f} W | {:.2f} W | {:.2f} | {} '.format(episode_index, max_episodes_to_run, 
                                                                                      agent.exploration_rate,
                                                                                      0, 
                                                                                      np.nan,
                                                                                      np.nan,
                                                                                      pt_serving, pt_interferer, 
                                                                                      0, action))   


        total_reward = 0
        done = False
        actions = [action]
        
        sinr_progress = [] # needed for the SINR based on the episode.
        sinr_ue2_progress = [] # needed for the SINR based on the episode.
        serving_tx_power_progress = []
        interfering_tx_power_progress = []
        
        episode_loss = []
        episode_q = []

        for timestep_index in 1 + np.arange(max_timesteps_per_episode):
            # Take a step
            next_observation, reward, done, abort = env.step(action)
            (_, _, _, _, pt_serving, pt_interferer, _, _) = next_observation
                        
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
                            
            successful = done and (total_reward > 0) and (abort == False)
            
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
            
            # Update for the next time step
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

            optimal = 'Episode {}/{} generated the highest reward {}.'.format(max_episode, MAX_EPISODES_DEEP, max_reward)
            print(optimal)
                            
            # Store all values in files
            filename = 'figures/optimal_{}.txt'.format(max_episode)
            file = open(filename, 'w')
            file.write(optimal)    
            file.close()

            
        #        print('SINR progress')
        #        print(sinr_progress)
        #        print('Serving BS transmit power progress')
        #        print(serving_tx_power_progress)
        #        print('Interfering BS transmit power progress')
        #        print(interfering_tx_power_progress)
                        
# Note.. remove the break on line 199
# for looping against all episodes
        
            # Plot the episode...
            if (plotting): 
                plot_measurements(sinr_progress, sinr_ue2_progress, serving_tx_power_progress, interfering_tx_power_progress, max_timesteps_per_episode, episode_index, episode_index)
                plot_actions(actions, max_timesteps_per_episode, episode_index, episode_index)
          
            plot_performance_function_deep(losses, episode_index, is_loss=True)
            plot_performance_function_deep(Q_values, episode_index, is_loss=False)
            
        #if (episode_index > 100*env.M_ULA):
         #   break
                 
    agent.save('deep_rl.model')

    if (len(episode_successful) == 0):
        print("Goal cannot be reached after {} episodes.  Try to increase maximum episodes.".format(max_episodes_to_run))

    
def plot_performance_function_deep(values, episode_count, is_loss=False):
    return
#    print(values)
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
    plt.savefig('figures/{}_{}_seed{}.pdf'.format(filename, episode_count,seed), format="pdf")
    
    # Store all values in files
    file = open('figures/{}_{}_seed{}.txt'.format(filename, episode_count,seed), 'w')
    file.write('values: {}'.format(values))
    file.close()
    
#    plt.show(block=True)
    plt.close(fig)
    
def plot_measurements(sinr_progress, sinr_ue2_progress, serving_tx_power_progress, interfering_tx_power_progress, max_timesteps_per_episode, episode_index, max_episodes_to_run):
    # Do some nice plotting here
    #ig = plt.figure(figsize=(8,5))
    
    #plt.rc('text', usetex=True)
    #plt.rc('font', family='serif')
    #matplotlib.rcParams['text.usetex'] = True
    #matplotlib.rcParams['font.size'] = 16
    #matplotlib.rcParams['text.latex.preamble'] = [
     #   r'\usepackage{amsmath}',
    #    r'\usepackage{amssymb}']
    
    #plt.xlabel('Transmit Time Interval (1 ms)')
    
    # Only integers                                
    #ax = fig.gca()
    #ax.xaxis.set_major_formatter(tick.FormatStrFormatter('%0g'))
    #ax.xaxis.set_ticks(np.arange(0, max_timesteps_per_episode))
    
    #ax_sec = ax.twinx()
    
    #ax.set_autoscaley_on(False)
    #ax_sec.set_autoscaley_on(False)
    
    #ax.plot(sinr_progress, marker='o', color='b', label='SINR for UE 0')
    #ax.plot(sinr_ue2_progress, marker='*', color='m', label='SINR for UE 2')
    #ax_sec.plot(serving_tx_power_progress, linestyle='--', color='k', label='Serving BS')
    #ax_sec.plot(interfering_tx_power_progress, linestyle='--', color='c', label='Interfering BS')
    
   # ax.set_xlim(xmin=0, xmax=max_timesteps_per_episode - 1)
    
    #ax.axhline(y=env.min_sinr, xmin=0, color="red", linewidth=1.5, label='SINR min')
#    ax.axhline(y=env.sinr_target, xmin=0, color="green",  linewidth=1.5, label='SINR target')
    #ax.set_ylabel('DL Received SINR (dB)')
    #ax_sec.set_ylabel('BS Transmit Power (dBm)')
    
    #max_sinr = max(max(sinr_progress), max(sinr_ue2_progress))
    #ax.set_ylim(env.min_sinr - 1, max_sinr + 1)
    #ax_sec.set_ylim(0, np.ceil(10*np.log10(1e3 * max(env.max_tx_power_interference, env.max_tx_power))))
    
    #ax.legend(loc="lower left")
    #ax_sec.legend(loc='upper right')
    
   # plt.title('Episode {0} / {1} ($\epsilon = {2:0.3}$)'.format(episode_index, max_episodes_to_run, agent.exploration_rate))
   # plt.grid(True)
   # plt.tight_layout()
    
    # Store all values in files
    filename = 'figures/measurements_{}_seed{}.txt'.format(episode_index,seed)
    file = open(filename, 'w')
    file.write('sinr_progress: {}'.format(sinr_progress))
    file.write('sinr_ue2_progress: {}'.format(sinr_ue2_progress))
    file.write('serving_tx_power: {}'.format(serving_tx_power_progress))
    file.write('interf_tx_power: {}'.format(interfering_tx_power_progress))
    file.close()
    
    #plt.savefig('figures/measurements_episode_{}_seed{}.pdf'.format(episode_index,seed), format="pdf")
#    plt.show(block=True)
    #plt.close(fig)
    
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
    plt.tight_layout()
    
    # Store all values in files
    filename = 'figures/actions_{}.txt'.format(episode_index)
    file = open(filename, 'w')
    file.write('actions: {}'.format(actions))    
    file.close()
    
    plt.step(np.arange(len(actions)), actions, color='b', label='Actions')
    plt.savefig('figures/actions_episode_{}_seed{}.pdf'.format(episode_index, seed), format="pdf")
#    plt.show(block=True)
    plt.close(fig)
    return
    
########################################################################################

radio_frame = 10
#seeds = np.arange(50).astype(int).tolist() 

seeds = [0] # for the optimal case.

for seed in seeds:
    print('Now running seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
 
    env = radio_environment(seed=seed)

    run_agent_optimal(env)

    #agent = QLearner(seed=seed) # only for the deep
    #run_agent_deep(env)
#    K.clear_session() # free up GPU memory
 #   del agent  

########################################################################################
