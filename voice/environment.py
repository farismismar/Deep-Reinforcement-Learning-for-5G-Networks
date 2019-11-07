#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 11:34:42 2019
@author: farismismar
"""

import math
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from numpy import linalg as LA

# An attempt to follow
# https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py

# Environment parameters
# cell radius
# UE movement speed
# BS max tx power
# BS antenna
# UE noise figure
# Center frequency
# Transmit antenna isotropic gain
# Antenna heights
# Shadow fading margin
# Number of ULA antenna elements
# Oversampling factor

class radio_environment:
    '''    
        Observation: 
            Type: Box(6)
            Num Observation                                    Min      Max
            0   User1 server X                                 -r       r
            1   User1 server Y                                 -r       r
            2   User2 server X                                 isd-r    isd+r
            3   User2 server Y                                 -r       r
            4   Serving BS Power                               5        40W
            5   Neighbor BS power                              5        40W
            
        Actions:
            Type: Discrete(8)
            Num	Action
            *	Binary mapping IC[3] IC[2] PC[1] PC[0]
            *   00 = -3, 01 = -1, 10 = 1, 11 = 3.
            
    '''     
    def __init__(self, seed):
        self.cell_radius = 350 # in meters.
        self.inter_site_distance = 3 * self.cell_radius / 2.
        self.num_users = 30 # number of users.
        self.sinr_target = 0 # in dB
        self.min_sinr = -3 # in dB
        self.max_sinr = 15 # in dB
        self.max_tx_power = 40 # in Watts
        self.max_tx_power_interference = 40 # in Watts
        self.f_c = 2.1e9 # Hz
        self.G_ant_no_beamforming = 11 # dBi
        self.prob_LOS = 0.9 # Probability of LOS transmission

        self.num_actions = 16
        
        # Where are the base stations?
        self.x_bs_1, self.y_bs_1 = 0, 0
        self.x_bs_2, self.y_bs_2 = self.inter_site_distance, 0
        
        # for Beamforming
        self.use_beamforming = False
        self.M_ULA = 1
        self.k_oversample = 1 # oversampling factor
        self.Np = 15
        self.F = np.zeros([self.M_ULA, self.k_oversample*self.M_ULA], dtype=complex)
        self.theta_n = math.pi * np.arange(start=0., stop=1., step=1./(self.k_oversample*self.M_ULA))
        # Beamforming codebook F
        for n in np.arange(self.k_oversample*self.M_ULA):
            f_n = self._compute_bf_vector(self.theta_n[n])
            self.F[:,n] = f_n
        self.f_n_bs1 = None  # The index in the codebook for serving BS
        self.f_n_bs2 = None  # The index in the codebook for interfering BS

        # for Reinforcement Learning
        self.reward_min = -20
        self.reward_max = 100
        
        bounds_lower = np.array([
            -self.cell_radius,
            -self.cell_radius,
            self.inter_site_distance-self.cell_radius,
            -self.cell_radius,
            1,
            1
            ])

        bounds_upper = np.array([
            self.cell_radius,
            self.cell_radius,
            self.inter_site_distance+self.cell_radius,
            self.cell_radius,
            self.max_tx_power,
            self.max_tx_power_interference
            ])

        self.action_space = spaces.Discrete(self.num_actions) # action size is here
        self.observation_space = spaces.Box(bounds_lower, bounds_upper, dtype=np.float32) # spaces.Discrete(2) # state size is here 
        
        self.seed(seed=seed)
        
        self.state = None
#        self.steps_beyond_done = None
        self.received_sinr_dB = None
        self.serving_transmit_power_dB = None
        self.interfering_transmit_power_dB = None
      
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    def reset(self):
        # Initialize f_n of both cells
        self.f_n_bs1 = self.np_random.randint(self.M_ULA)
        self.f_n_bs2 = self.np_random.randint(self.M_ULA)
        
        self.state = [self.np_random.uniform(low=-self.cell_radius, high=self.cell_radius),
                      self.np_random.uniform(low=-self.cell_radius, high=self.cell_radius),
                      self.np_random.uniform(low=self.inter_site_distance-self.cell_radius, high=self.inter_site_distance+self.cell_radius),
                      self.np_random.uniform(low=-self.cell_radius, high=self.cell_radius),
                      self.np_random.uniform(low=1, high=self.max_tx_power/2),
                      self.np_random.uniform(low=1, high=self.max_tx_power_interference/2)
                      ]
        
#        self.steps_beyond_done = None
        

        return np.array(self.state)
    
    def step(self, action):
#        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        reward = 0
        x_ue_1, y_ue_1, x_ue_2, y_ue_2, pt_serving, pt_interferer = state
        P = [-3, -1, 1, 3]
        
        if (action == -1):
            reward += 0 # do nothing.. this is for FPA.
        else:
            IC = (action & 0b1100) >> 2
            PC = (action & 0b0011)
            
            IC = P[IC]
            PC = P[PC]
            
            reward = P[PC] - P[IC]
            pt_serving *= 10**(PC/10.)
            pt_interferer *= 10**(IC/10.)
        
            
        if (action > self.num_actions - 1):
            print('WARNING: Invalid action played!')
            reward = 0
            return [], 0, False, True
        
        # move the UEs at a speed of v, in a random direction
        v = 2 # km/h.

        v *= 5./18 # in m/sec
        theta_1, theta_2 = self.np_random.uniform(low=-math.pi, high=math.pi, size=2)
        
        dx_1 = v * math.cos(theta_1)
        dy_1 = v * math.sin(theta_1)

        dx_2 = v * math.cos(theta_2)
        dy_2 = v * math.sin(theta_2)
        
        # Move UE 1
        x_ue_1 += dx_1
        y_ue_1 += dy_1
        
        # Move UE 2
        x_ue_2 += dx_2
        y_ue_2 += dy_2
    
        received_power, interference_power, received_sinr = self._compute_rf(x_ue_1, y_ue_1, pt_serving, pt_interferer, is_ue_2=False)
        received_power_ue2, interference_power_ue2, received_ue2_sinr = self._compute_rf(x_ue_2, y_ue_2, pt_serving, pt_interferer, is_ue_2=True)
        
        # Now provide an SINR gain based on the code rate...         
        received_sinr = self.effective_sinr(received_sinr)
        received_ue2_sinr = self.effective_sinr(received_ue2_sinr)

        # keep track of quantities...
        self.received_sinr_dB = received_sinr 
        self.received_ue2_sinr_dB = received_ue2_sinr
        self.serving_transmit_power_dBm = 10*np.log10(pt_serving*1e3)
        self.interfering_transmit_power_dBm = 10*np.log10(pt_interferer*1e3)

        # Did we find a FEASIBLE NON-DEGENERATE solution?
        done = (received_sinr >= self.sinr_target) and (pt_serving <= self.max_tx_power) and (pt_serving >= 0) and (pt_interferer <= self.max_tx_power_interference) and (pt_interferer >= 0) and (received_ue2_sinr >= self.sinr_target) \
        		and  (received_sinr <= self.max_sinr) and (received_ue2_sinr <= self.max_sinr) 
                
        abort = (pt_serving > self.max_tx_power) or (pt_interferer > self.max_tx_power_interference) or (received_sinr < self.min_sinr) or (received_ue2_sinr < self.min_sinr) \
                or (received_sinr > self.max_sinr) or (received_ue2_sinr > self.max_sinr) or (received_sinr < self.sinr_target) or (received_ue2_sinr < self.sinr_target) \
                or (np.isnan(self.received_sinr_dB)) or (np.isnan(self.received_ue2_sinr_dB)) 
                
#        print('{:.2f} dB | {:.2f} dB | {:.2f} W | {:.2f} W '.format(received_sinr, received_ue2_sinr, pt_serving, pt_interferer), end='')
#        print('Done: {}'.format(done))
#        print('UE moved to ({0:0.3f},{1:0.3f}) and their received SINR became {2:0.3f} dB.'.format(x,y,received_sinr))
        
        # Update the state.
        self.state = (x_ue_1, y_ue_1, x_ue_2, y_ue_2, pt_serving, pt_interferer) #, f_n_bs1, f_n_bs2)
     
        if done == True:
            abort = False
            reward += self.reward_max
        elif abort:
            reward = self.reward_min

#        print(done, (received_sinr >= self.sinr_target) , (pt_serving <= self.max_tx_power) , (pt_serving >= 0) , \
#                (pt_interferer <= self.max_tx_power_interference) , (pt_interferer >= 0) , (received_ue2_sinr >= self.sinr_target))
        
        # True True True True True True
        # 2352/40000 | 9 | 4.47 dB | 4.43 dB | 5.67 W | 8.22 W | -3.00 | ABORTED.
        return np.array(self.state), reward, done, abort

    def _compute_bf_vector(self, theta):
        c = 3e8 # speed of light
        wavelength = c / self.f_c
        
        d = wavelength / 2. # antenna spacing 
        k = 2. * math.pi / wavelength
    
        exponent = 1j * k * d * math.cos(theta) * np.arange(self.M_ULA)
        
        f = 1. / math.sqrt(self.M_ULA) * np.exp(exponent)
        
        # Test the norm square... is it equal to unity? YES.
    #    norm_f_sq = LA.norm(f, ord=2) ** 2
     #   print(norm_f_sq)
    
        return f

    def _compute_channel(self, x_ue, y_ue, x_bs, y_bs):
        # Np is the number of paths p
        PLE_L = 2
        PLE_N = 4
        G_ant = 3 # dBi for beamforming mmWave antennas
        
        # Override the antenna gain if no beamforming
        if self.use_beamforming == False:
            G_ant = self.G_ant_no_beamforming
            
        # theta is the steering angle.  Sampled iid from unif(0,pi).
        theta = np.random.uniform(low=0, high=math.pi, size=self.Np)
    
        is_mmWave = (self.f_c > 25e9)
        
        if is_mmWave:
            path_loss_LOS = 10 ** (self._path_loss_mmWave(x_ue, y_ue, PLE_L, x_bs, y_bs) / 10.)
            path_loss_NLOS = 10 ** (self._path_loss_mmWave(x_ue, y_ue, PLE_N, x_bs, y_bs) / 10.)
        else:
            path_loss_LOS = 10 ** (self._path_loss_sub6(x_ue, y_ue, x_bs, y_bs) / 10.)
            path_loss_NLOS = 10 ** (self._path_loss_sub6(x_ue, y_ue, x_bs, y_bs) / 10.)
            
        # Bernoulli for p
        alpha = np.zeros(self.Np, dtype=complex)
        p = np.random.binomial(1, self.prob_LOS)
        
        if (p == 1):
            self.Np = 1
            alpha[0] = 1. / math.sqrt(path_loss_LOS)
        else:
            ## just changed alpha to be complex in the case of NLOS
            alpha = (np.random.normal(size=self.Np) + 1j * np.random.normal(size=self.Np)) / math.sqrt(path_loss_NLOS)
                
        rho = 1. * 10 ** (G_ant / 10.)
        
        # initialize the channel as a complex variable.
        h = np.zeros(self.M_ULA, dtype=complex)
        
        for p in np.arange(self.Np):
            a_theta = self._compute_bf_vector(theta[p])
            h += alpha[p] / rho * a_theta.T # scalar multiplication into a vector
        
        h *= math.sqrt(self.M_ULA)
        
#        print ('Warning: channel gain is {} dB.'.format(10*np.log10(LA.norm(h, ord=2))))
        return h

    def _compute_rf(self, x_ue, y_ue, pt_bs1, pt_bs2, is_ue_2=False):
        T = 290 # Kelvins
        B = 15000 # Hz
        k_Boltzmann = 1.38e-23
        eps = 1e-20
        
        noise_power = k_Boltzmann*T*B # this is in Watts

        if is_ue_2 == False:
            # Without loss of generality, the base station is at the origin
            # The interfering base station is x = cell_radius, y = 0
            x_bs_1, y_bs_1 = self.x_bs_1, self.y_bs_1
            x_bs_2, y_bs_2 = self.x_bs_2, self.y_bs_2

            # Now the channel h, which is a vector in beamforming.
            # This computes the channel for user in serving BS from the serving BS.
            h_1 = self._compute_channel(x_ue, y_ue, x_bs=x_bs_1, y_bs=y_bs_1) 
            # This computes the channel for user in serving BS from the interfering BS.
            h_2 = self._compute_channel(x_ue, y_ue, x_bs=x_bs_2, y_bs=y_bs_2)
              
            # if this is not beamforming, there is no precoder:
            if (self.use_beamforming):
                received_power = pt_bs1 * abs(np.dot(h_1.conj(), self.F[:, self.f_n_bs1])) ** 2
                interference_power = pt_bs2 * abs(np.dot(h_2.conj(), self.F[:, self.f_n_bs2])) ** 2
            else: # the gain is ||h||^2
                received_power = pt_bs1 * LA.norm(h_1, ord=2) ** 2
                interference_power = pt_bs2 * LA.norm(h_2, ord=2) ** 2
        else:
            x_bs_1, y_bs_1 = self.x_bs_1, self.y_bs_1
            x_bs_2, y_bs_2 = self.x_bs_2, self.y_bs_2            
            
            # Now the channel h, which is a vector in beamforming.
            # This computes the channel for user in serving BS from the serving BS.
            h_1 = self._compute_channel(x_ue, y_ue, x_bs=x_bs_2, y_bs=y_bs_2) 
            # This computes the channel for user in serving BS from the interfering BS.
            h_2 = self._compute_channel(x_ue, y_ue, x_bs=x_bs_1, y_bs=y_bs_1) 

            # if this is not beamforming, there is no precoder:
            if (self.use_beamforming):
                received_power = pt_bs2 * abs(np.dot(h_1.conj(), self.F[:, self.f_n_bs2])) ** 2
                interference_power = pt_bs1 * abs(np.dot(h_2.conj(), self.F[:, self.f_n_bs1])) ** 2
            else: # the gain is ||h||^2
                received_power = pt_bs2 * LA.norm(h_1, ord=2) ** 2
                interference_power = pt_bs1 * LA.norm(h_2, ord=2) ** 2
                
        interference_plus_noise_power = interference_power + noise_power + eps
        received_sinr = 10*np.log10(received_power / interference_plus_noise_power)

        return [received_power, interference_power, received_sinr]
    
    # https://ieeexplore-ieee-org.ezproxy.lib.utexas.edu/stamp/stamp.jsp?tp=&arnumber=7522613
    def _path_loss_mmWave(self, x, y, PLE, x_bs=0, y_bs=0):
        # These are the parameters for f = 28000 MHz.
        c = 3e8 # speed of light
        wavelength = c / self.f_c
        A = 0.0671
        Nr = self.M_ULA
        sigma_sf = 9.1
        #PLE = 3.812
        
        d = math.sqrt((x - x_bs)**2 + (y - y_bs)**2) # in meters
        
        fspl = 10 * np.log10(((4*math.pi*d) / wavelength) ** 2)
        pl = fspl + 10 * np.log10(d ** PLE) * (1 - A*np.log2(Nr))
    
        chi_sigma = np.random.normal(0, sigma_sf) # log-normal shadowing 
        L = pl + chi_sigma
    
        return L # in dB    
        
    def _path_loss_sub6(self, x, y, x_bs=0, y_bs=0):
        f_c = self.f_c
        c = 3e8 # speed of light
        d = math.sqrt((x - x_bs)**2 + (y - y_bs)**2)
        h_B = 20
        h_R = 1.5

#        print('Distance from cell site is: {} km'.format(d/1000.))
        # FSPL
        L_fspl = -10*np.log10((4.*math.pi*c/f_c / d) ** 2)
        
        # COST231        
        C = 3
        a = (1.1 * np.log10(f_c/1e6) - 0.7)*h_R - (1.56*np.log10(f_c/1e6) - 0.8)
        L_cost231  = 46.3 + 33.9 * np.log10(f_c/1e6) + 13.82 * np.log10(h_B) - a + (44.9 - 6.55 * np.log10(h_B)) * np.log10(d/1000.) + C
    
        L = L_cost231
        
        return L # in dB

    def effective_sinr(self, sinr_value):
        # derive a code rate based on the sinr_value in dB
        code_rate = (sinr_value - self.min_sinr) / (self.max_sinr - self.min_sinr)
        
        if code_rate != 0:
            code_rate = -10 * np.log10(code_rate)
        
        code_rate = min(code_rate, 3) # should not exceed 3 dB 
                   
        return code_rate + sinr_value