#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 19:38:03 2018

@author: farismismar
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc

import scipy.special
import matplotlib.ticker as tick

import os

import matplotlib2tikz
import pandas as pd

os.chdir('/Users/farismismar/Desktop/voice')

def generate_ccdf(data1, data2, data3):
    fig = plt.figure(figsize=(10.24,7.68))
    
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.size'] = 20
    matplotlib.rcParams['text.latex.preamble'] = [
        r'\usepackage{amsmath}',
        r'\usepackage{amssymb}']
    
    num_bins = 50
    for data in [data1, data2, data3]:
       
        data_ = data

        counts, bin_edges = np.histogram(data_, bins=num_bins, density=True)
        ccdf = 1 - np.cumsum(counts) / counts.sum()
        ccdf = np.insert(ccdf, 0, 1)
        bin_edges = np.insert(bin_edges[1:], 0, bin_edges[0] - (bin_edges[2] - bin_edges[1]))
        ax = fig.gca()
        ax.plot(bin_edges, ccdf)

    labels = ['Tabular $Q$-learning', 'Deep $Q$-learning (proposed)', 'Fixed Power Allocation (FPA)']
    ax.set_xlabel('$\gamma$')
    ax.set_ylabel('$1 - F_\Gamma(\gamma)$')        
    ax.set_ylim([0,1])
    ax.legend(labels, loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('figures/voice_ccdf.pdf', format="pdf")
    matplotlib2tikz.save('figures/voice_ccdf.tikz')
    plt.close(fig)   


def _____generate_ccdf_(data1, data2, data3):
    fig = plt.figure(figsize=(10.24,7.68))
    
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.size'] = 20
    matplotlib.rcParams['text.latex.preamble'] = [
        r'\usepackage{amsmath}',
        r'\usepackage{amssymb}']
    
    for data in [data1, data2, data3]:
        # sort the data:
        data_sorted = np.sort(data)
        
        # calculate the proportional values of samples
        p = 1. * np.arange(len(data)) / (len(data) - 1)
        
        ax = fig.gca()
    
        # plot the sorted data:
        ax.plot(data_sorted, 1 - p)
    

    labels = ['Tabular $Q$-learning', 'Deep $Q$-learning (proposed)', 'Fixed Power Allocation (FPA)']
    ax.set_xlabel('$\gamma$')
    ax.set_ylabel('$1 - F_\Gamma(\gamma)$')        
    ax.legend(labels, loc="lower left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('figures/voice_ccdf.pdf', format="pdf")
    matplotlib2tikz.save('figures/voice_ccdf.tikz')
    plt.close(fig)    

def read_output(filename):
    df = pd.read_csv(filename)
    df = df.T
    df = df.reset_index()
    df.drop(df.index[-1], axis=0, inplace=True)
    df = df.astype(float)

    return list(df.iloc[:,0])

def main():
        
    # TODO, put all reported SINRs for both UEs in a vector
    ue_deep_ue1 = read_output('figures_deep/ue_1_sinr.txt')
    ue_tabular_ue1 = read_output('figures_tabular/ue_1_sinr.txt')
    ue_fpa_ue1 = read_output('figures_fpa/ue_1_sinr.txt')
    
    ue_deep_ue2 = read_output('figures_deep/ue_2_sinr.txt')
    ue_tabular_ue2 = read_output('figures_tabular/ue_2_sinr.txt')
    ue_fpa_ue2 = read_output('figures_fpa/ue_2_sinr.txt')
    
    ue_deep = np.array(ue_deep_ue1+ue_deep_ue2)
    ue_tabular = np.array(ue_tabular_ue1+ue_tabular_ue2)
    ue_fpa = np.array(ue_fpa_ue1+ue_fpa_ue2)

    generate_ccdf(ue_tabular, ue_deep, ue_fpa)
    
if __name__ == '__main__':
    main()
    


