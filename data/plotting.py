#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 2019

@author: farismismar
"""

import os
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
#import matplotlib.ticker as tick
#from matplotlib.ticker import MultipleLocator, FuncFormatter

import pandas as pd
import matplotlib2tikz

os.chdir('/Users/farismismar/Desktop/deep/')

MIN_EPISODES_DEEP = 0
MAX_EPISODES_DEEP = 2500

def sum_rate(sinr):
    output = []
    for s in sinr:
        s_linear = 10 ** (s / 10.)
        c = np.log2(1 + s_linear)
        output.append(c)

    return output    

def plot_ccdf(T, labels, filename='ccdf'):
    fig = plt.figure(figsize=(10.24, 7.68))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.size'] = 20
    matplotlib.rcParams['text.latex.preamble'] = [
        r'\usepackage{amsmath}',
        r'\usepackage{amssymb}']   
    
    num_bins = 10
    i = 0
    for data in T:
        data_ = T[data].dropna()

        counts, bin_edges = np.histogram(data_, bins=num_bins, density=True)
        ccdf = 1 - np.cumsum(counts) / counts.sum()
        ccdf = np.insert(ccdf, 0, 1)
        bin_edges = np.insert(bin_edges[1:], 0, bin_edges[0] - (bin_edges[2] - bin_edges[1]))
        lw = 1 + i
        ax = fig.gca()
        style = '-'
        ax.plot(bin_edges, ccdf, style, linewidth=lw)

    plt.grid(True)
    plt.tight_layout()
    ax.set_xlabel('$\gamma$')
    ax.set_ylabel('$1 - F_\Gamma(\gamma)$')        
    ax.legend(labels, loc="lower left")
    plt.savefig('figures/{}.pdf'.format(filename), format='pdf')
    matplotlib2tikz.save('figures/{}.tikz'.format(filename))
    plt.close(fig)    
    
def plot_primary(X,Y, xlabel, ylabel, ymin=0, ymax=MAX_EPISODES_DEEP, filename='plot.pdf'):
    fig = plt.figure(figsize=(10.24,7.68))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.size'] = 20
    matplotlib.rcParams['text.latex.preamble'] = [
        r'\usepackage{amsmath}',
        r'\usepackage{amssymb}']
    
    plt.xlabel(xlabel)
    
    #X = np.log2(np.array(X))
    
    ax = fig.gca()
    
    #ax.xaxis.set_major_locator(MultipleLocator(1))
    # Format the ticklabel to be 2 raised to the power of `x`
    #ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: int(2**x)))
    
    ax.set_autoscaley_on(False)
    
    plot_, = ax.plot(X, Y, 'k^-')  
    
    ax.set_ylabel(ylabel)
    ax.set_ylim(ymin, ymax)
    plt.grid(True)

    fig.tight_layout()
    plt.savefig('figures/{}'.format(filename), format='pdf')
    matplotlib2tikz.save('figures/{}.tikz'.format(filename))

##############################

def compute_distributions(optimal=False):
    df_final = pd.DataFrame()
    gamma_0 = 5 # dB as in the environment file.
    
    for M in [4, 8, 16, 32, 64]:
        if optimal:
            df_1 = pd.read_csv('figures M={} optimal/ue_1_sinr.txt'.format(M), sep=',', header=None, index_col=False).transpose().dropna()
            df_2 = pd.read_csv('figures M={} optimal/ue_2_sinr.txt'.format(M), sep=',', header=None, index_col=False).transpose().dropna()
        else:
            df_1 = pd.read_csv('figures M={}/ue_1_sinr.txt'.format(M), sep=',', header=None, index_col=False).transpose().dropna()
            df_2 = pd.read_csv('figures M={}/ue_2_sinr.txt'.format(M), sep=',', header=None, index_col=False).transpose().dropna()
        
        cutoff = gamma_0 + 10*np.log2(M)
        df = pd.concat([df_1, df_2], axis=0, ignore_index=True)
        sinr = df.astype(float)
        indexes = (sinr <= cutoff) & ~np.isnan(sinr)
        df_sinr = pd.DataFrame(sinr[indexes])
        df_sinr.columns = ['sinr_{}'.format(M)]
        
        if optimal:
            df_3 = pd.read_csv('figures M={} optimal/ue_1_power.txt'.format(M), sep=',', header=None, index_col=False).transpose().dropna()
            df_4 = pd.read_csv('figures M={} optimal/ue_2_power.txt'.format(M), sep=',', header=None, index_col=False).transpose().dropna()
        else:
            df_3 = pd.read_csv('figures M={}/ue_1_power.txt'.format(M), sep=',', header=None, index_col=False).transpose().dropna()
            df_4 = pd.read_csv('figures M={}/ue_2_power.txt'.format(M), sep=',', header=None, index_col=False).transpose().dropna()
        
        df = pd.concat([df_3, df_4], axis=0, ignore_index=True)
        df = df.astype(float)
        
        df_power = df.copy()
        df_power.columns = ['tx_power_{}'.format(M)]
    
        df_final = pd.concat([df_final, df_sinr, df_power], axis=1)
        
    return df_final

################################################################################################################################################################
def plot_secondary(X,Y1, Y2, Y3, Y4, xlabel, y1label, y2label, y1max, y2max, filename='plot.pdf'):
    fig = plt.figure(figsize=(10.24,7.68))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.size'] = 20
    matplotlib.rcParams['text.latex.preamble'] = [
        r'\usepackage{amsmath}',
        r'\usepackage{amssymb}']
    
    #plt.title(title)
    plt.xlabel(xlabel)
    plt.grid(True, axis='both', which='both')
        
    ax = fig.gca()
#    ax.xaxis.set_major_locator(MultipleLocator(1))
    # Format the ticklabel to be 2 raised to the power of `x`
 #   ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: int(2**x)))
    
    ax.set_autoscaley_on(False)
    ax_sec = ax.twinx()
    
    plot1_, = ax.plot(X, Y1, 'k^-')
    plot2_, = ax.plot(X, Y2, 'bo--')
    plot3_, = ax_sec.plot(X, Y3, 'r^-')
    plot4_, = ax_sec.plot(X, Y4, 'go--')

    ax.set_ylabel(y1label)
    ax_sec.set_ylabel(y2label)
    
    ax.set_ylim(0, y1max)
    ax_sec.set_ylim(0, y2max)
    
    plt.grid(True)
    plt.legend([plot1_, plot2_, plot3_, plot4_], ['TX Power JB-PCIC', 'TX Power Optimal', 'SINR JB-PCIC', 'SINR Optimal'], loc='lower right')
    fig.tight_layout()
    plt.savefig('figures/{}'.format(filename), format='pdf')
    matplotlib2tikz.save('figures/{}.tikz'.format(filename))
    
def plot_primary_two(X, Y1, Y2, xlabel, ylabel, filename='plot.pdf'):
    fig = plt.figure(figsize=(10.24,7.68))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.size'] = 20
    matplotlib.rcParams['text.latex.preamble'] = [
        r'\usepackage{amsmath}',
        r'\usepackage{amssymb}']
    
    plt.xlabel(xlabel)
    
    ax = fig.gca()
#    ax.xaxis.set_major_locator(MultipleLocator(1))
    # Format the ticklabel to be 2 raised to the power of `x`
 #   ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: int(2**x)))
    
    ax.set_autoscaley_on(False)
    
    plot1_, = ax.plot(X, Y1, 'k^-')
    plot2_, = ax.plot(X, Y2, 'ro--')

    ax.set_ylabel(ylabel)
    ax.set_ylim(min(min(Y1), min(Y2))*0.98, max(max(Y1), max(Y2))*1.02)
    
    plt.grid(True)
    plt.legend([plot1_, plot2_], ['JB-PCIC', 'Optimal'], loc='best')
    fig.tight_layout()
    plt.savefig('figures/{}'.format(filename), format='pdf')
    matplotlib2tikz.save('figures/{}.tikz'.format(filename))
    
##############################
df_final_ = compute_distributions()
df_final = df_final_.values
df_final = df_final.T
sinr_4, tx_power_4, sinr_8, tx_power_8, sinr_16, tx_power_16, sinr_32, tx_power_32, sinr_64, tx_power_64 = df_final 

sinr_4= sinr_4[~np.isnan(sinr_4)]
tx_power_4= tx_power_4[~np.isnan(tx_power_4)]

sinr_8= sinr_8[~np.isnan(sinr_8)]
tx_power_8= tx_power_8[~np.isnan(tx_power_8)]

sinr_16= sinr_16[~np.isnan(sinr_16)]
tx_power_16= tx_power_16[~np.isnan(tx_power_16)]

sinr_32= sinr_32[~np.isnan(sinr_32)]
tx_power_32= tx_power_32[~np.isnan(tx_power_32)]

sinr_64= sinr_64[~np.isnan(sinr_64)]
tx_power_64= tx_power_64[~np.isnan(tx_power_64)]

plot_ccdf(df_final_[['sinr_4', 'sinr_8', 'sinr_16', 'sinr_32', 'sinr_64']], [r'$M = 4$', r'$M  = 8$', r'$M = 16$', r'$M = 32$', r'$M = 64$'])
q = 100

tx_power_4_agg = 10*np.log10(10 ** (np.nanpercentile(tx_power_4, q)/10.))
tx_power_8_agg = 10*np.log10(10 ** (np.nanpercentile(tx_power_8, q)/10.))
tx_power_16_agg = 10*np.log10(10 ** (np.nanpercentile(tx_power_16, q)/10.))
tx_power_32_agg = 10*np.log10(10 ** (np.nanpercentile(tx_power_32, q)/10.))
tx_power_64_agg = 10*np.log10(10 ** (np.nanpercentile(tx_power_64, q)/10.))

sinr_avg_4 = 10*np.log10(10 ** (np.nanpercentile(sinr_4, q)/10.))
sinr_avg_8 = 10*np.log10(10 ** (np.nanpercentile(sinr_8, q)/10.))
sinr_avg_16 = 10*np.log10(10 ** (np.nanpercentile(sinr_16, q)/10.))
sinr_avg_32 = 10*np.log10(10 ** (np.nanpercentile(sinr_32, q)/10.))
sinr_avg_64 = 10*np.log10(10 ** (np.nanpercentile(sinr_64, q)/10.))

##############################

# 1) SINR and transmit power vs M
df_final_opt_ = compute_distributions(optimal=True)
df_final_opt = df_final_opt_.values
df_final_opt = df_final_opt.T
sinr_4_optimal, tx_power_4_optimal, sinr_8_optimal, tx_power_8_optimal, sinr_16_optimal, tx_power_16_optimal, sinr_32_optimal, tx_power_32_optimal, sinr_64_optimal, tx_power_64_optimal = df_final_opt 

##################################################################################################################
# 2) Convergence vs Antenna Size
X = [4,8,16,32,64]
Y = pd.DataFrame()

for M in X:
    try:
        df_i = pd.read_csv('figures M={}/convergence.txt'.format(M), sep=',', header=None, index_col=False).dropna()
        df_i.columns = ['episode', 'reward']
        df_i['M'] = M
        Y = pd.concat([Y, df_i])
    except:
        pass

Y.to_csv('figures/convergence_vs_ant.csv', index=False)
df_convergence = Y.groupby('M')['episode'].nth(10).reset_index() # close to median
df_convergence = pd.merge(df_convergence, Y)
df_convergence['episode'] /= df_convergence['episode'].sum()   

plot_primary(df_convergence['M'], df_convergence['episode'], r'$M$', r'Normalized Convergence Episode ($\zeta$)', 0, 0.5, 'convergence.pdf')

#X=np.log2(np.array(X)).astype(int)
tx_power_4_optimal_agg = 10*np.log10(10 ** (np.nanmax(tx_power_4_optimal) / 10.))
tx_power_8_optimal_agg = 10*np.log10(10 ** (np.nanmax(tx_power_8_optimal) / 10.))
tx_power_16_optimal_agg = 10*np.log10(10 ** (np.nanmax(tx_power_16_optimal) / 10.))
tx_power_32_optimal_agg = 10*np.log10(10 ** (np.nanmax(tx_power_32_optimal) /10.))
tx_power_64_optimal_agg = 10*np.log10(10 ** (np.nanmax(tx_power_64_optimal) / 10.))

sinr_avg_4_optimal = 10*np.log10(10 ** (np.nanmax(sinr_4_optimal) / 10.))
sinr_avg_8_optimal = 10*np.log10(10 ** (np.nanmax(sinr_8_optimal) / 10.))
sinr_avg_16_optimal = 10*np.log10(10 ** (np.nanmax(sinr_16_optimal) / 10.))
sinr_avg_32_optimal = 10*np.log10(10 **  (np.nanmax(sinr_32_optimal) / 10.))
sinr_avg_64_optimal = 10*np.log10(10 ** (np.nanmax(sinr_64_optimal) / 10.))

Y1 = [tx_power_4_agg, tx_power_8_agg, tx_power_16_agg, tx_power_32_agg, tx_power_64_agg]
Y2 = [sinr_avg_4, sinr_avg_8, sinr_avg_16, sinr_avg_32, sinr_avg_64]
Y3 = [tx_power_4_optimal_agg, tx_power_8_optimal_agg, tx_power_16_optimal_agg, tx_power_32_optimal_agg, tx_power_64_optimal_agg] # power optimal
Y4 = [sinr_avg_4_optimal, sinr_avg_8_optimal, sinr_avg_16_optimal, sinr_avg_32_optimal, sinr_avg_64_optimal] # sinr optimal


# Normalize Y1 and Y3
Y1 = Y1/ sum(Y1)
Y3 = Y3/ sum(Y3)
Y1 = [round(x, 2) for x in Y1] 
Y3 = [round(x, 2) for x in Y3] 

plot_secondary(X,Y1,Y3,Y2,Y4, r'$M$', r'Normalized Transmit Power', r'Achievable SINR [dB]', 0.5, 67, 'achievable_sinr_power.pdf')


##############################
# 3) Average achievable rate (or SNR) vs number of antennas
plot_primary_two(X,sum_rate(Y2),sum_rate(Y4), r'$M$', r'$C$ [bps/Hz]', 'sumrate.pdf')
##############################