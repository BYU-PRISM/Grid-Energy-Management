#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 09:23:58 2021

@author: nathanielgates
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import gekko_load_follow as g1
import gekko_co_gen as g2
import gekko_tri_gen as g3
import gekko_storage_periodic as g4
import gekko_load_follow_storage as g5
import gekko_co_gen_storage_both as g6

import utilities as util
from time import time

#%% Run models

end = 8 # 11 # 5 # 11 # 11 # 5 # 10 # 11 # 14 # 11 # 9 # 7 # 9 # 5 # 11
time_steps = [int(2**i) for i in range(2, end)]

imodes = [6, 9]
# imodes = [6] # 0.76 min (for nodes = [0, 3, 4, 5] with end=5)
# imodes = [9] # 5.03 min (for nodes = [0, 1, 2, 3] with end=5)
# 0.91 min for imodes = [6], nodes = [3], end = 9
# 3.72 min for imodes = [6], nodes = [3], end = 11 (51 and 118 sec added)
# 2.45 min for imodes = [6], nodes = [3], end = 11 (with GEKKO server)
# 5.79 min for imodes = [6, 9], nodes = [2], end = 11, imode_9_lim = 64 (BYU)

# 4.86 min for imodes = [6], nodes = [2], end = 11 with new benchmarks
# 5.67 min for imodes = [6, 9], nodes = [2], end = 11, imode_9_lim = 64 (BYU)
## with new benchmarks

imode_9_lim = 128 # 64
imode_9_lim = 64

imode_9_model_1_lim = 64
imode_9_model_2_lim = 128
imode_9_model_3_lim = 256

# Set objective function
cv_type = 1 # l-1 norm
# cv_type = 2 # squared error

# Set max time
max_time = 200 # seconds


name1 = 'Load-Follow'#'ing'
name2 = 'Cogen'#'eration'
name3 = 'Tri-gen'#'eration'
name4 = 'Stor-Const'
name5 = 'Stor-Load'
name6 = 'Stor-Cogen'
# name1 = 'Benchmark I'
# name2 = 'Benchmark II'
# name3 = 'Benchmark III'
# name4 = 'Benchmark IV'
# name5 = 'Benchmark V'
# name6 = 'Benchmark VI'

names = [name1, name2, name3, name4, name5, name6]
models = [g1, g2, g3, g4, g5, g6]
numbers = [1, 2, 3, 4, 5, 6]

# names = [
#     # name1, 
#     name2, 
#     # name3
#     ]
# models = [
#     # g1, 
#     g2, 
#     # g3
#     ]
# numbers = [
#     # 1, 
#     2, 
#     # 3
#     ]

# nodes = [0, 1, 2, 3, 4, 5]
# 5min for N0-N5 and end=9
# nodes = [2, 3, 4, 5, 6]
nodes = [2]

d = {}
df = {}

for imode in imodes:
    print('\n---- iMode {} ----'.format(imode))
    
    d[imode] = {}
    df[imode] = {}
    
    time_start = time()
    for n in time_steps:
        # if (imode == 9) & (n > imode_9_lim):
        #     continue
        print('Timesteps: {}'.format(n))
        t = np.linspace(0, 1, n)
        
        # Insert finer resolution at start
        add = [0.01]#[0.01]#, 0.02]
        t = np.array(list(sorted(set(list(t) + add))))
        
        d[imode][n] = {}
        df[imode][n] = {}
        
        for node in nodes:
            print('  Nodes: {}'.format(node))

            df[imode][n][node] = {}
            d[imode][n][node] = {}

            time1 = time()
            for model, name, number in zip(models, names, numbers):
                # if (imode == 9) & (number > 3):
                #     continue
                if (imode == 9):
                    if (number == 1) & (n > imode_9_model_1_lim):
                        continue
                    elif (number == 2) & (n > imode_9_model_2_lim):
                        continue
                    elif (number == 3) & (n > imode_9_model_3_lim):
                        continue

                # if (len(t) > 1000) & (number == 6):
                #     continue
                
                try:
                    print('    Model: {}-{}'.format(number, name))
                    
                    time1_a = time()
                    
                    # Solve the optimization prooblem
                    sol, res = model.model(
                        t, imode=imode, nodes=node, disp=True, solver=3, 
                        cv_type=cv_type, max_time=max_time) # solver=2) # Try this...
                    
                    time2_a = time()
                    time_sum_a = time2_a - time1_a
                    print('      Time: {:.2f}s'.format(time_sum_a))
                    
                    df[imode][n][node][number] = sol
                    d[imode][n][node][number] = res
                except:
                    print('Failed to solve')
                    
                    time2_a = time()
                    time_sum_a = time2_a - time1_a
                    print('      Time: {:.2f}s'.format(time_sum_a))
                    
                    df[imode][n][node][number] = {}
                    d[imode][n][node][number] = {}
                    
            time2 = time()
            time_sum = time2 - time1
            print('    Time: {:.2f}s'.format(time_sum))

time_end = time()
time_tot = time_end - time_start

print('Total time: {:.2f}min'.format(time_tot/60))

df_raw = df.copy()

#%% Process data
  
df = df_raw.copy()

for imode in imodes:
    for n in time_steps:
        # if (imode == 9) and (n > imode_9_lim):
        #     continue
        for node in nodes:
            df[imode][n][node] = (pd.DataFrame(df[imode][n][node])
                                  .T
                                  .reset_index()
                                  .rename(columns={'index': 'number'})
                                  )
        df[imode][n] = pd.concat(df[imode][n])
    df[imode] = pd.concat(df[imode])
    
df = (pd.concat(df)
      .reset_index()
      .rename(columns={'level_0': 'imode', 
                       'level_1': 'step', 
                       'level_2': 'nodes'})
       .drop(columns=['level_3'])
      )

df = df.set_index(['imode', 'number', 'nodes', 'step'])

#%% Visualize data


imode_name = {6: 'Simultaneous', 9: 'Sequential'}

markers = ['o']*4 #, 's', '^']
# markers = ['o', 's', '^', '.', '_']
linestyles = [':', '--', '-.', '-']

# nodes = [0, 3, 4, 5]
# nodes = [0, 3]


if 0:
    names = ['Load-Follow', 'Cogen', 'Tri-gen', 
             'Stor-Const', 'Stor-Load', 'Stor-Cogen']
    names = ['Load-Follow', 'Cogen', 'Tri-gen', 
             'Stor-Constant', 'Stor-Load-Follow', 'Stor-Cogen']

var = 'time (s)'

dat = {}

plt.figure(figsize=(5, 4.5))
# for imode, line in zip(imodes[::-1], ['-', '--'][::-1]):
for imode, line in zip(imodes[::-1], ['-', ':'][::-1]):
    dat[imode] = {}
    for node, marker, linestyle in zip(nodes, markers, linestyles):
        dat[imode][node] = {}
        for i in range(len(models)):
            if (imode == 9) & (numbers[i] > 3):
                continue
            dp = df.loc[imode].loc[numbers[i]].loc[node].copy()
            dp = dp.dropna()
            dat[imode][node][numbers[i]] = dp
            dp[var].plot(
            # dp['ITERATIONS'].plot(
            # (dp['time (s)'] / dp['ITERATIONS']).plot(
            # dp['DOF'].plot(
                color='C'+str(i),
                # linestyle=linestyle,
                linestyle=line,
                marker='o',
                markersize=5,
                # marker=marker, 
                # markeredgecolor='C'+str(i), 
                # markerfacecolor='None',
                # label='{}, N={}, {}-{}'.format(
                #     imode_name[imode][0:3], node, numbers[i], names[i]),
                label='{}. {}-{}'.format(
                    imode_name[imode][0:3], numbers[i], names[i]),
                logy=True,
                logx=True
                )
            # x = dp['time (s)'].index[-1]
            # y = dp['time (s)'].iloc[-1]
            # plt.text(x*1.1, y, numbers[i], ha='left', va='center')
            
            # if i != 3:
            #     x = dp['time (s)'].index[0]
            #     y = dp['time (s)'].iloc[0]
            #     plt.text(x*0.9, y, numbers[i], ha='right', va='center')
    
    ax = plt.gca()
    ax.set_xlabel('Number of Timesteps')
    ax.set_ylabel('Solve Time (s)')
    # ax.set_ylabel('Iterations')
    # ax.set_ylabel('Solve Time per Iteration')
    # ax.set_ylabel('Degrees of Freedom')
    # ax.legend(ncol=1)
    ax.legend(ncol=2, bbox_to_anchor=(0.5, 1.0), loc='lower center', 
              frameon=True)
    # ax.legend(ncol=1, bbox_to_anchor=(1.01, 0.5), loc='center left', 
    #           frameon=True)
    # ax.set_title(imode_name[imode])
    # ax.grid(linestyle=':', alpha=0.6, c='k', linewidth=0.6)
    ax.grid(which='major', linestyle='-', alpha=0.6, c='gray',
            linewidth=0.6)
    ax.grid(which='minor', linestyle=':', alpha=0.3, c='k',
            linewidth=0.5)
    
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_formatter(mtick.ScalarFormatter())
    
    # ax.yaxis.set_major_formatter(mtick.ScalarFormatter())
    # ax.yaxis.set_minor_formatter(mtick.ScalarFormatter())
    # plt.autoscale(enable=True, axis='y')
    
    for imode, line in zip(imodes[::-1], ['-', ':'][::-1]):
        for node, marker, linestyle in zip(nodes, markers, linestyles):
            for i in range(len(models)):
                if (imode == 9) & (numbers[i] > 3):
                    continue
                print('-', imode, numbers[i])
                dp2 = df.loc[imode].loc[numbers[i]].loc[node]['time (s)']
                dp2 = dp2.dropna()
                
                # Deal with failed timing results
                dp2 = dp2.loc[~(dp2 == 1)] 
                
                # End
                skip = False
                if (imode == 6) & (numbers[i] in [1, 2, 3, 4]):
                    skip = True
                if skip:
                    print(skip,' 1')
                else:
                    x = dp2.index[-1]
                    y = dp2.iloc[-1]
                    plt.text(x*1.1, y, numbers[i], ha='left', va='center')

                x = dp2.index[-1]
                y = dp2.iloc[-1]
                if (imode == 6) & (numbers[i] == 4):
                    plt.text(x*1.1, y-1, numbers[i], ha='left', va='center')
                if (imode == 6) & (numbers[i] == 1):
                    plt.text(x*1.1, y+1, numbers[i], ha='left', va='center')
                if (imode == 6) & (numbers[i] == 2):
                    plt.text(x*1.1, y-3, numbers[i], ha='left', va='center')
                if (imode == 6) & (numbers[i] == 3):
                    plt.text(x*1.1, y+5, numbers[i], ha='left', va='center')
                
                # Start
                skip = False
                if (imode == 9) & (numbers[i] in [1, 2, 3]):
                    skip = True
                elif (imode == 6) & (numbers[i] in [1, 2, 3, 4, 5, 6]):
                    skip = True
                if skip:
                    print(skip,' 2', imode, numbers[i])
                else:
                    x = dp2.index[0]
                    y = dp2.iloc[0]
                    plt.text(x*0.9, y, numbers[i], ha='right', va='center')

                
plt.tight_layout()

xmin, xmax = ax.get_xlim()
ax.set_xlim((0.9)*xmin, (1.1)*xmax) # This works due to the log scale

plt.savefig('timing_{}.pdf'.format(imode_name[imode].lower()),
            bbox_inches='tight')

#%%

dat.keys()
pd.DataFrame(dat[6])
