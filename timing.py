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
import utilities as util
from time import time

#%% Run models

end = 11 # 5 # 14 # 11 # 9 # 7 # 9 # 5 # 11
time_steps = [int(2**i) for i in range(2, end)]

imodes = [6, 9]
# imodes = [6] # 0.76 min (for nodes = [0, 3, 4, 5] with end=5)
# imodes = [9] # 5.03 min (for nodes = [0, 1, 2, 3] with end=5)
# 0.91 min for imodes = [6], nodes = [3], end = 9
# 3.72 min for imodes = [6], nodes = [3], end = 11 (51 and 118 sec added)
# 2.45 min for imodes = [6], nodes = [3], end = 11 (with GEKKO server)
# 5.79 min for imodes = [6, 9], nodes = [2], end = 11, imode_9_lim = 64 (BYU)
# 16.56min for imodes = [6, 9], nodes = [2], end = 11, imode_9_lim = 128 (BYU)



# imode_9_lim = 128 # 64 
imode_9_lim = 64

name1 = 'Load-follow'#'ing'
name2 = 'Co-gen'#'eration'
name3 = 'Tri-gen'#'eration'

names = [name1, name2, name3]
models = [g1, g2, g3]
numbers = [1, 2, 3]

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
        if (imode == 9) & (n > imode_9_lim):
            continue
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
                print('    Model: {}-{}'.format(number, name))
                
                time1_a = time()
                
                # Solve the optimization prooblem
                sol, res = model.model(t, imode=imode, nodes=node, disp=True,
                                        solver=3)
                                       # solver=2) # Try this...
                
                time2_a = time()
                time_sum_a = time2_a - time1_a
                print('      Time: {:.2f}s'.format(time_sum_a))
                
                df[imode][n][node][number] = sol
                d[imode][n][node][number] = res
                
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
        if (imode == 9) and (n > imode_9_lim):
            continue
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
linestyles = [':', '--', '-.', '-']

# nodes = [0, 3, 4, 5]
# nodes = [0, 3]

plt.figure()#figsize=(8,6))
for imode, line in zip(imodes[::-1], ['-', '--'][::-1]):
    imode_name
    for node, marker, linestyle in zip(nodes, markers, linestyles):
        for i in range(len(models)):
            dp = df.loc[imode].loc[numbers[i]].loc[node]
            dp['time (s)'].plot(
            # dp['ITERATIONS'].plot(
            # (dp['time (s)'] / dp['ITERATIONS']).plot(
            # dp['DOF'].plot(
                color='C'+str(i),
                # linestyle=linestyle,
                linestyle=line,
                # marker='.',
                marker=marker, 
                markeredgecolor='C'+str(i), 
                markerfacecolor='None',
                label='{}, N={}, {}-{}'.format(imode_name[imode][0:3], node, 
                                               numbers[i], names[i]),
                logy=True,
                logx=True
                )
    
    ax = plt.gca()
    ax.set_xlabel('Number of Timesteps')
    ax.set_ylabel('Solve Time (s)')
    # ax.set_ylabel('Iterations')
    # ax.set_ylabel('Solve Time per Iteration')
    # ax.set_ylabel('Degrees of Freedom')
    # ax.legend(ncol=1)
    ax.legend(ncol=2, bbox_to_anchor=(0.5, 1.05), loc='lower center', 
              frameon=True)
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
    
    plt.tight_layout()
    
    # plt.savefig('timing_{}.pdf'.format(imode_name[imode].lower()))

plt.savefig('timing_test.pdf')

#%%
