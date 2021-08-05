#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 15:43:17 2021

@author: nathanielgates
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from IPython.utils.capture import capture_output

#%%

def set_equal_ylim(axes, ymin=np.nan, ymax=np.nan):
    for ax in axes:
        
        if np.isnan(ymin):
            ymin, _ = ax.get_ylim()
        if np.isnan(ymax):
            _, ymax = ax.get_ylim()
        
        ymin_new, ymax_new = ax.get_ylim()
        if ymax_new > ymax:
            ymax = ymax_new
        if ymin_new < ymin:
            ymin = ymin_new
    for ax in axes:
        ax.set_ylim(ymin, ymax)
        
def set_equal_xlim(axes, xmin=np.nan, xmax=np.nan):
    for ax in axes:
        
        if np.isnan(xmin):
            xmin, _ = ax.get_xlim()
        if np.isnan(xmax):
            _, xmax = ax.get_xlim()
        
        xmin_new, xmax_new = ax.get_xlim()
        if xmax_new > xmax:
            xmax = xmax_new
        if xmin_new < xmin:
            xmin = xmin_new
    for ax in axes:
        ax.set_xlim(xmin, xmax)

#%%

def solve_and_get_txt(m):
    '''
    This function solves the Gekko model with disp=True and returns the text
    output as well.
    
    ToDo: modify this so that it captures the txt output even when disp=False
    '''
            
    # Solve and save optimization output as txt
    with capture_output() as c:
        m.solve(disp=True)
    c()
    txt = c.stdout
    
    return txt

    # # Solve and save optimization output as txt
    # try:
    #     with capture_output() as c:
    #         m.solve(disp=True)
    #     msg = 'None'
    # except:
    #     msg = 'Solve failed'
    # c()
    # txt = c.stdout
    
    # return txt, msg

def get_apm_values(txt, 
                   values=['DOF', 'ITERATIONS', 'STATE_VAR', 'EQUATIONS',
                           'SLACK_VAR','EQUALITY_CON', 'INEQUALITY_CON']
                   ):
    '''
    This function parses the APMonitor text output file to extract
    additional information.
    Inputs:
        txt - the APMonitor output text file
        values - a list of the values to extract
    Outputs:
        results - a dictionary of the values and corresponding results
    Example usage:
        values = ['DOF', 'ITERATIONS', 'STATE_VAR', 'EQUATIONS', 
                  'SLACK_VAR', 'EQUALITY_CON', 'INEQUALITY_CON']
        out = get_apm_values(txt, values)
    '''
    
    value_dict = {
        'DOF': 'Degrees of freedom',
        'ITERATIONS': 'Number of Iterations',
        'STATE_VAR': 'Number of state variables',
        'EQUATIONS': 'Number of total equations',
        'SLACK_VAR': 'Number of slack variables',
        'EQUALITY_CON': 'Total number of equality constraints',
        'INEQUALITY_CON': 'Total number of inequality constraints'
        }
    
    if values == '':
        print('Specify values')
        exit
                
    if ~isinstance(values, list):                
        values = list(values)
    
    results = {}
    
    for value in values:
        string = value_dict[value]
        if value == 'ITERATIONS':
            # Parse iterations differently (for sequential)
            instances = txt.split(string)[1:]
            outs = []
            for instance in instances:
                end = instance.find('\n')
                out = int(instance[:end].split(' ')[-1])
                outs.append(out)
            out = int(np.sum(outs))
        else:
            start = txt.find(string)
            out = txt[start:]
            end = out.find('\n')
            out = int(out[:end].split(' ')[-1])
        results[value] = out

    return results

def plot_data(df, model_name, save=True):

    fig, axes = plt.subplots(2, 3, figsize=(8, 6), sharex=True)
    axes = axes.ravel()

    variables = ['time (s)', 'TIME/ITERATION', 'f',
                 'DOF', 'ITERATIONS', 'total err']
    labels = ['Solve Time (s)', 'Time/Iteration (s)', 'Objective Function', 
              'Degrees of Freedom', 'Total Iterations', 'Total Error']
    logy_list = [True, True, True, False, False, False]
    for i, (var, label) in enumerate(zip(variables, labels)):
        ax = axes[i]

        logx = True
        logy = logy_list[i]

        df.loc[6][var].plot(ax=ax,
                            marker='o', 
                            markeredgecolor='C0', 
                            markerfacecolor='None',
                            label='Simultaneous',
                            logx=logx,
                            logy=logy)
        df.loc[9][var].plot(ax=ax,
                            marker='o', 
                            markeredgecolor='C1', 
                            markerfacecolor='None',
                            label='Sequential',
                            logx=logx,
                            logy=logy)

        ax.set_xlabel('Number of Timesteps')
        # ax.set_ylabel(label)
        ax.set_title(label)
        if i == 0:
            ax.legend()
        ax.grid(which='major', linestyle='-', alpha=0.6, c='gray',
                linewidth=0.6)
        ax.grid(which='minor', linestyle=':', alpha=0.3, c='k',
                linewidth=0.5)
        
        ax.yaxis.set_major_formatter(mtick.ScalarFormatter())
        ax.xaxis.set_major_formatter(mtick.ScalarFormatter())
        
    plt.suptitle(model_name)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save:
        plt.savefig(model_name.replace(' ', '_')+'_results.pdf')

def plot_storage_results(df):
    # fig, axes = plt.subplots(4, 1, sharex=True)
    nrow = 2
    fig, axes = plt.subplots(nrow, 1, sharex=True, figsize=(5, 1.0*nrow + 0.1))
        
    ax = axes[0]
    ax.plot(df.t, df.d, 'r-', label='Demand ($d$)')
    ax.plot(df.t, df.p, 'b:', linewidth=2, label='Production ($g$)')
    
    ax = axes[1]
    ax.plot(df.t, df.s, 'k-', label='Storage ($e$)')
    ax.plot(df.t, df.stored, 'g--', label='Stored ($e_{in}$)')
    ax.plot(df.t, df.recovery, 'b:', label='Recovered ($e_{out}$)',
            linewidth=2)
    # ax.text(1.05, 20, 'Energy')
    
    # ax = axes[2]
    # ax.plot(df.t, df.s, 'g-', label='Storage ($e$)')
    
    # ax = axes[3]
    # ax.plot(df.t, df.vx, 'C2-', label='$S_1$')
    # ax.plot(df.t, df.vy, 'C3--', label='$S_2$')
    ax.set_xlabel('Time')
    
    for ax in axes:
        ax.legend(bbox_to_anchor=(1.01, 0.5), loc='center left', frameon=False)
        ax.grid()
        # ax.set_xlim(0, 24)
        # loc = mtick.MultipleLocator(base=6)
        ax.set_xlim(0, 1)
        # loc = mtick.MultipleLocator(base=1)
        # ax.xaxis.set_major_locator(loc)
    
    plt.tight_layout()


def plot_load_follow(data, version=1):
    
    names = {'t': 'Time', 'load': 'Demand ($d$)', 'gen': 'Production ($g$)', 
             'dgen': 'Ramp Rate ($r$)'}
    df = pd.DataFrame(data).rename(columns=names)
    df = df[names.values()]
    
    colors = ['r', 'b', 'k']
    # colors = ['C3', 'C0', 'k']
    linestyles = ['-', ':', '--']
    
    if version == 1:
        df = df.set_index(names['t'])
        fig, ax = plt.subplots()
        for i, col in enumerate(df):
            df[col].plot(ax=ax, color=colors[i], linestyle=linestyles[i])
        ax.legend()
        ax.grid(linestyle=':')
        
        import matplotlib.ticker as mtick
    
        ax.xaxis.set_minor_locator(mtick.MultipleLocator(base=0.1))
        ax.yaxis.set_minor_locator(mtick.MultipleLocator(base=0.5))
        
        plt.tight_layout()
        
    elif version == 2:
        
        nrow = 2
        fig, axes = plt.subplots(nrow, 1, sharex=True, figsize=(5, 1.1*nrow + 0.1))
        
        ax = axes[0]
        ax.plot(df[names['t']], df[names['load']], 'r-', label='Demand ($d$)')
        ax.plot(df[names['t']], df[names['gen']], 'b:', label='Production ($g$)',
                linewidth=2)
        
        ax = axes[1]
        ax.plot(df[names['t']], df[names['dgen']], 'k--', label='Ramp Rate ($r$)')
        
        ax.set_xlabel('Time')
        
        for ax in axes:
            ax.legend(bbox_to_anchor=(1.01, 0.5), loc='center left', frameon=False)
            ax.grid()
            # ax.set_xlim(0, 24)
            # loc = mtick.MultipleLocator(base=6)
            ax.set_xlim(0, 1)
            # loc = mtick.MultipleLocator(base=1)
            # ax.xaxis.set_major_locator(loc)
            
            ax.grid(linestyle=':')
        
            import matplotlib.ticker as mtick
        
            ax.xaxis.set_minor_locator(mtick.MultipleLocator(base=0.1))
            ax.yaxis.set_minor_locator(mtick.MultipleLocator(base=0.5))
        
        plt.tight_layout()     
        
    plt.savefig('1-dispatch.pdf', bbox_inches='tight')

def plot_co_gen(data, save=True, version=1):
    
    names = {'t': 'Time', 'load1': 'Demand 1 ($d_1$)', 'gen1': 'Production 1 ($g_1$)', 
             'load2': 'Demand 2 ($d_2$)', 'gen2': 'Production 2 ($g_2$)', 'dgen1': 'Ramp Rate ($r$)'}
    df = pd.DataFrame(data).rename(columns=names)
    df = df[names.values()]
    df = df.set_index(names['t'])
    
    colors = ['r', 'b']*2 + ['k']
    # colors = ['C3', 'C0', 'k']
    linestyles = ['-']*2 + ['--']*2 + [':']
    
    if version == 1:
        
        fig, ax = plt.subplots()#figsize=(6,4))
        for i, col in enumerate(df):
            df[col].plot(ax=ax, color=colors[i], linestyle=linestyles[i])
        ax.legend(loc='lower center')#bbox_to_anchor=(1.01, 0.5), loc='center left', frameon=False)
        ax.grid(linestyle=':')
        
        import matplotlib.ticker as mtick
    
        ax.xaxis.set_minor_locator(mtick.MultipleLocator(base=0.1))
        ax.yaxis.set_minor_locator(mtick.MultipleLocator(base=1))
        
        plt.tight_layout()
    
    elif version == 2:
                
        nrow = 3
        fig, axes = plt.subplots(nrow, 1, sharex=True, figsize=(5, 1.0*nrow + 0.1))
        
        ax = axes[0]
        ax.plot(df.index, df[names['load1']], 'r-', label='Demand 1 ($d_1$)')
        ax.plot(df.index, df[names['gen1']], 'b:', label='Production 1 ($g_1$)',
                linewidth=2)

        ax = axes[1]
        ax.plot(df.index, df[names['load2']], 'r-', label='Demand 2 ($d_2$)')
        ax.plot(df.index, df[names['gen2']], 'b:', label='Production 2 ($g_2$)',
                linewidth=2)
        
        ax = axes[2]
        ax.plot(df.index, df[names['dgen1']], 'k--', label='Ramp Rate ($r$)')
        
        ax.set_xlabel('Time')
        
        for ax in axes:
            ax.legend(bbox_to_anchor=(1.01, 0.5), loc='center left', frameon=False)
            ax.grid()
            # ax.set_xlim(0, 24)
            # loc = mtick.MultipleLocator(base=6)
            ax.set_xlim(0, 1)
            # loc = mtick.MultipleLocator(base=1)
            # ax.xaxis.set_major_locator(loc)
            
            ax.grid(linestyle=':')
        
            import matplotlib.ticker as mtick
        
            ax.xaxis.set_minor_locator(mtick.MultipleLocator(base=0.1))
            ax.yaxis.set_minor_locator(mtick.MultipleLocator(base=0.5))
        
        plt.tight_layout()    
        
    if save:
        plt.savefig('2-dispatch.pdf', bbox_inches='tight')
   
def plot_load_follow_storage(data, version=1):
    
    df = pd.DataFrame(data)

    # fig, axes = plt.subplots(4, 1, sharex=True)
    fig, axes = plt.subplots(3, 1, sharex=True)
    
    ax = axes[0]
    ax.plot(df.t, df.load, 'r-', label='Demand ($d$)')
    ax.plot(df.t, df.gen, 'b:', label='Production ($g$)')
    ax.plot(df.t, df.dgen, 'k--', label='Ramp Rate ($r$)')
    
    ax = axes[1]
    ax.plot(df.t, df.stored, 'C3-', label='Stored Energy')
    ax.plot(df.t, df.recovery, 'C0-.', label='Recovered Energy')
    
    ax = axes[2]
    ax.plot(df.t, df.s, 'C2-', label='Energy Inventory')
    
    # ax = axes[3]
    # ax.plot(df.t, df.vx, 'C2-', label='$S_1$')
    # ax.plot(df.t, df.vy, 'C3--', label='$S_2$')
    ax.set_xlabel('Time')
    
    for ax in axes:
        if version == 1:
            ax.legend(bbox_to_anchor=(1.01, 0.5), loc='center left', frameon=False)
        elif version == 2:
            ax.legend()
        ax.grid()
        # ax.set_xlim(0, 24)
        # loc = mtick.MultipleLocator(base=6)
        ax.set_xlim(0, 1)
        # loc = mtick.MultipleLocator(base=1)
        # ax.xaxis.set_major_locator(loc)
    
    plt.tight_layout()
    
    plt.savefig('1-dispatch-storage.pdf', bbox_inches='tight')

def plot_load_follow_storage_solar(data, version=1):
    
    df = pd.DataFrame(data)

    nrow = 3
    fig, axes = plt.subplots(nrow, 1, sharex=True, figsize=(5, 1.0*nrow + 0.1))    
    
    ax = axes[0]
    ax.plot(df.t, df.load, 'r-', label='Demand ($d$)')
    ax.plot(df.t, df.gen, 'b:', linewidth=2, label='Production ($g$)')
    # ax.plot(df.t, df.r, 'C1--', label='Source ($src$)')
    ax.plot(df.t, df.load - df.r, 'k--', label='Net Demand',
            zorder=1)
    
    ax = axes[1]
    ax.plot(df.t, df.r, 'b-', label='Source ($R$)')#, linewidth=2)
    ax.plot(df.t, df.dgen, 'k--', label='Ramp Rate ($r$)')
    
    ax = axes[2]
    ax.plot(df.t, df.s, 'k-', label='Storage ($e$)')
    ax.plot(df.t, df.stored, 'g--', label='Stored ($e_{in}$)')
    ax.plot(df.t, df.recovery, 'b:', label='Recovered ($e_{out}$)',
            linewidth=2)
    
    # ax = axes[3]
    # ax.plot(df.t, df.s, 'C2-', label='Energy Inventory')
    
    # ax = axes[3]
    # ax.plot(df.t, df.vx, 'C2-', label='$S_1$')
    # ax.plot(df.t, df.vy, 'C3--', label='$S_2$')
    ax.set_xlabel('Time')
    
    for ax in axes:
        if version == 1:
            ax.legend(bbox_to_anchor=(1.01, 0.5), loc='center left', frameon=False)
        elif version == 2:
            ax.legend()
        ax.grid()
        # ax.set_xlim(0, 24)
        # loc = mtick.MultipleLocator(base=6)
        ax.set_xlim(0, 1)
        # loc = mtick.MultipleLocator(base=1)
        # ax.xaxis.set_major_locator(loc)
    
    plt.tight_layout()
    
    # plt.savefig('1-dispatch-storage-solar.pdf', bbox_inches='tight')
    plt.savefig('5-energy_storage.pdf', bbox_inches='tight')
    
#%% In development

def plot_co_gen_storage(data):
    
    names = {'t': 'Time', 
             'load1': 'Demand 1 ($d_1$)', 
             'gen1': 'Production 1 ($g_1$)', 
             'load2': 'Demand 2 ($d_2$)', 
             'gen2': 'Production 2 ($g_2$)', 
             'dgen1': 'Ramp Rate ($r$)'}
    df = pd.DataFrame(data).rename(columns=names)
    df = df[names.values()]
    df = df.set_index(names['t'])
    
    colors = ['r', 'b']*2 + ['k']
    # colors = ['C3', 'C0', 'k']
    linestyles = ['-']*2 + ['--']*2 + [':']
    fig, ax = plt.subplots()#figsize=(6,4))
    for i, col in enumerate(df):
        df[col].plot(ax=ax, color=colors[i], linestyle=linestyles[i])
    ax.legend(loc='lower center')#bbox_to_anchor=(1.01, 0.5), loc='center left', frameon=False)
    ax.grid(linestyle=':')
    
    import matplotlib.ticker as mtick

    ax.xaxis.set_minor_locator(mtick.MultipleLocator(base=0.1))
    ax.yaxis.set_minor_locator(mtick.MultipleLocator(base=1))
    
    plt.tight_layout()
    plt.savefig('2-dispatch.pdf', bbox_inches='tight')
    
def plot_co_gen_storage(data, save=True, version=1):
    
    df = pd.DataFrame(data)

    # fig, axes = plt.subplots(4, 1, sharex=True)
    fig, axes = plt.subplots(3, 1, sharex=True)
    
    ax = axes[0]
    ax.plot(df.t, df.load1, 'r-', label='Demand 1 ($d_1$)')
    ax.plot(df.t, df.gen1, 'b-', label='Production 1 ($g_1$)')
    ax.plot(df.t, df.load2, 'r--', label='Demand 2 ($d_2$)')
    ax.plot(df.t, df.gen2, 'b--', label='Production 2 ($g_2$)')
    ax.plot(df.t, df.dgen1, 'k:', label='Ramp Rate ($r$)')
    
    ax = axes[1]
    ax.plot(df.t, df.stored, 'C3-', label='Stored Energy')
    ax.plot(df.t, df.recovery, 'C0-.', label='Recovered Energy')
    
    ax = axes[2]
    ax.plot(df.t, df.s, 'C2-', label='Energy Inventory')
    
    # ax = axes[3]
    # ax.plot(df.t, df.vx, 'C2-', label='$S_1$')
    # ax.plot(df.t, df.vy, 'C3--', label='$S_2$')
    # # ax.set_xlabel('Time (hr)')
    ax.set_xlabel('Time')
    
    for ax in axes:
        if version == 1:
            ax.legend(bbox_to_anchor=(1.01, 0.5), loc='center left', frameon=False)
        elif version == 2:
            ax.legend()
        ax.grid()
        # ax.set_xlim(0, 24)
        # loc = mtick.MultipleLocator(base=6)
        ax.set_xlim(0, 1)
        # loc = mtick.MultipleLocator(base=1)
        # ax.xaxis.set_major_locator(loc)
    
    plt.tight_layout()
    
    if save:
        plt.savefig('2-dispatch-storage.pdf', bbox_inches='tight')
