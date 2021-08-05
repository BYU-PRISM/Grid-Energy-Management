#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 21:34:49 2021

Gekko implementation of the simple energy storage model found here:
    https://www.sciencedirect.com/science/article/abs/pii/S030626191500402X

Useful link:
    http://apmonitor.com/wiki/index.php/Apps/PeriodicBoundaryConditions    

Note:
    The printing line-by-line is being affected by the solve function with txt
    
@author: nathanielgates
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utilities as util
# import feasibility as fs
from gekko import GEKKO

#%% Gekko Model

def model(t, plot=False, disp=False, ramp=1, imode=6, nodes='',
            # server='https://gekko.apmonitor.com'):
            server='http://byu.apmonitor.com'):
    '''
    server = 'http://byu.apmonitor.com'
    imode = 6
    nodes = 4 # 2 is fastest
    t = np.linspace(0, 24, 24*3+1)/24 # 25)
    '''
    
    m = GEKKO(remote=True)
    m._server = server
    
    m.time = t
    
    m.options.SOLVER = 3
    m.options.IMODE = imode
    m.options.NODES = nodes
    m.options.CV_TYPE = 2 # 1 = Linear penalty from a dead-band trajectory
    m.options.MAX_ITER = 400 #300 # Default is 100
    
    p = m.FV() # production (constant)
    p.STATUS = 1
    s = m.Var(0.1, lb=0) # storage inventory
    stored = m.SV() # store energy rate
    recovery = m.SV() # recover energy rate
    vx = m.SV(lb=0) # recover slack variable
    vy = m.SV(lb=0) # store slack variable
    
    # m.periodic(s)
    # m.Obj(1e4*(s[len(t)]-s[0])**2)
    
    
    # r = m.Param(-20*np.cos(np.pi*t/12*24)+50) #renewable energy source
    renewable = 30*np.cos(np.pi*t/6*24)+30 #renewable energy source
    center = np.ones(len(t))
    num = len(t)
    center[0:int(num/4)] = 0
    center[-int(num/4):] = 0
    renewable *= center
    r = m.Param(renewable)
    
    eps = 0.85 # Storage efficiency
    
    d = m.MV(-30*np.sin(np.pi*t/12*24)+100)
    
    m.Equations([p + r + recovery/eps - stored >= d,
                 p + r - d == vx - vy,
                 stored == p + r - d + vy,
                 recovery == d - p - r + vx,
                 s.dt() == stored - recovery/eps,
                 # vx * vy <= 0,
                 stored * recovery <= 0])
    m.Minimize(p)
    # m.solve(disp=True)
    
    # Need to debug why the solve below is failing.
    
    try:
        # # Solve the optimization model (enforces disp=True)
        # txt = util.solve_and_get_txt(m)
        
        if 0:
            m.solve(disp=True)
            txt = '  '
        else:
        
            #### TEMPORARY
            from IPython.utils.capture import capture_output
            with capture_output() as c:
                m.solve(disp=True)
            c()
            txt = c.stdout
            ####
        
        # Get additional APMonitor values
        out = util.get_apm_values(txt)
        
        M = m.options
        message = M.APPINFO
        if message == 0:
            message = "Optimization terminated successfully"
        ## Need to define feasibility function for this model before this will work
        # consCheck = [
        #         gen.value, t,
        #         dgen.value
        #         ]
        # feasible, error1 = fs.load_feasibility(consCheck, tol=1e-6)
        feasible, error1 = 'Not tested', 'NA'
    except:
        print('Error')
        M = m.options
        message = M.APPINFO
        if message == 0:
            message = "Solution not found"
        feasible = False
        error1 = "NA"
        out = {}
    info = {
        'Model':'Gekko load-following',
        'time_steps':len(t),
        'fcalls':M.ITERATIONS,
        'gcalls':'NA',
        'f':M.OBJFCNVAL,
        'feasible':feasible,
        'ramp err':error1,
        'total err':error1,
        'time (s)':M.SOLVETIME,
        'message':message,
        'status':M.APPSTATUS,
        'path':m._path
        }
    data = {
        't': t,
        'd': d,
        'p': p,
        's': s,
        'r': r,
        'stored': stored,
        'recovery': recovery,
        'vx': vx,
        'vy': vy
        }
    
    # Add in the APMonitor data
    info = {**info, **out}    

    return info, data
    
if __name__ == '__main__':
    
    option = 1
    # option = 2
    model_name = '4-energy_storage'
    
    if option == 1:
        #%%
        
        t = np.linspace(0, 24, 24*3+1)/24 # 25)
        imode = 6
        # imode = 9
        info, data = model(t, imode=imode, nodes=4)
        
        df = pd.DataFrame(data)
        
        def plot_storage_results(df):
            fig, axes = plt.subplots(4, 1, sharex=True)
            
            ax = axes[0]
            ax.plot(df.t, df.stored, 'C3-', label='Stored Energy')
            ax.plot(df.t, df.recovery, 'C0-.', label='Recovered Energy')
            
            ax = axes[1]
            ax.plot(df.t, df.d, 'k-', label='Electricity Demand')
            ax.plot(df.t, df.p, 'C3--', label='Power Production')
            ax.plot(df.t, df.r, 'c--', label='Renewable Production')
            
            ax = axes[2]
            ax.plot(df.t, df.s, 'C2-', label='Energy Inventory')
            
            ax = axes[3]
            ax.plot(df.t, df.vx, 'C2-', label='$S_1$')
            ax.plot(df.t, df.vy, 'C3--', label='$S_2$')
            # ax.set_xlabel('Time (hr)')
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

        # Visualize results
        plot_storage_results(df) 
        
        plt.savefig(model_name.replace(' ', '_')+'.pdf')

        
    #%%
    elif option == 2:
        
        #%%
        df = {}
        d = {}
        imodes = [6, 9]
        for imode in imodes:
            print('iMode: {}'.format(imode))
            # steps = [3, 6, 9] # [5, 10, 20]#, 40, 80]#, 160, 320]
            base = 2
            end = 8 # 5 # 7 # 9
            # base = 1.5 # 2
            # end = 13 # 8 # 5 # 7 # 9
            steps = [int(base**i) for i in range(2, end)]
            df[imode] = {}
            d[imode] = {}
            for n in steps:
                t = np.linspace(0, 1, n+1)
                add = [0.01]#, 0.02]
                t = np.array(list(sorted(list(t) + add)))
                print(n)
                sol, res = model(t, disp=True, imode=imode)
                df[imode][n] = sol
                d[imode][n] = res
        for imode in imodes:
            df[imode] = pd.DataFrame(df[imode]).T.reset_index().rename(columns={'index': 'step'})
        df = (pd.concat(df)
              .reset_index()
              .rename(columns={'level_0': 'imode'})
              .drop(columns='level_1')
          )
        
        df[['imode', 'step']] = df[['imode', 'step']].astype(int)
        df = df.set_index(['imode', 'time_steps'])
        
        df['TIME/ITERATION'] = df['time (s)'] / df.ITERATIONS
        
        for imode in imodes:
            for n in steps:
                d[imode][n] = pd.DataFrame(d[imode][n], 
                                           index=np.arange(n+1+len(add)))
            d[imode] = pd.concat(d[imode])
        d = (pd.concat(d)
             .reset_index()
             .rename(columns={'level_0': 'imode', 'level_1': 'step'})
             .drop(columns='level_2')
             .set_index(['imode', 'step'])
         )
        
        #%% Plot results
        
        fig, axes = plt.subplots(len(steps), 2, sharex=True, sharey=True, 
                                 figsize=(8, 6))
        
        imode_name = {6: 'Simultaneous', 9: 'Sequential'}
        
        step_dict = df.reset_index()[['step', 'time_steps']].drop_duplicates()
        step_dict = dict(zip(step_dict['step'], step_dict.time_steps))
        
        for j, imode in enumerate(imodes):
            # time_steps = list(sorted(list(set(d.loc[imode].index))))
            for i, step in enumerate(steps):
                ax = axes[i, j]
                dp = d.loc[imode].loc[step].set_index('t')
                dp.plot(ax=ax, legend=False, 
                        marker='.')
                        # marker='o',
                        # markerfacecolor='None', markersize=5)
                if j == 0:
                    ax.set_ylabel('$t_n=${}  '.format(step_dict[step]), 
                                  rotation=0, ha='right', va='center')
                if i == 0:
                    ax.set_title(imode_name[imode])
                ax.grid(linestyle=':', alpha=0.6, c='k', linewidth=0.6)
        axes[1, 1].legend(bbox_to_anchor=(1.05, 0.5), loc='center left',
                          frameon=False)
        util.set_equal_ylim(axes.ravel())
        plt.suptitle(model_name)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.subplots_adjust(hspace=0.1)
        plt.savefig(model_name.replace(' ', '_')+'_data.pdf')

#%%
if 0:
    #%%

    t = np.linspace(0, 24, 24*3+1)/24 # 25)
    d = -20/2*np.sin(np.pi*t/12*24)+100

    plt.figure()
    plt.plot(t, d)
