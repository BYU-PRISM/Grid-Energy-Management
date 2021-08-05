#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 09:28:49 2021

@author: nathanielgates
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utilities as util
import feasibility as fs
from gekko import GEKKO


def model(t, plot=False, disp=False, ramp=1, imode=6, nodes='', solver=3,
          mv_step_hor='', cv_type=1, max_time='',
          # server='https://gekko.apmonitor.com'):
          server='http://byu.apmonitor.com'):
    
    '''
    Test options:
        n = 16
        t = np.linspace(0, 1, n+1)
        plot=False
        disp=True
        ramp=1
        imode=6
        nodes=3
        server='https://gekko.apmonitor.com'
    '''
    
    # t = np.linspace(0, 1, 101)
    m = GEKKO(remote=True)
    m._server = server

    m.time = t
    
    # m.options.MAX_ITER = 1000
    if max_time == '':
        pass
    else:
        m.options.MAX_TIME = max_time

    load = m.Param(np.cos(2*np.pi*t)+3)
    # load = m.Param(np.cos(2*np.pi*t)/2 + 3.5)
    gen = m.Var(load[0])

    err = m.CV(0)
    err.STATUS = 1
    err.SPHI = err.SPLO = 0
    err.WSPHI = 1000
    err.WSPLO = 1

    dgen = m.MV(0, lb=-ramp, ub=ramp) # ramp rate
    dgen.STATUS = 1

    m.Equations([gen.dt() == dgen,  err == load-gen])
    
    m.Obj(err**2 / len(t)) # Added
    
    if nodes == '':
        pass
    else:
        m.options.NODES = nodes # 4
        
    m.options.SOLVER = solver
    m.options.IMODE = imode
    # m.options.CV_TYPE = 2 # 1 = Linear penalty from a dead-band trajectory
    m.options.CV_TYPE = cv_type # 1 = Linear penalty from a dead-band trajectory

    if mv_step_hor != '':
        m.options.MV_STEP_HOR = mv_step_hor
        
    try:
        
        # Solve the optimization model (enforces disp=True)
        txt = util.solve_and_get_txt(m)
        
        # Get additional APMonitor values
        out = util.get_apm_values(txt)
        
        if plot:
            import matplotlib.pyplot as plt
            plt.plot(t, load)
            plt.plot(t, gen)
            plt.plot(t, dgen)
            plt.show()
        M = m.options
        message = M.APPINFO
        if message == 0:
            message = "Optimization terminated successfully"
        consCheck = [
                gen.value, t,
                dgen.value
        ]
        feasible, error1 = fs.load_feasibility(consCheck, tol=1e-6)
    except:
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
        'load': load,
        'gen': gen,
        'dgen': dgen,
        't': t
        }
    
    # Add in the APMonitor data
    info = {**info, **out}
    
    return info, data 

#%%

if __name__ == "__main__":
    
    option = 0 # Run model once and plot data
    # option = 2 # Grid refinement study, sim vs seq
    # option = 3 # Grid refinement, sim vs seq, changing DOF with mv_step_hor
    model_name = '1 - Load Following'
    
    # plt.close('all')
    
    if option == 0:
        t = np.linspace(0, 1, 101)
        # t = np.linspace(0, 1, 73)
        imode = 6
        # imode = 9
        info, data = model(t, plot=False, disp=True, imode=imode)
        print(info['fcalls'])
        
        util.plot_load_follow(data, version=2)
        
    elif option == 1:
        d = {}
        df = {}
        ramps = [0.25, 0.5, 1, 2, 4, 8]
        t = np.linspace(0, 1, 101) # Need to weight ramp by timestep length...
        for ramp in ramps:
            print(ramp)
            sol, res = model(t, plot=False, ramp=ramp)
            df[ramp] = sol
            d[ramp] = res
        df = pd.DataFrame(df).T.reset_index().rename(columns={'index': 'ramp'})
        for ramp in ramps:
            d[ramp] = pd.DataFrame(d[ramp], index=np.arange(len(t)))
        # test = pd.DataFrame(d).T.reset_index().rename(columns={'index': 'ramp'})

        #%%
        
        plt.figure(); plt.plot(df.ramp, df['time (s)'])
        # Need to run n times and take the average
        
        fig, axes = plt.subplots(3, 2, sharex=True)
        axes = axes.ravel()
        
        for i, ramp in enumerate(ramps):
            ax = axes[i]
            d[ramp].plot(ax=ax, legend=False)
            ax.set_title(ramp)
            ax.set_ylim(-2*np.pi*1.1, 2*np.pi*1.1)
        plt.tight_layout()
    #%%
    
    elif option == 2:
        
    #%%
        df = {}
        d = {}
        imodes = [6, 9]
        # imodes = [9, 6]
        for imode in imodes:
            print('iMode: {}'.format(imode))
            # steps = [3, 6, 9] # [5, 10, 20]#, 40, 80]#, 160, 320]
            base = 2
            end = 5 # 8 # 5 # 7 # 9
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
        
        #%% Plot time
        
        if 0:
            var = 'time (s)'
            label = 'Solve Time (s)'
    
            plt.figure()
            logx = logy = True
            df.loc[6][var].plot(marker='o', 
                                       markeredgecolor='C0', 
                                       markerfacecolor='None',
                                       label='Simultaneous',
                                       logx=logx,
                                       logy=logy)
            df.loc[9][var].plot(marker='o', 
                                       markeredgecolor='C1', 
                                       markerfacecolor='None',
                                       label='Sequential',
                                       logx=logx,
                                       logy=logy)
    
            ax = plt.gca()
            ax.set_xlabel('Number of Timesteps')
            ax.set_ylabel(label)
            ax.legend()
            ax.set_title(model_name)
            ax.grid(linestyle=':', alpha=0.6, c='k', linewidth=0.6)
            plt.tight_layout()
            plt.savefig(model_name.replace(' ', '_')+'_time.pdf')
        
        #%% Plot iterations
        
        if 0:
            var = 'ITERATIONS'
            label = 'Number of Iterations'
    
            plt.figure()
            logx = True
            logy = False
            df.loc[6][var].plot(marker='o', 
                                       markeredgecolor='C0', 
                                       markerfacecolor='None',
                                       label='Simultaneous',
                                       logx=logx,
                                       logy=logy)
            df.loc[9][var].plot(marker='o', 
                                       markeredgecolor='C1', 
                                       markerfacecolor='None',
                                       label='Sequential',
                                       logx=logx,
                                       logy=logy)
    
            ax = plt.gca()
            ax.set_xlabel('Number of Timesteps')
            ax.set_ylabel(label)
            ax.legend()
            model_name = '1 - Load Following'
            ax.set_title(model_name)
            ax.grid(linestyle=':', alpha=0.6, c='k', linewidth=0.6)
            plt.tight_layout()
            plt.savefig(model_name.replace(' ', '_')+'_iterations.pdf')
        
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
        
        #%% Plot quadrant of data
        
        util.plot_data(df, model_name)
        
    #%%
        
    elif option == 3:
        #%%
        df = {}
        d = {}
        imodes = [6, 9]
        # imodes = [9, 6]
        for imode in imodes:
            print('iMode: {}'.format(imode))
            # steps = [3, 6, 9] # [5, 10, 20]#, 40, 80]#, 160, 320]
            base = 2
            end = 5 # 8 # 5 # 7 # 9
            # base = 1.5 # 2
            # end = 13 # 8 # 5 # 7 # 9
            steps = [int(base**i) for i in range(2, end)]
            # steps = [5, 10, 20]#, 40]
            # mv_step_hors = [1, 2, 4]#, 8]
            steps = [10, 20, 40]#, 80, 160]
            mv_step_hors = [5, 10, 20]
            # mv_step_hors = [2]*len(steps) #[1, 2, 3]
            df[imode] = {}
            d[imode] = {}
            for i, n in enumerate(steps):
                t = np.linspace(0, 1, n+1)
                add = []# Removed to keep things constant with DOF [0.01]#, 0.02]
                add = list(np.linspace(0+t[1]/n, t[1], (mv_step_hors[i]+2)*2)[:-1])
                t = np.array(list(sorted(list(t) + add)))
                print(n)
                sol, res = model(t, disp=True, imode=imode, 
                                 mv_step_hor=mv_step_hors[i])
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
                d[imode][n] = pd.DataFrame(d[imode][n])#, 
                                           # index=np.arange(n+1+len(add)))
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
        # plt.savefig(model_name.replace(' ', '_')+'_data.pdf')
        
        #%% Plot quadrant of data
        
        util.plot_data(df, model_name, save=False)
        

