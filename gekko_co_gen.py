#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 15:02:46 2021

Notes:
    I can get sequential to match simultaneous by adding time-points at the
    start and the SSE objective function. Why is the second needed though?

@author: nathanielgates
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utilities as util
import feasibility as fs
from gekko import GEKKO

def model(t, plot=False, disp=False, imode=6, nodes='', solver=3, 
          cv_type=1, max_time='',
           # server='https://gekko.apmonitor.com'):
           server='http://byu.apmonitor.com'):
    
    # t = np.linspace(0, 1, 101)
    m = GEKKO(remote=True)
    m.time = t
    m._server = server
    
    # m.options.MAX_ITER = 1000
    if max_time == '':
        pass
    else:
        m.options.MAX_TIME = max_time
        
    load1 = m.Param(np.cos(2*np.pi*t)+3)
    # load2 = m.Param(1.5*np.sin(2*np.pi*t)+7)
    # load1 = m.Param(np.cos(2*np.pi*t)/2 + 3.5)
    load2 = m.Param(1.5*np.sin(2*np.pi*t)+7)
    # load2 = m.Param(1.0*np.sin(2*np.pi*t)+7)
    gen1 = m.Var(load1[0])
    gen2 = m.Intermediate(gen1*2)

    err1 = m.CV(0)
    err1.STATUS = 1
    err1.SPHI = err1.SPLO = 0
    err1.WSPHI = 1000
    err1.WSPLO = 1
    err2 = m.CV(0)
    err2.STATUS = 1
    err2.SPHI = err2.SPLO = 0
    err2.WSPHI = 1000
    err2.WSPLO = 1
    dgen1 = m.MV(0, lb=-1, ub=1)
    # dgen1 = m.MV(0, lb=-2, ub=2)
    dgen1.STATUS = 1

    m.Equations([gen1.dt() == dgen1, err1 == load1-gen1, err2 == load2-gen2])

    # m.Obj(err1**2 + err2**2) # Added. This makes the difference in solving seq
    # m.Obj(err1**2)
    # m.Obj(err2**2)
    m.Obj(err2**2 / len(t)) # Still need to scale the CV-controlled objective

    m.options.IMODE = imode
    m.options.SOLVER = solver
    # m.options.MV_STEP_HOR = 3
    # m.options.CV_TYPE = 2 # 1 = Linear penalty from a dead-band trajectory
    m.options.CV_TYPE = cv_type # 1 = Linear penalty from a dead-band trajectory

    if nodes == '':
        pass
    else:        
        m.options.NODES = nodes # 4

    # Solve the optimization model (enforces disp=True)
    txt = util.solve_and_get_txt(m)
    
    # Get additional APMonitor values
    out = util.get_apm_values(txt)
    
    if plot:
        import matplotlib.pyplot as plt
        plt.subplot(3, 1, 1)
        plt.plot(t, load1, label='load 1')
        plt.plot(t, gen1, label='gen')
        plt.legend()
        plt.subplot(3, 1, 2)
        plt.plot(t, load2, label='load 2')
        plt.plot(t, gen2, label='dh')
        plt.legend()
        plt.subplot(3, 1, 3)
        plt.plot(t, dgen1)
        plt.ylabel(r'$\frac{d\ gen}{dt}$')
        plt.show()
     
    M = m.options
    consCheck = [
            t, gen1.value, gen2.value,
            dgen1.value
    ]
    message = M.APPINFO
    if message == 0:
        message = "Optimization terminated successfully"
    feasible, error1, error2 = fs.co_feasibility(consCheck, tol=1e-6)
    
    info = {
        'Model':'Gekko co-gen',
        'time_steps':len(t),
        'fcalls':M.ITERATIONS,
        'gcalls':'NA',
        'f':M.OBJFCNVAL,
        'feasible':feasible,
        'ramp err':error1,
        'total err':error2,
        'time (s)':M.SOLVETIME,
        'message':message,
        'status':M.APPSTATUS,
        'path':m._path
        }
    data = {
        'load1': load1,
        'gen1': gen1,
        'load2': load2,
        'gen2': gen2,
        'dgen1': dgen1,
        't': t
        }
    
    # Add in the APMonitor data
    info = {**info, **out}

    return info, data

if __name__ == "__main__":
    
    option = 1
    # option = 2
    model_name = '2 - Co-generation'
    
    # plt.close('all')
    
    if option == 1:
        t = np.linspace(0, 1, 101)
        # t = np.linspace(0, 1, 73)
        info, data = model(t)
        print(info)
        
        util.plot_co_gen(data, version=2)
        
    elif option == 2:
#%%
        df = {}
        d = {}
        imodes = [6, 9]
        for imode in imodes:
            print('iMode: {}'.format(imode))
            # steps = [3, 6, 9] # [5, 10, 20]#, 40, 80]#, 160, 320]
            end = 5 # 8 # 5 # 7 # 9
            base = 2
            # base = 1.5
            # end = 13
            steps = [int(base**i) for i in range(2, end)]
            df[imode] = {}
            d[imode] = {}
            for n in steps:
                t = np.linspace(0, 1, n+1)
                add = [0.01] #[0.01, 0.02]
                t = np.array(list(sorted(list(t) + add)))
                print(n)
                sol, res = model(t, imode=imode)
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
        
        #%%
        
        if 0:
            plt.figure()
            logx = logy = True
            df.loc[6]['time (s)'].plot(marker='o', 
                                       markeredgecolor='C0', 
                                       markerfacecolor='None',
                                       label='Simultaneous',
                                       logx=logx,
                                       logy=logy)
            df.loc[9]['time (s)'].plot(marker='o', 
                                       markeredgecolor='C1', 
                                       markerfacecolor='None',
                                       label='Sequential',
                                       logx=logx,
                                       logy=logy)
    
            ax = plt.gca()
            ax.set_xlabel('Number of Timesteps')
            ax.set_ylabel('Solve Time (s)')
            ax.legend()
            ax.set_title(model_name)
            plt.savefig(model_name.replace(' ', '_')+'_time.pdf')
        
        #%%
        
        fig, axes = plt.subplots(len(steps), 2, sharex=True, sharey=True, 
                                 figsize=(8, 6))
        
        imode_name = {6: 'Simultaneous', 9: 'Sequential'}
        
        step_dict = df.reset_index()[['step', 'time_steps']].drop_duplicates()
        step_dict = dict(zip(step_dict['step'], step_dict.time_steps))
        
        for j, imode in enumerate(imodes):
            for i, step in enumerate(steps):
                ax = axes[i, j]
                dp = d.loc[imode].loc[step].set_index('t')
                dp.plot(ax=ax, legend=False, 
                        marker='.')
                        # marker='o',
                        # markerfacecolor='None', markersize=5)
                if j == 0:
                    ax.set_ylabel('$t_n=${}  '.format(step_dict[step]), rotation=0,
                                  ha='right', va='center')
                if i == 0:
                    ax.set_title(imode_name[imode])
                ax.grid(linestyle=':', alpha=0.6, c='k', linewidth=0.6)
                
                # if i != len(steps)-1:
                #     ax.set_xticks([])
                #     ax.set_xticks([], minor=True)
                # if j == 0:
                #     ax.set_yticks([])
                #     ax.set_yticks([], minor=True)
                
        axes[1, 1].legend(bbox_to_anchor=(1.05, 0.5), loc='center left',
                          frameon=False)
        util.set_equal_ylim(axes.ravel())
        plt.suptitle(model_name)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        # plt.subplots_adjust(hspace=0.25)
        plt.subplots_adjust(hspace=0.15, wspace=0.1, right=0.85,
                            left=0.125, top=0.925, bottom=0.075)
        plt.savefig(model_name.replace(' ', '_')+'_data.pdf')
        
        #%% Plot quadrant of data
        
        util.plot_data(df, model_name)

