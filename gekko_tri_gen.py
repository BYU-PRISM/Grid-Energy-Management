#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 15:09:01 2021

@author: nathanielgates
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utilities as util
import feasibility as fs
from gekko import GEKKO


def model(t, plot=False, disp=False, imode=6, nodes=2, solver=3, 
          cv_type=1, max_time='',
            # server='https://gekko.apmonitor.com'):
            server='http://byu.apmonitor.com'):
    
    # t = np.linspace(0, 1, 101)
    m = GEKKO(remote=True)
    m.time = t
    m._server = server
    
    # m.options.MAX_ITER = 1000
    # m.options.MAX_ITER = 600
    if max_time == '':
        pass
    else:
        m.options.MAX_TIME = max_time
    
    '''
    Variable naming
     (n) = number of each type
     (d) = dispatchable consumer / producer
     (f) = forecasted (non-dispatchable) consumer / producer
     (e/h/k/z) = electricity / heat / cooling / chemical (H2)
    
    Design variables
     (r) = ramp rate
     (s) = size (max production / consumption)
     (m) = minimum operating level before shutdown required
    '''



    fe = m.Param(np.cos(2*np.pi*t)+3)                       # d1 : elec. demand
    # fe = m.Param(np.cos(2*np.pi*t)/2+3.5)                       # d1 : elec. demand
    fh = m.Param(1.5*np.sin(2*np.pi*t)+7)                   # d2 : heat demand
    # fh = m.Param(1.9*np.sin(2*np.pi*t)+7)                   # d2 : heat demand
    # # fh = m.Param(1.0*np.sin(2*np.pi*t)+7)                   # d2 : heat demand
    # # fh = m.Param(-1.5*np.sin(2*np.pi*t)+7)                   # d2 : heat demand
    fz = m.Param(np.clip(-0.2*np.sin(2*np.pi*t), 0, None))  # d3?
    de = m.Var(fe[0])                                       # g1 : g1(0) = 4
    dh = m.Intermediate(de*2)                               # g2
    dz = m.Var(0, lb=0)                                     # g3
    dze = m.Var(0)
    dzh = m.Var(0)
    te = m.Intermediate(fe+dze)
    th = m.Intermediate(fh+dzh)

    e = m.CV(0)
    e.STATUS = 1
    e.SPHI = e.SPLO = 0
    e.WSPHI = 1000
    e.WSPLO = 1
    h = m.CV(0)
    h.STATUS = 1
    h.SPHI = h.SPLO = 0
    h.WSPHI = 1000
    h.WSPLO = 1
    z = m.CV(0)
    z.STATUS = 1
    z.SPHI = z.SPLO = 0
    z.WSPHI = 1000
    z.WSPLO = 1
    ramp = 1
    # ramp = 2
    der = m.MV(0, lb=-ramp, ub=ramp)                              # r1 = dg1/dt
    der.STATUS = 1
    dzr = m.MV(0, lb=-ramp, ub=ramp)                              # r3 = dg3/dt
    dzr.STATUS = 1

    m.Equations([de.dt() == der,                            # 
                  e == te-de,                                #  
                  h == th-dh])
    m.Equations([dz.dt() == dzr, 
                  z == fz-dz, 
                  dze == dz*2, 
                  dzh == dz*3
                  # dze == dz*3, 
                  # dzh == dz*2
                  ])
    m.Maximize(dz)
    # m.Maximize(dz/10)
    
    # if imode == 9:
        # m.Minimize(e**2 + z**2)
        # m.Minimize(te**2 + th**2)
        # m.Minimize(e**2 + h**2)
        # m.Minimize(e**2 + h**2 * z**2)
        # m.Minimize(e**2)
        # m.Minimize(z**2)
        # m.Minimize(h**2)
    m.Minimize(1e-7*(h**2 + z**2) )#/ len(t)) # failed with 1e-15, wiggly with 1e-10
        # 1e-8 is good, but flickery with higher resolution

    m.options.IMODE = imode
    m.options.SOLVER = solver
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
        """
        plt.figure(figsize=(10,10))
        plt.subplot(4, 1, 1)
        plt.plot(t, fe, 'r:', label='Demand 1')
        plt.plot(t, de, 'b:', label="Production 1")
        plt.legend(loc='lower right')

        plt.subplot(4, 1, 2)
        plt.plot(t, fh, 'r--', label='Demand 2')
        plt.plot(t, dh, 'b--', label="Production 2")
        plt.legend(loc='lower right')
        
        plt.subplot(4, 1, 3)
        plt.plot(t, dz, 'b-', label="Production 3")
        plt.plot(t, fz,'r-', label='Demand 3')
        plt.legend(loc='lower right')
        
        plt.subplot(4, 1, 4)
        plt.plot(t, der, 'k:', label='Ramp Rate 1')
        plt.plot(t, dzr, 'k--', label='Ramp Rate 3')
        plt.legend(loc='lower right')
        plt.xlabel('Time')
        plt.savefig('trigen.eps', format='eps')
        plt.show()
        #"""
        
        plt.figure(figsize=(10,10))
        plt.subplot(3, 1, 1)
        plt.plot(t, fe, label='Demand 1')
        plt.plot(t, fh, label='Demand 2')
        plt.plot(t, fz, label='Demand 3')
        plt.legend(loc='lower right')

        plt.subplot(3, 1, 2)
        plt.plot(t, de, label="Production 1")
        plt.plot(t, dh, label="Production 2")
        plt.plot(t, dz, label="Production 3")
        plt.legend(loc='lower right')

        plt.subplot(3, 1, 3)
        plt.plot(t, der, label='Ramp Rate 1')
        plt.plot(t, dzr, label='Ramp Rate 3')
        plt.legend(loc='lower right')
        plt.xlabel('Time')
        plt.show()
  
    M = m.options
    consCheck = [
            t, de.value, dh.value, dz.value,
            te.value, th.value, fe.value,
            fh.value, der.value, dzr.value
    ]
    message = M.APPINFO
    if message == 0:
        message = "Optimization terminated successfully"
    feasible, error1, error2 = fs.tri_feasibility(consCheck, tol=1e-6)
    
    # make Gekko solutions into csv to try in other framework
    """  add a # before these quotes to create gekko guesvals
    import pandas as pd
    GuessVal = pd.DataFrame({'Time':t.T})
    GuessVal['de'] = np.array(de.value).T
    GuessVal['dz'] = np.array(dz.value).T
    print(GuessVal.head())
    GuessVal.to_csv('../../data/GekkoSolutions.csv', index=False)
    # """
    info = {
        'Model':'Gekko tri-gen',
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
        'Demand 1': fe,
        'Demand 2': fh,
        'Demand 3': fz,
        'Production 1': de,
        'Production 2': dh,
        'Production 3': dz,
        'Ramp Rate 1': der,
        'Ramp Rate 2': dzr,
        't': t
        }
    
    # Add in the APMonitor data
    info = {**info, **out}
    
    return info, data

if __name__ == "__main__":
    
    # option = 1
    option = 2
    model_name = '3 - Tri-generation'
    
    # plt.close('all')
    
    if option == 1:
        t = np.linspace(0, 1, 101)
        # t = np.linspace(0, 1, 73)
        # t = np.linspace(0, 1, 8)
        info, data = model(t)
        print(info)
        
        # def plot_co_gen(data):
    
        names = {'t': 'Time'}
        names = {'t': 'Time', 'Demand 1': 'Demand 1 ($d_1$)', 'Production 1': 'Production 1 ($g_1$)', 
                 'Demand 2': 'Demand 2 ($d_2$)', 'Production 2': 'Production 2 ($g_2$)', 
                 'Demand 3': 'Demand 3 ($d_3$)', 'Production 3': 'Production 3 ($g_3$)', 
                 'Ramp Rate 1': 'Ramp Rate 1 ($r_1$)', 'Ramp Rate 2': 'Ramp Rate 3 ($r_3$)'}

        df = pd.DataFrame(data).rename(columns=names)
        df = df.set_index(names['t'])
        
        version = 2
        
        if version == 1:
            
            fig, axes = plt.subplots(4, 1, figsize=(7, 6), sharex=True)
        
            ax = axes[0]
            colors = ['r', 'b']
            linestyles = ['-']*2
            for i, col in enumerate(['Demand 1', 'Production 1']):
                df[names[col]].plot(ax=ax, color=colors[i], linestyle=linestyles[i])
            ax.legend()

            ax = axes[1]
            colors = ['r', 'b']
            linestyles = ['--']*2
            for i, col in enumerate(['Demand 2', 'Production 2']):
                df[names[col]].plot(ax=ax, color=colors[i], linestyle=linestyles[i])
            ax.legend()            
    
            ax = axes[2]
            colors = ['r', 'b']
            linestyles = [':']*2
            for i, col in enumerate(['Demand 3', 'Production 3']):
                df[names[col]].plot(ax=ax, color=colors[i], linestyle=linestyles[i], linewidth=2)
            ax.legend(loc='upper left')
            
            ax = axes[3]
            colors = ['k']*2
            linestyles = [':', '--']
            linewidths = [2, 1.5]
            for i, col in enumerate(['Ramp Rate 1', 'Ramp Rate 2']):
                df[names[col]].plot(ax=ax, color=colors[i], linestyle=linestyles[i], linewidth=linewidths[i])
            ax.legend(loc='center')
            
            import matplotlib.ticker as mtick
            for ax in axes:
                ax.grid(linestyle=':')
                ax.xaxis.set_minor_locator(mtick.MultipleLocator(base=0.1))
                ax.yaxis.set_minor_locator(mtick.MultipleLocator(base=1))
            
            plt.tight_layout()
        
        elif version == 2:
            
            nrow = 4
            fig, axes = plt.subplots(nrow, 1, sharex=True, 
                                     figsize=(5, 1.0*nrow + 0.1))
            
            linestyles = ['-', ':']

            ax = axes[0]
            colors = ['r', 'b']
            # linestyles = ['-']*2
            for i, col in enumerate(['Demand 1', 'Production 1']):
                if i == 0:
                    linewidth = 1.5
                elif i == 1:
                    linewidth = 2
                df[names[col]].plot(ax=ax, color=colors[i], linestyle=linestyles[i], linewidth=linewidth)
            
            ax = axes[1]
            colors = ['r', 'b']
            # linestyles = ['--']*2
            for i, col in enumerate(['Demand 2', 'Production 2']):
                if i == 0:
                    linewidth = 1.5
                elif i == 1:
                    linewidth = 2
                df[names[col]].plot(ax=ax, color=colors[i], linestyle=linestyles[i], linewidth=linewidth)
    
            ax = axes[2]
            colors = ['r', 'b']
            # linestyles = [':']*2
            for i, col in enumerate(['Demand 3', 'Production 3']):
                if i == 0:
                    linewidth = 1.5
                elif i == 1:
                    linewidth = 2
                df[names[col]].plot(ax=ax, color=colors[i], linestyle=linestyles[i], linewidth=linewidth)
            
            ax = axes[3]
            colors = ['k']*2
            linestyles = [':', '--']
            linewidths = [2, 1.5]
            for i, col in enumerate(['Ramp Rate 1', 'Ramp Rate 2']):
                df[names[col]].plot(ax=ax, color=colors[i], linestyle=linestyles[i], linewidth=linewidths[i])
            
            import matplotlib.ticker as mtick
            for ax in axes:
                ax.legend(bbox_to_anchor=(1.01, 0.5), loc='center left', frameon=False)
                ax.grid(linestyle=':')
                ax.xaxis.set_minor_locator(mtick.MultipleLocator(base=0.1))
                ax.yaxis.set_minor_locator(mtick.MultipleLocator(base=1))
            
            plt.tight_layout()
        plt.savefig('3-dispatch.pdf', bbox_inches='tight')
        
        #%% Plot oversupply as well
        
        # fig, axes = plt.subplots(5, 1, figsize=(7, 7), sharex=True)
        
        # ax = axes[0]
        # colors = ['r', 'b']
        # linestyles = ['--']*2
        # for i, col in enumerate(['Demand 2', 'Production 2']):
        #     df[names[col]].plot(ax=ax, color=colors[i], linestyle=linestyles[i])
        # ax.legend()
        
        # ax = axes[1]
        # colors = ['r', 'b']
        # linestyles = ['-']*2
        # for i, col in enumerate(['Demand 1', 'Production 1']):
        #     df[names[col]].plot(ax=ax, color=colors[i], linestyle=linestyles[i])
        # ax.legend()

        # ax = axes[2]
        # colors = ['r', 'b']
        # linestyles = [':']*2
        # for i, col in enumerate(['Demand 3', 'Production 3']):
        #     df[names[col]].plot(ax=ax, color=colors[i], linestyle=linestyles[i], linewidth=2)
        # ax.legend(loc='upper left')
        
        # ax = axes[3]
        # colors = ['k']*2
        # linestyles = [':', '--']
        # linewidths = [2, 1.5]
        # for i, col in enumerate(['Ramp Rate 1', 'Ramp Rate 2']):
        #     df[names[col]].plot(ax=ax, color=colors[i], linestyle=linestyles[i], linewidth=linewidths[i])
        # ax.legend(loc='center')

        # ax = axes[4]
        # colors = ['k']*2
        # linestyles = [':', '--']
        # linewidths = [2, 1.5]
        # over1 = df[names['Production 1']] - df[names['Demand 1']]
        # over2 = df[names['Production 2']] - df[names['Demand 2']]
        # over3 = df[names['Production 3']] - df[names['Demand 3']]
        # dp = pd.DataFrame({'Oversupply 1': over1, 'Oversupply 2': over2, 
        #                    'Oversupply 3': over3})
        # linestyles = ['-', '--', ':']
        # for i, col in enumerate(dp):
        #     dp[col].plot(ax=ax, linestyle=linestyles[i])#, linewidth=linewidths[i])
        # ax.legend(loc='center')
        
        # import matplotlib.ticker as mtick
        # for ax in axes:
        #     ax.grid(linestyle=':')
        #     ax.xaxis.set_minor_locator(mtick.MultipleLocator(base=0.1))
        #     ax.yaxis.set_minor_locator(mtick.MultipleLocator(base=1))
        
        # plt.tight_layout()
        # # plt.savefig('3-dispatch.pdf', bbox_inches='tight')
        
        
        #%%
        
    elif option == 2:
    #%%
        df = {}
        d = {}
        imodes = [6, 9]
        for imode in imodes:
            print('iMode: {}'.format(imode))
            # steps = [3, 6, 9, 12, 15] # [5, 10, 20]#, 40, 80]#, 160, 320]
            end = 5 # 7 # 3 # 4 # 5 # 8 # 5 # 7 # 9
            base = 2 # 2
            steps = [int(base**i) for i in range(2, end)]
            df[imode] = {}
            d[imode] = {}
            for n in steps:
                t_end = 1 #/ 2
                t = np.linspace(0, t_end, n+1)
                add = [0.01]#, 0.02]
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
        df = df.set_index(['imode', 'step'])
        
        df['TIME/ITERATION'] = df['time (s)'] / df.ITERATIONS
        
        for imode in imodes:
            for n in steps:
                d[imode][n] = pd.DataFrame(d[imode][n], index=np.arange(n+1+len(add)))
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
        
        if len(steps) == 1:
            idx = 0
        else:
            idx = 1
        
        fig, axes = plt.subplots(len(steps), 2, sharex=True, sharey=True, 
                                 figsize=(8, 6))
        
        if len(axes.shape) == 1:
            axes = np.expand_dims(axes, axis=0)
        
        imode_name = {6: 'Simultaneous', 9: 'Sequential'}
        
        for j, imode in enumerate(imodes):
            for i, step in enumerate(steps):
                ax = axes[i, j]
                    
                dp = d.loc[imode].loc[step].set_index('t')
                dp.plot(ax=ax, legend=False, 
                        marker='.')
                        # marker='o',
                        # markerfacecolor='None', markersize=5)
                if j == 0:
                    ax.set_ylabel('$t_n=${}  '.format(step), rotation=0,
                                  ha='right', va='center')
                if i == 0:
                    ax.set_title(imode_name[imode])
                ax.grid(linestyle=':', alpha=0.6, c='k', linewidth=0.6)
        axes[idx, 1].legend(bbox_to_anchor=(1.05, 0.5), loc='center left',
                          frameon=False)
        util.set_equal_ylim(axes.ravel())
        plt.suptitle(model_name)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.subplots_adjust(hspace=0.25)
        plt.savefig(model_name.replace(' ', '_')+'_data.pdf')

        #%% Plot quadrant of data
        
        util.plot_data(df, model_name)
