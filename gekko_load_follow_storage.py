#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 13:00:42 2021

@author: nathanielgates
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utilities as util
import feasibility as fs
from gekko import GEKKO


def model(t, plot=False, disp=False, ramp=4, imode=6, nodes='', solver=3,
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
    
    m.options.MAX_ITER = 600
    m.options.MAX_ITER = 1000
    
    if max_time == '':
        pass
    else:
        m.options.MAX_TIME = max_time
    
    renewable = 3*np.cos(np.pi*t/6*24)+3 #renewable energy source
    center = np.ones(len(t))
    num = len(t)
    center[0:int(num/4)] = 0
    center[-int(num/4):] = 0
    renewable *= center
    r = m.Param(renewable)
    
    t_first_half = np.r_[np.ones(51), np.zeros(50)]
    t_second_half = np.r_[np.zeros(51), np.ones(50)]
    t_periodic_start = np.zeros(len(t))
    t_periodic_start[0] = t_periodic_start[1] = 1
    t_periodic_end = np.zeros(len(t))
    t_periodic_end[-1] = 1
    t_periodic_start = m.Param(t_periodic_start)
    t_periodic_end = m.Param(t_periodic_end)
    
    # load = m.Param(-2*np.cos(1.75*np.pi*t) + 1.75) # - t_first_half + t_second_half)
    load = m.Param(-2*np.sin(2*np.pi*t) + 7) # - t_first_half + t_second_half)
    # gen = m.Var(2.5)
    gen = m.Var(load[0])

    err = m.CV(0)
    err.STATUS = 1
    err.SPHI = err.SPLO = 0
    err.WSPHI = 1000
    err.WSPLO = 1
    
    dgen = m.MV(0, lb=-ramp, ub=ramp) # ramp rate
    dgen.STATUS = 1
    
    ##### Storage
    s = m.Var(0, lb=0)#, ub=0.75) # storage inventory (Used to be 0.1)
    # s = m.Var(0.1, lb=0, ub=0.7) # storage inventory (Used to be 0.1)
    # Constrain the storage charge/discharge rate
    # and/or max value
    stored = m.SV() # store energy rate
    recovery = m.SV() # recover energy rate
    vx = m.SV(lb=0) # recover slack variable
    vy = m.SV(lb=0) # store slack variable
    
    if imode == 6:
        m.periodic(s) # Makes it infeasible
    # elif imode == 9:
        # m.Equation(s*t_periodic_start == s*t_periodic_end)
        # m.Equation(s*t_periodic_end == 0)
    # m.Obj(1e4*(s[len(t)]-s[0])**2)
    # m.Obj(1e4*s)#*t_periodic_end)
    
    eps = 0.85 # Storage efficiency
    ##### End Storage

    m.Equations([gen.dt() == dgen,  
                 err == load - gen - r + recovery/eps - stored,
                                  
                 gen + r - load == vx - vy,
                 stored == gen + r - load + vy,
                 recovery == load - gen - r + vx,
                 s.dt() == stored - recovery/eps,                 
                 # vx * vy <= 0,
                 # s*t_periodic_start == s*t_periodic_end,
                 stored * recovery <= 0])
    
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
        'r': r,
        'dgen': dgen,
        't': t,
        
        's': s,
        'stored': stored,
        'recovery': recovery,
        'vx': vx,
        'vy': vy
        }
    
    # Add in the APMonitor data
    info = {**info, **out}
    
    return info, data 

#%%

if __name__ == "__main__":
    
    option = 0 # Run model once and plot data
    model_name = '1 - Load Following'
        
    if option == 0:
        imode = 6
        t = np.linspace(0, 1, 101)
        # t = np.linspace(0, 1, 72)
        # imode = 9
        # t = np.linspace(0, 1, 8)
        info, data = model(t, plot=False, disp=True, imode=imode)
        print(info['fcalls'])
        
        util.plot_load_follow_storage_solar(data, version=1)
        
#%%

if 0:
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
        ax.legend(bbox_to_anchor=(1.01, 0.5), loc='center left', frameon=False)
        ax.grid()
        # ax.set_xlim(0, 24)
        # loc = mtick.MultipleLocator(base=6)
        ax.set_xlim(0, 1)
        # loc = mtick.MultipleLocator(base=1)
        # ax.xaxis.set_major_locator(loc)
    
    plt.tight_layout()