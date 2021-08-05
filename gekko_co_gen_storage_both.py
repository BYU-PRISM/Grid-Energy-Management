import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gekko import GEKKO
import utilities as util


def model(t, plot=False, disp=False, ramp=4, imode=6, nodes='', solver=3,
          mv_step_hor='', cv_type=1, max_time='',
           # server='https://gekko.apmonitor.com'):
           server='http://byu.apmonitor.com'):
    
    m = GEKKO(remote=True, server=server)
    m.time = t
    
        
    m.options.SOLVER = solver # 3
    m.options.IMODE = imode # 6
    # m.options.NODES = 2
    if nodes == '':
        pass
    else:
        m.options.NODES = nodes # 4
    # m.options.CV_TYPE = 2 # 1 = Linear penalty from a dead-band trajectory
    # m.options.CV_TYPE = 1 # 1 = Linear penalty from a dead-band trajectory
    m.options.CV_TYPE = cv_type # 1 = Linear penalty from a dead-band trajectory
    m.options.MAX_ITER = 600 #300 # Default is 100
    m.options.MAX_ITER = 1000
    
    if max_time == '':
        pass
    else:
        m.options.MAX_TIME = max_time
        
    p = m.SV(10) # production (constant)
    #p.STATUS = 0
    s = m.Var(0.1, lb=0) # storage inventory
    stored = m.SV() # store energy rate
    recovery = m.SV() # recover energy rate
    vx = m.SV(lb=0) # recover slack variable
    vy = m.SV(lb=0) # store slack variable
    
    # m.periodic(s)
    # m.Obj(1e4*(s[len(t)]-s[0])**2)
        
    eps = 0.85 # Storage efficiency
        
    d = m.MV((-20*np.sin(np.pi*t/12*24)+100)/10)
    d_h = m.MV((15*np.cos(np.pi*t/12*24)+150)/10)
    
    p_h_initial = m.Intermediate(p*1.5)
    
    p_h = m.SV(p_h_initial)
    s_h = m.Var(0.5,lb=0)
    stored_h = m.SV()
    recovery_h = m.SV()
    
    renewable = (20*np.cos(np.pi*t/6*24)+20)/10 #renewable energy source
    center = np.ones(len(t))
    num = len(t)
    center[0:int(num/4)] = 0
    center[-int(num/4):] = 0
    renewable *= center
    r = m.Param(renewable)
    
    r1 = m.MV(ub=3,lb=-3)
    r1.STATUS=1
    
    m.periodic(s_h)
    
    zx = m.SV(lb=0)
    zy = m.SV(lb=0)
    
    eps_h = 0.8 # heat storage efficiency
    
    
    m.Equations([p + r + recovery/eps - stored >= d,
                 p + r - d == vx - vy,
                 stored == p + r - d + vy,
                 recovery == d - p - r + vx,
                 s.dt() == stored - recovery/eps,
                 p.dt() == r1,
                 # vx * vy <= 0,
                 stored * recovery <= 0,
                 p_h + recovery_h/eps_h - stored_h >= d_h,
                 p_h - d_h == zx - zy,
                 stored_h == p_h - d_h + zy,
                 recovery_h == d_h - p_h + zx,
                 s_h.dt() == stored_h - recovery_h/eps_h,
                 stored_h * recovery_h <= 0,
                 p_h == 1.5 * p
                 ]
               )
    m.Minimize(p)

    # Solve the optimization model (enforces disp=True)
    txt = util.solve_and_get_txt(m)
    
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
        'Demand 1': d,
        'Demand 2': d_h,
        'Production 1': p,
        'Production 2': p_h,
        'Storage 1': s,
        'Stored 1': stored,
        'Recovered 1': recovery,
        'Slack 1x': vx,
        'Slack 1y': vy,
        'Storage 2': s_h,
        'Stored 2': stored_h,
        'Recovered 2': recovery_h,
        'Slack 2x': zx,
        'Slack 2y': zy,
        'Ramp Rate 1': r1,
        'Renewable': r,
        't': t
        }
    
    # Add in the APMonitor data
    info = {**info, **out}
    
    return info, data    

#%%

if __name__ == '__main__':
    
    # Solve model
    t = np.linspace(0, 24, 24*3+1)/24 # 25)
    t = np.linspace(0, 1, 73)
    # t = np.linspace(0, 1, 101)
    # t = np.linspace(0, 1, 8)
    info, data = model(t, plot=False, disp=True, nodes=2, cv_type=2)
    print(info['fcalls'])
    
    t = np.array(data['t'])
    d = np.array(data['Demand 1'])
    p = np.array(data['Production 1'])
    d_h = np.array(data['Demand 2'])
    p_h = np.array(data['Production 2'])
    r = np.array(data['Renewable'])
    r1 = np.array(data['Ramp Rate 1'])
    s = np.array(data['Storage 1'])
    stored = np.array(data['Stored 1'])
    recovery = np.array(data['Recovered 1'])
    s_h = np.array(data['Storage 2'])
    stored_h = np.array(data['Stored 2'])
    recovery_h = np.array(data['Recovered 2'])
    
    #%%
        
    # Plot solution
    fig, axes = plt.subplots(5, 1, figsize=(5, 5.1), sharex=True)
    axes = axes.ravel()
    
    ax = axes[0]
    ax.plot(t, d, 'r-', label='Demand 1 ($d_1$)')
    ax.plot(t, p,'b:', label='Production 1 ($g_1$)', linewidth=2)
    # ax.plot(t,n[:-1], 'k--', label='Net ($d_1-R_1$)')
    ax.plot(t, d - r, 'k--', label='Net ($d_1-R_1$)')
    
    ax = axes[1]
    ax.plot(t,r, 'b-',label='Source 1 ($R_1$)')
    ax.plot(t,r1, 'k--', label='Ramp Rate ($r$)')
    
    ax = axes[2]
    ax.plot(t,s, 'k-', label='Storage 1 ($e_1$)')
    ax.plot(t,stored,'g--',label='Stored ($e_{in,1}$)')
    ax.plot(t,recovery,'b:',label='Recovered ($e_{out,1}$)', linewidth=2)
    
    ax = axes[3]
    ax.plot(t,d_h, 'r-', label='Demand 2 ($d_2$)')
    ax.plot(t[1:], p_h[1:],'b:', label='Production 2 ($g_2$)',
            linewidth=2)
    #ax.plot(t[1:],p_prod[1:], 'b--',label='Supplemental Heat Production')
    
    ax = axes[4]
    ax.plot(t,s_h, 'k-', label='Storage 2 ($e_2$)')
    ax.plot(t,stored_h,'g--',label='Stored ($e_{in,2}$)')
    ax.plot(t[1:],recovery_h[1:],'b:',label='Recovered ($e_{out,2}$)', 
            linewidth=2)
    
    ax.set_xlabel('Time')
    
    for ax in axes:
        ax.legend(loc='center left',bbox_to_anchor = (1,0.5),frameon=False)
        ax.grid()
        ax.set_xlim(0, 1)
    
    plt.tight_layout()
    
    plt.savefig('6-energy-storage.pdf', bbox_inches = 'tight')