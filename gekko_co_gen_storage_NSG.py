import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gekko import GEKKO

m = GEKKO(remote=True)
t = np.linspace(0, 24, 24*3+1)/24 # 25)
m.time = t
    
m.options.SOLVER = 3
m.options.IMODE = 6
m.options.NODES = 2
m.options.CV_TYPE = 2 # 1 = Linear penalty from a dead-band trajectory
m.options.MAX_ITER = 400 #300 # Default is 100
    
p = m.FV() # production (constant)
p.STATUS = 1
s = m.Var(0.1, lb=0) # storage inventory
stored = m.SV() # store energy rate
recovery = m.SV() # recover energy rate
vx = m.SV(lb=0) # recover slack variable
vy = m.SV(lb=0) # store slack variable
    
#m.periodic(s)
# m.Obj(1e4*(s[len(t)]-s[0])**2)
    
eps = 0.7 # Storage efficiency
    
d = m.MV(-20*np.sin(np.pi*t/12*24)+100)
d_h = m.MV(15*np.cos(np.pi*t/12*24)+150)

p_h_initial = m.Intermediate(p*1.5)

p_h = m.SV()
s_h = m.Var(5,lb=0)
stored_h = m.SV()
recovery_h = m.SV()

renewable = 30*np.cos(np.pi*t/6*24)+30 #renewable energy source
center = np.ones(len(t))
num = len(t)
center[0:int(num/4)] = 0
center[-int(num/4):] = 0
renewable *= center
r = m.Param(renewable)


#m.free_initial(p_h)

#m.periodic(s_h)

zx = m.SV(lb=0)
zy = m.SV(lb=0)

eps_h = 0.8 # heat storage efficiency


m.Equations([p + r + recovery/eps - stored >= d,
                p + r - d == vx - vy,
                stored == p + r - d + vy,
                recovery == d - p - r + vx,
                s.dt() == stored - recovery/eps,
                # vx * vy <= 0,
                stored * recovery <= 0,
                p_h + recovery_h/eps_h - stored_h >= d_h,
                p_h - d_h == zx - zy,
                stored_h == p_h - d_h + zy,
                recovery_h == d_h - p_h +zx,
                s_h.dt() == stored_h - recovery_h/eps_h,
                stored_h * recovery_h <= 0,
                p_h == 1.5 * p]
           )
m.Obj(p)
m.solve()

plt.figure(figsize=(10,10))
plt.subplot(5,2,1)
plt.plot(t,p,'r--',label='Power Production')
plt.plot(t,d, 'k-', label='Power Demand')
plt.plot(t,r, 'c-', label='Renewable Production')
plt.legend()
plt.subplot(5,2,2)
plt.plot(t[1:],p_h[1:],'r--',label='Heat Production')
plt.plot(t,d_h, 'k-', label='Heat Demand')
plt.legend()
plt.subplot(5,2,3)
plt.plot(t,stored,'r-',label='Stored Energy')
plt.plot(t,recovery,'b-.',label='Recovered Energy')
plt.legend()
plt.subplot(5,2,4)
plt.plot(t,stored_h,'r-',label='Stored Heat')
plt.plot(t[1:],recovery_h[1:],'b-.',label='Recovered Heat')
plt.legend()
plt.subplot(5,2,5)
plt.plot(t,s, 'g-', label='Energy Inventory')
plt.xlabel('Time')
plt.legend()
plt.subplot(5,2,6)
plt.plot(t,s_h, 'g-', label='Heat Inventory')
plt.xlabel('Time')
plt.legend()
plt.show()
