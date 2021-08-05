
import numpy as np


def load_feasibility(gen_time, tol=1e-8):
    gen = gen_time[0]
    time = gen_time[1]
    r = np.zeros(len(time))
    r[0] = 0
    dt = time[1] - time[0]
    for i in range(1, len(time)):
        r[i] = (gen[i] - gen[i-1])/dt
    # r = gen_time[2]
    # r = np.diff(gen)/np.diff(time)
    # r = np.gradient(gen, time)
    feasible = True
    err = np.zeros(len(r))
    for i in range(len(r)):
        err[i] = r[i]**2 - 1
    # eliminate error from within constraints
    err = np.clip(err, 0, None)
    mess1 = sum(err)
    if sum(err) > tol:
        feasible = False
    # for i in range(len(r)):
    #   if r[i] > 1.0 or r[i] < 1.0:
    #       isTrue = False
    return feasible, mess1


def co_feasibility(consCheck, tol=1e-7):
    #consCheck is [time, gen1, gen2]
    # r = np.diff(consCheck[1])/np.diff(consCheck[0])
    time = consCheck[0]
    g1 = consCheck[1]
    g2 = consCheck[2]
    # r = consCheck[3]
    r = np.zeros(len(time))
    r[0] = 0
    dt = time[1] - time[0]
    for i in range(1, len(time)):
        r[i] = (g1[i] - g1[i-1])/dt
    feasible = True
    err = np.zeros(len(r))
    for i in range(len(r)):
        err[i] = r[i]**2 - 1
    err = np.clip(err, 0, None)
    mess1 = sum(err)
    err2 = np.zeros(len(g1))
    for i in range(len(g1)):
        err2[i] += (g2[i] - (2*g1[i]))**2
        # if r2 > 2 or r2 < -2:
        #   isTrue = False
    residual = sum(err) + sum(err2)
    mess2 = residual
    if sum(err) > tol:
        feasible = False
    return feasible, mess1, mess2


def tri_feasibility(consCheck, tol=1e-7):
    #consCheck is [time, gen1, gen2, gen3, tdemand1, tdemand2, load1, load2]
    # r = np.diff(consCheck[1])/np.diff(consCheck[0])
    # r3 = np.diff(consCheck[3])/np.diff(consCheck[0])
    time = consCheck[0]
    g1 = consCheck[1]
    g2 = consCheck[2]
    g3 = consCheck[3]
    r = np.zeros(len(time))
    r[0] = 0
    dt = time[1] - time[0]
    for i in range(1, len(time)):
        r[i] = (g1[i] - g1[i-1])/dt
    r3 = np.zeros(len(time))
    r3[0] = 0
    for i in range(1, len(time)):
        r3[i] = (g3[i] - g3[i-1])/dt
    dtot1 = consCheck[4]
    dtot2 = consCheck[5]
    d1 = consCheck[6]
    d2 = consCheck[7]
    # r = consCheck[8]
    # r3 = consCheck[9]
    feasible = True
    err = np.zeros(len(r))
    err2 = np.zeros(len(r))
    err3 = np.zeros(len(g1))
    for i in range(len(r)):
        err[i] = r[i]**2 - 1
    err = np.clip(err, 0, None)
    for i in range(len(r)):
        err2[i] = r3[i]**2 - 1
    err2 = np.clip(err2, 0, None)
    mess1 = sum(err) + sum(err2)
    for i in range(len(g1)):
        err3[i] += (g2[i] - (2*g1[i]))**2
        err3[i] += (dtot1[i] - d1[i] - 2*g3[i])**2
        err3[i] += (dtot2[i] - d2[i] - 3*g3[i])**2
    residual = sum(err) + sum(err2) + sum(err3)
    mess2 = residual
    if sum(err) > tol:
        feasible = False
    return feasible, mess1, mess2
