import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams, cm
from mpl_toolkits.mplot3d import Axes3D
from scipy import io, integrate
from sklearn import linear_model
import os


## Kuramoto-Sivashinsky equation (from Trefethen)
## u_t = -u*u_x - u_xx - nu*u_xxxx,  periodic BCs

def ksEquation (u_0, dt, t_end, t_max=10, nu=0.005, M = 16):
    
    if t_end > t_max:
        t_end = t_max

    N = len(u_0)
    v = np.fft.fft(u_0)

    # Spatial grid and initial condition:
    h = dt
    k = np.hstack((np.arange(0,N/2), 0, np.arange(-N/2+1,0)))
    L = k**2 - nu*k**4
    E = np.exp(h*L)
    E2 = np.exp(h*L/2)
    r = (np.exp(1j*np.pi*(np.arange(1,M+1) - .5)/M))
    LR = h*np.repeat(L.reshape((-1,1)), M, axis=1) + np.repeat(r.reshape((1,-1)),N,axis=0)

    Q = h*np.real(np.average((np.exp(LR/2)-1)/LR, axis=1))
    f1 = h*np.real(np.average((-4-LR+np.exp(LR)*(4-3*LR+LR**2))/LR**3, axis = 1))
    f2 = h*np.real(np.average((2+LR+np.exp(LR)*(-2+LR))/LR**3, axis = 1))
    f3 = h*np.real(np.average((-4-3*LR-LR**2+np.exp(LR)*(4-LR))/LR**3, axis = 1))

    # Main time-stepping loop:
    uu = u_0
    tt = 0
    nmax = int(t_max/h)
    nplt = np.floor((t_max/1000)/h)
    g = -0.5j*k
    tt = np.zeros(nmax)
    uu = np.zeros((N,nmax))

    for n in range(nmax):
        t = n*h
        Nv = g*np.fft.fft(np.real(np.fft.ifft(v))**2)
        a = E2*v + Q*Nv
        Na = g*np.fft.fft(np.real(np.fft.ifft(a))**2)
        b = E2*v + Q*Na
        Nb = g*np.fft.fft(np.real(np.fft.ifft(b))**2)
        c = E2*a + Q*(2*Nb-Nv)
        Nc = g*np.fft.fft(np.real(np.fft.ifft(c))**2)
        v = E*v + Nv*f1 + 2*(Na+Nb)*f2 + Nc*f3     
        if np.mod(n,nplt)==0:
            u = np.real(np.fft.ifft(v))
            uu[:,n] = u
            tt[n] = t

    cutoff = tt > 0
    cutoff = np.logical_and(cutoff, tt<t_end)

    t_sol = tt[cutoff]
    u_sol = (uu[:,cutoff]).T

    return u_sol, t_sol


def LorenzEquation (X_0, t, sigma, beta, rho):

    def eq_rhs(X,t):

        x, y, z = X
        x_d = sigma * (y - x)
        y_d = x * (rho - z) - y
        z_d = x * y - beta * z
        
        return [x_d, y_d, z_d]

    X = integrate.odeint(eq_rhs, X_0, t, rtol= 1e-11, atol=1e-10)

    return X    

def LorenzFixedPoints(sigma, beta, rho):

    X0 = np.array([np.sqrt(beta*(rho-1)), np.sqrt(beta*(rho-1)), rho-1])
    X1 = np.array([-np.sqrt(beta*(rho-1)), -np.sqrt(beta*(rho-1)), rho-1])
    X2 = np.array([0,0,0]) #-(rho+sigma)

    return X0, X1, X2

def LorenzSphere (sigma, rho):

    u = np.linspace(0, 2 * np.pi, 1000)
    v = np.linspace(0, np.pi, 1000)
    x = rho * np.outer(np.cos(u), np.sin(v))
    y = rho * np.outer(np.sin(u), np.sin(v))
    z_p = rho * np.outer(np.ones(np.size(u)), np.cos(v))
    z = z_p + rho + sigma

    return x,y,z

def LorenzEllipsoid (beta, rho):

    u = np.linspace(0, 2 * np.pi, 1000)
    v = np.linspace(0, np.pi, 1000)

    coeff = np.array([np.sqrt(rho*beta), np.sqrt([rho**2*beta]), rho])

    x = coeff[0] * np.outer(np.cos(u), np.sin(v))
    y = coeff[1] * np.outer(np.sin(u), np.sin(v))
    z_p = coeff[2] * np.outer(np.ones(np.size(u)), np.cos(v))

    z = z_p+rho

    return x,y,z



