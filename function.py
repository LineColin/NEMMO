#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 16:56:21 2024

@author: linecolin
"""

import numpy as np


def euler(dt: float, dr: float, T0: float, T: float, cooling_func: float) -> float:
    """
    Euler method for temperature evolution
    """
    return T + dt * cooling_func(dr, T0, T)
    
# =================================

def newton(f, x, T, *args):
    h=1.0E-6
    epsilon=1.0E-8
    NbIterationMax = 100
    n = 0
    while (np.abs(f(x, T, *args)) > epsilon) and (n < NbIterationMax):
        f_prime = (f(x + h, T, *args) - f(x - h, T, *args)) / (2 * h)
        x = x - f(x, T, *args) / f_prime
        n += 1
    return x if n < NbIterationMax else None

# =================================

def rk4(dfdt, y, t, dt):
    k1 = dfdt(y, t)
    k2 = dfdt(y + k1 * dt / 2, t + dt / 2)
    k3 = dfdt(y + k2 * dt / 2, t + dt / 2)
    k4 = dfdt(y + k3 * dt / 2, t + dt / 2)
    return y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

# =================================

def calculate_integral(T, radius, coef, R_MO):
    
    num_intervals = len(radius)
    r_values = radius

    # Calcul de la valeur de l'intégrale dans chaque sous-intervalle
    integral_values = []
    for i in range(num_intervals - 1):
        r = r_values[i]
        dr = r_values[i+1] - r_values[i]
        integral_value = T[i] * R_MO**2 * dr
        integral_values.append(integral_value * coef)

    return integral_values

# =================================
def non_uniform_grid(start, stop, num_points, edge_factor=1/5):
    # Générer des points uniformément espacés
    linear_points = np.linspace(start, stop, num_points)
    
    # Appliquer une fonction non linéaire pour augmenter le nombre de points aux bords
    non_linear_points = start + (stop - start) * np.sin(np.linspace(0, np.pi/2, num_points))**edge_factor
    
    return non_linear_points
# =================================

from scipy.integrate import quad
from scipy.misc import derivative

def calculate_expression(R, R_moon, T):
    
    def integrand(r):
        return T * 4 * np.pi * r**2

    def derivative_func(r):
        return T * 4 * np.pi * r**2

    integral_result, _ = quad(integrand, R, R_moon)
    #derivative_result = derivative(integral_result, R)

    return integral_result

# =================================


def heat_production_distribution(radius, h0=25e-12, D=1e-3, r_top=1737e3, r_bot=390e3, rho=3.3e3):
    HPE_0 = h0*D*rho
    return HPE_0 * ((r_top**3 - r_bot**3)/(r_top**3 - radius**3))**(1-D)


# =================================

def non_uniform_grid(start, stop, num_points, edge_factor=1/2):
    # Générer des points uniformément espacés
    linear_points = np.linspace(start, stop, num_points)
    
    # Appliquer une fonction non linéaire pour augmenter le nombre de points aux bords
    non_linear_points = start + (stop - start) * np.sin(np.linspace(0, np.pi/2, num_points))**edge_factor
    
    return non_linear_points
    
# =================================

def diffusion(y, n, dt, K, T_top, T_bot, R_top, R_bot, dy):

    #NOT TESTED

    """
    Construction of the diffusion matrix using finite volume method

    Parameters
    ----------
    y : array(n)
        rescaling of the radius
    n : int
        len of array
    dt : float
        time step
    dy : float
        space step
    K : float
        thermal diffusivity
    T_top : float
        temperature at the top boundary.
    T_bot : float
        temperature at the bottom.
    R_top : float
        radius at the top.
    R_bot : float
        radius at the bottom.

    Returns
    -------
    M : array((n,n))
        matrix for the diffusion problem
    R : array(n)
        rest
    r_center : array(n)
        dimensional center of the cells

    """


    rest = np.zeros(n)

    L = R_top - R_bot

    # cells boundary

    y_boundary = np.linspace(y[0], y[-1], n+1)#non_uniform_grid(y[0], y[-1], n+1)
    
    r_boundary = (y_boundary - 1)*(R_top - R_bot) + R_bot
    #y_boundary = (r_boundary - R_bot)/(R_top - R_bot) + 1
    
#    print(dy)

    # cells center

    y_center = (y_boundary[:-1] + y_boundary[1:])/2
    

    r_center = (y_center - 1)*(R_top - R_bot) + R_bot
    #dy = np.mean(np.abs(np.diff(r_center)))

    s = dt*K / (dy*dy*r_center*r_center * L**2)

    rest[0] = 2*T_top*r_boundary[0]*r_boundary[0]*s[0]
    rest[-1] = 2*T_bot*r_boundary[-1]*r_boundary[-1]*s[-1]

    a = s*r_boundary[1:]*r_boundary[1:]
    b = - s*r_boundary[1:]*r_boundary[1:] - s*r_boundary[:-1]*r_boundary[:-1]
    c = s*r_boundary[:-1]*r_boundary[:-1]


    M = (np.diag(c[1:], -1) + np.diag(b, 0) + np.diag(a[:-1], +1))

    M[0,0] = - 2*s[0]*r_boundary[0]*r_boundary[0] - s[0]*r_boundary[1]*r_boundary[1]
    M[0,1] = s[0]*r_boundary[1]*r_boundary[1]

    M[-1, -2] = s[-1]*r_boundary[-2]*r_boundary[-2]
    M[-1, -1] = - s[-1]*r_boundary[-2]*r_boundary[-2] - 2*s[-1]*r_boundary[-1]*r_boundary[-1]

    return M, rest, r_center

# =================================

def advection(u, dy, dt):
    # NOT TESTED
    
    """
    Advection matrix using #mettre la méthode
    Parameters
    ----------
    u : array
        matrix of the advection speed
    dy : float
        space step
    dt : float
        time step

    Returns
    -------
    A : array((n,n))
        matrix of advection

    """
    
    u_abs = np.abs(u)
    f = dt/(dy*4)
    
    a_a = -f*(u[1:] + u_abs[1:])
    b_a = f*3*(u_abs - u)
    c_a = f*(5*u[:-1] - 3*u_abs[:-1])
    d_a = f*(u_abs[:-2] - u[:-2])
    
    A = (np.diag(a_a, -1) + np.diag(b_a, 0) + np.diag(c_a, +1) + np.diag(d_a, +2))
        
#    A[0,0] = f*3*(u_abs[0] - u[0]) - f*3*(u_abs[0] - u[0])
#    A[0,1] = f*(5*u[0] - 3*u_abs[0])
#    A[0,2] = f*(u_abs[0] - u[0])
#
#    A[-1,-2] = -f*(u[-1] + u_abs[-1])
#    A[-1,-1] = f*3*(u_abs[-1] - u[-1]) - f*(5*u[-1] - 3*u_abs[-1])
    A[0, 0] = b_a[0] - a_a[0]
    A[-1, -1] = b_a[-1] - c_a[-1] -  2 * d_a[-1]*np.abs(dy) - d_a[-1]
    
    rtop = a_a[0]
    rbot = c_a[-1] + d_a[-1] * dy
    
    
    #M = sparse.csc_matrix(M)
    
    return A, rtop, rbot

# =================================

def F_plus(gamma, eps, T, u, gN, gN2, flux="normal", B=None):
    u_abs = np.abs(u)
    Tm = np.zeros_like(T)
    Tp = np.zeros_like(T)
    nx = len(T)
    Fp = np.zeros_like(T)
    if flux=="normal":
         for i in range(nx - 2):
             Tm[i] = T[i] + 0.5 * gamma(eps[i]) * (T[i+1] - T[i])
             Tp[i] = T[i+1] - 0.5 * gamma(eps[i+1]) * (T[i+2] - T[i+1])
             Fp[i] = 0.5 * u[i] * (Tp[i] + Tm[i]) - 0.5 * u_abs[i] * (Tp[i] - Tm[i])
         Tp[-2] = T[-1] - 0.5 * gamma(eps[-1]) * (gN - T[-1])
         Tm[-2] =  T[-2] + 0.5 * gamma(eps[-2]) * (T[-1] - T[-2])
         Fp[-2] = 0.5 * u[-2] * (Tp[-2] + Tm[-2]) - 0.5 * u_abs[-2] * (Tp[-2] - Tm[-2])
    
         Tm[-1] = T[-1] + 0.5 * gamma(eps[-1]) * (gN - T[-1])
         epsN = (gN - T[-1])/(gN2 - gN)
         Tp[-1] = gN - 0.5 * gamma(epsN) * (gN2 - gN)
         Fp[-1] = 0.5 * u[-1] * (Tp[-1] + Tm[-1]) - 0.5 * u_abs[-1] * (Tp[-1] - Tm[-1])
#    else:
#        for i in range(nx - 2):
#             Tm[i] = T[i] + 0.5 * gamma(eps[i], B[i]) * (T[i+1] - T[i])
#             Tp[i] = T[i+1] - 0.5 * gamma(eps[i+1], B[i+1]) * (T[i+2] - T[i+1])
#             Fp[i] = 0.5 * u[i] * (Tp[i] + Tm[i]) - 0.5 * u_abs[i] * (Tp[i] - Tm[i])
#          Tp[-2] = T[-1] - 0.5 * gamma(eps[-1], B[-1]) * (gN - T[-1])
#          Tm[-2] =  T[-2] + 0.5 * gamma(eps[-2], B[-2]) * (T[-1] - T[-2])
#          Fp[-2] = 0.5 * u[-2] * (Tp[-2] + Tm[-2]) - 0.5 * u_abs[-2] * (Tp[-2] - Tm[-2])
#    
#         Tm[-1] = T[-1] + 0.5 * gamma(eps[-1], B[-1]) * (gN - T[-1])
#         epsN = (gN - T[-1])
#         BN = (gN2 - gN)
#         Tp[-1] = gN - 0.5 * gamma(epsN, BN) * (gN2 - gN)
#         Fp[-1] = 0.5 * u[-1] * (Tp[-1] + Tm[-1]) - 0.5 * u_abs[-1] * (Tp[-1] - Tm[-1])
    return Fp

def F_minus(gamma, eps, T, u, g0, g1, gN):
    u_abs = np.abs(u)
    Tm = np.zeros_like(T)
    Tp = np.zeros_like(T)
    nx = len(T)
    Fm = np.zeros_like(T)
    for i in range(1, nx-1):
        Tm[i] = T[i-1] + 0.5 * gamma(eps[i-1])*(T[i] - T[i-1])
        Tp[i] = T[i] - 0.5*gamma(eps[i]) * (T[i+1] - T[i])
        Fm[i] = 0.5 * u[i] * (Tp[i] + Tm[i]) - 0.5 * u_abs[i] * (Tp[i] - Tm[i])

    eps1 = (g0 - g1)/(T[0] - g0)
    Tm[0] = g0 + 0.5 * gamma(eps1)*(T[0] - g0)
    Tp[0] = T[0] - 0.5*gamma(eps[0]) * (T[1] - T[0])
    Fm[0] = 0.5 * u[0] * (Tp[0] + Tm[0]) - 0.5 * u_abs[0] * (Tp[0] - Tm[0])
    Tm[-1] = T[-2] + 0.5 * gamma(eps[-2])*(T[-1] - T[-2])
    Tp[-1] = T[-1] - 0.5*gamma(eps[-1]) * (gN - T[-1])
    Fm[-1] = 0.5 * u[-1] * (Tp[-1] + Tm[-1]) - 0.5 * u_abs[-1] * (Tp[-1] - Tm[-1])
    return Fm
