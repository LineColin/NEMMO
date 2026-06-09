#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 16:56:21 2024

@author: linecolin
"""

import numpy as np
from scipy.integrate import quad
from scipy.misc import derivative


def euler(dt: float, dr: float, T0: float, T: float, cooling_func: float) -> float:
    """
    Approximates the temperature evolution using the explicit Euler method.

    Advances the temperature T by one time step dt using the cooling function
    provided. Suitable for slowly varying systems where simplicity is preferred
    over accuracy.

    Parameters
    ----------
    dt : float
        Time step [s]
    dr : float
        Spatial step [m]
    T0 : float
        Reference temperature [K]
    T : float
        Current temperature [K]
    cooling_func : callable
        Function returning the time derivative of temperature dT/dt [K/s]

    Returns
    -------
    float
        Temperature at the next time step [K]
    """
    return T + dt * cooling_func(dr, T0, T)
    
# =================================

def newton(f, x, T, *args):
    """
    Finds the root of f(x, T) using the Newton-Raphson iterative method.

    Used to solve the implicit energy balance equation at the surface, where
    the surface temperature cannot be expressed analytically. Convergence is
    reached when |f(x)| < 1e-8 or after 100 iterations.

    Parameters
    ----------
    f : callable
        Function whose root is sought, of the form f(x, T, *args)
    x : float
        Initial guess for the root (typically a temperature [K])
    T : float
        Current magma ocean temperature [K]
    *args : 
        Additional arguments passed to f

    Returns
    -------
    float or None
        Root of f if convergence is achieved, None otherwise
    """
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
    """
    Advances y by one time step using the 4th-order Runge-Kutta method.

    Provides higher accuracy than Euler for the same time step, at the cost
    of 4 evaluations of dfdt per step. Recommended for stiff or rapidly
    varying thermal evolution equations.

    Parameters
    ----------
    dfdt : callable
        Function returning dy/dt, of the form dfdt(y, t)
    y : float or ndarray
        Current state variable (e.g. temperature [K] and composition c)
    t : float
        Current time [s]
    dt : float
        Time step [s]

    Returns
    -------
    float or ndarray
        State variable at the next time step
    """

    k1 = dfdt(y, t)
    k2 = dfdt(y + k1 * dt / 2, t + dt / 2)
    k3 = dfdt(y + k2 * dt / 2, t + dt / 2)
    k4 = dfdt(y + k3 * dt / 2, t + dt / 2)

    return y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

# =================================

def diffusion(y, n, dt, K, T_top, T_bot, R_top, R_bot, dy):

    """
    Builds the implicit diffusion matrix for heat conduction in a spherical shell.

    Discretises the spherical heat equation using a finite volume method on a
    rescaled coordinate y. Boundary conditions are of Dirichlet type: fixed
    temperatures T_top and T_bot are imposed at the top and bottom of the shell.
    Intended for use in the conductive crust or cumulate layers.

    Parameters
    ----------
    y : ndarray
        Rescaled radial coordinate (dimensionless)
    n : int
        Number of cells
    dt : float
        Time step [s]
    K : float
        Thermal diffusivity [m^2/s]
    T_top : float
        Temperature at the upper boundary [K]
    T_bot : float
        Temperature at the lower boundary [K]
    R_top : float
        Physical radius at the top of the layer [m]
    R_bot : float
        Physical radius at the bottom of the layer [m]
    dy : float
        Rescaled spatial step [-]

    Returns
    -------
    M : ndarray, shape (n, n)
        Diffusion matrix
    rest : ndarray, shape (n,)
        Boundary condition vector
    r_center : ndarray, shape (n,)
        Physical radius at cell centres [m]
        
    """


    rest = np.zeros(n)

    L = R_top - R_bot

    y_boundary = np.linspace(y[0], y[-1], n+1)
    
    r_boundary = (y_boundary - 1)*(R_top - R_bot) + R_bot

    y_center = (y_boundary[:-1] + y_boundary[1:])/2
    

    r_center = (y_center - 1)*(R_top - R_bot) + R_bot

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
    """
    Builds the advection matrix for thermal transport in the magma ocean.

    Discretises the advection equation using an upwind scheme to ensure
    numerical stability. The sign of the velocity field u determines the
    direction of heat transport at each grid point.

    Parameters
    ----------
    u : ndarray
        Advection velocity field [m/s]
    dy : float
        Spatial step in rescaled coordinates [-]
    dt : float
        Time step [s]

    Returns
    -------
    A : ndarray, shape (n, n)
        Advection matrix
    rtop : float
        Boundary flux contribution at the top
    rbot : float
        Boundary flux contribution at the bottom

    Notes
    -----
    NOT TESTED
    """
    
    u_abs = np.abs(u)
    f = dt/(dy*4)
    
    a_a = -f*(u[1:] + u_abs[1:])
    b_a = f*3*(u_abs - u)
    c_a = f*(5*u[:-1] - 3*u_abs[:-1])
    d_a = f*(u_abs[:-2] - u[:-2])
    
    A = (np.diag(a_a, -1) + np.diag(b_a, 0) + np.diag(c_a, +1) + np.diag(d_a, +2))
    A[0, 0] = b_a[0] - a_a[0]
    A[-1, -1] = b_a[-1] - c_a[-1] -  2 * d_a[-1]*np.abs(dy) - d_a[-1]
    
    rtop = a_a[0]
    rbot = c_a[-1] + d_a[-1] * dy
    
    return A, rtop, rbot

# =================================

def F_plus(gamma, eps, T, u, gN, gN2, flux="normal", B=None):
    """
    Computes the upwind numerical flux F+ at each cell interface.

    Evaluates the positive-direction component of the advective flux using
    a flux limiter gamma to reduce numerical diffusion while preserving
    monotonicity near sharp gradients (e.g. thermal fronts).

    Parameters
    ----------
    gamma : callable
        Flux limiter function, of the form gamma(eps)
    eps : ndarray
        Ratio of consecutive gradients (smoothness indicator) [-]
    T : ndarray
        Temperature field [K]
    u : ndarray
        Advection velocity [m/s]
    gN : float
        Ghost cell temperature at the upper boundary [K]
    gN2 : float
        Second ghost cell temperature beyond the upper boundary [K]
    flux : str, optional
        Flux scheme selector, default "normal"
    B : ndarray or None, optional
        Reserved for alternative flux schemes, default None

    Returns
    -------
    ndarray
        Upwind flux F+ at each grid point [K·m/s]
    """
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

    return Fp

def F_minus(gamma, eps, T, u, g0, g1, gN):
    """
    Computes the upwind numerical flux F- at each cell interface.

    Evaluates the negative-direction component of the advective flux,
    symmetric to F_plus. Together, F+ and F- form the complete flux-limiter
    scheme for advective heat transport.

    Parameters
    ----------
    gamma : callable
        Flux limiter function, of the form gamma(eps)
    eps : ndarray
        Ratio of consecutive gradients (smoothness indicator) [-]
    T : ndarray
        Temperature field [K]
    u : ndarray
        Advection velocity [m/s]
    g0 : float
        Ghost cell temperature at the lower boundary [K]
    g1 : float
        Second ghost cell temperature beyond the lower boundary [K]
    gN : float
        Ghost cell temperature at the upper boundary [K]

    Returns
    -------
    ndarray
        Upwind flux F- at each grid point [K·m/s]
    """
    
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
