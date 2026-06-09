#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 17:40:09 2023

@author: linecolin
"""
from __future__ import annotations

import typing
from dataclasses import dataclass


if typing.TYPE_CHECKING:
    from typing import Callable, Optional#,Mapping, , Union

    from numpy.typing import NDArray

from function import rk4, euler, newton, diffusion, advection
import numpy as np
from scipy import interpolate
import scipy.sparse.linalg as LA
import scipy.sparse


@dataclass
class Sun:
    """
    Stores the physical constants of the Sun used to compute the equilibrium
    temperature of the body.

    These values are fixed and should be modified if the model is applied
    to a different stellar system.

    Attributes
    ----------
    RADIUS : float
        Solar radius [m], default 700e3
    TEMP : float
        Effective surface temperature of the Sun [K], default 5780
    """

    RADIUS = 700e3
    TEMP = 5780

@dataclass
class PhysicalBody:
    """
    Defines the physical parameters and derived quantities of the planetary body
    under study.

    This class is the base class for all components of the model (magma ocean,
    crust, cumulates, core). It computes at initialisation the equilibrium
    temperature with the Sun, the liquidus slope, and the initial heat
    production of the magma ocean.

    Parameters
    ----------
    r_body : float
        Radius of the body [m]
    r_core : float
        Core radius [m]
    albedo : float
        Bond albedo of the surface [-]
    rho : float
        Mean density [kg/m^3]
    gravity : float
        Surface gravitational acceleration [m/s^2]
    initial_heat_production : float
        Initial bulk heat production rate [W/kg]
    c0 : float
        Initial anorthite fraction in the magma ocean [-]
    ce : float
        Eutectic anorthite fraction [-]
    k_crust : float
        Thermal conductivity of the crust [W/m/K]
    D : float
        Partition coefficient for heat-producing elements [-]
    heat_source : bool
        False for a global magma ocean, True for a non-global magma ocean
    r_flottability : float or None
        Radius at the base of the non-global magma ocean [m];
        None if the magma ocean is global
    distance_sun_object : float
        Distance from the Sun to the body [m]
    n_factor : int
        Resolution factor for the cumulate grid [-]
    overturn : bool or None
        False: no overturn; True: with overturn;
        None: no overturn and no flux from cumulates
    func_overturn : callable or None
        Function returning the overturn heat flux as a function of time [W]
    dir_save : str
        Directory path for saving output files

    Attributes (computed at initialisation)
    ----------------------------------------
    VOLUME : float
        Volume of the mantle shell [m^3]
    M : float
        Liquidus slope in the system [K]
    T_EQ : float
        Radiative equilibrium temperature with the Sun [K]
    H_LMO : float
        Volumetric heat production of the magma ocean [W/m^3]

    Constants
    ---------
    T_E : float
        Eutectic temperature [K], default 1600
    SIGMA : float
        Stefan–Boltzmann constant [W/m^2/K^4]
    EMISSIVITY : float
        Surface emissivity [-], default 1.0
    ALPHA : float
        Thermal expansivity [K^-1], default 5e-5
    CP : float
        Heat capacity [J/kg/K], default 1000
    LATENT_HEAT : float
        Latent heat of crystallisation [J/kg], default 5e5
    HEAT_DECAY : float
        Radioactive decay constant [s^-1]
    """

    r_body: float
    r_core: float
    albedo: float
    rho: float
    gravity: float
    initial_heat_production: float
    c0: float
    ce: float
    k_crust: float
    D: float
    heat_source: bool
    r_flottability: float
    distance_sun_object: float
    n_factor: int
    overturn: Optional[bool]
    t_overturn: float

    T_E: float = 1600
    P: float = 2150
    SIGMA: float = 5.67e-8
    EMISSIVITY: float = 1.0
    ALPHA: float = 5E-5
    CP: float = 1000
    LATENT_HEAT: float = 5e5
    VISCOSITY: float = 1
    INITIAL_SURFACE_TEMPERATURE: float = 1173
    HEAT_DECAY = np.abs(np.log(21/25) / 300e6)

    def __post_init__(self):
        self.VOLUME = 4 * np.pi * (self.r_body**3 - self.r_core**3) / 3
        self.M = - (self.P - self.T_E) / self.ce
        self.T_EQ = Sun.TEMP * np.sqrt(Sun.RADIUS / (2 * self.distance_sun_object)) * (1 - self.albedo) ** 0.25
        self.c  = self.c0
        if self.heat_source == True:
            self.H_LMO = self.initial_heat_production * self.rho * ((self.r_body**3 - self.r_core**3)/(self.r_body**3 - self.r_flottability**3))
            self.r_core = self.r_flottability
        else:
            self.H_LMO = self.initial_heat_production * self.rho
            self.r_flottability = self.r_core

        
    def rayleigh(self, dT: float, d: float, mu: float, K: float) -> float:
        """
        Computes the Rayleigh number for convective instability assessment.

        Parameters
        ----------
        dT : float
            Temperature contrast across the layer [K]
        d : float
            Thickness of the layer [m]
        mu : float
            Dynamic viscosity [Pa·s]
        K : float
            Thermal diffusivity [m^2/s]

        Returns
        -------
        float
            Rayleigh number [-]
        """
        return self.ALPHA*self.rho*self.gravity*dT*d**3 /(mu*K)
          
    
@dataclass
class CoreCooling(PhysicalBody):
    """
    Models the thermal evolution of the core as a uniformly cooling sphere.

    Inherits all parameters from PhysicalBody and adds core-specific thermal
    properties. The core loses heat by conduction through its boundary with
    the mantle. Its internal heat production is set to zero for a global
    magma ocean, or proportionally allocated for a non-global one.

    Attributes
    ----------
    CP_CORE : float
        Heat capacity of core material [J/kg/K], default 840
    k : float
        Thermal conductivity of core material [W/m/K], default 4
    RHO_CORE : float
        Density of core material [kg/m^3], default 7200
    T : float or None
        Current core temperature [K]
    """
    
    CP_CORE: float = 840
    k: float = 4
    RHO_CORE: float = 7200
    T: float = None
    
    def __post_init__(self):
        super().__post_init__()
        self.radius = self.r_core
        if self.heat_source == True:
            self.H = self.initial_heat_production * self.rho * ((self.r_core**3 - self.r_flottability**3)/(self.r_body**3 - self.r_flottability**3))
            self.r_core = self.r_flottability
        else:
            self.H = 0
        
    
    def cooling(self) -> Callable[[float, float, float, float], float]:
        """
        Returns the function describing the rate of temperature change of the core.
        See section 3.2.2 and eqs. 17 and 18.

        Returns
        -------
        callable
            Function cooling_func(dr, T0, T) -> dT/dt [K/s]
        """
        def cooling_func(dr: float, T0: float, T: float) -> float:
            a = - 3 * 2 * self.k * (T - T0) / (self.RHO_CORE * self.CP_CORE * self.radius * dr)
            b = + 3 * self.H / (4 * np.pi * self.radius**3 * self.RHO_CORE * self.CP_CORE)
        
            return a + b
        
        return cooling_func
    
@dataclass
class Cumulates(PhysicalBody):
    """
    Models the cumulate growing at the base of the magma ocean
    during Stage 1.

    As the magma ocean cools, crystals settle and accumulate above the core,
    progressively reducing the volume of liquid. This class tracks the
    temporal evolution of the cumulate radius, the anorthite composition,
    and the surface temperature.

    Attributes
    ----------
    k : float
        Thermal conductivity [W/m/K], default 4
    viscosity : float
        Dynamic viscosity of the magma ocean [Pa·s], default 1e17
    Ts : float
        Surface temperature [K], initialised to INITIAL_SURFACE_TEMPERATURE
    RAC : float
        Critical Rayleigh number for the boundary layer [-], default 1500
    BETA : float
        Exponent of the convective heat flux scaling law [-], default 1/3
    T : float
        Liquidus temperature at current composition [K]
    h_lmo : float
        Volumetric heat production of the magma ocean [W/m^3]
    h : float
        Volumetric heat production of the cumulates [W/m^3]
    """
    
    k: float = 4
    viscosity: float = 1e17
    Ts: float = PhysicalBody.INITIAL_SURFACE_TEMPERATURE
    V: Optional[float] = None
    T: float = None
    RAC: float = 1500
    BETA: float = 1/3
    
    dr: float = None
    T_: NDArray = None
    y: NDArray = None
    dT: float = None
    dr_dt: float = None
    

    def __post_init__(self):
        super().__post_init__() 
        self.r = self.r_core
        self.V = 4 * np.pi * (self.r_body ** 3 - self.r ** 3) / 3
        self.T = self.M * self.c + self.P
        self.h_lmo = self.H_LMO
        #print(self.h_lmo)
        self.h = self.D*self.h_lmo
        self.K = self.k/(self.rho*self.CP)
        
   
    def temperature_surface(self, Ts: float, T: float) -> float:
        """
        Evaluates the surface energy balance used to determine the surface temperature.

        Parameters
        ----------
        Ts : float
            Surface temperature (unknown to solve for) [K]
        T : float
            Interior magma ocean temperature [K]

        Returns
        -------
        float
            Residual of the energy balance [W/m^2]
        """
        cste = self.k*(self.ALPHA*self.rho*self.gravity/\
                       (self.K*self.VISCOSITY*self.RAC))**(self.BETA)
    
        return self.SIGMA*(Ts**4 - self.T_EQ**4) - cste*(T - Ts)**(4/3)
    
    def dfdt(self, y, t):
        """
        Defines the coupled ODE system governing Stage 1 solidification.

        Parameters
        ----------
        y : ndarray, shape (2,)
            State vector [c, R], where c is the anorthite fraction [-]
            and R is the cumulate radius [m]
        t : float
            Current time [s]

        Returns
        -------
        ndarray, shape (2,)
            Time derivatives [dc/dt, dR/dt]
        """

        c = y[0]
        R = y[1]
        H = self.h_lmo * np.exp(- self.HEAT_DECAY * t / 3.15e7) * 4 * np.pi * (self.r_body**3 - R**3)/3
        a = self.rho * self.LATENT_HEAT + self.rho * self.CP * (self.P - (self.P + self.M * self.c0 * (self.r_body**3 -self.r_core**3)/(self.r_body**3 - R**3)))
        drdt = (self.SIGMA * self.EMISSIVITY * (self.Ts ** 4 - self.T_EQ ** 4) * 4 * np.pi * self.r_body ** 2 - H) / (4 * np.pi * a * R ** 2)
        dcdt = c * 3 * drdt * R ** 2 / (self.r_body ** 3 - R ** 3)
        f = np.array([dcdt, drdt])

        return f
    
class Crust(PhysicalBody):

    """
    Models the anorthosite flotation crust forming at the surface of the
    magma ocean during Stage 2.

    Unlike the cumulates which settle at the base, anorthosite crystals are
    less dense than the magma ocean and float upward, forming an insulating lid.
    This class tracks the growth of this lid and the conductive temperature
    profile within it.

    Attributes
    ----------
    k : float
        Thermal conductivity of the crust [W/m/K], set from k_crust
    r : float
        Inner radius of the crust (base) [m]
    dr : float
        Initial crust thickness [m], default 100
    K : float
        Thermal diffusivity of the crust [m^2/s]
    T_ : ndarray or None
        Temperature profile within the crust [K]
    hr : ndarray or None
        Heat production profile within the crust [W/m^3]
    """
    
    k: float = None
    viscosity: float = 1e17
    Ts: float = None
    V: Optional[float] = None
    T: float = None
    dr: float = 100
    r: float = None
    dT: float = None
    dr_dt: float = -1e-8
    hr: NDArray = None
    h: float = None
    T_: NDArray = None
    y: NDArray = None
    
    def __post_init__(self):
        super().__post_init__()  
        self.k = self.k_crust
        self.r = self.r_body - self.dr
        self.V = 4 * np.pi * self.r_body**2 * self.dr
        self.K = self.k/(self.rho*self.CP)
        
        
    def temperature_surface(self, Ts, T, dy):
        """
        Evaluates the surface energy balance for the crusted stage (Stage 2).

        Parameters
        ----------
        Ts : float
            Surface temperature (unknown to solve for) [K]
        T : float
            Non-dimensional temperature at the base of the crust [-]
        dy : float
            Grid spacing at the surface [m]

        Returns
        -------
        float
            Residual of the energy balance [W/m^2]; zero at the solution
        """
        return self.SIGMA*(Ts**4 - self.T_EQ**4) + 2*self.k*(Ts - (T*(self.T_E - self.T_EQ) + self.T_EQ))/dy
    
    

