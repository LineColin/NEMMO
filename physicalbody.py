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
    Set the parameters for the sun.
    Be careful to change the distance to the sun
    """
    RADIUS = 700e3
    TEMP = 5780

@dataclass
class PhysicalBody:
    """
    Defines the physical parameters of the study object
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

    T_E: float = 1600  # eutectic temperature
    P: float = 2150
    SIGMA: float = 5.67e-8  # Stephan-Boltzman cst
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
        Calculation of the rayleigh number

        """
        return self.ALPHA*self.rho*self.gravity*dT*d**3 /(mu*K)
          
    
@dataclass
class CoreCooling(PhysicalBody):
    """
    Defines core parameters
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
        Uniform core cooling
        """
        def cooling_func(dr: float, T0: float, T: float) -> float:
            a = - 3 * 2 * self.k * (T - T0) / (self.RHO_CORE * self.CP_CORE * self.radius * dr)
            b = + 3 * self.H / (4 * np.pi * self.radius**3 * self.RHO_CORE * self.CP_CORE)
        
            return a + b
        
        return cooling_func
    
@dataclass
class Cumulates(PhysicalBody):
    """
    Defines cumulates parameters
    The methods oh this class are only valid for the first phase.
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
        super().__post_init__()  # Appeler la methode __post_init__ de la classe parente
        self.r = self.r_core
        self.V = 4 * np.pi * (self.r_body ** 3 - self.r ** 3) / 3
        self.T = self.M * self.c + self.P
        self.h_lmo = self.H_LMO
        #print(self.h_lmo)
        self.h = self.D*self.h_lmo
        self.K = self.k/(self.rho*self.CP)
        
   
    def temperature_surface(self, Ts: float, T: float) -> float:
        """
        calcul of the surface temperature using heat balance between radiativ flux
        and convection flux in the magma ocean (limit boundary layer theory)

        """
        cste = self.k*(self.ALPHA*self.rho*self.gravity/\
                       (self.K*self.VISCOSITY*self.RAC))**(self.BETA)
    
        return self.SIGMA*(Ts**4 - self.T_EQ**4) - cste*(T - Ts)**(4/3)
    
    def dfdt(self, y, t):
        c = y[0]
        R = y[1]
        H = self.h_lmo * self.V * np.exp(- self.HEAT_DECAY * t / 3.15e7)
        a = self.rho * self.LATENT_HEAT + self.rho * self.CP * self.P
        drdt = (self.SIGMA * self.EMISSIVITY * (self.Ts ** 4 - self.T_EQ ** 4) * self.r_body ** 2 - H) / (a * R ** 2)
        dcdt = c * 3 * drdt * R ** 2 / (self.r_body ** 3 - R ** 3)
        f = np.array([dcdt, drdt])
        return f
    
class Crust(PhysicalBody):
    
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
        super().__post_init__()  # Appeler la méthode __post_init__ de la classe parente
        self.k = self.k_crust
        self.r = self.r_body - self.dr
        self.V = 4 * np.pi * self.r_body**2 * self.dr
        self.K = self.k/(self.rho*self.CP)
        
        
    def temperature_surface(self, Ts, T, dy):
        return self.SIGMA*(Ts**4 - self.T_EQ**4) + 2*self.k*(Ts - (T*(self.T_E - self.T_EQ) + self.T_EQ))/dy
    
    

