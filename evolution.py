#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 09:55:34 2023

@author: linecolin
"""
from __future__ import annotations

import typing
from dataclasses import dataclass


if typing.TYPE_CHECKING:
    from typing import Callable, Optional#,Mapping, , Union

    from numpy.typing import NDArray

from function import rk4, euler, newton, diffusion, advection, F_plus, F_minus
from physicalbody import Sun, PhysicalBody, Cumulates, CoreCooling, Crust
import numpy as np
from scipy import interpolate
import scipy.sparse.linalg as LA
import scipy.sparse
import matplotlib.pyplot as plt
np.seterr(divide='ignore', invalid='ignore')
import time

@dataclass
class Stage1Analysis(Cumulates):
    """
    Class for the analysis of the first stage of crystallisation
    """
    
    t: float = 0
    
    
    def __post_init__(self, *args, **kwargs):
        super().__post_init__(*args, **kwargs)
        
        self.h = self.D*self.h_lmo
        self.hr_history = [self.h]
        self.r_history = [self.r]
        self.T_history = [self.T]
        self.Ts_history = [self.Ts]
        self.time_history = [self.t]
        self.h_lmo_history = [self.h_lmo]
        self.h_solid_history = [0]
        self.dV_history = [0]
        self.profilT = {}
        self.profilr = {}
        self.dr_history = [0]
        
    def update_volume(self):
        return 4*np.pi*(self.r_body**3 - self.r**3)/3
        
    def update_temperature(self):
        return self.M * self.c + self.P
        
    def update_heat_producing(self, dr):
        dV = 4*np.pi*dr*self.r**2
        phi = dV/self.update_volume()
        return self.h_lmo/(1 - phi + self.D*phi)

    def evolve_r(self, time_step):
        
        R = self.r
        self.V = self.update_volume()
             
        y = np.array([self.c, self.r])
        
        self.c, self.r = rk4(self.dfdt, y, self.t, time_step)
        self.r_history.append(self.r)
        self.dr = self.r - R
        self.t += time_step
        self.time_history.append(self.t)
        dV = 4*np.pi*self.dr*self.r**2
        self.dV_history.append(dV)
        self.dr_history.append(self.dr/time_step)
        #d = self.r - self.r_core
        self.h_lmo = self.update_heat_producing(self.dr)
        self.h = self.D*self.h_lmo
        self.hr_history.append(self.h)
       
             
        self.T = self.update_temperature()
             
        self.Ts = newton(self.temperature_surface, self.Ts, self.T)
        self.Ts_history.append(self.Ts)
        self.T_history.append(self.T)
        
        self.h_lmo_history.append(self.h_lmo)
        
        return dV
        
    def run_stage1_analysis(self, time_step):
        a = 0
        i = 0
        T0 = self.T
        while self.T > PhysicalBody.T_E:
            dV = self.evolve_r(time_step)
            a += self.D * self.h_lmo * dV
            self.h_solid_history.append(a)
            
            if i>=51:
                n_it = int(i/50)
                if len(self.T_history[::n_it])==51:
                    self.profilT[f"T, i={i}"] = [(T - self.T)/(T0 - self.T) for T in self.T_history[::n_it]]
                    r = np.linspace(self.r_core, self.r, 51)
                    self.profilr[f"r, i={i}"] = (r - self.r_core)/(self.r - self.r_core) + 1
                
            i += 1

    def get_time_history(self):
        return np.array(self.time_history)
    
    def get_r_history(self):
        return np.array(self.r_history)
    
    def get_T_history(self):
        return np.array(self.T_history)
    
    def get_Ts_history(self):
        return np.array(self.Ts_history)
    
    def get_hr_history(self):
        return np.array(self.hr_history)
    
    def get_h_solid_history(self):
        return np.array(self.h_solid_history)
    
    def get_h_lmo_history(self):
        return np.array(self.h_lmo_history)
    
    def get_dV_history(self):
        return np.array(self.dV_history)
    
    def get_T_profil(self):
        return self.profilT
    
@dataclass
class Stage2Analysis(PhysicalBody):
    """
    Class for the analysis of the second stage
    """
    
    n: int = 100
    cfl: float = 0.4
    t: float = None
    h_lmo: float = None
    dt: float = None
    Qmax:float = 0
    decay: float = 0
    Q:float = 0
    t0:float = 0
    
    def __post_init__(self):
        super().__post_init__()
        self.core = CoreCooling(self.r_body, self.r_core, self.albedo, self.rho, self.gravity, self.initial_heat_production, self.c0, self.ce, self.k_crust, self.D, self.heat_source, self.r_flottability, self.distance_sun_object, self.n_factor, self.overturn, self.t_overturn)
        self.solid = Cumulates(self.r_body, self.r_core, self.albedo, self.rho, self.gravity, self.initial_heat_production, self.c0, self.ce, self.k_crust, self.D, self.heat_source, self.r_flottability, self.distance_sun_object, self.n_factor, self.overturn, self.t_overturn)
        self.crust = Crust(self.r_body, self.r_core, self.albedo, self.rho, self.gravity, self.initial_heat_production, self.c0, self.ce, self.k_crust, self.D, self.heat_source, self.r_flottability, self.distance_sun_object, self.n_factor, self.overturn, self.t_overturn)
        self.stage1 = Stage1Analysis(self.r_body, self.r_core, self.albedo, self.rho, self.gravity, self.initial_heat_production, self.c0, self.ce, self.k_crust, self.D, self.heat_source, self.r_flottability, self.distance_sun_object, self.n_factor, self.overturn, self.t_overturn)
        
        
        
        #data storage
        
        self.T_core_history = []
        self.r_solid_history = []
        self.r_crust_history = []
        self.t_history = []
        self.dt_history = []
        self.h_lmo_history = []
        self.h_crust_history = []
        self.h_solid_history = []
        self.Ts_history = []
        self.r_profil_crust = []
        self.T_profil_crust = []
        self.r_profil_solid = []
        self.T_profil_solid = []
        self.hr_profil_crust = []
        self.hr_profil_solid = []
        self.drdt_crust_history = []
        self.drdt_solid_history = []
        self.flux_crust = []
        self.flux_solid = []
        self.flux_hlmo = []
        self.flux_lat = []
        self.dv_crust_history = []
        self.dV_solid_history = []
        self.Tbot_crust = []
        self.Ttop_solid = []
        self.flux_overturn = []
    
    
    def T_analytique(self):
        return lambda x: self.P + self.M*(self.c0*(self.r_body**3 - self.r_core**3)/(self.r_body**3 - x**3))
        
    def initialisation(self, time_step=1e6):
        
        self.stage1.run_stage1_analysis(time_step)
        
        self.t = self.stage1.get_time_history()[-1]
        t0 = self.crust.dr ** 2 *self.LATENT_HEAT * self.rho / (2 * self.crust.k * self.T_E)
        self.crust.dr_dt = self.crust.dr/t0
        self.t += t0
        self.t0 = self.t
        print(f"time stage 1: {self.t/3.15e7} years")
        self.crust.Ts = self.stage1.get_Ts_history()[-1]
        self.core.T = self.stage1.get_T_history()[0]
        self.solid.r = self.stage1.get_r_history()[-1]
        print(f"radius stage 1: {self.solid.r/1000} km")
        
        dV_crust = 4 * np.pi * (self.r_body**3 - self.crust.r**3)/3
        dV_solid = (1 - self.ce) * dV_crust / self.ce
        V_lmo = 4 * np.pi * (self.r_body**3 - self.solid.r**3)
        phi_crust = dV_crust / V_lmo
        phi_solid = dV_solid / V_lmo
        V_lmo = 4 * np.pi * (self.crust.r - self.solid.r)
        self.solid.dr_dt = 1
        
        self.crust.dT = self.T_E - self.crust.T_EQ
        self.solid.dT = self.core.T - self.T_E
        self.h_lmo = self.stage1.get_h_lmo_history()[-1] / (self.D * phi_crust \
                      + self.D * phi_solid + 1 - phi_crust - phi_solid)
            
        self.crust.h = self.D * self.h_lmo
        self.solid.h = self.D*self.h_lmo
        
        T0_crust = np.linspace(self.crust.Ts, self.T_E, self.n+1)
        T0_crust = (T0_crust[1:] + T0_crust[:-1])/2
        r_crust = np.linspace(self.r_body, self.crust.r, self.n+1)
        rc_crust = (r_crust[1:] + r_crust[:-1])/2
        T_solid  = interpolate.interp1d(self.stage1.get_r_history(), self.stage1.get_T_history())
        r_solid = np.linspace(self.r_core, self.solid.r, self.n_factor*self.n+1)
        self.r_solid = (r_solid[1:] + r_solid[:-1])/2
        T0_solid = T_solid(self.r_solid)
        
        self.crust.T_ = (T0_crust - self.T_EQ) / self.crust.dT
        self.solid.T_ = (T0_solid - self.T_E) / self.solid.dT
        print("Ttop", self.solid.T_[-1])
        
        self.crust.hr = np.ones(self.n) * self.crust.h#np.concatenate((lin1, lin2))#np.linspace(0, self.crust.h, self.n)#
        solid_hr = interpolate.interp1d(self.stage1.get_r_history(), self.stage1.get_hr_history())
        self.solid.hr = solid_hr(self.r_solid)
        
        #save data
        self.T_core_history.append(self.core.T)
        self.r_solid_history.append(self.solid.r)
        self.r_crust_history.append(self.crust.r)
        self.t_history.append(self.t)
        self.h_lmo_history.append(self.h_lmo)
        self.h_crust_history.append(self.crust.h)
        self.h_solid_history.append(self.solid.h)
        self.Ts_history.append(self.crust.Ts)
        self.flux_hlmo.append(self.h_lmo * np.exp(-self.HEAT_DECAY * self.t/3.1536e7) * 4 * np.pi * (self.crust.r**3 - self.solid.r**3)/3)
        self.dv_crust_history.append(dV_crust)
        self.dV_solid_history.append(dV_solid)
        self.Tbot_crust.append(self.crust.T_[-1])
        self.Ttop_solid.append(self.solid.T_[-1])
        
        #plot initial profils
        
#        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
#        ax1.plot(self.solid.T_, self.r_solid, 'g-')
#        ax2.plot(self.solid.hr, self.r_solid, 'b-')
#        ax1.set_xlabel('temperature')
#        ax1.set_ylabel('radius')
#        ax2.set_xlabel('h(r)')
#        
    
    def update_time(self):
        
        dt1 = self.cfl*self.crust.dr**2 /self.crust.K
        dt2 = self.cfl*self.crust.dr/np.abs(self.crust.dr_dt)
        #dt3 = self.cfl*self.solid.dr/np.abs(self.solid.dr_dt)
        
        self.dt = min([dt1, dt2])#, dt3])
    
    def update_radius(self, V_lmo):
        
        a =  self.crust.k*self.crust.dT * 2 * (1 - self.crust.T_[-1]) /(self.crust.dr)
        
        if self.overturn == None:
            b = - self.solid.k * self.solid.r**2 * 2 * self.solid.dT * self.solid.T_[-1] \
                /(self.solid.dr * self.crust.r**2)
        elif self.overturn == True:
            b = - self.Qmax * np.exp(-self.decay * self.t)/(4 * np.pi * self.crust.r**2)
        elif self.overturn == False:
            b = 0

        c = - self.h_lmo * np.exp(-self.HEAT_DECAY * self.t/3.1536e7) * V_lmo\
                /(4 * np.pi * self.crust.r**2)
        self.crust.dr_dt = - (a + b + c) * self.ce / (self.rho * self.LATENT_HEAT)
                
        d = - (1 - self.ce)/self.ce
        self.solid.dr_dt = self.crust.dr_dt * d * self.crust.r**2 / self.solid.r**2
    
    def update_dy(self):
            r_crust = np.linspace(self.r_body, self.crust.r, self.n+1)
            r_crust = (r_crust[1:] + r_crust[:-1])/2
            self.crust.y = (r_crust - self.crust.r)/(self.r_body - self.crust.r) + 1
            r_solid = np.linspace(self.r_core, self.solid.r, self.n_factor*self.n+1)
            r_solid = (r_solid[1:] + r_solid[:-1])/2
            self.solid.y = (r_solid - self.r_core)/(self.solid.r - self.r_core) + 1
            
            self.crust.dr = (self.r_body - self.crust.r)/self.n
            self.solid.dr = (self.solid.r - self.r_core)/(self.n_factor*self.n)
            
            return -1/self.n, 1/(self.n_factor*self.n)
        
    def run_stage2_analysis(self):
        
        print("start stage 2")
        debut = time.time()
        self.initialisation()
        reset = False
        
        Ttop_crust = 0
        Tbot_crust = 1
        Ttop_solid = 1
        Tbot_solid = 0
        
        r_crust = np.linspace(self.r_body, self.crust.r, self.n+1)
        r_crust = (r_crust[1:] + r_crust[:-1])/2
        self.crust.y = (r_crust - self.crust.r)/(self.r_body - self.crust.r) + 1
        r_solid = np.linspace(self.r_core, self.solid.r, self.n_factor*self.n+1)
        r_solid = (r_solid[1:] + r_solid[:-1])/2
        self.solid.y = (r_solid - self.r_core)/(self.solid.r - self.r_core) + 1
            
        self.crust.dr = (self.r_body - self.crust.r)/self.n
        self.solid.dr = (self.solid.r - self.r_core)/(self.n_factor*self.n)
        dy_crust = -1/self.n
        dy_solid = 1/(self.n_factor*self.n)
        
        acrust = 0
        asolid = np.trapz(4 * np.pi * self.solid.hr * r_solid **2, r_solid)
        
        self.dt = self.cfl*self.crust.dr**2 /self.crust.K
        
        i = 1
        I = np.identity(self.n)
#        figc1, (ax1c, ax2c) = plt.subplots(1, 2, sharey=True)
#        figc, axc = plt.subplots(1, 1, sharey=True)
#        figs1, (ax1s, ax2s) = plt.subplots(1, 2, sharey=True)
#        figs, axs = plt.subplots(1, 1, sharey=True)
#        figh, (axhc, axhs) = plt.subplots(1, 2)
        print(self.CP)
        while self.crust.r - self.solid.r > 500 and self.t/3.15e13<500:
            
            self.crust.dT = self.T_E - self.crust.Ts
                
            if self.crust.Ts > self.T_EQ:
                self.crust.Ts = newton(self.crust.temperature_surface, self.crust.Ts, self.crust.T_[1], self.crust.dr)
                
            V_lmo = 4*np.pi*(self.crust.r**3 - self.solid.r**3)/3
            
            self.update_time()
            self.t += self.dt
            self.update_radius(V_lmo)
            
            self.crust.dV =  - 4 * np.pi * self.crust.dr_dt * self.dt * self.crust.r**2
            self.solid.dV = 4 * np.pi * self.solid.dr_dt * self.dt * self.solid.r**2
            
            #print(self.solid.dr_dt * self.dt)
            self.crust.r += self.dt * self.crust.dr_dt
            self.solid.r += self.dt * self.solid.dr_dt

            dy_crust, dy_solid = self.update_dy()
            
            u_crust =  self.crust.dr_dt*(self.crust.y - 2)/(self.r_body - self.crust.r)
            u_solid = - self.solid.dr_dt*(self.solid.y - 1)/(self.solid.r - self.r_core)

            phi_crust = self.crust.dV/V_lmo
            phi_solid = self.solid.dV/V_lmo
            self.h_lmo = self.h_lmo/(self.D*phi_crust + self.D*phi_solid + 1 - phi_crust - phi_solid)
            
            self.crust.h = self.D*self.h_lmo
            self.solid.h = self.D*self.h_lmo
            
            gamma = lambda x :max(0, min(1, 2*x), min(2, x)) #superbee
            if self.D!=0:
           #advection of HPE in the crust
                #self.crust.hr[-1] = (2 * self.crust.h + self.crust.hr[-2])/3
                gamma_crust = lambda x : 1
                hn_crust = self.crust.hr.copy()
                eps = np.ones_like(hn_crust)
#                eps[1:-1] = (hn_crust[1:-1] - hn_crust[:-2])/(hn_crust[2:] - hn_crust[1:-1])
                gh0 = hn_crust[0]
                gh1 = hn_crust[0]
                ghN = 2 * self.crust.h - self.crust.hr[-1]#(hn_crust[-1] - hn_crust[-2]) + hn_crust[-1]
                ghN2 = (ghN - hn_crust[-1]) + ghN
#                print(ghN, ghN2)
#                eps[0] = (hn_crust[0] - gh0)/(hn_crust[1] - hn_crust[0])
#                eps[-1] = (hn_crust[-1] - hn_crust[-2])/(ghN - hn_crust[-1])
                Fph_c = F_plus(gamma_crust, eps, hn_crust, u_crust, ghN, ghN2)
                Fmh_c = F_minus(gamma_crust, eps, hn_crust, u_crust, gh0, gh1, ghN)
                self.crust.hr = hn_crust - self.dt * (Fph_c - Fmh_c)/dy_crust
                
#                advection in cumulates
                
                ghN2 = (ghN - self.solid.hr[-1]) + ghN
                ghN = 2 * self.solid.h - self.solid.hr[-1]#(hn_solid[-1] - hn_solid[-2]) + hn_solid[-1]
                #self.solid.hr[-1] = (2 * self.solid.h + self.solid.hr[-2])/3
                hn_solid = self.solid.hr.copy()
                eps = np.ones_like(hn_solid)
                eps[1:-1] = (hn_solid[1:-1] - hn_solid[:-2])/(hn_solid[2:] - hn_solid[1:-1])
                gh0 = hn_solid[0]
                gh1 = gh0
                
                
                #print(self.solid.hr[-1], ghN, ghN2)
                eps[0] = 0
                eps[-1] = (hn_solid[-1] - hn_solid[-2])/(ghN - hn_solid[-1])
                Fp_s = F_plus(gamma, eps, hn_solid, u_solid, ghN, ghN2)
                Fm_s = F_minus(gamma, eps, hn_solid, u_solid, gh0, gh1, ghN)
                self.solid.hr = hn_solid - self.dt * (Fp_s - Fm_s)/dy_solid
            
            #print(u_solid)
            
            D_crust, res_crust, rc_crust = diffusion([2., 1.], self.n, self.dt, self.crust.K, \
                                                      Ttop_crust, Tbot_crust, self.r_body, self.crust.r, dy_crust)
            Tn_crust = self.crust.T_.copy()
            eps = np.ones_like(Tn_crust)
            eps[1:-1] = (Tn_crust[1:-1] - Tn_crust[:-2])/(Tn_crust[2:] - Tn_crust[1:-1])
            g1_c = 2 * Ttop_crust - self.crust.T_[0]
            g2_c = 2 * g1_c  #2 * (Ttop_crust - self.crust.T_[0]) * dy_crust + self.crust.T_[0]
            gNc = 2 * Tbot_crust - self.crust.T_[-1]
            gN2 = 2 * gNc# 2 * (Tbot_crust - self.crust.T_[-1]) + self.crust.T[-1]
            eps[-1] = (Tn_crust[-1] - Tn_crust[-2])/(gNc - Tn_crust[-1])
            Fp_crust = F_plus(gamma, eps, Tn_crust, u_crust, gNc, gN2)
            Fm_crust = F_minus(gamma, eps, Tn_crust, u_crust, g1_c, g2_c, gNc)
            Tn_crust = Tn_crust - self.dt * (Fp_crust - Fm_crust) / dy_crust
            V_crust = Tn_crust + res_crust + (self.dt/(self.rho*self.CP*self.crust.dT))*self.crust.hr*np.exp(-self.HEAT_DECAY*self.t/3.15E7)
            M_crust = I - D_crust
            M_crust = scipy.sparse.csc_matrix(M_crust)
            self.crust.T_ = LA.spsolve(M_crust, V_crust)
            
            
            D_solid, res_solid, rc_solid = diffusion([1., 2.], self.n_factor*self.n, self.dt, self.solid.K, \
                                            Ttop_solid, Tbot_solid, self.solid.r, self.r_core, dy_solid)
            
            Tn_solid = self.solid.T_.copy()
            eps = np.ones_like(Tn_solid)
            eps[1:-1] = (Tn_solid[1:-1] - Tn_solid[:-2])/(Tn_solid[2:] - Tn_solid[1:-1])
            g1_s = 2 * 1 - self.solid.T_[0]
            g2_s = 2 * g1_s#2 * (1 - self.solid.T_[0]) * dy_solid + self.solid.T_[0]
            gN = 2 * 0 - self.solid.T_[-1]
            gN2 = 2 * gN#2 * (0 - self.solid.T_[-1]) * dy_solid + self.solid.T_[-1]
            eps[-1] = (Tn_solid[-1] - Tn_solid[-2])/(gN - Tn_solid[-2])
            Fp_solid = F_plus(gamma, eps, Tn_solid, u_solid, gN, gN2)
            Fm_solid = F_minus(gamma, eps, Tn_solid, u_solid, g1_s, g2_s, gN)
            Tn_solid = Tn_solid - self.dt * (Fp_solid - Fm_solid) / dy_solid
            V_solid = Tn_solid + res_solid + (self.dt/(self.rho*self.CP*self.solid.dT)\
                *self.solid.hr*np.exp(-self.HEAT_DECAY*self.t/3.15E7))
            M_solid = np.identity(self.n_factor*self.n) - D_solid
            M_solid = scipy.sparse.csc_matrix(M_solid)

            self.solid.T_ = LA.spsolve(M_solid, V_solid)
            
           
            
            Ttop_crust = (self.crust.Ts - self.T_EQ)/(self.T_E - self.T_EQ)
            Tbot_var = self.solid.T_[0]*(self.core.T - self.T_E) + self.T_E
            
            cooling_func = self.core.cooling()
            self.core.T = euler(dt=self.dt, dr=self.solid.dr, T0=Tbot_var, T=self.core.T, cooling_func=cooling_func)

            self.solid.T_[0] = (Tbot_var - self.T_E)/(self.core.T - self.T_E)
            self.solid.dT = self.core.T - self.T_E
            
            acrust += self.crust.h*self.crust.dV
            asolid += self.solid.h*self.solid.dV
            
            if (self.t - self.t0)  / 3.15e7 > 0.1 and not reset:
                temp = self.solid.T_ * self.solid.dT + self.T_E
                func = 4 * np.pi * self.rho * self.CP * (temp - self.T_E) * rc_solid **2
                dr = np.diff(rc_solid)
                self.Q = np.trapz(func, dx=dr)
                self.Qmax = (self.rho * self.LATENT_HEAT * 4 * np.pi * (- self.crust.r**2 * self.crust.dr_dt + self.solid.r**2 * self.solid.dr_dt))/self.t_overturn
                self.decay = self.Qmax/self.Q
                reset = True
                print(self.Qmax, 1/self.decay/3.15e13, self.Q)

#            if i%5000==0:
#                print(self.crust.dr_dt)
#                print(self.crust.h)
#                print("i", i, ",r", self.crust.r/1e3, ",t", self.t/3.15e13, "My, dt", self.dt)
#                print((time.time() - debut)/60, "min")
#                #y = (self.r_crust - self.crust.r)/(self.r_body - self.crust.r) + 1
#                ax1c.plot(self.crust.T_[:5], self.crust.y[:5], '.')
#                ax2c.plot(self.crust.T_, self.crust.y, '.')
#                ax1c.set_xlabel('temperature')
#                ax1c.set_ylabel('radius')
#                ax2c.set_xlabel('h(r)')
#                figc1.suptitle('Crust')
#                plt.savefig("T_crust.pdf")
#                # = (self.crust.dT * self.crust.T_ + self.T_EQ)
#                axc.plot(self.crust.T_, rc_crust)
##                ax2s.plot(self.solid.hr, self.r_solid)
#                axc.set_xlabel('temperature')
#                axc.set_ylabel('radius')
#                #ax2s.set_xlabel('h(r)')
#                figc.suptitle('crust')
#
#                y = (self.r_solid - self.r_core)/(self.solid.r - self.r_core) + 1
#                ax1s.plot(self.solid.T_[:5], y[:5], '.')
#                ax2s.plot(self.solid.T_[-5:], y[-5:], '.')
#                ax1s.set_xlabel('temperature')
#                ax1s.set_ylabel('radius')
##                ax2s.set_xlabel('h(r)')
#                figs1.suptitle('solid')
#                plt.savefig("T_solid.pdf")
#                Tsol = (self.solid.dT * self.solid.T_ + self.T_E)
#                axs.plot(self.solid.T_, rc_solid)
#                axs.set_ylim(1.2e6, 1.7e6)
##                ax2s.plot(self.solid.hr, self.r_solid)
#                axs.set_xlabel('temp')
#                axs.set_ylabel('radius')
#                #ax2s.set_xlabel('h(r)')
#                figs.suptitle('solid')
#
#                axhc.plot(self.crust.hr, rc_crust, '.')
#                axhs.plot(self.solid.hr, rc_solid, '.')
#                axhs.set_ylim(1.4e6, 1.7e6)
#                figh.suptitle('h(r)')
            
            
            #save data
            if i%100==0:
                
                self.T_core_history.append(self.core.T)
                self.r_solid_history.append(self.solid.r)
                self.r_crust_history.append(self.crust.r)
                self.t_history.append(self.t)
                self.h_lmo_history.append(self.h_lmo)
                self.h_crust_history.append(self.crust.h)
                self.h_solid_history.append(self.solid.h)
                self.Ts_history.append(self.crust.Ts)
                self.flux_crust.append(self.crust.k * self.crust.dT * 2 * (1 - self.crust.T_[-1]) /(self.crust.dr))
                self.flux_solid.append(self.solid.k * 2 * self.solid.dT * self.solid.T_[-1] \
                    /(self.solid.dr))
                self.flux_lat.append(self.rho * self.LATENT_HEAT * 4 * np.pi * (- self.crust.r**2 * self.crust.dr_dt + self.solid.r**2 * self.solid.dr_dt))
                self.flux_hlmo.append(self.h_lmo * np.exp(-self.HEAT_DECAY * self.t/3.1536e7) * 4 * np.pi * (self.crust.r**3 - self.solid.r**3)/3)
                self.flux_overturn.append(self.Qmax * np.exp(-self.decay * self.t))
                self.dv_crust_history.append(self.crust.dV)
                self.dV_solid_history.append(self.solid.dV)
                self.r_profil_crust.append(rc_crust)
                self.T_profil_crust.append(self.crust.T_)
                self.r_profil_solid.append(rc_solid)
                self.T_profil_solid.append(self.solid.T_)
                self.hr_profil_crust.append(self.crust.hr)
                self.hr_profil_solid.append(self.solid.hr)
                self.Tbot_crust.append(self.crust.T_[-1])
                self.Ttop_solid.append(self.solid.T_[-1])
            if i%10000==0:
                print((time.time() - debut)/60, "min")
                print(f"thickness: {(self.r_body - self.crust.r)/1e3}")
            
            i += 1
            
#        ax1c.plot(0, 2, '*')
#        ax2c.plot(1, 1, '*')
#        ax1s.plot(1, 1, '*')
#        ax2s.plot(0, 2, '*')
#        plt.show()
        h = self.h_lmo * np.exp(-self.HEAT_DECAY * self.t/3.1536e7) * 4 * np.pi * (self.crust.r**3 - self.solid.r**3)/3
        total = self.rho * self.LATENT_HEAT * 4 * np.pi * (- self.crust.r**2 * self.crust.dr_dt + self.solid.r**2 * self.solid.dr_dt) + h + self.solid.k * 2 * self.solid.dT * self.solid.T_[-1] / (self.solid.dr) * 4 * np.pi * self.solid.r**2
        self.t  = self.t + (self.rho*self.LATENT_HEAT*4/3  * np.pi * (self.crust.r**3 - self.solid.r**3) + h)/total
        r_min_crust = pow(self.r_body**3 - self.c0 * (self.r_body**3 - self.r_core**3), 1/3)
        self.r_solid_history.append(r_min_crust)
        self.r_crust_history.append(r_min_crust)
        self.t_history.append(self.t)
        print((time.time() - debut)/60, "min")
        print(f"final thickness: {(self.r_body - self.crust.r)/1000} km")
        print(f"final time: {self.t/3.15e13} Myr" )
        
    def get_time_history(self):
        return self.t_history
        
    def get_radius(self):
        return self.r_crust_history, self.r_solid_history
        
    def get_T_history(self):
        return self.Ts_history, self.T_core_history
        
    def get_h_history(self):
        return self.h_lmo_history, self.h_crust_history, self.h_solid_history
        
    def get_drdt_history(self):
        return self.drdt_crust_history, self.drdt_solid_history
        
            
