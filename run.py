#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 09:03:15 2024

@author: linecolin
"""

from pathlib import Path
from physicalbody import PhysicalBody
from evolution import Stage2Analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#%%
# Define the body to study

# Moon = PhysicalBody(r_body = 1737e3,
#                     r_core = 390e3,
#                     albedo = 0.12,
#                     rho = 3.3e3,
#                     gravity = 1.62,
#                     initial_heat_production = 25e-12,
#                     c0 = 0.076,
#                     ce = 0.37,#0.075/0.2,
#                     k_crust =2.7,
#                     D = 1e-3,
#                     heat_source = False,
#                     r_flottability = None,
#                     distance_sun_object = 1495e5,
#                     n_factor=1)



Moon = Stage2Analysis(r_body = 1737e3,
                    r_core = 390e3,
                    albedo = 0.12,
                    rho = 3.3e3,
                    gravity = 1.62,
                    initial_heat_production = 25e-12,
                    c0 = 0.075,
                    ce = 0.37,#0.075/0.2,
                    k_crust = 2,
                    D = 1e-3,
                    heat_source = False,
                    r_flottability = None,
                    distance_sun_object = 1495e5,
                    n_factor=1,
                    overturn=None,
                    t_overturn=0.1)

th_crust = (Moon.r_body - pow(Moon.r_body**3 - Moon.c0 * (Moon.r_body**3 - Moon.r_core**3), 1/3))/1e3
print(th_crust)
print(Moon.k_crust)
print(Moon.overturn)


th_crust = format(th_crust, '.0f')
kc_print = format(Moon.k_crust, '.1f')
kc = kc_print.replace('.', '-')
ce = format(Moon.ce, '.2f')
ce = ce.replace('.', '-')
hp = format(Moon.initial_heat_production, '.1e')
if Moon.heat_source == True:
    depth_LMO = (Moon.r_body - Moon.r_flottability)/1e3
    depth_LMO = format(depth_LMO, '.0f')
    save_name = f"{th_crust}km_kc{kc}_CE{ce}_overturn_{Moon.overturn}_depth_LMO_{depth_LMO}_ihp_{hp}"
else:
    save_name = f"{th_crust}km_kc{kc}_CE{ce}_overturn_{Moon.overturn}_ihp_{hp}"
print(save_name)


Moon.run_stage2_analysis()
print("end analysis, start save data")


#%%


time_evolution = [i/3.15e13 for i in Moon.get_time_history()]
r_crust, r_cumul = Moon.get_radius()

drdt_crust, drdt_cumul = Moon.get_drdt_history()

radius_crust = np.array([(Moon.r_body - i)/1e3  for i in r_crust])
radius_cumul = np.array([(Moon.r_body - i)/1e3  for i in r_cumul])

T_surf_evolution, T_core_evolution = Moon.get_T_history()
Tbot_crust = Moon.Tbot_crust
Ttop_solid = Moon.Ttop_solid

h_lmo, h_crust, h_cumul = Moon.get_h_history()

volume_lmo = 4*np.pi*(np.array(r_crust)**3 - np.array(r_cumul)**3)/3
#%%
print(1/Moon.decay/3.15e13)

# if 0 < 1/Moon.decay/3.15e13 <=10:
#     dir_save = "Datas/Decay_a_1Myr"
# elif 10 < 1/Moon.decay/3.15e13 <=100:
#     dir_save = "Datas/Decay_a_10Myr"
# else:
#     dir_save = "Datas/Decay_a_100Myr"
dir_save = "Datas/ExplorationTimeOverturn"
dir_save = Path(dir_save)
print(dir_save)

data = {"time": time_evolution, 
        "radius crust":r_crust, 
        "radius cumul": r_cumul}
data_reference = pd.DataFrame(data)
data_reference.to_csv(dir_save/f"{save_name}.csv")

data_temp = {"surface temp":T_surf_evolution,
            "core temp":T_core_evolution, 
            "crust":Tbot_crust, 
            "cumul":Ttop_solid}
data_temp = pd.DataFrame(data_temp)
data_temp.to_csv(dir_save/f"{save_name}_temp.csv")

data_drdt = {"crust":drdt_crust, 
              "cumul":drdt_cumul}
data_ref = pd.DataFrame(data_drdt)
data_ref.to_csv(dir_save/f"{save_name}_drdt.csv")

data_heat = {"h lmo":h_lmo, 
              "h crust":h_crust, 
              "h cumul":h_cumul}
data_heat = pd.DataFrame(data_heat)
data_heat.to_csv(dir_save/f"{save_name}_heat.csv")

data_flux = {"L":Moon.flux_lat,
              "flux crust":Moon.flux_crust, 
              "flux solid": Moon.flux_solid,
              "q LMO":Moon.flux_hlmo[1:],
              "overturn":Moon.flux_overturn}
             
data_flux = pd.DataFrame(data_flux)
data_flux.to_csv(dir_save/f"{save_name}_flux.csv")




# print("start save crust profils")

# data_crust = {"radius":Moon.r_profil_crust, 
#               "Temp":Moon.T_profil_crust, 
#               "h(r)":Moon.hr_profil_crust}
# data_crust = pd.DataFrame(data_crust)
# data_crust.to_csv(dir_save/f"{save_name}_crust_profil.csv")

# print("start save cumulates profils")

# data_solid = {"radius":Moon.r_profil_solid, 
#               "Temp":Moon.T_profil_solid, 
#               "h(r)":Moon.hr_profil_solid}
# data_solid = pd.DataFrame(data_solid)
# data_solid.to_csv(dir_save/f"{save_name}_solid_profil.csv")

# print("end")

#%%

time = [t/3.15e13 for t in Moon.t_history]
rad_crust, rad_solid = Moon.get_radius()
# save data reference case

temp_surf = Moon.Ts_history
temp_core = Moon.T_core_history
h_crust = Moon.h_crust_history
h_solid = Moon.h_solid_history
h_lmo = Moon.h_lmo_history
dV_crust = Moon.dv_crust_history
dV_solid = Moon.dV_solid_history
q_crust = Moon.flux_crust
q_solid = Moon.flux_solid
q_lmo = Moon.flux_hlmo[1:]
q_latent_heat = Moon.flux_lat
q_overturn = Moon.flux_overturn
rc_crust = Moon.r_profil_crust
temp_crust = Moon.T_profil_crust
hr_crust = Moon.hr_profil_crust
rc_solid = Moon.r_profil_solid
temp_solid = Moon.T_profil_solid
hr_solid = Moon.hr_profil_solid
 
#%%

#radial evolution

plt.figure()
plt.plot(time, rad_crust)
plt.plot(time, rad_solid)
plt.xlabel("time [My]")
plt.ylabel("radius [km]")
plt.savefig(save_name + "radius_ref_case.pdf")

#%%

plt.figure()
plt.plot(time[:-1], temp_surf, '.')
plt.xlabel("time [My]")
plt.ylabel("surface temp [K]")
plt.xlim(-0.2, 0.5)
plt.savefig(save_name + "surf_temp_ref_case.pdf")
#%%

plt.figure()
plt.plot(time[:-1], temp_core)
plt.xlabel("time [My]")
plt.ylabel("core temp [K]")
plt.savefig(save_name + "core_temp_ref_case.pdf")


#%%

# fluxes
n = len(time)

Q_solid = [q_solid[i] * 4 * np.pi * rad_solid[i+1]**2 for i in range(n-2)]
Q_crust = [q_crust[i] * 4 * np.pi * rad_crust[i+1]**2 for i in range(n-2)]

total = [q_lmo[i] + q_latent_heat[i] + q_overturn[i] for i in range(n-2)]

plt.figure()
plt.plot(time[1:-1], Q_crust, label="crust")
plt.plot(time[1:-1], total, '--', label="Sum")
plt.plot(time[1:-1], q_lmo, label = "LMO")
plt.plot(time[1:-1], q_latent_heat, label = "latent heat")
#plt.plot(time[1:-1], Q_solid, label="solid")
plt.plot(time[1:-1], q_overturn, label="overturn")
plt.yscale("log")
plt.xlabel("time [My]")
plt.legend()
plt.savefig(save_name + "fluxes.pdf")
#%%

# def vecteur_float(vecteur):
#     vecteur_str = vecteur.replace('[', '').replace(']', '').replace('\n', '')
#     valeurs_str = vecteur_str.split()
#     valeurs_float = [float(val) for val in valeurs_str]
#     return valeurs_float

# figcT, axcT = plt.subplots(1, 1, sharey=True)
# figch, axch = plt.subplots(1, 1, sharey=True)
# figsT, axsT = plt.subplots(1, 1, sharey=True)
# figsh, axsh = plt.subplots(1, 1)

# for i in range(1, n, 10000):
#     axch.plot(hr_crust[i], rc_crust[i], '.')
#     axcT.plot(temp_crust[i], rc_crust[i])
    
#     axsh.plot(hr_solid[i], rc_solid[i],)
#     axsT.plot(temp_solid[i], rc_solid[i])

# axcT.set_xlabel("temperature")
# axcT.set_ylabel("radius [m]")

# axch.set_xlabel("h")
# axch.set_ylabel("radius [m]")

# axsT.set_xlabel("temperature")
# axsT.set_ylabel("radius [m]")  

# axsh.set_xlabel("h")
# axsh.set_ylabel("radius [m]")
# axsh.set_ylim(1.6e6, 1.7e6)

# figcT.savefig("crust_temp.pdf")
# figsT.savefig("solid_temp.pdf")
# figch.savefig("crust_h.pdf")
# figsh.savefig("solid_h.pdf")

#%%

# production de chaleur 

# heat production
# t = np.array(time) * 3.15e13

# total_heat = 25e-12 * Moon.rho * 4 * np.pi * (Moon.r_body**3 - Moon.r_core**3) * np.exp( - Moon.HEAT_DECAY * t/3.1536e7) /3
# int_hr_crust = []
# int_hr_solid = []
# somme = []
# erreur = []
#m = len(vecteur_float(rc_crust.iloc[0]))
# for i in range(n-2):
#     # rc_rust = np.array(vecteur_float(rc_crust.iloc[i]))
#     # hcrust = np.array(vecteur_float(hr_crust.iloc[i]))
    
#     # rsolid = np.array(vecteur_float(rc_solid.iloc[i]))
#     # hsolid = np.array(vecteur_float(hr_solid.iloc[i]))
    
#     a = np.trapz(4 * np.pi * rc_crust[i+1]**2 * hr_crust[i], rc_crust[i+1])
#     b = np.trapz(4 * np.pi * rc_solid[i+1]**2 * hr_solid[i], rc_solid[i+1])
#     int_hr_crust.append(a)
#     int_hr_solid.append(b)

#     somme.append(np.abs(a) + b + q_lmo[i])
#     erreur.append((np.abs(a) + b + q_lmo[i] - total_heat[i])/total_heat[i])
# #%%    
# plt.plot(time, total_heat)
# plt.plot(time[1:-1], np.abs(int_hr_crust))
# plt.plot(time[1:-1], int_hr_solid)
# plt.plot(time[1:-1], somme, '--')
# # plt.plot(time, h_solid, '--')
# # plt.plot(time, h_crust, '--')
# plt.plot(time[1:], q_lmo)
# plt.yscale("log")
# plt.ylim(1.6e12, 1.85e12)

# #%%

# # vérification avce profils de temperature

# qbis_solid = []
# qbis_crust = []
# qbis_lmo = []
# dT_solid = Moon.solid.dT
# dT_crust = Moon.crust.dT

# for i in range(1, n, 1000):
#     dr_solid = (rc_solid[i][0] - rc_solid[i][-1]) / 100
#     qbis_solid.append((4 * 2 * temp_solid[i][-1] * dT_solid / dr_solid))# * 4 * np.pi * rad_solid[i+1]**2)
    
#     dr_crust = (rc_crust[i][0] - rc_crust[i][-1]) /100
#     qbis_crust.append(2 * 2 * dT_crust * (1 - temp_crust[i][-1]) /dr_crust)
    
    
# plt.plot(time[1::1000], np.abs(qbis_solid))
# plt.plot(time[1:-1], q_solid, '--')

# plt.plot(time[1::1000], qbis_crust)
# plt.plot(time[1:-1], q_crust, '--')
# plt.yscale('log')