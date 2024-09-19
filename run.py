#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: linecolin

 _   _  ___   __  __  __  __  _____ 
| \ | | ____||  \/  ||  \/  ||  _  |
|  \| |  _|  | |\/| || |\/| || | | |
| |\  | |___ | |  | || |  | || |_| |
|_| \_|_____||_|  |_||_|  |_||_____| 
  
       o       /`-._
      o o     /_,.._`:-     
        o ,.-'  ,   `-:..-')   
         : o ):';      _  {   
          `-._ `'__,.-'\`-.)
              `\\  \,.-'`

"""

from pathlib import Path
from evolution import Stage2Analysis
import numpy as np
import matplotlib.pyplot as plt


# Defining the problem
# Here, the example for the reference case, 
# i.e. crust thickness 44 km, global magma ocean, no overturn.

Moon = Stage2Analysis(r_body = 1737e3, # body radius
                    r_core = 390e3, # core radius
                    albedo = 0.12, # albedo
                    rho = 3.3e3, # density
                    gravity = 1.62, # gravity
                    initial_heat_production = 0, # initial heat production
                    c0 = 0.075, # initial composition in anorthite component 
                    ce = 0.37, # eutectic composition
                    k_crust = 2, #thermal conductivity of the crust
                    D = 1e-3, # partition coefficient of the heat producing elements
                    heat_source = False, # in the case of a non-global magma ocean this value have to be True
                    r_flottability = None, # radius of the depth of the non global magma ocean
                    distance_sun_object = 1495e5, # distance sun body
                    n_factor = 1, # factor to increase the precision for the resolution of advection in the cumulates
                    overturn = None, # in the case of an overturn these value have to be True
                    t_overturn = 100 # factor to modify the decay of the overturn
                    )

Moon.run_stage2_analysis()
time = [i/3.15e13 for i in Moon.get_time_history()] # convert in Myr
r_crust, r_cumul = Moon.get_radius_history()
save_name = Moon.get_name()

print(type(Moon.get_crust_profil()[0]))
fig, ax = plt.subplots()
ax.plot(time, r_crust, 'k-') #in km
ax.plot(time, r_cumul, 'k-') #in km
ax.set_xlabel("time [myr]")
ax.set_ylabel("radius [km]")
fig.savefig(save_name + ".pdf", transparent=True)

