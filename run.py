#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: linecolin : linecolin-at-proton.me
  
       o       /`-._
      o o     /_,.._`:-     
        o ,.-'  ,   `-:..-')   
         : o ):';      _  {   
          `-._ `'__,.-'\`-.)
              `\\  \,.-'`

Refer to the user guide for a detailed description of the input parameters and output functions.

"""

from evolution import Stage2Analysis
import numpy as np
import matplotlib.pyplot as plt


# Defining the body to study
# Here, the example for the thermal evolution of the lunar magma ocean 
# for the reference case, i.e. crust thickness 44 km, global magma ocean, no overturn.

Moon = Stage2Analysis(r_body = 1737e3, 
                    r_core = 390e3, 
                    albedo = 0.12, 
                    rho = 3.3e3, 
                    gravity = 1.62, 
                    initial_heat_production = 0, 
                    c0 = 0.075, 
                    ce = 0.37, 
                    k_crust = 2, 
                    D = 1e-3, 
                    heat_source = False, 
                    r_flottability = None, 
                    distance_sun_object = 1495e5, 
                    n_factor = 1, 
                    overturn = None, 
                    t_overturn = 100 
                    )

Moon.run_stage2_analysis()
time = Moon.get_time_history()/3.15e13
r_crust, r_cumul = Moon.get_radius_history()
save_name = Moon.get_name()

print(type(Moon.get_crust_profil()[0]))
fig, ax = plt.subplots()
ax.plot(time, r_crust, 'k-') 
ax.plot(time, r_cumul, 'k-') 
ax.set_xlabel("time [myr]")
ax.set_ylabel("radius [km]")
fig.savefig(save_name + ".pdf", transparent=True)

