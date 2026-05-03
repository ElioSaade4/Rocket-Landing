import numpy as np
from dataclasses import dataclass


@dataclass
class RocketParams:
    m_dry   = 49000.0               # dry mass (kg)
    S_ref   = 8.0                   # reference area (m^2)
    C_D     = 0.25                  # drag coefficient
    Isp     = 443.0                 # specific impulse (s)
    T_min   = 412700.0              # minimum thrust (N)
    T_max   = 1375600.0             # maximum thrust (N)
    alpha_min = np.deg2rad( -10 )   # minimum angle of attack (rad)
    alpha_max = np.deg2rad( 10 )    # maximum angle of attack (rad)
    V_safe = 1.0                    # max landing speed (m/s)
    alpha_safe = np.deg2rad( 2.0 )  # max landing angle of attack (rad)

    # Derived       
    z_dry   = np.log( m_dry )