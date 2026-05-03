"""
Convex Trajectory Optimization and Model Predictive Control for Fuel-Optimal Rocket Landing
References: 
[1] Wang & Song (2018), "Convex Model Predictive Control for Rocket Vertical Landing"
[2] Liu, "Fuel-Optimal Rocket Landing with Aerodynamic Controls"
"""

import numpy as np
import matplotlib.pyplot as plt

from RocketParams import RocketParams
from ReferenceTrajectory import ReferenceTrajectory
from TrackingMPC import TrackingMPC


if __name__ == "__main__":

    # Intiial conditions
    x0 = np.array([
        3000.0,             # h0  : initial altitude (m)
        0.0,                # s0  : initial horizontal pos (m)
        280.0,              # V0  : initial speed (m/s)
        np.deg2rad( -65 ),  # gamma0 : initial flight path angle (rad)
        np.log( 55000.0 ),  # z0  : ln(m0)
        0.0                 # t0  : initial time (s)
    ])

    # Target position for landing
    target_pos = np.array([
        0.0,              # hf  : final altitude (landing)
        1000.0,           # sf  : final horizontal pos (m)
    ])
    
    rocket = RocketParams()
    ref_traj = ReferenceTrajectory( rocket )
    tracking_mpc = TrackingMPC( rocket, ref_traj )

    print("=" * 60)
    print("STEP 1: Fuel-Optimal Trajectory Optimization")
    print("=" * 60)

    ref_traj.solve( 
        x0=x0,
        target_pos=target_pos,
        N=30, 
        max_iter=20, 
        tol=1e-2, 
        tf_guess=20.0
    )

    ref_traj.plot()

    # MPC
    print("\n" + "="*60)
    print("STEP 2: MPC Guidance")
    print("="*60)

    tracking_mpc.solve( 
        N=20,
        dt=0.1,
        max_iter=20,
        tol=1e-2
    )

    tracking_mpc.plot()

    plt.show( block=True )