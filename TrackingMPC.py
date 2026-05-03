import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from utils import air_density, build_trapezoidal_constraints


g0  = 9.80665               # Standard gravity (m/s^2)
R0  = 6.3781e6              # Earth radius (m)


class TrackingMPC:
    def __init__( self, rocket, ref_traj ):
        self.rocket = rocket
        self.ref_traj = ref_traj

        self.t_ref = None 
        self.X_ref = None 
        self.U_ref = None 
        
        self.X_mpc = np.empty( ( 5, 0 ) )   # states for MPC (exclude 't' state)
        self.U_mpc = np.empty( ( 3, 0 ) )   # controls for MPC (exclude 'tf' control)
        
        # state tracking weights
        self.Q = np.diag( [ 
            1 / 1**2,       # h
            1 / 10**2,      # s
            1 / 20**2,      # V
            1 / 0.1**2,     # gamma
            1 / 0.05**2     # z
        ] )  

        # control tracking weights 
        self.R = np.diag([
            1/10**2,    # u1
            1/5**2,     # u2 
            1/10**2,    # u3 
        ])

        # state terminal weights
        self.Qf = self.Q.copy() * 100


    def interpolate_reference( self, dt ): 
        """
        Interpolate reference trajectory to uniform timestep.
        
        X: ( 6, N ) array - states [ r, s, V, gamma, z, t ]
        U: ( 4, N ) array - controls [ u1, u2, u3, tf ]
        dt: timestep for interpolation ( default 0.1s )
        
        Returns t_uniform, X_interp, U_interp at uniform dt timestep.
        """
        X = self.ref_traj.X_opt
        U = self.ref_traj.U_opt

        # The actual time values are stored in X[ 5, : ] ( the 't' state )
        t_nodes = X[ 5, : ]
        t_end_rounded = np.round(t_nodes[-1] / dt ) * dt
        X[5, -1] = t_end_rounded

        t_nodes = X[ 5, : ]
        t_start = t_nodes[ 0 ]
        t_end   = t_nodes[ -1 ]
        t_uniform = np.arange( t_start, t_end + dt, dt )
        t_uniform[ -1 ] = t_end  # ensure last point is exactly t_end (because of float precision)
        
        X_interp = np.zeros( ( X.shape[ 0 ]-1, len( t_uniform ) ) )

        for i in range( X.shape[ 0 ]-1):
            f = interp1d( t_nodes, X[ i, : ], kind='linear' )
            X_interp[ i, : ] = f( t_uniform )
        
        # Interpolate u1, u2, u3; tf is constant
        U_interp = np.zeros( ( U.shape[ 0 ]-1, len( t_uniform ) ) )

        for i in range( U.shape[ 0 ]-1 ):  # u1, u2, u3
            f = interp1d( t_nodes, U[ i, : ], kind='linear' )
            U_interp[ i, : ] = f( t_uniform )
        
        self.t_ref = t_uniform
        self.X_ref = X_interp
        self.U_ref = U_interp

    
    def drag_accel( self, h, V, z ):
        """
        D/m term in the equations of motion.
        D = 0.5 * rho * V^2 * Sref * CD
        m = exp(z)
        Returns D/m.
        """
        rho = air_density( h )
        D   = 0.5 * rho * ( V **2 ) * self.rocket.S_ref * self.rocket.C_D
        m   = np.exp( z )
        return D / m
    

    def dynamics( self, x, u ):
        """
        Nonlinear continuous dynamics dx/dtau = f(x, u).
        tau in [0,1] is the normalized independent variable (Eq. 8).
        u = [u1, u2, u3, tf]
        """
        h, s, V, gamma, z = x
        u1, u2, u3 = u

        r = h + R0
        Dm  = self.drag_accel( h, V, z )          # D/m

        h_dot     = V * np.sin( gamma )
        s_dot     = V * np.cos( gamma )
        V_dot     = -u1 - Dm - ( g0 * np.sin( gamma ) ) 
        gamma_dot = -( u2 / V ) - ( g0 * np.cos( gamma ) / V ) 
        z_dot     = u3 / ( g0 * self.rocket.Isp )

        return np.array( [ h_dot, s_dot, V_dot, gamma_dot, z_dot ] )
    
    
    def compute_ABC( self, xk, uk ):
        """
        Compute linearization matrices A, B, and affine offset c at (xk, uk).

        Linearized dynamics:
            dx/dtau = A @ x + B @ u + c

        where c = f(xk,uk) - A @ xk - B @ uk  (Eq. 20)

        Returns A (5x5), B (5x3), c (5,)
        """
        h, s, V, gamma, z = xk
        u1, u2, u3 = uk

        r = h + R0

        rho = air_density( h )
        m   = np.exp( z )
        D   = 0.5 * rho * V**2 * self.rocket.S_ref * self.rocket.C_D
        Dm  = D / m                            # D/m

        # ── Matrix A = df/dx ──────────────────────────────────────────────────────
        A = np.zeros( ( 5, 5 ) )

        # Row 0 : h_dot = tf * V * sin(gamma)
        A[0, 2] = np.sin( gamma )           # d/dV
        A[0, 3] = V * np.cos( gamma )       # d/d_gamma

        # Row 1 : s_dot = tf * V * cos(gamma)
        A[1, 2] = np.cos( gamma )           # d/dV
        A[1, 3] = - V * np.sin( gamma )      # d/d_gamma

        # Row 2 : V_dot = tf*(-u1 - D/m - g0 sin(gamma))
        A[2, 2] = - 2 * Dm / V                 # d/dV
        A[2, 3] = - g0 * np.cos( gamma )      # d/d_gamma
        A[2, 4] = Dm                           # d/z 

        # Row 3 : gamma_dot = tf*(-u2/V - g0 cos(gamma))
        A[3, 2] = ( u2 +  g0 * np.cos( gamma ) ) / ( V**2 )                # d/dV
        A[3, 3] = g0 * np.sin( gamma ) / V                                  # d/d_gamma

        # Row 4 (z_dot) has no state dependence
        # => remain zero

        # ── Matrix B = df/du ──────────────────────────────────────────────────────
        B = np.zeros( ( 5, 3 ) )

        # Row 2 : V_dot = tf*(-u1 - D/m - sin(gamma)/r^2)
        B[2, 0] = -1                        # d/d_u1

        # Row 3 : gamma_dot = tf*(-u2/V - cos(gamma)/(r^2*V))
        B[3, 1] = -1 / V                      # d/d_u2

        # Row 4 : z_dot = -tf*u3/Isp
        B[4, 2] = -1 / ( g0 * self.rocket.Isp )                  # d/d_u3

        # ── Affine offset c = f(xk,uk) - A@xk - B@uk ─────────────────────────────
        fk = self.dynamics( xk, uk )
        C  = fk - A @ xk - B @ uk

        return A, B, C
    

    def step_env(self, x, u, dt ):
        h, s, V, gamma, z = x
        u1, u2, u3 = u

        r = h + R0
        Dm  = self.drag_accel( h, V, z )          # D/m

        h_next     = h + ( V * np.sin( gamma ) ) * dt
        s_next     = s + ( V * np.cos( gamma ) ) * dt
        V_next     = V + ( -u1 - Dm - ( g0 * np.sin( gamma ) ) ) * dt
        gamma_next = gamma + ( -( u2 / V ) - ( g0 * np.cos( gamma ) / V ) ) * dt
        z_next     = z + ( - u3 / ( g0 * self.rocket.Isp ) ) * dt

        return np.array( [ h_next, s_next, V_next, gamma_next, z_next ] )
                        
    
    def solve( self, N, dt, max_iter, tol ):

        self.interpolate_reference( dt )

        # Vector of dynamics
        x0 = self.X_ref[ :, 0 ].copy()
        self.X_mpc = np.column_stack( [ self.X_mpc, x0 ] )

        for i in range( len( self.t_ref ) - 1 ):

            if( i + N + 1 > len( self.t_ref ) ):
                # Number of points left smaller than MPC horizon, use whatever is left
                N_mpc = len( self.t_ref ) - i - 1
            else:
                N_mpc = N

            print( f"MPC step {i}/{len( self.t_ref )-1} | horizon: {i} to {i+N_mpc+1}" )

            # Horizon reference trajectory
            X_ref_hor = self.X_ref[ :, i : i+N_mpc+1 ]
            U_ref_hor = self.U_ref[ :, i : i+N_mpc+1 ]

            X_prev = X_ref_hor.copy()
            U_prev = U_ref_hor.copy()

            for iteration in range( max_iter ):
                # Compute linearization matrices at each node
                Ak_list = []
                Bk_list = []
                Ck_list = []

                for k in range( N_mpc + 1 ):
                    A, B, C = self.compute_ABC( X_prev[:, k], U_prev[:, k] )
                    Ak_list.append( A )
                    Bk_list.append( B )
                    Ck_list.append( C )

                # ── CVXPY decision variables ──────────────────────────────────────────
                X = cp.Variable( ( 5, N_mpc + 1 ) )   # states
                U = cp.Variable( ( 3, N_mpc + 1 ) )   # controls

                # ── Warm Start ──────────────────────────────────────────
                X.value = X_prev
                U.value = U_prev

                constraints = []

                # ── Dynamics constraints (trapezoidal, Eq. 26) ───────────────────────
                constraints += build_trapezoidal_constraints(
                    X, U, Ak_list, Bk_list, Ck_list, dt
                )

                # ── Initial boundary conditions (Eq. 3) ──────────────────────────────
                constraints.append( X[:, 0] == x0 )

                # Try these terminal constraints, if not, add a terminal cost to the objective
                # Terminal constraint to keep it in line with the reference
                # If I don't the rocket might spend more fuel and might not have enough to land, because MPC is looking at a short horizon
                # constraints.append( X[0, N_mpc] == X_ref_mpc[0, -1] ) 
                # constraints.append( X[1, N_mpc] == X_ref_mpc[1, -1] )     
                # constraints.append( X[2, N_mpc] == X_ref_mpc[2, -1] )     
                # constraints.append( X[3, N_mpc] == X_ref_mpc[3, -1] )    
                # constraints.append( X[4, N_mpc] >= X_ref_mpc[4, -1] )       # Can relax mass to be bigger or equal 


                # Angle of attack terminal constraint (Eq. 13, linearized form):
                # |u2/u1| <= tan(alpha_safe)  at final node
                constraints.append(
                    cp.abs( U[1, N_mpc] ) <= np.tan( self.rocket.alpha_safe ) * U[0, N_mpc]
                )

                # ── Process constraints (Eq. 14) ─────────────────────────────────────
                for i in range( N_mpc + 1 ):

                    ez_k = np.exp( -X_prev[4, i] )        # e^{-z^k} (fixed, from linearization)

                    # T_min * e^{-z} <= u3 <= T_max * e^{-z}   (from Eq. 14, after e^z linearization)
                    constraints.append( self.rocket.T_min * ez_k <= U[2, i] )
                    constraints.append( U[2, i] <= self.rocket.T_max * ez_k )

                    # Angle of attack: |u2/u1| <= tan(alpha_max)  =>  |u2| <= tan(alpha_max)*u1
                    constraints.append(
                        cp.abs( U[1, i] ) <= np.tan( self.rocket.alpha_max ) * U[0, i]
                    )

                    # SOCP / lossless convexification (Eq. 22):
                    # u1^2 + u2^2 <= u3^2
                    constraints.append(
                        cp.norm( U[:2, i], 2 ) <= U[2, i]
                    )

                # ── Objective: minimize tracking error from reference ─────────
                objective = cp.Minimize(
                    sum(
                        cp.quad_form( X[:, k] - X_ref_hor[:, k], self.Q ) +
                        cp.quad_form( U[:, k] - U_ref_hor[:, k], self.R )
                        for k in range( N_mpc + 1 )
                    ) 
                    + cp.quad_form( X[:, -1] - X_ref_hor[:, -1], self.Qf )
                )

                # ── Solve ─────────────────────────────────────────────────────────────
                prob = cp.Problem( objective, constraints )
                prob.solve( solver=cp.SCS, verbose=False, warm_start=True )

                if prob.status not in [ "optimal", "optimal_inaccurate" ]:
                    print( f"  Iter {iteration+1}: solver status = {prob.status}. Stopping." )
                    break

                X_sol = X.value
                U_sol = U.value

                # ── Convergence check (Eq. 24) ────────────────────────────────────────
                dx = np.max( np.abs( X_sol - X_prev ) )
                du = np.max( np.abs( U_sol - U_prev ) )

                # print(f"  Iter {iteration+1}:"
                #     f"fuel = {(np.exp(x0[4]) - np.exp(X_sol[4,-1])):.1f} kg | "
                #     f"dx={dx:.2e}, du={du:.2e}")

                X_prev = X_sol.copy()
                U_prev = U_sol.copy()

                if dx < tol and du < tol:
                    print(f"Converged at iteration {iteration+1}.")
                    break

            self.U_mpc = np.column_stack( [ self.U_mpc, U_sol[:, 0] ] )  # first control input of the optimal sequence

            # Update dynamics by applying the first control input and simulating the nonlinear dynamics for one step
            x0 = self.step_env( x0, U_sol[:,0], dt )
            self.X_mpc = np.column_stack( [ self.X_mpc, x0 ] )

            print( f"h={x0[0]:.1f} m, s={x0[1]:.1f} m, V={x0[2]:.1f} m/s, gamma={np.rad2deg(x0[3]):.1f} deg, mass={np.exp(x0[4]):.1f} kg" )
            print( f"u1={U_sol[0,0]:.2f}, u2={U_sol[1,0]:.2f}, u3={U_sol[2,0]:.2f}" )
            print()

        print( f"MPC landed at h={self.X_mpc[0, -1 ]:.2f}m, s={self.X_mpc[ 1, -1 ]:.1f}m, V={self.X_mpc[ 2, -1 ]:.1f}m/s, gamma={np.rad2deg(self.X_mpc[ 3, -1 ]):.1f}deg" )


    def plot( self ):
        """Plot states and controls of the fuel-optimal reference trajectory"""

        fig1, axes1 = plt.subplots( 4, 1, figsize=(3.5, 7), sharex=False, constrained_layout=True )
        axes1.flatten()

        # States
        # Trajectory: altitude vs horizontal position
        axes1[0].plot( self.X_ref[1], self.X_ref[0], label="Ref" )
        axes1[0].plot( self.X_mpc[1], self.X_mpc[0], 'r--', label="MPC" )
        axes1[0].set_xlabel( "Downrange s (m)" )
        axes1[0].set_ylabel( "Altitude h (m)" )
        axes1[0].set_title( "Trajectory" )
        axes1[0].grid(True)
        axes1[0].legend()

        # Speed
        axes1[1].plot( self.t_ref, self.X_ref[2], label="Ref" )
        axes1[1].plot( self.t_ref, self.X_mpc[2], 'r--', label="MPC" )
        axes1[1].set_xlabel( "time (s)" )
        axes1[1].set_ylabel( "V (m/s)" )
        axes1[1].set_title( "Velocity" )
        axes1[1].grid( True )
        axes1[1].legend()
        
        # Flight path angle
        axes1[2].plot( self.t_ref, np.rad2deg( self.X_ref[ 3 ] ), label="Ref" )
        axes1[2].plot( self.t_ref, np.rad2deg( self.X_mpc[ 3 ] ), 'r--', label="MPC" )
        axes1[2].set_xlabel( "time (s)" )
        axes1[2].set_ylabel( "gamma (deg)" )
        axes1[2].set_title( "Flight Path Angle" )
        axes1[2].grid( True )
        axes1[2].legend()

        # Mass
        axes1[3].plot( self.t_ref, np.exp( self.X_ref[4] ), label="Ref" )
        axes1[3].plot( self.t_ref, np.exp( self.X_mpc[4] ), 'r--', label="MPC" )
        axes1[3].set_xlabel( "time (s)" )
        axes1[3].set_ylabel( "mass (kg)" )
        axes1[3].set_title( "Mass" )
        axes1[3].grid( True )
        axes1[3].legend()

        # Font scaling
        for ax in axes1:
            ax.title.set_fontsize(9)
            ax.xaxis.label.set_fontsize(8)
            ax.yaxis.label.set_fontsize(8)
            ax.tick_params(labelsize=7)

        fig1.savefig( "results/mpc_states.png",  dpi=600, bbox_inches='tight' )

        # Controls
        fig2, axes2 = plt.subplots( 2, 1, figsize=(3.5, 3.5), sharex=False, constrained_layout=True )
        axes2.flatten()

        t = self.t_ref[ :self.U_mpc.shape[1] ]  # time vector for controls (same length as U_mpc)
        X = self.X_ref[ :, :self.U_mpc.shape[1] ]  # corresponding states for controls
        
        T_vals_ref     = self.U_ref[2] * np.exp( self.X_ref[4] )   # u3 * m = T (N)
        alpha_vals_ref = np.arctan2( self.U_ref[1], self.U_ref[0] ) # angle of attack

        T_vals_mpc     = self.U_mpc[2] * np.exp( X[4] )   # u3 * m = T (N)
        alpha_vals_mpc = np.arctan2( self.U_mpc[1], self.U_mpc[0] ) # angle of attack

        # Thrust
        axes2[0].plot( self.t_ref, T_vals_ref / 1e3, label="Ref" )
        axes2[0].plot( t, T_vals_mpc / 1e3, 'r--', label="MPC" )
        axes2[0].set_xlabel( "time (s)" )
        axes2[0].set_ylabel( "T (kN)" )
        axes2[0].set_title( "Thrust" )
        axes2[0].grid( True )
        axes2[0].legend()

        # Angle of attack
        axes2[1].plot( self.t_ref, np.rad2deg(alpha_vals_ref), label="Ref" )
        axes2[1].plot( t, np.rad2deg(alpha_vals_mpc), 'r--', label="MPC" )
        axes2[1].set_xlabel( "time (s)" )
        axes2[1].set_ylabel( "alpha (deg)" )
        axes2[1].set_title( "Angle of Attack" )
        axes2[1].grid( True )
        axes2[1].legend()

        # Font scaling
        for ax in axes2:
            ax.title.set_fontsize(9)
            ax.xaxis.label.set_fontsize(8)
            ax.yaxis.label.set_fontsize(8)
            ax.tick_params(labelsize=7)

        fig2.savefig( "results/mpc_controls.png",  dpi=600, bbox_inches='tight' )