import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

from utils import air_density, build_trapezoidal_constraints


g0  = 9.80665               # Standard gravity (m/s^2)
R0  = 6.3781e6              # Earth radius (m)


class ReferenceTrajectory:
    def __init__( self, rocket ):
        self.rocket = rocket
        self.X_opt = None
        self.U_opt = None
        self.tf = None

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
        h, s, V, gamma, z, t = x
        u1, u2, u3, tf = u

        r = h + R0
        Dm  = self.drag_accel( h, V, z )          # D/m

        h_dot     = tf * V * np.sin( gamma )
        s_dot     = tf * V * np.cos( gamma )
        V_dot     = tf * ( -u1 - Dm - ( g0 * np.sin( gamma ) ) )
        gamma_dot = tf * ( -( u2 / V ) - ( g0 * np.cos( gamma ) / V ) )
        z_dot     = -tf * u3 / ( g0 * self.rocket.Isp )
        t_dot     = tf

        return np.array( [ h_dot, s_dot, V_dot, gamma_dot, z_dot, t_dot ] ) 


    def compute_ABC( self, xk, uk ):
        """
        Compute linearization matrices A, B, and affine offset c at (xk, uk).

        Linearized dynamics:
            dx/dtau = A @ x + B @ u + c

        where c = f(xk,uk) - A @ xk - B @ uk  (Eq. 20)

        Returns A (6x6), B (6x4), c (6,)
        """
        h, s, V, gamma, z, t = xk
        u1, u2, u3, tf       = uk

        r = h + R0

        rho = air_density( h )
        m   = np.exp( z )
        D   = 0.5 * rho * V**2 * self.rocket.S_ref * self.rocket.C_D
        Dm  = D / m                            # D/m

        # ── Matrix A = df/dx ──────────────────────────────────────────────────────
        A = np.zeros( ( 6, 6 ) )

        # Row 0 : h_dot = tf * V * sin(gamma)
        A[0, 2] = tf * np.sin( gamma )           # d/dV
        A[0, 3] = tf * V * np.cos( gamma )       # d/d_gamma

        # Row 1 : s_dot = tf * V * cos(gamma)
        A[1, 2] = tf * np.cos( gamma )           # d/dV
        A[1, 3] = -tf * V * np.sin( gamma )      # d/d_gamma

        # Row 2 : V_dot = tf*(-u1 - D/m - g0 sin(gamma))
        A[2, 2] = - 2 * tf * Dm / V                 # d/dV
        A[2, 3] = -tf * g0 * np.cos( gamma )      # d/d_gamma
        A[2, 4] = tf * Dm                           # d/z 

        # Row 3 : gamma_dot = tf*(-u2/V - g0 cos(gamma))
        A[3, 2] = tf * ( u2 +  g0 * np.cos( gamma ) ) / ( V**2 )                # d/dV
        A[3, 3] = tf * g0 * np.sin( gamma ) / V                                  # d/d_gamma

        # Rows 4 & 5 (z_dot, t_dot) have no state dependence
        # => remain zero

        # ── Matrix B = df/du ──────────────────────────────────────────────────────
        B = np.zeros( ( 6, 4 ) )

        # Row 0 : h_dot = tf*V*sin(gamma)
        B[0, 3] = V * np.sin( gamma )            # d/d_tf

        # Row 1 : s_dot = tf*V*cos(gamma)
        B[1, 3] = V * np.cos( gamma )            # d/d_tf

        # Row 2 : V_dot = tf*(-u1 - D/m - sin(gamma)/r^2)
        B[2, 0] = -tf                          # d/d_u1
        B[2, 3] = -u1 - Dm - ( g0 * np.sin( gamma ) ) # d/d_tf

        # Row 3 : gamma_dot = tf*(-u2/V - cos(gamma)/(r^2*V))
        B[3, 1] = -tf / V                      # d/d_u2
        B[3, 3] = -( u2 / V ) - ( g0 * np.cos( gamma ) / V )  # d/d_tf

        # Row 4 : z_dot = -tf*u3/Isp
        B[4, 2] = -tf / ( g0 * self.rocket.Isp )                  # d/d_u3
        B[4, 3] = -u3 / ( g0 * self.rocket.Isp )                  # d/d_tf

        # Row 5 : t_dot = tf
        B[5, 3] = 1.0                          # d/d_tf

        # ── Affine offset c = f(xk,uk) - A@xk - B@uk ─────────────────────────────
        fk = self.dynamics( xk, uk )
        C  = fk - A @ xk - B @ uk

        return A, B, C

    
    def solve( self, x0, target_pos, N=20, max_iter=20, tol=1e-2, tf_guess=20.0 ):
        """
        Solve the fuel-optimal landing trajectory via sequential convex programming.

        x0       : (6,) initial state
        N        : number of discrete intervals (N+1 nodes, paper uses 20 intervals / 21 points)
        max_iter : max SCP outer iterations
        tol      : convergence tolerance (Eq. 24)
        tf_guess : initial guess for terminal time (s)

        Returns
        -------
        X_opt : (6, N+1) optimal state trajectory
        U_opt : (4, N+1) optimal control trajectory
        """

        tau_nodes = np.linspace( 0, 1, N + 1 )   # fixed grid on [0,1]
        dtau = tau_nodes[ 1 ] - tau_nodes[ 0 ]

        # ── Initial guess: linear interpolation of states, constant control 
        # This is needed to be able to compute the linearization matrices A, B, C at each node for the first optimization iteration
        xf_guess = np.array([
            0.0,
            1000.0,
            1.0,
            np.deg2rad( -90 ),
            self.rocket.z_dry,
            tf_guess
        ])

        X_ref = np.array([
            x0 + ( xf_guess - x0 ) * tau for tau in tau_nodes
        ]).T                                    # shape (6, N+1)

        # Rough control guess: hover thrust direction, constant tf
        u_guess = np.array([
            g0,
            0.0, 
            g0, 
            tf_guess
        ])

        U_ref = np.tile( u_guess[ :, None ], ( 1, N + 1 ) )   # shape (4, N+1)

        X_prev = X_ref.copy()
        U_prev = U_ref.copy()

        for iteration in range( max_iter ):

            # ── Compute linearization at every node ──────────────────────────────
            Ak_list = []
            Bk_list = []
            Ck_list = []

            for i in range( N + 1 ):
                A, B, C = self.compute_ABC( X_prev[:, i], U_prev[:, i] )
                Ak_list.append( A )
                Bk_list.append( B )
                Ck_list.append( C )

            # ── CVXPY decision variables ──────────────────────────────────────────
            X = cp.Variable( ( 6, N + 1 ) )   # states
            U = cp.Variable( ( 4, N + 1 ) )   # controls
            # Note: tf is U[3, :] -- same value at every node (enforced below)

            # ── Warm Start ──────────────────────────────────────────
            X.value = X_prev
            U.value = U_prev

            constraints = []

            # ── Dynamics constraints (trapezoidal, Eq. 26) ───────────────────────
            constraints += build_trapezoidal_constraints(
                X, U, Ak_list, Bk_list, Ck_list, dtau
            )

            # ── tf must be the same scalar at every node ──────────────────────────
            for i in range( 1, N + 1 ):
                constraints.append( U[ 3, i ] == U[ 3, 0 ] )

            # ── Initial boundary conditions (Eq. 3) ──────────────────────────────
            constraints.append( X[ :, 0 ] == x0 )

            # ── Terminal boundary conditions (Eq. 3 & 4) ─────────────────────────
            constraints.append( X[ 0, N ] == target_pos[ 0 ] )        # hf = 0
            constraints.append( X[ 1, N ] == target_pos[ 1 ] )     # sf = 1000 m
            constraints.append( X[ 2, N ] <= self.rocket.V_safe )     # Vf <= Vsafe
            constraints.append( X[ 3, N ] == np.deg2rad( -90 ) )    # gamma_f = -90 deg

            # Fuel constraint: z_f >= z_dry  (Eq. 4: mf >= m_dry)
            constraints.append( X[ 4, N ] >= self.rocket.z_dry )

            # Angle of attack terminal constraint (Eq. 13, linearized form):
            # |u2/u1| <= tan(alpha_safe)  at final node
            constraints.append(
                cp.abs( U[ 1, N ] ) <= np.tan( self.rocket.alpha_safe ) * U[ 0, N ]
            )

            # ── Process constraints (Eq. 14) ─────────────────────────────────────
            for i in range(N + 1):
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

            # tf must be positive
            constraints.append( U[3, 0] >= 5.0 )

            # ── Objective: maximize final mass = minimize -z_f  (Eq. 12) ─────────
            objective = cp.Minimize( -X[4, N] )

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
            tf_val = U_sol[ 3, 0 ]

            print(f"  Iter {iteration+1}: tf = {tf_val:.3f} s | "
                f"fuel = {(np.exp(x0[4]) - np.exp(X_sol[4,-1])):.1f} kg | "
                f"dx={dx:.2e}, du={du:.2e}")

            X_prev = X_sol.copy()
            U_prev = U_sol.copy()

            if dx < tol and du < tol:
                print(f"  Converged at iteration {iteration+1}.")
                break

        self.X_opt = X_sol
        self.U_opt = U_sol
        self.tf = U_sol[ 3, 0 ]

        fuel_opt = np.exp( x0[4] ) - np.exp( self.X_opt[4, -1] )
        print(f"\nOptimal tf   = {self.tf:.2f} s")
        print(f"dt    = {self.tf / 30:.2f} s")
        print(f"Fuel used    = {fuel_opt:.1f} kg")
        print(f"Final mass   = {np.exp(self.X_opt[4,-1]):.1f} kg")



    
    def plot( self ):
        """Plot states and controls of the fuel-optimal reference trajectory"""

        tau = np.linspace( 0, 1, self.X_opt.shape[1] )
        t = tau * self.tf

        fig1, axes1 = plt.subplots( 4, 1, figsize=(3.5, 7), sharex=False, constrained_layout=True )
        axes1.flatten()

        # States
        # Trajectory: altitude vs horizontal position
        axes1[0].plot(self.X_opt[1], self.X_opt[0], marker='o', markersize=4, fillstyle='none' )
        axes1[0].set_xlabel( "Downrange s (m)" )
        axes1[0].set_ylabel( "Altitude h (m)" )
        axes1[0].set_title( "Trajectory" )
        axes1[0].grid(True)

        # Speed
        axes1[1].plot( t, self.X_opt[2], marker='o', markersize=4, fillstyle='none')
        axes1[1].set_xlabel( "time (s)" )
        axes1[1].set_ylabel("V (m/s)")
        axes1[1].set_title("Velocity")
        axes1[1].grid( True )
        
        # Flight path angle
        axes1[2].plot(t, np.rad2deg( self.X_opt[ 3 ] ), marker='o', markersize=4, fillstyle='none' )
        axes1[2].set_xlabel( "time (s)" )
        axes1[2].set_ylabel( "gamma (deg)" )
        axes1[2].set_title( "Flight Path Angle" )
        axes1[2].grid( True )

        # Mass
        axes1[3].plot( t, np.exp( self.X_opt[4] ), marker='o', markersize=4, fillstyle='none' )
        axes1[3].set_xlabel( "time (s)" )
        axes1[3].set_ylabel( "mass (kg)" )
        axes1[3].set_title( "Mass" )
        axes1[3].grid( True )

        # Font scaling
        for ax in axes1:
            ax.title.set_fontsize(9)
            ax.xaxis.label.set_fontsize(8)
            ax.yaxis.label.set_fontsize(8)
            ax.tick_params(labelsize=7)

        fig1.savefig( "results/reference_states.png",  dpi=600, bbox_inches='tight' )

        # Controls
        fig2, axes2 = plt.subplots( 2, 1, figsize=(3.5, 3.5), sharex=False, constrained_layout=True )
        axes2.flatten()

        T_vals     = self.U_opt[2] * np.exp( self.X_opt[4] )   # u3 * m = T (N)
        alpha_vals = np.arctan2( self.U_opt[1], self.U_opt[0] ) # angle of attack

        # Thrust
        axes2[0].plot( t, T_vals / 1e3, marker='o', markersize=4, fillstyle='none' )
        axes2[0].set_xlabel( "time (s)" )
        axes2[0].set_ylabel( "T (kN)" )
        axes2[0].set_title( "Thrust" )
        axes2[0].grid( True )

        # Angle of attack
        axes2[1].plot(t, np.rad2deg(alpha_vals), marker='o', markersize=4, fillstyle='none' )
        axes2[1].set_xlabel( "time (s)" )
        axes2[1].set_ylabel( "alpha (deg)" )
        axes2[1].set_title( "Angle of Attack" )
        axes2[1].grid( True )

        # Font scaling
        for ax in axes2:
            ax.title.set_fontsize(9)
            ax.xaxis.label.set_fontsize(8)
            ax.yaxis.label.set_fontsize(8)
            ax.tick_params(labelsize=7)

        fig2.savefig( "results/reference_controls.png",  dpi=600, bbox_inches='tight' )


        