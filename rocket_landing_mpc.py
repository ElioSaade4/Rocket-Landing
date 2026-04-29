"""
Convex MPC for Rocket Vertical Landing
Based on: Wang & Song (2018), "Convex Model Predictive Control for Rocket Vertical Landing"

State vector:  x = [r, s, V, gamma, z, t]
                    r     : altitude (m)
                    s     : horizontal position (m)
                    V     : speed (m/s)
                    gamma : flight path angle (rad)
                    z     : ln(mass)  -- log of rocket mass
                    t     : time (s)  -- physical time (state, not independent variable)

Control vector: u = [u1, u2, u3, tf]
                    u1 = T*cos(alpha)/m   (normalized thrust, axial)
                    u2 = T*sin(alpha)/m   (normalized thrust, lateral)
                    u3 = T/m             (normalized thrust magnitude)
                    tf                   (terminal time, free optimization variable)

Independent variable: tau in [0, 1]  (normalized time, NOT physical time)
"""

from dataclasses import dataclass

import numpy as np
import cvxpy as cp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# 1. ROCKET PARAMETERS  (Table 1 of the paper)
# ─────────────────────────────────────────────
@dataclass
class RocketParams:
    m0      = 55000.0        # initial mass (kg)
    m_dry   = 49000.0        # dry mass (kg)
    S_ref   = 8.0            # reference area (m^2)
    C_D     = 0.25           # drag coefficient
    T_min   = 412700.0       # minimum thrust (N)
    T_max   = 1375600.0      # maximum thrust (N)
    Isp     = 443.0          # specific impulse (s)
    g0      = 9.80665        # standard gravity (m/s^2)
    rho0    = 1.225          # sea-level air density (kg/m^3)
    beta    = 0.0001         # density scale height (1/m)
    R0      = 6.3781e6       # Earth radius (m)

    # Derived
    z0      = np.log( m0 )
    z_dry   = np.log( m_dry )

p = RocketParams()

# ─────────────────────────────────────────────
# 2. INITIAL & TERMINAL CONDITIONS  (Table 2)
# ─────────────────────────────────────────────

# Note: paper uses dimensionless variables; here we keep SI units for clarity.
# You may want to nondimensionalise before solving.

# Dimensionless final conditions
xf_target = np.array([
    0.0,              # rf  : final altitude (landing)
    1000.0,           # sf  : final horizontal pos (m)
    # V, gamma, z, t are constrained separately below
])

# Dimensionless terminal constraints (Eq. 4)
V_safe     = 1.0             # max landing speed (m/s)
alpha_safe = np.deg2rad( 2.0 )  # max landing angle of attack (rad)
gamma_f    = np.deg2rad( -90 )  # must land vertically

# Process constraints (Eq. 5)
alpha_min = np.deg2rad( -10 )
alpha_max = np.deg2rad( 10 )

# ─────────────────────────────────────────────
# 3. HELPER FUNCTIONS
# ─────────────────────────────────────────────

# Dimensional air density model: rho = rho0 * exp(-beta * h)
def air_density( h ):
    """Atmospheric density at altitude h (m)."""
    # h = r - p.R0
    # h = r
    return p.rho0 * np.exp( -p.beta * h )


def drag_accel(h, V, z):
    """
    D/m term in the equations of motion.
    D = 0.5 * rho * V^2 * Sref * CD
    m = exp(z)
    Returns D/m.
    """
    rho = air_density( h )
    D   = 0.5 * rho * ( V **2 ) * p.S_ref * p.C_D
    m   = np.exp( z )
    return D / m

def dynamics( tau, x, u ):
    """
    Nonlinear continuous dynamics dx/dtau = f(x, u).
    tau in [0,1] is the normalized independent variable (Eq. 8).
    u = [u1, u2, u3, tf]
    """
    h, s, V, gamma, z, t = x
    u1, u2, u3, tf       = u

    r = h + p.R0
    Dm  = drag_accel( h, V, z )          # D/m

    h_dot     = tf * V * np.sin( gamma )
    s_dot     = tf * V * np.cos( gamma )
    V_dot     = tf * ( -u1 - Dm - ( p.g0 * np.sin( gamma ) ) )
    gamma_dot = tf * ( -( u2 / V ) - ( p.g0 * np.cos( gamma ) / V ) )
    z_dot     = -tf * u3 / ( p.g0 * p.Isp )
    t_dot     = tf

    return np.array( [ h_dot, s_dot, V_dot, gamma_dot, z_dot, t_dot ] ) 

# ─────────────────────────────────────────────
# 4. LINEARIZATION  (Eq. 17-20)
# ─────────────────────────────────────────────

def compute_ABC( xk, uk ):
    """
    Compute linearization matrices A, B, and affine offset c at (xk, uk).

    Linearized dynamics:
        dx/dtau = A @ x + B @ u + c

    where c = f(xk,uk) - A @ xk - B @ uk  (Eq. 20)

    Returns A (6x6), B (6x4), c (6,)
    """
    h, s, V, gamma, z, t = xk
    u1, u2, u3, tf       = uk

    r = h + p.R0

    rho = air_density( h )
    m   = np.exp(z)
    D   = 0.5 * rho * V**2 * p.S_ref * p.C_D
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
    A[2, 3] = -tf * p.g0 * np.cos( gamma )      # d/d_gamma
    A[2, 4] = tf * Dm                           # d/z 

    # Row 3 : gamma_dot = tf*(-u2/V - g0 cos(gamma))
    A[3, 2] = tf * ( u2 +  p.g0 * np.cos( gamma ) ) / ( V**2 )                # d/dV
    A[3, 3] = tf * p.g0 * np.sin( gamma ) / V                                  # d/d_gamma

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
    B[2, 3] = -u1 - Dm - ( p.g0 * np.sin( gamma ) ) # d/d_tf

    # Row 3 : gamma_dot = tf*(-u2/V - cos(gamma)/(r^2*V))
    B[3, 1] = -tf / V                      # d/d_u2
    B[3, 3] = -( u2 / V ) - ( p.g0 * np.cos( gamma ) / V )  # d/d_tf

    # Row 4 : z_dot = -tf*u3/Isp
    B[4, 2] = -tf / ( p.g0 * p.Isp )                  # d/d_u3
    B[4, 3] = -u3 / ( p.g0 * p.Isp )                  # d/d_tf

    # Row 5 : t_dot = tf
    B[5, 3] = 1.0                          # d/d_tf

    # ── Affine offset c = f(xk,uk) - A@xk - B@uk ─────────────────────────────
    fk = dynamics( None, xk, uk )
    C  = fk - A @ xk - B @ uk

    return A, B, C

# ─────────────────────────────────────────────
# 5. DISCRETIZATION  (Eq. 26 - trapezoidal rule)
# ─────────────────────────────────────────────

def build_trapezoidal_constraints( X, U, Ak_list, Bk_list, Ck_list, dtau ):
    """
    Build the list of CVXPY trapezoidal collocation constraints.

    X : cp.Variable (6, N+1)   -- states  at each node
    U : cp.Variable (4, N+1)   -- controls at each node
    Ak_list, Bk_list, Ck_list  -- linearization matrices at each node (numpy)
    dtau : float               -- uniform step in tau
    """
    constraints = []
    N = X.shape[1] - 1

    for i in range(N):
        # dynamics at node i and i+1
        fi  = Ak_list[i]  @ X[:, i]   + Bk_list[i]  @ U[:, i]   + Ck_list[i]
        fi1 = Ak_list[i+1] @ X[:, i+1] + Bk_list[i+1] @ U[:, i+1] + Ck_list[i+1]

        # trapezoidal rule: x_{i+1} = x_i + dtau/2 * (f_i + f_{i+1})
        constraints.append(
            X[:, i+1] == X[:, i] + (dtau / 2.0) * (fi + fi1)
        )

    return constraints

# ─────────────────────────────────────────────
# 6. SEQUENTIAL CONVEX OPTIMIZATION (Problem 2)
# ─────────────────────────────────────────────

def solve_trajectory_optimization( x0, N=20, max_iter=15, tol=1e-4, tf_guess=20.0 ):
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
        p.z_dry,
        tf_guess
    ])

    X_ref = np.array([
        x0 + ( xf_guess - x0 ) * tau for tau in tau_nodes
    ]).T                                    # shape (6, N+1)

    # Rough control guess: hover thrust direction, constant tf
    u_guess = np.array([
        p.g0,
        0.0, 
        p.g0, 
        tf_guess
    ])

    U_ref   = np.tile( u_guess[ :, None ], ( 1, N + 1 ) )   # shape (4, N+1)

    X_prev = X_ref.copy()
    U_prev = U_ref.copy()

    for iteration in range( max_iter ):

        # ── Compute linearization at every node ──────────────────────────────
        Ak_list = []
        Bk_list = []
        Ck_list = []

        for i in range( N + 1 ):
            A, B, C = compute_ABC( X_prev[:, i], U_prev[:, i] )
            Ak_list.append( A )
            Bk_list.append( B )
            Ck_list.append( C )

        # ── CVXPY decision variables ──────────────────────────────────────────
        X = cp.Variable( (6, N + 1) )   # states
        U = cp.Variable( (4, N + 1) )   # controls
        # Note: tf is U[3, :] -- same value at every node (enforced below)

        constraints = []

        # ── Dynamics constraints (trapezoidal, Eq. 26) ───────────────────────
        constraints += build_trapezoidal_constraints(
            X, U, Ak_list, Bk_list, Ck_list, dtau
        )

        # ── tf must be the same scalar at every node ──────────────────────────
        for i in range(1, N + 1):
            constraints.append( U[3, i] == U[3, 0] )

        tf_var = U[3, 0]   # convenient alias

        # ── Initial boundary conditions (Eq. 3) ──────────────────────────────
        constraints.append(X[:, 0] == x0)

        # ── Terminal boundary conditions (Eq. 3 & 4) ─────────────────────────
        constraints.append(X[0, N] == 0.0)        # hf = 0
        constraints.append(X[1, N] == 1000.0)     # sf = 1000 m
        constraints.append(X[2, N] <= V_safe)     # Vf <= Vsafe
        constraints.append(X[3, N] == gamma_f)    # gamma_f = -90 deg

        # Fuel constraint: z_f >= z_dry  (Eq. 4: mf >= m_dry)
        constraints.append(X[4, N] >= p.z_dry)

        # Angle of attack terminal constraint (Eq. 13, linearized form):
        # |u2/u1| <= tan(alpha_safe)  at final node
        constraints.append(
            cp.abs(U[1, N]) <= np.tan(alpha_safe) * U[0, N]
        )

        # ── Process constraints (Eq. 14) ─────────────────────────────────────
        for i in range(N + 1):
            ez_k = np.exp(-X_prev[4, i])        # e^{-z^k} (fixed, from linearization)

            # T_min * e^{-z} <= u3 <= T_max * e^{-z}   (from Eq. 14, after e^z linearization)
            constraints.append(p.T_min * ez_k <= U[2, i])
            constraints.append(U[2, i] <= p.T_max * ez_k)

            # Angle of attack: |u2/u1| <= tan(alpha_max)  =>  |u2| <= tan(alpha_max)*u1
            constraints.append(
                cp.abs(U[1, i]) <= np.tan(alpha_max) * U[0, i]
            )

            # SOCP / lossless convexification (Eq. 22):
            # u1^2 + u2^2 <= u3^2
            constraints.append(
                cp.norm(U[:2, i], 2) <= U[2, i]
            )

        # tf must be positive
        constraints.append( U[3, 0] >= 5.0)
        # constraints.append(U[3, 0] <= 60.0)

        # ── Objective: maximize final mass = minimize -z_f  (Eq. 12) ─────────
        objective = cp.Minimize( -X[4, N] )

        # ── Solve ─────────────────────────────────────────────────────────────
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS, verbose=False)

        if prob.status not in ["optimal", "optimal_inaccurate"]:
            print(f"  Iter {iteration+1}: solver status = {prob.status}. Stopping.")
            break

        X_sol = X.value
        U_sol = U.value

        # ── Convergence check (Eq. 24) ────────────────────────────────────────
        dx = np.max(np.abs(X_sol - X_prev))
        du = np.max(np.abs(U_sol - U_prev))
        tf_val = U_sol[3, 0]
        print(f"  Iter {iteration+1}: tf = {tf_val:.3f} s | "
              f"fuel = {(np.exp(x0[4]) - np.exp(X_sol[4,-1])):.1f} kg | "
              f"dx={dx:.2e}, du={du:.2e}")
        
        # print( X_sol )
        # print( U_sol )

        X_prev = X_sol.copy()
        U_prev = U_sol.copy()

        if dx < tol and du < tol:
            print(f"  Converged at iteration {iteration+1}.")
            break

    return X_sol, U_sol


# ─────────────────────────────────────────────
# 7. PLOTTING
# ─────────────────────────────────────────────

def plot_trajectory(X, U, title="Rocket Landing Trajectory"):
    """Plot states and controls, matching Fig. 2 & 3 of the paper."""
    tau = np.linspace(0, 1, X.shape[1])
    tf  = U[3, 0]
    t   = tau * tf

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(title, fontsize=14)

    # States
    axes[0, 0].plot(X[1], X[0])
    axes[0, 0].set_xlabel("s (m)")
    axes[0, 0].set_ylabel("r / altitude (m)")
    axes[0, 0].set_title("Trajectory")
    axes[0, 0].grid(True)

    axes[0, 1].plot(t, X[2])
    axes[0, 1].set_xlabel("time (s)")
    axes[0, 1].set_ylabel("V (m/s)")
    axes[0, 1].set_title("Speed")
    axes[0, 1].grid(True)

    axes[0, 2].plot(t, np.rad2deg(X[3]))
    axes[0, 2].set_xlabel("time (s)")
    axes[0, 2].set_ylabel("gamma (deg)")
    axes[0, 2].set_title("Flight Path Angle")
    axes[0, 2].grid(True)

    # Controls
    T_vals     = U[2] * np.exp(X[4])   # u3 * m = T (N)
    alpha_vals = np.arctan2(U[1], U[0]) # angle of attack

    axes[1, 0].plot(t, T_vals / 1e3)
    axes[1, 0].set_xlabel("time (s)")
    axes[1, 0].set_ylabel("T (kN)")
    axes[1, 0].set_title("Thrust")
    axes[1, 0].grid(True)

    axes[1, 1].plot(t, np.rad2deg(alpha_vals))
    axes[1, 1].set_xlabel("time (s)")
    axes[1, 1].set_ylabel("alpha (deg)")
    axes[1, 1].set_title("Angle of Attack")
    axes[1, 1].grid(True)

    axes[1, 2].plot(t, np.exp(X[4]))
    axes[1, 2].set_xlabel("time (s)")
    axes[1, 2].set_ylabel("mass (kg)")
    axes[1, 2].set_title("Mass")
    axes[1, 2].grid(True)

    plt.tight_layout()
    plt.show()

# ─────────────────────────────────────────────
# 9. MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 60)
    print("STEP 1: Trajectory Optimization (Sequential Convex Programming)")
    print("=" * 60)

    # Dimensional initial conditions
    x0 = np.array([
        3000.0,             # h0  : initial altitude (m)
        0.0,                # s0  : initial horizontal pos (m)
        280.0,              # V0  : initial speed (m/s)
        np.deg2rad( -65 ),  # gamma0 : initial flight path angle (rad)
        p.z0,               # z0  : ln(m0)
        0.0                 # t0  : initial time (s)
    ])
    
    X_opt, U_opt = solve_trajectory_optimization( 
        x0 = x0,
        N=20, 
        max_iter=100, 
        tol=1e-4, 
        tf_guess=20.0
    )

    tf_opt   = U_opt[ 3, 0 ]
    fuel_opt = np.exp( x0[4] ) - np.exp( X_opt[4, -1] )
    print(f"\nOptimal tf   = {tf_opt:.2f} s")
    print(f"Fuel used    = {fuel_opt:.1f} kg")
    print(f"Final mass   = {np.exp(X_opt[4,-1]):.1f} kg")

    plot_trajectory(X_opt, U_opt, title="Trajectory Optimization Result")

    # ── Uncomment to run MPC ──────────────────────────────────────────────────
    # print("\n" + "="*60)
    # print("STEP 2: MPC Guidance")
    # print("="*60)
    # X_mpc, U_mpc = run_mpc(X_opt, U_opt, guidance_dt=0.1)
    # print(f"MPC landed at h={X_mpc[-1,0]:.2f}m, s={X_mpc[-1,1]:.1f}m")
