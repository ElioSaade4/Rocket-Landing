import numpy as np

rho0    = 1.225                 # sea-level air density (kg/m^3)
beta    = 0.0001                # density scale height (1/m)


def air_density( h ):
    """Atmospheric density at altitude h (m)."""
    return rho0 * np.exp( -beta * h )


def build_trapezoidal_constraints( X, U, Ak_list, Bk_list, Ck_list, dt ):
    """
    Build the list of CVXPY trapezoidal collocation constraints.

    X : cp.Variable (6, N+1)   -- states  at each node
    U : cp.Variable (4, N+1)   -- controls at each node
    Ak_list, Bk_list, Ck_list  -- linearization matrices at each node (numpy)
    dt: float                  -- step time
    """
    constraints = []
    N = X.shape[ 1 ] - 1

    for i in range( N ):
        # dynamics at node i and i+1
        fi  = Ak_list[ i ]  @ X[ :, i ]   + Bk_list[ i ]  @ U[ :, i ]   + Ck_list[ i ]
        fi1 = Ak_list[ i+1 ] @ X[ :, i+1 ] + Bk_list[ i+1 ] @ U[ :, i+1 ] + Ck_list[ i+1 ]

        # trapezoidal rule: x_{i+1} = x_i + dtau/2 * (f_i + f_{i+1})
        constraints.append(
            X[ :, i+1 ] == X[ :, i ] + ( dt / 2.0 ) * ( fi + fi1 )
        )

    return constraints