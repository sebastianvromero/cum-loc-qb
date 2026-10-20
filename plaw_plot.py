from __future__ import annotations


import argparse
import math
import os
from pathlib import Path
from typing import Tuple


# # Prevent BLAS oversubscription inside joblib workers.
# os.environ.setdefault("OMP_NUM_THREADS", "1")
# os.environ.setdefault("MKL_NUM_THREADS", "1")
# os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
# os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from numba import njit, set_num_threads
from scipy.integrate import quad


plt.rcParams.update({
    'pgf.texsystem': 'pdflatex',
    'text.usetex': True,
    'font.family': 'serif',
    'pgf.preamble': '\n'.join([
        r'\usepackage{amsfonts,amsmath}',
    ]),
    'text.latex.preamble': '\n'.join([
        r'\usepackage{amsfonts,amsmath}',
    ])
})




# =============================================================================
# User defaults
# =============================================================================


MAIN_N_MAX = 10_000
INSET_N_MIN = MAIN_N_MAX // 100
INSET_N_MAX = MAIN_N_MAX + 1
N_REALIZATIONS = 100
N_JOBS = min(N_REALIZATIONS, os.cpu_count() or 1)
SEED = 13
J_DEFAULT = 1.0


# Number of p-values kept in the inset sweep:
#   first two p-groups -> panel (a) and panel (b) insets.
INSET_P_COUNT = 8


P_VALUES = np.asarray(
    [0.0, 0.25, 0.5, 0.75,
     1.0, 1.25, 1.5, 1.75,
     2.0, 3.0, 4.0],
    dtype=np.float64,
)


P_GROUPS = (
    (0.0, 0.25, 0.5, 0.75),
    (1.0, 1.25, 1.5, 1.75),
    (2.0, 3.0, 4.0),
)


# Same tab20 ordering as the previous benchmark.
_TAB20 = plt.get_cmap("tab20").colors
P_COLORS = {
    float(p): _TAB20[i % len(_TAB20)]
    for i, p in enumerate(P_VALUES)
}


ETA_CRIT = 0.6529308944619922
EULER_GAMMA = 0.5772156649015329




# =============================================================================
# Exact finite-N charger normalization
# =============================================================================




def normalization_sigma_table(
    n_max: int,
    p_values: np.ndarray,
    J: float = 1.0,
) -> np.ndarray:
    """
    Exact finite-N Hilbert-Schmidt normalization:


        sigma_p(N)^2 = 2 J^2 N /
            [N + 2 sum_{L=2}^N (N+1-L) L^{-2p}].


    Row N-2 contains sigma_p(N).
    """
    if n_max < 2:
        return np.empty((0, p_values.size), dtype=np.float64)


    N_arr = np.arange(2, n_max + 1, dtype=np.float64)
    L_all = np.arange(2, n_max + 1, dtype=np.float64)
    out = np.empty((n_max - 1, p_values.size), dtype=np.float64)


    idx = np.arange(n_max - 1)
    for ip, p in enumerate(p_values):
        w = L_all ** (-2.0 * p)
        c0 = np.cumsum(w)
        c1 = np.cumsum(L_all * w)
        sum_off = (N_arr + 1.0) * c0[idx] - c1[idx]
        denom = N_arr + 2.0 * sum_off
        out[:, ip] = J * np.sqrt(2.0 * N_arr / denom)


    return out




@njit(cache=True, inline="always")
def total_coefficient_count(N: int) -> float:
    """N_max = 2 N (N+2) (2N-1)/3."""
    return 2.0 * N * (N + 2.0) * (2.0 * N - 1.0) / 3.0




# =============================================================================
# Small Numba helpers
# =============================================================================




@njit(cache=True, inline="always")
def xlogx(x: float) -> float:
    if x <= 0.0:
        return 0.0
    return x * math.log(x)




@njit(cache=True, inline="always")
def add_x_logx(x: float, S: float, R: float) -> Tuple[float, float]:
    S += x
    if x > 0.0:
        R += x * math.log(x)
    return S, R




@njit(cache=True, inline="always")
def pair_sector_six(
    beta_c: float,
    gamma_c: float,
    delta_c: float,
    delta_n: float,
    q0: float,
    q1: float,
    q2: float,
    q3: float,
    left_orientation: bool,
) -> Tuple[float, float, float, float, float, float]:
    """
    Six coefficient magnitudes for one central-site / neighbour relation.


    The first four are the C4-like coefficients and the last two are the C3-
    like coefficients.
    """
    rho = math.sqrt(beta_c * beta_c + gamma_c * gamma_c)
    if rho > 0.0:
        c = beta_c / rho
        s = gamma_c / rho
    else:
        c = 1.0
        s = 0.0


    if left_orientation:
        # central site is the left endpoint; neighbour lies to its right.
        xp = -s * q0 + c * q1
        yp = c * q0 + s * q1
        xm = c * q2 - s * q3
        ym = s * q2 + c * q3


        p1 = abs(-delta_n * yp + delta_c * ym)
        p2 = abs(-delta_n * xm + delta_c * xp)
        p3 = abs(delta_n * xp - delta_c * xm)
        p4 = abs(-delta_n * ym + delta_c * yp)
    else:
        # central site is the right endpoint; neighbour lies to its left.
        yp = c * q0 + s * q1
        xp = -s * q0 + c * q1
        xm = c * q2 - s * q3
        ym = s * q2 + c * q3


        p1 = abs(-delta_c * yp + delta_n * ym)
        p2 = abs(-delta_c * xm + delta_n * xp)
        p3 = abs(delta_c * xp - delta_n * xm)
        p4 = abs(-delta_c * ym + delta_n * yp)


    p5 = abs(rho * q0)
    p6 = abs(rho * q2)
    return p1, p2, p3, p4, p5, p6




# =============================================================================
# One combined realization kernel
# =============================================================================




@njit(cache=True, fastmath=False)
def simulate_one_realization_kernel(
    beta: np.ndarray,
    gamma: np.ndarray,
    delta: np.ndarray,
    diag_z: np.ndarray,
    pair_seed: int,
    p_values: np.ndarray,
    sigma_bound: np.ndarray,
    main_n_max: int,
    inset_n_min: int,
    inset_n_max: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate one full disorder realization and compute:


      main_part[N,p]    for N=2..main_n_max,
      inset_part[N,p]   for N=inset_n_min..inset_n_max and the first
                        INSET_P_COUNT p-values (first two p-groups),
      bounds[N,p]       for N=2..main_n_max.


    The same charger pair variables are used for the participation and bound
    estimators whenever N <= main_n_max.  Pair variables beyond main_n_max are
    still generated from the same pair stream, so the extended inset is the
    continuation of the same disorder realization.


    The C2 contribution is updated in O(1) per outer pair using interval
    difference arrays, because for fixed central k the four charger Gaussian
    magnitudes multiply both |beta_k| and |gamma_k|.
    """
    M = beta.size
    P = p_values.size
    P_inset = min(INSET_P_COUNT, P)


    main_part = np.full((max(main_n_max - 1, 0), P), np.nan, dtype=np.float64)
    bounds = np.full((max(main_n_max - 1, 0), P), np.nan, dtype=np.float64)


    if inset_n_max >= inset_n_min and inset_n_min >= 2:
        n_inset = inset_n_max - inset_n_min + 1
        inset_part = np.full((n_inset, P_inset), np.nan, dtype=np.float64)
    else:
        inset_part = np.empty((0, P_inset), dtype=np.float64)


    # Per-central-site local + neighbor-pair entropy accumulators.
    pair_S = np.zeros((P, M), dtype=np.float64)
    pair_R = np.zeros((P, M), dtype=np.float64)
    local_S = np.zeros(M, dtype=np.float64)
    local_R = np.zeros(M, dtype=np.float64)


    # C2 interval updates, one pair of difference arrays for each p.
    c2_diff_W = np.zeros((P, M + 1), dtype=np.float64)
    c2_diff_U = np.zeros((P, M + 1), dtype=np.float64)


    # Local contribution for every central site is known from the beginning.
    for k in range(M):
        x1 = abs(diag_z[k] * beta[k])
        x2 = abs(diag_z[k] * gamma[k])
        local_S[k] = x1 + x2
        local_R[k] = xlogx(x1) + xlogx(x2)


    # Seed Numba's independent RNG only for charger pair variables.
    np.random.seed(pair_seed)


    # Running C2 interval sums as we scan central sites at each N.
    c2_running_W = np.zeros(P, dtype=np.float64)
    c2_running_U = np.zeros(P, dtype=np.float64)


    for m in range(1, M):
        # ------------------------------------------------------------------
        # Add every pair (k,m), k < m.
        # ------------------------------------------------------------------
        for k in range(m):
            # Four independent standard-normal charger combinations.
            q0 = np.random.normal()
            q1 = np.random.normal()
            q2 = np.random.normal()
            q3 = np.random.normal()


            L = float(m - k + 1)
            logL = math.log(L)


            vals_k = pair_sector_six(
                beta[k], gamma[k], delta[k], delta[m],
                q0, q1, q2, q3, True,
            )
            vals_m = pair_sector_six(
                beta[m], gamma[m], delta[m], delta[k],
                q0, q1, q2, q3, False,
            )


            # C2 outer-pair random factors: the same four q's generate the
            # eight coefficients |beta_c| |q_a| and |gamma_c| |q_a| for every
            # central c strictly between k and m.
            g0 = abs(q0)
            g1 = abs(q1)
            g2 = abs(q2)
            g3 = abs(q3)
            G = g0 + g1 + g2 + g3
            Glog = xlogx(g0) + xlogx(g1) + xlogx(g2) + xlogx(g3)


            # Whenever N=m+1 <= main_n_max we need all P p-values; otherwise
            # we only need the first P_inset values for the extended inset.
            p_limit = P if (m + 1 <= main_n_max) else P_inset
            for ip in range(p_limit):
                w = L ** (-p_values[ip])


                # Add the six pair-sector coefficients to central k.
                s6 = 0.0
                r6 = 0.0
                for z in vals_k:
                    s6 += z
                    r6 += xlogx(z)
                pair_S[ip, k] += w * s6
                pair_R[ip, k] += w * r6


                # Add the six pair-sector coefficients to central m.
                s6m = 0.0
                r6m = 0.0
                for z in vals_m:
                    s6m += z
                    r6m += xlogx(z)
                pair_S[ip, m] += w * s6m
                pair_R[ip, m] += w * r6m


                # C2 interval contribution to every central site c in
                # k+1,...,m-1.  For fixed c the raw entropy terms are
                #   beta_c * q_a * w,
                #   gamma_c * q_a * w.
                # Define
                #   W = sum_a q_a w,
                #   U = sum_a q_a w (log q_a - p log L),
                # then
                #   S_C2 = (|beta|+|gamma|) W,
                #   R_C2 = (|beta|log|beta|+|gamma|log|gamma|)W
                #             + (|beta|+|gamma|)U.
                U = w * (Glog - p_values[ip] * logL * G)
                W = w * G
                lo = k + 1
                hi = m
                c2_diff_W[ip, lo] += W
                c2_diff_W[ip, hi] -= W
                c2_diff_U[ip, lo] += U
                c2_diff_U[ip, hi] -= U


        # ------------------------------------------------------------------
        # Current prefix N=m+1.  The per-k participation ratio needs the
        # cumulative C2 interval values at EVERY central site k.
        # This scan is O(P N) per N, hence O(P N^2) overall.
        # ------------------------------------------------------------------
        N = m + 1


        # Only materialize eta_N where it is actually needed.  The main figure
        # uses every N<=main_n_max; the inset uses every N>=inset_n_min.
        need_main = N <= main_n_max
        need_inset = (
            inset_n_max >= inset_n_min
            and N >= inset_n_min
            and N <= inset_n_max
        )


        if need_main or need_inset:
            p_limit_scan = P if need_main else P_inset


            for ip in range(p_limit_scan):
                c2_running_W[ip] = 0.0
                c2_running_U[ip] = 0.0


            eta_sum = np.zeros(p_limit_scan, dtype=np.float64)


            for k in range(N):
                for ip in range(p_limit_scan):
                    c2_running_W[ip] += c2_diff_W[ip, k]
                    c2_running_U[ip] += c2_diff_U[ip, k]


                    b = abs(beta[k])
                    g = abs(gamma[k])
                    blog = xlogx(b)
                    glog = xlogx(g)


                    Sc2 = (b + g) * c2_running_W[ip]
                    Rc2 = (
                        (blog + glog) * c2_running_W[ip]
                        + (b + g) * c2_running_U[ip]
                    )


                    S = local_S[k] + pair_S[ip, k] + Sc2
                    R = local_R[k] + pair_R[ip, k] + Rc2


                    if S > 0.0 and math.isfinite(S) and math.isfinite(R):
                        eta_sum[ip] += S * math.exp(-R / S)


            den = total_coefficient_count(N)
            eta_N = eta_sum / den


            if need_main:
                row = N - 2
                for ip in range(P):
                    main_part[row, ip] = eta_N[ip]


            if need_inset:
                irow = N - inset_n_min
                for ip in range(P_inset):
                    inset_part[irow, ip] = eta_N[ip]


        # ------------------------------------------------------------------
        # Lower-row tight norm bound.  This is deliberately evaluated only
        # for the main range because the requested figure uses N<=1000 there.
        # It is filled by the companion kernel simulate_bounds_stream().
        # ------------------------------------------------------------------


    return main_part, inset_part, bounds




# =============================================================================
# Exact bound kernel (compact pair stream, no pair matrix storage)
# =============================================================================




@njit(cache=True, fastmath=True)
def simulate_bounds_stream(
    beta: np.ndarray,
    gamma: np.ndarray,
    delta: np.ndarray,
    diag_z: np.ndarray,
    pair_seed: int,
    p_values: np.ndarray,
    sigma_table: np.ndarray,
    main_n_max: int,
) -> np.ndarray:
    """Same realization-level tight bound used previously, but pair data are streamed."""
    M = main_n_max
    P = p_values.size
    out = np.empty((M - 1, P), dtype=np.float64)

    # Prefix sums let the original pairwise C2 bound evaluate each interior
    # battery sum in O(1), without changing the C3/C4 calculations.
    pref_b2 = np.zeros(M + 1, dtype=np.float64)
    pref_g2 = np.zeros(M + 1, dtype=np.float64)
    for i in range(M):
        pref_b2[i + 1] = pref_b2[i] + beta[i] * beta[i]
        pref_g2[i + 1] = pref_g2[i] + gamma[i] * gamma[i]

    c1_raw = (
        abs(diag_z[0])
        * math.sqrt(beta[0] * beta[0] + gamma[0] * gamma[0])
    )
    # Original locality-resolved C2 bound, accumulated pair by pair.
    c2_pairwise = np.zeros(P, dtype=np.float64)

    # W_c(N,p) = sum_{k<m} (m-k-1) L^{-2p} (|A_km|^2+|B_km|^2).
    # The second C2 bound below is B2 <= 2*sqrt(2*W_b*W_c).
    c2_coupling_weight = np.zeros(P, dtype=np.float64)


    left_c3_sq = np.zeros((P, M), dtype=np.float64)
    right_c3_sq = np.zeros((P, M), dtype=np.float64)
    x_c4_sq = np.zeros((P, M), dtype=np.float64)
    y_c4_sq = np.zeros((P, M), dtype=np.float64)


    left_c3_sum = np.zeros(P, dtype=np.float64)
    right_c3_sum = np.zeros(P, dtype=np.float64)
    x_c4_sum = np.zeros(P, dtype=np.float64)
    y_c4_sum = np.zeros(P, dtype=np.float64)


    np.random.seed(pair_seed)


    for m in range(1, M):
        N = m + 1
        c1_raw += (
            abs(diag_z[m])
            * math.sqrt(beta[m] * beta[m] + gamma[m] * gamma[m])
        )
        new_right_sq = np.zeros(P, dtype=np.float64)


        for k in range(m):
            q0 = np.random.normal()
            q1 = np.random.normal()
            q2 = np.random.normal()
            q3 = np.random.normal()


            L = float(m - k + 1)
            q_sq_sum = q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3
            g_abs_sum = abs(q0) + abs(q1) + abs(q2) + abs(q3)
            interior_count = float(m - k - 1)

            # Interior sites are k+1,...,m-1. This is the squared battery
            # norm entering the original pairwise C2 bound.
            inner_sq = (
                (pref_b2[m] - pref_b2[k + 1])
                + (pref_g2[m] - pref_g2[k + 1])
            )

            # Pair-sector bound coefficients (same as the previous tight
            # implementation).
            k31 = beta[k] * q2 - gamma[m] * q0
            k32 = beta[k] * q1 + gamma[m] * q3
            k33 = -beta[m] * q3 - gamma[k] * q0
            k34 = beta[m] * q1 - gamma[k] * q2


            k41 = -delta[k] * q2 + delta[m] * q3
            k42 = -delta[k] * q1 + delta[m] * q0
            k43 = delta[k] * q0 - delta[m] * q1
            k44 = -delta[k] * q3 + delta[m] * q2


            for ip in range(P):
                w = L ** (-p_values[ip])
                w2 = w * w

                # Original locality-resolved estimate:
                #   sum_{k<m} L^{-p} (sum_a |q_a|)
                #       * sqrt(sum_{j=k+1}^{m-1}(beta_j^2+gamma_j^2)).
                if inner_sq > 0.0:
                    c2_pairwise[ip] += (
                        w * math.sqrt(inner_sq) * g_abs_sum
                    )

                # |A_km|^2 + |B_km|^2 = 0.5 * sum_j q_j^2.
                c2_coupling_weight[ip] += (
                    0.5 * interior_count * w2 * q_sq_sum
                )

                old = left_c3_sq[ip, k]
                new = old + w2 * (k31 * k31 + k32 * k32)
                left_c3_sq[ip, k] = new
                left_c3_sum[ip] += math.sqrt(new) - math.sqrt(old)


                new_right_sq[ip] += w2 * (k33 * k33 + k34 * k34)


                oldx = x_c4_sq[ip, k]
                newx = oldx + w2 * (k41 * k41 + k42 * k42)
                x_c4_sq[ip, k] = newx
                x_c4_sum[ip] += math.sqrt(newx) - math.sqrt(oldx)


                oldy = y_c4_sq[ip, k]
                newy = oldy + w2 * (k43 * k43 + k44 * k44)
                y_c4_sq[ip, k] = newy
                y_c4_sum[ip] += math.sqrt(newy) - math.sqrt(oldy)


        # W_b(N) = sum_l min(l, N-l-1) (beta_l^2 + gamma_l^2).
        # Zero-based indices; the power-law charger includes every pair range.
        battery_rank_weight = 0.0
        for ell in range(N):
            rank = min(ell, N - ell - 1)
            battery_rank_weight += rank * (
                beta[ell] * beta[ell] + gamma[ell] * gamma[ell]
            )

        for ip in range(P):
            right_c3_sq[ip, m] = new_right_sq[ip]
            right_c3_sum[ip] += math.sqrt(new_right_sq[ip])


            sigma = sigma_table[N - 2, ip]
            c2_refined = 2.0 * math.sqrt(
                2.0 * battery_rank_weight * c2_coupling_weight[ip]
            )
            # Both expressions are valid upper bounds on the same C2 term.
            # Their minimum remains an upper bound and preserves the sharper
            # locality-sensitive N log N / N regimes for p >= 3/2.
            c2_bound = min(c2_pairwise[ip], c2_refined)
            raw_total = (
                c1_raw
                + c2_bound
                + left_c3_sum[ip]
                + right_c3_sum[ip]
                + x_c4_sum[ip]
                + y_c4_sum[ip]
            )
            out[N - 2, ip] = sigma * raw_total


    return out




# =============================================================================
# One realization wrapper
# =============================================================================




def simulate_one_realization(
    realization_id: int,
    realization_n_max: int,
    p_values: np.ndarray,
    sigma_table: np.ndarray,
    main_n_max: int,
    inset_n_min: int,
    inset_n_max: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Independent disorder realization used by joblib."""
    set_num_threads(1)


    ss = np.random.SeedSequence([int(seed), int(realization_id)])
    rng = np.random.default_rng(ss)


    beta = np.ascontiguousarray(rng.standard_normal(realization_n_max))
    gamma = np.ascontiguousarray(rng.standard_normal(realization_n_max))
    delta = np.ascontiguousarray(rng.standard_normal(realization_n_max))
    diag_z = np.ascontiguousarray(rng.standard_normal(realization_n_max))


    # Use a deterministic, independent Numba RNG stream for all charger-pair
    # variables.  Calling the same kernel with the same pair seed reproduces
    # exactly the same pair stream for the bound calculation.
    pair_seed = int(
        (seed + 104729 * (realization_id + 1)) % 2_000_000_000
    )


    main_part, inset_part, _ = simulate_one_realization_kernel(
        beta,
        gamma,
        delta,
        diag_z,
        pair_seed,
        p_values,
        sigma_table,
        main_n_max,
        inset_n_min,
        inset_n_max,
    )


    bounds = simulate_bounds_stream(
        beta[:main_n_max],
        gamma[:main_n_max],
        delta[:main_n_max],
        diag_z[:main_n_max],
        pair_seed,
        p_values,
        sigma_table,
        main_n_max,
    )


    if not np.all(np.isfinite(main_part)):
        raise FloatingPointError(
            f"Non-finite main participation values in realization "
            f"{realization_id}."
        )
    if inset_part.size and not np.all(np.isfinite(inset_part)):
        raise FloatingPointError(
            f"Non-finite inset participation values in realization "
            f"{realization_id}."
        )
    if not np.all(np.isfinite(bounds)):
        raise FloatingPointError(
            f"Non-finite bound values in realization {realization_id}."
        )


    return main_part, inset_part, bounds




# =============================================================================
# Aggregation
# =============================================================================




def mean_and_sem(stack: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.mean(stack, axis=0)
    if stack.shape[0] > 1:
        sem = np.std(stack, axis=0, ddof=1) / math.sqrt(stack.shape[0])
    else:
        sem = np.full_like(mean, np.nan)
    return mean, sem




# =============================================================================
# eta_crit validation
# =============================================================================




def eta_r(r: float) -> float:
    """Analytic conditional participation ratio eta(r)."""
    pref = math.exp(EULER_GAMMA / 2.0) / (2.0 * math.sqrt(math.pi))
    if r <= 0.0:
        return pref
    return pref * (1.0 + r) * math.exp(
        -(r / (1.0 + r)) * math.log(r)
    )




def eta_crit_quadrature() -> float:
    """Independent numerical evaluation of the supplied eta_crit integral."""
    def integrand(r: float) -> float:
        return eta_r(r) * 2.0 / (math.pi * (1.0 + r * r))


    val, _ = quad(
        integrand,
        0.0,
        np.inf,
        epsabs=1e-11,
        epsrel=1e-11,
        limit=500,
    )
    return float(val)




# =============================================================================
# Output helpers
# =============================================================================




def save_csv(
    path: Path,
    N_values: np.ndarray,
    p_values: np.ndarray,
    mean: np.ndarray,
    sem: np.ndarray,
    prefix: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["N"]
    for p in p_values:
        header += [f"{prefix}_p{p:g}", f"{prefix}_err_p{p:g}"]


    data = np.empty(
        (N_values.size, 1 + 2 * p_values.size),
        dtype=np.float64,
    )
    data[:, 0] = N_values
    col = 1
    for ip in range(p_values.size):
        data[:, col] = mean[:, ip]
        data[:, col + 1] = sem[:, ip]
        col += 2


    np.savetxt(
        path,
        data,
        delimiter=",",
        header=",".join(header),
        comments="",
    )




# =============================================================================
# Plotting
# =============================================================================




def positive_values(mean: np.ndarray, sem: np.ndarray) -> np.ndarray:
    lo = mean - np.where(np.isfinite(sem), sem, 0.0)
    hi = mean + np.where(np.isfinite(sem), sem, 0.0)
    vals = np.concatenate((lo.ravel(), hi.ravel()))
    return vals[np.isfinite(vals) & (vals > 0.0)]




def add_group(ax, N_values, group, mean, sem, ylabel, label):
    p_to_index = {float(p): i for i, p in enumerate(P_VALUES)}
    for p in group:
        ip = p_to_index[float(p)]
        color = P_COLORS[float(p)]
        y = mean[:, ip]
        e = sem[:, ip]
        ax.plot(N_values, y, color=color, lw=1.25, label=fr"$p={p:g}$")
        if np.all(np.isfinite(e)):
            ax.fill_between(
                N_values,
                np.maximum(y - e, np.finfo(float).tiny),
                np.maximum(y + e, np.finfo(float).tiny),
                color=color,
                alpha=0.12,
                linewidth=0.0,
            )


    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.16)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if label in ['a', 'b', 'c']:
        ax.text(
            0.018, 0.55, f"({label})",
            transform=ax.transAxes,
            ha="left", va="top",
        )
        ax.legend(
            loc="lower left",
            frameon=True,
            handlelength=1.5,
            borderpad=0.25,
            labelspacing=0.15,
            handletextpad=0.3,
            fontsize=8,
        )
    else:
        ax.text(
            0.018, 0.975, f"({label})",
            transform=ax.transAxes,
            ha="left", va="top",
        )
        ax.legend(
            loc="lower right",
            frameon=True,
            handlelength=1.5,
            borderpad=0.25,
            labelspacing=0.15,
            handletextpad=0.3,
            fontsize=8,
        )




def _draw_panel_inset(
    ax_parent,
    inset_N: np.ndarray,
    inset_mean: np.ndarray,
    inset_sem: np.ndarray,
    p_group: tuple[float, ...],
    column_offset: int,
    ylim: Tuple[float, float],
    draw_eta_crit_guide: bool = False,
) -> None:
    """
    Draw a zoomed inset inside ``ax_parent``.


    ``column_offset`` selects which block of columns of ``inset_mean``
    corresponds to ``p_group`` (0 for panel (a), 4 for panel (b), ...).
    """
    ax_inset = ax_parent.inset_axes([0.45, 0.15, 0.5, 0.5])


    for ip, p in enumerate(p_group):
        color = P_COLORS[float(p)]
        y = inset_mean[:, column_offset + ip]
        e = inset_sem[:, column_offset + ip]
        ax_inset.plot(
            inset_N,
            y,
            color=color,
            linewidth=1.0,
        )
        if np.all(np.isfinite(e)):
            ax_inset.fill_between(
                inset_N,
                np.maximum(y - e, np.finfo(float).tiny),
                np.maximum(y + e, np.finfo(float).tiny),
                color=color,
                alpha=0.12,
                linewidth=0.0,
            )


    ax_inset.set_xlim(float(inset_N[0]), float(inset_N[-1]))
    ax_inset.set_xscale('log')
    ax_inset.set_ylim(*ylim)
    ax_inset.grid(True, which="both", alpha=0.22)
    ax_inset.tick_params(
        which="both",
        direction="in",
        top=True,
        right=True,
        labelsize=7,
    )
    # ax_inset.set_xlabel(r"$N$", fontsize=7, labelpad=1.0)
    # ax_inset.set_ylabel(r"$\eta_N(p)$", fontsize=7, labelpad=1.0)
    for spine in ax_inset.spines.values():
        spine.set_linewidth(0.8)


    if draw_eta_crit_guide:
        ax_inset.axhline(ETA_CRIT, color="k", ls=":", lw=0.8)
    else:
        ax_inset.set_yscale('log')




def plot_six_panel(
    N_values: np.ndarray,
    part_mean: np.ndarray,
    part_sem: np.ndarray,
    bound_mean: np.ndarray,
    bound_sem: np.ndarray,
    inset_N: np.ndarray,
    inset_mean: np.ndarray,
    inset_sem: np.ndarray,
    path: Path,
) -> None:
    """
    Six log-log panels plus the requested upper-left / upper-middle insets.


    Panel (a) inset shows the first p-group; panel (b) inset shows the second
    p-group.  Both use the SAME ``inset_mean`` / ``inset_sem`` arrays, just
    different column blocks (0..3 vs 4..7).
    """
    fig, axes = plt.subplots(2, 3, figsize=(9, 3), sharex=True, squeeze=True)
    axes[0, 0].set_xlim(2, MAIN_N_MAX)
    fig.subplots_adjust(
        left=0.075,
        right=0.995,
        top=0.985,
        bottom=0.095,
        wspace=0.0,
        hspace=0.0,
    )


    top_labels = ("a", "b", "c")
    bottom_labels = ("d", "e", "f")


    for j, group in enumerate(P_GROUPS):
        add_group(
            axes[0, j],
            N_values,
            group,
            part_mean,
            part_sem,
            r"$\eta_N(p)$" if j == 0 else None,
            top_labels[j],
        )
        add_group(
            axes[1, j],
            N_values,
            group,
            bound_mean,
            bound_sem,
            r"$\widetilde{\mathcal{U}}_p(N)$" if j == 0 else None,
            bottom_labels[j],
        )


    # ------------------------------------------------------------------
    # Requested black dashed references.
    # ------------------------------------------------------------------
    N_red = N_values[11:].astype(float)
    if N_red.size:
        axes[1, 0].plot(N_red, 5.0 * N_red**2, "k--", lw=1.0)
        axes[1, 0].text(1e2, 5e5, r"$\sim N^2$", fontsize=8)


        axes[1, 0].plot(
            N_red,
            1.5 * N_red**2 / np.sqrt(np.log(N_red)),
            "k--", lw=1.0,
        )
        axes[1, 0].text(4e2, 2e4, r"$\sim N^2/\sqrt{\log N}$", fontsize=8)


        axes[1, 1].plot(
            N_red,
            10 * N_red ** (2.5 - P_GROUPS[1][0]),
            "k--", lw=1.0,
        )
        axes[1, 1].text(1e2, 1e5, r"$\sim N^{3/2}$", fontsize=8)


        axes[1, 1].plot(
            N_red,
            3 * N_red * np.log(N_red),
            "k--", lw=1.0,
        )
        axes[1, 1].text(1e2, 2e2, r"$\sim N\log N$", fontsize=8)


        N_red = N_values[101:].astype(float)


        axes[0, 2].plot(
            N_red,
            1.5 * np.log(N_red) / N_red,
            "k--", lw=1.0,
        )
        axes[0, 2].text(1e3, 2e-2, r"$\sim N^{-1}\log N$", fontsize=8)


        axes[0, 2].plot(
            N_red,
            1.5 / N_red**2,
            "k--", lw=1.0,
        )
        axes[0, 2].text(100, 3e-6, r"$\sim N^{-2}$", fontsize=8)


        axes[1, 2].plot(
            N_red,
            N_red,
            "k--", lw=1.0,
        )
        axes[1, 2].text(3e2, 50, r"$\sim N$", fontsize=8)


        # Fixed y-limits requested by user.
        for j in range(3):
            axes[0, j].set_ylim(5e-7, 1.0)   # upper row
            axes[1, j].set_ylim(1.0, 5e8)    # lower row


    # X labels/tick labels only on bottom panels.
    for j in range(3):
        axes[0, j].tick_params(
            axis="x", which="both", labelbottom=False,
        )
        axes[1, j].set_xlabel(r"$N$")


    # Y labels/tick labels only on leftmost panels.
    for row in range(2):
        axes[row, 0].tick_params(
            axis="y", which="both", labelleft=True,
        )
        for j in (1, 2):
            axes[row, j].tick_params(
                axis="y", which="both", labelleft=False,
            )


    for row in range(2):
        for j in range(3):
            axes[row, j].tick_params(
                which="both",
                direction="in",
                top=True,
                right=True,
            )


    # ------------------------------------------------------------------
    # Upper-row insets.
    #
    #   Panel (a) inset -> P_GROUPS[0]  (columns 0..3 of inset_mean)
    #   Panel (b) inset -> P_GROUPS[1]  (columns 4..7 of inset_mean)
    # ------------------------------------------------------------------
    if inset_N.size:
        # Panel (a) inset: first p-group.
        _draw_panel_inset(
            axes[0, 0],
            inset_N,
            inset_mean,
            inset_sem,
            P_GROUPS[0],
            column_offset=0,
            ylim=(0.5, 0.7),
            draw_eta_crit_guide=True,
        )


        # Panel (b) inset: second p-group.  Requires inset_mean/inset_sem to
        # have at least 8 columns (INSET_P_COUNT >= 8).
        if inset_mean.shape[1] >= 8:
            _draw_panel_inset(
                axes[0, 1],
                inset_N,
                inset_mean,
                inset_sem,
                P_GROUPS[1],
                column_offset=4,
                ylim=(6e-3, 0.6),
                draw_eta_crit_guide=False,
            )


        # Shade the parent-panel regions that motivate the insets.
        if N_values[-1] >= 50:
            axes[0, 0].axvspan(
                50.0,
                float(N_values[-1]),
                color="0.7",
                alpha=0.08,
                linewidth=0.0,
                zorder=0,
            )
            axes[0, 1].axvspan(
                50.0,
                float(N_values[-1]),
                color="0.7",
                alpha=0.08,
                linewidth=0.0,
                zorder=0,
            )


    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
        # pad_inches=0.03,
    )
    plt.close(fig)




# =============================================================================
# Self-tests
# =============================================================================




def self_check() -> None:
    """Basic exact checks, including the supplied eta_crit integral."""
    val = eta_crit_quadrature()
    if not math.isclose(val, ETA_CRIT, rel_tol=2e-10, abs_tol=2e-10):
        raise AssertionError(f"eta_crit mismatch: {val} vs {ETA_CRIT}")


    for N in (2, 3, 16, 100):
        direct = (
            2.0 * N
            + 6.0 * N * (N - 1)
            + (4.0 / 3.0) * N * (N - 1) * (N - 2)
        )
        formula = float(total_coefficient_count(N))
        if not math.isclose(
            direct, formula, rel_tol=1e-12, abs_tol=1e-12,
        ):
            raise AssertionError(
                f"Coefficient count mismatch at N={N}: "
                f"{formula} vs {direct}"
            )


    sig = normalization_sigma_table(1000, np.asarray([0.0, 4.0]))
    expected_p0 = math.sqrt(2.0 / 1000.0)
    if not math.isclose(
        sig[-1, 0], expected_p0, rel_tol=1e-12,
    ):
        raise AssertionError("p=0 normalization check failed.")


    print(f"eta_crit quadrature = {val:.12f}")
    print("Coefficient-count and normalization checks passed.")




# =============================================================================
# CLI
# =============================================================================




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--main-n-max", type=int, default=MAIN_N_MAX,
    )
    parser.add_argument(
        "--inset-n-min", type=int, default=INSET_N_MIN,
    )
    parser.add_argument(
        "--inset-n-max", type=int, default=INSET_N_MAX,
    )
    parser.add_argument(
        "--n-realizations", type=int, default=N_REALIZATIONS,
    )
    parser.add_argument(
        "--n-jobs", type=int, default=N_JOBS,
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
    )
    parser.add_argument(
        "--J", type=float, default=J_DEFAULT,
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("powerlaw_mc_results_corrected"),
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()




def main() -> None:
    args = parse_args()


    if args.main_n_max < 2:
        raise ValueError("--main-n-max must be at least 2")
    if args.inset_n_max > 0 and args.inset_n_min < 2:
        raise ValueError("--inset-n-min must be at least 2")
    if args.inset_n_max > 0 and args.inset_n_max < args.inset_n_min:
        raise ValueError("--inset-n-max must be >= --inset-n-min")
    if args.n_realizations < 1:
        raise ValueError("--n-realizations must be >= 1")
    if args.n_jobs < 1:
        raise ValueError("--n-jobs must be >= 1")
    if args.J <= 0:
        raise ValueError("--J must be positive")


    if args.self_test:
        self_check()


    inset_enabled = args.inset_n_max > 0
    realization_n_max = max(
        args.main_n_max,
        args.inset_n_max if inset_enabled else args.main_n_max,
    )


    main_N = np.arange(2, args.main_n_max + 1, dtype=np.int64)
    if inset_enabled:
        inset_N = np.arange(
            args.inset_n_min, args.inset_n_max + 1, dtype=np.int64,
        )
    else:
        inset_N = np.empty(0, dtype=np.int64)


    # Bound normalization only needs the main range.
    sigma_table = normalization_sigma_table(
        args.main_n_max, P_VALUES, J=args.J,
    )


    print(
        "Corrected direct Monte-Carlo participation + tight "
        "norm-bound calculation"
    )
    print("=" * 78)
    print(f"Main N range: 2..{args.main_n_max}")
    if inset_enabled:
        print(
            f"Inset N range: {args.inset_n_min}..{args.inset_n_max}"
        )
    else:
        print("Inset: disabled")
    print(
        f"p values: {tuple(float(p) for p in P_VALUES)}"
    )
    print(f"realizations: {args.n_realizations}")
    print(
        f"joblib workers: "
        f"{min(args.n_jobs, args.n_realizations)}"
    )
    print(f"J: {args.J:g}")
    print("Participation definition: sum_k N_eff^(k) / N_max")
    print(
        f"Target p=0 thermodynamic limit: "
        f"eta_crit = {ETA_CRIT:.12f}"
    )


    seeds = [
        args.seed + 104729 * r
        for r in range(args.n_realizations)
    ]


    results = Parallel(
        n_jobs=min(args.n_jobs, args.n_realizations),
    )(
        delayed(simulate_one_realization)(
            r,
            realization_n_max,
            P_VALUES,
            sigma_table,
            args.main_n_max,
            args.inset_n_min if inset_enabled else 0,
            args.inset_n_max if inset_enabled else 0,
            seeds[r],
        )
        for r in tqdm(range(args.n_realizations))
    )


    main_part_stack = np.stack([r[0] for r in results], axis=0)
    if inset_enabled:
        inset_part_stack = np.stack([r[1] for r in results], axis=0)
    else:
        inset_part_stack = np.empty((0, 0, 0))
    bound_stack = np.stack([r[2] for r in results], axis=0)


    part_mean, part_sem = mean_and_sem(main_part_stack)
    bound_mean, bound_sem = mean_and_sem(bound_stack)


    if inset_enabled:
        inset_mean, inset_sem = mean_and_sem(inset_part_stack)
    else:
        inset_mean = np.empty((0, INSET_P_COUNT), dtype=np.float64)
        inset_sem = np.empty((0, INSET_P_COUNT), dtype=np.float64)


    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)


    main_csv = outdir / "powerlaw_participation_and_bounds_main.csv"
    inset_csv = outdir / "powerlaw_participation_inset.csv"
    bound_csv = outdir / "powerlaw_bounds_main.csv"
    npz_file = (
        outdir / "powerlaw_participation_and_bounds_corrected.npz"
    )
    figure_file = (
        outdir
        / "participation_plaw.pdf"
    )


    save_csv(
        main_csv, main_N, P_VALUES,
        part_mean, part_sem, "eta",
    )
    save_csv(
        bound_csv, main_N, P_VALUES,
        bound_mean, bound_sem, "bound",
    )


    if inset_enabled:
        save_csv(
            inset_csv,
            inset_N,
            P_VALUES[:INSET_P_COUNT],
            inset_mean,
            inset_sem,
            "eta",
        )


    np.savez_compressed(
        npz_file,
        main_N=main_N,
        p=P_VALUES,
        participation_mean=part_mean,
        participation_sem=part_sem,
        bound_mean=bound_mean,
        bound_sem=bound_sem,
        inset_N=inset_N,
        inset_p=P_VALUES[:INSET_P_COUNT],
        inset_participation_mean=inset_mean,
        inset_participation_sem=inset_sem,
        eta_crit=ETA_CRIT,
    )


    plot_six_panel(
        main_N,
        part_mean,
        part_sem,
        bound_mean,
        bound_sem,
        inset_N,
        inset_mean,
        inset_sem,
        figure_file,
    )


    # Explicitly report the p=0 validation point in the main range and the
    # last inset point when available.
    p0_idx = int(np.where(np.isclose(P_VALUES, 0.0))[0][0])
    print("\nValidation")
    print("=" * 78)
    print(
        f"main eta(N={main_N[-1]}, p=0) = "
        f"{part_mean[-1, p0_idx]:.8f} "
        f"+/- {part_sem[-1, p0_idx]:.8f}"
    )
    if inset_enabled:
        print(
            f"inset eta(N={inset_N[-1]}, p=0) = "
            f"{inset_mean[-1, 0]:.8f} "
            f"+/- {inset_sem[-1, 0]:.8f}"
        )
        print(
            f"inset eta(N={inset_N[-1]}, p=1) = "
            f"{inset_mean[-1, 4]:.8f} "
            f"+/- {inset_sem[-1, 4]:.8f}"
        )
    print(f"eta_crit = {ETA_CRIT:.8f}")


    print("\nOutputs")
    print("=" * 78)
    print(main_csv.resolve())
    print(bound_csv.resolve())
    if inset_enabled:
        print(inset_csv.resolve())
    print(npz_file.resolve())
    print(figure_file.resolve())
    print("Done.")




if __name__ == "__main__":
    main()
