from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Sequence, Tuple

# # Avoid BLAS oversubscription inside joblib workers.
# os.environ.setdefault("OMP_NUM_THREADS", "1")
# os.environ.setdefault("MKL_NUM_THREADS", "1")
# os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
# os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib.collections as mcoll
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from numba import njit, set_num_threads
from scipy.optimize import curve_fit
from matplotlib.colors import LinearSegmentedColormap, PowerNorm, to_rgb

plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "text.usetex": True,
    "font.family": "serif",
    "pgf.preamble": "\n".join([r"\usepackage{amsmath}"]),
    "text.latex.preamble": "\n".join([r"\usepackage{amsmath}"]),
})


# =============================================================================
# User settings
# =============================================================================

N_MIN = 2
N_MAX = 10_000
N_REALIZATIONS = 100
N_JOBS = min(N_REALIZATIONS, os.cpu_count() or 1)
SEED = 13
J = 1.0

N_L_VALUES = np.asarray(
    [
        2,
        5,
        8,
        10,
        20,
        50,
        80,
        100,
        200,
        500,
        800,
        1000,
        2000,
        5000,
        8000,
        10000,
    ],
    dtype=np.int64,
)

ETA_CRIT = 0.6529308944619922

BASE_COLORS = [
    "seagreen",
    "chartreuse",
    "mediumturquoise",
][::-1]


# =============================================================================
# Plot colour interpolation
# =============================================================================

def interpolate_colors(
    base_colors: Sequence[str],
    n_output: int,
):
    rgb_base = np.asarray(
        [to_rgb(c) for c in base_colors],
        dtype=np.float64,
    )

    if n_output < 1:
        raise ValueError("n_output must be >= 1")

    if len(base_colors) == 1:
        rgb_interp = np.tile(
            rgb_base,
            (n_output, 1),
        )
    else:
        x_base = np.linspace(0.0, 1.0, len(base_colors))
        x_interp = np.linspace(0.0, 1.0, n_output)
        rgb_interp = np.zeros((n_output, 3), dtype=np.float64)

        for channel in range(3):
            rgb_interp[:, channel] = np.interp(
                x_interp,
                x_base,
                rgb_base[:, channel],
            )

    return [tuple(np.clip(c, 0.0, 1.0)) for c in rgb_interp]


COLOURS = interpolate_colors(BASE_COLORS, len(N_L_VALUES))


# =============================================================================
# Exact GLOBAL coefficient count used in eta_N normalization
# =============================================================================

@njit(cache=True, inline="always")
def global_coefficient_count(N: int) -> float:
    return 2.0 * N * (N + 2.0) * (2.0 * N - 1.0) / 3.0


# =============================================================================
# Exact finite-N charger normalization for a hard cutoff
# =============================================================================

@njit(cache=True, inline="always")
def charger_sigma_cutoff(N: int, nL: int, J: float) -> float:
    """sigma^2 = 2 J^2 N / [N + (q-1)(2N-q)], q=min(nL,N)."""
    q = nL
    if q > N:
        q = N
    denom = N + (q - 1) * (2 * N - q)
    return J * math.sqrt(2.0 * N / denom)


# =============================================================================
# Small numerical helpers
# =============================================================================

@njit(cache=True, inline="always")
def xlogx(x: float) -> float:
    if x <= 0.0:
        return 0.0
    return x * math.log(x)


@njit(cache=True, inline="always")
def pair_six_coefficients(
    beta_c: float,
    gamma_c: float,
    delta_c: float,
    delta_n: float,
    q0: float,
    q1: float,
    q2: float,
    q3: float,
    central_is_left: bool,
) -> Tuple[float, float, float, float, float, float]:
    """Six coefficient magnitudes for one central-cell/neighbour pair."""
    rho = math.sqrt(beta_c * beta_c + gamma_c * gamma_c)

    if rho > 0.0:
        c = beta_c / rho
        s = gamma_c / rho
    else:
        c = 1.0
        s = 0.0

    if central_is_left:
        xp = -s * q0 + c * q1
        yp = c * q0 + s * q1
        xm = c * q2 - s * q3
        ym = s * q2 + c * q3

        p1 = abs(-delta_n * yp + delta_c * ym)
        p2 = abs(-delta_n * xm + delta_c * xp)
        p3 = abs(delta_n * xp - delta_c * xm)
        p4 = abs(-delta_n * ym + delta_c * yp)
    else:
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
# Combined participation + norm kernel
# =============================================================================

@njit(cache=True, fastmath=False)
def simulate_nL_stream(
    beta: np.ndarray,
    gamma: np.ndarray,
    delta: np.ndarray,
    diag_z: np.ndarray,
    nL_values: np.ndarray,
    J: float,
    pair_seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """One realization. Returns participation and the combined norm bound."""
    M = beta.size
    Q = nL_values.size

    participation = np.empty((M - 1, Q), dtype=np.float64)
    bounds = np.empty((M - 1, Q), dtype=np.float64)

    # Per-central-site local S/R terms.
    local_S = np.zeros(M, dtype=np.float64)
    local_R = np.zeros(M, dtype=np.float64)

    for k in range(M):
        x1 = abs(diag_z[k] * beta[k])
        x2 = abs(diag_z[k] * gamma[k])
        local_S[k] = x1 + x2
        local_R[k] = xlogx(x1) + xlogx(x2)

    # Per-central-site neighbour-pair S/R accumulators.
    pair_S = np.zeros((Q, M), dtype=np.float64)
    pair_R = np.zeros((Q, M), dtype=np.float64)

    # C2 participation interval-difference arrays (unchanged).
    c2_diff_W = np.zeros((Q, M + 1), dtype=np.float64)
    c2_diff_U = np.zeros((Q, M + 1), dtype=np.float64)

    # Prefix sums for the battery factors used in C2.
    pref_b2 = np.zeros(M + 1, dtype=np.float64)
    pref_g2 = np.zeros(M + 1, dtype=np.float64)
    for i in range(M):
        pref_b2[i + 1] = pref_b2[i] + beta[i] * beta[i]
        pref_g2[i + 1] = pref_g2[i] + gamma[i] * gamma[i]

    # Norm-bound block accumulators.
    left_c3_sq = np.zeros((Q, M), dtype=np.float64)
    x_c4_sq = np.zeros((Q, M), dtype=np.float64)
    y_c4_sq = np.zeros((Q, M), dtype=np.float64)

    right_c3_sum = np.zeros(Q, dtype=np.float64)
    left_c3_sum = np.zeros(Q, dtype=np.float64)
    x_c4_sum = np.zeros(Q, dtype=np.float64)
    y_c4_sum = np.zeros(Q, dtype=np.float64)

    # Refined C2 coupling factor W_c for each cutoff:
    # sum_{k<m} (m-k-1)(|A_km|^2+|B_km|^2).
    c2_coupling_weight = np.zeros(Q, dtype=np.float64)

    # Unscaled C1 contribution.
    c1_raw = abs(diag_z[0]) * math.sqrt(beta[0] * beta[0] + gamma[0] * gamma[0])

    # Deterministic Numba RNG stream.
    np.random.seed(pair_seed)

    for m in range(1, M):
        N = m + 1
        c1_raw += abs(diag_z[m]) * math.sqrt(beta[m] * beta[m] + gamma[m] * gamma[m])

        # Right-endpoint C3 blocks for the newly introduced site m.
        new_right_c3_sq = np.zeros(Q, dtype=np.float64)

        # Add the pair (k,m), for every k<m.
        for k in range(m):
            q0 = np.random.normal()
            q1 = np.random.normal()
            q2 = np.random.normal()
            q3 = np.random.normal()

            L = m - k + 1
            q_start = 0
            while q_start < Q and nL_values[q_start] < L:
                q_start += 1
            if q_start >= Q:
                continue

            # Participation pair-sector coefficients (unchanged).
            vals_k = pair_six_coefficients(
                beta[k], gamma[k], delta[k], delta[m],
                q0, q1, q2, q3, True,
            )
            vals_m = pair_six_coefficients(
                beta[m], gamma[m], delta[m], delta[k],
                q0, q1, q2, q3, False,
            )

            s6_k = 0.0
            r6_k = 0.0
            s6_m = 0.0
            r6_m = 0.0
            for z in vals_k:
                s6_k += z
                r6_k += xlogx(z)
            for z in vals_m:
                s6_m += z
                r6_m += xlogx(z)

            # C2 participation coefficients (unchanged).
            g0 = abs(q0)
            g1 = abs(q1)
            g2 = abs(q2)
            g3 = abs(q3)
            G = g0 + g1 + g2 + g3
            Glog = xlogx(g0) + xlogx(g1) + xlogx(g2) + xlogx(g3)

            # Norm-bound pair coefficients (unchanged C3/C4 definitions).
            k31 = beta[k] * q2 - gamma[m] * q0
            k32 = beta[k] * q1 + gamma[m] * q3
            k33 = -beta[m] * q3 - gamma[k] * q0
            k34 = beta[m] * q1 - gamma[k] * q2

            k41 = -delta[k] * q2 + delta[m] * q3
            k42 = -delta[k] * q1 + delta[m] * q0
            k43 = delta[k] * q0 - delta[m] * q1
            k44 = -delta[k] * q3 + delta[m] * q2

            # ----------------------------------------------------------
            # REFINED C2 COUPLING FACTOR for the pair (k,m).
            #
            # q0=Re(A)+Re(B), q1=Re(A)-Re(B),
            # q2=Im(A)+Im(B), q3=Im(A)-Im(B), so
            #
            # |A|^2+|B|^2 = 1/2 * sum_j q_j^2.
            #
            # The multiplicity of the pair in sum_l ||M^(l)||_F^2 is
            # m-k-1, i.e. the number of intermediate battery sites.
            # ----------------------------------------------------------
            distance_weight = m - k - 1
            pair_coupling_weight = 0.5 * distance_weight * (
                q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3
            )

            # Activate all cutoffs nL >= L.
            for iq in range(q_start, Q):
                # Participation (unchanged).
                pair_S[iq, k] += s6_k
                pair_R[iq, k] += r6_k
                pair_S[iq, m] += s6_m
                pair_R[iq, m] += r6_m

                lo = k + 1
                hi = m
                c2_diff_W[iq, lo] += G
                c2_diff_W[iq, hi] -= G
                c2_diff_U[iq, lo] += Glog
                c2_diff_U[iq, hi] -= Glog

                # Refined C2 upper bound: accumulate W_c here.
                c2_coupling_weight[iq] += pair_coupling_weight

                # Left endpoint C3 block (unchanged).
                old_l = left_c3_sq[iq, k]
                new_l = old_l + k31 * k31 + k32 * k32
                left_c3_sq[iq, k] = new_l
                left_c3_sum[iq] += math.sqrt(new_l) - math.sqrt(old_l)

                # Right endpoint C3 block (unchanged).
                new_right_c3_sq[iq] += k33 * k33 + k34 * k34

                # C4 X block (unchanged).
                old_x = x_c4_sq[iq, k]
                new_x = old_x + k41 * k41 + k42 * k42
                x_c4_sq[iq, k] = new_x
                x_c4_sum[iq] += math.sqrt(new_x) - math.sqrt(old_x)

                # C4 Y block (unchanged).
                old_y = y_c4_sq[iq, k]
                new_y = old_y + k43 * k43 + k44 * k44
                y_c4_sq[iq, k] = new_y
                y_c4_sum[iq] += math.sqrt(new_y) - math.sqrt(old_y)

        # Finalize the right C3 block of the newly introduced site.
        for iq in range(Q):
            right_c3_sum[iq] += math.sqrt(new_right_c3_sq[iq])

        # Evaluate eta_N(nL) for the CURRENT N.
        c2_run_W = np.zeros(Q, dtype=np.float64)
        c2_run_U = np.zeros(Q, dtype=np.float64)
        eta_sum = np.zeros(Q, dtype=np.float64)

        # Refined C2 battery factor:
        # W_b = sum_l r_l (beta_l^2+gamma_l^2),
        # where in zero-based indexing
        # r_l = min(nL-2, l, N-l-1).
        battery_rank_weight = np.zeros(Q, dtype=np.float64)

        for k in range(N):
            for iq in range(Q):
                c2_run_W[iq] += c2_diff_W[iq, k]
                c2_run_U[iq] += c2_diff_U[iq, k]

                q_rank = int(nL_values[iq]) - 2
                if q_rank < 0:
                    q_rank = 0
                r_l = min(q_rank, k, N - k - 1)
                battery_rank_weight[iq] += r_l * (
                    beta[k] * beta[k] + gamma[k] * gamma[k]
                )

                # Participation-ratio calculation (unchanged).
                b = abs(beta[k])
                g = abs(gamma[k])
                Sc2 = (b + g) * c2_run_W[iq]
                Rc2 = (xlogx(b) + xlogx(g)) * c2_run_W[iq]
                Rc2 += (b + g) * c2_run_U[iq]

                S = local_S[k] + pair_S[iq, k] + Sc2
                R = local_R[k] + pair_R[iq, k] + Rc2

                if S > 0.0 and math.isfinite(S) and math.isfinite(R):
                    eta_sum[iq] += S * math.exp(-R / S)

        denominator = global_coefficient_count(N)

        for iq in range(Q):
            participation[N - 2, iq] = eta_sum[iq] / denominator

            sigma = charger_sigma_cutoff(N, int(nL_values[iq]), J)

            # Tight parameter-resolved C2 bound:
            #
            # ||C2|| <= 2 sqrt(2 W_b W_c).
            #
            # W_b is the rank-weighted battery sum above, and W_c is the
            # distance-weighted coupling sum accumulated for each cutoff.
            c2_refined = 2.0 * math.sqrt(
                2.0 * battery_rank_weight[iq] * c2_coupling_weight[iq]
            )

            raw_bound = (
                c1_raw
                + c2_refined
                + left_c3_sum[iq]
                + right_c3_sum[iq]
                + x_c4_sum[iq]
                + y_c4_sum[iq]
            )
            bounds[N - 2, iq] = sigma * raw_bound

    return participation, bounds


# =============================================================================
# All-cutoff norm-bound kernel for dense exponent extraction
# =============================================================================

@njit(cache=True, fastmath=True)
def _fill_family_root_from_stream(
    beta: np.ndarray,
    gamma: np.ndarray,
    delta: np.ndarray,
    M: int,
    pair_seed: int,
    family: int,
) -> np.ndarray:
    """Build cumulative square-root block data for one C3/C4 family."""
    root = np.zeros((M, M - 1), dtype=np.float64)
    np.random.seed(pair_seed)

    for m in range(1, M):
        for k in range(m):
            q0 = np.random.normal()
            q1 = np.random.normal()
            q2 = np.random.normal()
            q3 = np.random.normal()

            k31 = beta[k] * q2 - gamma[m] * q0
            k32 = beta[k] * q1 + gamma[m] * q3
            k33 = -beta[m] * q3 - gamma[k] * q0
            k34 = beta[m] * q1 - gamma[k] * q2
            k41 = -delta[k] * q2 + delta[m] * q3
            k42 = -delta[k] * q1 + delta[m] * q0
            k43 = delta[k] * q0 - delta[m] * q1
            k44 = -delta[k] * q3 + delta[m] * q2

            d = m - k - 1
            if family == 0:
                z = k31 * k31 + k32 * k32
                endpoint = k
            elif family == 1:
                z = k33 * k33 + k34 * k34
                endpoint = m
            elif family == 2:
                z = k41 * k41 + k42 * k42
                endpoint = k
            else:
                z = k43 * k43 + k44 * k44
                endpoint = k
            root[endpoint, d] = z

    for k in range(M):
        acc = 0.0
        for d in range(M - 1):
            acc += root[k, d]
            root[k, d] = math.sqrt(acc)

    return root


@njit(cache=True, fastmath=True)
def _all_cutoff_family_contribution(root: np.ndarray, M: int) -> np.ndarray:
    """Dense (N,nL) sum for left-endpoint anticommuting block families."""
    out = np.zeros((M - 1, M - 1), dtype=np.float64)
    first = np.zeros(M, dtype=np.float64)

    for N in range(2, M + 1):
        for c in range(2, N + 1):
            k = N - c
            if k >= 0:
                first[c - 2] += root[k, c - 2]

        anti = 0.0
        for c in range(2, N + 1):
            if c >= 3:
                k = N - c + 1
                d = c - 3
                if k >= 0 and d < M - 1:
                    anti += root[k, d]
            out[N - 2, c - 2] = first[c - 2] + anti

    return out


@njit(cache=True, fastmath=True)
def _all_cutoff_right_endpoint_family(root: np.ndarray, M: int) -> np.ndarray:
    """Dense (N,nL) sum for a right-endpoint/left-neighbour block family."""
    out = np.zeros((M - 1, M - 1), dtype=np.float64)
    running = np.zeros(M - 1, dtype=np.float64)

    for N in range(2, M + 1):
        endpoint = N - 1
        for d in range(M - 1):
            if d < endpoint:
                running[d] += root[endpoint, d]
            else:
                running[d] += root[endpoint, endpoint - 1]

        for d in range(N - 1):
            out[N - 2, d] = running[d]

    return out


@njit(cache=True, fastmath=True)
def _all_cutoff_c2_raw(
    beta: np.ndarray,
    gamma: np.ndarray,
    M: int,
    pair_seed: int,
) -> np.ndarray:
    """
    Dense refined C2 bound for every (N,nL), 2<=nL<=N.

    For each N and cutoff nL,
        C2_raw = 2 sqrt(2 W_b W_c),

    where

        W_b = sum_l min(nL-2,l,N-l-1)(beta_l^2+gamma_l^2),

        W_c = sum_{k<m, m-k+1<=nL}
              (m-k-1)(|A_km|^2+|B_km|^2).

    With q0=Re(A)+Re(B), q1=Re(A)-Re(B), q2=Im(A)+Im(B),
    q3=Im(A)-Im(B), we have |A|^2+|B|^2 = 0.5*sum_j q_j^2.

    The return value is unscaled; sigma is applied by the caller.
    """
    out = np.zeros((M - 1, M - 1), dtype=np.float64)

    # Prefix sums of w_l = beta_l^2 + gamma_l^2.
    pref_w = np.zeros(M + 1, dtype=np.float64)
    for i in range(M):
        pref_w[i + 1] = pref_w[i] + beta[i] * beta[i] + gamma[i] * gamma[i]

    # running[d] is the total coupling weight for all pairs whose distance
    # index is d=m-k-1. W_c is a prefix sum of these distance weights.
    row = np.zeros(M - 1, dtype=np.float64)
    running = np.zeros(M - 1, dtype=np.float64)

    # The same pair sequence is generated as in simulate_nL_stream.
    np.random.seed(pair_seed)

    for m in range(1, M):
        for d in range(M - 1):
            row[d] = 0.0

        for k in range(m):
            q0 = np.random.normal()
            q1 = np.random.normal()
            q2 = np.random.normal()
            q3 = np.random.normal()

            d = m - k - 1

            pair_energy = 0.5 * (
                q0 * q0
                + q1 * q1
                + q2 * q2
                + q3 * q3
            )

            row[d] += d * pair_energy

        # Accumulate all pairs that have become available by endpoint m.
        for d in range(m):
            running[d] += row[d]

        N = m + 1
        battery_weight = 0.0
        coupling_weight = 0.0

        # Column d represents locality cutoff nL=d+2. Increasing d by one
        # adds the battery sites where the rank weight increases by one.
        # This computes W_b exactly while coupling_weight is the prefix
        # sum W_c over all pair-distance indices <= d.
        for d in range(N - 1):
            if d > 0:
                upper = N - d
                if upper > d:
                    battery_weight += pref_w[upper] - pref_w[d]

            coupling_weight += running[d]

            out[N - 2, d] = 2.0 * math.sqrt(
                2.0 * battery_weight * coupling_weight
            )

    return out


@njit(cache=True, fastmath=True)
def simulate_all_cutoff_bounds(
    beta: np.ndarray,
    gamma: np.ndarray,
    delta: np.ndarray,
    diag_z: np.ndarray,
    J: float,
    pair_seed: int,
) -> np.ndarray:
    """Realization-level norm bound for all hard cutoffs nL=2,...,M."""
    M = beta.size

    # C1 contribution as a cumulative function of N.
    c1 = np.zeros(M - 1, dtype=np.float64)
    acc = 0.0
    for N in range(1, M + 1):
        i = N - 1
        acc += abs(diag_z[i]) * math.sqrt(beta[i]*beta[i] + gamma[i]*gamma[i])
        if N >= 2:
            c1[N - 2] = acc

    # Refined C2 contribution for all cutoffs.
    c2 = _all_cutoff_c2_raw(beta, gamma, M, pair_seed)

    # C3-left, C3-right, C4-X, C4-Y (unchanged).
    left_c3 = _all_cutoff_family_contribution(
        _fill_family_root_from_stream(beta, gamma, delta, M, pair_seed, 0), M
    )
    right_c3 = _all_cutoff_right_endpoint_family(
        _fill_family_root_from_stream(beta, gamma, delta, M, pair_seed, 1), M
    )
    x_c4 = _all_cutoff_family_contribution(
        _fill_family_root_from_stream(beta, gamma, delta, M, pair_seed, 2), M
    )
    y_c4 = _all_cutoff_family_contribution(
        _fill_family_root_from_stream(beta, gamma, delta, M, pair_seed, 3), M
    )

    raw = np.zeros((M - 1, M - 1), dtype=np.float64)
    for N in range(2, M + 1):
        row = N - 2
        for c in range(2, N + 1):
            col = c - 2
            raw[row, col] = (
                c1[row]
                + c2[row, col]
                + left_c3[row, col]
                + right_c3[row, col]
                + x_c4[row, col]
                + y_c4[row, col]
            )

    bounds = np.zeros_like(raw)
    for N in range(2, M + 1):
        row = N - 2
        for c in range(2, N + 1):
            q = c
            denom = N + (q - 1) * (2 * N - q)
            sigma = J * math.sqrt(2.0 * N / denom)
            bounds[row, c - 2] = sigma * raw[row, c - 2]

        # nL>=N is the same saturated global model for this system size.
        global_val = bounds[row, N - 2]
        for c in range(N + 1, M + 1):
            bounds[row, c - 2] = global_val

    return bounds


# =============================================================================
# Independent realization wrapper
# =============================================================================

def one_realization(
    realization_id: int,
    n_max: int,
    nL_values: np.ndarray,
    seed: int,
    J: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate one independent disorder realization and evaluate both observables."""
    set_num_threads(1)

    ss = np.random.SeedSequence([int(seed), int(realization_id)])
    rng = np.random.default_rng(ss)

    beta = np.ascontiguousarray(rng.standard_normal(n_max))
    gamma = np.ascontiguousarray(rng.standard_normal(n_max))
    delta = np.ascontiguousarray(rng.standard_normal(n_max))
    diag_z = np.ascontiguousarray(rng.standard_normal(n_max))

    # Identical deterministic charger-pair stream for both observables.
    numba_seed = int((seed + 104729 * (realization_id + 1)) % 2_000_000_000)

    participation, bounds = simulate_nL_stream(
        beta, gamma, delta, diag_z, nL_values, J, numba_seed
    )

    all_cutoff_bounds = simulate_all_cutoff_bounds(
        beta, gamma, delta, diag_z, J, numba_seed
    )

    if not np.all(np.isfinite(participation)):
        raise FloatingPointError(
            f"Non-finite participation values in realization {realization_id}."
        )
    if not np.all(np.isfinite(bounds)):
        raise FloatingPointError(
            f"Non-finite norm bounds in realization {realization_id}."
        )
    if not np.all(np.isfinite(all_cutoff_bounds)):
        raise FloatingPointError(
            f"Non-finite dense norm bounds in realization {realization_id}."
        )

    # Cross-check displayed cutoffs against the dense-cutoff calculation.
    for iq, nL in enumerate(nL_values):
        if int(nL) <= n_max:
            ref = all_cutoff_bounds[:, int(nL) - 2]
            if not np.allclose(ref, bounds[:, iq], rtol=2e-12, atol=2e-12):
                raise AssertionError(
                    f"Dense-cutoff norm mismatch at nL={int(nL)} "
                    f"in realization {realization_id}."
                )

    return participation, bounds, all_cutoff_bounds


# =============================================================================
# Ensemble mean / SEM
# =============================================================================

def mean_sem(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.mean(arr, axis=0)
    if arr.shape[0] > 1:
        sem = np.std(arr, axis=0, ddof=1) / math.sqrt(arr.shape[0])
    else:
        sem = np.full_like(mean, np.nan)
    return mean, sem


# =============================================================================
# Exact eta_crit validation
# =============================================================================

def eta_r(r: float) -> float:
    pref = math.exp(0.5772156649015329 / 2.0) / (2.0 * math.sqrt(math.pi))
    if r <= 0.0:
        return pref
    return pref * (1.0 + r) * math.exp(-(r / (1.0 + r)) * math.log(r))


def eta_crit_integral() -> float:
    def integrand(r: float) -> float:
        return eta_r(r) * 2.0 / (math.pi * (1.0 + r * r))

    from scipy.integrate import quad
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
# Power-law fit for norm curves
# =============================================================================

def model_func(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * x**b


def fit_exponent(N_values, y, nL):
    Nf = np.asarray(N_values, dtype=float)
    yf = np.asarray(y, dtype=float)
    N_max = int(Nf[-1])

    if int(nL) >= N_max:
        # nL = N_max: use all data points (global saturated curve).
        mask = np.isfinite(yf) & (yf > 0.0)
    else:
        # nL < N_max: fit only over N in [nL, N_max].
        mask = (Nf >= int(nL)) & np.isfinite(yf) & (yf > 0.0)

    xfit = Nf[mask]
    yfit = yf[mask]

    if xfit.size < 2:
        raise RuntimeError(
            f"Not enough points to fit nL={nL} "
            f"(only {xfit.size} points in [nL, N_max])."
        )

    lx = np.log(xfit)
    ly = np.log(yfit)

    if xfit.size == 2:
        # Exactly two points: slope is exact, but covariance is undefined,
        # so np.polyfit(..., cov=True) would return NaN and emit a warning.
        # Compute the slope directly instead.
        b = (ly[1] - ly[0]) / (lx[1] - lx[0])
        loga = ly[0] - b * lx[0]
        a = math.exp(loga)
        return a, b, math.nan, math.nan

    (b, loga), cov = np.polyfit(lx, ly, 1, full=False, cov=True)
    perr0, perr1 = np.sqrt(np.diag(cov)).tolist()
    a = np.exp(loga)
    return a, b, perr0, perr1


# =============================================================================
# Plot
# =============================================================================

def make_plot(
    N_values: np.ndarray,
    participation_mean: np.ndarray,
    participation_sem: np.ndarray,
    bound_mean: np.ndarray,
    bound_sem: np.ndarray,
    exponents: np.ndarray,
    exponent_err: np.ndarray,
    inset_nL_values: np.ndarray,
    inset_exponents: np.ndarray,
    inset_exponent_err: np.ndarray,
    outfile: Path,
) -> None:
    fig, axes = plt.subplot_mosaic(
        [["ratio"], ["power"]],
        figsize=(5, 5.7),
        constrained_layout=False,
        sharex=True,
    )
    fig.subplots_adjust(
        left=0.14,
        right=0.98,
        top=0.99,
        bottom=0.08,
        wspace=0.0,
        hspace=0.0,
    )

    # Top panel: eta(N).
    ax_left = axes["ratio"]
    for iq, nL in enumerate(N_L_VALUES):
        color = COLOURS[iq]
        ax_left.plot(
            N_values,
            participation_mean[:, iq],
            color=color,
            linewidth=1.1,
            label=rf"$n_L={nL}$",
        )
        if np.all(np.isfinite(participation_sem[:, iq])):
            ax_left.fill_between(
                N_values,
                np.maximum(participation_mean[:, iq] - participation_sem[:, iq], 1e-300),
                participation_mean[:, iq] + participation_sem[:, iq],
                color=color,
                alpha=0.08,
                linewidth=0.0,
            )

    x_ref = np.asarray(N_values[18:], dtype=np.float64)
    ax_left.loglog(x_ref, 5.0 * x_ref**(-2.0), "k--", linewidth=1.0)
    ax_left.text(180, 1e-5, r"$\sim N^{-2}$", fontsize=11)
    ax_left.set_xlim(N_MIN, N_MAX)
    ax_left.set_ylim(1e-7, 1.0)
    ax_left.set_ylabel(r"$\eta_N(n_L)$")
    ax_left.grid(which="both", alpha=0.3)
    ax_left.legend(
        ncol=2,
        fontsize=7.6,
        loc="lower left",
        frameon=True,
        handlelength=1.5,
        columnspacing=0.7,
    )
    ax_left.annotate(
        "(a)",
        xy=(0, 0.6),
        xycoords="axes fraction",
        xytext=(2, -2),
        textcoords="offset points",
        fontsize=12,
        va="top",
        fontfamily="serif",
    )
    ax_left.set_xscale("log")
    ax_left.set_yscale("log")

    # Bottom panel: commutator bound.
    ax_right = axes["power"]
    for iq, nL in enumerate(N_L_VALUES):
        color = COLOURS[iq]
        ax_right.plot(N_values, bound_mean[:, iq], color=color, linewidth=1.1)
        if np.all(np.isfinite(bound_sem[:, iq])):
            ax_right.fill_between(
                N_values,
                np.maximum(bound_mean[:, iq] - bound_sem[:, iq], 1e-300),
                bound_mean[:, iq] + bound_sem[:, iq],
                color=color,
                alpha=0.08,
                linewidth=0.0,
            )

    x_ref2 = np.asarray(N_values, dtype=np.float64)
    ax_right.plot(x_ref2, 2.0 * x_ref2, "k--", linewidth=1.0)
    ax_right.text(170, 100, r"$\sim N$", fontsize=11)
    ax_right.plot(x_ref2, 5.0 * x_ref2**2, "k--", linewidth=1.0)
    ax_right.text(170, 6e5, r"$\sim N^2$", fontsize=11)
    ax_right.set_xlim(N_MIN, N_MAX)
    ax_right.set_ylim(1.0, 5e8)
    ax_right.set_xlabel(r"$N$")
    ax_right.set_ylabel(r"$\mathcal{U}_{n_L}(N)$")
    ax_right.grid(which="both", alpha=0.3)
    ax_right.set_xscale("log")
    ax_right.set_yscale("log")
    ax_right.annotate(
        "(b)",
        xy=(1, 0.03),
        xycoords="axes fraction",
        xytext=(-4, -2),
        textcoords="offset points",
        fontsize=12,
        va="bottom",
        ha="right",
        fontfamily="serif",
    )

    ax_left.tick_params(axis="x", which="both", labelbottom=False)

    # Inset: ell(nL)=b(nL)-1.
    inset_ax = ax_right.inset_axes(
        [
            0.1,
            0.65,
            0.37,
            0.32,
        ]
    )

    x_orig = (inset_nL_values / N_MAX).astype(
        np.float64
    )

    y_orig = (
        inset_exponents - 1.0
    )

    y_orig = np.clip(
        y_orig,
        0.0,
        1.0,
    )

    x_smooth = np.geomspace(x_orig.min(), x_orig.max(), 1000)
    y_smooth = np.interp(np.log(x_smooth), np.log(x_orig), y_orig)

    cmap = LinearSegmentedColormap.from_list("custom_gradient", BASE_COLORS)
    norm = PowerNorm(gamma=1.0, vmin=float(x_smooth.min()), vmax=float(x_smooth.max()))
    points = np.asarray([x_smooth, y_smooth]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = mcoll.LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(x_smooth)
    lc.set_linewidth(2.0)
    inset_ax.add_collection(lc)

    inset_ax.set_xlim(0.0, 1.0)
    inset_ax.set_ylim(-0.01, 1.01)
    inset_ax.set_xlabel(r"$n_L/N$", fontsize=9)
    inset_ax.set_ylabel(r"$\ell(n_L)$", fontsize=9)
    inset_ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    inset_ax.set_yticklabels(
        ["0.0", "", "0.2", "", "0.4", "", "0.6", "", "0.8", "", "1.0"],
        fontsize=8,
    )
    inset_ax.tick_params(labelsize=8, direction="in", top=True, right=True)
    inset_ax.grid(True, which="both", alpha=0.3)
    inset_ax.patch.set_edgecolor("gray")
    inset_ax.patch.set_linewidth(0.5)

    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, bbox_inches="tight", dpi=300)
    plt.show()


# =============================================================================
# Data output
# =============================================================================

def save_csv(
    outfile: Path,
    N_values: np.ndarray,
    nL_values: np.ndarray,
    mean: np.ndarray,
    sem: np.ndarray,
    prefix: str,
) -> None:
    header = ["N"]
    for nL in nL_values:
        header.extend([f"{prefix}_nL{int(nL)}", f"{prefix}_sem_nL{int(nL)}"])

    data = np.empty((N_values.size, 1 + 2 * nL_values.size), dtype=np.float64)
    data[:, 0] = N_values
    col = 1
    for iq in range(nL_values.size):
        data[:, col] = mean[:, iq]
        data[:, col + 1] = sem[:, iq]
        col += 2

    np.savetxt(outfile, data, delimiter=",", header=",".join(header), comments="")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N-max", type=int, default=N_MAX)
    parser.add_argument("--n-realizations", type=int, default=N_REALIZATIONS)
    parser.add_argument("--n-jobs", type=int, default=N_JOBS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--J", type=float, default=J)
    parser.add_argument("--outdir", type=Path, default=Path("participation_norm_vs_nL_results"))
    args = parser.parse_args()

    if args.N_max < 2:
        raise ValueError("N-max must be >= 2.")
    if args.n_realizations < 1:
        raise ValueError("n-realizations must be >= 1.")
    if args.n_jobs < 1:
        raise ValueError("n-jobs must be >= 1.")
    if args.J <= 0.0:
        raise ValueError("J must be positive.")

    N_values = np.arange(N_MIN, args.N_max + 1, dtype=np.int64)

    # Analytic eta_crit sanity check.
    eta_crit_num = eta_crit_integral()
    print(f"eta_crit (quadrature) = {eta_crit_num:.12f}")
    if not math.isclose(eta_crit_num, ETA_CRIT, rel_tol=2e-10, abs_tol=2e-10):
        raise RuntimeError("eta_crit quadrature check failed.")

    # Parallel realizations.
    seeds = [args.seed + 104729 * r for r in range(args.n_realizations)]
    print("\nRunning corrected finite-cutoff participation + tight-bound benchmark")
    print(f"N = {N_MIN}..{args.N_max}")
    print("nL = " + ", ".join(str(int(x)) for x in N_L_VALUES))
    print(f"realizations = {args.n_realizations}")
    print(f"joblib workers = {min(args.n_jobs, args.n_realizations)}")

        # ------------------------------------------------------------------
    # Parallel realizations, processed in batches.
    #
    # one_realization returns a dense all_cutoff_bounds matrix of shape
    # (N_max-1, N_max-1).  At N_max=10^4 this is ~800 MB per realization,
    # so keeping all realizations in memory at once is impossible.
    # Only the mean and SEM of that matrix are ever used downstream, so
    # we accumulate them online (Chan et al. parallel-variance formula)
    # and never materialise the full stack.  participation and bounds
    # are small (16 columns) and are still stacked normally.
    # ------------------------------------------------------------------
    batch_size = max(1, min(args.n_jobs, args.n_realizations))
    print(
        f"\nProcessing {args.n_realizations} realizations "
        f"in batches of {batch_size}"
    )

    participation_list = []
    bound_list = []

    acb_mean = None      # running mean of all_cutoff_bounds
    acb_M2 = None        # running sum of squared deviations
    acb_count = 0

    for batch_start in range(0, args.n_realizations, batch_size):

        batch_end = min(
            batch_start + batch_size,
            args.n_realizations,
        )

        batch_indices = list(
            range(batch_start, batch_end)
        )

        print(
            f"  batch {batch_start}-{batch_end - 1} "
            f"({batch_end - batch_start} realizations)"
        )

        batch_results = Parallel(
            n_jobs=len(batch_indices),
            verbose=10,
        )(
            delayed(one_realization)(
                r,
                args.N_max,
                N_L_VALUES,
                seeds[r],
                args.J,
            )
            for r in batch_indices
        )

        batch_K = len(batch_results)

        # Batch mean of all_cutoff_bounds, computed without copying
        # the dense arrays out of batch_results.
        batch_mean = batch_results[0][2].copy()
        for i in range(1, batch_K):
            batch_mean += batch_results[i][2]
        batch_mean /= batch_K

        # Batch sum of squared deviations, in-place to bound memory.
        batch_M2 = np.zeros_like(batch_mean)
        scratch = np.empty_like(batch_mean)
        for i in range(batch_K):
            np.subtract(batch_results[i][2], batch_mean, out=scratch)
            np.multiply(scratch, scratch, out=scratch)
            batch_M2 += scratch
        del scratch

        # Extract the small participation/bounds before freeing the
        # dense batch arrays.
        for participation, bounds, _acb in batch_results:
            participation_list.append(participation)
            bound_list.append(bounds)

        # Now the dense arrays are unreferenced and will be freed.
        del batch_results

        # Combine batch statistics with the running ones.
        if acb_mean is None:
            acb_mean = batch_mean
            acb_M2 = batch_M2
            acb_count = batch_K
        else:
            delta = batch_mean - acb_mean
            total_n = acb_count + batch_K
            acb_mean = acb_mean + delta * (batch_K / total_n)
            acb_M2 = (
                acb_M2
                + batch_M2
                + delta * delta
                * (acb_count * batch_K / total_n)
            )
            acb_count = total_n

        del batch_mean, batch_M2

    # ------------------------------------------------------------------
    # Finalise the ensemble statistics.
    # ------------------------------------------------------------------
    all_cutoff_bound_mean = acb_mean
    if acb_count > 1:
        all_cutoff_bound_sem = np.sqrt(
            acb_M2 / (acb_count * (acb_count - 1))
        )
    else:
        all_cutoff_bound_sem = np.full_like(acb_mean, np.nan)

    del acb_mean, acb_M2

    participation_stack = np.stack(participation_list, axis=0)
    bound_stack = np.stack(bound_list, axis=0)
    del participation_list, bound_list

    participation_mean, participation_sem = mean_sem(participation_stack)
    bound_mean, bound_sem = mean_sem(bound_stack)

    # Dense cutoff exponent extraction: nL=2,...,10000. For nL>N the
    # corresponding finite-N curve is the saturated global curve.
    inset_nL_values = np.arange(2, N_MAX + 1, dtype=np.int64)
    inset_exponents = np.empty(inset_nL_values.size, dtype=np.float64)
    inset_exponent_err = np.empty(inset_nL_values.size, dtype=np.float64)
    inset_fit_amplitudes = np.empty(inset_nL_values.size, dtype=np.float64)
    inset_fit_amplitude_err = np.empty(inset_nL_values.size, dtype=np.float64)

    exponents = np.empty(N_L_VALUES.size, dtype=np.float64)
    exponent_err = np.empty(N_L_VALUES.size, dtype=np.float64)
    fit_amplitudes = np.empty(N_L_VALUES.size, dtype=np.float64)
    fit_amplitude_err = np.empty(N_L_VALUES.size, dtype=np.float64)

    print("\nPower-law fits y=a*N^b")
    print("=" * 78)

    for iq, nL in enumerate(inset_nL_values):
        c = min(int(nL), args.N_max)
        col = c - 2
        y = all_cutoff_bound_mean[:, col]
        a, b, ea, eb = fit_exponent(N_values, y, int(nL))
        inset_fit_amplitudes[iq] = a
        inset_exponents[iq] = b
        inset_fit_amplitude_err[iq] = ea
        inset_exponent_err[iq] = eb

    print("\nSelected cutoff power-law fits y=a*N^b")
    for iq, nL in enumerate(N_L_VALUES):
        a = inset_fit_amplitudes[int(nL) - 2]
        b = inset_exponents[int(nL) - 2]
        ea = inset_fit_amplitude_err[int(nL) - 2]
        eb = inset_exponent_err[int(nL) - 2]
        fit_amplitudes[iq] = a
        exponents[iq] = b
        fit_amplitude_err[iq] = ea
        exponent_err[iq] = eb
        print(
            f"nL={int(nL):5d}: a={a:.8e} +/- {ea:.3e}, "
            f"ell={b:.8f} +/- {eb:.3e}, ell-1={b-1.0:.8f}"
        )

    # Cutoffs nL>=N_max should be exactly identical within the same ensemble.
    global_indices = np.where(N_L_VALUES >= args.N_max)[0]
    if global_indices.size >= 2:
        ref = participation_mean[:, global_indices[0]]
        for iq in global_indices[1:]:
            if not np.allclose(ref, participation_mean[:, iq], rtol=0.0, atol=0.0):
                raise AssertionError(
                    "nL >= N_MAX participation curves should be identical in the same realization ensemble."
                )
            if not np.allclose(
                bound_mean[:, global_indices[0]],
                bound_mean[:, iq],
                rtol=0.0,
                atol=0.0,
            ):
                raise AssertionError(
                    "nL >= N_MAX norm-bound curves should be identical in the same realization ensemble."
                )

    # Global/cumulative-locality eta diagnostic.
    global_candidates = np.where(N_L_VALUES >= args.N_max)[0]
    if global_candidates.size == 0:
        raise ValueError("N_L_VALUES must include at least one cutoff >= --N-max.")
    global_iq = int(global_candidates[0])
    print("\nGlobal/cumulative-locality eta check")
    print(
        f"eta(N={args.N_max}, nL={int(N_L_VALUES[global_iq])}) = "
        f"{participation_mean[-1, global_iq]:.8f} +/- {participation_sem[-1, global_iq]:.8f}"
    )
    print(f"eta_crit = {ETA_CRIT:.8f}")

    # Save numerical data.
    args.outdir.mkdir(parents=True, exist_ok=True)
    participation_csv = args.outdir / "participation_vs_nL.csv"
    bound_csv = args.outdir / "norm_bound_vs_nL.csv"
    fit_csv = args.outdir / "norm_exponents_vs_nL.csv"
    npz_file = args.outdir / "participation_norm_vs_nL.npz"
    figure_file = args.outdir / "participation_norm_vs_nL.pdf"

    save_csv(participation_csv, N_values, N_L_VALUES, participation_mean, participation_sem, "eta")
    save_csv(bound_csv, N_values, N_L_VALUES, bound_mean, bound_sem, "bound")

    np.savetxt(
        fit_csv,
        np.column_stack([
            inset_nL_values,
            inset_fit_amplitudes,
            inset_fit_amplitude_err,
            inset_exponents,
            inset_exponent_err,
        ]),
        delimiter=",",
        header="nL,amplitude,amplitude_err,exponent,exponent_err",
        comments="",
    )

    np.savez_compressed(
        npz_file,
        N=N_values,
        nL=N_L_VALUES,
        participation_mean=participation_mean,
        participation_sem=participation_sem,
        bound_mean=bound_mean,
        bound_sem=bound_sem,
        fit_amplitude=fit_amplitudes,
        fit_amplitude_err=fit_amplitude_err,
        fit_exponent=exponents,
        fit_exponent_err=exponent_err,
        inset_nL=inset_nL_values,
        inset_fit_amplitude=inset_fit_amplitudes,
        inset_fit_amplitude_err=inset_fit_amplitude_err,
        inset_fit_exponent=inset_exponents,
        inset_fit_exponent_err=inset_exponent_err,
        all_cutoff_bound_mean=all_cutoff_bound_mean,
        all_cutoff_bound_sem=all_cutoff_bound_sem,
        eta_crit=ETA_CRIT,
    )

    make_plot(
        N_values,
        participation_mean,
        participation_sem,
        bound_mean,
        bound_sem,
        exponents,
        exponent_err,
        inset_nL_values,
        inset_exponents,
        inset_exponent_err,
        figure_file,
    )

    print("\nSaved:")
    print(participation_csv.resolve())
    print(bound_csv.resolve())
    print(fit_csv.resolve())
    print(npz_file.resolve())
    print(figure_file.resolve())


if __name__ == "__main__":
    main()