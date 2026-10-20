import json
import math
import csv
import numpy as np
import matplotlib.pyplot as plt
import openfermion as of
from joblib import Parallel, delayed
from scipy.sparse.linalg import expm_multiply, eigsh
from numba import njit
from matplotlib.patches import Rectangle, ConnectionPatch

plt.rcParams.update({
    'pgf.texsystem': 'pdflatex',
    'text.usetex': True,
    'font.family': 'serif',
    'pgf.preamble': '\n'.join([
        r'\usepackage{amsmath}',
    ]),
    'text.latex.preamble': '\n'.join([
        r'\usepackage{amsmath}',
    ])
})

# ============================================================
# USER PARAMETERS
# ============================================================
N_REALIZATIONS = 32
N_JOBS = -1                    # -1 = all available CPU cores
EXACT_N_MAX = 16
SEED = 13
N_MIN = 2
N_MAX = 1000
N_L_SETTING = 1000         # "global" or integer
OUTPUT_FILE = f"data_general_disorder_averaged_{N_L_SETTING}.json"
CSV_FILE = f"data_per_realization_{N_L_SETTING}.csv"
FIGURE_FILE = f"general_disorder_averaged_{N_L_SETTING}.pdf"
POWER_FIGURE_FILE = f"general_disorder_power_vs_time_{N_L_SETTING}.pdf"

# Number of time points for the dynamical calculation.
POWER_N_TIME = 100

# N values shown in the P(t)/N figure.
POWER_PLOT_N_VALUES = range(2, EXACT_N_MAX+1, 2)

# ============================================================
# N-DEPENDENT POWER TIME WINDOW
# ============================================================

def power_t_max(N):
    """
    Linear interpolation between
        N = 2   -> t_max = 1.0
        N = 16  -> t_max = 0.3
    The function is clipped to these values outside [2,16].
    """
    N0, N1 = 2, 16

    # Support both the integer locality setting and "global".
    if N_L_SETTING == "global" or (
        isinstance(N_L_SETTING, int) and N_L_SETTING > 2
    ):
        t0, t1 = 1.0, 0.35
    else:
        t0, t1 = 1.0, 1.0

    if N <= N0:
        return t0
    if N >= N1:
        return t1

    return t0 + (t1 - t0) * (N - N0) / (N1 - N0)


# ============================================================
# CHARGER
# ============================================================

def charger_sigma(N, nL, J=1):
    nL = min(nL, N)
    return J * math.sqrt(
        2.0 * N / (N + (nL - 1) * (2 * N - nL))
    )


def generate_banded_matrices(N, nL, rng):
    nL = min(nL, N)

    Af = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    Bf = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))

    # Af = np.ones((N,N)) + 1j * np.ones((N,N))
    # Bf = np.ones((N,N)) + 1j * np.ones((N,N))

    # Keep only diagonals 0, 1, ..., nL-1.
    A = np.triu(Af, k=0) - np.triu(Af, k=nL)
    B = np.triu(Bf, k=0) - np.triu(Bf, k=nL)

    A = (A + A.conj().T) / 2.0
    B = (B - B.T) / 2.0
    np.fill_diagonal(B, 0.0)

    # Symmetrization/antisymmetrization halves each retained off-diagonal
    # complex entry. Restore E[|A_kl|^2] = E[|B_kl|^2] = 1 for k < l,
    # as assumed by charger_sigma; the diagonal A_kk variance remains 1.
    off_diagonal = ~np.eye(N, dtype=np.bool_)
    A[off_diagonal] *= math.sqrt(2.0)
    B[off_diagonal] *= math.sqrt(2.0)

    return A, B


def normalize_charger(A, B, nL):
    sigma = charger_sigma(A.shape[0], nL)
    return sigma * A, sigma * B, sigma


# ============================================================
# ANALYTICAL COMMUTATOR BOUNDS (NUMBA-ACCELERATED)
# ============================================================

@njit(cache=True, fastmath=True)
def _compute_norm_bounds_numba(
    rA, iA, rB, iB, beta, gamma, delta, nL
):
    N = beta.shape[0]

    C1_tight = 0.0

    left_C3_sq = np.zeros(N, dtype=np.float64)
    right_C3_sq = np.zeros(N, dtype=np.float64)

    x_C4_sq = np.zeros(N, dtype=np.float64)
    y_C4_sq = np.zeros(N, dtype=np.float64)

    Lambda1 = Lambda2 = Lambda3 = Lambda4 = 0.0

    for k in range(N):
        k11 = -rA[k, k] * gamma[k]
        k12 = rA[k, k] * beta[k]

        C1_tight += np.sqrt(k11 * k11 + k12 * k12)
        Lambda1 = max(Lambda1, abs(k11), abs(k12))

        lmax = min(N, k + nL)

        for l in range(k + 1, lmax):
            ar = rA[k, l]
            ai = iA[k, l]
            br = rB[k, l]
            bi = iB[k, l]

            rp = ar + br
            rm = ar - br
            ip = ai + bi
            im = ai - bi

            # ------------------------------------------------
            # C2: retain Lambda2 coefficients; compute the refined
            # system-parameter C2 bound below.
            # ------------------------------------------------
            if l >= k + 2:
                for m in range(k + 1, l):
                    be = beta[m]
                    ga = gamma[m]

                    k21 = ga * rp
                    k22 = -be * rp
                    k23 = -be * rm
                    k24 = ga * rm
                    k25 = ga * im
                    k26 = -be * im
                    k27 = -be * ip
                    k28 = ga * ip

                    Lambda2 = max(
                        Lambda2,
                        abs(k21), abs(k22),
                        abs(k23), abs(k24),
                        abs(k25), abs(k26),
                        abs(k27), abs(k28),
                    )

            # ------------------------------------------------
            # C3
            # ------------------------------------------------
            k31 = beta[k] * (ai + bi) - gamma[l] * (ar + br)
            k32 = beta[k] * (ar - br) + gamma[l] * (ai - bi)
            k33 = -beta[l] * (ai - bi) - gamma[k] * (ar + br)
            k34 = beta[l] * (ar - br) - gamma[k] * (ai + bi)

            left_C3_sq[k] += k31 * k31 + k32 * k32
            right_C3_sq[l] += k33 * k33 + k34 * k34

            Lambda3 = max(
                Lambda3,
                abs(k31), abs(k32),
                abs(k33), abs(k34),
            )

            # ------------------------------------------------
            # C4
            # ------------------------------------------------
            k41 = -delta[k] * (ai + bi) + delta[l] * (ai - bi)
            k42 = -delta[k] * (ar - br) + delta[l] * (ar + br)
            k43 = delta[k] * (ar + br) - delta[l] * (ar - br)
            k44 = -delta[k] * (ai - bi) + delta[l] * (ai + bi)

            x_C4_sq[k] += k41 * k41 + k42 * k42
            y_C4_sq[k] += k43 * k43 + k44 * k44

            Lambda4 = max(
                Lambda4,
                abs(k41), abs(k42),
                abs(k43), abs(k44),
            )

    C3_tight = (
        np.sqrt(left_C3_sq).sum()
        + np.sqrt(right_C3_sq).sum()
    )

    C4_tight = (
        np.sqrt(x_C4_sq).sum()
        + np.sqrt(y_C4_sq).sum()
    )

    # --------------------------------------------------------
    # Refined C2 bound from the nuclear/Frobenius norm estimate:
    #
    # ||C2|| <= 2 sqrt(
    #     2 * sum_l r_l (beta_l^2 + gamma_l^2)
    #       * sum_{k<m} (m-k-1)(|A_km|^2 + |B_km|^2)
    # )
    #
    # Here q = nL-2 and r_l=min(q,l-1,N-l).
    # --------------------------------------------------------
    q = max(0, nL - 2)

    battery_rank_weight = 0.0
    for ell in range(1, N - 1):
        r_l = min(q, ell, N - ell - 1)

        battery_rank_weight += r_l * (
            beta[ell] * beta[ell] + gamma[ell] * gamma[ell]
        )

    # Each pair k<m with 2 <= m-k <= nL-1 occurs for (m-k-1)
    # intermediate sites. The matrices passed here have already been
    # multiplied by sigma_L in run_realization, so do not scale again.
    weighted_coupling_norm_sq = 0.0

    for k in range(N):
        m_stop = min(N, k + nL)

        for m in range(k + 2, m_stop):
            distance_weight = m - k - 1

            weighted_coupling_norm_sq += distance_weight * (
                rA[k, m] * rA[k, m]
                + iA[k, m] * iA[k, m]
                + rB[k, m] * rB[k, m]
                + iB[k, m] * iB[k, m]
            )

    C2_tight = 2.0 * np.sqrt(
        2.0 * battery_rank_weight * weighted_coupling_norm_sq
    )

    # --------------------------------------------------------
    # Lambda bounds
    # --------------------------------------------------------
    C1_lambda = np.sqrt(2.0) * Lambda1 * N

    S2 = 0.0
    for k in range(N - 2):
        for l in range(k + 2, min(N, k + nL)):
            S2 += np.sqrt(l - k - 1)

    C2_lambda = 4.0 * np.sqrt(2.0) * Lambda2 * S2

    S34 = 0.0
    for r in range(1, nL):
        S34 += np.sqrt(r)

    if N > nL:
        S34 += (N - nL) * np.sqrt(nL - 1)

    C3_lambda = 2.0 * np.sqrt(2.0) * Lambda3 * S34
    C4_lambda = 2.0 * np.sqrt(2.0) * Lambda4 * S34

    tight = (
        C1_tight
        + C2_tight
        + C3_tight
        + C4_tight
    )

    lam = (
        C1_lambda
        + C2_lambda
        + C3_lambda
        + C4_lambda
    )

    return (
        tight,
        lam,
        C1_tight,
        C2_tight,
        C3_tight,
        C4_tight,
        C1_lambda,
        C2_lambda,
        C3_lambda,
        C4_lambda,
        Lambda1,
        Lambda2,
        Lambda3,
        Lambda4,
    )


def compute_norm_bounds(
    rA, iA, rB, iB, beta, gamma, delta, nL
):
    vals = _compute_norm_bounds_numba(
        rA, iA, rB, iB, beta, gamma, delta, int(nL)
    )

    return {
        "tight": vals[0],
        "lambda": vals[1],
        "C1_tight": vals[2],
        "C2_tight": vals[3],
        "C3_tight": vals[4],
        "C4_tight": vals[5],
        "C1_lambda": vals[6],
        "C2_lambda": vals[7],
        "C3_lambda": vals[8],
        "C4_lambda": vals[9],
        "Lambda1": vals[10],
        "Lambda2": vals[11],
        "Lambda3": vals[12],
        "Lambda4": vals[13],
    }


# ============================================================
# HAMILTONIANS
# ============================================================

def build_battery_qubit_operator(beta, gamma, delta):
    H = of.QubitOperator()

    for i, (b, g, d) in enumerate(
        zip(beta, gamma, delta)
    ):
        if b != 0.0:
            H += of.QubitOperator(
                ((i, "X"),),
                float(b)
            )

        if g != 0.0:
            H += of.QubitOperator(
                ((i, "Y"),),
                float(g)
            )

        if d != 0.0:
            H += of.QubitOperator(
                ((i, "Z"),),
                float(d)
            )

    return H


def build_charger_fermion_operator(A, B):
    N = A.shape[0]
    H = of.FermionOperator()

    for k in range(N):
        for l in range(N):
            if A[k, l] != 0:
                H += of.FermionOperator(
                    ((k, 1), (l, 0)),
                    complex(A[k, l])
                )

            if B[k, l] != 0:
                H += 0.5 * of.FermionOperator(
                    ((k, 1), (l, 1)),
                    complex(B[k, l])
                )

                H += 0.5 * of.FermionOperator(
                    ((l, 0), (k, 0)),
                    complex(B[k, l]).conjugate()
                )

    return H


def build_many_body_matrices(A, B, beta, gamma, delta):
    N = len(beta)

    H_b = of.get_sparse_operator(
        build_battery_qubit_operator(
            beta, gamma, delta
        ),
        n_qubits=N
    ).tocsr()

    H_c = of.get_sparse_operator(
        of.jordan_wigner(
            build_charger_fermion_operator(A, B)
        ),
        n_qubits=N
    ).tocsr()

    return H_b, H_c


# ============================================================
# EXACT COMMUTATOR NORM
# ============================================================

def exact_general_commutator_norm(comm):
    dimension = comm.shape[0]

    if dimension <= 256:
        norm_exact = float(
            np.linalg.svd(
                comm.toarray(),
                compute_uv=False,
            )[0]
        )
    else:
        # 1j * comm is Hermitian because comm is anti-Hermitian.
        # Symmetrize to remove tiny floating-point Hermitian part.
        M = 0.5j * (comm - comm.getH())

        w = eigsh(
            M,
            k=1,
            which="LM",
            return_eigenvectors=False,
        )

        norm_exact = float(abs(w[0]))

    anti = comm + comm.getH()

    anti_error = (
        float(np.max(np.abs(anti.data)))
        if anti.nnz
        else 0.0
    )

    return norm_exact, anti_error, comm


# ============================================================
# GROUND STATE OF H_b
# ============================================================

def battery_ground_state(beta, gamma, delta):
    """
    H_b is a sum of one-site terms, so its ground state is
    a tensor product of one-qubit ground states.
    """
    psi = np.array(
        [1.0 + 0.0j]
    )

    for b, g, d in zip(
        beta, gamma, delta
    ):
        h = np.array(
            [
                [d, b - 1j * g],
                [b + 1j * g, -d]
            ],
            dtype=complex
        )

        _, vecs = np.linalg.eigh(h)

        # Lowest one-qubit eigenvector.
        psi = np.kron(
            psi,
            vecs[:, 0]
        )

    return psi / np.linalg.norm(psi)


# ============================================================
# POWER DYNAMICS
# ============================================================

def compute_power_vs_time(
    comm,
    H_c,
    beta,
    gamma,
    delta,
    times
):
    """
    Evolution is generated by H_c ONLY:
        |psi(t)> = exp(-i H_c t) |psi(0)>,
    where |psi(0)> is the ground state of H_b.

    The power is
        P(t) = -i <psi(t)|[H_b,H_c]|psi(t)>.

    Returns P(t) and max_t |P(t)|.
    """
    psi0 = battery_ground_state(
        beta,
        gamma,
        delta
    )

    states = expm_multiply(
        -1j * H_c,
        psi0,
        start=float(times[0]),
        stop=float(times[-1]),
        num=len(times)
    )

    power = np.empty(
        len(times),
        dtype=float
    )

    # Chunked matrix-vector products to improve efficiency
    chunk_size = 64

    for start in range(0, len(times), chunk_size):
        stop = min(start + chunk_size, len(times))

        block = states[start:stop].T          # shape: dim x chunk
        comm_block = comm @ block             # shape: dim x chunk

        vals = np.sum(block.conj() * comm_block, axis=0)

        power[start:stop] = np.real(-1j * vals)

    return (
        power,
        float(np.max(np.abs(power)))
    )


# ============================================================
# STATISTICS
# ============================================================

def mean_and_sem(values):
    values = np.asarray(
        values,
        dtype=float
    )

    mean = float(
        np.mean(values)
    )

    if len(values) > 1:
        std = float(
            np.std(values, ddof=1)
        )

        sem = std / math.sqrt(
            len(values)
        )
    else:
        std = 0.0
        sem = 0.0

    return mean, std, sem


# ============================================================
# ONE DISORDER REALIZATION
# ============================================================

def run_realization(
    N,
    nL,
    seed,
    compute_exact,
    times
):
    rng = np.random.default_rng(
        seed
    )

    # --------------------------------------------------------
    # Random charger
    # --------------------------------------------------------
    A_raw, B_raw = generate_banded_matrices(
        N,
        nL,
        rng
    )

    A, B, _ = normalize_charger(
        A_raw,
        B_raw,
        nL
    )

    # --------------------------------------------------------
    # Random battery
    # --------------------------------------------------------
    beta = rng.standard_normal(N)
    gamma = rng.standard_normal(N)
    delta = rng.standard_normal(N)

    # --------------------------------------------------------
    # Analytical bounds
    # --------------------------------------------------------
    result = compute_norm_bounds(
        A.real,
        A.imag,
        B.real,
        B.imag,
        beta,
        gamma,
        delta,
        nL
    )

    # --------------------------------------------------------
    # Exact / dynamical quantities only for N <= cutoff
    # --------------------------------------------------------
    if compute_exact:
        H_b, H_c = build_many_body_matrices(
            A,
            B,
            beta,
            gamma,
            delta
        )

        # Compute commutator once and reuse it
        comm = (H_b @ H_c - H_c @ H_b).tocsr()

        (
            exact_norm,
            anti_error,
            _
        ) = exact_general_commutator_norm(
            comm
        )

        power, power_star = compute_power_vs_time(
            comm,
            H_c,
            beta,
            gamma,
            delta,
            times
        )

        result.update({
            "exact_norm": exact_norm,
            "antihermiticity_error": anti_error,
            "power_star": power_star,
            "power_vs_time": power
        })

    return result


# ============================================================
# MAIN SIMULATION
# ============================================================

if __name__ == "__main__":
    if any(
        N > EXACT_N_MAX
        for N in POWER_PLOT_N_VALUES
    ):
        raise ValueError(
            "POWER_PLOT_N_VALUES must contain only "
            "values <= EXACT_N_MAX."
        )

    data = {}

    # For per-realization CSV output
    per_realization_rows = [dict() for _ in range(N_REALIZATIONS)]

    keys_to_average = [
        "tight",
        "lambda",
        "C1_tight",
        "C2_tight",
        "C3_tight",
        "C4_tight",
        "C1_lambda",
        "C2_lambda",
        "C3_lambda",
        "C4_lambda",
        "Lambda1",
        "Lambda2",
        "Lambda3",
        "Lambda4",
    ]

    for N in range(
        N_MIN,
        N_MAX + 1
    ):

        # ----------------------------------------------------
        # Locality
        # ----------------------------------------------------
        if N_L_SETTING == "global":
            nL = N
        elif isinstance(
            N_L_SETTING,
            int
        ):
            nL = min(
                N_L_SETTING,
                N
            )
        else:
            raise ValueError(
                "N_L_SETTING must be 'global' "
                "or an integer."
            )

        # ----------------------------------------------------
        # Expensive calculation cutoff
        # ----------------------------------------------------
        compute_exact = (
            N <= EXACT_N_MAX
        )

        # ----------------------------------------------------
        # N-dependent time window
        # ----------------------------------------------------
        if compute_exact:
            t_max = power_t_max(N)

            times = np.linspace(
                0.0,
                t_max,
                POWER_N_TIME
            )
        else:
            times = None

        # ----------------------------------------------------
        # Independent reproducible seeds
        # ----------------------------------------------------
        seeds = [
            int(
                np.random.SeedSequence(
                    [SEED, N, r]
                ).generate_state(
                    1,
                    dtype=np.uint64
                )[0]
            )
            for r in range(
                N_REALIZATIONS
            )
        ]

        # ----------------------------------------------------
        # Parallel disorder realizations
        # ----------------------------------------------------
        results = Parallel(
            n_jobs=N_JOBS,
            backend="loky",
            verbose=0
        )(
            delayed(run_realization)(
                N,
                nL,
                seeds[r],
                compute_exact,
                times
            )
            for r in range(
                N_REALIZATIONS
            )
        )

        # ----------------------------------------------------
        # Store per-realization data for CSV
        # ----------------------------------------------------
        for r, res in enumerate(results):
            if compute_exact:
                per_realization_rows[r][f"power_{N}"] = res.get("power_star", np.nan)
                per_realization_rows[r][f"norm_comm_{N}"] = res.get("exact_norm", np.nan)
            else:
                per_realization_rows[r][f"power_{N}"] = np.nan
                per_realization_rows[r][f"norm_comm_{N}"] = np.nan

            per_realization_rows[r][f"bound_{N}"] = res["tight"]

        averaged = {}

        # ====================================================
        # Average analytical quantities
        # ====================================================
        for key in keys_to_average:
            vals = [
                result[key]
                for result in results
            ]

            mean, std, sem = mean_and_sem(
                vals
            )

            averaged[key] = mean
            averaged[f"{key}_std"] = std
            averaged[f"{key}_sem"] = sem

        # ====================================================
        # Small-N exact quantities
        # ====================================================
        if compute_exact:

            # ------------------------------------------------
            # Exact norm
            # ------------------------------------------------
            exact_vals = [
                result["exact_norm"]
                for result in results
            ]

            (
                exact_mean,
                exact_std,
                exact_sem
            ) = mean_and_sem(
                exact_vals
            )

            averaged.update({
                "exact_norm": exact_mean,
                "exact_norm_std": exact_std,
                "exact_norm_sem": exact_sem,
                "exact_available": True
            })

            # ------------------------------------------------
            # P_*
            # ------------------------------------------------
            pstar_vals = [
                result["power_star"]
                for result in results
            ]

            (
                pstar_mean,
                pstar_std,
                pstar_sem
            ) = mean_and_sem(
                pstar_vals
            )

            averaged.update({
                "power_star": pstar_mean,
                "power_star_std": pstar_std,
                "power_star_sem": pstar_sem,
                "power_available": True
            })

            # ------------------------------------------------
            # Full P(t), averaged over realizations
            # ------------------------------------------------
            power_traces = np.asarray([
                result["power_vs_time"]
                for result in results
            ])

            p_mean = np.mean(
                power_traces,
                axis=0
            )

            if N_REALIZATIONS > 1:
                p_std = np.std(
                    power_traces,
                    axis=0,
                    ddof=1
                )
            else:
                p_std = np.zeros_like(
                    p_mean
                )

            p_sem = (
                p_std
                / math.sqrt(N_REALIZATIONS)
            )

            averaged["power_times"] = (
                times.tolist()
            )

            averaged["power_t_max"] = (
                t_max
            )

            averaged["power_vs_time"] = (
                p_mean.tolist()
            )

            averaged["power_vs_time_std"] = (
                p_std.tolist()
            )

            averaged["power_vs_time_sem"] = (
                p_sem.tolist()
            )

            # Requested quantity:
            #
            #     <P(t)> / N
            #
            averaged["power_per_N_vs_time"] = (
                (p_mean / N).tolist()
            )

            averaged["power_per_N_vs_time_std"] = (
                (p_std / N).tolist()
            )

            averaged["power_per_N_vs_time_sem"] = (
                (p_sem / N).tolist()
            )

            # ------------------------------------------------
            # Diagnostics
            # ------------------------------------------------
            anti_vals = [
                result["antihermiticity_error"]
                for result in results
            ]

            averaged[
                "antihermiticity_error_mean"
            ] = float(
                np.mean(anti_vals)
            )

            averaged[
                "antihermiticity_error_max"
            ] = float(
                np.max(anti_vals)
            )

            # ------------------------------------------------
            # Ratios
            # ------------------------------------------------
            tight_ratios = [
                result["exact_norm"] / result["tight"]
                for result in results
                if result["tight"] > 0
            ]

            lambda_ratios = [
                result["exact_norm"] / result["lambda"]
                for result in results
                if result["lambda"] > 0
            ]

            (
                mean,
                std,
                sem
            ) = mean_and_sem(
                tight_ratios
            )

            averaged[
                "exact_vs_tight_ratio"
            ] = mean

            averaged[
                "exact_vs_tight_ratio_std"
            ] = std

            averaged[
                "exact_vs_tight_ratio_sem"
            ] = sem

            (
                mean,
                std,
                sem
            ) = mean_and_sem(
                lambda_ratios
            )

            averaged[
                "exact_vs_lambda_ratio"
            ] = mean

            averaged[
                "exact_vs_lambda_ratio_std"
            ] = std

            averaged[
                "exact_vs_lambda_ratio_sem"
            ] = sem

        else:

            # ------------------------------------------------
            # No expensive calculation beyond EXACT_N_MAX
            # ------------------------------------------------
            averaged.update({
                "exact_norm": None,
                "exact_norm_std": None,
                "exact_norm_sem": None,
                "exact_available": False,

                "power_star": None,
                "power_star_std": None,
                "power_star_sem": None,
                "power_available": False,

                "power_times": None,
                "power_t_max": None,
                "power_vs_time": None,
                "power_vs_time_std": None,
                "power_vs_time_sem": None,

                "power_per_N_vs_time": None,
                "power_per_N_vs_time_std": None,
                "power_per_N_vs_time_sem": None,

                "antihermiticity_error_mean": None,
                "antihermiticity_error_max": None,

                "exact_vs_tight_ratio": None,
                "exact_vs_tight_ratio_std": None,
                "exact_vs_tight_ratio_sem": None,

                "exact_vs_lambda_ratio": None,
                "exact_vs_lambda_ratio_std": None,
                "exact_vs_lambda_ratio_sem": None
            })

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------
        averaged.update({
            "N_REALIZATIONS": N_REALIZATIONS,
            "N_JOBS": N_JOBS,
            "nL": nL,
            "sigma_L": charger_sigma(N, nL),
            "POWER_N_TIME": (
                POWER_N_TIME
                if compute_exact
                else None
            )
        })

        # ----------------------------------------------------
        # Save after each N
        # ----------------------------------------------------
        data[str(N)] = averaged

        with open(
            OUTPUT_FILE,
            "w"
        ) as f:
            json.dump(
                data,
                f,
                indent=2
            )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------
        if compute_exact:
            print(
                f"N={N:3d}: "
                f"t_max={t_max:.3f}, "
                f"<tight>={averaged['tight']:.6g}, "
                f"<lambda>={averaged['lambda']:.6g}, "
                f"<exact>={averaged['exact_norm']:.6g}, "
                f"<P_*>={averaged['power_star']:.6g}"
            )
        else:
            print(
                f"N={N:3d}: "
                f"<tight>={averaged['tight']:.6g}, "
                f"<lambda>={averaged['lambda']:.6g}",
                end="\r"
            )

    print()
    print("Simulation complete.")
    print(
        f"Saved data to: {OUTPUT_FILE}"
    )

    # --------------------------------------------------------
    # Write per-realization CSV
    # --------------------------------------------------------
    columns = []

    for N in range(N_MIN, N_MAX + 1):
        columns.append(f"power_{N}")
        columns.append(f"norm_comm_{N}")
        columns.append(f"bound_{N}")

    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(per_realization_rows)

    print(f"Saved per-realization CSV to: {CSV_FILE}")

    # ========================================================
    # FIGURE 1: N SCALING
    # ========================================================
    N_array = np.array(
        sorted(
            map(int, data.keys())
        ),
        dtype=float
    )

    tight = np.array([
        data[str(int(N))]["tight"]
        for N in N_array
    ])

    tight_sem = np.array([
        data[str(int(N))]["tight_sem"]
        for N in N_array
    ])

    exact_N = np.array([
        N
        for N in N_array
        if data[str(int(N))]["exact_norm"] is not None
    ])

    exact = np.array([
        data[str(int(N))]["exact_norm"]
        for N in exact_N
    ])

    exact_sem = np.array([
        data[str(int(N))]["exact_norm_sem"]
        for N in exact_N
    ])

    power_N = np.array([
        N
        for N in N_array
        if data[str(int(N))]["power_star"] is not None
    ])

    power_star = np.array([
        data[str(int(N))]["power_star"]
        for N in power_N
    ])

    power_star_sem = np.array([
        data[str(int(N))]["power_star_sem"]
        for N in power_N
    ])

    # Create the main figure
    fig = plt.figure(figsize=(6.5, 4.2))
    ax = fig.add_subplot(111)

    # Main plot: only the tight bound across all N
    ax.plot(
        N_array,
        tight,
        linewidth=1.5,
        label=r"$\|[H_b,H_c]\|$ tight bound"
    )

    ax.fill_between(
        N_array,
        np.maximum(tight - tight_sem, 1e-300),
        tight + tight_sem,
        alpha=0.20
    )

    ax.set_xlabel(r"$N$")
    ax.set_ylabel(r"$\|[H_b,H_c]\|$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower right", ncol=3)
    plt.tight_layout()

    # Add a zoomed inset for N = 2 .. EXACT_N_MAX
    # Position: [left, bottom, width, height] in figure coordinates
    inset_ax = fig.add_axes([0.15, 0.65, 0.35, 0.3])

    # Select tight bound data for N <= EXACT_N_MAX
    mask = N_array <= EXACT_N_MAX
    N_inset = N_array[mask]
    tight_inset = tight[mask]
    tight_sem_inset = tight_sem[mask]

    inset_ax.plot(
        N_inset,
        tight_inset,
        linewidth=1.5,
        color="C0",
        label=r"tight bound"
    )

    inset_ax.fill_between(
        N_inset,
        np.maximum(tight_inset - tight_sem_inset, 1e-300),
        tight_inset + tight_sem_inset,
        alpha=0.20,
        color="C0"
    )

    # Exact commutator norm
    if len(exact_N) > 0:
        inset_ax.errorbar(
            exact_N,
            exact,
            yerr=exact_sem,
            fmt="o",
            markersize=4,
            capsize=2,
            label=r"exact norm",
            color="C1"
        )

    # Maximum power
    if len(power_N) > 0:
        inset_ax.errorbar(
            power_N,
            power_star,
            yerr=power_star_sem,
            fmt="s",
            markersize=4,
            capsize=2,
            label=r"$P_*$",
            color="C2"
        )

    # Formatting the inset axes
    inset_ax.set_xlabel(r"$N$", fontsize=8)
    # inset_ax.set_ylabel("Value", fontsize=8)
    inset_ax.set_xscale("log")
    inset_ax.set_yscale("log")
    inset_ax.set_xlim(2, EXACT_N_MAX)
    inset_ax.grid(True, which="both", alpha=0.25)
    inset_ax.legend(fontsize=7, loc="upper left")

    # Set integer tick locations on the x-axis
    tick_positions = list(range(2, EXACT_N_MAX + 1))
    inset_ax.set_xticks(tick_positions)

    inset_ax.set_xticklabels(
        [str(int(tick)) for tick in tick_positions],
        fontsize=8   # same font size as main plot's tick labels
    )

    inset_ax.tick_params(axis='y', labelsize=8)

    # ---- Manual zoom indication with SOLID lines ----
    # Determine the y-limits for the zoom region (same as inset)
    all_vals = np.concatenate([tight_inset, exact, power_star])

    y_zoom_min = np.min(all_vals) * 0.5   # padding below
    y_zoom_max = np.max(all_vals) * 2.0   # padding above

    # Set the inset axes limits to match the rectangle exactly
    inset_ax.set_ylim(y_zoom_min, y_zoom_max)

    # Draw rectangle on the main axes (solid line)
    rect = Rectangle(
        (2, y_zoom_min),
        EXACT_N_MAX - 2,
        y_zoom_max - y_zoom_min,
        fill=False,
        edgecolor="black",
        linestyle="-",           # solid line
        linewidth=1.0,
        zorder=5
    )

    ax.add_patch(rect)

    # Draw connection lines from top vertices of rectangle
    # to bottom vertices of inset axes (solid lines)
    # Left connection: (2, y_zoom_max) in main -> (2, y_zoom_min) in inset
    con_left = ConnectionPatch(
        xyA=(2, y_zoom_max),          # top-left vertex of rectangle
        xyB=(2, y_zoom_min),          # bottom-left vertex of inset
        coordsA="data",
        coordsB="data",
        axesA=ax,
        axesB=inset_ax,
        color="black",
        linestyle="-",                # solid line
        linewidth=1.0,
        clip_on=False,
        zorder=6
    )

    fig.add_artist(con_left)

    # Right connection: (EXACT_N_MAX, y_zoom_max) -> (EXACT_N_MAX, y_zoom_min)
    con_right = ConnectionPatch(
        xyA=(EXACT_N_MAX, y_zoom_max), # top-right vertex of rectangle
        xyB=(EXACT_N_MAX, y_zoom_min), # bottom-right vertex of inset
        coordsA="data",
        coordsB="data",
        axesA=ax,
        axesB=inset_ax,
        color="black",
        linestyle="-",                # solid line
        linewidth=1.0,
        clip_on=False,
        zorder=6
    )

    fig.add_artist(con_right)

    plt.savefig(
        FIGURE_FILE,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    # ========================================================
    # FIGURE 2: <P(t)> / N
    # ========================================================
    plt.figure(
        figsize=(6.5, 4.2)
    )

    for N in POWER_PLOT_N_VALUES:
        key = str(N)

        if (
            key not in data
            or data[key]["power_per_N_vs_time"] is None
        ):
            continue

        t = np.asarray(
            data[key]["power_times"],
            dtype=float
        )

        p = np.asarray(
            data[key]["power_per_N_vs_time"],
            dtype=float
        )

        p_sem = np.asarray(
            data[key]["power_per_N_vs_time_sem"],
            dtype=float
        )

        plt.plot(
            t,
            p,
            linewidth=1.5,
            label=fr"$N={N}$"
        )

        plt.fill_between(
            t,
            p - p_sem,
            p + p_sem,
            alpha=0.15
        )

    plt.axhline(
        0.0,
        linewidth=0.8
    )

    plt.xlabel(
        r"$t$"
    )

    plt.ylabel(
        r"$\langle P(t)\rangle_{\mathrm{dis}}/N$"
    )

    plt.grid(
        True,
        alpha=0.25
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        POWER_FIGURE_FILE,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(
        f"Saved power-vs-time figure to: "
        f"{POWER_FIGURE_FILE}"
    )