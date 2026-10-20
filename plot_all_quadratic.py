import glob
import csv
import re
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Rectangle, ConnectionPatch
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple

# ============================================================
# Global plot settings
# ============================================================
plt.rcParams.update({
    'pgf.texsystem': 'pdflatex',
    'text.usetex': True,
    'font.family': 'serif',
    'pgf.preamble': '\n'.join([r'\usepackage{amsmath}']),
    'text.latex.preamble': '\n'.join([r'\usepackage{amsmath}']),
})

# ============================================================
# Figure with two vertically stacked panels sharing x
# ============================================================
fig, axes = plt.subplot_mosaic(
    [['a'], ['b']],
    figsize=(5, 6),
    gridspec_kw={'height_ratios': [3, 2], 
                 'hspace': 0.0},
    sharex=True,
)
ax  = axes['a']
ax2 = axes['b']

labels_abc = ['(a)', '(b)', '(c)', '(d)']

# ============================================================
# ====================== PANEL (a) ===========================
# ============================================================
FILE_PATTERN = "data_per_realization_*.csv"
OUTPUT_FIGURE = "./img/all_quadratic.pdf"

# -------------------------------
# Helper: mean and SEM
# -------------------------------
def mean_and_sem(values):
    vals = np.asarray(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return np.nan, np.nan
    mean = np.mean(vals)
    if len(vals) > 1:
        std = np.std(vals, ddof=1)
        sem = std / np.sqrt(len(vals))
    else:
        std = 0.0
        sem = 0.0
    return mean, sem

# -------------------------------
# Helper: format exponent with uncertainty
# -------------------------------
def format_exponent(b, sb):
    if not np.isfinite(b) or not np.isfinite(sb) or sb <= 0:
        return f"{b:.2f}"
    exp = int(np.floor(np.log10(sb)))
    digit = int(round(sb / 10**exp))
    if digit == 10:
        exp += 1
        digit = 1
    decimal_places = max(0, -exp)
    b_rounded = round(b, decimal_places)
    sb_rounded = round(sb, decimal_places)
    b_str = f"{b_rounded:.{decimal_places}f}"
    sb_digit = digit
    return f"{b_str}({sb_digit})"

# -------------------------------
# Read all CSV files
# -------------------------------
csv_files = sorted(glob.glob(FILE_PATTERN))
if not csv_files:
    raise FileNotFoundError(f"No files matching {FILE_PATTERN} found.")

data_dict = {}
all_N = None

for filepath in csv_files:
    basename = filepath.split('/')[-1]
    label = basename[len("data_per_realization_"):-len(".csv")]

    with open(filepath, 'r', newline='') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        rows = list(reader)

    N_set = set()
    pattern = re.compile(r"^(power|norm_comm|bound)_(\d+)$")
    for h in headers:
        m = pattern.match(h)
        if m:
            N_set.add(int(m.group(2)))

    N_list = sorted(N_set)
    if all_N is None:
        all_N = set(N_list)
    else:
        all_N &= set(N_list)

    N_array = np.array(N_list, dtype=float)

    bound_mean = np.full(len(N_array), np.nan)
    bound_sem  = np.full(len(N_array), np.nan)
    exact_mean = np.full(len(N_array), np.nan)
    exact_sem  = np.full(len(N_array), np.nan)
    power_mean = np.full(len(N_array), np.nan)
    power_sem  = np.full(len(N_array), np.nan)

    for i, N in enumerate(N_list):
        bound_col = f"bound_{N}"
        norm_col  = f"norm_comm_{N}"
        power_col = f"power_{N}"

        bound_vals = [float(row[bound_col]) for row in rows]
        m, s = mean_and_sem(bound_vals)
        bound_mean[i] = m
        bound_sem[i]  = s

        norm_vals = []
        for row in rows:
            val = row[norm_col]
            if val not in ('', 'NaN', 'nan', 'None', None):
                norm_vals.append(float(val))
        if norm_vals:
            m, s = mean_and_sem(norm_vals)
            exact_mean[i] = m
            exact_sem[i]  = s

        power_vals = []
        for row in rows:
            val = row[power_col]
            if val not in ('', 'NaN', 'nan', 'None', None):
                power_vals.append(float(val))
        if power_vals:
            m, s = mean_and_sem(power_vals)
            power_mean[i] = m
            power_sem[i]  = s

    data_dict[label] = {
        'N': N_array,
        'bound_mean': bound_mean, 'bound_sem': bound_sem,
        'exact_mean': exact_mean, 'exact_sem': exact_sem,
        'power_mean': power_mean, 'power_sem': power_sem,
    }

N_min = min(all_N)
N_max = max(all_N)

N_inset_max = N_min
for label, d in data_dict.items():
    valid_exact_N = d['N'][~np.isnan(d['exact_mean'])]
    if len(valid_exact_N) > 0:
        N_inset_max = max(N_inset_max, int(valid_exact_N.max()))

# -------------------------------
# Power-law fitting
# -------------------------------
fit_colors = ['coral', 'chartreuse', 'mediumturquoise'][::-1]
fit_results = {}

for idx, (label, d) in enumerate(data_dict.items()):
    N = d['N']
    bound = d['bound_mean']

    if label.lower() == "global":
        nL_val = N_max
    else:
        try:
            nL_val = int(label)
        except ValueError:
            nL_val = N_max

    if nL_val == 2:
        fit_lower = 10
    elif 2 < nL_val < N_max:
        fit_lower = nL_val
    elif nL_val == N_max:
        fit_lower = 10
    else:
        fit_lower = 10

    fit_upper = min(N_max, 1000)

    fit_mask = (N >= fit_lower) & (N <= fit_upper) & (~np.isnan(bound))
    N_fit = N[fit_mask]
    bound_fit = bound[fit_mask]

    if len(N_fit) >= 2:
        coeffs, cov = np.polyfit(np.log10(N_fit), np.log10(bound_fit), 1, cov=True)
        b = coeffs[0]
        log_a = coeffs[1]
        a = 10 ** log_a
        sb = np.sqrt(cov[0, 0])
        fit_results[label] = (a, b, sb)
    else:
        fit_results[label] = (np.nan, np.nan, np.nan)

# -------------------------------
# Panel (a) label
# -------------------------------
ax.annotate(
    labels_abc[0],
    xy=(1, 0), xycoords='axes fraction',
    xytext=(-1.8, +3.5), textcoords='offset fontsize',
    fontsize='medium',
    verticalalignment='top',
    bbox=dict(facecolor='none', edgecolor='none', pad=3.0)
)

raw_colors = ['firebrick', 'seagreen', 'mediumblue'][::-1]
text_positions_by_idx = [(300, 800), (300, 4e4), (300, 1.5e6)]

# -------------------------------
# Main plot
# -------------------------------
for idx, (label, d) in enumerate(data_dict.items()):
    color = raw_colors[idx]
    N = d['N']
    bound = d['bound_mean']
    sem = d['bound_sem']

    label_str = f"$n_L$={str(int(label))}" if label != "global" else "$n_L=N$"
    ax.plot(N, bound, linewidth=2, color=color, label=label_str)
    ax.fill_between(
        N,
        np.maximum(bound - sem, 1e-300),
        bound + sem,
        alpha=0.15,
        color=color
    )

    a, b, sb = fit_results.get(label, (np.nan, np.nan, np.nan))
    if np.isfinite(a) and np.isfinite(b) and np.isfinite(sb):
        fit_color = fit_colors[idx % len(fit_colors)]
        if idx != 1:
            N_fit_plot = np.logspace(np.log10(10), np.log10(fit_upper), 200)
            fitted = a * N_fit_plot ** b
            ax.plot(N_fit_plot, fitted, linestyle='-', linewidth=1,
                    color=fit_color, label=None)
        else:
            N_fit_plot = np.logspace(np.log10(10), np.log10(fit_upper), 200)
            N_fit_plot = N_fit_plot[N_fit_plot >= 100]
            fitted = a * N_fit_plot ** b
            ax.plot(N_fit_plot, fitted, linestyle='-', linewidth=1,
                    color=fit_color, label=None)

        if idx < len(text_positions_by_idx):
            x_text, y_text = text_positions_by_idx[idx]
        else:
            x_text = N_fit_plot[-1] * 1.02 - 400
            y_text = a * N_fit_plot[-1] ** b

        exponent_text = format_exponent(b, sb)
        ax.text(x_text, y_text, rf"$\sim N^{{{exponent_text}}}$",
                color='k', fontsize=9, ha='left', va='center')

ax.set_ylabel(r"$\mathcal{U}_{n_L}(N)$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(2, 1000)
ax.set_ylim(5, 4e6)
ax.grid(True, which="both", alpha=0.25)
ax.legend(loc="lower right", ncol=3)

# -------------------------------
# Inset (in axes coordinates of panel (a))
# -------------------------------
inset_ax = ax.inset_axes([0.06, 0.57, 0.49, 0.42])

inset_x_min = 2
inset_x_max = 16

for idx, (label, d) in enumerate(data_dict.items()):
    color = raw_colors[idx]
    N = d['N']
    mask = (N >= inset_x_min) & (N <= inset_x_max)

    N_inset     = N[mask]
    bound_inset = d['bound_mean'][mask]
    sem_inset   = d['bound_sem'][mask]

    inset_ax.plot(N_inset, bound_inset, linewidth=1.5, color=color)
    inset_ax.fill_between(
        N_inset,
        np.maximum(bound_inset - sem_inset, 1e-300),
        bound_inset + sem_inset,
        alpha=0.15, color=color
    )

    exact_mask = mask & (~np.isnan(d['exact_mean']))
    if np.any(exact_mask):
        N_exact    = N[exact_mask]
        exact_vals = d['exact_mean'][exact_mask]
        exact_sems = d['exact_sem'][exact_mask]

        inset_ax.plot(N_exact, exact_vals, linestyle='--', linewidth=1.5, color=color)
        inset_ax.fill_between(
            N_exact,
            np.maximum(exact_vals - exact_sems, 1e-300),
            exact_vals + exact_sems,
            alpha=0.15, color=color
        )

    power_mask = mask & (~np.isnan(d['power_mean']))
    if np.any(power_mask):
        inset_ax.errorbar(
            N[power_mask],
            d['power_mean'][power_mask],
            yerr=d['power_sem'][power_mask],
            fmt='o', markersize=4, capsize=2,
            color=color, linestyle='None'
        )

    inset_ax.plot([1, 17], [1, 17], 'k--', linewidth=0.75)
    inset_ax.plot([1, 17], 1.7*np.array([1, 17**1.5]), 'k--', linewidth=0.75)
    inset_ax.plot([1, 17], 2*np.array([1, 17**2]), 'k--', linewidth=0.75)
    inset_ax.text(11, 7, r'$\sim N$', fontsize=8)
    inset_ax.text(10, 90, r'$\sim N^{3/2}$', fontsize=8)
    inset_ax.text(5.2, 100, r'$\sim N^2$', fontsize=8)

bound_line  = Line2D([0], [0], color='black', lw=1.5,
                     label=r'$\mathcal{U}_{n_L}(N)$')
norm_line   = Line2D([0], [0], color='black', lw=1.5, linestyle='--',
                     label=r'$\|[H_\text{b},H_\text{c}]\|$')
power_marker = Line2D([0], [0], color='black', marker='o', markersize=4,
                      linestyle='None')

generic_handles = [bound_line, norm_line, power_marker]
generic_labels  = [r'$\mathcal{U}_{n_L}(N)$',
                   r'$\|[H_\text{b},H_\text{c}]\|$',
                   r'$P_*$']

inset_ax.legend(
    generic_handles,
    generic_labels,
    handler_map={tuple: HandlerTuple(ndivide=None)},
    fontsize=6,
    loc='upper left'
)

inset_ax.set_xscale("log")
inset_ax.set_xlim(inset_x_min, inset_x_max)
inset_ax.set_yscale("log")
inset_ax.grid(True, which="both", alpha=0.25)
inset_ax.tick_params(axis='both', which='both', direction='in', labelsize=8)

even_ticks = list(range(2, 17))
inset_ax.xaxis.set_major_locator(ticker.FixedLocator(even_ticks))
inset_ax.xaxis.set_major_formatter(
    ticker.FuncFormatter(lambda x, pos: str(int(x)) if int(x) % 2 == 0 else '')
)

for label in inset_ax.xaxis.get_minorticklabels():
    label.set_visible(False)

all_vals = []
for d in data_dict.values():
    mask = (d['N'] >= inset_x_min) & (d['N'] <= inset_x_max)
    all_vals.extend(d['bound_mean'][mask][~np.isnan(d['bound_mean'][mask])])
    ex = d['exact_mean'][mask][~np.isnan(d['exact_mean'][mask])]
    pw = d['power_mean'][mask][~np.isnan(d['power_mean'][mask])]
    all_vals.extend(ex)
    all_vals.extend(pw)

if all_vals:
    y_zoom_min = 3
    y_zoom_max = 400
    inset_ax.set_ylim(y_zoom_min, y_zoom_max)

    rect = Rectangle(
        (inset_x_min, y_zoom_min),
        inset_x_max - inset_x_min,
        y_zoom_max - y_zoom_min,
        fill=True, facecolor="lightgray", edgecolor="gray",
        linestyle="-", linewidth=1.0, zorder=-1, alpha=0.6
    )
    # ax.add_patch(rect)

# ============================================================
# ====================== PANEL (b) ===========================
# ============================================================
class HandlerTupleCentered(HandlerTuple):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        handler_map = legend.get_legend_handler_map()
        center = ydescent
        dy = 4
        artists = []
        for i, handle in enumerate(orig_handle):
            handler = legend.get_legend_handler(handler_map, handle)
            y = center + (-dy if i == 0 else dy)
            artists.extend(handler.create_artists(
                legend, handle, xdescent, y, width, height, fontsize, trans
            ))
        return artists

nL_settings = [2, 100, 1000]
csv_pattern = "disorder_realizations_nL{}.csv"

colors  = ['firebrick', 'seagreen', 'mediumblue'][::-1]
colors2 = ['r', 'g', 'b'][::-1]

line_comm_list = []
line_p_list = []
labels_b = []

for idx, nL in enumerate(nL_settings):
    csv_file = csv_pattern.format(nL)
    try:
        with open(csv_file, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = [row for row in reader]
    except FileNotFoundError:
        print(f"Warning: {csv_file} not found, skipping nL={nL}")
        continue

    N_sorted = []
    for col in header:
        m = re.match(r"P_N(\d+)", col)
        if m:
            N_sorted.append(int(m.group(1)))
    N_sorted = sorted(N_sorted)

    data = np.array([[float(x) for x in row] for row in rows])
    col_index = {col: i for i, col in enumerate(header)}

    P_mean, P_min, P_max = [], [], []
    comm_mean, comm_min, comm_max = [], [], []

    for N in N_sorted:
        p_col = f"P_N{N}"
        c_col = f"comm_N{N}"
        p_vals = data[:, col_index[p_col]]
        c_vals = data[:, col_index[c_col]]

        P_mean.append(np.mean(p_vals))
        P_min.append(np.min(p_vals))
        P_max.append(np.max(p_vals))

        comm_mean.append(np.mean(c_vals))
        comm_min.append(np.min(c_vals))
        comm_max.append(np.max(c_vals))

    N_array = np.array(N_sorted, dtype=float)
    P_mean = np.array(P_mean); P_min = np.array(P_min); P_max = np.array(P_max)
    comm_mean = np.array(comm_mean)
    comm_min = np.array(comm_min); comm_max = np.array(comm_max)

    color  = colors[idx]
    color2 = colors2[idx]

    l_comm, = ax2.plot(N_array, comm_mean, color=color, linestyle="-")
    ax2.fill_between(N_array, comm_min, comm_max, color=color2, alpha=0.1)

    l_p, = ax2.plot(N_array, P_mean, color=color, linestyle="--")
    ax2.fill_between(N_array, P_min, P_max, color=color2, alpha=0.1)

    line_comm_list.append(l_comm)
    line_p_list.append(l_p)
    labels_b.append(f"$n_L = {nL}$")

# Reference line ~ N
ref_line, = ax2.plot([2, 1000], [2/2, 1000/2], "k--")
ax2.text(300, 100, r'$\sim N$')

# Legend with one entry per nL (solid + dashed pair)
handles = [(lc, lp) for lc, lp in zip(line_comm_list, line_p_list)]
# ax2.legend(
#     handles,
#     labels_b,
#     handler_map={tuple: HandlerTupleCentered()},
#     loc="upper left",
# )

ax2.set_xlabel(r"$N$")
ax2.set_ylabel(r"$\lVert[H_\text{b},H_\text{c}]\rVert$, $P_*$")
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlim(2, 1000)
ax2.set_ylim(1, 2e3)
ax2.grid(True, which="both", alpha=0.25)

ax2.annotate(
    labels_abc[1],
    xy=(1, 0), xycoords='axes fraction',
    xytext=(-1.8, +1.4), textcoords='offset fontsize',
    fontsize='medium',
    verticalalignment='top',
    bbox=dict(facecolor='none', edgecolor='none', pad=3.0)
)

# ============================================================
# Save
# ============================================================
plt.savefig(OUTPUT_FIGURE, dpi=300, bbox_inches="tight")
plt.show()

print(f"Combined figure saved to: {OUTPUT_FIGURE}")