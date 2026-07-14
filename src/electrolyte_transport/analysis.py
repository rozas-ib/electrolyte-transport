#!/usr/bin/env python3
"""Vectorized Helfand--Einstein transport analysis for MD trajectories.

The module evaluates charge-weighted conductivity terms with multi-time-origin
averaging, arbitrary atomic or centre-of-mass species, charge restoration,
optional MDAnalysis PBC transformations, and replica averaging. It also reports
species and cross terms, Onsager/diffusion-like diagnostics, and binary
transference-number estimates. The expensive lag/origin accumulation is
vectorized with NumPy and chunked to limit memory use.

Use :func:`run_conductivity_analysis` from Python or the
``electrolyte-transport`` command with a TOML configuration file.
"""

import argparse
import datetime as _dt
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive, fast for saving PNGs
import matplotlib.pyplot as plt
from tqdm import tqdm

import tomllib
import MDAnalysis as mda

import warnings
#from Bio import BiopythonDeprecationWarning
#
#warnings.filterwarnings(
#    "ignore",
#    message="The Bio.Application modules and modules relying on it have been deprecated.*",
#    category=BiopythonDeprecationWarning,
#)


warnings.filterwarnings(
    "ignore",
    message=r"NoJump detected that the interval between frames is unequal.*",
    category=UserWarning,
)

warnings.filterwarnings(
    "ignore",
    message=r"NoJump transform is only accurate when positions.*half a box length.*",
    category=UserWarning,
)

# Physical constants
kB = 1.380649e-23          # J/K
e_charge = 1.602176634e-19 # Coulomb


# ----------------------------
# Logging: "tee" stdout to file
# ----------------------------
class TeeStdout:
    def __init__(self, filepath: str):
        self.file = open(filepath, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, msg: str):
        self.stdout.write(msg)
        self.file.write(msg)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


# ----------------------------
# Config utilities
# ----------------------------
def load_toml(path: str) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_template_config(path: str):
    template = """[files]
# Backward-compatible single trajectory:
# topology = "system.tpr"
# trajectory = "traj.xtc"

# Recommended for replicas with the same topology:
topology = "system.tpr"
trajectories = ["replica1.xtc", "replica2.xtc", "replica3.xtc"]

# Alternative: one topology per replica:
# topologies = ["replica1.tpr", "replica2.tpr", "replica3.tpr"]
# trajectories = ["replica1.xtc", "replica2.xtc", "replica3.xtc"]

# Alternative: explicit replicas outside this [files] table:
# [[replicas]]
# label = "rep1"
# topology = "replica1.tpr"
# trajectory = "replica1.xtc"

[system]
temperature_K = 300.0
verbose = true

[trajectory]
start = 0        # frame index
stop  = -1       # -1 means last frame
stride = 1       # read every Nth frame

[pbc]
# "none", "unwrap", "nojump", "unwrap+nojump"
mode = "nojump"
# If true, apply "unwrap" first (make molecules whole), then nojump.
make_molecules_whole = false

[charges]
# Choose ONE approach (comment the other out)

# Option A: multiply charges by this factor
charge_multiplicative_factor = 1.0

# Option B: charges were rescaled by this factor in the simulation
# charge_rescaled_by = 0.8

[ions]
species = [
    # atomic ions: keep atomic
    { name = "Li", selection = "name Li", mode = "atomic" },

    # polyatomic anion: COM mode (group molecules by residue)
    { name = "FSI", selection = "resname FSI", mode = "com", groupby = "residue" }
]

[analysis]
# Manual fit window in LAG TIME tau (ps). This same window is used for
# conductivity, species terms, Onsager diffusion-like terms, cross terms,
# and transport/transference numbers.
fit_start_ps = 50000.0
fit_end_ps = 200000.0

# Sliding window (ps) for local slope plot
local_slope_window_ps = 25000.0

# Multi-origin settings
origin_stride_frames = 5
max_lag_ps = 200000.0

# How to average replicas: "counts" weights by number of time origins per lag;
# "replica" gives each replica equal weight.
replica_average = "counts"

# Chunk size for origins (controls memory use; smaller = less memory, slower)
origins_chunk = 200

# For transference numbers (binary electrolyte only)
cation_name = "Li"
anion_name  = "FSI"

[output]
prefix = "HE"

# Time axis for plots: "ps", "ns", or "s"
time_unit = "ns"

# Conductivity units: "S/m", "S/cm", or "mS/cm"
conductivity_unit = "mS/cm"

# Also save the individual replica time-series CSVs.
save_replica_csv = true

# Also write a dedicated PNG with Onsager/diffusion-like diagnostics.
save_diffusion_plot = true

# Unit for diffusion coefficients in the diffusion diagnostic plot:
# "m2/s", "cm2/s", or "10^-10 m2/s".
diffusion_unit = "m2/s"

# Rolling log-log window, in lag points, for the diffusion diagnostic plot.
# Must be odd; even values are rounded up internally.
diffusion_alpha_window_points = 21
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(template)


# ----------------------------
# Charge restoration + reporting
# ----------------------------
def restore_charges_C(atomgroup, charge_cfg: dict, verbose: bool) -> Tuple[np.ndarray, str]:
    q = atomgroup.charges.copy()  # usually in e, depends on topology

    if "charge_multiplicative_factor" in charge_cfg:
        factor = float(charge_cfg["charge_multiplicative_factor"])
        desc = f"q_restored = q_sim * {factor}"
        if verbose:
            print(f"[charges] Using multiplicative factor: {desc}")
        q = q * factor

    elif "charge_rescaled_by" in charge_cfg:
        s = float(charge_cfg["charge_rescaled_by"])
        desc = f"q_restored = q_sim / {s}  (because q_sim = q_phys * {s})"
        if verbose:
            print(f"[charges] Using rescaled_by factor: {desc}")
        q = q / s

    else:
        desc = "q_restored = q_sim (no change)"
        if verbose:
            print("[charges] No restoration specified: using raw charges from topology")

    return q * e_charge, desc


def report_total_charge(atomgroup, qC_restored, qdesc: str, verbose: bool):
    q_raw_e = atomgroup.charges.copy()
    q_rest_e = qC_restored / e_charge

    raw_sum_e = float(np.sum(q_raw_e))
    rest_sum_e = float(np.sum(q_rest_e))
    rest_sum_C = float(np.sum(qC_restored))

    if verbose:
        print(f"    total charge (raw)      : {raw_sum_e:+.6f} e")
        print(f"    total charge (restored) : {rest_sum_e:+.6f} e  ({rest_sum_C:+.6e} C)")
        print(f"    restoration rule        : {qdesc}")

def build_species_entities(u, sp_cfg, charge_cfg, verbose):
    """
    Return (name, mode, positions_getter, charges_C)

    positions_getter(ts) must return positions for the chosen entities at this frame
    with shape (N_entities, 3) in Å.

    charges_C must be (N_entities,) in Coulomb, consistent with positions_getter.
    """
    name = sp_cfg["name"]
    sel = sp_cfg["selection"]
    mode = sp_cfg.get("mode", "atomic").lower()

    ag = u.select_atoms(sel)
    if ag.n_atoms == 0:
        raise ValueError(f"[ions] Species '{name}' selection '{sel}' matched 0 atoms.")

    # Atomic mode: entities are atoms
    if mode == "atomic":
        qC, qdesc = restore_charges_C(ag, charge_cfg, verbose)
        if verbose:
            print(f"  - {name}: mode=atomic, selection='{sel}', atoms={ag.n_atoms}")
            report_total_charge(ag, qC, qdesc, verbose=True)

        def get_positions(_ts):
            return ag.positions.copy()  # (Natoms, 3)

        return name, mode, get_positions, qC, ag

    # COM mode: entities are molecules (typically residues)
    if mode == "com":
        groupby = sp_cfg.get("groupby", "residue").lower()
        if groupby not in ("residue", "segment"):
            raise ValueError(f"[ions] Species '{name}': invalid groupby='{groupby}' (use 'residue' or 'segment').")

        if groupby == "residue":
            # list of residues (each residue is a molecule)
            residues = ag.residues
            if len(residues) == 0:
                raise ValueError(f"[ions] Species '{name}': selection has no residues.")
            # charges per residue: sum atomic charges in that residue, then restore, then convert to C
            # easiest is: take residue atomgroup charges, apply restoration rule consistently
            # We'll do it by restoring per-atom charges first, then summing per residue.
            qC_atoms, qdesc = restore_charges_C(ag, charge_cfg, verbose)
            # map atom charges to residues via residue indices
            # residues.atoms is the same ag, but order consistent; use residue indices per atom:

            residues = ag.residues
            n_res = len(residues)
            qC_res = np.zeros(n_res, dtype=float)
            
            # global atom index -> local index inside 'ag'
            global_to_local = {idx: i for i, idx in enumerate(ag.atoms.indices)}
            
            for r_idx, res in enumerate(residues):
                s = 0.0
                for idx in res.atoms.indices:
                    s += qC_atoms[global_to_local[idx]]
                qC_res[r_idx] = s
            

            if verbose:
                # raw and restored totals at residue level
                raw_sum_e = float(np.sum(ag.charges))
                rest_sum_C = float(np.sum(qC_res))
                print(f"  - {name}: mode=com(residue), selection='{sel}', residues={n_res}, atoms={ag.n_atoms}")
                print(f"    total charge (raw, atoms)      : {raw_sum_e:+.6f} e")
                print(f"    total charge (restored, residues) : {(rest_sum_C/e_charge):+.6f} e  ({rest_sum_C:+.6e} C)")
                print(f"    restoration rule              : {qdesc}")
                # show residue charge stats (useful sanity check)
                q_res_e = qC_res / e_charge
                print(f"    residue charges (restored, e): min={q_res_e.min():+.6f}, max={q_res_e.max():+.6f}")

            def get_positions(_ts):
                # COM per residue; MDAnalysis returns (n_res, 3) for compound='residues'
                return ag.center_of_mass(compound="residues")

            return name, mode, get_positions, qC_res, ag

        if groupby == "segment":
            segments = ag.segments
            qC_atoms, qdesc = restore_charges_C(ag, charge_cfg, verbose)
            seg_indices = ag.atoms.segindices
            n_seg = len(segments)
            qC_seg = np.zeros(n_seg, dtype=float)
            for i_atom in range(ag.n_atoms):
                qC_seg[seg_indices[i_atom]] += qC_atoms[i_atom]

            if verbose:
                print(f"  - {name}: mode=com(segment), selection='{sel}', segments={n_seg}, atoms={ag.n_atoms}")
                print(f"    restoration rule: {qdesc}")

            def get_positions(_ts):
                return ag.center_of_mass(compound="segments")

            return name, mode, get_positions, qC_seg, ag

    raise ValueError(f"[ions] Species '{name}': unknown mode '{mode}'")

        

# ----------------------------
# PBC handling (MDAnalysis transformations)
# ----------------------------
def setup_pbc_transformations(u: mda.Universe, cfg: dict, verbose: bool):
    pbc = cfg.get("pbc", {})
    mode = pbc.get("mode", "none").lower()
    make_whole = bool(pbc.get("make_molecules_whole", False))

    if mode == "none":
        if verbose:
            print("[pbc] No PBC transformation applied (assumes already unwrapped trajectory).")
        return "none"

    transformations = []

    if mode in ("unwrap", "unwrap+nojump") or make_whole:
        from MDAnalysis.transformations.wrap import unwrap
        transformations.append(unwrap(u.atoms))
        if verbose:
            print("[pbc] Added transformation: unwrap(u.atoms)")

    if mode in ("nojump", "unwrap+nojump"):
        from MDAnalysis.transformations.nojump import NoJump
        transformations.append(NoJump())
        if verbose:
            print("[pbc] Added transformation: NoJump()")

    u.trajectory.add_transformations(*transformations)
    return mode


# ----------------------------
# Fit + diagnostics
# ----------------------------
@dataclass
class FitResult:
    slope: float
    intercept: float
    r2: float
    t_fit: np.ndarray
    y_fit: np.ndarray
    y_pred: np.ndarray
    residuals: np.ndarray


def scale_time_for_plot(tau_ps: np.ndarray, tau_s: np.ndarray, unit: str):
    u = unit.strip().lower()
    if u == "ps":
        return tau_ps, "lag time τ (ps)"
    if u == "ns":
        return tau_ps * 1e-3, "lag time τ (ns)"
    if u == "s":
        return tau_s, "lag time τ (s)"
    raise ValueError(f"Unknown time unit: {unit}")


def linear_fit_with_r2(t: np.ndarray, y: np.ndarray) -> FitResult:
    coeff = np.polyfit(t, y, 1)
    slope, intercept = float(coeff[0]), float(coeff[1])
    y_pred = slope * t + intercept
    residuals = y - y_pred

    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - np.mean(y))**2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return FitResult(slope, intercept, r2, t, y, y_pred, residuals)



def compute_local_slopes(times_s: np.ndarray, values: np.ndarray, window_s: float) -> Tuple[np.ndarray, np.ndarray]:
    t_centers = []
    slopes = []
    n = len(times_s)

    for i in range(n):
        t0 = times_s[i]
        t1 = t0 + window_s
        mask = (times_s >= t0) & (times_s <= t1)
        if np.sum(mask) < 3:
            continue
        fit = linear_fit_with_r2(times_s[mask], values[mask])
        t_centers.append(0.5 * (times_s[mask][0] + times_s[mask][-1]))
        slopes.append(fit.slope)

    return np.array(t_centers), np.array(slopes)


def make_diagnostic_plot(
    out_png: str,
    tau_s: np.ndarray,
    total: np.ndarray,
    selfv: np.ndarray,
    distinct: np.ndarray,
    fit_total: FitResult,
    fit_window_s: Tuple[float, float],
    local_t: np.ndarray,
    fit_x_plot: np.ndarray,
    local_slope: np.ndarray,
    x_label: str
):
    t0, t1 = fit_window_s
    fig = plt.figure(figsize=(10, 10))

    #e_charge = 1.602176634e-19 # Coulomb
    s_to_ns = 1e-9
    scale = 0.01 / (e_charge**2) # 0.01 is for going from angstrom2 to nm2
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(tau_s, total*scale, label="TOTAL  M(τ)=<|G|²>")
    ax1.plot(tau_s, selfv*scale, label="SELF   <Σ q² |Δr|²>")
    ax1.plot(tau_s, distinct*scale, label="DISTINCT = TOTAL - SELF")
    ax1.axvspan(t0, t1, alpha=0.2, label="fit window")
    ax1.set_xlabel(x_label)
    ax1.set_ylabel("M(τ) (e²·nm²)")
    ax1.legend()
    ax1.set_title("Helfand–Einstein diagnostics")

    ax2 = fig.add_subplot(3, 1, 2)
    ax2.plot(fit_x_plot, fit_total.residuals*scale)
    ax2.axhline(0.0)
    ax2.set_xlabel(x_label)
    ax2.set_ylabel("residuals (e²·nm²)")
    ax2.set_title("Linear-fit residuals")

    ax3 = fig.add_subplot(3, 1, 3)
    ax3.set_yscale('log')
    ax3.plot(local_t[:-1], local_slope[:-1]*scale*s_to_ns)
    ax3.axvspan(t0, t1, alpha=0.2)
    ax3.set_xlabel(x_label)
    ax3.set_ylabel("local slope (e²·nm² / ns)")
    ax3.set_title("Local slope (sliding-window)")

    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# ----------------------------
# Output helpers
# ----------------------------
def save_csv(path: str, data: Dict[str, np.ndarray]):
    keys = list(data.keys())
    n = len(data[keys[0]])
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for i in range(n):
            f.write(",".join(str(data[k][i]) for k in keys) + "\n")


# ----------------------------
# Conductivity and transference
# ----------------------------
def conductivity_from_slope(slope_C2A2_per_s: float, V_m3: float, T_K: float) -> float:
    slope_C2m2_per_s = slope_C2A2_per_s * (1e-10 ** 2)  # Å^2 -> m^2
    return (1.0 / (6.0 * V_m3 * kB * T_K)) * slope_C2m2_per_s

def convert_conductivity(value_S_per_m: float, unit: str) -> float:
    """
    Convert conductivity from S/m to requested unit.
    """
    u = unit.strip().lower()
    if u == "s/m":
        return value_S_per_m
    if u == "s/cm":
        return value_S_per_m * 0.01
    if u == "ms/cm":
        return value_S_per_m * 10.0
    raise ValueError(f"Unknown conductivity unit: {unit}")

def compute_transference_numbers(
    names: List[str],
    tau_s: np.ndarray,
    mask: np.ndarray,
    ts_data: Dict[str, np.ndarray],
    V_m3: float,
    T_K: float,
    cation_name: str,
    anion_name: str,
) -> Dict[str, float]:
    """
    Compute two cation transference/transport numbers for a binary electrolyte:

    1) t_plus_NE  : Nernst–Einstein (self-only) transference number
    2) rho_plus   : correlation-aware bulk current fraction (from slopes of charge-displacement terms)

    Requirements:
      - species indices for cation_name and anion_name exist in `names`
      - ts_data contains:
          species_{i}_self, species_{i}_G2, and cross_{i}_{j} (for i<j)

    Notes:
      - Uses the SAME fit window (mask) as conductivity.
      - Works whether the cation is represented atomically or by COM entities.
    """
    if cation_name not in names:
        raise ValueError(f"[transference] cation_name '{cation_name}' not found in species names: {names}")
    if anion_name not in names:
        raise ValueError(f"[transference] anion_name '{anion_name}' not found in species names: {names}")

    ip = names.index(cation_name)
    im = names.index(anion_name)

    # ---- NE transference number: based on self terms only ----
    # fit slopes of species self Helfand curves
    fit_self_p = linear_fit_with_r2(tau_s[mask], ts_data[f"species_{ip}_self"][mask])
    fit_self_m = linear_fit_with_r2(tau_s[mask], ts_data[f"species_{im}_self"][mask])

    sigma_self_p = conductivity_from_slope(fit_self_p.slope, V_m3, T_K)  # S/m
    sigma_self_m = conductivity_from_slope(fit_self_m.slope, V_m3, T_K)  # S/m

    denom_ne = sigma_self_p + sigma_self_m
    t_plus_ne = sigma_self_p / denom_ne if denom_ne != 0 else np.nan

    # ---- Correlation-aware bulk current fraction rho+ ----
    # Need slopes of <|G_p|^2>, <|G_m|^2>, and 2< G_p·G_m >
    fit_g2_p = linear_fit_with_r2(tau_s[mask], ts_data[f"species_{ip}_G2"][mask])
    fit_g2_m = linear_fit_with_r2(tau_s[mask], ts_data[f"species_{im}_G2"][mask])

    # cross key uses smaller index first
    a, b = (ip, im) if ip < im else (im, ip)
    cross_key = f"cross_{a}_{b}"
    if cross_key not in ts_data:
        raise ValueError(f"[transference] Missing cross term '{cross_key}' in ts_data.")
    fit_cross2 = linear_fit_with_r2(tau_s[mask], ts_data[cross_key][mask])  # this is slope of 2<dot>

    slope_pp = fit_g2_p.slope
    slope_mm = fit_g2_m.slope
    slope_pm = 0.5 * fit_cross2.slope  # because your series is 2<dot>

    slope_tot = slope_pp + slope_mm + 2.0 * slope_pm
    slope_p_tot = slope_pp + slope_pm

    rho_plus = slope_p_tot / slope_tot if slope_tot != 0 else np.nan

    # Some helpful debugging values (optional)
    return {
        "t_plus_NE": float(t_plus_ne),
        "rho_plus": float(rho_plus),
        "sigma_self_plus_S_per_m": float(sigma_self_p),
        "sigma_self_minus_S_per_m": float(sigma_self_m),
        "cross_key": cross_key,
    }


# ----------------------------
# Data collection (positions)
# ----------------------------
def collect_positions(trajectory, pos_getters: List, verbose: bool, nojump_checker=None, com_drift_checker=None):
    """
    pos_getters: list of functions get_positions(ts)->(N_entities,3)

    Returns:
      time_ps (F,)
      vols_A3 (F,)
      pos_by_species: list of arrays, each (F, N_entities, 3)
    """
    time_ps = []
    vols_A3 = []
    pos_by_species = [ [] for _ in pos_getters ]
    
    if verbose:
        print("\n[collect] Reading trajectory and storing positions for selected entities...")
        print(f"[collect] Frames: {len(trajectory)}")

    for ts in tqdm(trajectory, total=len(trajectory), desc="[collect] frames"):
        time_ps.append(float(ts.time))
        vols_A3.append(float(ts.volume))

        com_drift_checker.update(ts, trajectory)
        
        for a, get_pos in enumerate(pos_getters):
            pos_by_species[a].append(np.array(get_pos(ts), copy=True))

        if nojump_checker is not None:
            curr_pos_list = [pos_by_species[a][-1] for a in range(len(pos_by_species))]
            #nojump_checker.update(ts, curr_pos_list)
   
    com_drift_checker.finalize()
        
    time_ps = np.array(time_ps)
    vols_A3 = np.array(vols_A3)
    pos_arrays = [np.array(lst) for lst in pos_by_species]

    if verbose:
        dt_ps = np.median(np.diff(time_ps)) if len(time_ps) > 1 else np.nan
        print(f"[collect] Trajectory time: {len(trajectory)*dt_ps/1000} ns")
        print(f"[collect] Frames read: {len(trajectory)}")
        print(f"[collect] Done. Median frame spacing: {dt_ps:.3f} ps")

    if verbose and nojump_checker is not None:
        nojump_checker.report()
        
    return time_ps, vols_A3, pos_arrays

class SystemCOMDriftChecker:
    """
    Stateful checker for system COM drift.
    Accumulates COM(t) on the fly and checks for linear drift at the end.
    If drift exceeds threshold, writes COM-vs-time plot and raises RuntimeError.
    """

    def __init__(self, u, drift_threshold_nm_per_ns=1e-2, out_prefix="system_COM"):
        self.universe = u
        self.times_ps = []
        self.com_A = []
        self.drift_threshold = float(drift_threshold_nm_per_ns)
        self.out_prefix = out_prefix

    def update(self, ts, traj):
        """Call once per stored frame."""
        self.times_ps.append(float(ts.time))
        self.com_A.append(self.universe.atoms.center_of_mass())

    def finalize(self):
        """Check drift and fail loudly if needed."""
        time_ps = np.asarray(self.times_ps, dtype=float)
    
        if time_ps.size < 2:
            print("[drift] Skipping COM drift check (less than 2 frames).")
            return
    
        # Force proper 2D shape (F,3)
        com_A = np.vstack(self.com_A).astype(float)
    
        # Fit COM(t) = v*t + b for each axis
        v = np.zeros(3)
        for k in range(3):
            v[k] = np.polyfit(time_ps, com_A[:, k], 1)[0]  # Å/ps
    
        # Convert Å/ps -> nm/ns
        v_mag_nm_per_ns = np.linalg.norm(v) * 100.0
    
        print(f"[drift] System COM drift velocity = {v_mag_nm_per_ns:.3e} nm/ns")
    
        if v_mag_nm_per_ns <= self.drift_threshold:
            return  # PASS
    
        # ---- Drift is unacceptable: write diagnostic plot ----
        import matplotlib.pyplot as plt
    
        t_ns = time_ps * 1e-3
        com_nm = com_A * 0.1
    
        plt.figure(figsize=(6, 4))
        plt.plot(t_ns, com_nm[:, 0], label="COM x")
        plt.plot(t_ns, com_nm[:, 1], label="COM y")
        plt.plot(t_ns, com_nm[:, 2], label="COM z")
        plt.xlabel("time (ns)")
        plt.ylabel("COM position (nm)")
        plt.title("System COM drift")
        plt.legend()
        plt.tight_layout()
    
        fname = f"{self.out_prefix}_vs_time.png"
        plt.savefig(fname, dpi=150)
        plt.close()
    
        raise RuntimeError(
            f"[drift] FATAL: system COM drift detected.\n"
            f"[drift] |v| = {v_mag_nm_per_ns:.3e} nm/ns "
            f"(threshold {self.drift_threshold:.1e} nm/ns)\n"
            f"[drift] Diagnostic plot written to '{fname}'.\n"
            f"[drift] ACTION: remove COM motion in GROMACS (comm-mode), "
            f"reduce trajectory stride, or fix the simulation."
        )


class NoJumpStrideChecker:
    """
    Checks whether displacement between consecutive *stored* frames exceeds ~half the box length.
    This is a diagnostic for whether NoJump may become unreliable when using trajectory stride > 1.
    """

    def __init__(self, frac_half_box: float = 0.5):
        self.frac_half_box = float(frac_half_box)
        self.prev_pos_by_species = None  # list of arrays (N_a, 3)
        self.max_jump_seen = 0.0
        self.exceed_count = 0
        self.frames_checked = 0
        self.last_threshold = None

    def update(self, ts, curr_pos_by_species: List[np.ndarray]):
        """
        ts: MDAnalysis timestep (provides dimensions)
        curr_pos_by_species: list of arrays, each (N_a, 3) in Å
        """
        # Box lengths in Å
        Lx, Ly, Lz = ts.dimensions[:3]
        Lmin = min(float(Lx), float(Ly), float(Lz))
        threshold = self.frac_half_box * Lmin
        self.last_threshold = threshold

        if self.prev_pos_by_species is not None:
            for a in range(len(curr_pos_by_species)):
                dr = curr_pos_by_species[a] - self.prev_pos_by_species[a]  # (N, 3)
                # max displacement magnitude (Å)
                max_dr = float(np.max(np.linalg.norm(dr, axis=1)))
                if max_dr > self.max_jump_seen:
                    self.max_jump_seen = max_dr
                if max_dr > threshold:
                    self.exceed_count += 1
            self.frames_checked += 1

        # store a copy so later modifications won't affect it
        self.prev_pos_by_species = [p.copy() for p in curr_pos_by_species]

    def report(self, prefix: str = "[nojump-check]"):
        if self.frames_checked == 0:
            print(f"{prefix} Not enough frames to check displacements.")
            return

        thr = self.last_threshold if self.last_threshold is not None else float("nan")
        print(f"{prefix} Max displacement between stored frames: {self.max_jump_seen:.3f} Å")
        print(f"{prefix} Exceedances (> {thr:.3f} Å = {self.frac_half_box:.2f}*Lmin): {self.exceed_count}")
        if self.exceed_count > 0:
            raise RuntimeError(
                f"{prefix} FATAL: NoJump reliability violated.\n"
                f"{prefix} Detected {self.exceed_count} frame/species exceedances.\n"
                f"{prefix} Max displacement between stored frames: {self.max_jump_seen:.3f} Å\n"
                f"{prefix} Threshold = {self.frac_half_box:.2f} * Lmin = {thr:.3f} Å\n"
                f"{prefix} This means atoms moved more than half the box length between frames.\n"
                f"{prefix} Results would be physically incorrect.\n"
                f"{prefix} ACTION: reduce trajectory stride, or apply NoJump on the full trajectory "
                f"and subsample only after unwrapping."
            )

# ----------------------------
# Vectorized multi-origin accumulation (chunked)
# ----------------------------
def accumulate_multi_origin_vectorized(
    time_ps: np.ndarray,
    pos_by_species: List[np.ndarray],        # species -> (F, N_a, 3) in Å
    q_by_species_C: List[np.ndarray],        # species -> (N_a,) in C
    origin_stride_frames: int,
    max_lag_ps: float,
    origins_chunk: int,
    verbose: bool,
) -> Dict[str, np.ndarray]:
    F = len(time_ps)
    if F < 2:
        raise ValueError("Need at least 2 frames.")

    dt_ps = float(np.median(np.diff(time_ps)))
    if dt_ps <= 0:
        raise ValueError("Non-positive dt detected.")

    max_lag_frames = int(np.floor(max_lag_ps / dt_ps))
    max_lag_frames = min(max_lag_frames, F - 1)
    if max_lag_frames < 1:
        raise ValueError("max_lag_ps too small.")

    nsp = len(pos_by_species)

    if verbose:
        print("\n[multi-origin] Vectorized accumulation")
        print(f"[multi-origin] nframes = {F}")
        print(f"[multi-origin] dt_ps = {dt_ps:.3f} ps")
        print(f"[multi-origin] origin_stride_frames = {origin_stride_frames}")
        print(f"[multi-origin] max_lag_frames = {max_lag_frames}")
        print(f"[multi-origin] origins_chunk = {origins_chunk}")

    # Lag axis and counts
    tau_ps = np.zeros(max_lag_frames + 1)
    counts = np.zeros(max_lag_frames + 1, dtype=int)

    total = np.zeros(max_lag_frames + 1)
    selfv = np.zeros(max_lag_frames + 1)
    distinct = np.zeros(max_lag_frames + 1)

    species_G2 = [np.zeros(max_lag_frames + 1) for _ in range(nsp)]
    species_self = [np.zeros(max_lag_frames + 1) for _ in range(nsp)]
    species_distinct = [np.zeros(max_lag_frames + 1) for _ in range(nsp)]

    # Uncharged Onsager/Helfand quantities:
    # R_a = sum_i dr_i, not charge-weighted.
    ons_species_R2 = [np.zeros(max_lag_frames + 1) for _ in range(nsp)]
    ons_species_self = [np.zeros(max_lag_frames + 1) for _ in range(nsp)]
    ons_species_distinct = [np.zeros(max_lag_frames + 1) for _ in range(nsp)]

    pair_keys = []
    cross = {}
    ons_cross = {}
    for a in range(nsp):
        for b in range(a + 1, nsp):
            key = f"cross_{a}_{b}"
            pair_keys.append(key)
            cross[key] = np.zeros(max_lag_frames + 1)

            # Uncharged cross terms: <R_a · R_b>, not 2<R_a · R_b>
            ons_key = f"ons_cross_{a}_{b}"
            ons_cross[ons_key] = np.zeros(max_lag_frames + 1)
    # Main lag loop (vectorized inside)
    for d in tqdm(range(1, max_lag_frames + 1), desc="[multi-origin] lags"):
        # origins indices with stride
        origins = np.arange(0, F - d, origin_stride_frames, dtype=int)
        n_orig = len(origins)
        if n_orig == 0:
            continue

        # tau for this lag: average actual time differences for used origins
        tau_ps[d] = float(np.mean(time_ps[origins + d] - time_ps[origins]))
        counts[d] = n_orig

        # We will accumulate sums over origins, then divide by n_orig at the end
        total_sum = 0.0
        self_sum = 0.0
        distinct_sum = 0.0

        sp_g2_sum = [0.0 for _ in range(nsp)]
        sp_self_sum = [0.0 for _ in range(nsp)]
        sp_dist_sum = [0.0 for _ in range(nsp)]
        cross_sum = {k: 0.0 for k in pair_keys}
        ons_R2_sum = [0.0 for _ in range(nsp)]
        ons_self_sum = [0.0 for _ in range(nsp)]
        ons_dist_sum = [0.0 for _ in range(nsp)]
        ons_cross_sum = {f"ons_cross_{a}_{b}": 0.0
                         for a in range(nsp)
                         for b in range(a + 1, nsp)}

        # Chunk over origins to limit memory use
        for start in range(0, n_orig, origins_chunk):
            o_chunk = origins[start:start + origins_chunk]
            # For each species, compute dr for this chunk and build G_species and self_species
            Gsp = []
            Selfsp = []
            Rsp = []
            OnsSelfsp = []

            for a in range(nsp):
                pos = pos_by_species[a]      # (F, N_a, 3)
                q = q_by_species_C[a]        # (N_a,)
                q2 = q * q

                # dr: (C, N_a, 3)
                dr = pos[o_chunk + d] - pos[o_chunk]

                # G_a: sum_i q_i * dr_i  -> (C, 3)
                # Using einsum to keep it clear: 'i, c i k -> c k'
                G_a = np.einsum("i,cik->ck", q, dr)

                # self_a: sum_i q_i^2 * |dr_i|^2 -> (C,)
                dr2 = np.sum(dr * dr, axis=2)               # (C, N_a)
                self_a = np.einsum("i,ci->c", q2, dr2)       # (C,)
                # Uncharged species Helfand displacement:
                # R_a = sum_i dr_i
                R_a = np.sum(dr, axis=1)                     # (C, 3)

                # Uncharged self part:
                # sum_i |dr_i|^2
                ons_self_a = np.sum(dr2, axis=1)             # (C,)

                Rsp.append(R_a)
                OnsSelfsp.append(ons_self_a)

                R2_a = np.sum(R_a * R_a, axis=1)             # (C,)
                ons_R2_sum[a] += float(np.sum(R2_a))
                ons_self_sum[a] += float(np.sum(ons_self_a))
                ons_dist_sum[a] += float(np.sum(R2_a - ons_self_a))

                Gsp.append(G_a)
                Selfsp.append(self_a)

                # species totals for this chunk
                g2a = np.sum(G_a * G_a, axis=1)             # (C,)
                sp_g2_sum[a] += float(np.sum(g2a))
                sp_self_sum[a] += float(np.sum(self_a))
                sp_dist_sum[a] += float(np.sum(g2a - self_a))

            # Total G for this chunk: sum over species
            Gtot = np.zeros_like(Gsp[0])
            for a in range(nsp):
                Gtot += Gsp[a]

            g2tot = np.sum(Gtot * Gtot, axis=1)             # (C,)
            self_chunk = np.zeros_like(Selfsp[0])
            for a in range(nsp):
                self_chunk += Selfsp[a]

            total_sum += float(np.sum(g2tot))
            self_sum += float(np.sum(self_chunk))
            distinct_sum += float(np.sum(g2tot - self_chunk))

            # Cross terms between species: 2 * (G_a · G_b)
            for a in range(nsp):
                for b in range(a + 1, nsp):
                    key = f"cross_{a}_{b}"
                    dotab = np.sum(Gsp[a] * Gsp[b], axis=1)   # (C,)
                    cross_sum[key] += float(np.sum(2.0 * dotab))

            # Uncharged Onsager cross terms: R_a · R_b
            for a in range(nsp):
                for b in range(a + 1, nsp):
                    key = f"ons_cross_{a}_{b}"
                    dotab = np.sum(Rsp[a] * Rsp[b], axis=1)   # (C,)
                    ons_cross_sum[key] += float(np.sum(dotab))

        # Convert sums to averages
        total[d] = total_sum / n_orig
        selfv[d] = self_sum / n_orig
        distinct[d] = distinct_sum / n_orig

        for a in range(nsp):
            species_G2[a][d] = sp_g2_sum[a] / n_orig
            species_self[a][d] = sp_self_sum[a] / n_orig
            species_distinct[a][d] = sp_dist_sum[a] / n_orig
        
        for a in range(nsp):
            ons_species_R2[a][d] = ons_R2_sum[a] / n_orig
            ons_species_self[a][d] = ons_self_sum[a] / n_orig
            ons_species_distinct[a][d] = ons_dist_sum[a] / n_orig

        for key in pair_keys:
            cross[key][d] = cross_sum[key] / n_orig
        
        for key in ons_cross_sum:
            ons_cross[key][d] = ons_cross_sum[key] / n_orig

    out = {
        "tau_ps": tau_ps,
        "tau_s": tau_ps * 1e-12,
        "counts": counts.astype(float),
        "total_G2": total,
        "self_G2": selfv,
        "distinct_G2": distinct,
    }
    for a in range(nsp):
        out[f"species_{a}_G2"] = species_G2[a]
        out[f"species_{a}_self"] = species_self[a]
        out[f"species_{a}_distinct"] = species_distinct[a]

    for a in range(nsp):
        out[f"ons_species_{a}_R2"] = ons_species_R2[a]
        out[f"ons_species_{a}_self"] = ons_species_self[a]
        out[f"ons_species_{a}_distinct"] = ons_species_distinct[a]

    for key in pair_keys:
        out[key] = cross[key]
    for key, val in ons_cross.items():
        out[key] = val
    return out

def onsager_from_slope_A2_per_s(slope_A2_per_s: float, V_m3: float, T_K: float) -> float:
    """
    Convert slope of <R_i · R_j> from Å²/s to Onsager coefficient:

        L_ij = slope / (6 V kB T)

    where slope is first converted from Å²/s to m²/s.
    """
    slope_m2_per_s = slope_A2_per_s * 1e-20
    return slope_m2_per_s / (6.0 * V_m3 * kB * T_K)


def diffusion_like_from_slope_A2_per_s(slope_A2_per_s: float, N_norm: int) -> float:
    """
    Convert Helfand slope to a diffusion-like coefficient in m²/s:

        D_ij^(norm by i) = slope_ij / (6 N_i)

    This is useful for reporting, but use L_ij for transference numbers.
    """
    return slope_A2_per_s * 1e-20 / (6.0 * float(N_norm))




def convert_diffusion_m2_s(value_m2_s: float, unit: str) -> float:
    """Convert a diffusion coefficient from m²/s to the requested display unit."""
    u = unit.strip().lower().replace(" ", "")
    if u in ("m2/s", "m^2/s"):
        return value_m2_s
    if u in ("cm2/s", "cm^2/s"):
        return value_m2_s * 1e4
    if u in ("10^-10m2/s", "1e-10m2/s", "1e-10m^2/s"):
        return value_m2_s / 1e-10
    raise ValueError("Unsupported diffusion_unit. Use 'm2/s', 'cm2/s', or '10^-10 m2/s'.")


def diffusion_unit_label(unit: str) -> str:
    u = unit.strip().lower().replace(" ", "")
    if u in ("m2/s", "m^2/s"):
        return "m²/s"
    if u in ("cm2/s", "cm^2/s"):
        return "cm²/s"
    if u in ("10^-10m2/s", "1e-10m2/s", "1e-10m^2/s"):
        return "10⁻¹⁰ m²/s"
    return unit


def ensure_odd_window(window: int, minimum: int = 5) -> int:
    window = int(window)
    if window < minimum:
        window = minimum
    if window % 2 == 0:
        window += 1
    return window


def rolling_mean_same(y: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling mean with same-length output, ignoring NaNs inside each window."""
    y = np.asarray(y, dtype=float)
    if window <= 1 or y.size == 0:
        return y.copy()
    window = ensure_odd_window(window, minimum=3)
    pad = window // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    out = np.empty_like(y, dtype=float)
    for i in range(y.size):
        seg = ypad[i:i + window]
        out[i] = np.nanmean(seg)
    return out


def rolling_rms_same(y: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling RMS with same-length output, ignoring NaNs inside each window."""
    y = np.asarray(y, dtype=float)
    if window <= 1 or y.size == 0:
        return np.abs(y).copy()
    window = ensure_odd_window(window, minimum=3)
    pad = window // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    out = np.empty_like(y, dtype=float)
    for i in range(y.size):
        seg = ypad[i:i + window]
        out[i] = np.sqrt(np.nanmean(seg * seg))
    return out


def local_loglog_alpha_points(tau_s: np.ndarray, y: np.ndarray, window_points: int) -> np.ndarray:
    """
    Rolling local log-log slope alpha = d log(y) / d log(tau).
    Non-positive or non-finite windows are returned as NaN.
    """
    tau_s = np.asarray(tau_s, dtype=float)
    y = np.asarray(y, dtype=float)
    window_points = ensure_odd_window(window_points, minimum=5)
    half = window_points // 2
    out = np.full_like(tau_s, np.nan, dtype=float)
    valid = np.isfinite(tau_s) & np.isfinite(y) & (tau_s > 0.0) & (y > 0.0)

    for i in range(half, len(tau_s) - half):
        idx = slice(i - half, i + half + 1)
        if not np.all(valid[idx]):
            continue
        fit = linear_fit_with_r2(np.log(tau_s[idx]), np.log(y[idx]))
        out[i] = fit.slope

    return out


def make_diffusion_diagnostic_plot(
    out_png: str,
    names: List[str],
    tau_s: np.ndarray,
    mask: np.ndarray,
    ts_data: Dict[str, np.ndarray],
    diffusion_results: Dict[str, object],
    analysis: dict,
    out_cfg: dict,
):
    """
    Write a separate PNG summarizing the uncharged Onsager/diffusion-like analysis.

    The layout follows the spirit of the self-diffusion diagnostic plot:
    curves + fit region, log-log slope diagnostics, MSD/tau-like stability,
    residuals, and final coefficient summaries. No automatic fit window is used;
    every fit shown here is the shared manual TOML fit window.
    """
    tau_s = np.asarray(tau_s, dtype=float)
    tau_ns = tau_s * 1e9
    tau_ps = tau_s * 1e12
    fit_start_ns = float(tau_ns[mask][0]) if np.any(mask) else np.nan
    fit_end_ns = float(tau_ns[mask][-1]) if np.any(mask) else np.nan

    diff_unit = str(out_cfg.get("diffusion_unit", "m2/s"))
    diff_label = diffusion_unit_label(diff_unit)
    alpha_window = ensure_odd_window(int(out_cfg.get("diffusion_alpha_window_points", 21)), minimum=5)
    local_window_s = float(analysis.get("local_slope_window_ps", 25000.0)) * 1e-12
    smooth_window = ensure_odd_window(int(out_cfg.get("diffusion_smooth_window_points", alpha_window)), minimum=3)

    species_results = diffusion_results.get("species", [])
    cross_results = diffusion_results.get("cross", [])

    fig, axs = plt.subplots(4, 2, figsize=(15, 18))
    axs = axs.flatten()

    # ------------------------------------------------------------------
    # (1) Diagonal uncharged Helfand curves: total, self, distinct
    # ------------------------------------------------------------------
    ax = axs[0]
    for item in species_results:
        a = int(item["index"])
        name = str(item["name"])
        N = float(item["N"])
        curves = [
            ("total", ts_data[f"ons_species_{a}_R2"], item["fit_total"]),
            ("self", ts_data[f"ons_species_{a}_self"], item["fit_self"]),
            ("distinct", ts_data[f"ons_species_{a}_distinct"], item["fit_dist"]),
        ]
        for term, y_A2, fit in curves:
            y_norm_nm2 = (np.asarray(y_A2, dtype=float) / N) * 0.01
            ax.plot(tau_ns, y_norm_nm2, lw=1.1, label=f"{name} {term}")
            yfit_nm2 = ((fit.slope * tau_s + fit.intercept) / N) * 0.01
            ax.plot(tau_ns[mask], yfit_nm2[mask], "--", lw=1.0, alpha=0.85)
    ax.axvspan(fit_start_ns, fit_end_ns, alpha=0.08, label="fit window")
    ax.set_xlabel("lag time (ns)")
    ax.set_ylabel("normalized curve (nm² / entity)")
    ax.set_title("Diagonal diffusion-like curves and manual-window fits")
    ax.legend(loc="best", fontsize=8, frameon=True)

    # ------------------------------------------------------------------
    # (2) Log-log self MSD-like curves with slope=1 references
    # ------------------------------------------------------------------
    ax = axs[1]
    for item in species_results:
        a = int(item["index"])
        name = str(item["name"])
        N = float(item["N"])
        y_self = np.asarray(ts_data[f"ons_species_{a}_self"], dtype=float) / N
        positive = np.isfinite(y_self) & (y_self > 0.0) & np.isfinite(tau_ns) & (tau_ns > 0.0)
        if np.count_nonzero(positive) < 3:
            continue
        y_nm2 = y_self * 0.01
        ax.loglog(tau_ns[positive], y_nm2[positive], lw=1.4, label=f"{name} self")
        fit_positive = positive & mask
        if np.count_nonzero(fit_positive) >= 2:
            ax.loglog(tau_ns[fit_positive], y_nm2[fit_positive], "o", ms=3)
            ref = tau_s * (np.nanmedian(y_nm2[fit_positive] / tau_s[fit_positive]))
            ref_positive = np.isfinite(ref) & (ref > 0.0) & positive
            ax.loglog(tau_ns[ref_positive], ref[ref_positive], "--", lw=1.0, alpha=0.55)
    ax.set_xlabel("lag time (ns)")
    ax.set_ylabel("self MSD-like curve (nm² / entity)")
    ax.set_title("Log-log self curves with slope=1 references")
    ax.legend(loc="best", fontsize=8, frameon=True)

    # ------------------------------------------------------------------
    # (3) Rolling local alpha for self curves
    # ------------------------------------------------------------------
    ax = axs[2]
    for item in species_results:
        a = int(item["index"])
        name = str(item["name"])
        y_self = ts_data[f"ons_species_{a}_self"]
        alpha = local_loglog_alpha_points(tau_s, y_self, alpha_window)
        ax.plot(tau_ns, alpha, lw=1.4, label=f"{name} self")
    ax.axhline(1.0, ls="--", alpha=0.8)
    ax.axvspan(fit_start_ns, fit_end_ns, alpha=0.08)
    ax.set_xlabel("lag time (ns)")
    ax.set_ylabel("local log-log slope α")
    ax.set_title(f"Rolling self α, window={alpha_window} lag points")
    ax.legend(loc="best", fontsize=8, frameon=True)

    # ------------------------------------------------------------------
    # (4) Local/running D_self from sliding linear windows
    # ------------------------------------------------------------------
    ax = axs[3]
    for item in species_results:
        a = int(item["index"])
        name = str(item["name"])
        N = int(item["N"])
        y_self = ts_data[f"ons_species_{a}_self"]
        tloc, slopes = compute_local_slopes(tau_s, y_self, local_window_s)
        if slopes.size == 0:
            continue
        dloc = np.array([convert_diffusion_m2_s(diffusion_like_from_slope_A2_per_s(s, N), diff_unit) for s in slopes])
        ax.plot(tloc * 1e9, dloc, lw=1.3, label=f"{name} local")
        ax.axhline(convert_diffusion_m2_s(item["D_self"], diff_unit), ls="--", lw=1.0, alpha=0.75)
    ax.axvspan(fit_start_ns, fit_end_ns, alpha=0.08)
    ax.set_xlabel("lag time (ns)")
    ax.set_ylabel(f"D_self ({diff_label})")
    ax.set_title("Local self diffusion coefficients")
    ax.legend(loc="best", fontsize=8, frameon=True)

    # ------------------------------------------------------------------
    # (5) Self residuals and rolling RMS residuals
    # ------------------------------------------------------------------
    ax = axs[4]
    for item in species_results:
        a = int(item["index"])
        name = str(item["name"])
        N = float(item["N"])
        fit = item["fit_self"]
        y_self = np.asarray(ts_data[f"ons_species_{a}_self"], dtype=float)
        resid_nm2 = ((y_self - (fit.slope * tau_s + fit.intercept)) / N) * 0.01
        ax.plot(tau_ns, resid_nm2, lw=1.0, alpha=0.8, label=f"{name} residual")
        ax.plot(tau_ns, rolling_rms_same(resid_nm2, smooth_window), lw=1.2, alpha=0.8, label=f"{name} RMS")
    ax.axhline(0.0, lw=1.0)
    ax.axvspan(fit_start_ns, fit_end_ns, alpha=0.08)
    ax.set_xlabel("lag time (ns)")
    ax.set_ylabel("self residual (nm² / entity)")
    ax.set_title("Self linear-fit residuals")
    ax.legend(loc="best", fontsize=8, frameon=True)

    # ------------------------------------------------------------------
    # (6) Final D_total / D_self / D_distinct bars
    # ------------------------------------------------------------------
    ax = axs[5]
    labels = []
    values = []
    for item in species_results:
        name = str(item["name"])
        for term, key in (("total", "D_total"), ("self", "D_self"), ("distinct", "D_dist")):
            labels.append(f"{name}\n{term}")
            values.append(convert_diffusion_m2_s(item[key], diff_unit))
    if labels:
        x = np.arange(len(labels))
        ax.bar(x, values)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.axhline(0.0, lw=1.0)
    ax.set_ylabel(f"D ({diff_label})")
    ax.set_title("Manual-window diffusion-like coefficients")

    # ------------------------------------------------------------------
    # (7) Cross Onsager curves and fits
    # ------------------------------------------------------------------
    ax = axs[6]
    if cross_results:
        for item in cross_results:
            a = int(item["a"])
            b = int(item["b"])
            name = f"{names[a]}×{names[b]}"
            N_cat = float(item["N_a"])
            N_an = float(item["N_b"])
            key = str(item["key"])
            y_cross = np.asarray(ts_data[key], dtype=float)
            norm = np.sqrt(N_cat * N_an) if N_cat > 0 and N_an > 0 else 1.0
            y_norm_nm2 = (y_cross / norm) * 0.01
            fit = item["fit_cross"]
            yfit_nm2 = ((fit.slope * tau_s + fit.intercept) / norm) * 0.01
            ax.plot(tau_ns, y_norm_nm2, lw=1.2, label=f"{name} cross")
            ax.plot(tau_ns[mask], yfit_nm2[mask], "--", lw=1.0, alpha=0.85)
        ax.axvspan(fit_start_ns, fit_end_ns, alpha=0.08)
        ax.axhline(0.0, lw=1.0)
        ax.legend(loc="best", fontsize=8, frameon=True)
    else:
        ax.text(0.5, 0.5, "No cross terms\n(single species)", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("lag time (ns)")
    ax.set_ylabel("cross curve (nm² / √entities)")
    ax.set_title("Uncharged cross terms and manual-window fits")

    # ------------------------------------------------------------------
    # (8) Summary text: D values, R², transport/transference
    # ------------------------------------------------------------------
    ax = axs[7]
    ax.axis("off")
    lines = [
        "Diffusion/Onsager summary",
        f"fit window: {tau_ps[mask][0]:.6g}–{tau_ps[mask][-1]:.6g} ps" if np.any(mask) else "fit window: n/a",
        f"D unit: {diff_label}",
        "",
    ]
    for item in species_results:
        lines.append(
            f"{item['name']}: Dself={convert_diffusion_m2_s(item['D_self'], diff_unit):.4g}, "
            f"Dtot={convert_diffusion_m2_s(item['D_total'], diff_unit):.4g}, "
            f"Ddist={convert_diffusion_m2_s(item['D_dist'], diff_unit):.4g}; "
            f"R² self={item['fit_self'].r2:.4f}"
        )
    if cross_results:
        lines.append("")
        for item in cross_results:
            lines.append(
                f"{names[item['a']]}×{names[item['b']]}: "
                f"Dcross/{names[item['a']]}={convert_diffusion_m2_s(item['D_ab_norm_a'], diff_unit):.4g}, "
                f"Dcross/{names[item['b']]}={convert_diffusion_m2_s(item['D_ab_norm_b'], diff_unit):.4g}; "
                f"R²={item['fit_cross'].r2:.4f}"
            )
    tinfo = diffusion_results.get("transference", None)
    if tinfo is not None:
        lines.extend([
            "",
            f"t+ Onsager/correlation-aware = {tinfo['t_plus']:.6f}",
            f"z+={tinfo['z_p']:+.4f}, z-={tinfo['z_m']:+.4f}",
        ])
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=9)

    fig.suptitle("Onsager / diffusion-like diagnostics (manual shared fit window)")
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def report_onsager_diffusion_terms(
    names: List[str],
    q_by_species_C: List[np.ndarray],
    tau_s: np.ndarray,
    mask: np.ndarray,
    ts_data: Dict[str, np.ndarray],
    V_m3: float,
    T_K: float,
    cation_name: str = None,
    anion_name: str = None,
):
    """
    Reports uncharged Onsager coefficients and diffusion-like
    self/distinct/cross terms.

    The SAME manual fit window (mask) is used for every diagonal and cross
    term so that the Onsager matrix and the derived transference number are
    internally consistent.

    Requires the ons_* time series added in accumulate_multi_origin_vectorized().
    """
    nsp = len(names)
    L = np.zeros((nsp, nsp), dtype=float)
    species_results = []
    cross_results = []
    transference = None

    print("\n=== ONSAGER / DIFFUSION-LIKE TERMS ===")
    print("Uncharged Helfand form: L_ij = slope(<R_i · R_j>) / (6 V kB T)")
    print("Diffusion-like values are reported in m²/s.")
    print("All Onsager and diffusion-like terms use the same manual fit window.")
    print("Use L_ij, not the normalized D-like values, for general transference-number calculations.")

    # Diagonal self/distinct/total terms
    for a, name in enumerate(names):
        N_a = len(q_by_species_C[a])

        y_total = ts_data[f"ons_species_{a}_R2"]
        y_self = ts_data[f"ons_species_{a}_self"]
        y_dist = ts_data[f"ons_species_{a}_distinct"]

        fit_total = linear_fit_with_r2(tau_s[mask], y_total[mask])
        fit_self = linear_fit_with_r2(tau_s[mask], y_self[mask])
        fit_dist = linear_fit_with_r2(tau_s[mask], y_dist[mask])

        L_total = onsager_from_slope_A2_per_s(fit_total.slope, V_m3, T_K)
        L_self = onsager_from_slope_A2_per_s(fit_self.slope, V_m3, T_K)
        L_dist = onsager_from_slope_A2_per_s(fit_dist.slope, V_m3, T_K)

        D_total = diffusion_like_from_slope_A2_per_s(fit_total.slope, N_a)
        D_self = diffusion_like_from_slope_A2_per_s(fit_self.slope, N_a)
        D_dist = diffusion_like_from_slope_A2_per_s(fit_dist.slope, N_a)

        L[a, a] = L_total

        species_results.append({
            "index": a,
            "name": name,
            "N": int(N_a),
            "fit_total": fit_total,
            "fit_self": fit_self,
            "fit_dist": fit_dist,
            "L_total": float(L_total),
            "L_self": float(L_self),
            "L_dist": float(L_dist),
            "D_total": float(D_total),
            "D_self": float(D_self),
            "D_dist": float(D_dist),
        })

        print(f"\n  [{a}] {name}")
        print(f"    N entities                  : {N_a}")
        print(f"    L_{name}{name} total         : {L_total:.6e}")
        print(f"    L_{name}{name} self          : {L_self:.6e}")
        print(f"    L_{name}{name} distinct      : {L_dist:.6e}")
        print(f"    D_like total                : {D_total:.6e} m²/s")
        print(f"    D_self / tracer             : {D_self:.6e} m²/s")
        print(f"    D_distinct                  : {D_dist:.6e} m²/s")
        print(f"    R² total/self/distinct       : {fit_total.r2:.5f} / {fit_self.r2:.5f} / {fit_dist.r2:.5f}")

    # Cross terms
    if nsp >= 2:
        print("\n=== ONSAGER CROSS TERMS ===")
        for a in range(nsp):
            for b in range(a + 1, nsp):
                key = f"ons_cross_{a}_{b}"
                if key not in ts_data:
                    continue

                fit_cross = linear_fit_with_r2(tau_s[mask], ts_data[key][mask])
                L_ab = onsager_from_slope_A2_per_s(fit_cross.slope, V_m3, T_K)

                N_a = len(q_by_species_C[a])
                N_b = len(q_by_species_C[b])

                D_ab_norm_a = diffusion_like_from_slope_A2_per_s(fit_cross.slope, N_a)
                D_ab_norm_b = diffusion_like_from_slope_A2_per_s(fit_cross.slope, N_b)

                L[a, b] = L_ab
                L[b, a] = L_ab

                cross_results.append({
                    "a": int(a),
                    "b": int(b),
                    "key": key,
                    "N_a": int(N_a),
                    "N_b": int(N_b),
                    "fit_cross": fit_cross,
                    "L_ab": float(L_ab),
                    "D_ab_norm_a": float(D_ab_norm_a),
                    "D_ab_norm_b": float(D_ab_norm_b),
                })

                print(f"\n  {names[a]} x {names[b]}")
                print(f"    L_{names[a]}{names[b]} cross : {L_ab:.6e}")
                print(f"    D_cross normalized by {names[a]} : {D_ab_norm_a:.6e} m²/s")
                print(f"    D_cross normalized by {names[b]} : {D_ab_norm_b:.6e} m²/s")
                print(f"    R² cross                    : {fit_cross.r2:.5f}")

    # Binary cation transference number from Onsager matrix
    if cation_name is not None and anion_name is not None:
        if cation_name not in names:
            raise ValueError(f"[Onsager t+] cation_name '{cation_name}' not found in {names}")
        if anion_name not in names:
            raise ValueError(f"[Onsager t+] anion_name '{anion_name}' not found in {names}")

        ip = names.index(cation_name)
        im = names.index(anion_name)

        # Estimate species valences from average restored charge per entity.
        z_p = float(np.mean(q_by_species_C[ip]) / e_charge)
        z_m = float(np.mean(q_by_species_C[im]) / e_charge)

        Lpp = L[ip, ip]
        Lmm = L[im, im]
        Lpm = L[ip, im]

        denom = z_p*z_p*Lpp + z_m*z_m*Lmm + 2.0*z_p*z_m*Lpm
        numer_p = z_p * (z_p*Lpp + z_m*Lpm)

        t_plus = numer_p / denom if denom != 0 else np.nan
        transference = {
            "cation_name": cation_name,
            "anion_name": anion_name,
            "z_p": float(z_p),
            "z_m": float(z_m),
            "Lpp": float(Lpp),
            "Lmm": float(Lmm),
            "Lpm": float(Lpm),
            "t_plus": float(t_plus),
        }

        print("\n=== TRANSFERENCE NUMBER FROM ONSAGER MATRIX ===")
        print(f"cation = {cation_name}, z+ = {z_p:+.6f}")
        print(f"anion  = {anion_name}, z- = {z_m:+.6f}")
        print(f"L++ = {Lpp:.6e}")
        print(f"L-- = {Lmm:.6e}")
        print(f"L+- = {Lpm:.6e}")
        print(f"t+ Onsager/correlation-aware = {t_plus:.6f}")

        if abs(z_p - 1.0) < 1e-3 and abs(z_m + 1.0) < 1e-3:
            t_plus_11 = (Lpp - Lpm) / (Lpp + Lmm - 2.0*Lpm)
            transference["t_plus_11"] = float(t_plus_11)
            print(f"t+ 1:1 simplified check       = {t_plus_11:.6f}")

    return {
        "L": L,
        "species": species_results,
        "cross": cross_results,
        "transference": transference,
    }
# ----------------------------
# Main
# ----------------------------
# ----------------------------
# Multi-replica helpers + Main
# ----------------------------
@dataclass
class ReplicaResult:
    index: int
    label: str
    topology: str
    trajectory: str
    ts_data: Dict[str, np.ndarray]
    V_A3: float
    stdvol_A3: float
    rsd_percent: float
    names: List[str]
    modes: List[str]
    q_by_species_C: List[np.ndarray]


def make_trajectory_slice(traj_cfg: dict) -> slice:
    traj_start = int(traj_cfg.get("start", 0))
    traj_stop = traj_cfg.get("stop", -1)
    traj_stride = int(traj_cfg.get("stride", 1))

    if traj_stop == -1:
        return slice(traj_start, None, traj_stride)
    return slice(traj_start, int(traj_stop), traj_stride)


def get_replica_specs(cfg: dict) -> List[dict]:
    """
    Flexible input parser. Supported TOML styles:

    1) Backward-compatible single trajectory:
       [files]
       topology = "system.tpr"
       trajectory = "rep1.xtc"

    2) Common topology, several trajectories:
       [files]
       topology = "system.tpr"
       trajectories = ["rep1.xtc", "rep2.xtc", "rep3.xtc"]

    3) One topology per trajectory:
       [files]
       topologies = ["rep1.tpr", "rep2.tpr", "rep3.tpr"]
       trajectories = ["rep1.xtc", "rep2.xtc", "rep3.xtc"]

    4) Explicit array of tables:
       [[replicas]]
       label = "rep1"
       topology = "rep1.tpr"
       trajectory = "rep1.xtc"
    """
    files = cfg["files"]

    if "replicas" in cfg:
        raw_reps = cfg["replicas"]
        if not isinstance(raw_reps, list):
            raise ValueError("Top-level [[replicas]] must be an array of tables.")
        reps = []
        for i, rep in enumerate(raw_reps, start=1):
            reps.append({
                "label": rep.get("label", f"rep{i}"),
                "topology": rep["topology"],
                "trajectory": rep["trajectory"],
            })
        return reps

    if "replicas" in files:
        raw_reps = files["replicas"]
        if not isinstance(raw_reps, list):
            raise ValueError("files.replicas must be an array of inline tables.")
        reps = []
        for i, rep in enumerate(raw_reps, start=1):
            reps.append({
                "label": rep.get("label", f"rep{i}"),
                "topology": rep["topology"],
                "trajectory": rep["trajectory"],
            })
        return reps

    if "trajectories" in files:
        trajectories = list(files["trajectories"])
        if "topologies" in files:
            topologies = list(files["topologies"])
            if len(topologies) != len(trajectories):
                raise ValueError("files.topologies and files.trajectories must have the same length.")
        else:
            if "topology" not in files:
                raise ValueError("files.topology is required when using files.trajectories without files.topologies.")
            topologies = [files["topology"] for _ in trajectories]

        return [
            {"label": f"rep{i}", "topology": top, "trajectory": traj}
            for i, (top, traj) in enumerate(zip(topologies, trajectories), start=1)
        ]

    # Backward-compatible original input
    return [{
        "label": "rep1",
        "topology": files["topology"],
        "trajectory": files["trajectory"],
    }]


def weighted_average_series(values_stack: np.ndarray, weights_stack: np.ndarray) -> np.ndarray:
    """
    Average over replicas at each lag. If weights are zero at a lag, fall back
    to an unweighted mean. weights_stack normally contains the number of
    time origins contributing to each lag in each replica.
    """
    wsum = np.sum(weights_stack, axis=0)
    unweighted = np.mean(values_stack, axis=0)
    return np.divide(
        np.sum(values_stack * weights_stack, axis=0),
        wsum,
        out=unweighted.copy(),
        where=wsum > 0,
    )


def average_replica_timeseries(results: List[ReplicaResult], method: str = "counts") -> Dict[str, np.ndarray]:
    """
    Average Helfand/MSD-like time series from several replicas.

    method="counts" weights each lag by the number of time origins in that
    replica, which is statistically preferable if replica lengths differ.
    method="replica" gives every replica equal weight at every lag.
    """
    if len(results) == 0:
        raise ValueError("No replica results were provided.")

    method = method.lower()
    if method not in ("counts", "replica"):
        raise ValueError("analysis.replica_average must be 'counts' or 'replica'.")

    min_len = min(len(r.ts_data["tau_ps"]) for r in results)
    if len(set(len(r.ts_data["tau_ps"]) for r in results)) > 1:
        print(f"[replicas] WARNING: replicas have different lag-axis lengths; truncating all to {min_len} points.")

    ref_tau = results[0].ts_data["tau_ps"][:min_len]
    for r in results[1:]:
        tau_i = r.ts_data["tau_ps"][:min_len]
        if not np.allclose(tau_i, ref_tau, rtol=1e-5, atol=1e-8):
            print(
                "[replicas] WARNING: lag-time grids are not identical. "
                "Averaging is being done by lag index using weighted mean tau. "
                "For strongly different frame spacings, use the same saved-frame interval for all replicas."
            )
            break

    # Keep only keys that exist in every replica.
    ref_keys = list(results[0].ts_data.keys())
    common_keys = [k for k in ref_keys if all(k in r.ts_data for r in results)]

    counts_stack = np.stack([r.ts_data["counts"][:min_len] for r in results], axis=0)
    if method == "replica":
        weights_stack = np.ones_like(counts_stack, dtype=float)
    else:
        weights_stack = counts_stack.astype(float)

    out: Dict[str, np.ndarray] = {}
    out["counts"] = np.sum(counts_stack, axis=0)

    tau_ps_stack = np.stack([r.ts_data["tau_ps"][:min_len] for r in results], axis=0)
    out["tau_ps"] = weighted_average_series(tau_ps_stack, weights_stack)
    out["tau_s"] = out["tau_ps"] * 1e-12

    for key in common_keys:
        if key in ("tau_ps", "tau_s", "counts"):
            continue
        values_stack = np.stack([r.ts_data[key][:min_len] for r in results], axis=0)
        out[key] = weighted_average_series(values_stack, weights_stack)

        # Unweighted replica scatter, useful for error bars/post-processing.
        if len(results) > 1:
            out[f"{key}_replica_std"] = np.std(values_stack, axis=0, ddof=1)
            out[f"{key}_replica_sem"] = out[f"{key}_replica_std"] / np.sqrt(len(results))

    return out


def fit_main_curves(ts_data: Dict[str, np.ndarray], analysis: dict, V_m3: float, T_K: float) -> dict:
    tau_s = ts_data["tau_s"]
    total = ts_data["total_G2"]
    selfv = ts_data["self_G2"]
    distinct = ts_data["distinct_G2"]

    fit_start_s = float(analysis["fit_start_ps"]) * 1e-12
    fit_end_s = float(analysis["fit_end_ps"]) * 1e-12
    mask = (tau_s >= fit_start_s) & (tau_s <= fit_end_s)
    if np.sum(mask) < 3:
        raise ValueError("Fit window has <3 points. Increase max_lag_ps or widen fit window.")

    fit_total = linear_fit_with_r2(tau_s[mask], total[mask])
    fit_self = linear_fit_with_r2(tau_s[mask], selfv[mask])
    fit_dist = linear_fit_with_r2(tau_s[mask], distinct[mask])

    sigma_total = conductivity_from_slope(fit_total.slope, V_m3, T_K)
    sigma_self = conductivity_from_slope(fit_self.slope, V_m3, T_K)
    sigma_dist = conductivity_from_slope(fit_dist.slope, V_m3, T_K)
    haven = sigma_self / sigma_total if sigma_total != 0 else np.nan

    return {
        "mask": mask,
        "fit_start_s": fit_start_s,
        "fit_end_s": fit_end_s,
        "fit_total": fit_total,
        "fit_self": fit_self,
        "fit_dist": fit_dist,
        "sigma_total": sigma_total,
        "sigma_self": sigma_self,
        "sigma_dist": sigma_dist,
        "haven": haven,
    }


def print_fit_report(label: str, fit_info: dict, cond_unit: str):
    scale = 0.01 / (e_charge**2)
    s_to_ns = 1e-9
    fit_total = fit_info["fit_total"]
    mask = fit_info["mask"]

    print(f"\n[fit] TOTAL M(τ) linear fit: {label}")
    print(f"  slope     = {fit_total.slope*scale*s_to_ns:.6e}  (e²·nm² / ns)")
    print(f"  intercept = {fit_total.intercept*scale:.6e} (e²·nm²)")
    print(f"  R²        = {fit_total.r2:.6f}")
    print(f"  points    = {int(np.sum(mask))}")

    sigma_total_u = convert_conductivity(fit_info["sigma_total"], cond_unit)
    sigma_self_u = convert_conductivity(fit_info["sigma_self"], cond_unit)
    sigma_dist_u = convert_conductivity(fit_info["sigma_dist"], cond_unit)

    print(f"\n=== RESULTS for {label} ({cond_unit}) ===")
    print(f"Total conductivity    : {sigma_total_u:.6e}")
    print(f"Self conductivity     : {sigma_self_u:.6e}")
    print(f"Distinct conductivity : {sigma_dist_u:.6e}")
    print(f"Haven ratio (self/total): {fit_info['haven']:.6f}")


def print_species_and_cross_report(
    names: List[str],
    tau_s: np.ndarray,
    mask: np.ndarray,
    ts_data: Dict[str, np.ndarray],
    V_m3: float,
    T_K: float,
    cond_unit: str,
):
    print(f"\n=== SPECIES BREAKDOWN ({cond_unit}) ===")
    for a, name in enumerate(names):
        g2a = ts_data[f"species_{a}_G2"]
        selfa = ts_data[f"species_{a}_self"]
        dicta = ts_data[f"species_{a}_distinct"]

        fit_g2a = linear_fit_with_r2(tau_s[mask], g2a[mask])
        fit_selfa = linear_fit_with_r2(tau_s[mask], selfa[mask])
        fit_dicta = linear_fit_with_r2(tau_s[mask], dicta[mask])

        sigma_g2a = conductivity_from_slope(fit_g2a.slope, V_m3, T_K)
        sigma_selfa = conductivity_from_slope(fit_selfa.slope, V_m3, T_K)
        sigma_dicta = conductivity_from_slope(fit_dicta.slope, V_m3, T_K)

        print(f"\n  [{a}] {name}")
        print(f"    sigma(|G_species|²) : {convert_conductivity(sigma_g2a, cond_unit):.6e}")
        print(f"    sigma_self(species)  : {convert_conductivity(sigma_selfa, cond_unit):.6e}")
        print(f"    sigma_distinct(species)  : {convert_conductivity(sigma_dicta, cond_unit):.6e}")

    if len(names) >= 2:
        print(f"\n=== CROSS TERMS BETWEEN SPECIES (2*G_a·G_b) ({cond_unit}) ===")
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                key = f"cross_{a}_{b}"
                series = ts_data[key]
                fit_cross = linear_fit_with_r2(tau_s[mask], series[mask])
                sigma_cross = conductivity_from_slope(fit_cross.slope, V_m3, T_K)
                print(f"  {names[a]} x {names[b]} : {convert_conductivity(sigma_cross, cond_unit):.6e}")


def validate_replica_species_consistency(results: List[ReplicaResult]) -> None:
    ref = results[0]
    for r in results[1:]:
        if r.names != ref.names or r.modes != ref.modes:
            raise ValueError(
                "Species names/modes differ between replicas. "
                f"Reference={list(zip(ref.names, ref.modes))}; "
                f"{r.label}={list(zip(r.names, r.modes))}"
            )
        if len(r.q_by_species_C) != len(ref.q_by_species_C):
            raise ValueError(f"Species count differs between replicas: reference vs {r.label}")
        for i, (q_ref, q_rep) in enumerate(zip(ref.q_by_species_C, r.q_by_species_C)):
            if len(q_ref) != len(q_rep):
                raise ValueError(
                    f"Number of entities for species '{ref.names[i]}' differs between replicas: "
                    f"reference={len(q_ref)}, {r.label}={len(q_rep)}"
                )
            if not np.allclose(q_ref, q_rep, rtol=1e-8, atol=1e-30):
                raise ValueError(
                    f"Restored charges for species '{ref.names[i]}' differ between replicas. "
                    "Use consistent topology/charge restoration across all replicas."
                )


def process_replica(
    cfg: dict,
    spec: dict,
    rep_index: int,
    traj_slice: slice,
    verbose: bool,
) -> ReplicaResult:
    system = cfg["system"]
    analysis = cfg["analysis"]
    charge_cfg = cfg.get("charges", {})
    ions = cfg["ions"]["species"]
    T_K = float(system["temperature_K"])

    label = spec.get("label", f"rep{rep_index}")
    topology = spec["topology"]
    trajectory = spec["trajectory"]

    print(f"\n\n=== REPLICA {rep_index}: {label} ===")
    print(f"[files] topology   : {topology}")
    print(f"[files] trajectory : {trajectory}")

    u = mda.Universe(topology, trajectory)
    pbc_mode = setup_pbc_transformations(u, cfg, verbose)

    names = []
    modes = []
    pos_getters = []
    qC_list = []
    grand_rest_C = 0.0

    print("[select] Species recap:")
    for sp in ions:
        name, mode, get_pos, qC_entities, ag = build_species_entities(u, sp, charge_cfg, verbose)
        names.append(name)
        modes.append(mode)
        pos_getters.append(get_pos)
        qC_list.append(qC_entities)
        grand_rest_C += float(np.sum(qC_entities))

    print(f"\n[select] TOTAL restored charge over all selected species:")
    print(f"  restored total : {(grand_rest_C/e_charge):+.6f} e  ({grand_rest_C:+.6e} C)")

    print(f"\n[system] Temperature: {T_K} K")
    print(f"[system] PBC mode: {pbc_mode}")

    u.trajectory[0]
    nojump_checker = NoJumpStrideChecker(frac_half_box=0.5)
    com_drift_checker = SystemCOMDriftChecker(u, out_prefix=f"{label}_system_COM")
    time_ps, vols_A3, pos_by_species = collect_positions(
        u.trajectory[traj_slice],
        pos_getters,
        verbose=verbose,
        nojump_checker=nojump_checker,
        com_drift_checker=com_drift_checker,
    )

    V_A3 = float(vols_A3.mean())
    stdvol_A3 = float(vols_A3.std())
    rsd = (stdvol_A3 / V_A3) * 100.0

    print(f"Avg Vol: {V_A3:.2f} Å³ | StdDev: {stdvol_A3:.2f} Å³ | RSD: {rsd:.2f}%")
    if rsd < 5:
        print("Status: [PASS] Volume is stable.")
    else:
        print(f"Status: [WARNING] High fluctuation ({rsd:.2f}%). Check equilibration.")

    origin_stride_frames = int(analysis.get("origin_stride_frames", 1))
    max_lag_ps = float(analysis.get("max_lag_ps", analysis["fit_end_ps"]))
    origins_chunk = int(analysis.get("origins_chunk", 200))

    ts_data = accumulate_multi_origin_vectorized(
        time_ps=time_ps,
        pos_by_species=pos_by_species,
        q_by_species_C=qC_list,
        origin_stride_frames=origin_stride_frames,
        max_lag_ps=max_lag_ps,
        origins_chunk=origins_chunk,
        verbose=verbose,
    )

    return ReplicaResult(
        index=rep_index,
        label=label,
        topology=topology,
        trajectory=trajectory,
        ts_data=ts_data,
        V_A3=V_A3,
        stdvol_A3=stdvol_A3,
        rsd_percent=rsd,
        names=names,
        modes=modes,
        q_by_species_C=[q.copy() for q in qC_list],
    )


def main(argv: list[str] | None = None) -> dict[str, object] | None:
    parser = argparse.ArgumentParser(description="Vectorized Helfand–Einstein conductivity with optional replica averaging.")
    parser.add_argument("config", nargs="?", help="Path to config TOML.")
    parser.add_argument("--write-template", metavar="PATH", help="Write a template config TOML and exit.")
    args = parser.parse_args(argv)

    if args.write_template:
        write_template_config(args.write_template)
        print(f"Wrote template config to: {args.write_template}")
        return {"template": args.write_template}

    if not args.config:
        print("Error: need a config TOML (or use --write-template).", file=sys.stderr)
        sys.exit(2)

    now = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg = load_toml(args.config)
    cfg_text = read_text(args.config)

    prefix = cfg.get("output", {}).get("prefix", "HE")
    out_cfg = cfg.get("output", {})
    time_unit = out_cfg.get("time_unit", "s")
    cond_unit = out_cfg.get("conductivity_unit", "S/m")
    save_replica_csv = bool(out_cfg.get("save_replica_csv", True))
    save_diffusion_plot = bool(out_cfg.get("save_diffusion_plot", True))

    log_path = f"{prefix}_conductivity_vec_{now}.log"
    png_path = f"{prefix}_diagnostic_vec_{now}.png"
    diffusion_png_path = f"{prefix}_diffusion_diagnostics_vec_{now}.png"
    csv_path = f"{prefix}_timeseries_vec_{now}.csv"

    tee = TeeStdout(log_path)
    sys.stdout = tee

    try:
        print("=== Helfand–Einstein Conductivity (vectorized, manual fit window) ===")
        print(f"[run] timestamp: {now}")
        print(f"[run] command: {' '.join(sys.argv)}")
        print(f"[run] config file: {args.config}")
        print("\n--- CONFIG (raw TOML) ---")
        print(cfg_text.strip())
        print("--- END CONFIG ---\n")

        system = cfg["system"]
        analysis = cfg["analysis"]
        traj_cfg = cfg.get("trajectory", {})
        verbose = bool(system.get("verbose", True))
        T_K = float(system["temperature_K"])
        traj_slice = make_trajectory_slice(traj_cfg)

        cation_name = analysis.get("cation_name", None)
        anion_name = analysis.get("anion_name", None)
        if cation_name is None or anion_name is None:
            print("\n[transference] Skipping transference numbers (set analysis.cation_name and analysis.anion_name in TOML).")

        replica_specs = get_replica_specs(cfg)
        print(f"[replicas] Number of replicas to process: {len(replica_specs)}")
        for i, spec in enumerate(replica_specs, start=1):
            print(f"  {i}. {spec.get('label', f'rep{i}')}: {spec['trajectory']}")

        results: List[ReplicaResult] = []
        for i, spec in enumerate(replica_specs, start=1):
            result = process_replica(cfg, spec, i, traj_slice, verbose)
            results.append(result)

            # Optional per-replica fit and CSV. Useful for diagnosing outliers.
            V_m3_rep = result.V_A3 * 1e-30
            rep_fit = fit_main_curves(result.ts_data, analysis, V_m3_rep, T_K)
            print_fit_report(result.label, rep_fit, cond_unit)
            if save_replica_csv:
                rep_csv = f"{prefix}_{result.label}_timeseries_vec_{now}.csv"
                save_csv(rep_csv, result.ts_data)
                print(f"[output] Wrote replica timeseries CSV: {rep_csv}")

        validate_replica_species_consistency(results)
        ref_names = results[0].names
        ref_qC_list = results[0].q_by_species_C

        # Average the Helfand/MSD curves, then fit the averaged curve.
        average_method = analysis.get("replica_average", "counts")
        if len(results) == 1:
            print("\n[replicas] Single trajectory supplied; using its time series directly.")
            ts_data = results[0].ts_data
        else:
            print(f"\n[replicas] Averaging time series from {len(results)} replicas using method='{average_method}'.")
            ts_data = average_replica_timeseries(results, method=average_method)

        tau_s = ts_data["tau_s"]
        total = ts_data["total_G2"]
        selfv = ts_data["self_G2"]
        distinct = ts_data["distinct_G2"]

        V_A3_mean = float(np.mean([r.V_A3 for r in results]))
        V_A3_between = float(np.std([r.V_A3 for r in results], ddof=1)) if len(results) > 1 else 0.0
        V_m3 = V_A3_mean * 1e-30
        print(f"\n[replicas] Mean replica volume used in conductivity/Onsager prefactors: {V_A3_mean:.2f} Å³")
        if len(results) > 1:
            print(f"[replicas] StdDev of mean volumes across replicas: {V_A3_between:.2f} Å³")

        fit_info = fit_main_curves(ts_data, analysis, V_m3, T_K)
        mask = fit_info["mask"]
        fit_total = fit_info["fit_total"]
        fit_start_s = fit_info["fit_start_s"]
        fit_end_s = fit_info["fit_end_s"]
        print_fit_report("replica-averaged curve", fit_info, cond_unit)

        print_species_and_cross_report(
            names=ref_names,
            tau_s=tau_s,
            mask=mask,
            ts_data=ts_data,
            V_m3=V_m3,
            T_K=T_K,
            cond_unit=cond_unit,
        )

        if cation_name is not None and anion_name is not None:
            tn = compute_transference_numbers(
                names=ref_names,
                tau_s=tau_s,
                mask=mask,
                ts_data=ts_data,
                V_m3=V_m3,
                T_K=T_K,
                cation_name=cation_name,
                anion_name=anion_name,
            )

            print("\n=== TRANSFERENCE / TRANSPORT NUMBERS (binary, replica-averaged curve) ===")
            print(f"cation = {cation_name} ; anion = {anion_name}")
            print(f"t+ (Nernst–Einstein, self-only)          : {tn['t_plus_NE']:.6f}")
            print(f"rho+ (correlation-aware bulk fraction)  : {tn['rho_plus']:.6f}")
            print(f"[debug] cross term used: {tn['cross_key']}")

        diffusion_results = report_onsager_diffusion_terms(
            names=ref_names,
            q_by_species_C=ref_qC_list,
            tau_s=tau_s,
            mask=mask,
            ts_data=ts_data,
            V_m3=V_m3,
            T_K=T_K,
            cation_name=cation_name,
            anion_name=anion_name,
        )

        if save_diffusion_plot:
            make_diffusion_diagnostic_plot(
                out_png=diffusion_png_path,
                names=ref_names,
                tau_s=tau_s,
                mask=mask,
                ts_data=ts_data,
                diffusion_results=diffusion_results,
                analysis=analysis,
                out_cfg=out_cfg,
            )
            print(f"[output] Wrote diffusion diagnostic plot: {diffusion_png_path}")

        # Local slope diagnostic on the averaged curve.
        window_ps = float(analysis.get("local_slope_window_ps", 25000.0))
        window_s = window_ps * 1e-12
        local_t, local_sl = compute_local_slopes(tau_s, total, window_s)

        if time_unit.lower() == "ps":
            fit_window_x = (analysis["fit_start_ps"], analysis["fit_end_ps"])
            local_x = local_t * 1e12
        elif time_unit.lower() == "ns":
            fit_window_x = (analysis["fit_start_ps"] * 1e-3, analysis["fit_end_ps"] * 1e-3)
            local_x = local_t * 1e9
        else:
            fit_window_x = (fit_start_s, fit_end_s)
            local_x = local_t

        tau_ps = ts_data["tau_ps"]
        x_plot, x_label = scale_time_for_plot(tau_ps, tau_s, time_unit)
        fit_x_plot = x_plot[mask]

        make_diagnostic_plot(
            out_png=png_path,
            tau_s=x_plot,
            total=total,
            selfv=selfv,
            distinct=distinct,
            fit_total=fit_total,
            fit_window_s=fit_window_x,
            local_t=local_x,
            fit_x_plot=fit_x_plot,
            local_slope=local_sl,
            x_label=x_label,
        )
        print(f"\n[output] Wrote diagnostic plot: {png_path}")

        save_csv(csv_path, ts_data)
        print(f"[output] Wrote averaged timeseries CSV: {csv_path}")
        print(f"[output] Wrote log file: {log_path}")
        return {
            "replicas_processed": len(results),
            "conductivity_s_m": float(fit_info["sigma_total"]),
            "self_conductivity_s_m": float(fit_info["sigma_self"]),
            "distinct_conductivity_s_m": float(fit_info["sigma_dist"]),
            "haven_ratio": float(fit_info["haven"]),
            "timeseries_csv": csv_path,
            "diagnostic_plot": png_path,
            "log_file": log_path,
        }

    finally:
        sys.stdout = tee.stdout
        tee.close()


def run_conductivity_analysis(config_path: str) -> dict[str, object]:
    """Run a TOML-configured analysis and return its output summary."""
    result = main([config_path])
    if result is None:
        raise RuntimeError("Analysis did not produce a result summary.")
    return result


if __name__ == "__main__":
    main()
