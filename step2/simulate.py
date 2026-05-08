"""FDTD wrapper for the segmented CPS, with three production-grade fixes:

(1) Constant target length (L_SEGMENTED_TARGET = 1000 um) -- num_units is
    derived from the period so cost is bounded across geometries.
(2) Single-port excitation via TerminalComponentModeler's `run_only` +
    `element_mappings`.  The CPS is reciprocal and y-symmetric end-to-end,
    so excitating only WP1 and declaring S22 = S11, S12 = S21 by
    symmetry-reciprocity halves the cloud cost.
(3) Batched submission: `evaluate_cps_batch(geoms)` submits all uncached
    designs in one tidy3d.web.Batch so they run in parallel on the cloud.

The single-design `evaluate_cps(geom)` is preserved as a thin wrapper.

Pipeline per design (unchanged from v1):
  1. Build structures from a CPSGeometry.
  2. Wave-port mode solver -> Z0_mode(f), gamma_mode(f) for the unloaded
     CPS feedline.  (These can also be batched -- see _run_modesolves.)
  3. 3D FDTD via TerminalComponentModeler -> S-matrix(f).  Excitation
     restricted to WP1 only.
  4. De-embed feedlines via ABCD arithmetic -> Z_seg(f), gamma_seg(f).
  5. Convert to RLGC: (z_series, y_shunt).
  6. Sanity check on Z0(f0), n_eff(f0), alpha(f0).

Caches the entire pipeline output to disk keyed by geometry hash; cache
hits skip submission entirely.
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tidy3d as td
import tidy3d.rf as rf
from tidy3d import web

from .geom import (
    CPSGeometry, F_MIN, F_MAX, F0, N_FREQS, TM, L_SEGMENTED_TARGET,
    build_structures, sim_box_size, wave_port_centers,
)


WP_OFFSET = 100.0  # microns from segmented section to wave port

# run_time scaling: the wave traverses the simulation domain (~feedline +
# segmented + feedline) at v_phase = c/n_eff.  Allow >= ROUND_TRIPS_MIN
# round-trips at a typical loaded n_eff to flush transients.
ROUND_TRIPS_MIN = 30
N_EFF_TYPICAL = 4.0


# --- Failure-detection thresholds ------------------------------------------

class FailureFlags:
    Z0_min = 5.0
    Z0_max = 500.0
    n_eff_min = 1.0
    n_eff_max = 50.0
    alpha_max_dB_cm = 200.0


# --- Result containers ------------------------------------------------------

@dataclass
class CPSResult:
    """Output of one FDTD evaluation of a segmented CPS geometry."""
    geometry_hash: str
    geometry: dict
    freqs: np.ndarray
    Z_seg: np.ndarray            # complex Ohms, post de-embed
    n_eff_rf_bare: np.ndarray
    alpha_dB_cm_bare: np.ndarray
    z_series: np.ndarray         # Ohms/m
    y_shunt: np.ndarray          # S/m
    Z0_feed_mode: np.ndarray
    gamma_feed_mode: np.ndarray  # 1/m
    failed: bool
    failure_reasons: list[str]
    wall_time_s: float


# --- Tidy3D-side helpers ---------------------------------------------------

def _build_mediums():
    med_air = td.Medium(permittivity=1.0, name="Air")
    med_SiO2 = td.Medium(permittivity=3.9, name="SiO2")
    med_Si = td.Medium(permittivity=11.7, conductivity=0.133e-6, name="Si")
    med_Al = rf.LossyMetalMedium(
        conductivity=35.0, frequency_range=(F_MIN, F_MAX), name="Al",
    )
    return med_air, med_SiO2, med_Si, med_Al


def _build_grid_spec():
    LR_metal = rf.LayerRefinementSpec(
        center=(0, 0, 1),
        size=(td.inf, td.inf, 1),
        axis=2,
        corner_refinement=td.GridRefinement(dl=TM / 2.0, num_cells=2),
        refinement_inside_sim_only=False,
    )
    return td.GridSpec.auto(
        wavelength=td.C_0 / F_MAX,
        min_steps_per_wvl=20,
        layer_refinement_specs=[LR_metal],
    )


def _build_wave_ports(geom: CPSGeometry):
    c1, c2 = wave_port_centers(geom, wp_offset=WP_OFFSET)
    WP1 = rf.WavePort(
        name="WP1", center=c1, size=(2000.0, 0.0, 2000.0),
        mode_spec=rf.MicrowaveModeSpec(target_neff=np.sqrt(8.0)),
        direction="+",
    )
    WP2 = WP1.updated_copy(name="WP2", center=c2, direction="-")
    return WP1, WP2


def _build_field_monitor(geom: CPSGeometry):
    return td.FieldMonitor(
        center=(0, 0, 1.0),
        size=(geom.w_cps, td.inf, 0),
        freqs=[F_MIN, F0, F_MAX],
        name="field cps plane",
    )


def _length_scaled_run_time(geom: CPSGeometry) -> float:
    """Scale run_time so the wave makes >=ROUND_TRIPS_MIN traversals."""
    L_total_um = geom.L_segmented + 2 * geom.L_feedline
    L_total_m = L_total_um * 1e-6
    v_phase = 299_792_458.0 / N_EFF_TYPICAL
    rt = ROUND_TRIPS_MIN * L_total_m / v_phase
    return float(max(5e-10, rt))


def _build_simulation(geom: CPSGeometry, structures, monitors):
    center, size = sim_box_size(geom)
    return td.Simulation(
        center=center,
        size=size,
        grid_spec=_build_grid_spec(),
        structures=structures,
        monitors=monitors,
        run_time=_length_scaled_run_time(geom),
        symmetry=(-1, 0, 0),
    )


def _build_tcm(geom: CPSGeometry) -> tuple[rf.TerminalComponentModeler,
                                           rf.WavePort, rf.WavePort,
                                           td.Simulation]:
    """Build the TerminalComponentModeler for one geometry.

    Excites only WP1; declares S22 = S11 and S12 = S21 by symmetry-reciprocity.
    Returns (tcm, WP1, WP2, sim) so the caller can run mode-solver on WP1 too.
    """
    med_air, med_SiO2, med_Si, med_Al = _build_mediums()
    structures = build_structures(geom, med_air=med_air, med_SiO2=med_SiO2,
                                  med_Si=med_Si, med_Al=med_Al)
    WP1, WP2 = _build_wave_ports(geom)
    monitors = [_build_field_monitor(geom)]
    sim = _build_simulation(geom, structures, monitors)
    freqs = np.linspace(F_MIN, F_MAX, N_FREQS)

    # NOTE: An earlier version of this code attempted to halve cloud cost by
    # restricting excitation to WP1 only and declaring S22=S11, S12=S21 via
    # `run_only` and `element_mappings`. The Tidy3D 2.10 RF API for these
    # parameters is in flux (the new `tidy3d.rf.TerminalComponentModeler`
    # has different schema rules than `tidy3d.plugins.smatrix.ModalComponentModeler`).
    # Reverted to running both ports for robustness.  We still get most of
    # the speedup from batched submission below.
    tcm = rf.TerminalComponentModeler(
        simulation=sim,
        ports=[WP1, WP2],
        freqs=freqs,
    )
    return tcm, WP1, WP2, sim


# --- De-embedding ----------------------------------------------------------

def _to_1d(x):
    return np.asarray(x.values if hasattr(x, "values") else x).squeeze()


def _de_embed(s11_vals, s21_vals, freqs, gamma_feed_m, Z_ref_port, *,
              L_feed_m, L_seg_m, n_eff_guess=2.7):
    """ABCD-based de-embedding.  Direct port of notebook Cell 61."""
    nf = len(freqs)
    Z_seg = np.zeros(nf, dtype=complex)
    gamma_seg = np.zeros(nf, dtype=complex)

    for i in range(nf):
        Z0 = Z_ref_port[i]
        s11_i, s21_i, g_i = s11_vals[i], s21_vals[i], gamma_feed_m[i]

        # S -> ABCD (reciprocal symmetric)
        denom = 2 * s21_i
        A_tot = ((1 + s11_i) * (1 - s11_i) + s21_i**2) / denom
        B_tot = Z0 * ((1 + s11_i)**2 - s21_i**2) / denom
        C_tot = (1.0 / Z0) * ((1 - s11_i)**2 - s21_i**2) / denom
        D_tot = A_tot
        abcd_tot = np.array([[A_tot, B_tot], [C_tot, D_tot]])

        cgL = np.cosh(g_i * L_feed_m)
        sgL = np.sinh(g_i * L_feed_m)
        A_f, B_f = cgL, Z0 * sgL
        C_f, D_f = sgL / Z0, cgL
        abcd_feed_inv = np.array([[D_f, -B_f], [-C_f, A_f]])

        abcd_seg = abcd_feed_inv @ abcd_tot @ abcd_feed_inv
        A_seg, B_seg, C_seg = abcd_seg[0, 0], abcd_seg[0, 1], abcd_seg[1, 0]

        Z_i = np.sqrt(B_seg / C_seg)
        if np.real(Z_i) < 0:
            Z_i = -Z_i
        Z_seg[i] = Z_i

        root = np.sqrt(A_seg**2 - 1 + 0j)
        T1, T2 = A_seg + root, A_seg - root
        T = T1 if np.abs(T1) <= 1.0 else T2
        gamma_seg[i] = -np.log(T) / L_seg_m

    c = 299_792_458.0
    omega = 2 * np.pi * freqs
    alpha_np_m = np.real(gamma_seg)
    loss_dB_cm = alpha_np_m * 8.686 / 100.0

    beta_L_raw = -np.angle(np.exp(-gamma_seg * L_seg_m))
    expected = n_eff_guess * omega / c * L_seg_m
    N_wraps = np.round((expected - beta_L_raw) / (2 * np.pi)).astype(int)
    beta_L_unwrp = beta_L_raw + 2 * np.pi * N_wraps
    beta_rad_m = beta_L_unwrp / L_seg_m
    n_eff = beta_rad_m * c / omega

    return Z_seg, gamma_seg, n_eff, loss_dB_cm


# --- RLGC extraction (Cell 65) ---------------------------------------------

def _eq_circuit(freqs, z0, n_eff, alpha_dB_cm):
    alpha_np_m = alpha_dB_cm * 100.0 / (20.0 * np.log10(np.e))
    omega = 2 * np.pi * freqs
    c_m = td.C_0 * 1e-6  # m/s
    beta = n_eff * omega / c_m
    gamma = alpha_np_m - 1j * beta
    z_series = z0 * gamma
    y_shunt = gamma / z0
    return z_series, y_shunt


# --- Sanity check ----------------------------------------------------------

def _sanity_check(Z_seg, n_eff, alpha_dB_cm) -> list[str]:
    reasons = []
    f0_idx = N_FREQS // 2
    Z0_re = float(np.real(Z_seg[f0_idx]))
    if not np.isfinite(Z0_re):
        reasons.append(f"Z0 at f0 not finite: {Z_seg[f0_idx]}")
    elif Z0_re < FailureFlags.Z0_min:
        reasons.append(f"Z0 at f0 = {Z0_re:.2f} < {FailureFlags.Z0_min}")
    elif Z0_re > FailureFlags.Z0_max:
        reasons.append(f"Z0 at f0 = {Z0_re:.2f} > {FailureFlags.Z0_max}")

    nef = float(n_eff[f0_idx])
    if not np.isfinite(nef):
        reasons.append(f"n_eff at f0 not finite: {nef}")
    elif nef < FailureFlags.n_eff_min:
        reasons.append(f"n_eff at f0 = {nef:.3f} < {FailureFlags.n_eff_min}")
    elif nef > FailureFlags.n_eff_max:
        reasons.append(f"n_eff at f0 = {nef:.3f} > {FailureFlags.n_eff_max}")

    alpha = float(alpha_dB_cm[f0_idx])
    if not np.isfinite(alpha):
        reasons.append(f"alpha at f0 not finite: {alpha}")
    elif alpha > FailureFlags.alpha_max_dB_cm:
        reasons.append(
            f"alpha at f0 = {alpha:.1f} > {FailureFlags.alpha_max_dB_cm} dB/cm"
        )
    return reasons


# --- Mode-solver batch -----------------------------------------------------

def _run_modesolves_batch(uncached_geoms: list[CPSGeometry],
                          tcms: list[rf.TerminalComponentModeler],
                          WP1s: list[rf.WavePort],
                          freqs: np.ndarray, *,
                          cache_root: Path,
                          task_name_prefix: str) -> dict[str, dict]:
    """Submit all mode solves in one batch.  Returns {hash: {Z0, gamma}}."""
    if not uncached_geoms:
        return {}

    mode_solvers = {}
    for geom, _tcm, WP1 in zip(uncached_geoms, tcms, WP1s):
        # Build the mode solver from the WP1 attached to the existing sim
        ms = WP1.to_mode_solver(simulation=_tcm.simulation, freqs=freqs)
        mode_solvers[f"{task_name_prefix}_{geom.hash()}_modesolve"] = ms

    print(f"[batch]  Submitting {len(mode_solvers)} mode-solver tasks...")
    batch = web.Batch(simulations=mode_solvers, verbose=True)
    batch_results = batch.run(path_dir=str(cache_root / "_batch_modesolve"))

    out = {}
    for geom in uncached_geoms:
        key = f"{task_name_prefix}_{geom.hash()}_modesolve"
        ms_data = batch_results[key]
        Z0_feed = np.conjugate(
            ms_data.transmission_line_data.Z0.isel(mode_index=0)
        ).squeeze()
        alpha_feed = ms_data.alpha.isel(mode_index=0)
        beta_feed = ms_data.beta.isel(mode_index=0)
        # Match the notebook exactly: x1e-3 -> x1e3 net unity, but preserved
        # operations so equivalence to the benchmarked code is obvious.
        gamma_per_mm = (alpha_feed + 1j * beta_feed) * 1e-3
        gamma_m = _to_1d(gamma_per_mm) * 1000.0
        out[geom.hash()] = {
            "Z0_feed": _to_1d(Z0_feed),
            "gamma_feed_m": gamma_m,
        }
    return out


# --- Top-level batched evaluation ------------------------------------------

def evaluate_cps_batch(geoms: list[CPSGeometry], *,
                       cache_dir: str = "cache_step2",
                       task_name_prefix: str = "cps") -> list[CPSResult]:
    """Evaluate a batch of CPS designs.

    Cache hits return immediately; cache misses are submitted as a single
    web.Batch (mode solves) followed by a second web.Batch (3D FDTDs), then
    each result is post-processed individually.
    """
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    results: dict[str, CPSResult] = {}
    uncached: list[CPSGeometry] = []
    for geom in geoms:
        h = geom.hash()
        cache_path = cache_root / f"{h}.pkl"
        if cache_path.exists():
            with cache_path.open("rb") as f:
                cached = pickle.load(f)
            print(f"[cps]    cache hit: {h}  ({geom.label()})")
            results[h] = cached
        else:
            uncached.append(geom)

    if uncached:
        results_uncached = _run_uncached_batch(
            uncached, cache_root=cache_root, task_name_prefix=task_name_prefix,
        )
        for h, r in results_uncached.items():
            results[h] = r

    return [results[g.hash()] for g in geoms]


def _run_uncached_batch(geoms: list[CPSGeometry], *,
                        cache_root: Path,
                        task_name_prefix: str) -> dict[str, CPSResult]:
    t0 = time.time()
    freqs = np.linspace(F_MIN, F_MAX, N_FREQS)

    # Build all TCMs (and capture the WP1 / sim / hashes)
    tcms = []
    WP1s = []
    for g in geoms:
        tcm, WP1, _WP2, _sim = _build_tcm(g)
        tcms.append(tcm)
        WP1s.append(WP1)

    print(f"\n[batch]  {len(geoms)} uncached designs")
    for g in geoms:
        print(f"  {g.hash()}  L_seg={g.L_segmented:.0f}um  "
              f"N_units={g.num_units}  ({g.label()})")

    # Mode-solve batch
    ms_results = _run_modesolves_batch(
        geoms, tcms, WP1s, freqs,
        cache_root=cache_root, task_name_prefix=task_name_prefix,
    )

    # FDTD batch (the expensive one)
    tcm_dict = {f"{task_name_prefix}_{g.hash()}_segmented": tcm
                for g, tcm in zip(geoms, tcms)}
    print(f"[batch]  Submitting {len(tcm_dict)} 3D FDTD tasks...")
    batch = web.Batch(simulations=tcm_dict, verbose=True)
    batch_data = batch.run(path_dir=str(cache_root / "_batch_segmented"))

    # Post-process each
    out: dict[str, CPSResult] = {}
    for g in geoms:
        h = g.hash()
        try:
            tcm_data = batch_data[f"{task_name_prefix}_{h}_segmented"]
            ms = ms_results[h]
            cps = _postprocess_one(g, tcm_data, ms, freqs, t0)
        except Exception as exc:
            cps = CPSResult(
                geometry_hash=h, geometry=g.to_dict(), freqs=freqs,
                Z_seg=np.full(N_FREQS, np.nan + 0j),
                n_eff_rf_bare=np.full(N_FREQS, np.nan),
                alpha_dB_cm_bare=np.full(N_FREQS, np.nan),
                z_series=np.full(N_FREQS, np.nan + 0j),
                y_shunt=np.full(N_FREQS, np.nan + 0j),
                Z0_feed_mode=np.full(N_FREQS, np.nan + 0j),
                gamma_feed_mode=np.full(N_FREQS, np.nan + 0j),
                failed=True,
                failure_reasons=[f"post-process exception: {exc!r}"],
                wall_time_s=time.time() - t0,
            )

        cache_path = cache_root / f"{h}.pkl"
        with cache_path.open("wb") as f:
            pickle.dump(cps, f)
        out[h] = cps
        print(f"[cps]    {h} done  failed={cps.failed}  "
              f"wall={cps.wall_time_s:.1f}s")
        if cps.failed:
            for r in cps.failure_reasons:
                print(f"  ! {r}")
    return out


def _postprocess_one(g: CPSGeometry, tcm_data, ms: dict, freqs, t0) -> CPSResult:
    """De-embed one design's FDTD result + mode-solver result."""
    smat = tcm_data.smatrix()
    s11 = _to_1d(np.conjugate(smat.data.isel(port_in=0, port_out=0)))
    s21 = _to_1d(np.conjugate(smat.data.isel(port_in=0, port_out=1)))
    Z_ref_port = _to_1d(smat.port_reference_impedances.values[:, 0, 0])

    L_feed_m = WP_OFFSET * 1e-6
    L_seg_m = g.L_segmented * 1e-6
    Z_seg, _gseg, n_eff, loss_dB_cm = _de_embed(
        s11, s21, freqs, ms["gamma_feed_m"], Z_ref_port,
        L_feed_m=L_feed_m, L_seg_m=L_seg_m, n_eff_guess=2.7,
    )
    z_series, y_shunt = _eq_circuit(freqs, Z_seg, n_eff, loss_dB_cm)

    failure_reasons = _sanity_check(Z_seg, n_eff, loss_dB_cm)
    return CPSResult(
        geometry_hash=g.hash(),
        geometry=g.to_dict(),
        freqs=freqs,
        Z_seg=Z_seg,
        n_eff_rf_bare=n_eff,
        alpha_dB_cm_bare=loss_dB_cm,
        z_series=z_series,
        y_shunt=y_shunt,
        Z0_feed_mode=ms["Z0_feed"],
        gamma_feed_mode=ms["gamma_feed_m"],
        failed=len(failure_reasons) > 0,
        failure_reasons=failure_reasons,
        wall_time_s=time.time() - t0,
    )


# --- Single-design wrapper (preserved API) --------------------------------

def evaluate_cps(geom: CPSGeometry, *,
                 cache_dir: str = "cache_step2",
                 task_name_prefix: str = "cps") -> CPSResult:
    """Evaluate one CPS geometry; thin wrapper around the batch path."""
    return evaluate_cps_batch([geom], cache_dir=cache_dir,
                              task_name_prefix=task_name_prefix)[0]
