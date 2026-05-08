"""PN junction modulator characterization.

Refactored from TWModulator_VpiL_Loss.ipynb. Preserves all physics; only
structural changes: wraps the workflow in `evaluate_design(mult)` and caches
the heavy simulation outputs to disk so re-runs don't re-bill Tidy3D.

The single control variable is `mult`: a scalar that scales both core dopings
together as p_doping = 5e17 * mult, n_doping = 3e17 * mult. Everything else
(geometry, access dopings, voltage sweep, mesh, optical stack) is fixed.

Outputs at 9 reverse-bias voltages from -0.5 to 1.5 V:
  * C  [pF/cm]   capacitance per unit length
  * R  [Ohm.cm]  series resistance per unit length (computed at bias=0,
                 reused across voltages -- matches notebook behavior)
  * f3dB [GHz]   1 / (2 pi R C)
  * VpiL [V.cm]  pi / (d phi/dV per cm).  NaN at first/last voltage because
                 np.gradient uses one-sided differences there and the
                 endpoints are unreliable.
  * loss [dB/cm] from 4 pi Im(n_eff) / wavelength
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import photonforge as pf
import siepic_forge as siepic
import tidy3d as td
from tidy3d import web

from laplace import LaplaceSolver

td.config.logging_level = "ERROR"


# ---------------------------------------------------------------------------
# Fixed geometry  (all units in micrometers)
# ---------------------------------------------------------------------------
W_CORE = 0.5
H_CORE = 0.22
W_CLEARANCE = 2.0
H_CLEARANCE = 0.09
W_SIDE = 1.0
H_SIDE = 0.22

TOX_THICKNESS = 1.2
BOX_THICKNESS = 2.0
OXIDE_THICKNESS = TOX_THICKNESS + BOX_THICKNESS
TL_THICKNESS = 2.0

# RF / contacts
SIGNAL_WIDTH = 7.0
GAP_WIDTH = 3.0
GROUND_WIDTH = 75.0
W_CONTACT = 1.0
H_CONTACT = TOX_THICKNESS - H_CORE

W_TOT = 2 * W_SIDE + 2 * W_CLEARANCE + W_CORE
RES = H_CLEARANCE / 10  # mesh resolution unit

# Heavily-doped boundary positions (notebook constants)
Y_P_P = -W_CORE / 2 - 0.2
Y_N_P = W_CORE / 2 + 0.2
Y_P_PP = -W_CORE / 2 - 0.9
Y_N_PP = W_CORE / 2 + 0.9

# Fixed access dopings (only the *core* p/n dopings vary)
P_P_DOPING = 1.5e19
N_P_DOPING = 1.2e19
P_PP_DOPING = 1.0e20
N_PP_DOPING = 1.0e20

# Voltage sweep -- always the same 9 points; agent never changes this
VOLTAGES = np.linspace(-0.5, 1.5, 9)
INTERIOR_MASK = np.ones_like(VOLTAGES, dtype=bool)
INTERIOR_MASK[0] = False
INTERIOR_MASK[-1] = False  # endpoints are unreliable for VpiL gradient

# Optical
WVL_UM = 1.55
FREQ0 = td.C_0 / WVL_UM


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class DesignResult:
    """Per-design simulation results.  Arrays have shape (9,) over VOLTAGES."""
    mult: float
    p_doping: float
    n_doping: float
    voltages: np.ndarray
    C_pF_cm: np.ndarray         # capacitance per cm at each voltage
    R_ohm_cm: np.ndarray        # series resistance, per voltage (was scalar)
    f3dB_GHz: np.ndarray        # 1 / (2 pi R(V) C(V))
    VpiL_V_cm: np.ndarray       # pi / (dphi/dV per cm); endpoints kept as-is
    loss_dB_cm: np.ndarray      # from Im(n_eff)
    n_eff_baseline: complex     # at first voltage; phase reference
    x_p_m: np.ndarray           # depletion edge p-side, per voltage
    x_n_m: np.ndarray           # depletion edge n-side, per voltage
    charge_task_id: str
    mode_solver_batch_dir: str

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
            elif isinstance(v, complex):
                d[k] = {"real": v.real, "imag": v.imag}
        return d


# ---------------------------------------------------------------------------
# Geometry construction
# ---------------------------------------------------------------------------
def _build_doping(p_doping: float, n_doping: float) -> td.MultiPhysicsMedium:
    """Build the doped-silicon multiphysics medium for given core dopings."""
    acceptor_p_p = td.GaussianDoping.from_bounds(
        rmin=[-np.inf, Y_P_PP, -0.3],
        rmax=[np.inf, Y_P_P + 0.003, H_CLEARANCE],
        concentration=P_P_DOPING, ref_con=1e6, width=0.001, source="zmax",
    )
    donor_n_p = td.GaussianDoping.from_bounds(
        rmin=[-np.inf, Y_N_P - 0.003, -0.3],
        rmax=[np.inf, Y_N_PP, H_CLEARANCE],
        concentration=N_P_DOPING, ref_con=1e6, width=0.001, source="zmax",
    )
    acceptor_p_pp = td.GaussianDoping.from_bounds(
        rmin=[-np.inf, -W_TOT, -0.3],
        rmax=[np.inf, Y_P_PP + 0.003, H_CORE],
        concentration=P_PP_DOPING, ref_con=1e6, width=0.001, source="zmax",
    )
    donor_n_plus = td.GaussianDoping.from_bounds(
        rmin=[-np.inf, Y_N_PP - 0.003, -0.3],
        rmax=[np.inf, W_TOT, H_CORE],
        concentration=N_PP_DOPING, ref_con=1e6, width=0.001, source="zmax",
    )
    acceptor_p = td.GaussianDoping.from_bounds(
        rmin=[-np.inf, Y_P_P, 0.0],
        rmax=[np.inf, -W_CORE / 2 + 0.003, H_CORE],
        concentration=p_doping, ref_con=1e6, width=0.001, source="zmax",
    )
    donor_n = td.GaussianDoping.from_bounds(
        rmin=[-np.inf, W_CORE / 2 - 0.003, 0.0],
        rmax=[np.inf, Y_N_P, H_CORE],
        concentration=n_doping, ref_con=1e6, width=0.001, source="zmax",
    )
    acceptor_p_wg = td.GaussianDoping.from_bounds(
        rmin=[-np.inf, -W_CORE / 2, 0.0],
        rmax=[np.inf, 0.0, H_CORE],
        concentration=p_doping, ref_con=1e6, width=0.001, source="zmax",
    )
    donor_n_wg = td.GaussianDoping.from_bounds(
        rmin=[-np.inf, 0.0, 0.0],
        rmax=[np.inf, W_CORE / 2, H_CORE],
        concentration=n_doping, ref_con=1e6, width=0.001, source="zmax",
    )

    intrinsic_si = td.material_library["cSi"].variants["Si_MultiPhysics"].medium
    return td.MultiPhysicsMedium(
        charge=intrinsic_si.charge.updated_copy(
            N_d=[donor_n_plus, donor_n_p, donor_n, donor_n_wg],
            N_a=[acceptor_p_pp, acceptor_p_p, acceptor_p, acceptor_p_wg],
        ),
        name="Si_doping",
    )


def _build_structures(si_doping: td.MultiPhysicsMedium) -> tuple:
    """Returns (all_structures, contact_p, contact_n, oxide_name)."""
    sio2_charge = td.MultiPhysicsMedium(
        optical=td.material_library["SiO2"]["Palik_Lossless"],
        charge=td.ChargeInsulatorMedium(permittivity=4.2),
        name="SiO2",
    )
    al = td.MultiPhysicsMedium(
        charge=td.ChargeConductorMedium(conductivity=38),
        name="Aluminium",
    )

    oxide = td.Structure(
        geometry=td.Box(
            center=(0, 0, OXIDE_THICKNESS / 2 - BOX_THICKNESS),
            size=(td.inf, td.inf, OXIDE_THICKNESS),
        ),
        medium=sio2_charge, name="oxide",
    )
    core = td.Structure(
        geometry=td.Box(center=(0, 0, H_CORE / 2),
                        size=(td.inf, W_CORE, H_CORE)),
        medium=si_doping, name="core",
    )
    slab = td.Structure(
        geometry=td.Box(center=(0, 0, H_CLEARANCE / 2),
                        size=(td.inf, W_TOT, H_CLEARANCE)),
        medium=si_doping, name="slab",
    )
    side_p = td.Structure(
        geometry=td.Box(
            center=(0, -W_CORE / 2 - W_CLEARANCE - W_SIDE / 2, H_SIDE / 2),
            size=(td.inf, W_SIDE, H_SIDE),
        ),
        medium=si_doping, name="side_p",
    )
    side_n = td.Structure(
        geometry=td.Box(
            center=(0, W_CORE / 2 + W_CLEARANCE + W_SIDE / 2, H_SIDE / 2),
            size=(td.inf, W_SIDE, H_SIDE),
        ),
        medium=si_doping, name="side_n",
    )
    contact_p = td.Structure(
        geometry=td.Box(
            center=(0, -W_CORE / 2 - W_CLEARANCE - W_SIDE + W_CONTACT / 2,
                    H_SIDE + H_CONTACT / 2),
            size=(td.inf, W_CONTACT, H_CONTACT),
        ),
        medium=al, name="contact_p",
    )
    contact_n = td.Structure(
        geometry=td.Box(
            center=(0, W_CORE / 2 + W_CLEARANCE + W_SIDE - W_CONTACT / 2,
                    H_SIDE + H_CONTACT / 2),
            size=(td.inf, W_CONTACT, H_CONTACT),
        ),
        medium=al, name="contact_n",
    )

    return [oxide, core, slab, side_p, side_n, contact_p, contact_n], contact_p, contact_n


def _build_mesh(oxide_name: str) -> td.DistanceUnstructuredGrid:
    """Reproduce the notebook's mesh refinement, all keyed off RES."""
    dl_b = RES * 0.12
    dist_b = dl_b * 2
    dist_bulk = dist_b * 20

    regions = [
        td.GridRefinementRegion(
            center=(0, 0.0, H_CORE / 2),
            size=(0, W_CORE, H_CORE),
            dl_internal=RES * 0.5, transition_thickness=RES * 60,
        ),
        td.GridRefinementRegion(
            center=(0, Y_N_P, H_CLEARANCE / 2),
            size=(0, 0.05, H_CLEARANCE),
            dl_internal=RES * 0.4, transition_thickness=RES * 40,
        ),
        td.GridRefinementRegion(
            center=(0, Y_P_P, H_CLEARANCE / 2),
            size=(0, 0.05, H_CLEARANCE),
            dl_internal=RES * 0.4, transition_thickness=RES * 40,
        ),
        td.GridRefinementRegion(
            center=(0, Y_N_PP, H_CLEARANCE / 2),
            size=(0, 0.05, H_CLEARANCE),
            dl_internal=RES * 0.4, transition_thickness=RES * 40,
        ),
        td.GridRefinementRegion(
            center=(0, Y_P_PP, H_CLEARANCE / 2),
            size=(0, 0.05, H_CLEARANCE),
            dl_internal=RES * 0.4, transition_thickness=RES * 40,
        ),
        td.GridRefinementRegion(
            center=(0, -W_TOT / 2 + W_CONTACT / 2, H_SIDE + H_CONTACT / 2),
            size=(0, W_CONTACT, H_CONTACT),
            dl_internal=RES * 2, transition_thickness=RES * 50,
        ),
        td.GridRefinementRegion(
            center=(0, W_TOT / 2 - W_CONTACT / 2, H_SIDE + H_CONTACT / 2),
            size=(0, W_CONTACT, H_CONTACT),
            dl_internal=RES * 2, transition_thickness=RES * 50,
        ),
    ]
    lines = [
        td.GridRefinementLine(r1=(0, -W_TOT / 2, 0.0), r2=(0, W_TOT / 2, 0.0),
                              dl_near=dl_b, distance_near=dist_b, distance_bulk=dist_bulk),
        td.GridRefinementLine(r1=(0, -(W_CORE / 2 + W_CLEARANCE), H_CLEARANCE),
                              r2=(0, -W_CORE / 2, H_CLEARANCE),
                              dl_near=dl_b, distance_near=dist_b, distance_bulk=dist_bulk),
        td.GridRefinementLine(r1=(0, W_CORE / 2, H_CLEARANCE),
                              r2=(0, W_CORE / 2 + W_CLEARANCE, H_CLEARANCE),
                              dl_near=dl_b, distance_near=dist_b, distance_bulk=dist_bulk),
        td.GridRefinementLine(r1=(0, -W_CORE / 2, H_CORE), r2=(0, W_CORE / 2, H_CORE),
                              dl_near=dl_b, distance_near=dist_b, distance_bulk=dist_bulk),
        td.GridRefinementLine(r1=(0, -W_CORE / 2, H_CLEARANCE), r2=(0, -W_CORE / 2, H_CORE),
                              dl_near=dl_b, distance_near=dist_b, distance_bulk=dist_bulk),
        td.GridRefinementLine(r1=(0, W_CORE / 2, H_CLEARANCE), r2=(0, W_CORE / 2, H_CORE),
                              dl_near=dl_b, distance_near=dist_b, distance_bulk=dist_bulk),
        td.GridRefinementLine(r1=(0, -W_TOT / 2, H_SIDE),
                              r2=(0, -W_TOT / 2 + W_SIDE, H_SIDE),
                              dl_near=dl_b, distance_near=dist_b, distance_bulk=dist_bulk),
        td.GridRefinementLine(r1=(0, W_TOT / 2, H_SIDE),
                              r2=(0, W_TOT / 2 - W_SIDE, H_SIDE),
                              dl_near=dl_b, distance_near=dist_b, distance_bulk=dist_bulk),
        td.GridRefinementLine(r1=(0, -W_TOT / 2 + W_SIDE, H_CLEARANCE),
                              r2=(0, -W_TOT / 2 + W_SIDE, H_SIDE),
                              dl_near=dl_b, distance_near=dist_b, distance_bulk=dist_bulk),
        td.GridRefinementLine(r1=(0, W_TOT / 2 - W_SIDE, H_CLEARANCE),
                              r2=(0, W_TOT / 2 - W_SIDE, H_SIDE),
                              dl_near=dl_b, distance_near=dist_b, distance_bulk=dist_bulk),
    ]
    return td.DistanceUnstructuredGrid(
        dl_interface=RES * 1.2,
        dl_bulk=RES * 20,
        distance_interface=RES * 1,
        distance_bulk=dist_bulk,
        relative_min_dl=0,
        sampling=500,
        non_refined_structures=[oxide_name],
        mesh_refinements=regions + lines,
    )


# ---------------------------------------------------------------------------
# Resistance via Caughey-Thomas + 2D Laplace
#
# Convention: in this device, "voltage" is the bias on the n-contact while the
# p-contact is grounded (see notebook Cell 16). So **positive voltage = reverse
# bias**, negative voltage = forward bias.  The depletion-region formula uses
# V_total = V_bi + voltage, clipped to a small positive value to avoid the
# depletion approximation collapsing under near-flat-band forward bias.
# ---------------------------------------------------------------------------
def _caughey_thomas(mu_max, mu_min, ref_N, exp_N, N):
    """Caughey-Thomas mobility (T=300 K).  N in cm^-3, returns cm^2/(V s)."""
    return mu_min + (mu_max - mu_min) / (1 + (N / ref_N) ** exp_N)


# Physical constants (SI)
_Q = 1.602e-19
_EPS0 = 8.854e-12
_EPS_S = _EPS0 * 11.7
_KB = 1.381e-23
_T = 300.0
_NI = 1.0e16  # m^-3, intrinsic Si density


def _depletion_edges_m(p_doping: float, n_doping: float,
                       voltage: float) -> tuple[float, float, float, float]:
    """Return (Wd, x_p, x_n, V_total) in meters / volts at given reverse bias."""
    N_A = p_doping * 1e6  # cm^-3 -> m^-3
    N_D = n_doping * 1e6
    V_bi = (_KB * _T / _Q) * np.log((N_A * N_D) / (_NI ** 2))
    # Clip to avoid V_total going to zero or negative under forward bias.
    V_total = max(V_bi + voltage, V_bi * 0.05)
    Wd = np.sqrt(2 * _EPS_S * V_total * (N_A + N_D) / (N_A * N_D * _Q))
    x_p = -Wd * (N_D / (N_A + N_D))
    x_n = Wd * (N_A / (N_A + N_D))
    return Wd, x_p, x_n, V_total


def _series_resistance_at_voltage(p_doping: float, n_doping: float,
                                  voltage: float, *,
                                  device_depth_um: float = 10000.0,
                                  laplace_grid: int = 100) -> dict:
    """Series resistance R [Ohm.cm] at a single reverse-bias voltage.

    Hybrid model: analytical for slab/access regions, 2D Laplace for the two
    channel regions whose lengths shrink with reverse bias.  Returns a dict
    with all 8 sub-resistances, total, and depletion edges (the latter are
    used to place mode-solver mesh refinements).
    """
    N_A = p_doping * 1e6
    N_D = n_doping * 1e6

    mu_n = _caughey_thomas(1471, 52.2, 9.68e16, 0.68, n_doping) * 1e-4
    mu_p = _caughey_thomas(470.5, 44.9, 2.23e17, 0.719, p_doping) * 1e-4
    mu_n_p = _caughey_thomas(1471, 52.2, 9.68e16, 0.68, N_P_DOPING) * 1e-4
    mu_p_p = _caughey_thomas(470.5, 44.9, 2.23e17, 0.719, P_P_DOPING) * 1e-4
    mu_n_pp = _caughey_thomas(1471, 52.2, 9.68e16, 0.68, N_PP_DOPING) * 1e-4
    mu_p_pp = _caughey_thomas(470.5, 44.9, 2.23e17, 0.719, P_PP_DOPING) * 1e-4

    w = W_CORE * 1e-6
    t_slab = H_CLEARANCE * 1e-6
    t_chan = H_CORE * 1e-6
    L = device_depth_um * 1e-6

    Wd, x_p, x_n, V_total = _depletion_edges_m(p_doping, n_doping, voltage)

    # Channel lengths (voltage-dependent through x_p, x_n)
    l1 = -w / 2 - Y_P_P * 1e-6
    l2 = x_p + w / 2          # depends on voltage
    l3 = w / 2 - x_n          # depends on voltage
    l4 = Y_N_P * 1e-6 - w / 2
    l5 = -(Y_P_PP - Y_P_P) * 1e-6
    l6 = (Y_N_PP - Y_N_P) * 1e-6
    l7 = (W_CLEARANCE + Y_P_PP) * 1e-6 + w / 2
    l8 = (W_CLEARANCE - Y_N_PP) * 1e-6 + w / 2

    # Guard: under strong reverse bias the channel can disappear (l2 or l3
    # < 0 means depletion has eaten the entire core/slab boundary).  Clip at
    # a small positive value -- the resulting Rp/Rn will be tiny but finite.
    l2_eff = max(l2, 1e-9)
    l3_eff = max(l3, 1e-9)

    den1 = _Q * mu_p * N_A * t_slab * L
    den4 = _Q * mu_n * N_D * t_slab * L
    den5 = _Q * mu_p_p * (P_P_DOPING * 1e6) * t_slab * L
    den6 = _Q * mu_n_p * (N_P_DOPING * 1e6) * t_slab * L
    den7 = _Q * mu_p_pp * (P_PP_DOPING * 1e6) * t_slab * L
    den8 = _Q * mu_n_pp * (N_PP_DOPING * 1e6) * t_slab * L

    sigma_p = _Q * mu_p * N_A
    sigma_n = _Q * mu_n * N_D

    solver_p = LaplaceSolver(w=l2_eff, h=t_chan, ny=laplace_grid, nz=laplace_grid)
    solver_p.set_bc(side="left", value=1.0, start=0, end=t_slab)
    solver_p.set_bc(side="right", value=0.0)
    solver_p.solve()

    solver_n = LaplaceSolver(w=l3_eff, h=t_chan, ny=laplace_grid, nz=laplace_grid)
    solver_n.set_bc(side="left", value=1.0, start=0, end=t_slab)
    solver_n.set_bc(side="right", value=0.0)
    solver_n.solve()

    Rp = 1 / (L * solver_p.calculate_current_across_y(l2_eff / 2 * 0.95, sigma_p))
    Rn = 1 / (L * solver_n.calculate_current_across_y(l3_eff / 2 * 0.95, sigma_n))

    R = {
        "R1_p_slab":   l1 / den1,
        "R2_p_chan":   Rp,
        "R3_n_chan":   Rn,
        "R4_n_slab":   l4 / den4,
        "R5_p_p_slab": l5 / den5,
        "R6_n_p_slab": l6 / den6,
        "R7_p_pp_slab": l7 / den7,
        "R8_n_pp_slab": l8 / den8,
    }
    R["R_total_ohm_cm"] = sum(R.values())
    R["x_p_m"] = x_p
    R["x_n_m"] = x_n
    R["V_total_V"] = V_total
    R["voltage_V"] = voltage
    return R


def _series_resistance_sweep(p_doping: float, n_doping: float, *,
                             cache_path: Path | None = None) -> list[dict]:
    """Compute R(V) at every voltage in VOLTAGES.  Cached on disk."""
    if cache_path is not None and cache_path.exists():
        with cache_path.open("rb") as f:
            print(f"[R(V)]   cache hit: {cache_path.name}")
            return pickle.load(f)
    out = [_series_resistance_at_voltage(p_doping, n_doping, float(v))
           for v in VOLTAGES]
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as f:
            pickle.dump(out, f)
        print(f"[R(V)]   cached -> {cache_path.name}")
    return out


# ---------------------------------------------------------------------------
# Charge simulation
# ---------------------------------------------------------------------------
def _run_charge_sim(p_doping: float, n_doping: float, *,
                    cache_path: Path,
                    task_name: str = "charge_junction"):
    """Run (or load cached) charge simulation. Returns (data, monitor_names)."""
    monitor_names = {"carrier": "carriers", "cap": "capacitance_global_mnt"}

    if cache_path.exists():
        with cache_path.open("rb") as f:
            print(f"[charge] cache hit: {cache_path.name}")
            return pickle.load(f), monitor_names

    si_doping = _build_doping(p_doping, n_doping)
    structures, contact_p, contact_n = _build_structures(si_doping)

    air = td.MultiPhysicsMedium(heat=td.FluidSpec(), name="air")
    carrier_mnt = td.SteadyFreeCarrierMonitor(
        center=(0, 0, 0), size=(td.inf, td.inf, td.inf),
        name=monitor_names["carrier"], unstructured=True,
    )
    cap_mnt = td.SteadyCapacitanceMonitor(
        center=(0, 0, 0.11), size=(0, td.inf, td.inf),
        name=monitor_names["cap"],
    )

    bc_p = td.HeatChargeBoundarySpec(
        condition=td.VoltageBC(source=td.DCVoltageSource(voltage=0)),
        placement=td.StructureBoundary(structure=contact_p.name),
    )
    bc_n = td.HeatChargeBoundarySpec(
        condition=td.VoltageBC(source=td.DCVoltageSource(voltage=VOLTAGES)),
        placement=td.StructureBoundary(structure=contact_n.name),
    )

    convergence = td.ChargeToleranceSpec(
        rel_tol=1e-5, abs_tol=1e5, max_iters=400, ramp_up_iters=1
    )
    analysis = td.IsothermalSteadyChargeDCAnalysis(
        temperature=300, convergence_dv=10, tolerance_settings=convergence,
    )

    sim = td.HeatChargeSimulation(
        sources=[],
        monitors=[carrier_mnt, cap_mnt],
        analysis_spec=analysis,
        center=(0, 0, 0), size=(0, 10, 5),
        structures=structures, medium=air,
        boundary_spec=[bc_p, bc_n],
        grid_spec=_build_mesh("oxide"),
        symmetry=(0, 0, 0),
    )

    print(f"[charge] submitting task '{task_name}' to Tidy3D...")
    data = web.run(sim, task_name=task_name)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump(data, f)
    print(f"[charge] cached -> {cache_path.name}")
    return data, monitor_names


# ---------------------------------------------------------------------------
# Optical mode-solver sweep
# ---------------------------------------------------------------------------
def _run_mode_solver_batch(charge_data, monitor_names: dict,
                           x_p_m: float, x_n_m: float, *,
                           cache_path: Path,
                           batch_dir: str = "data_batch_modesolver"):
    """Run mode solver at each of VOLTAGES, return complex n_eff array."""
    if cache_path.exists():
        with cache_path.open("rb") as f:
            print(f"[mode]   cache hit: {cache_path.name}")
            return pickle.load(f)

    si = td.material_library["cSi"]["Palik_Lossless"]
    n_si, k_si = si.nk_model(frequency=FREQ0)
    si_non_perturb = td.Medium.from_nk(n=n_si, k=k_si, freq=FREQ0)
    perturbation_model = td.NedeljkovicSorefMashanovich(ref_freq=FREQ0)
    si_perturb = td.PerturbationMedium.from_unperturbed(
        medium=si_non_perturb,
        perturbation_spec=td.IndexPerturbation(
            delta_n=perturbation_model.delta_n(),
            delta_k=perturbation_model.delta_k(),
            freq=FREQ0,
        ),
    )

    tl_metal = {
        "optical": td.material_library["Al"]["Rakic1995"],
        "electrical": td.LossyMetalMedium(
            conductivity=38, frequency_range=[0.1e9, 200e9],
            fit_param=td.SurfaceImpedanceFitterParam(max_num_poles=16),
        ),
    }

    wg_spec = pf.PortSpec(
        description="Rib wg",
        width=3,
        limits=(-1, 1.22),
        num_modes=1,
        target_neff=3.5,
        path_profiles=[(W_CORE, 0, (1, 0)), (W_TOT, 0, (2, 0))],
    )

    mode_solvers = {}
    voltage_keys = []
    refine_geo_p = td.Box(center=(0, x_p_m * 1e6, H_CORE / 2),
                          size=(td.inf, 0.005, H_CORE))
    refine_geo_n = td.Box(center=(0, x_n_m * 1e6, H_CORE / 2),
                          size=(td.inf, 0.005, H_CORE))
    mesh_override_dl = (0.01, 0.001, 0.01)
    refine_box_p = td.MeshOverrideStructure(geometry=refine_geo_p, dl=mesh_override_dl)
    refine_box_n = td.MeshOverrideStructure(geometry=refine_geo_n, dl=mesh_override_dl)
    refined_grid_spec = td.GridSpec.auto(
        wavelength=WVL_UM, min_steps_per_wvl=20,
        override_structures=[refine_box_p, refine_box_n],
    )

    for v in VOLTAGES:
        e_data = charge_data[monitor_names["carrier"]].electrons.sel(voltage=v)
        h_data = charge_data[monitor_names["carrier"]].holes.sel(voltage=v)
        si_perturb_new = si_perturb.perturbed_copy(
            electron_density=e_data, hole_density=h_data
        )
        perturbed_tech = siepic.ebeam(
            top_oxide_thickness=TOX_THICKNESS,
            bottom_oxide_thickness=BOX_THICKNESS,
            si_thickness=H_CORE,
            si_slab_thickness=H_CLEARANCE,
            si=si_perturb_new,
            use_parametric_cache=False,
        )
        ms = wg_spec.to_tidy3d(frequencies=[FREQ0], technology=perturbed_tech)
        sim = ms.simulation.copy(update={"grid_spec": refined_grid_spec})
        ms = ms.copy(update={"simulation": sim})
        key = f"voltage_{v:.4f}"
        voltage_keys.append(key)
        mode_solvers[key] = ms

    print(f"[mode]   submitting batch of {len(mode_solvers)} simulations...")
    batch = web.Batch(simulations=mode_solvers, verbose=True)
    batch_results = batch.run(path_dir=batch_dir)

    n_eff_freq0 = []
    for key in voltage_keys:
        sim_data = batch_results[key]
        n_eff_freq0.append(sim_data.n_complex.isel(mode_index=0).item())
    n_eff_array = np.array(n_eff_freq0)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump(n_eff_array, f)
    print(f"[mode]   cached -> {cache_path.name}")
    return n_eff_array


# ---------------------------------------------------------------------------
# Top-level evaluation
# ---------------------------------------------------------------------------
def _setup_technology():
    """Set the default photonforge technology if not already set."""
    if pf.config.default_technology is None:
        tl_metal = {
            "optical": td.material_library["Al"]["Rakic1995"],
            "electrical": td.LossyMetalMedium(
                conductivity=38, frequency_range=[0.1e9, 200e9],
                fit_param=td.SurfaceImpedanceFitterParam(max_num_poles=16),
            ),
        }
        tech = siepic.ebeam(
            si_thickness=H_CORE,
            si_slab_thickness=H_CLEARANCE,
            metal_si_separation=TOX_THICKNESS - H_CORE,
            router_thickness=TL_THICKNESS,
            top_oxide_thickness=TOX_THICKNESS,
            bottom_oxide_thickness=BOX_THICKNESS,
            include_top_opening=True,
            include_substrate=True,
            router_metal=tl_metal,
            heater_metal=tl_metal,
        )
        pf.config.default_technology = tech


def _mult_to_label(mult: float) -> str:
    """Stable string label for caching, e.g. mult=5.0 -> 'mult_05_000'."""
    return f"mult_{mult:08.3f}".replace(".", "_")


def evaluate_design(mult: float, *, cache_dir: str = "cache") -> DesignResult:
    """Evaluate a single (mult,) PN junction design across the voltage sweep.

    Reuses cached results if available; otherwise submits jobs to Tidy3D.
    """
    _setup_technology()

    p_doping = 5e17 * mult
    n_doping = 3e17 * mult
    label = _mult_to_label(mult)

    cache_root = Path(cache_dir)
    charge_cache = cache_root / f"{label}_charge.pkl"
    mode_cache = cache_root / f"{label}_modes.pkl"
    rsweep_cache = cache_root / f"{label}_R_sweep.pkl"
    batch_dir = str(cache_root / f"{label}_batch")

    # 1. Charge simulation
    charge_data, monitor_names = _run_charge_sim(
        p_doping, n_doping, cache_path=charge_cache,
        task_name=f"charge_{label}",
    )

    # 2. Capacitance from charge result
    cap_mnt = charge_data[monitor_names["cap"]]
    mnt_v = np.array(cap_mnt.electron_capacitance.coords["v"].data)
    mnt_ce = np.array(cap_mnt.electron_capacitance.data)
    mnt_ch = np.array(cap_mnt.hole_capacitance.data)
    C_pF_cm = -0.5 * (mnt_ce + mnt_ch) * 10  # pF/cm

    # Sanity: capacitance voltages should match VOLTAGES
    if not np.allclose(mnt_v, VOLTAGES, rtol=1e-3):
        raise RuntimeError(
            f"Capacitance voltage axis mismatch.  Expected {VOLTAGES}, got {mnt_v}"
        )

    # 3. Voltage-dependent series resistance (analytical + Laplace per V)
    R_sweep = _series_resistance_sweep(p_doping, n_doping, cache_path=rsweep_cache)
    R_arr = np.array([r["R_total_ohm_cm"] for r in R_sweep])
    x_p_arr = np.array([r["x_p_m"] for r in R_sweep])
    x_n_arr = np.array([r["x_n_m"] for r in R_sweep])
    f3dB_GHz = 1e12 / (2 * np.pi * R_arr * np.array(C_pF_cm)) / 1e9

    # 4. Mode-solver batch.  We use the bias=0 depletion edges to position the
    # mesh-refinement boxes (these are "where to refine the optical mesh"
    # heuristics, not the physics; using a single representative position
    # keeps caching simple and matches the notebook).
    arg_v0 = int(np.argmin(np.abs(VOLTAGES)))
    n_eff_arr = _run_mode_solver_batch(
        charge_data, monitor_names,
        x_p_m=x_p_arr[arg_v0], x_n_m=x_n_arr[arg_v0],
        cache_path=mode_cache, batch_dir=batch_dir,
    )

    # 5. VpiL and loss from complex n_eff.  Endpoints are kept as-is (less
    # accurate one-sided gradient, but useful for diagnostics).
    delta_neff = np.real(n_eff_arr - n_eff_arr[0])
    rel_phase_change = 2 * np.pi * delta_neff / WVL_UM * 1e4  # rad/cm
    dphiL_dv = np.gradient(rel_phase_change, VOLTAGES)
    with np.errstate(divide="ignore", invalid="ignore"):
        VpiL = np.where(np.abs(dphiL_dv) > 1e-12, np.pi / dphiL_dv, np.nan)

    loss_dB_cm = (10 * 4 * np.pi * np.imag(n_eff_arr) / WVL_UM
                  * 1e4 * np.log10(np.exp(1)))

    return DesignResult(
        mult=float(mult),
        p_doping=float(p_doping),
        n_doping=float(n_doping),
        voltages=VOLTAGES.copy(),
        C_pF_cm=np.array(C_pF_cm),
        R_ohm_cm=R_arr,
        f3dB_GHz=np.array(f3dB_GHz),
        VpiL_V_cm=np.array(VpiL),
        loss_dB_cm=np.array(loss_dB_cm),
        n_eff_baseline=complex(n_eff_arr[0]),
        x_p_m=x_p_arr,
        x_n_m=x_n_arr,
        charge_task_id=str(charge_cache),
        mode_solver_batch_dir=batch_dir,
    )


if __name__ == "__main__":
    # Quick smoke test if run directly
    import sys
    mult = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    print(f"Evaluating mult={mult}...")
    result = evaluate_design(mult)
    print("\nResults (per voltage):")
    print(f"  voltages = {result.voltages}")
    print(f"  C        = {result.C_pF_cm}")
    print(f"  R        = {result.R_ohm_cm}")
    print(f"  f3dB     = {result.f3dB_GHz}")
    print(f"  VpiL     = {result.VpiL_V_cm}")
    print(f"  loss     = {result.loss_dB_cm}")
