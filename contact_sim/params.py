"""Every number the nut-on-rod contact simulation depends on, with where it came from.

Sources:
  rgb_fit      fitted from the two-view RGB capture (outputs/ur_nut/*.json); replace with calipers
  urdf         data/ur5e_robotiq/ur5e_robotiq85.urdf
  factory      Isaac Lab 2.0.2 Factory task settings (tuned for small-clearance contact)
  placeholder  a guess so the simulation runs; must be measured on the hardware (plan steps 0-1)
  choice       a design choice, not a property of the hardware

Measured values go in a JSON file with the same nesting (e.g. {"sensor": {"force_noise_n": 0.3}})
and are loaded with `load(path)`; `unmeasured(params)` lists what is still a placeholder or a fit.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


@dataclass
class Geometry:
    """Nut and rod. The bore clearance is the knob the experiments sweep (printed rings)."""
    ring_outer_r: float = 0.016079        # rgb_fit
    ring_bore_r: float = 0.010708         # rgb_fit
    ring_height: float = 0.015742         # rgb_fit
    ring_segments: int = 32               # choice: convex sectors approximating the ring
    rod_r: float = 0.0078                 # rgb_fit
    rod_z0: float = 0.0943                # rgb_fit (top of the spacers)
    rod_tip_z: float = 0.1801             # rgb_fit
    rod_chamfer: float = 0.0              # placeholder: unknown until measured
    rod_sides: int = 64                   # choice
    rod_xy: tuple[float, float] = (0.0857, -0.6381)  # rgb_fit
    spacer_r: float = 0.018               # rgb_fit
    spacer_z0: float = 0.0457             # rgb_fit
    table_z: float = -0.02                # rgb_fit (scene.json table_z)
    rod_threaded: bool | None = None      # placeholder: unknown

    @property
    def radial_clearance(self) -> float:
        return self.ring_bore_r - self.rod_r


@dataclass
class InHand:
    """The ring hangs from the gripper on a 6-DoF spring-damper (in-hand slip and rocking).

    grasp_model "finray" (default): the real gripper's fin-ray fingers (seen in
    earlier peg-insertion videos) pinch the ring's outside at their tips. The spring pivots at the
    middle of the gripped band, with its axes along the closing direction (x), across the
    fingers (y) and the ring axis (z), each with its own stiffness; the simulated fingers get
    visual-only fin-ray blades and close to touch the ring. grasp_model "center": the earlier
    runs (isotropic k_lin / k_ang about the ring centre, stock finger visuals, closing 0.55 rad).
    """
    mass: float = 0.02                    # placeholder: weigh the ring
    grasp_model: str = "finray"           # choice: "center" reproduces the runs before 2026-11-21 evening
    tip_below_ring_top: float = 0.013     # placeholder: generator grasp (TCP 17 mm over the top, tips 30 mm below TCP); measure
    k_lin_xyz: tuple[float, float, float] = (6000.0, 1500.0, 2500.0)   # placeholder N/m: closing, across, axial
    k_ang_xyz: tuple[float, float, float] = (0.3, 1.0, 0.3)            # placeholder N m/rad: about closing (rocking), across, ring axis
    d_lin_xyz: tuple[float, float, float] = (40.0, 20.0, 30.0)         # placeholder N s/m
    d_ang_xyz: tuple[float, float, float] = (0.005, 0.01, 0.005)       # placeholder N m s/rad
    finray_tip_below_tcp: float = 0.030   # placeholder: FINGERTIP_BELOW_TCP of the generator
    ring_bottom_below_flange: float | None = None  # measured grip: ring bottom below the tool flange (m); None = generator grasp
    finray_top_above_tcp: float = 0.038   # placeholder (video estimate): where the blade leaves the inner finger
    finray_thickness: tuple[float, float] = (0.016, 0.004)   # placeholder (video): at the top, at the tip
    finray_width: tuple[float, float] = (0.022, 0.014)       # placeholder (video): at the top, at the tip
    # grasp_model "center" (earlier runs)
    k_lin: float = 4000.0                 # placeholder N/m
    k_ang: float = 2.0                    # placeholder N m/rad
    d_lin: float = 40.0                   # placeholder N s/m
    d_ang: float = 0.02                   # placeholder N m s/rad


@dataclass
class Contact:
    static_friction: float = 0.4          # placeholder
    dynamic_friction: float = 0.4         # placeholder
    restitution: float = 0.0              # factory
    contact_offset: float = 0.002         # choice (factory uses 0.005; smaller keeps contact pairs local)
    rest_offset: float = 0.0              # factory


@dataclass
class Sensor:
    """Wrist wrench chain, applied in order: payload gravity, bias + drift, noise, reference point
    and frame, sample-and-hold, latency, causal low-pass."""
    rate_hz: float = 500.0                # placeholder: RTDE e-Series nominal; measure the delivered rate
    latency_s: float = 0.008              # placeholder
    force_noise_n: float = 0.5            # placeholder: per-axis std; UR quotes 4 N accuracy, not noise
    torque_noise_nm: float = 0.02         # placeholder
    force_bias_n: float = 1.0             # placeholder: per-episode bias std
    torque_bias_nm: float = 0.05          # placeholder
    drift_n_per_s: float = 0.05           # placeholder: force-bias random-walk std per sqrt(s)
    drift_nm_per_s: float = 0.0025        # placeholder: torque-bias random-walk std per sqrt(s)
    lowpass_hz: float = 20.0              # choice: the policy channel; raw channel is unfiltered
    gripper_mass: float = 0.925           # placeholder: 2F-85 datasheet ~0.9 kg (the URDF links sum to 0.41 kg)
    gripper_cog_tool0: tuple[float, float, float] = (0.0, 0.0, 0.07)  # placeholder (tool0 frame)
    payload_mass_error: float = 0.0       # sampled in randomisation: configured minus true payload mass
    payload_cog_error: tuple[float, float, float] = (0.0, 0.0, 0.0)  # sampled in randomisation: m, tool0 frame
    reference: str = "tcp"                # placeholder: where RTDE actual_TCP_force is referenced; verify
    frame: str = "base"                   # placeholder: frame RTDE expresses it in; verify
    sign: float = 1.0                     # placeholder: +1 = force the environment applies on the tool


@dataclass
class Controller:
    """Task-frame admittance around the reference, then differential IK to joint targets for a stiff
    position-controlled arm (how admittance runs on a UR through servoJ)."""
    rate_hz: float = 120.0                # choice
    mass: tuple[float, ...] = (2.0, 2.0, 2.0, 0.02, 0.02, 0.02)          # choice, task frame x y z rx ry rz
    damping: tuple[float, ...] = (300.0, 300.0, 400.0, 2.0, 2.0, 2.0)    # choice
    stiffness: tuple[float, ...] = (600.0, 600.0, 0.0, 6.0, 6.0, 6.0)    # choice: z free (damping control)
    max_offset: tuple[float, ...] = (0.01, 0.01, 0.03, 0.2, 0.2, 0.2)    # choice: m, rad
    arm_tracking_tau_s: float = 0.0       # placeholder: first-order lag on joint targets (0 = none)
    deadband: tuple[float, ...] = (0.5, 0.5, 0.5, 0.03, 0.03, 0.03)  # choice: N, N m; ignore sensor noise and drift


@dataclass
class Sim:
    dt: float = 1 / 240                   # choice (factory runs 1/120 with SDF collision)
    position_iterations: int = 192        # factory
    velocity_iterations: int = 1          # factory
    bounce_threshold_velocity: float = 0.2  # factory
    friction_offset_threshold: float = 0.01  # factory
    friction_correlation_distance: float = 0.00625  # factory
    gpu_max_num_partitions: int = 1       # factory: "important for stable simulation"


@dataclass
class Params:
    geometry: Geometry = field(default_factory=Geometry)
    in_hand: InHand = field(default_factory=InHand)
    contact: Contact = field(default_factory=Contact)
    sensor: Sensor = field(default_factory=Sensor)
    controller: Controller = field(default_factory=Controller)
    sim: Sim = field(default_factory=Sim)


SOURCES = {
    "geometry": "rgb_fit except ring_segments/rod_sides (choice) and rod_chamfer/rod_threaded (placeholder)",
    "in_hand": "placeholder",
    "contact": "placeholder friction; factory restitution/rest offset",
    "sensor": "placeholder except lowpass_hz (choice)",
    "controller": "choice except arm_tracking_tau_s (placeholder)",
    "sim": "factory except dt (choice)",
}


def _apply(obj, over: dict, path: str = ""):
    for k, v in over.items():
        if k.startswith("_") or v is None:  # notes and unfilled fields in measurement files
            continue
        if not hasattr(obj, k):
            raise KeyError(f"unknown parameter {path}{k}")
        cur = getattr(obj, k)
        if is_dataclass(cur):
            _apply(cur, v, f"{path}{k}.")
        else:
            setattr(obj, k, tuple(v) if isinstance(cur, tuple) else v)


def load(path: Path | None = None, overrides: dict | None = None) -> Params:
    p = Params()
    if path is not None and Path(path).exists():
        _apply(p, json.loads(Path(path).read_text()))
    if overrides:
        _apply(p, overrides)
    return p


def unmeasured(p: Params, measured: dict | None = None) -> list[str]:
    """Hardware-dependent parameters not yet set from a measurement file."""
    measured = measured or {}
    out = []
    for group in ("geometry", "in_hand", "contact", "sensor"):
        for f in fields(getattr(p, group)):
            if f.name in ("ring_segments", "rod_sides", "lowpass_hz", "payload_mass_error", "payload_cog_error"):
                continue
            if f.name not in measured.get(group, {}):
                out.append(f"{group}.{f.name}")
    return out


def to_dict(p: Params) -> dict:
    return asdict(p)
