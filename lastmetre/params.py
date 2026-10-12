"""All tunable parameters in one place.

Units are SI (metres, seconds, radians). Values are chosen for a small
laboratory-scale scenario, not for any real device or procedure.
"""
from dataclasses import dataclass


@dataclass
class Params:
    # --- field ---------------------------------------------------------
    field_w: float = 12.0           # suspected hazardous area, x extent (m)
    field_h: float = 12.0           # suspected hazardous area, y extent (m)
    perimeter: float = 2.0          # cleared access strip around the area (m)
    res: float = 0.1                # belief-grid resolution (m)

    # --- robot (circular footprint UGV, unicycle kinematics) ------------
    robot_radius: float = 0.30
    v_max: float = 0.40
    v_min: float = -0.20            # limited reversing
    w_max: float = 1.0
    dt: float = 0.1

    # --- boom-mounted near-surface sensor array ------------------------
    sensor_offset: float = 1.2      # robot centre -> sensor bar (m)
    sensor_width: float = 1.8
    sensor_samples: int = 13
    sensor_height: float = 0.10
    sensor_noise: float = 0.04
    sample_radius: float = 0.15     # grid footprint of one sample (m); >= sample spacing

    # --- belief (Bayesian log-odds grid) --------------------------------
    prior: float = 0.02             # prior P(anomaly) per cell inside the area
    mu_obj: float = 0.30            # H1: signal ~ N(mu_obj, sigma_obj)
    sigma_obj: float = 0.15
    sigma_bg: float = 0.05          # H0: signal ~ N(0, sigma_bg)
    update_weight: float = 0.3      # de-correlation weight per sample
    l_min: float = -9.0
    l_max: float = 6.0
    hazard_radius_max: float = 0.5  # ASSUMED bound on a device's trigger footprint (m)

    # --- safety ----------------------------------------------------------
    eps: float = 0.005              # max tolerated P(hazard) under the robot
    cbf_alpha: float = 1.0
    blur_sigma_cells: float = 1.0
    blocked_p: float = 0.05         # above this, a cell is "evidence of hazard"

    # --- survey planning -----------------------------------------------
    lane_spacing: float = 1.4
    lookahead: float = 0.6
    survey_time_limit: float = 2400.0

    # --- baseline "reactive" policy --------------------------------------
    react_threshold: float = 0.15   # stop when any sample exceeds this
    react_radius: float = 0.8       # fixed keep-out disc around a mark (m)

    # --- neutralisation chain (all simulated, inert) ------------------
    boom_reach: float = 1.8         # max horizontal reach of the placer (m)
    unit_radius: float = 0.05       # footprint of the (inert) neutralisation unit
    gap_max: float = 0.10           # unit must end within this gap of the item
    k_sigma: float = 2.0            # uncertainty margin used for placement
    camera_sigma: float = 0.02      # position error after uncover + RGB-D identify
    body_rel_sigma: float = 0.10    # relative error of the estimated body radius
    p_identify: float = 0.95        # P(correct device / clutter call)
    withdraw_distance: float = 5.0  # sim stand-in for the operator's safety distance
    p_neutralise: float = 0.85      # sim outcome of the operator-initiated action
    verify_reset_radius: float = 0.5
