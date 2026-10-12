# Design (Stage A)

Everything here is a laboratory abstraction with inert surrogates. Numbers describe sensor contrast and geometry of surrogates, not any real device.

## 1. World
A suspected hazardous area `[0, W] × [0, H]` (12 m × 12 m by default) surrounded by a 2 m **cleared access strip** (known clear — the robot starts there, as real teams work from a cleared base line). Items (`lastmetre/field.py`): `ied_ap` (small, low metal content), `ied_at` (larger), `clutter` (harmless metal). Each hazardous item has a **trigger footprint** of radius `hazard_r` around its body that the robot does not know; it only knows an upper bound `R_h = hazard_radius_max` (0.5 m).

## 2. Robot and sensor
Unicycle UGV with a circular footprint (radius `r = 0.3 m`). A boom carries a near-surface sensor bar (1.8 m wide, 13 samples) at `l_s = 1.2 m` ahead of the robot centre — far enough that the sensor always looks at ground **before** the robot's footprint and the possible trigger footprint of anything there can reach it (`l_s > r + R_h`).

Signal model (normalised dipole-like fall-off, plus white noise σ = 0.04):

    s(p) = Σ_i a_i · ( z_i² / (‖p − x_i‖² + z_i²) )^{3/2} + n,     z_i = h + depth_i

## 3. Belief
Log-odds grid (0.1 m) of `P_obj(c)` = P(an anomaly is centred in cell c). Each sample updates the cells within 0.15 m by

    l(c) ← l(c) + w · log [ N(s; μ₁, σ₁) / N(s; 0, σ₀) ],   w = 0.3 (de-correlation), clamped to [−9, 6]

Prior inside the area: 0.02. Perimeter: clamped clear.

Robot-centre risk: `F(c) = max { P_obj(c') : ‖c' − c‖ ≤ R_h + r }` (a disc dilation: a device anywhere within `R_h + r` could have its footprint under the robot). Unsensed cells keep the prior 0.02 > ε, so **unsensed ground is forbidden by construction**.

## 4. Probabilistic barrier and safety filter
`h(p) = log ε − (G ∗ log F)(p)` with ε = 0.005 and `G` a small Gaussian blur (for a usable gradient). `h ≥ 0` ⇔ (smoothed) P(hazard under the footprint) ≤ ε.

Unicycle `ẋ = v cos θ, ẏ = v sin θ, θ̇ = ω`. Since `h` depends on position only, `ḣ = (∇h · [cos θ, sin θ]) v`, and the CBF condition `ḣ ≥ −α h` is a bound on `v` alone, solved in closed form each step (`lastmetre/sim.py::filter`). Rotation is unconstrained (circular footprint). In discrete time the minimum observed `h` is slightly negative (−0.05 in log units), i.e. the risk bound is exceeded by at most ~5 % relative — the formal discrete-time treatment is thesis work (C1).

**Baseline `reactive`:** identical CBF, but `h = distance to the nearest mark − 0.8 m`, where a mark is any sample whose raw signal exceeds 0.15. It has no notion of unsensed ground or calibrated probability. **Baseline `naive`:** no safety layer.

## 5. Survey
Boustrophedon lanes (1.4 m spacing < 1.8 m sensor width), lane ends in the cleared strip. New ground is only ever entered **head-on with the sensor leading**. If a lane is blocked by a suspect, the robot returns over verified-safe ground and works the same lane from the other end. Planning (`dijkstra` on an 8-connected grid) uses the *same* smoothed field as the safety filter (cells with `h ≥ 0.5` are "safe"), with a cost penalty near the edge of the safe set; if the robot stops on its safety boundary it first backs off along `∇h` (deadlock recovery).

## 6. The last-metre chain (`lastmetre/mission.py::process_suspect`)
1. **Approach** — shortest path over verified-safe cells to a pose within placer reach (1.8 m).
2. **Uncover + identify** — abstracted in Stage A: position error after identification σ = 2 cm, body-radius error 10 %, identification correct with p = 0.95. (Stage B/C2: contactless uncovering with a transmitted-load bound.)
3. **Place** — unit placed on the line item→robot at `r̂_body + k σ + r_unit` from the estimated centre (k = 2); contact with the true body and the true gap are scored. Success = no contact and gap ≤ 10 cm.
4. **Withdraw** — over verified-safe ground to ≥ 5 m (simulation stand-in for the operator's danger area).
5. **Operator decides and initiates** — simulated; outcome "neutralised" with p = 0.85. The robot has no initiation capability.
6. **Verify** — evidence around the spot is **invalidated** (reset to the prior: the world changed), the robot returns head-on and re-senses; verdict = residual hazard / verified / inconclusive, scored against the truth.

## 7. Metrics (`lastmetre/metrics.py`)
Coverage of the area, probability of detection (suspect within 0.5 m of a hazardous item), clutter alarms, false alarms per hectare, **virtual triggers** (robot footprint overlapped a true trigger footprint), time, distance, minimum barrier value; chain: placements within spec, contacts, verification correct / wrong / inconclusive, devices missed at identification.

## 8. Known limitations (honest list)
- 2-D kinematics; no terrain, slip or vegetation; perfect localisation.
- Uncovering and identification are abstracted; no granular physics (that is Stage B/C2 on real hardware — no simulator models it well).
- The sensor model is idealised (no soil response, no orientation effects); the hazard-footprint bound `R_h` is assumed known.
- Discrete-time CBF without formal guarantees; ε is per-step, not per-mission.
- Seeds are few (5); results are illustrative, not statistically powered.

## 9. C2 prototype — load-bounded placement (`lastmetre/contact.py`)
1-D vertical, vectorised Monte Carlo, all loads normalised by the unit weight W.
- **Soil:** Hunt–Crossley `F = k δ^1.5 (1 + β δ̇)`, with `k` log-uniform in [350, 2800] W/m^1.5 (W carried at 20 mm to 5 mm sinkage) and β = 1 s/m.
- **Transfer to the buried object:** Boussinesq point load, `σ_z = 3P z³ / (2π (r² + z²)^{5/2})`, so `L = γ(r, z) F` with `γ ≈ σ_z A_obj`.
- **Requirement:** `L(t) ≤ θ` at every instant (θ = 0.05 W, a normalised placeholder).
- **Planner (chance constraint):** the smallest lateral offset `r` such that the 97.5 % (or 99.9 %) quantile of `γ` under depth/position uncertainty, times `W` and a 1.25 headroom, is ≤ θ. The controller then receives `F_lim = θ / γ̄`.
- **Stopping-distance barrier:** the worst-case force after the sensing delay τ and a full stop at `a_max`, for any stiffness ≤ K:
  `F_pred = (F_m^{2/3} + K^{2/3} (vτ + v²/2a))^{3/2} (1 + βv)`.
  Because this is monotone in K, using an upper bound K is safe. The robot uses the highest speed with `F_pred ≤ F_lim`.
  - *Robust* version: K = k_max.
  - *Adaptive* version: K = 1.3 × the running maximum of `F_m / δ̂^{1.5}`, with δ̂ measured from the detected contact point. This is biased upwards, i.e. conservative.
- **Baselines:**
  - position control to the estimated surface − 1 cm, then release (a release in the air drops the unit);
  - guarded move: constant speed until the measured force reaches W, then stop.
- **Limitations:**
  - Boussinesq is elastic-half-space theory; sand is granular, plastic and heterogeneous.
  - The object is treated as a point-like area; there is no lateral load and no uncovering.
  - The real transfer has to be measured in a soil bin with instrumented surrogates (C4).

## 10. Mop-up and second look
After the chain, lanes whose swath still holds ≥ 5 unsensed cells are worked again. These are pockets that were enclosed by suspects before neutralisation. Where something was called clutter, the operator removes it and the evidence is invalidated. If the spot is detected again, it is re-processed as a device, a safety net for identification errors. A device wrongly called clutter is *not* removed in the simulation (the human check is the backstop), so the mop-up can find it again.
