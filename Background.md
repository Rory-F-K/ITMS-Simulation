# ITMS Engineering Model & Simulation — Section 3.2 Write-Up

**Team 324 — MIET2385 Assignment 4 — Detail Design and System Verification**
**Section: 3.2 Engineering Models and Mathematical Analysis**
**Primary Design Scenario: S3 — Emergency Vehicle Priority**

This document explains what the simulation does, the engineering principles and
equations behind each model, the assumptions made, and how to read the results.
It's written so it can be lifted almost directly into the Section 3.2 write-up —
adapt the wording to your own voice and add the actual literature citations
(see the *References to verify* note at the end; the values used here are
realistic, typical figures, not pulled from a specific cited source, so check
them against a real reference before submitting).

---

## 1. What this models and why

Section 3.2 needs mathematical/engineering models that support the detailed
design, with explained assumptions, equations, derivation and verified
results. The Primary Design Scenario selected for Part B is **S3 — Emergency
Vehicle Priority**, which exercises the full closed control loop plus both
override paths described in Section 2.6:

```
Sensing → Central Optimisation → Signal Actuation      (baseline loop, always running)
        ↳ Emergency Vehicle Priority override           (can temporarily pre-empt the loop)
        ↳ Fault Detection / Fallback override            (activates on sensor/comms failure)
```

The simulation builds a numerical model of each part of that loop and checks
the result against the System Requirements table (SR1–SR8 / SR-F01–F07,
SR-N01–N05) from Section 2.2. Two deliverables come out of it:

1. **`itms_simulation.py`** — runs in plain Python (no notebook needed).
   It does two things when you run it:
   - **Live narrated run** (~90 simulated seconds): prints what the system
     is doing, second by second, to the console — sensing updates, phase
     changes, the emergency vehicle request and how conflict C1 is resolved,
     and a fault → fallback → notification sequence. This is the "watch it
     run" view.
   - **Statistical run** (30 simulated minutes, run twice — fixed-time
     baseline vs adaptive control): produces the numbers and charts used for
     verification. Outputs `itms_verification_results.csv` and four PNG
     figures.
2. **Interactive browser demo** — a visual, clickable version of the same
   logic (intersection view, live SR checklist, event log). Good for a
   presentation or for building intuition; the Python script is what you'd
   cite as the actual engineering model.

### How to run it

```bash
pip install numpy matplotlib   # if not already installed
python3 itms_simulation.py
```

No arguments needed. It prints the live run to the console first, then the
statistical run, then saves the CSV and four figures to the working
directory.

---

## 2. Assumptions (state these explicitly in your report)

| Parameter | Value used | Basis |
|---|---|---|
| Saturation (discharge) flow, `s` | 0.50 veh/s/lane (~1800 veh/h) | Typical urban saturation flow range 1700–1900 veh/h/lane (HCM-style figure) |
| Lost time per phase | 4.0 s | Typical start-up + clearance lost time, 3–5 s range |
| Minimum green | 10 s (design) / scaled to 4 s in the browser demo for watchability | Safety/pedestrian floor |
| Maximum green | 60 s (design) / 14 s scaled in demo | Fairness to other approaches |
| Pedestrian clearance time | 7 s | Illustrative: crossing distance 10 m ÷ walking speed 1.2 m/s + ~1 s reaction. **Replace with the actual AS 1742.2 calculation for your assumed crossing width if you cite this.** |
| Peak-period demand | Sinusoidal, average ≈ 0.11 veh/s/approach (per-approach y = q/s ≈ 0.22, sum Y ≈ 0.87 across 4 phases) | Kept deliberately sub-saturation (Y < 1) — see note below |
| MTBF (controller) | 4000 h | Representative figure for a field ITS controller/roadside unit — **verify against a manufacturer datasheet or reliability study before citing** |
| MTTR | 4 h | Representative figure — same caveat |
| Simulation step | 1 s | Fine enough to resolve second-level SR targets without excessive runtime |

**Why Y < 1 matters:** the demand is deliberately kept below saturation
(sum of flow ratios ≈ 0.87, not >1). An oversaturated intersection (Y ≥ 1)
has queues that grow without bound regardless of control strategy — that
would make *any* comparison between fixed-time and adaptive control
meaningless, since neither could keep up with demand. Keeping Y < 1 is what
makes the 65–75% queue reduction result (Section 5) a genuine property of
the *control strategy*, not an artefact of an impossible traffic condition.

---

## 3. The seven models

### 3.1 Traffic flow / queueing model

Each approach has a queue `Q`. Arrivals are drawn from a Poisson process
with time-varying rate `λ(t)` (demand rises and falls over the 30-minute
window, representing a peak load). While an approach has a green signal, it
discharges at the saturation flow rate `s`, limited by how many vehicles are
actually waiting:

```
Q(t+dt) = Q(t) + arrivals(t)·dt − min(Q(t), s·dt)   [only during green]
```

This is a standard deterministic/stochastic queueing approximation used in
traffic engineering for signal-timing analysis — it is not a full
microsimulation (no individual vehicle positions, car-following, or
lane-changing), which is an appropriate level of fidelity for a
system-requirements verification model.

### 3.2 Fixed-time baseline — Webster's optimal cycle

The comparison baseline is a conventional fixed-time plan, computed once
offline using **Webster's formula** (Webster, 1958), which minimises average
intersection delay for a given set of flow ratios:

```
C_o = (1.5·L + 5) / (1 − Y)

where:
  L = total lost time per cycle = n_phases × lost_time_per_phase
  Y = Σ y_i, the sum of the critical flow ratios y_i = q_i / s_i
  g_i = (C_o − L) · y_i / Y        (proportional green split per phase)
```

This plan is calculated once from the *long-run average* demand and then
never changes — exactly how a real time-of-day fixed-time signal operates.

### 3.3 Adaptive control — fully vehicle-actuated logic

The adaptive controller does **not** just recompute Webster's formula more
often (an earlier version of this model tried that and it was numerically
unstable — Webster's formula is very sensitive near saturation and produces
wildly swinging cycle lengths if re-solved on a few seconds of noisy count
data). Instead it uses standard **vehicle-actuated signal control** logic
(see Roess, Prassas & McShane, *Traffic Engineering*):

- Each phase serves its approach for at least `MIN_GREEN`.
- It continues to hold green only while that approach still has a queue
  ("**extend**").
- It ends the phase as soon as the queue clears ("**gap-out**"), or at
  `MAX_GREEN` if demand doesn't let up ("**max-out**").
- A phase with an empty queue is **skipped** entirely.

This is what SR-F01–SR-F03 are actually specifying the mechanism for: the
sense → detect → transmit loop is what lets the controller make this
extend/gap-out/skip decision in real time rather than on a fixed clock.

### 3.4 Sensing / control-loop latency model (SR-F01–F03)

Every sensing cycle draws:

- **Data update interval** — uniform(3 s, 5 s), verifying SR-F01 (≤5 s)
- **Detection latency** — |Normal(1.0 s, 0.4 s)|, capped at 2.5 s, verifying SR-F02 (≤2 s)
- **Transmission latency** — |Normal(0.9 s, 0.4 s)|, capped at 2.5 s, verifying SR-F03 (≤2 s)

These are bounded stochastic models representing realistic jitter in a
sensor-to-controller communications link, not measured hardware data — flag
them as *assumed* distributions in your report, not measured results.

### 3.5 Emergency priority + conflict C1 (SR-F04, F05, N01)

Implements the precedence rule established in Part A:
**safety > compliance > emergency > efficiency**.

- If a pedestrian clearance interval is already running on the requested
  approach, the emergency request is **deferred, never denied** — it waits
  for the remaining clearance time, then is granted. This is conflict C1.
- If there is no conflict, the request is granted immediately (after a
  small actuation-latency draw).
- The dashboard status update latency is drawn per event and checked
  against SR-F05 (≤2 s).
- Because the design deferrs rather than truncates, the pedestrian
  clearance interval is *never* shortened → SR-N01 (0 violations) holds by
  construction, and every request eventually reaches "granted" → SR-F04
  (100%) holds by construction. This is worth saying explicitly in your
  report: these two results follow from the *design rule*, and the
  simulation is really verifying that the rule is applied consistently
  under load, not "discovering" a 100% pass rate empirically.

### 3.6 Fault detection & fallback (SR-F06, F07, SR-N02)

A fault (sensor/comms failure) is injected, then:

```
detect_latency   ~ |Normal(4.0 s, 2.0 s)|, capped 9.8 s   -> SR-F06 (<=10s)
fallback_latency ~ |Normal(4.5 s, 2.2 s)|, capped 9.8 s   -> SR-F07 (<=10s), from detection
notify_latency   ~ |Normal(14.0 s, 6.0 s)|, capped 29.0 s -> SR-F06 (<=30s), from detection
```

Once fallback activates, the affected intersection reverts to the safe
fixed-time (Webster) plan until the fault clears — this is the SR-N02
"single failure does not remove traffic control" requirement.

### 3.7 Reliability / availability (SR-N04)

Standard reliability-engineering steady-state availability:

```
A = MTBF / (MTBF + MTTR)
```

With MTBF = 4000 h and MTTR = 4 h, `A ≈ 99.90%`, which satisfies the
SR-N04 target of ≥99%. **This is an analytical calculation, not something
derived from the discrete-event simulation** — say so explicitly. It's a
standard back-of-envelope reliability estimate; a rigorous version would
use failure data from comparable deployed ITS controllers.

---

## 4. Results

Full numbers are in `itms_verification_results.csv`; figures are
`fig1`–`fig4`. Headline results from the most recent run:

| Requirement | Target | Result | Status |
|---|---|---|---|
| SR-F01 data update ≤5 s | 5.0 s | worst-case 4.99 s | PASS |
| SR-F02 detection ≤2 s | 2.0 s | worst-case 2.11 s | **marginal FAIL** |
| SR-F03 transmit ≤2 s | 2.0 s | worst-case 1.91 s | PASS |
| SR-F04 emergency requests granted | 100% | 100% | PASS |
| SR-F05 priority status ≤2 s | 2.0 s | worst-case 1.70 s | PASS |
| SR-N01 pedestrian violations | 0 | 0 | PASS |
| SR-F06 fault detected ≤10 s | 10 s | worst-case 4.6 s | PASS |
| SR-F07 fallback activated ≤10 s | 10 s | worst-case 7.2 s | PASS |
| SR-F06 notification ≤30 s | 30 s | worst-case 16.3 s | PASS |
| SR-N04 availability ≥99% | 99% | 99.90% | PASS |
| Queue reduction, adaptive vs fixed-time | — | **65–75%** (varies slightly run to run) | Headline performance result |

**Figure guide:**
- `fig1_queue_comparison.png` — queue length over the 30-minute run,
  fixed-time vs adaptive. This is your main "the design works" figure for
  Section 4.1/4.2.
- `fig2_latency_distributions.png` — histograms of the SR-F01/F02/F03
  latencies against their targets. Use this to show the SR-F02 near-miss
  visually.
- `fig3_emergency_priority.png` — grant delay per intersection leg, with
  legs deferred by conflict C1 highlighted. Good evidence for how conflict
  C1 is resolved.
- `fig4_fault_fallback.png` — detect/fallback/notify timeline for each
  injected fault against the 10 s / 10 s / 30 s targets.

### The SR-F02 near-miss — keep this, don't hide it

Across repeated runs, the worst-case detection latency occasionally lands
just over the 2 s target (typically 2.0–2.3 s), while the mean sits
comfortably under 1 s. This is a genuinely useful finding for **Section 4.4
(Validation, Evidence and Performance Gaps)** and **Section 5.1 (Critical
Evaluation)**: it shows the detection sub-function has a thin margin against
its target under the assumed latency distribution, not that the design is
broken. Two honest ways to frame it:

- The 2 s target may need a small implementation margin (e.g. specify
  hardware/software with a 99th-percentile detection latency below 1.5 s,
  not just "under 2 s on average").
- Alternatively, this could motivate tightening SR-F02's verification
  method in acceptance testing (percentile-based rather than pass/fail on a
  single trial).

A report that shows 100% pass on every single requirement with no
discussion of margin tends to read as less credible than one that shows
where the design is tight — use this one.

---

## 5. What's simplified / out of scope

Be upfront about these in your report - a marker will recognise a model
that's honest about its limits far more favourably than one presented as
more complete than it is:

- **Not a microsimulation.** No individual vehicle positions, car-following,
  lane-changing, or turning movements - queues are treated as aggregate
  counts per approach.
- **One intersection modelled in depth.** The Python model simulates one
  representative intersection along the S3 route. A
  fuller model would replicate this across the 220-intersection initial
  deployment and account for coordination/offsets between adjacent
  intersections (not currently modelled).
- **Latency distributions are assumed, not measured.** They are realistic
  and bounded to be consistent with the SR targets, but they are not derived
  from hardware/network test data.

---

## 6. References to verify before citing

- Webster's optimal cycle length formula — [Webster, B.V. (1958), *Traffic
  Signal Settings*, Road Research Technical Paper No. 39, HMSO, London.](https://scispace.com/papers/traffic-signal-settings-3k4m8rw47b)
- Saturation flow rate (~1900 veh/h/lane) — [Determination of Saturation Flows in
Melbourne (2019)](https://australasiantransportresearchforum.org.au/wp-content/uploads/2022/03/ATRF2019_resubmission_29.pdf)
- Pedestrian clearance time / walking speed — [Austroads Guide to Road Design Part 4A, 2023, page 28](https://www.scribd.com/document/681722685/AGRD04A-23-Guide-to-Road-Design-Part-4A-Unsignalised-and-Signalised-Intersections-Ed3-2)
- Vehicle-actuated control logic (extend/gap-out/max-out) — a traffic
  signal control textbook, e.g. Roess, R.P., Prassas, E.S. and McShane,
  W.R., *Traffic Engineering*.
- MTBF/MTTR figures for ITS roadside controllers — a manufacturer
  datasheet or a reliability study of deployed traffic-signal controllers,
  if you can find one; otherwise state clearly that these are illustrative
  assumptions.

---

## 7. Interactive demo

A visual, clickable version of this closed loop (one intersection, live SR
checklist, "send emergency vehicle" and "inject fault" buttons) is
published separately as a Claude artifact — useful for a live walkthrough
in your presentation, but the Python script above is the citable engineering
model for the written report.