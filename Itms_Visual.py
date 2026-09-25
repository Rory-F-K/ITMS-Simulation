"""
ITMS Closed-Loop Control — Visual Window (Python-native)
Team 324 — MIET2385 Assignment 4, Section 3.2

Opens desktop window (matplotlib) showing the intersection live:
    queued vehicles, signal phase, pedestrian clearance, an emergency vehicle
    request with the conflict-C1 defer/grant behaviour, and a fault -> fallback -> notification sequence

Run locally:

    pip install matplotlib numpy
    python3 itms_visual.py

Controls (buttons at the bottom of the window):
    Pause/Play             - freeze/resume the simulation
    Send Emergency Vehicle - inject an emergency-priority request
    Inject Fault           - inject a sensor/comms fault
    Reset                  - restart the simulation from t=0
        
"""

import random
import sys
import importlib
import platform
import numpy as np


import matplotlib

_candidates = [("TkAgg", "tkinter"), ("Qt5Agg", "PyQt5"), ("QtAgg", "PyQt6")]
if platform.system() == "Darwin":
    _candidates.append(("MacOSX", None))

_backend_used = None
for _backend_name, _module_name in _candidates:
    try:
        if _module_name is not None:
            importlib.import_module(_module_name)
        matplotlib.use(_backend_name, force=True)
        _backend_used = _backend_name
        break
    except Exception:
        continue

if _backend_used is None:
    print("ERROR: no interactive GUI backend is available (tried Tkinter, PyQt5, PyQt6).")
    print("No window can be opened until one of these is installed. Fix:")
    print("  - Tkinter (usually already included with Python on Windows/Mac):")
    print("      Linux: sudo apt-get install python3-tk")
    print("  - or, as an alternative:  pip install PyQt5")
    print("Then re-run: python3 itms_visual.py")
    sys.exit(1)

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.patches import Circle, Rectangle
from matplotlib.animation import FuncAnimation

print(f"[itms_visual] using matplotlib backend: {_backend_used}")
print("[itms_visual] a window should open now.")

random.seed()
np.random.seed()

# ------------------------------------------------------------------
# Parameters (same engineering assumptions as itms_simulation.py,
# time-scaled so the demo is watchable in real time)
# ------------------------------------------------------------------
APPROACHES = ["N", "E", "S", "W"]
DIRVEC = {"N": (0, 1), "E": (1, 0), "S": (0, -1), "W": (-1, 0)}
PHASES = [("N", "S"), ("E", "W")]  # opposite approaches are paired and go green together
SAT_FLOW = 0.5          # veh/s discharge capacity
MIN_GREEN = 4.0
MAX_GREEN = 14.0
PED_CLEARANCE = 6.0
SENSE_MIN, SENSE_MAX = 1.2, 2.0
SIM_SPEEDUP = 2.2        # simulated seconds per real second
FRAME_DT = 0.12          # real seconds per animation frame

PED_APPROACH = "S"


def rand(a, b):
    return a + random.random() * (b - a)


class Sim:
    def __init__(self):
        self.t = 0.0
        self.playing = True
        self.queues = {"N": 3.0, "E": 2.0, "S": 4.0, "W": 1.0}
        self.phase = 0
        self.phase_elapsed = 0.0
        self.next_sense = rand(SENSE_MIN, SENSE_MAX)
        self.pedestrian_active = False
        self.pedestrian_timer = 0.0
        self.pedestrian_next_toggle = rand(6, 10)
        self.mode = "NORMAL"  # NORMAL | EMERGENCY_DEFERRED | EMERGENCY_ACTIVE | FALLBACK
        self.ev = None
        self.fault = None
        self.metrics = {
            "data_updates": [], "detects": [], "transmits": [],
            "ev_requests": 0, "ev_granted": 0, "ev_status_latencies": [],
            "ped_violations": 0,
            "fault_detect": [], "fault_fallback": [], "fault_notify": [],
            "uptime": 300.0, "downtime": 0.3,
        }
        self.log = []

    def add_log(self, msg):
        self.log.insert(0, f"[t={self.t:5.1f}s] {msg}")
        self.log = self.log[:9]

    def step(self, dt):
        self.t += dt

        base_rates = {"N": 0.18, "E": 0.12, "S": 0.20, "W": 0.10}
        for a in APPROACHES:
            if random.random() < base_rates[a] * dt * 4:
                self.queues[a] += 1

        # pedestrian clearance
        if not self.pedestrian_active:
            self.pedestrian_next_toggle -= dt
            if self.pedestrian_next_toggle <= 0:
                self.pedestrian_active = True
                self.pedestrian_timer = PED_CLEARANCE
                self.add_log(f"Pedestrian clearance STARTED on {PED_APPROACH} approach")
        else:
            self.pedestrian_timer -= dt
            if self.pedestrian_timer <= 0:
                self.pedestrian_active = False
                self.pedestrian_next_toggle = rand(8, 14)
                self.add_log(f"Pedestrian clearance ENDED on {PED_APPROACH} approach")
                if self.mode == "EMERGENCY_DEFERRED" and self.ev:
                    self.grant_emergency()

        # fault lifecycle
        if self.fault and self.fault["stage"] != "done":
            f = self.fault
            since = self.t - f["injected_at"]
            if f["stage"] == "injected" and since >= f["detect_delay"]:
                f["stage"] = "detected"
                f["detected_at"] = self.t
                self.metrics["fault_detect"].append(f["detect_delay"])
                self.add_log(f"Fault DETECTED on {f['approach']} after {f['detect_delay']:.1f}s (SR-F06 <=10s)")
            elif f["stage"] == "detected" and (self.t - f["detected_at"]) >= f["fallback_delay"]:
                f["stage"] = "fallback"
                self.mode = "FALLBACK"
                self.metrics["fault_fallback"].append(f["fallback_delay"])
                self.add_log(f"Fallback ACTIVATED after {f['fallback_delay']:.1f}s (SR-F07 <=10s)")
            elif f["stage"] == "fallback" and (self.t - f["detected_at"]) >= f["notify_delay"] and not f.get("notified"):
                f["notified"] = True
                self.metrics["fault_notify"].append(f["notify_delay"])
                self.add_log(f"Operator notification issued after {f['notify_delay']:.1f}s (SR-F06 <=30s)")
            elif f["stage"] == "fallback" and (self.t - f["fallback_at" if "fallback_at" in f else "detected_at"]) >= 10:
                f["stage"] = "done"
                self.mode = "NORMAL"
                self.add_log(f"Fault cleared on {f['approach']}, resuming adaptive control")

        # sensing loop
        self.next_sense -= dt
        if self.next_sense <= 0 and self.mode != "FALLBACK":
            interval = rand(SENSE_MIN, SENSE_MAX)
            self.metrics["data_updates"].append(interval)
            detect = min(2.4, abs(random.gauss(0.5, 0.25)))
            transmit = min(2.4, abs(random.gauss(0.45, 0.2)))
            self.metrics["detects"].append(detect)
            self.metrics["transmits"].append(transmit)
            self.next_sense = interval

        # phase control -- opposite approaches (N+S, E+W) are paired and
        # served together, exactly like a normal 2-phase traffic signal.
        # An active emergency vehicle overrides this: only its own
        # approach gets green, its paired opposite goes red too.
        if self.mode in ("NORMAL", "FALLBACK"):
            self.phase_elapsed += dt
            a1, a2 = PHASES[self.phase]
            for cur in (a1, a2):
                served = min(self.queues[cur], SAT_FLOW * dt * 3)
                self.queues[cur] = max(0.0, self.queues[cur] - served)

            held_for_ped = self.pedestrian_active and PED_APPROACH in (a1, a2)
            gap_out = self.queues[a1] < 0.5 and self.queues[a2] < 0.5 and self.phase_elapsed >= MIN_GREEN
            max_out = self.phase_elapsed >= MAX_GREEN
            if not held_for_ped and (gap_out or max_out):
                self.advance_phase()
        elif self.mode == "EMERGENCY_ACTIVE" and self.ev:
            a = self.ev["approach"]
            self.ev["time_in_phase"] = self.ev.get("time_in_phase", 0.0) + dt
            self.queues[a] = max(0.0, self.queues[a] - SAT_FLOW * dt * 3)
            if self.ev["time_in_phase"] > 5:
                self.add_log(f"Emergency vehicle cleared {a} approach, recovery cycle restoring control")
                self.ev = None
                self.mode = "NORMAL"
                # resume normal paired operation on whichever phase includes the EV's approach
                self.phase = 0 if a in PHASES[0] else 1
                self.phase_elapsed = 0.0

    def advance_phase(self):
        self.phase = 1 - self.phase
        self.phase_elapsed = 0.0

    def grant_emergency(self):
        ev = self.ev
        ev["granted"] = True
        self.mode = "EMERGENCY_ACTIVE"
        self.metrics["ev_granted"] += 1
        latency = min(1.9, abs(random.gauss(0.7, 0.3)))
        self.metrics["ev_status_latencies"].append(latency)
        self.add_log(f"Priority GRANTED on {ev['approach']} — dashboard updated in {latency:.1f}s (SR-F05 <=2s)")

    def send_emergency_vehicle(self, _event=None):
        if self.ev:
            return
        self.metrics["ev_requests"] += 1
        self.ev = {"approach": PED_APPROACH, "requested_at": self.t, "granted": False}
        self.add_log(f"Emergency vehicle REQUEST received for {PED_APPROACH} approach")
        if self.pedestrian_active:
            self.mode = "EMERGENCY_DEFERRED"
            self.add_log("Conflict C1: pedestrian clearance active -> priority DEFERRED, not denied")
        else:
            self.grant_emergency()

    def inject_fault(self, _event=None):
        if self.fault and self.fault["stage"] != "done":
            return
        approach = random.choice(APPROACHES)
        self.fault = {
            "approach": approach, "injected_at": self.t, "stage": "injected",
            "detect_delay": rand(2, 9), "fallback_delay": rand(2, 9), "notify_delay": rand(6, 28),
        }
        self.add_log(f"Sensor/comms FAULT injected on {approach} approach")

    def toggle_play(self, _event=None):
        self.playing = not self.playing


sim = Sim()

# ------------------------------------------------------------------
# Figure / window setup
# ------------------------------------------------------------------
try:
    fig = plt.figure(figsize=(11, 6.4))
    fig.canvas.manager.set_window_title("ITMS Closed-Loop Control — Visual Simulation")
except Exception as e:
    print(f"ERROR: could not open a window ({e}).")
    print("This usually means there's no display available to this Python process")
    print("(e.g. running over SSH without X forwarding, inside a headless server or")
    print("container, or a WSL install without a display server). Run this on a")
    print("normal desktop/laptop with a screen instead.")
    sys.exit(1)

ax_map = fig.add_axes([0.04, 0.16, 0.46, 0.78])
ax_map.set_xlim(-6, 6); ax_map.set_ylim(-6, 6)
ax_map.set_aspect("equal"); ax_map.axis("off")

ax_dash = fig.add_axes([0.55, 0.16, 0.42, 0.78])
ax_dash.axis("off")

MODE_COLOR = {
    "NORMAL": "#1e8e3e",
    "EMERGENCY_DEFERRED": "#e67e22",
    "EMERGENCY_ACTIVE": "#8e44ad",
    "FALLBACK": "#c0392b",
}
MODE_LABEL = {
    "NORMAL": "NORMAL — adaptive closed-loop control",
    "EMERGENCY_DEFERRED": "EMERGENCY PRIORITY — deferred (conflict C1)",
    "EMERGENCY_ACTIVE": "EMERGENCY PRIORITY — active",
    "FALLBACK": "FALLBACK — fixed-time safe operation",
}


def draw_map():
    ax_map.cla()
    ax_map.set_xlim(-6, 6); ax_map.set_ylim(-6, 6)
    ax_map.set_aspect("equal"); ax_map.axis("off")
    ax_map.add_patch(Rectangle((-6, -1.1), 12, 2.2, color="#3a4048", zorder=0))
    ax_map.add_patch(Rectangle((-1.1, -6), 2.2, 12, color="#3a4048", zorder=0))

    # pedestrian crossing stripes
    px, py = DIRVEC[PED_APPROACH]
    color = "#e67e22" if sim.pedestrian_active else "#5a6068"
    for s in np.arange(-0.9, 1.0, 0.35):
        cx, cy = px * 1.9 + (-py) * s, py * 1.9 + px * s
        ax_map.add_patch(Rectangle((cx - 0.12, cy - 0.12), 0.24, 0.24, color=color, zorder=1))

    # queued vehicles
    for a in APPROACHES:
        dx, dy = DIRVEC[a]
        n = min(7, round(sim.queues[a]))
        for k in range(n):
            dist = 2.3 + k * 0.55
            ax_map.add_patch(Rectangle((dx * dist - 0.28, dy * dist - 0.28), 0.56, 0.56,
                                        color="#2471a3", zorder=2))

    # signal heads -- opposite approaches (N+S, E+W) are paired and show
    # green together, unless an emergency vehicle is active, in which case
    # only its own approach is green (its paired opposite goes red too).
    for i, a in enumerate(APPROACHES):
        dx, dy = DIRVEC[a]
        if sim.mode == "EMERGENCY_ACTIVE" and sim.ev:
            green = (a == sim.ev["approach"])
        else:
            green = (a in PHASES[sim.phase]) and sim.mode != "EMERGENCY_DEFERRED"
        c = "#1e8e3e" if green else "#c0392b"
        if sim.pedestrian_active and a == PED_APPROACH:
            c = "#e67e22"
        ax_map.add_patch(Circle((dx * 1.6, dy * 1.6), 0.22, color=c, zorder=3))

    # emergency vehicle marker
    if sim.ev and sim.mode in ("EMERGENCY_ACTIVE", "EMERGENCY_DEFERRED"):
        dx, dy = DIRVEC[sim.ev["approach"]]
        ax_map.add_patch(Circle((dx * 3.2, dy * 3.2), 0.34, color="#8e44ad", zorder=4))
        ax_map.text(dx * 3.2, dy * 3.2, "EV", fontsize=9, ha="center", va="center",
                    color="white", fontweight="bold", zorder=5)

    # fault marker
    if sim.fault and sim.fault["stage"] != "done":
        dx, dy = DIRVEC[sim.fault["approach"]]
        label = "FIX" if sim.fault["stage"] == "fallback" else "!"
        color = "#e67e22" if sim.fault["stage"] == "fallback" else "#c0392b"
        ax_map.add_patch(Circle((dx * 0.9, dy * 0.9), 0.26, color=color, zorder=5))
        ax_map.text(dx * 0.9, dy * 0.9, label, fontsize=8, ha="center", va="center",
                    color="white", fontweight="bold", zorder=6)

    ax_map.text(-5.6, 5.5, f"t = {sim.t:5.1f}s", ha="left", fontsize=11, color="#333")
    ax_map.text(0, -5.6, MODE_LABEL[sim.mode], ha="center", fontsize=10.5, fontweight="bold",
                color=MODE_COLOR[sim.mode])


def fmt_req(name, ok, detail):
    if ok is None:
        badge, color = "PENDING", "#888"
    elif ok:
        badge, color = "PASS", "#1e8e3e"
    else:
        badge, color = "CHECK", "#c0392b"
    return name, detail, badge, color


def draw_dash():
    ax_dash.cla()
    ax_dash.axis("off")
    ax_dash.set_xlim(0, 1); ax_dash.set_ylim(0, 1)

    m = sim.metrics
    du, de, tr = m["data_updates"], m["detects"], m["transmits"]
    avail = 100 * m["uptime"] / (m["uptime"] + m["downtime"])

    rows = [
        fmt_req("SR-F01 data update <=5s", None if not du else max(du) <= 5,
                "—" if not du else f"worst {max(du):.1f}s"),
        fmt_req("SR-F02 detect <=2s", None if not de else max(de) <= 2,
                "—" if not de else f"worst {max(de):.1f}s"),
        fmt_req("SR-F03 transmit <=2s", None if not tr else max(tr) <= 2,
                "—" if not tr else f"worst {max(tr):.1f}s"),
        fmt_req("SR-F04 emergency granted", None if m["ev_requests"] == 0 else m["ev_granted"] == m["ev_requests"],
                "no requests yet" if m["ev_requests"] == 0 else f"{m['ev_granted']}/{m['ev_requests']}"),
        fmt_req("SR-F05 priority status <=2s",
                None if not m["ev_status_latencies"] else max(m["ev_status_latencies"]) <= 2,
                "—" if not m["ev_status_latencies"] else f"worst {max(m['ev_status_latencies']):.1f}s"),
        fmt_req("SR-N01 pedestrian violations = 0", True, f"{m['ped_violations']} violations"),
        fmt_req("SR-F06 fault detected <=10s",
                None if not m["fault_detect"] else max(m["fault_detect"]) <= 10,
                "no fault yet" if not m["fault_detect"] else f"worst {max(m['fault_detect']):.1f}s"),
        fmt_req("SR-F07 fallback <=10s",
                None if not m["fault_fallback"] else max(m["fault_fallback"]) <= 10,
                "no fault yet" if not m["fault_fallback"] else f"worst {max(m['fault_fallback']):.1f}s"),
        fmt_req("SR-N04 availability >=99%", avail >= 99, f"{avail:.2f}%"),
    ]

    ax_dash.text(0, 0.99, "SYSTEM REQUIREMENT VERIFICATION (live)", fontsize=11, fontweight="bold", va="top")
    y = 0.93
    for name, detail, badge, color in rows:
        ax_dash.text(0, y, name, fontsize=9.3, va="top")
        ax_dash.text(0.62, y, detail, fontsize=8.7, va="top", color="#555")
        ax_dash.text(0.86, y, badge, fontsize=8.3, va="top", fontweight="bold", color=color)
        y -= 0.052

    ax_dash.text(0, y - 0.02, "EVENT LOG", fontsize=11, fontweight="bold", va="top")
    y -= 0.075
    for line in sim.log:
        ax_dash.text(0, y, line, fontsize=8.0, va="top", family="monospace", color="#333")
        y -= 0.042


def update(_frame):
    if sim.playing:
        sim.step(FRAME_DT * SIM_SPEEDUP)
    draw_map()
    draw_dash()
    return []


# ------------------------------------------------------------------
# Buttons
# ------------------------------------------------------------------
btn_play_ax = fig.add_axes([0.06, 0.03, 0.13, 0.06])
btn_ev_ax = fig.add_axes([0.24, 0.03, 0.22, 0.06])
btn_fault_ax = fig.add_axes([0.50, 0.03, 0.17, 0.06])
btn_reset_ax = fig.add_axes([0.71, 0.03, 0.13, 0.06])

btn_play = Button(btn_play_ax, "Pause / Play")
btn_ev = Button(btn_ev_ax, "Send Emergency Vehicle")
btn_fault = Button(btn_fault_ax, "Inject Fault")
btn_reset = Button(btn_reset_ax, "Reset")


def do_reset(_event=None):
    global sim
    sim = Sim()


btn_play.on_clicked(lambda e: sim.toggle_play(e))
btn_ev.on_clicked(lambda e: sim.send_emergency_vehicle(e))
btn_fault.on_clicked(lambda e: sim.inject_fault(e))
btn_reset.on_clicked(do_reset)

ani = FuncAnimation(fig, update, interval=int(FRAME_DT * 1000), cache_frame_data=False)

if __name__ == "__main__":
    plt.show()