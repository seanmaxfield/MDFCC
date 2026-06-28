import io
import json
import math
import sys
import threading
import tempfile
import os
import tkinter as tk
from tkinter import ttk
from tkinter import simpledialog
from tkinter import messagebox

import numpy as np

import main as m
import rocket_data as rd
import balmis as bm
import short_search as ss
import footprintv2 as fp
import intersection as isec
import ground_based_early_warning as gbewr
from fcc_constants import R_e, beta_step, dist_param, bm_range_limit
import fcc_constants

try:
    from geographiclib.geodesic import Geodesic
    _geod = Geodesic(R_e, 0)
except ImportError:
    _geod = None

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt
    CARTOPY_AVAILABLE = True
except Exception:
    CARTOPY_AVAILABLE = False

try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    MPL_TK_AVAILABLE = True
except Exception:
    MPL_TK_AVAILABLE = False

try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    MATPLOTLIB_TK_AVAILABLE = True
except Exception:
    MATPLOTLIB_TK_AVAILABLE = False


class _TeeStream:
    """Write stdout to multiple streams (terminal + capture buffer)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()


def _build_flight_trajectory_figure(trj, title):
    """Two-panel trajectory chart (path + speed/acceleration), matching fp_gui.gui_trajcharts."""
    fig = Figure(figsize=(8.5, 3.4), dpi=100)
    fig.suptitle(title, fontsize=11)

    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax22 = ax2.twinx()

    time = trj[:, 0]
    x_pt = np.copy(trj[:, 1])
    y_pt = np.copy(trj[:, 2])
    x_ea = np.copy(trj[:, 1])
    y_ea = np.copy(trj[:, 2])
    x_at = np.copy(trj[:, 1])
    y_at = np.copy(trj[:, 2])

    i_max = len(time) - 1
    al_max = trj[i_max, 2]
    am2 = al_max / 2
    d = R_e * math.sin(am2)

    for i_tc in range(len(time)):
        x_tc = d - x_pt[i_tc] * math.sin(am2 - y_pt[i_tc])
        y_tc = x_pt[i_tc] * math.cos(y_pt[i_tc] - am2) - R_e * math.cos(am2)
        x_pt[i_tc] = x_tc / 1000
        y_pt[i_tc] = y_tc / 1000

        x_tx = d - R_e * math.sin(am2 - y_ea[i_tc])
        y_tx = R_e * math.cos(y_ea[i_tc] - am2) - R_e * math.cos(am2)
        x_ea[i_tc] = x_tx / 1000
        y_ea[i_tc] = y_tx / 1000

        x_tz = d / 1000 - (R_e / 1000 + 100) * math.sin(am2 - y_at[i_tc])
        y_tz = (R_e / 1000 + 100) * math.cos(y_at[i_tc] - am2) - R_e / 1000 * math.cos(am2)
        x_at[i_tc] = x_tz
        y_at[i_tc] = y_tz

    v_pt = trj[:, 3] / 1000
    a_pt = trj[:, 4]

    ax1.set_xlabel("x distance, km")
    ax1.set_ylabel("y distance, km")
    ax2.set_xlabel("time, s")
    ax2.set_ylabel("speed, km/s")
    ax22.set_ylabel("acceleration, m/s²")

    ax1.plot(x_ea, y_ea, c="black", linestyle="-", linewidth=1, label="earth")
    ax1.plot(x_pt, y_pt, c="red", linestyle="-", linewidth=1, label="trajectory")
    ax1.plot(x_at, y_at, c="blue", linestyle=":", linewidth=1, label="atmosphere")
    ax2.plot(time, v_pt, c="blue", linestyle="-", linewidth=1, label="speed")
    ax22.plot(time, a_pt, c="green", linestyle="-", linewidth=1, label="acceleration")
    ax22.axhline(y=0, c="black", linewidth=0.75, linestyle="--")

    ax1.set_title("trajectory", fontsize=10)
    ax2.set_title("speed and acceleration", fontsize=10)
    ax1.legend(loc="best", frameon=False)
    ax2.legend(loc=(0.35, 0.925), frameon=False)
    ax22.legend(loc=(0.35, 0.88), frameon=False)
    ax1.axis("equal")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "fcc_config.json")
TILE_ROOT = os.path.join(BASE_DIR, "world_tiles")
TILE_SIZE = 256
ROCKET_DATA_PATH = os.path.join(BASE_DIR, "rocket_data.json")
GBEWR_DATA_PATH = os.path.join(BASE_DIR, "gbewr_data.json")
MIN_ZOOM = 1.0
MAX_ZOOM = 128.0

THEME = {
    "root_bg": "#0e1117",
    "panel_bg": "#181c27",
    "panel_inner": "#12151e",
    "titlebar": "#222736",
    "toolbar_bg": "#181c27",
    "border": "#2b3142",
    "fg": "#e7ecf6",
    "fg_muted": "#98a2b8",
    "accent": "#6d8cff",
    "accent_dim": "#3f56b8",
    "accent_fg": "#ffffff",
    "success": "#54d98c",
    "danger": "#f87171",
    "warning": "#fbbf24",
    "entry_bg": "#0d1019",
    "list_bg": "#0d1019",
    "list_fg": "#e7ecf6",
    "select_bg": "#2b3450",
    "map_stripe": "#12121c",
    # Secondary buttons: light chip + dark label (renders reliably on macOS native buttons)
    "tool_btn_bg": "#dde5f3",
    "tool_btn_fg": "#16203a",
    "tool_btn_hover": "#c6d2e9",
    "tool_btn_active_bg": "#aebdda",
    "tool_btn_active_fg": "#0b1020",
    # Primary action buttons: accent chip + white label (ttk honors the bg)
    "primary_btn_bg": "#6d8cff",
    "primary_btn_fg": "#ffffff",
    "primary_btn_hover": "#577af2",
    "primary_btn_active": "#4a63c0",
    # tk.Button often renders a LIGHT native chip on macOS regardless of bg, so for the
    # custom tk buttons use a dark, high-contrast label (+ accent border for "primary").
    "primary_text": "#16307a",
    "armed_text": "#7c2d12",   # placement-armed (warning) label
    "armed_bg": "#fde68a",     # placement-armed chip
}

HELP = {
    "win_title": "Plan interceptor launch points (ILPs), missile launch points (MLPs), and ground-based early-warning radars on the world map.",
    "library": "Edit missile types, interceptor types, and radar presets used by tokens. Data is stored in rocket_data.json and gbewr_data.json.",
    "inspector": "When you select a unit on the map, its position, footprint options, and configuration appear here. Click Apply to save edits to the live token.",
    "check_coverage": "Walkthrough: pick an ILP and MLP (by name or by clicking the map), then click a target point. Shows whether that point lies inside the computed footprint.",
    "measure_distance": "Click two points (existing units or anywhere on the map) to measure geodesic distance in kilometers.",
    "map_pan": "Drag with the left mouse button on empty map to pan. Right-drag also pans. Drag units or handles to move or aim.",
    "fp_layers": "When a defense site has multiple Calculation Modes or footprint layers, click its star marker once (without dragging) to switch Calculation Mode and show or hide individual layers on the map.",
    "toggle_library": "Show or hide the left library panel. Drag its title bar to move it.",
    "add_ilp": "Interceptor launch point — models where batteries could engage. Click the map to place; drag the unit or the green/gray handles to aim the sector.",
    "add_mlp": "Missile launch point — threat launch locations used with range rings and coverage checks.",
    "add_gbewr": "Place a ground-based early-warning radar; assign its preset from the inspector.",
    "apply": "Write inspector fields into the selected token and refresh the map overlay.",
    "go_footprint": "Run the footprint solver in the background and draw engagement regions on the map.",
    "update_range": "Recompute maximum range rings for the selected MLP’s missile types.",
    "delete_token": "Remove this token from the map (does not delete entries from the type library).",
    "sectoral": "Clip the footprint to the defense sector (center direction ± half width) instead of a full 360° envelope.",
    "dict_tab": "Reference for JSON field names used in the data files and in token configuration.",
    "insp_empty": "Choose a unit on the map, or use the toolbar to place a defense site, threat launch, or radar.",
    "display_name": "Shown next to the marker on the map (does not change internal simulation IDs).",
    "lat_lon": "Coordinates in decimal degrees (WGS84). Drag the marker on the map to adjust.",
    "radar_preset": "Which radar performance curve from the library this placed radar uses.",
    "boresight": "Azimuth of the center of the defended sector, degrees clockwise from north.",
    "sector_w": "Total angular width of the sector (center line ± half width). Drag sector handles on the map.",
    "threat_types": "Threat missile models to evaluate against this defense site (multi-select).",
    "show_center": "Draw the green boresight line from the site.",
    "show_sector": "Draw the gray sector boundary lines.",
    "sectoral_fp": "When computing footprints, clip results to the sector instead of full 360°.",
    "threat_arc": "Show sampled threat launch points (gold dots) on a curved line at the missile's range, spanning the sector — i.e. where the incoming missile could be coming from. Click a dot to isolate that single launch point and recompute the footprint for it.",
    "clear_isolation": "Clear an isolated launch point and recompute the footprint for the full sector again.",
    "from_mlp": "Compute the defended footprint against a threat originating from one fixed launch point you place on the map. Unlike the standard footprint (which assumes max-range trajectories everywhere), here the MLP→target range — and therefore the missile trajectory — changes for each analyzed target point.",
    "from_mlp_loft": "For target points closer than the missile's max range, two trajectories reach them: a lofted (high, slow) and a depressed (flat, fast) one. Choose lofted here; otherwise depressed is used.",
    "from_mlp_set": "Then click the map to place the threat launch point (a yellow dot). Drag the dot afterward to move it; the boresight auto-points at it.",
    "from_mlp_compute": "Compute the footprint(s) for the placed launch point, honoring the same interceptor, threat, detection, and engagement settings as the standard footprint.",
    "from_ranges_compute": "For every selected threat missile at once, place its launch point at the central (boresight) dot on its own max-range arc and compute the from-launch-point footprint. One layer per missile, each oriented along the boresight.",
    "full_sector_compute": "Fan the standard (max-range-everywhere) footprint across a set of boresights spanning the sector (count = Threat dots per sector) and layer them all. Each layer's name carries its boresight angle, missile, and interceptor.",
    "pct100": "After running Full Sector: show only the region defended in 100% of the sampled boresights (the intersection of all boresight footprints) per missile/interceptor.",
    "launch_box": "Drag a dotted box on the map; the system samples a grid of missile launch points inside it and overlays a from-launch-point footprint for each. A small panel above the box lets you pick which selected missile types the box fires.",
    "builder": "Open the Missile/Interceptor Builder: edit physics specs and see the trajectory, speed, and range update live (and drag the trajectory to change the launch angle). Autofill specs from a text description via AIP, then Save & Inject into the rocket library.",
    "det_override": "Use a fixed slant-range detection cue (km) instead of the native pairing value. This limits when the interceptor is allowed to start engaging the inbound threat — not the maximum ground distance of the footprint. Re-run the footprint calculation after changing this. Mutually exclusive with space-based override.",
    "det_sweep": "Plot multiple footprint overlays from a comma-separated set of detection ranges in km (e.g. 100,300,600).",
    "space_delay": "Override native detection with space-based cueing. Set delay in seconds (default 30). Mutually exclusive with fixed detection-range override.",
    "range_override": "Override the threat’s maximum range from rocket data for footprint math only.",
    "op_range_override": "For footprint display only: replace the interceptor’s operational range from rocket_data (km). Use for terminal-phase or notional envelope studies; does not change the interceptor trajectory table.",
    "assign_gbewr": "Early-warning radars whose detection arcs feed this defense site in Mode 2 / combined solves.",
    "procedures": "Launch detailed analysis windows from fp_gui.py using this site’s configuration.",
    "footprint_cfg": "Advanced solver parameters mirrored from fcc_config.json for this token.",
}

# User-facing labels for stored footprint calculation modes (ILP).
FOOTPRINT_CALC_MODES = {
    "classic": "Max range (each target at max range)",
    "from_mlp": "From launch point",
    "from_ranges": "Boresight at each max range",
    "full_sector": "All max distances, full sector",
    "pct100": "100% interceptable",
    "launch_box": "Launch box",
    "isolated_lp": "Isolated launch point",
}


VARIABLE_DICTIONARY = """
MISSILE / INTERCEPTOR DATA VARIABLES
====================================

m_key / i_key
  Missile or interceptor ID number.

type
  Human-readable label or description.

cd_type
  Drag coefficient model for boost phase:
  - "ls" = large solid-fuel rocket
  - "ll" = liquid-fuel rocket
  - "v2" = V2-style drag curve

m_st
  Stage mass (kg), comma-separated per stage.
  Example: "2059, 1190, 282" for 3 stages.

m_fu
  Fuel mass (kg) per stage.

v_ex
  Exhaust velocity (m/s) per stage.
  If value < 1000, interpreted as Isp (specific impulse, seconds) and multiplied by g.

t_bu
  Burn time (seconds) per stage.

t_delay
  Delay (seconds) between stage ignitions.

a_mid
  Cross-sectional area (m²) per stage, used for drag calculation.

m_warhead
  Warhead mass (kg). Empty = 0.

m_shroud
  Payload shroud mass (kg), added to payload.

t_shroud
  Time (seconds) of shroud jettison.

m_pl
  Payload mass (kg).

a_nz
  Nozzle area (m²) for atmospheric pressure thrust correction.
  0 = not used.

c_bal
  Ballistic coefficient (kg/m²), used for re-entry drag.

vert_launch_height
  Altitude (m) at which gravity turn starts.
  0 = ground launch. For ALBM: launch altitude.

grav_turn_angle
  Gravity turn angle (degrees) from vertical at turn start.

flight_path_angle
  (Interceptors) Flight path angle (degrees from horizontal) after vertical climb.

range
  Max range (km). For ALBM: "albm 830" = launch speed 830 m/s.

traj_type
  - "bal_mis" = ballistic missile
  - "int_exo" = exoatmospheric interceptor
  - "int_endo" = endoatmospheric interceptor

mpia
  (Interceptors) Minimum feasible intercept altitude (km).

op_range
  (Interceptors) Operational range (km). Empty = no limit.

det_range
  (Interceptors) Detection range (km) per missile type. Format: "m1:val, m2:val, ..."

note
  Optional notes.


TOKEN / FOOTPRINT CONFIG VARIABLES
==================================

mtype / itype
  Default missile type and interceptor type for footprint calculation.

h_int_min
  Minimum intercept altitude (km). Intercepts below this are rejected.

h_discr
  Warhead discrimination altitude (km). 0 = not used.

t_delay
  Interceptor launch delay (seconds) after detection or burn-out.

center_dir
  (ILP) Defense direction azimuth (degrees). 0 = North, 90 = East.

sector_width
  (ILP) Angular width (degrees) of the defense sector.

use_sectoral
  (ILP) Use sectoral footprint (intersection over sector angles).

det_range_override / range_override
  (ILP) Override detection or missile range from config.

use_op_range_override / op_range_override_km
  (ILP) Optional footprint-only replacement for interceptor operational range (km).
"""


def load_base_config():
    with open(CONFIG_PATH, "r") as cfg_file:
        return json.load(cfg_file)


def load_rocket_data():
    with open(ROCKET_DATA_PATH, "r") as data_file:
        return json.load(data_file)


def save_rocket_data(data):
    with open(ROCKET_DATA_PATH, "w") as data_file:
        json.dump(data, data_file, indent=4)


def save_gbewr_data(radars):
    with open(GBEWR_DATA_PATH, "w") as f:
        json.dump({"gbewr_radars": radars}, f, indent=4)


FOOTPRINT_XY_UNITS = "km"


def footprint_xy_to_meters(xy_points, units=FOOTPRINT_XY_UNITS):
    """
    Convert local footprint points to meters with explicit units.

    The footprint solver returns local x/y in kilometers; this function keeps
    the unit contract explicit to avoid accidental 1000x scaling errors.
    """
    if not xy_points:
        return []
    if units == "m":
        return [(float(x), float(y)) for x, y in xy_points]
    if units == "km":
        return [(float(x) * 1000.0, float(y) * 1000.0) for x, y in xy_points]
    raise ValueError(f"Unsupported footprint units: {units}")


def clamp(value, low, high):
    return max(low, min(high, value))


class Token:
    def __init__(self, token_id, name, lat, lon, base_config, kind="ilp"):
        self.token_id = token_id
        self.name = name
        self.lat = lat
        self.lon = lon
        self.kind = kind
        self.center_dir = 0.0
        self.sector_width = 60.0
        self.show_center = True
        self.show_sector = True
        self.use_sectoral = False
        # Threat-origin sampling arc (where the incoming missile is "coming from").
        self.show_threat_arc = True
        self.threat_arc_samples = 9
        self.isolated_azimuth = None       # deg; when set, footprint is computed/shown for this azimuth only
        self.isolated_missile_key = None   # the missile type of the isolated launch point
        self.threat_arc_dots = []          # runtime registry: {item, azimuth, missile_key, lat, lon}
        # Fixed launch point (single MLP) footprint mode
        self.fixed_mlp_latlon = None       # (lat, lon) of a placed threat launch point, or None
        self._saved_mlp_latlon = None      # launch point saved while an arc dot is isolated
        self.fixed_mlp_loft = False        # lofted (True) vs depressed (False) trajectory for sub-max ranges
        self.from_mlp_mode = False         # True when current footprints were computed from the fixed MLP
        self.full_sector_mode = False      # True when current footprints are the per-boresight full-sector set
        # Launch box: a drawn region MLPs are sampled from. dict with corners + firing missiles + grid.
        self.launch_box = None  # {"lat1","lon1","lat2","lon2","missile_keys":[...],"grid_n":int}
        self._mlp_btn_state = "idle"      # idle | placing | placed (stateful compute button)
        self._box_btn_state = "idle"      # idle | drawing | drawn
        self._calc_mode_offset = None     # (dx, dy) of the on-map calc-mode box relative to the ILP
        self.use_det_range_override = False
        self.det_range_override_km = 0.0
        self.use_det_sweep = False
        self.det_sweep_km_csv = ""
        self.use_space_based_delay = False
        self.use_range_override = False
        self.range_override_km = 0.0
        self.use_op_range_override = False
        self.op_range_override_km = 0.0
        self.config = json.loads(json.dumps(base_config))
        self.space_based_delay_s = float(self.config.get("set_sat_delay", fcc_constants.sat_delay))
        self.missile_keys = [int(self.config.get("mtype", 1))]
        self.interceptor_keys = [int(self.config.get("itype", 11))]
        self.footprints = []
        self.footprint_visibility = {}  # fp_id -> bool (display only; data kept in footprints)
        self.footprint_calculations = {}  # mode_id -> {label, footprints, footprint_visibility, ...}
        self.active_footprint_mode = None
        self.missile_ranges = []
        self.canvas_items = {}
        self.gbewr_assigned = []  # list of GBEWR token_ids assigned to this ILP
        self.r_key = None  # for gbewr tokens: radar spec key


class ScrollableFrame(tk.Frame):
    """Vertically scrollable region with dark styling."""

    def __init__(self, parent, width=420):
        super().__init__(parent, bg=THEME["panel_bg"])
        self.canvas = tk.Canvas(
            self,
            borderwidth=0,
            highlightthickness=0,
            width=width,
            bg=THEME["panel_inner"],
        )
        self.v_scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.h_scrollbar = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.inner = tk.Frame(self.canvas, bg=THEME["panel_inner"])

        self._win_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self._last_win_w = 0
        self.inner.bind("<Configure>", self._sync_scroll)
        self.canvas.bind("<Configure>", self._sync_scroll)
        self.canvas.configure(yscrollcommand=self.v_scrollbar.set, xscrollcommand=self.h_scrollbar.set)

        # vertical scrollbar always on the right; horizontal along the bottom
        self.v_scrollbar.pack(side="right", fill="y")
        self.h_scrollbar.pack(side="bottom", fill="x")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Shift-MouseWheel>", self._on_shift_mousewheel)

    def _sync_scroll(self, _event=None):
        # Make the inner window at least the canvas width (so content fills the panel),
        # but allow it to grow wider than the canvas so wide content scrolls horizontally
        # instead of being clipped.
        cw = self.canvas.winfo_width()
        req = self.inner.winfo_reqwidth()
        target = max(cw, req)
        if target != self._last_win_w:
            self._last_win_w = target
            self.canvas.itemconfigure(self._win_id, width=target)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_shift_mousewheel(self, event):
        self.canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")


class ToolTip:
    """Hover tooltip anchored to a widget."""

    def __init__(self, widget, text, delay_ms=400):
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self._win = None
        self._after = None
        widget.bind("<Enter>", self._schedule)
        widget.bind("<Leave>", self._hide)
        widget.bind("<ButtonPress>", self._hide)

    def set_text(self, text):
        self.text = text

    def _schedule(self, _event=None):
        self._cancel()
        self._after = self.widget.after(self.delay_ms, self._show)

    def _cancel(self):
        if self._after is not None:
            self.widget.after_cancel(self._after)
            self._after = None

    def _show(self):
        self._cancel()
        if not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self._win = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_attributes("-topmost", True)
        lbl = tk.Label(
            tw,
            text=self.text,
            justify="left",
            background="#2e3440",
            foreground=THEME["fg"],
            relief="solid",
            borderwidth=1,
            font=("TkDefaultFont", 10),
            wraplength=340,
            padx=10,
            pady=8,
        )
        lbl.pack()
        tw.geometry(f"+{x}+{y}")

    def _hide(self, _event=None):
        self._cancel()
        if self._win is not None:
            try:
                self._win.destroy()
            except tk.TclError:
                pass
            self._win = None


def _help_icon(parent, tip_text):
    """Small (?) control with tooltip."""
    try:
        pbg = parent.cget("bg")
    except tk.TclError:
        pbg = ""
    if not pbg:
        pbg = THEME["panel_inner"]
    b = tk.Label(
        parent,
        text="(?)",
        fg=THEME["accent"],
        bg=pbg,
        cursor="question_arrow",
        font=("TkDefaultFont", 9, "bold"),
    )
    ToolTip(b, tip_text, delay_ms=200)
    return b


class DraggablePanel(tk.Frame):
    """Floating panel with a draggable title bar."""

    def __init__(self, parent, title, width=320, height=520, on_drag_end=None, **kwargs):
        bg = kwargs.pop("bg", THEME["panel_bg"])
        super().__init__(parent, width=width, height=height, bg=bg, highlightthickness=1, highlightbackground=THEME["border"], **kwargs)
        self._parent = parent
        self._on_drag_end = on_drag_end

        self.title_bar = tk.Frame(self, bg=THEME["titlebar"], height=36)
        self.title_bar.pack(fill="x")
        self._grip = tk.Label(
            self.title_bar,
            text="  ⋮⋮  ",
            bg=THEME["titlebar"],
            fg=THEME["fg_muted"],
            font=("TkDefaultFont", 10),
        )
        self._grip.pack(side="left", padx=(6, 0))
        self.title_label = tk.Label(
            self.title_bar,
            text=title,
            bg=THEME["titlebar"],
            fg=THEME["fg"],
            font=("TkDefaultFont", 11, "bold"),
        )
        self.title_label.pack(side="left", padx=4, pady=8)

        for w in (self.title_bar, self.title_label, self._grip):
            w.bind("<ButtonPress-1>", self._drag_start)
            w.bind("<B1-Motion>", self._drag_motion)
            w.bind("<ButtonRelease-1>", self._drag_end)

        self.body = tk.Frame(self, bg=THEME["panel_bg"])
        self.body.pack(fill="both", expand=True)

        self.bind("<ButtonPress-1>", lambda e: self.lift())

    def set_title(self, title):
        self.title_label.configure(text=title)

    def _drag_start(self, event):
        self.lift()
        self._mx = event.x_root - self.winfo_rootx()
        self._my = event.y_root - self.winfo_rooty()

    def _drag_motion(self, event):
        if not hasattr(self, "_mx"):
            return
        nx = event.x_root - self._parent.winfo_rootx() - self._mx
        ny = event.y_root - self._parent.winfo_rooty() - self._my
        pw = self._parent.winfo_width()
        ph = self._parent.winfo_height()
        nw = self.winfo_width()
        nh = self.winfo_height()
        nx = max(0, min(nx, max(0, pw - nw)))
        ny = max(0, min(ny, max(0, ph - nh)))
        self.place(x=nx, y=ny)

    def _drag_end(self, _event):
        if hasattr(self, "_mx"):
            del self._mx
            del self._my
        if self._on_drag_end:
            self._on_drag_end()


class MapApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MDFCC — scenario map")

        self.base_config = load_base_config()
        self.rocket_data = load_rocket_data()
        self.tokens = {}
        self.selected_token = None
        self._missile_range_cache = {}  # m_key -> range km (cleared when rocket data changes)
        self._suppress_sidebar_trace = False  # guard: ignore field traces during programmatic var updates
        self.placing_fixed_mlp_for = None  # token_id awaiting a fixed-MLP placement click
        self.drawing_launch_box_for = None  # token_id awaiting a drag-drawn launch box
        self._launch_box_start = None       # (x, y) press point while rubber-banding the box
        self._launch_box_rubber = None      # canvas id of the rubber-band rectangle
        self._launch_box_popup = None       # missile-selection popup for the launch box
        self.add_mode = None
        self.drag_state = None
        self.map_image = None
        self.map_image_item = None
        self.map_render_job = None
        self.map_fast_render_job = None
        self.tile_images = []
        self.tile_cache = {}
        self._b1_press_xy = None
        self._left_pan_armed = False
        self._left_panning = False
        self._left_pan_last = None
        self.zoom = 1.0
        self.zoom_var = tk.DoubleVar(value=self.zoom)
        self._zoom_slider_updating = False
        self.view_center_lat = 0.0
        self.view_center_lon = 0.0
        self._is_macos = str(self.root.tk.call("tk", "windowingsystem")) == "aqua"
        self.left_panel_visible = True
        self._left_panel_saved_geom = (12, 64)
        self._right_panel_user_pos = False
        self.coverage_mode = None
        self.coverage_pins = []
        self.distance_mode = None
        self._token_press_moved = False
        self._footprint_layer_popup = None
        self._interceptor_flight_preview_win = None

        self._build_ui()
        self.root.after_idle(self._draw_base_map)

    def _apply_ttk_theme(self):
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")
        bg = THEME["panel_bg"]
        fg = THEME["fg"]
        style.configure(".", background=THEME["root_bg"], foreground=fg, fieldbackground=THEME["entry_bg"])
        style.configure("TFrame", background=bg)
        style.configure("TLabel", background=bg, foreground=fg)
        style.configure("TButton", padding=(12, 7), relief="flat", borderwidth=0,
                        background=THEME["titlebar"], foreground=fg, focuscolor=THEME["accent"])
        style.map(
            "TButton",
            background=[("pressed", THEME["accent_dim"]), ("active", THEME["border"]),
                        ("!disabled", THEME["titlebar"])],
            foreground=[("!disabled", fg)],
        )
        # Primary (accent) action button
        style.configure("Primary.TButton", padding=(12, 7), relief="flat", borderwidth=0,
                        background=THEME["accent"], foreground=THEME["accent_fg"])
        style.map("Primary.TButton",
                  background=[("pressed", THEME["primary_btn_active"]), ("active", THEME["primary_btn_hover"]),
                              ("!disabled", THEME["accent"])],
                  foreground=[("!disabled", THEME["accent_fg"])])
        style.configure("TNotebook", background=bg, borderwidth=0)
        style.configure("TNotebook.Tab", background=THEME["panel_inner"], foreground=THEME["fg_muted"],
                        padding=(14, 7), borderwidth=0)
        style.map("TNotebook.Tab",
                  background=[("selected", THEME["panel_bg"])],
                  foreground=[("selected", THEME["accent"])])
        # Entries with an accent focus ring
        style.configure("TEntry", fieldbackground=THEME["entry_bg"], foreground=fg, insertcolor=fg,
                        bordercolor=THEME["border"], lightcolor=THEME["border"], darkcolor=THEME["border"],
                        padding=4)
        style.map("TEntry",
                  bordercolor=[("focus", THEME["accent"])],
                  lightcolor=[("focus", THEME["accent"])],
                  darkcolor=[("focus", THEME["accent"])])
        style.configure("TCheckbutton", background=bg, foreground=fg, focuscolor=THEME["accent"])
        style.map("TCheckbutton",
                  indicatorcolor=[("selected", THEME["accent"]), ("!selected", THEME["entry_bg"])],
                  foreground=[("active", THEME["accent"])])
        style.configure("TRadiobutton", background=bg, foreground=fg, focuscolor=THEME["accent"])
        style.map("TRadiobutton",
                  indicatorcolor=[("selected", THEME["accent"]), ("!selected", THEME["entry_bg"])],
                  foreground=[("active", THEME["accent"])])
        style.configure(
            "TCombobox",
            fieldbackground=THEME["entry_bg"],
            background=THEME["entry_bg"],
            foreground=fg,
            arrowcolor=THEME["accent"],
            bordercolor=THEME["border"],
            padding=4,
        )
        style.map("TCombobox",
                  fieldbackground=[("readonly", THEME["entry_bg"])],
                  bordercolor=[("focus", THEME["accent"])])
        style.configure("TScale", background=bg, troughcolor=THEME["entry_bg"])
        style.configure("Vertical.TScrollbar", background=THEME["border"], troughcolor=THEME["panel_inner"],
                        bordercolor=THEME["panel_inner"], arrowcolor=THEME["fg_muted"], arrowsize=12)
        style.configure("Horizontal.TScrollbar", background=THEME["border"], troughcolor=THEME["panel_inner"],
                        bordercolor=THEME["panel_inner"], arrowcolor=THEME["fg_muted"], arrowsize=12)

    def _map_tool_btn(self, parent, text, command, tooltip=None, primary=False):
        bg = THEME["tool_btn_bg"]; hover = THEME["tool_btn_hover"]
        act_bg = THEME["tool_btn_active_bg"]
        if primary:
            fg = THEME["primary_text"]; act_fg = THEME["primary_text"]
            hl, hlc = 2, THEME["accent"]
        else:
            fg = THEME["tool_btn_fg"]; act_fg = THEME["tool_btn_active_fg"]
            hl, hlc = 0, bg
        b = tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg,
            fg=fg,
            activebackground=act_bg,
            activeforeground=act_fg,
            relief="flat",
            borderwidth=0,
            padx=14,
            pady=8,
            cursor="hand2",
            font=("TkDefaultFont", 10, "bold"),
            highlightthickness=hl,
            highlightbackground=hlc,
            highlightcolor=hlc,
        )

        def on_enter(_e):
            b.configure(bg=hover, fg=fg)

        def on_leave(_e):
            b.configure(bg=bg, fg=fg)

        b.bind("<Enter>", on_enter)
        b.bind("<Leave>", on_leave)
        if tooltip:
            ToolTip(b, tooltip)
        return b

    def _shell_place_panels(self, _event=None):
        """Keep floating panels sized within the map area."""
        self.map_shell.update_idletasks()
        th = self.map_toolbar.winfo_height()
        h = max(420, self.map_shell.winfo_height() - th - 20)
        if self.left_panel_visible:
            try:
                self.left_draggable.place_configure(height=h)
            except tk.TclError:
                pass
        try:
            self.right_draggable.place_configure(height=h)
        except tk.TclError:
            pass
        if not self._right_panel_user_pos:
            w = self.map_shell.winfo_width()
            rw = self.right_draggable.winfo_width()
            self.right_draggable.place(x=max(8, w - rw - 14), y=th + 8)

    def _on_right_panel_drag_end(self, event):
        DraggablePanel._drag_end(self.right_draggable, event)
        self._right_panel_user_pos = True

    def _build_ui(self):
        self.root.geometry("1400x800")
        self.root.configure(bg=THEME["root_bg"])
        self._apply_ttk_theme()
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=0)
        self.root.grid_rowconfigure(1, weight=1)

        attribution = tk.Label(
            self.root,
            text="MDFCC (MIT License) © 2025 tim-kad — includes MIT-licensed components.",
            anchor="center",
            bg=THEME["root_bg"],
            fg=THEME["fg_muted"],
            font=("TkDefaultFont", 9),
        )
        attribution.grid(row=0, column=0, sticky="ew", padx=6, pady=(6, 4))
        ToolTip(attribution, HELP["win_title"], delay_ms=600)

        self.map_shell = tk.Frame(self.root, bg=THEME["root_bg"])
        self.map_shell.grid(row=1, column=0, sticky="nsew")
        self.map_shell.grid_rowconfigure(1, weight=1)
        self.map_shell.grid_columnconfigure(0, weight=1)

        self.map_toolbar = tk.Frame(self.map_shell, bg=THEME["toolbar_bg"], height=52)
        self.map_toolbar.grid(row=0, column=0, sticky="ew")
        self.map_toolbar.grid_columnconfigure(9, weight=1)

        self.check_coverage_btn = self._map_tool_btn(
            self.map_toolbar, "Coverage check", self._start_coverage_check, HELP["check_coverage"]
        )
        self.check_coverage_btn.grid(row=0, column=0, padx=(12, 6), pady=8)

        self.measure_distance_btn = self._map_tool_btn(
            self.map_toolbar, "Measure distance", self._start_distance_measure, HELP["measure_distance"]
        )
        self.measure_distance_btn.grid(row=0, column=1, padx=6, pady=8)

        tk.Frame(self.map_toolbar, width=1, bg=THEME["border"]).grid(row=0, column=2, sticky="ns", padx=8, pady=10)

        self.toggle_left_btn = self._map_tool_btn(self.map_toolbar, "Hide library", self._toggle_left_panel, HELP["toggle_library"])
        self.toggle_left_btn.grid(row=0, column=3, padx=4, pady=8)

        self.add_ilp_btn = self._map_tool_btn(
            self.map_toolbar, "Add defense site", lambda: self._set_add_mode("ilp"), HELP["add_ilp"]
        )
        self.add_ilp_btn.grid(row=0, column=4, padx=4, pady=8)

        self.add_mlp_btn = self._map_tool_btn(
            self.map_toolbar, "Add threat launch", lambda: self._set_add_mode("mlp"), HELP["add_mlp"]
        )
        self.add_mlp_btn.grid(row=0, column=5, padx=4, pady=8)

        self.add_gbewr_btn = self._map_tool_btn(
            self.map_toolbar, "Add radar", lambda: self._set_add_mode("gbewr"), HELP["add_gbewr"]
        )
        self.add_gbewr_btn.grid(row=0, column=6, padx=4, pady=8, sticky="w")

        tk.Frame(self.map_toolbar, width=1, bg=THEME["border"]).grid(row=0, column=7, sticky="ns", padx=8, pady=10)

        self.builder_btn = self._map_tool_btn(
            self.map_toolbar, "✦ Builder", self._open_builder, HELP["builder"], primary=True
        )
        self.builder_btn.grid(row=0, column=8, padx=4, pady=8, sticky="w")

        # column 9 is the stretchy spacer; the busy spinner + status sit on the far right
        self._spin_canvas = tk.Canvas(self.map_toolbar, width=18, height=18,
                                      bg=THEME["toolbar_bg"], highlightthickness=0)
        self._spin_canvas.grid(row=0, column=10, padx=(8, 0), pady=8, sticky="e")
        self._spin_canvas.grid_remove()  # hidden until busy
        self._spin_angle = 0
        self._spin_job = None

        self.status_var = tk.StringVar(value="Ready")
        self.status_label = tk.Label(
            self.map_toolbar,
            textvariable=self.status_var,
            bg=THEME["toolbar_bg"],
            fg=THEME["success"],
            font=("TkDefaultFont", 10),
        )
        self.status_label.grid(row=0, column=11, padx=(6, 16), pady=8, sticky="e")
        # animate the spinner whenever a "Computing…" status is showing
        self.status_var.trace_add("write", self._on_status_change)

        self.map_canvas = tk.Canvas(self.map_shell, bg="#ffffff", highlightthickness=0)
        self.map_canvas.grid(row=1, column=0, sticky="nsew")

        self.zoom_slider_frame = tk.Frame(self.map_shell, bg=THEME["toolbar_bg"], highlightthickness=1, highlightbackground=THEME["border"])
        self.zoom_slider_frame.place(x=12, rely=1.0, y=-18, anchor="sw")
        tk.Label(
            self.zoom_slider_frame,
            text="Zoom",
            bg=THEME["toolbar_bg"],
            fg=THEME["fg"],
            font=("TkDefaultFont", 9, "bold"),
        ).pack(side="left", padx=(8, 6), pady=4)
        self.zoom_slider = tk.Scale(
            self.zoom_slider_frame,
            from_=MIN_ZOOM,
            to=MAX_ZOOM,
            orient="horizontal",
            variable=self.zoom_var,
            command=self._on_zoom_slider_drag,
            resolution=0.1,
            showvalue=False,
            length=170,
            bg=THEME["toolbar_bg"],
            fg=THEME["fg"],
            troughcolor=THEME["panel_inner"],
            activebackground=THEME["accent"],
            highlightthickness=0,
            borderwidth=0,
        )
        self.zoom_slider.pack(side="left", padx=(0, 8), pady=3)

        self.left_draggable = DraggablePanel(self.map_shell, "Type library", width=312, height=560)
        self.left_draggable.place(x=12, y=64)
        self._build_left_panel()

        self.right_draggable = DraggablePanel(
            self.map_shell,
            "Inspector",
            width=438,
            height=560,
            on_drag_end=lambda: setattr(self, "_right_panel_user_pos", True),
        )
        self.right_draggable.place(x=920, y=64)

        self.scrollable = ScrollableFrame(self.right_draggable.body, width=420)
        self.scrollable.pack(fill="both", expand=True)
        ToolTip(self.right_draggable.title_label, HELP["inspector"], delay_ms=500)

        self.sidebar_vars = {}
        self._build_sidebar(None)

        self.map_shell.bind("<Configure>", self._shell_place_panels)
        self.root.after_idle(self._shell_place_panels)

        self.map_canvas.bind("<ButtonPress-1>", self._on_map_button1_press)
        self.map_canvas.bind("<B1-Motion>", self._on_map_drag)
        self.map_canvas.bind("<ButtonRelease-1>", self._on_map_button1_release)
        self.map_canvas.bind("<Configure>", self._on_canvas_resize)
        self.map_canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.map_canvas.bind("<Button-4>", self._on_mousewheel)
        self.map_canvas.bind("<Button-5>", self._on_mousewheel)
        self.map_canvas.bind("<Button-3>", self._on_pan_start)
        self.map_canvas.bind("<B3-Motion>", self._on_pan_drag)
        self.map_canvas.bind("<ButtonRelease-3>", self._on_pan_end)

        self.root.bind("<Left>", lambda e: self._pan_by_pixels(-80, 0))
        self.root.bind("<Right>", lambda e: self._pan_by_pixels(80, 0))
        self.root.bind("<Up>", lambda e: self._pan_by_pixels(0, -80))
        self.root.bind("<Down>", lambda e: self._pan_by_pixels(0, 80))
        self.root.bind("<Shift-Left>", lambda e: self._pan_by_pixels(-240, 0))
        self.root.bind("<Shift-Right>", lambda e: self._pan_by_pixels(240, 0))
        self.root.bind("<Shift-Up>", lambda e: self._pan_by_pixels(0, -240))
        self.root.bind("<Shift-Down>", lambda e: self._pan_by_pixels(0, 240))

    def _build_left_panel(self):
        parent = self.left_draggable.body
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)
        notebook = ttk.Notebook(parent)
        notebook.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

        missiles_tab = ttk.Frame(notebook)
        interceptors_tab = ttk.Frame(notebook)
        gbewr_tab = ttk.Frame(notebook)
        dict_tab = ttk.Frame(notebook)
        notebook.add(missiles_tab, text="Missile types")
        notebook.add(interceptors_tab, text="Interceptor types")
        notebook.add(gbewr_tab, text="Radar presets")
        notebook.add(dict_tab, text="Field reference")

        self._build_type_editor(missiles_tab, "missile")
        self._build_type_editor(interceptors_tab, "interceptor")
        self._build_type_editor(gbewr_tab, "gbewr")
        self._build_dictionary_tab(dict_tab)

        ToolTip(self.left_draggable.title_label, HELP["library"], delay_ms=500)

    def _lib_btn(self, parent, text, command, fg_color=None, active_fg=None, primary=False, bg_color=None):
        # tk.Button bg is unreliable on macOS (light native chip), so keep the light chip
        # and use dark/high-contrast labels; "primary" adds a bold accent label + border.
        bg = bg_color if bg_color is not None else THEME["tool_btn_bg"]
        hover = THEME["tool_btn_hover"] if bg_color is None else bg
        if primary:
            fg = fg_color if fg_color is not None else THEME["primary_text"]
            act_fg = active_fg or fg
            hl, hlc = 2, THEME["accent"]
        else:
            fg = fg_color if fg_color is not None else THEME["tool_btn_fg"]
            act_fg = active_fg or THEME["tool_btn_active_fg"]
            hl, hlc = 0, bg
        b = tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg,
            fg=fg,
            activebackground=THEME["tool_btn_active_bg"] if bg_color is None else bg,
            activeforeground=act_fg,
            relief="flat",
            borderwidth=0,
            padx=12,
            pady=6,
            cursor="hand2",
            font=("TkDefaultFont", 9, "bold"),
            highlightthickness=hl,
            highlightbackground=hlc,
            highlightcolor=hlc,
        )

        def on_enter(_e):
            b.configure(bg=hover, fg=fg)

        def on_leave(_e):
            b.configure(bg=bg, fg=fg)

        b.bind("<Enter>", on_enter)
        b.bind("<Leave>", on_leave)
        return b

    @staticmethod
    def _type_editor_key_order(kind, item):
        keys = list(item.keys())
        if kind == "gbewr":
            order = ["r_key", "type", "range_km", "note"]
            return [k for k in order if k in keys] + sorted(k for k in keys if k not in order)
        pk = "m_key" if kind == "missile" else "i_key"
        priority = [pk, "type", "note"]
        rest = sorted(k for k in keys if k not in priority)
        return [k for k in priority if k in keys] + rest

    def _library_field_label(self, kind, key):
        gb = {
            "r_key": "Radar ID",
            "type": "Name",
            "range_km": "Range (km)",
            "note": "Notes",
        }
        if kind == "gbewr":
            return gb.get(key, self._config_label(key))
        pk = "m_key" if kind == "missile" else "i_key"
        if key == pk:
            return "Missile catalog #" if kind == "missile" else "Interceptor catalog #"
        return self._config_label(key)

    def _coerce_value_to_ref_type(self, raw, old_val):
        if isinstance(raw, str):
            s = raw.strip()
        else:
            s = str(raw).strip()
        if old_val is None:
            return s if s else None
        if isinstance(old_val, bool):
            return s.lower() in ("1", "true", "yes", "on")
        if isinstance(old_val, int) and not isinstance(old_val, bool):
            if not s:
                return 0
            try:
                return int(float(s))
            except ValueError:
                return old_val
        if isinstance(old_val, float):
            if not s:
                return 0.0
            try:
                return float(s)
            except ValueError:
                return old_val
        return s

    def _clear_type_fields(self, kind):
        ui = self.type_ui.get(kind)
        if not ui or "fields_inner" not in ui:
            return
        for child in ui["fields_inner"].winfo_children():
            child.destroy()
        ui["field_vars"].clear()
        ui["ref_item"] = None

    def _populate_type_fields(self, kind, item):
        ui = self.type_ui[kind]
        self._clear_type_fields(kind)
        ui["ref_item"] = dict(item)
        inner = ui["fields_inner"]
        ui["field_vars"] = {}
        for key in self._type_editor_key_order(kind, item):
            row = tk.Frame(inner, bg=THEME["panel_inner"])
            row.pack(fill="x", pady=3, padx=2)
            lab = tk.Label(
                row,
                text=self._library_field_label(kind, key),
                width=16,
                anchor="w",
                bg=THEME["panel_inner"],
                fg=THEME["fg_muted"],
                font=("TkDefaultFont", 9),
            )
            lab.pack(side="left", padx=(2, 6))
            val = item.get(key, "")
            if val is None:
                disp = ""
            else:
                disp = str(val)
            var = tk.StringVar(value=disp)
            ent = ttk.Entry(row, textvariable=var, width=28)
            ent.pack(side="left", fill="x", expand=True)
            ui["field_vars"][key] = var

    def _collect_type_editor(self, kind):
        ui = self.type_ui[kind]
        ref = ui.get("ref_item") or {}
        data = {}
        for key, var in ui["field_vars"].items():
            raw = var.get()
            old = ref.get(key, "")
            data[key] = self._coerce_value_to_ref_type(raw, old)
        return data

    def _bind_field_canvas_scroll(self, canvas):
        def wheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind("<MouseWheel>", wheel)
        canvas.bind("<Button-4>", lambda e: canvas.yview_scroll(-2, "units"))
        canvas.bind("<Button-5>", lambda e: canvas.yview_scroll(2, "units"))

    def _build_type_editor(self, parent, kind):
        parent.grid_rowconfigure(1, weight=1)
        parent.grid_columnconfigure(1, weight=1)

        list_frame = ttk.Frame(parent)
        list_frame.grid(row=0, column=0, rowspan=2, sticky="nsw", padx=(4, 2), pady=4)
        list_frame.grid_rowconfigure(0, weight=1)

        lb_kw = dict(
            width=28,
            height=18,
            exportselection=False,
            bg=THEME["list_bg"],
            fg=THEME["list_fg"],
            selectbackground=THEME["select_bg"],
            selectforeground=THEME["fg"],
            relief="flat",
            highlightthickness=1,
            highlightbackground=THEME["border"],
            font=("TkDefaultFont", 10),
        )
        listbox = tk.Listbox(list_frame, **lb_kw)
        list_scroll = ttk.Scrollbar(list_frame, orient="vertical", command=listbox.yview)
        listbox.configure(yscrollcommand=list_scroll.set)
        listbox.grid(row=0, column=0, sticky="ns")
        list_scroll.grid(row=0, column=1, sticky="ns")

        editor_wrap = tk.Frame(parent, bg=THEME["panel_bg"])
        editor_wrap.grid(row=1, column=1, sticky="nsew", padx=(2, 4), pady=(0, 4))
        editor_wrap.grid_rowconfigure(0, weight=1)
        editor_wrap.grid_columnconfigure(0, weight=1)

        fld_canvas = tk.Canvas(
            editor_wrap,
            bg=THEME["panel_inner"],
            highlightthickness=0,
            borderwidth=0,
        )
        fld_scroll = ttk.Scrollbar(editor_wrap, orient="vertical", command=fld_canvas.yview)
        fields_inner = tk.Frame(fld_canvas, bg=THEME["panel_inner"])
        fld_win = fld_canvas.create_window((0, 0), window=fields_inner, anchor="nw")

        def cfg_scroll(_event=None):
            fld_canvas.configure(scrollregion=fld_canvas.bbox("all"))

        def cfg_width(event):
            fld_canvas.itemconfigure(fld_win, width=event.width)

        fields_inner.bind("<Configure>", cfg_scroll)
        fld_canvas.bind("<Configure>", cfg_width)
        fld_canvas.configure(yscrollcommand=fld_scroll.set)
        fld_canvas.grid(row=0, column=0, sticky="nsew")
        fld_scroll.grid(row=0, column=1, sticky="ns")
        self._bind_field_canvas_scroll(fld_canvas)

        btn_row = tk.Frame(parent, bg=THEME["panel_bg"])
        btn_row.grid(row=0, column=1, sticky="ew", padx=(2, 4), pady=(4, 2))
        btn_row.grid_columnconfigure(2, weight=1)

        add_btn = self._lib_btn(btn_row, "New entry", lambda: self._add_type(kind))
        save_btn = self._lib_btn(btn_row, "Save to file", lambda: self._save_type(kind))
        refresh_btn = self._lib_btn(btn_row, "Reload list", lambda: self._refresh_type_list(kind))
        add_btn.grid(row=0, column=0, padx=2)
        save_btn.grid(row=0, column=1, padx=2)
        refresh_btn.grid(row=0, column=2, sticky="e", padx=2)
        ToolTip(add_btn, "Append a template entry or duplicate from JSON.")
        ToolTip(save_btn, "Write the edited fields back to the data file for the selected ID.")
        ToolTip(refresh_btn, "Reload lists from disk without restarting.")

        if not hasattr(self, "type_ui"):
            self.type_ui = {}
        self.type_ui[kind] = {
            "listbox": listbox,
            "fields_inner": fields_inner,
            "field_vars": {},
            "ref_item": None,
            "kind": kind,
        }

        listbox.bind("<<ListboxSelect>>", lambda e, k=kind: self._load_type_into_editor(k))
        self._refresh_type_list(kind)

    def _build_dictionary_tab(self, parent):
        parent.grid_rowconfigure(1, weight=1)
        parent.grid_columnconfigure(0, weight=1)
        hint = tk.Frame(parent, bg=THEME["panel_bg"])
        hint.grid(row=0, column=0, sticky="ew", padx=4, pady=(4, 0))
        tk.Label(
            hint,
            text="Technical names used in JSON data files",
            bg=THEME["panel_bg"],
            fg=THEME["fg_muted"],
            font=("TkDefaultFont", 9),
        ).pack(side="left")
        _help_icon(hint, HELP["dict_tab"]).pack(side="left", padx=4)

        dict_frame = ttk.Frame(parent)
        dict_frame.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        dict_frame.grid_rowconfigure(0, weight=1)
        dict_frame.grid_columnconfigure(0, weight=1)
        dict_text = tk.Text(
            dict_frame,
            wrap="word",
            font=("TkDefaultFont", 10),
            padx=8,
            pady=8,
            bg=THEME["entry_bg"],
            fg=THEME["fg"],
            relief="flat",
            highlightthickness=1,
            highlightbackground=THEME["border"],
        )
        dict_scroll = ttk.Scrollbar(dict_frame, orient="vertical", command=dict_text.yview)
        dict_text.configure(yscrollcommand=dict_scroll.set)
        dict_text.grid(row=0, column=0, sticky="nsew")
        dict_scroll.grid(row=0, column=1, sticky="ns")
        dict_text.insert("1.0", VARIABLE_DICTIONARY.strip())
        dict_text.configure(state="disabled")

    def _refresh_type_list(self, kind):
        ui = self.type_ui[kind]
        listbox = ui["listbox"]
        listbox.delete(0, tk.END)
        if kind == "gbewr":
            items = self._get_gbewr_radars()
            key_name = "r_key"
        else:
            items = self._get_missiles() if kind == "missile" else self._get_interceptors()
            key_name = "m_key" if kind == "missile" else "i_key"
        for item in items:
            key_val = item.get(key_name, "")
            name = item.get("type", "").strip() or "(no label)"
            if kind == "gbewr":
                label = f"Radar #{key_val} — {name} ({item.get('range_km', '?')} km)"
            elif kind == "missile":
                label = f"Missile #{key_val} — {name}"
            else:
                label = f"Interceptor #{key_val} — {name}"
            listbox.insert(tk.END, label)
        if items:
            listbox.selection_set(0)
            self._load_type_into_editor(kind)

    def _load_type_into_editor(self, kind):
        ui = self.type_ui[kind]
        listbox = ui["listbox"]
        if kind == "gbewr":
            items = self._get_gbewr_radars()
        else:
            items = self._get_missiles() if kind == "missile" else self._get_interceptors()
        selection = listbox.curselection()
        if not selection or selection[0] >= len(items):
            return
        item = items[selection[0]]
        self._populate_type_fields(kind, item)

    def _add_type(self, kind):
        if kind == "gbewr":
            items = self._get_gbewr_radars()
            key_name = "r_key"
            max_key = max([int(x.get(key_name, 0)) for x in items], default=0)
            new_key = max_key + 1
            new_item = {
                "r_key": new_key,
                "type": f"New radar {new_key}",
                "range_km": 1000,
                "note": ""
            }
            items.append(new_item)
            save_gbewr_data(items)
            self._refresh_type_list(kind)
            return
        items = self._get_missiles() if kind == "missile" else self._get_interceptors()
        key_name = "m_key" if kind == "missile" else "i_key"
        max_key = max([int(x.get(key_name, 0)) for x in items], default=0)
        new_key = max_key + 1
        if kind == "missile":
            new_item = {
                key_name: new_key,
                "type": f"New missile {new_key}",
                "cd_type": "v2",
                "m_st": "1330",
                "m_fu": "7400",
                "v_ex": "2254.0",
                "t_bu": "127.8",
                "t_delay": "",
                "a_mid": "d 1",
                "m_warhead": "",
                "m_shroud": "",
                "t_shroud": "",
                "m_pl": "500.0",
                "a_nz": "0.135",
                "c_bal": "7320",
                "vert_launch_height": "100",
                "grav_turn_angle": "0.7",
                "range": "1000",
                "traj_type": "bal_mis",
                "note": ""
            }
            self.rocket_data[0].append(new_item)
        else:
            new_item = {
                key_name: new_key,
                "type": f"New interceptor {new_key}",
                "cd_type": "ls",
                "m_st": "0",
                "m_fu": "80",
                "v_ex": "2800",
                "t_bu": "2",
                "t_delay": "",
                "a_mid": "0.042",
                "m_warhead": "",
                "m_shroud": "0",
                "t_shroud": "0",
                "m_pl": "150.0",
                "a_nz": "0",
                "c_bal": "14300",
                "vert_launch_height": "0.0",
                "flight_path_angle": "40",
                "range": "100",
                "traj_type": "int_endo",
                "mpia": "",
                "maxia": "",
                "op_range": "",
                "det_range": "1:100, 2:100, 3:100, 4:100, 5:100, 6:100, 7:100, 8:100, 9:100, 10:100, 11:100, 12:100, 13:100",
                "note": ""
            }
            self.rocket_data[1].append(new_item)
        save_rocket_data(self.rocket_data)
        self._missile_range_cache.clear()
        self._refresh_type_list(kind)

    def _save_type(self, kind):
        ui = self.type_ui[kind]
        listbox = ui["listbox"]
        if kind == "gbewr":
            items = self._get_gbewr_radars()
        else:
            items = self._get_missiles() if kind == "missile" else self._get_interceptors()
        selection = listbox.curselection()
        if not selection or selection[0] >= len(items):
            return
        try:
            data = self._collect_type_editor(kind)
        except (TypeError, ValueError) as exc:
            self.status_var.set(f"Invalid field value: {exc}")
            return

        key_name = "r_key" if kind == "gbewr" else ("m_key" if kind == "missile" else "i_key")
        if key_name not in data:
            self.status_var.set(f"Missing {key_name} in editor")
            return

        try:
            data[key_name] = int(data[key_name])
        except (TypeError, ValueError):
            self.status_var.set(f"{key_name} must be an integer")
            return
        target_key = data[key_name]
        updated = False
        if kind == "gbewr":
            for idx, item in enumerate(items):
                if int(item.get(key_name, -1)) == target_key:
                    items[idx] = data
                    updated = True
                    break
            if not updated:
                items.append(data)
            save_gbewr_data(items)
        else:
            container = self.rocket_data[0] if kind == "missile" else self.rocket_data[1]
            for idx, item in enumerate(container):
                if int(item.get(key_name, -1)) == target_key:
                    container[idx] = data
                    updated = True
                    break
            if not updated:
                container.append(data)
            save_rocket_data(self.rocket_data)
            self._missile_range_cache.clear()

        self._refresh_type_list(kind)
        if self.selected_token:
            self._build_sidebar(self.selected_token)

    def _toggle_left_panel(self):
        if self.left_panel_visible:
            self._left_panel_saved_geom = (self.left_draggable.winfo_x(), self.left_draggable.winfo_y())
            self.left_draggable.place_forget()
            self.left_panel_visible = False
            self.toggle_left_btn.configure(text="Show library")
        else:
            x, y = self._left_panel_saved_geom
            th = self.map_toolbar.winfo_height()
            self.left_draggable.place(x=x, y=y)
            self.left_panel_visible = True
            self.toggle_left_btn.configure(text="Hide library")
            self._shell_place_panels()
        self._on_canvas_resize(None)

    def _insp_container(self):
        return getattr(self, "_insp_host", None) or self.scrollable.inner

    def _inspector_sep(self):
        ttk.Separator(self._insp_container(), orient="horizontal").pack(fill="x", padx=6, pady=8)

    def _inspector_heading(self, text, help_key=None):
        row = tk.Frame(self._insp_container(), bg=THEME["panel_inner"])
        row.pack(fill="x", padx=6, pady=(10, 4))
        tk.Label(
            row,
            text=text,
            bg=THEME["panel_inner"],
            fg=THEME["fg"],
            font=("TkDefaultFont", 11, "bold"),
        ).pack(side="left")
        if help_key:
            _help_icon(row, HELP[help_key]).pack(side="left", padx=6)

    def _collapsible(self, title, help_key=None, opened=True):
        """A click-to-collapse section. Widgets created after this (via the layout
        helpers, or parented to the returned body) live inside the section."""
        root = self.scrollable.inner
        sec = tk.Frame(root, bg=THEME["panel_inner"])
        sec.pack(fill="x", padx=2, pady=(6, 0))
        header = tk.Frame(sec, bg=THEME["titlebar"], cursor="hand2")
        header.pack(fill="x")
        state = {"open": opened}
        arrow = tk.Label(header, text=("▾" if opened else "▸"), bg=THEME["titlebar"],
                         fg=THEME["accent"], font=("TkDefaultFont", 10, "bold"))
        arrow.pack(side="left", padx=(8, 4), pady=4)
        lbl = tk.Label(header, text=title, bg=THEME["titlebar"], fg=THEME["fg"],
                       font=("TkDefaultFont", 11, "bold"))
        lbl.pack(side="left", pady=4)
        if help_key:
            _help_icon(header, HELP[help_key]).pack(side="left", padx=6)
        body = tk.Frame(sec, bg=THEME["panel_inner"])
        if opened:
            body.pack(fill="x")

        def toggle(_e=None):
            state["open"] = not state["open"]
            arrow.config(text="▾" if state["open"] else "▸")
            if state["open"]:
                body.pack(fill="x")
            else:
                body.pack_forget()
        for w in (header, arrow, lbl):
            w.bind("<Button-1>", toggle)
        self._insp_host = body
        return body

    def _end_section(self):
        self._insp_host = self.scrollable.inner

    def _sidebar_list_kw(self, height):
        return dict(
            height=height,
            exportselection=False,
            bg=THEME["list_bg"],
            fg=THEME["list_fg"],
            selectbackground=THEME["select_bg"],
            selectforeground=THEME["fg"],
            relief="flat",
            highlightthickness=1,
            highlightbackground=THEME["border"],
            font=("TkDefaultFont", 10),
        )

    def _build_sidebar(self, token):
        for child in self.scrollable.inner.winfo_children():
            child.destroy()
        self.sidebar_vars.clear()

        inner = self.scrollable.inner
        self._insp_host = inner
        if token is None:
            f = tk.Frame(inner, bg=THEME["panel_inner"])
            f.pack(fill="x", padx=12, pady=24)
            tk.Label(f, text="Nothing selected", bg=THEME["panel_inner"], fg=THEME["fg"],
                     font=("TkDefaultFont", 13, "bold")).pack(anchor="w")
            tk.Label(f, text="Select a unit on the map to edit it, or place one from the toolbar:",
                     bg=THEME["panel_inner"], fg=THEME["fg_muted"], font=("TkDefaultFont", 10),
                     wraplength=300, justify="left").pack(anchor="w", pady=(6, 10))
            for icon, name in (("◆", "Defense site (interceptors)"),
                               ("⬢", "Threat launch"),
                               ("▲", "Early-warning radar")):
                row = tk.Frame(f, bg=THEME["panel_inner"]); row.pack(anchor="w", pady=1)
                tk.Label(row, text=icon, bg=THEME["panel_inner"], fg=THEME["accent"],
                         font=("TkDefaultFont", 12)).pack(side="left", padx=(0, 8))
                tk.Label(row, text=name, bg=THEME["panel_inner"], fg=THEME["fg_muted"],
                         font=("TkDefaultFont", 10)).pack(side="left")
            return

        kind_title = {"ilp": "Defense site (ILP)", "mlp": "Threat launch (MLP)", "gbewr": "Early-warning radar"}[token.kind]
        self._inspector_heading(f"{kind_title}: {token.name}", None)

        self._add_entry("Display name", token.name, "name", HELP["display_name"])
        self._add_entry("Latitude (°)", token.lat, "lat", HELP["lat_lon"])
        self._add_entry("Longitude (°)", token.lon, "lon", HELP["lat_lon"])

        if token.kind == "gbewr":
            self._inspector_heading("Radar performance", "radar_preset")
            radars = gbewr.load_gbewr_data()
            radar_names = [
                f"#{r.get('r_key')} — {r.get('type', '')} ({r.get('range_km', 0)} km)"
                for r in radars
            ]
            radar_combo = ttk.Combobox(inner, values=radar_names, state="readonly", width=40)
            idx = next((i for i, r in enumerate(radars) if r.get("r_key") == token.r_key), 0)
            radar_combo.current(idx)
            radar_combo.pack(anchor="w", padx=6, pady=2)
            self.sidebar_vars["gbewr_radar"] = radar_combo
            ToolTip(radar_combo, HELP["radar_preset"])
        elif token.kind == "ilp":
            self._add_entry("Boresight azimuth (°)", token.center_dir, "center_dir", HELP["boresight"])
            self._add_entry("Sector width (°)", token.sector_width, "sector_width", HELP["sector_w"])
            # Live-update the sector drawing and threat-origin dots while these are edited,
            # so the arc tracks the sector without needing to press Apply.
            for _fld in ("center_dir", "sector_width"):
                _v = self.sidebar_vars.get(_fld)
                if _v is not None:
                    _v.trace_add("write", lambda *a: self._on_sector_field_change())

            self._inspector_heading("Threat models for this site", "threat_types")
            missile_frame = ttk.Frame(inner)
            missile_frame.pack(fill="x", padx=6, pady=2)
            missile_list = tk.Listbox(missile_frame, selectmode="multiple", **self._sidebar_list_kw(6))
            missile_scroll = ttk.Scrollbar(missile_frame, orient="vertical", command=missile_list.yview)
            missile_list.configure(yscrollcommand=missile_scroll.set)
            missile_list.pack(side="left", fill="x", expand=True)
            missile_scroll.pack(side="right", fill="y")
            self.sidebar_vars["missile_list"] = missile_list
            self._populate_missile_list(token, missile_list)
            missile_list.bind("<<ListboxSelect>>", lambda e: self._on_missile_select())

            self._inspector_heading("Interceptor models for this site", None)
            interceptors = self._get_interceptors()
            int_labels = [f"i{item.get('i_key', '')}: {item.get('type', '')}" for item in interceptors]
            int_keys = [int(item.get("i_key", 0)) for item in interceptors]
            self.sidebar_vars["ilp_itype_keys"] = int_keys
            if int_labels:
                i_combo = ttk.Combobox(inner, values=int_labels, state="readonly", width=40)
                current_itype = int(token.config.get("itype", token.interceptor_keys[0] if token.interceptor_keys else 11))
                idx = next((i for i, k in enumerate(int_keys) if k == current_itype), 0)
                i_combo.current(idx)
                i_combo.pack(anchor="w", padx=6, pady=2)
                self.sidebar_vars["ilp_itype_combo"] = i_combo

            int_frame = ttk.Frame(inner)
            int_frame.pack(fill="x", padx=6, pady=2)
            int_list = tk.Listbox(int_frame, selectmode="multiple", **self._sidebar_list_kw(5))
            int_scroll = ttk.Scrollbar(int_frame, orient="vertical", command=int_list.yview)
            int_list.configure(yscrollcommand=int_scroll.set)
            int_list.pack(side="left", fill="x", expand=True)
            int_scroll.pack(side="right", fill="y")
            self.sidebar_vars["interceptor_list"] = int_list
            self._populate_interceptor_list(token, int_list)
        elif token.kind == "mlp":
            self._inspector_heading("Threat missile types at this launch", "threat_types")
            missile_frame = ttk.Frame(inner)
            missile_frame.pack(fill="x", padx=6, pady=2)
            missile_list = tk.Listbox(missile_frame, selectmode="multiple", **self._sidebar_list_kw(6))
            missile_scroll = ttk.Scrollbar(missile_frame, orient="vertical", command=missile_list.yview)
            missile_list.configure(yscrollcommand=missile_scroll.set)
            missile_list.pack(side="left", fill="x", expand=True)
            missile_scroll.pack(side="right", fill="y")
            self.sidebar_vars["mlp_missile_list"] = missile_list
            self._populate_missile_list(token, missile_list)
            missile_list.bind("<<ListboxSelect>>", lambda e: self._on_missile_select())

        if token.kind == "ilp":
            show_center_var = tk.BooleanVar(value=token.show_center)
            show_sector_var = tk.BooleanVar(value=token.show_sector)
            use_sectoral_var = tk.BooleanVar(value=token.use_sectoral)
            use_det_override_var = tk.BooleanVar(value=token.use_det_range_override)
            use_det_sweep_var = tk.BooleanVar(value=token.use_det_sweep)
            use_space_delay_var = tk.BooleanVar(value=token.use_space_based_delay)
            use_range_override_var = tk.BooleanVar(value=token.use_range_override)
            use_op_range_override_var = tk.BooleanVar(value=token.use_op_range_override)
            self.sidebar_vars["show_center"] = show_center_var
            self.sidebar_vars["show_sector"] = show_sector_var
            self.sidebar_vars["use_sectoral"] = use_sectoral_var
            self.sidebar_vars["use_det_range_override"] = use_det_override_var
            self.sidebar_vars["use_det_sweep"] = use_det_sweep_var
            self.sidebar_vars["use_space_based_delay"] = use_space_delay_var
            self.sidebar_vars["use_range_override"] = use_range_override_var
            self.sidebar_vars["use_op_range_override"] = use_op_range_override_var
            body = self._collapsible("Overlay & footprint options", "fp_layers", opened=False)
            fp_hint = tk.Frame(body, bg=THEME["panel_inner"])
            fp_hint.pack(fill="x", padx=6, pady=(2, 4))
            tk.Label(
                fp_hint,
                text="Tip: after running multiple footprint calculations, click the site star (no drag) to switch Calculation Mode and show/hide layers.",
                bg=THEME["panel_inner"],
                fg=THEME["fg_muted"],
                font=("TkDefaultFont", 9),
                wraplength=380,
                justify="left",
            ).pack(side="left", anchor="w")
            self._insp_check(body, "Show boresight line", show_center_var, HELP["show_center"], "show_center")
            self._insp_check(body, "Show sector edges", show_sector_var, HELP["show_sector"], "show_sector")
            self._insp_check(body, "Clip footprint to sector", use_sectoral_var, HELP["sectoral_fp"], "use_sectoral")
            self._insp_check(
                body,
                "Override detection range (fixed km)",
                use_det_override_var,
                HELP["det_override"],
                "use_det_range_override",
            )
            self._add_entry("Detection range (km)", token.det_range_override_km, "det_range_override", HELP["det_override"])
            self._insp_check(
                body,
                "Show multi-detection overlays",
                use_det_sweep_var,
                HELP["det_sweep"],
                "use_det_sweep",
            )
            self._add_entry("Detection sweep list (km)", token.det_sweep_km_csv, "det_sweep_km_csv", HELP["det_sweep"])
            self._insp_check(
                body,
                "Override with space-based detection",
                use_space_delay_var,
                HELP["space_delay"],
                "use_space_based_delay",
            )
            self._add_entry("Space-based delay (s)", token.space_based_delay_s, "space_based_delay_s", HELP["space_delay"])
            use_det_override_var.trace_add("write", lambda *_: self._on_det_mode_toggle("use_det_range_override"))
            use_space_delay_var.trace_add("write", lambda *_: self._on_det_mode_toggle("use_space_based_delay"))
            self._insp_check(
                body,
                "Override threat max range",
                use_range_override_var,
                HELP["range_override"],
                "use_range_override",
            )
            self._add_entry("Threat range override (km)", token.range_override_km, "range_override", HELP["range_override"])
            self._insp_check(
                body,
                "Override interceptor op range (footprint only)",
                use_op_range_override_var,
                HELP["op_range_override"],
                "use_op_range_override",
            )
            self._add_entry("Interceptor op range (km)", token.op_range_override_km, "op_range_override_km", HELP["op_range_override"])

            show_threat_arc_var = tk.BooleanVar(value=token.show_threat_arc)
            self.sidebar_vars["show_threat_arc"] = show_threat_arc_var
            self._insp_check(
                body,
                "Show threat-origin dots",
                show_threat_arc_var,
                HELP["threat_arc"],
                "show_threat_arc",
            )
            self._add_entry("Threat dots per sector", token.threat_arc_samples, "threat_arc_samples", HELP["threat_arc"])
            iso_row = tk.Frame(body, bg=THEME["panel_inner"])
            iso_row.pack(fill="x", padx=6, pady=(2, 4))
            iso_btn = self._lib_btn(iso_row, "Show full sector", self._reset_isolation_for_selected)
            iso_btn.pack(side="left", padx=4)
            ToolTip(iso_btn, HELP["clear_isolation"])

            body = self._collapsible("Footprint calculations", "from_mlp", opened=True)
            tk.Label(
                body,
                text="From-launch-point modes model a threat from a fixed spot (range/trajectory vary per target).",
                bg=THEME["panel_inner"], fg=THEME["fg_muted"], font=("TkDefaultFont", 9),
                wraplength=380, justify="left",
            ).pack(fill="x", padx=6, pady=(2, 4))
            fixed_mlp_loft_var = tk.BooleanVar(value=token.fixed_mlp_loft)
            self.sidebar_vars["fixed_mlp_loft"] = fixed_mlp_loft_var
            self._insp_check(body, "Lofted trajectory (vs depressed)", fixed_mlp_loft_var, HELP["from_mlp_loft"], "fixed_mlp_loft")

            # --- launch point: one stateful button drives place -> proceed -> compute ---
            mp_state = getattr(token, "_mlp_btn_state", "idle")
            mp_row = tk.Frame(body, bg=THEME["panel_inner"])
            mp_row.pack(fill="x", padx=6, pady=(4, 2))
            if mp_state == "placing":
                mb = self._lib_btn(mp_row, "● Click map to place launch point  (click here to cancel)",
                                   lambda: self._mlp_button_action(token),
                                   bg_color=THEME["armed_bg"], fg_color=THEME["armed_text"])
            elif mp_state == "placed":
                mb = self._lib_btn(mp_row, "✓ Place again (map) or proceed → compute",
                                   lambda: self._mlp_button_action(token), primary=True)
            else:
                mb = self._lib_btn(mp_row, "Compute from launch point", lambda: self._mlp_button_action(token))
            mb.pack(side="left", fill="x", expand=True, padx=4)
            ToolTip(mb, HELP["from_mlp_compute"])
            if getattr(token, "fixed_mlp_latlon", None):
                self._lib_btn(mp_row, "Clear", self._clear_fixed_mlp_for_selected).pack(side="left", padx=4)

            # --- launch box: one stateful button drives draw -> proceed -> compute ---
            bx_state = getattr(token, "_box_btn_state", "idle")
            bx_row = tk.Frame(body, bg=THEME["panel_inner"])
            bx_row.pack(fill="x", padx=6, pady=(0, 2))
            if bx_state == "drawing":
                bb = self._lib_btn(bx_row, "▭ Drag a box on the map  (click here to cancel)",
                                   lambda: self._box_button_action(token),
                                   bg_color=THEME["armed_bg"], fg_color=THEME["armed_text"])
            elif bx_state == "drawn":
                bb = self._lib_btn(bx_row, "✓ Draw again (map) or proceed → compute",
                                   lambda: self._box_button_action(token), primary=True)
            else:
                bb = self._lib_btn(bx_row, "Compute from launch box", lambda: self._box_button_action(token))
            bb.pack(side="left", fill="x", expand=True, padx=4)
            ToolTip(bb, HELP["launch_box"])
            if getattr(token, "launch_box", None):
                self._lib_btn(bx_row, "Clear box", self._clear_launch_box_for_selected).pack(side="left", padx=4)

            # the other from-launch-point computations (concise labels, detail in tooltips)
            for label, cmd, help_key in (
                ("From each max-range (boresight)", self._run_footprint_from_ranges, "from_ranges_compute"),
                ("All max distances, full sector", self._run_footprint_full_sector, "full_sector_compute"),
                ("100% interceptable", self._show_100pct_interceptable, "pct100"),
            ):
                r = tk.Frame(body, bg=THEME["panel_inner"]); r.pack(fill="x", padx=6, pady=(0, 3))
                btn = self._lib_btn(r, label, cmd)
                btn.pack(side="left", fill="x", expand=True)
                ToolTip(btn, HELP[help_key])

            # primary action — kept top-level and prominent
            self._end_section()
            fp_btn_row = tk.Frame(inner, bg=THEME["panel_inner"])
            fp_btn_row.pack(fill="x", padx=6, pady=(8, 8))
            go_b = self._lib_btn(fp_btn_row, "Compute footprint (max-range)", self._run_footprint, primary=True)
            go_b.pack(side="left", fill="x", expand=True, padx=4)
            ToolTip(go_b, HELP["go_footprint"])
            apply_b = self._lib_btn(fp_btn_row, "Apply", self._apply_sidebar)
            apply_b.pack(side="left", padx=4)
            ToolTip(apply_b, HELP["apply"])

            body = self._collapsible("Linked radars", "assign_gbewr", opened=False)
            gbewr_frame = ttk.Frame(body)
            gbewr_frame.pack(fill="x", padx=6, pady=2)
            gbewr_listbox = tk.Listbox(gbewr_frame, selectmode="multiple", **self._sidebar_list_kw(4))
            gbewr_scroll = ttk.Scrollbar(gbewr_frame, orient="vertical", command=gbewr_listbox.yview)
            gbewr_listbox.configure(yscrollcommand=gbewr_scroll.set)
            gbewr_listbox.pack(side="left", fill="x", expand=True)
            gbewr_scroll.pack(side="right", fill="y")
            self.sidebar_vars["gbewr_list"] = gbewr_listbox
            gbewr_token_ids = []
            for tid, tok in sorted(self.tokens.items()):
                if tok.kind == "gbewr":
                    spec = gbewr.get_radar_spec(tok.r_key) or {}
                    label = f"{tok.name}  ·  #{tok.r_key}  ·  {spec.get('range_km', '?')} km"
                    gbewr_listbox.insert(tk.END, label)
                    gbewr_token_ids.append(tid)
                    if tid in (token.gbewr_assigned or []):
                        gbewr_listbox.selection_set(gbewr_listbox.size() - 1)
            self.sidebar_vars["gbewr_token_ids"] = gbewr_token_ids
            gbewr_listbox.bind("<<ListboxSelect>>", lambda e: self._apply_sidebar())

        if token.kind not in ("gbewr", "mlp"):
            body = self._collapsible("Solver parameters", "footprint_cfg", opened=False)
            for key, value in token.config.items():
                if token.kind == "ilp" and key == "itype":
                    continue
                field_id = f"cfg::{key}"
                label = self._config_label(key)
                tip = self._config_tooltip(key)
                if isinstance(value, bool):
                    var = tk.BooleanVar(value=value)
                    self._insp_check(body, label, var, tip, field_id)
                else:
                    self._add_entry(label, value, field_id, tip)
            self._end_section()

        if token.kind == "ilp":
            body = self._collapsible("Analysis tools (fp_gui)", "procedures", opened=False)
            self._add_procedure_buttons(body)
            self._end_section()

            del_row = tk.Frame(inner, bg=THEME["panel_inner"])
            del_row.pack(fill="x", padx=6, pady=(10, 14))
            del_b = self._lib_btn(
                del_row,
                "Remove from map",
                self._delete_token,
                fg_color="#9f1239",
                active_fg="#4c0519",
            )
            del_b.pack(side="left", padx=4)
            ToolTip(del_b, HELP["delete_token"])
        else:
            self._inspector_sep()
            btn_row = tk.Frame(inner, bg=THEME["panel_inner"])
            btn_row.pack(fill="x", padx=6, pady=8)
            apply_b = self._lib_btn(btn_row, "Apply", self._apply_sidebar)
            apply_b.pack(side="left", padx=4)
            ToolTip(apply_b, HELP["apply"])
            if token.kind == "mlp":
                ur_b = self._lib_btn(btn_row, "Refresh range rings", self._run_mlp_ranges)
                ur_b.pack(side="left", padx=4)
                ToolTip(ur_b, HELP["update_range"])
            del_b = self._lib_btn(
                btn_row,
                "Remove from map",
                self._delete_token,
                fg_color="#9f1239",
                active_fg="#4c0519",
            )
            del_b.pack(side="left", padx=4)
            ToolTip(del_b, HELP["delete_token"])

    def _insp_check(self, parent, text, variable, tip, storage_key=None):
        row = tk.Frame(parent, bg=THEME["panel_inner"])
        row.pack(fill="x", padx=6, pady=2)
        cb = ttk.Checkbutton(row, text=text, variable=variable, command=self._apply_sidebar)
        cb.pack(side="left")
        if tip:
            _help_icon(row, tip).pack(side="left", padx=4)
        if storage_key is not None:
            self.sidebar_vars[storage_key] = variable

    def _add_entry(self, label, value, field_id, tip=None):
        frame = tk.Frame(self._insp_container(), bg=THEME["panel_inner"])
        frame.pack(fill="x", padx=6, pady=3)
        tk.Label(
            frame,
            text=label,
            width=22,
            anchor="w",
            bg=THEME["panel_inner"],
            fg=THEME["fg"],
            font=("TkDefaultFont", 10),
        ).pack(side="left")
        if tip:
            _help_icon(frame, tip).pack(side="left", padx=(0, 6))
        var = tk.StringVar(value=str(value))
        ent = ttk.Entry(frame, textvariable=var, width=32)
        ent.pack(side="left", fill="x", expand=True)
        self.sidebar_vars[field_id] = var

    def _on_det_mode_toggle(self, source_key):
        """Keep detection mode toggles mutually exclusive."""
        if self.selected_token is None or self.selected_token.kind != "ilp":
            return
        det_var = self.sidebar_vars.get("use_det_range_override")
        space_var = self.sidebar_vars.get("use_space_based_delay")
        if det_var is None or space_var is None:
            return
        if source_key == "use_det_range_override" and det_var.get() and space_var.get():
            space_var.set(False)
        elif source_key == "use_space_based_delay" and space_var.get() and det_var.get():
            det_var.set(False)

    def _set_add_mode(self, mode):
        if self.add_mode == mode:
            self.add_mode = None
        else:
            self.add_mode = mode
            self.coverage_mode = None
            self.distance_mode = None
        ilp_text = "Click map to place…" if self.add_mode == "ilp" else "Add defense site"
        mlp_text = "Click map to place…" if self.add_mode == "mlp" else "Add threat launch"
        gbewr_text = "Click map to place…" if self.add_mode == "gbewr" else "Add radar"
        self.add_ilp_btn.configure(text=ilp_text)
        self.add_mlp_btn.configure(text=mlp_text)
        self.add_gbewr_btn.configure(text=gbewr_text)

    def _hit_test_map_token(self, x, y):
        """Return (token_id, drag_tag) for topmost drawable under (x,y), or (None, None)."""
        items = self.map_canvas.find_overlapping(x - 3, y - 3, x + 3, y + 3)
        for item in reversed(items):
            tags = self.map_canvas.gettags(item)
            if "base" in tags and len(tags) == 1:
                continue
            token_id = self._token_id_from_tags(tags)
            if token_id is not None and token_id in self.tokens:
                return token_id, self._drag_tag_from_tags(tags)
        return None, None

    def _select_token_at_press(self, event):
        """Selection logic for a click (used on press for tokens, on release for empty map)."""
        token_id, drag_tag = self._hit_test_map_token(event.x, event.y)
        if token_id is None:
            self._select_token(None)
            return
        self._select_token(self.tokens[token_id])
        self.drag_state = {"token_id": token_id, "tag": drag_tag}

    def _on_map_button1_press(self, event):
        self._left_pan_armed = False
        self._left_panning = False
        self._left_pan_last = None
        self._b1_press_xy = (event.x, event.y)
        self._token_press_moved = False

        if self.drawing_launch_box_for is not None:
            self._launch_box_start = (event.x, event.y)
            if self._launch_box_rubber is not None:
                self.map_canvas.delete(self._launch_box_rubber)
            self._launch_box_rubber = self.map_canvas.create_rectangle(
                event.x, event.y, event.x, event.y,
                outline="#7c3aed", width=2, dash=(5, 4),
            )
            return

        if self.placing_fixed_mlp_for is not None:
            tid = self.placing_fixed_mlp_for
            if tid in self.tokens:
                lat, lon = self._xy_to_latlon(event.x, event.y)
                tok = self.tokens[tid]
                tok._mlp_btn_state = "placed"  # button becomes "place again or proceed"
                self._select_token(tok)        # rebuild sidebar in placed state
                self._set_fixed_mlp(tok, lat, lon)
                self.status_var.set("Launch point placed — click map to replace, or press Proceed")
                # stay armed so further map clicks re-place the point
            else:
                self.placing_fixed_mlp_for = None
                self.map_canvas.configure(cursor="")
            return

        if self.add_mode:
            lat, lon = self._xy_to_latlon(event.x, event.y)
            if self.add_mode == "ilp":
                self._create_token(lat, lon)
            elif self.add_mode == "mlp":
                self._create_mlp_token(lat, lon)
            elif self.add_mode == "gbewr":
                self._create_gbewr_token(lat, lon)
            self.add_mode = None
            self.add_ilp_btn.configure(text="Add defense site")
            self.add_mlp_btn.configure(text="Add threat launch")
            self.add_gbewr_btn.configure(text="Add radar")
            return

        if self.distance_mode:
            self._handle_distance_click(event)
            return

        if self.coverage_mode:
            self._handle_coverage_click(event)
            return

        dot_token, dot = self._hit_test_threat_dot(event.x, event.y)
        if dot is not None:
            self._isolate_threat_dot(dot_token, dot)
            return

        token_id, drag_tag = self._hit_test_map_token(event.x, event.y)
        if token_id is not None:
            self._select_token(self.tokens[token_id])
            self.drag_state = {"token_id": token_id, "tag": drag_tag}
            self.map_canvas.configure(cursor="hand2")
            return

        self._left_pan_armed = True
        self._left_pan_last = (event.x, event.y)

    def _on_map_drag(self, event):
        if self.drawing_launch_box_for is not None and self._launch_box_start is not None:
            sx, sy = self._launch_box_start
            self.map_canvas.coords(self._launch_box_rubber, sx, sy, event.x, event.y)
            return
        if self._b1_press_xy is not None:
            px, py = self._b1_press_xy
            if math.hypot(event.x - px, event.y - py) > 5:
                self._token_press_moved = True
        if self.drag_state:
            token = self.tokens[self.drag_state["token_id"]]
            tag = self.drag_state["tag"]
            if tag == "token":
                lat, lon = self._xy_to_latlon(event.x, event.y)
                token.lat = clamp(lat, -85, 85)
                token.lon = ((lon + 180) % 360) - 180
                self._update_token_draw(token)
                self._refresh_sidebar_token(token)
            elif tag == "dir_handle":
                token.center_dir = self._bearing_from_token(token, event.x, event.y)
                self._update_token_draw(token)
                self._refresh_sidebar_token(token)
            elif tag in ("left_handle", "right_handle"):
                angle = self._bearing_from_token(token, event.x, event.y)
                delta = self._angle_diff(angle, token.center_dir)
                token.sector_width = clamp(abs(delta) * 2.0, 5.0, 180.0)
                self._update_token_draw(token)
                self._refresh_sidebar_token(token)
            elif tag == "fixed_mlp":
                lat, lon = self._xy_to_latlon(event.x, event.y)
                self._set_fixed_mlp(token, lat, lon)
            return

        if not self._left_pan_armed or self._b1_press_xy is None:
            return

        px, py = self._b1_press_xy
        moved = math.hypot(event.x - px, event.y - py)
        if moved > 4:
            self._left_panning = True
            self.map_canvas.configure(cursor="fleur")

        if not self._left_panning or self._left_pan_last is None:
            return

        lx, ly = self._left_pan_last
        dx = event.x - lx
        dy = event.y - ly
        self._left_pan_last = (event.x, event.y)
        self._pan_by_pixels(dx, dy, debounce=True)

    def _on_map_button1_release(self, event):
        self.map_canvas.configure(cursor="")

        if self.drawing_launch_box_for is not None:
            tid = self.drawing_launch_box_for
            start = self._launch_box_start
            self._launch_box_start = None
            if self._launch_box_rubber is not None:
                self.map_canvas.delete(self._launch_box_rubber)
                self._launch_box_rubber = None
            self._b1_press_xy = None
            if start is None or tid not in self.tokens:
                self.drawing_launch_box_for = None
                return
            if math.hypot(event.x - start[0], event.y - start[1]) < 6:
                self.status_var.set("Launch box too small — drag a larger rectangle")
                return  # stay armed to try again
            lat1, lon1 = self._xy_to_latlon(*start)
            lat2, lon2 = self._xy_to_latlon(event.x, event.y)
            tok = self.tokens[tid]
            prev_firing = (getattr(tok, "launch_box", None) or {}).get("missile_keys")
            tok._box_btn_state = "drawn"  # button becomes "draw again or proceed"
            self._select_token(tok)        # rebuild sidebar in drawn state
            tok.launch_box = {
                "lat1": lat1, "lon1": lon1, "lat2": lat2, "lon2": lon2,
                "missile_keys": prev_firing or list(tok.missile_keys),
                "grid_n": (getattr(tok, "launch_box", None) or {}).get("grid_n", 3),
            }
            self._update_token_draw(tok)
            self._show_launch_box_popup(tok)
            self.status_var.set("Box drawn — drag again to redraw, choose firing missiles, then Proceed")
            # stay armed so a further drag redraws the box
            return

        if self.drag_state:
            tid = self.drag_state["token_id"]
            tag = self.drag_state["tag"]
            moved = self._token_press_moved
            self.drag_state = None
            if (
                not moved
                and tag == "token"
                and tid in self.tokens
            ):
                tok = self.tokens[tid]
                if tok.kind == "ilp" and self._ilp_has_footprint_popup(tok):
                    self._show_footprint_layer_popup(tok)
            elif moved and tag in ("dir_handle", "left_handle", "right_handle") and tid in self.tokens:
                # Sector/boresight changed: recompute the footprint for the new sector.
                tok = self.tokens[tid]
                if tok.kind == "ilp":
                    self._refresh_sidebar_token(tok)
                    tok.isolated_azimuth = None
                    tok.isolated_missile_key = None
                    if tok.footprints:
                        self.selected_token = tok
                        self._run_footprint()
                    else:
                        self._update_token_draw(tok)
            return

        if self._left_pan_armed:
            if self._left_panning:
                self._cancel_fast_map_render()
                self._draw_base_map()
            else:
                fake = type("E", (), {"x": self._b1_press_xy[0], "y": self._b1_press_xy[1]})()
                self._select_token_at_press(fake)

        self._left_pan_armed = False
        self._left_panning = False
        self._left_pan_last = None
        self._b1_press_xy = None

    def _on_pan_start(self, event):
        self._cancel_fast_map_render()
        self.pan_state = {"last_x": event.x, "last_y": event.y}

    def _on_pan_drag(self, event):
        if not hasattr(self, "pan_state") or self.pan_state is None:
            return
        dx = event.x - self.pan_state["last_x"]
        dy = event.y - self.pan_state["last_y"]
        self.pan_state["last_x"] = event.x
        self.pan_state["last_y"] = event.y
        self._pan_by_pixels(dx, dy, debounce=True)

    def _on_pan_end(self, event):
        self.pan_state = None
        self._cancel_fast_map_render()
        self._draw_base_map()

    def _cancel_fast_map_render(self):
        if self.map_fast_render_job is not None:
            self.root.after_cancel(self.map_fast_render_job)
            self.map_fast_render_job = None

    def _schedule_fast_map_render(self):
        self._cancel_fast_map_render()

        def _do():
            self.map_fast_render_job = None
            self._draw_base_map()

        self.map_fast_render_job = self.root.after(20, _do)

    def _pan_by_pixels(self, dx, dy, debounce=False):
        w = self.map_canvas.winfo_width() or 1000
        h = self.map_canvas.winfo_height() or 600
        z, world_px, left, top, view_w_px, view_h_px = self._view_world_bounds()
        lon_per_px = view_w_px / w
        lat_per_px = view_h_px / h
        center_x, center_y = self._latlon_to_world_px(self.view_center_lat, self.view_center_lon, z)
        # Invert dx: drag right (dx>0) moves map left on screen (grab-the-map behavior)
        center_x -= dx * lon_per_px
        # Invert dy: drag up (dy<0) moves map down on screen (grab-the-map / Google Maps style)
        center_y -= dy * lat_per_px
        center_lon, center_lat = self._world_px_to_latlon(center_x, center_y, z)
        self._set_view_center(center_lat, center_lon)
        if debounce:
            self._schedule_fast_map_render()
        else:
            self._draw_base_map()

    def _on_canvas_resize(self, event):
        self._schedule_map_render()
        for token in self.tokens.values():
            self._update_token_draw(token)

    def _on_mousewheel(self, event):
        # Normalize wheel direction across OSes; on macOS/aqua Tk delta sign is inverted
        # relative to the Linux/Windows behavior this map was written for.
        direction = 0
        if hasattr(event, "num") and event.num in (4, 5):
            direction = 1 if event.num == 4 else -1
        elif hasattr(event, "delta") and event.delta:
            direction = 1 if event.delta > 0 else -1
            if self._is_macos:
                direction *= -1

        if direction == 0:
            return

        factor = 1.1 if direction > 0 else 0.9

        before_lat, before_lon = self._xy_to_latlon(event.x, event.y)
        self.zoom = clamp(self.zoom * factor, MIN_ZOOM, MAX_ZOOM)
        self._sync_zoom_slider()
        self._recenter_on_cursor(before_lat, before_lon, event.x, event.y)
        self._cancel_fast_map_render()
        self._schedule_fast_map_render()

    def _sync_zoom_slider(self):
        self._zoom_slider_updating = True
        self.zoom_var.set(self.zoom)
        self._zoom_slider_updating = False

    def _on_zoom_slider_drag(self, value):
        if self._zoom_slider_updating:
            return
        try:
            target_zoom = float(value)
        except (TypeError, ValueError):
            return
        target_zoom = clamp(target_zoom, MIN_ZOOM, MAX_ZOOM)
        if abs(target_zoom - self.zoom) < 1e-9:
            return
        self.zoom = target_zoom
        self._cancel_fast_map_render()
        self._schedule_fast_map_render()

    def _recenter_on_cursor(self, lat, lon, x, y):
        w = self.map_canvas.winfo_width() or 1000
        h = self.map_canvas.winfo_height() or 600
        z, world_px, left, top, view_w_px, view_h_px = self._view_world_bounds()
        world_x, world_y = self._latlon_to_world_px(lat, lon, z)
        left = world_x - (x / w) * view_w_px
        top = world_y - (y / h) * view_h_px
        center_x = left + view_w_px / 2.0
        center_y = top + view_h_px / 2.0
        center_lon, center_lat = self._world_px_to_latlon(center_x, center_y, z)
        self._set_view_center(center_lat, center_lon)

    def _token_id_from_tags(self, tags):
        for tag in tags:
            if tag.startswith("token:"):
                return int(tag.split(":")[1])
        return None

    def _drag_tag_from_tags(self, tags):
        for tag in ("dir_handle", "left_handle", "right_handle", "fixed_mlp", "token"):
            if tag in tags:
                return tag
        return "token"

    def _create_token(self, lat, lon):
        token_id = max(self.tokens.keys(), default=0) + 1
        token = Token(token_id, f"Defense site {token_id}", lat, lon, self.base_config, kind="ilp")
        self.tokens[token_id] = token
        self._draw_token(token)
        self._select_token(token)

    def _create_mlp_token(self, lat, lon):
        token_id = max(self.tokens.keys(), default=0) + 1
        token = Token(token_id, f"Threat launch {token_id}", lat, lon, self.base_config, kind="mlp")
        token.missile_keys = [int(self.base_config.get("mtype", 1))]
        self.tokens[token_id] = token
        self._draw_token(token)
        self._select_token(token)

    def _create_gbewr_token(self, lat, lon):
        radars = gbewr.load_gbewr_data()
        if not radars:
            self.status_var.set("No GBEWR specs in gbewr_data.json")
            return
        r_key = radars[0].get("r_key", 1)
        token_id = max(self.tokens.keys(), default=0) + 1
        token = Token(token_id, f"Radar site {token_id}", lat, lon, self.base_config, kind="gbewr")
        token.r_key = r_key
        self.tokens[token_id] = token
        self._draw_token(token)
        self._select_token(token)

    def _select_token(self, token):
        if token is not self.selected_token:
            self._destroy_footprint_layer_popup()
        self.selected_token = token
        self._build_sidebar(token)
        for tok in self.tokens.values():
            self._set_token_highlight(tok, tok == token)

    def _draw_token(self, token):
        x, y = self._latlon_to_xy(token.lat, token.lon)
        token_tag = f"token:{token.token_id}"
        if token.kind == "gbewr":
            coords = self._radar_xy(x, y, 9)
            token.canvas_items["marker"] = self.map_canvas.create_polygon(
                *coords,
                fill="#1e3a5f",
                outline="#e2e8f0",
                width=2,
                joinstyle="round",
                tags=(token_tag, "token"),
            )
        elif token.kind == "mlp":
            coords = self._hex_xy(x, y, 8)
            token.canvas_items["marker"] = self.map_canvas.create_polygon(
                *coords,
                fill="#b45309",
                outline="#fff7ed",
                width=2,
                joinstyle="round",
                tags=(token_tag, "token"),
            )
        else:
            coords = self._star_xy(x, y, 11, 4.5, 5)
            token.canvas_items["marker"] = self.map_canvas.create_polygon(
                *coords,
                fill="#0d9488",
                outline="#ecfdf5",
                width=2,
                joinstyle="round",
                tags=(token_tag, "token"),
            )
        token.canvas_items["label"] = self.map_canvas.create_text(
            x + 12,
            y - 12,
            text=token.name,
            anchor="w",
            fill="#0f172a",
            font=("Helvetica", 9, "bold"),
            tags=(token_tag,),
        )
        self._update_token_draw(token)

    def _update_token_draw(self, token):
        x, y = self._latlon_to_xy(token.lat, token.lon)
        if token.kind == "gbewr":
            coords = self._radar_xy(x, y, 9)
        elif token.kind == "mlp":
            coords = self._hex_xy(x, y, 8)
        else:
            coords = self._star_xy(x, y, 11, 4.5, 5)
        self.map_canvas.coords(token.canvas_items["marker"], *coords)
        self.map_canvas.coords(token.canvas_items["label"], x + 12, y - 12)
        self.map_canvas.itemconfigure(token.canvas_items["label"], text=token.name)

        if token.kind == "gbewr":
            return
        if token.kind == "mlp":
            self._draw_mlp_ranges(token)
            return

        center_line_len = 60
        left_angle = token.center_dir - token.sector_width / 2.0
        right_angle = token.center_dir + token.sector_width / 2.0

        center_xy = self._point_from_bearing(x, y, token.center_dir, center_line_len)
        left_xy = self._point_from_bearing(x, y, left_angle, center_line_len)
        right_xy = self._point_from_bearing(x, y, right_angle, center_line_len)

        self._update_line_item(token, "center_line", x, y, *center_xy, token.show_center, "green")
        self._update_line_item(token, "left_line", x, y, *left_xy, token.show_sector, "gray")
        self._update_line_item(token, "right_line", x, y, *right_xy, token.show_sector, "gray")

        self._update_handle(token, "dir_handle", center_xy, token.show_center, "dir_handle")
        self._update_handle(token, "left_handle", left_xy, token.show_sector, "left_handle")
        self._update_handle(token, "right_handle", right_xy, token.show_sector, "right_handle")

        # Draw each overlay independently: a failure rendering one (e.g. a complex
        # footprint polygon) must not abort the others, or the threat-origin dots would
        # silently freeze (Tk swallows the exception) and stop tracking the sector.
        for _draw in (self._draw_footprint, self._draw_threat_arc, self._draw_fixed_mlp, self._draw_launch_box, self._draw_calc_mode_box):
            try:
                _draw(token)
            except Exception as exc:
                self._safe_status_update(f"{_draw.__name__} error: {exc}")

    def _update_line_item(self, token, key, x1, y1, x2, y2, show, color):
        token_tag = f"token:{token.token_id}"
        if key not in token.canvas_items:
            token.canvas_items[key] = self.map_canvas.create_line(
                x1,
                y1,
                x2,
                y2,
                fill=color,
                width=3,
                capstyle=tk.ROUND,
                tags=(token_tag,),
            )
        else:
            self.map_canvas.coords(token.canvas_items[key], x1, y1, x2, y2)
        state = "normal" if show else "hidden"
        self.map_canvas.itemconfigure(token.canvas_items[key], state=state)

    def _update_handle(self, token, key, xy, show, handle_tag):
        token_tag = f"token:{token.token_id}"
        r = 6
        x, y = xy
        d = self._diamond_xy(x, y, r)
        if key not in token.canvas_items:
            token.canvas_items[key] = self.map_canvas.create_polygon(
                *d,
                fill="#f8fafc",
                outline="#0f172a",
                width=2,
                tags=(token_tag, handle_tag),
            )
        else:
            self.map_canvas.coords(token.canvas_items[key], *d)
        state = "normal" if show else "hidden"
        self.map_canvas.itemconfigure(token.canvas_items[key], state=state)

    def _unwrap_longitude(self, lon, ref_lon):
        """Unwrap lon to be near ref_lon (view center). Returns lon in [ref_lon-180, ref_lon+180]."""
        while lon < ref_lon - 180:
            lon += 360
        while lon > ref_lon + 180:
            lon -= 360
        return lon

    def _split_latlon_at_dateline(self, points, view_center_lon=None):
        """
        Split a closed polygon (list of (lat, lon)) into segments that don't cross
        wrap boundaries. Uses view center to unwrap longitudes so polygons display
        correctly regardless of map pan/zoom. Returns list of segments; all output
        lons are in [-180, 180] so downstream _latlon_to_xy works correctly.
        """
        if len(points) < 3:
            return [points] if points else []
        ref = view_center_lon if view_center_lon is not None else self.view_center_lon
        unwrapped = [(lat, self._unwrap_longitude(lon, ref)) for lat, lon in points]
        # Normalize to [-180, 180] so _latlon_to_world_px gets valid input
        work = [(lat, ((lon + 180) % 360) - 180) for lat, lon in unwrapped]
        n = len(work)
        segments = []
        current = [work[0]]
        for i in range(1, n + 1):
            lat1, lon1 = work[i - 1]
            lat2, lon2 = work[i % n]
            dlon = lon2 - lon1
            if abs(dlon) > 180:
                lon_edge = 180.0 if dlon > 0 else -180.0
                if dlon > 0:
                    t = (180.0 - lon1) / (360.0 - lon1 + lon2) if (360.0 - lon1 + lon2) > 1e-10 else 0.5
                else:
                    t = (lon1 + 180.0) / (360.0 + lon1 - lon2) if (360.0 + lon1 - lon2) > 1e-10 else 0.5
                t = max(0, min(1, t))
                lat_edge = lat1 + t * (lat2 - lat1)
                current.append((lat_edge, lon_edge))
                if len(current) >= 3:
                    segments.append(current)
                current = [(lat_edge, lon_edge), (lat2, lon2)]
            else:
                if i < n:
                    current.append((lat2, lon2))
        if len(current) >= 3:
            segments.append(current)
        return segments if segments else [points]

    def _sync_footprint_ids_and_visibility(self, token, reset_visibility=False):
        """Assign stable fp_id to each footprint entry; optionally reset layer visibility."""
        fps = token.footprints or []
        for i, fp in enumerate(fps):
            fp["fp_id"] = i
        if reset_visibility:
            token.footprint_visibility = {i: True for i in range(len(fps))}
        else:
            prev = getattr(token, "footprint_visibility", None) or {}
            token.footprint_visibility = {i: bool(prev.get(i, True)) for i in range(len(fps))}
        self._persist_active_footprint_state(token)

    def _iter_footprint_calc_modes(self, token):
        calcs = getattr(token, "footprint_calculations", None) or {}
        ordered = [m for m in FOOTPRINT_CALC_MODES if m in calcs]
        for mode_id in calcs:
            if mode_id not in ordered:
                ordered.append(mode_id)
        return [(mode_id, calcs[mode_id]) for mode_id in ordered]

    def _ilp_has_footprint_popup(self, token):
        calcs = getattr(token, "footprint_calculations", None) or {}
        if len(calcs) > 1:
            return True
        return len(token.footprints or []) > 1

    def _apply_active_footprint_mode(self, token):
        mode_id = getattr(token, "active_footprint_mode", None)
        calcs = getattr(token, "footprint_calculations", None) or {}
        if mode_id and mode_id in calcs:
            entry = calcs[mode_id]
            token.footprints = entry.get("footprints") or []
            token.footprint_visibility = dict(entry.get("footprint_visibility") or {})
            token.from_mlp_mode = bool(entry.get("from_mlp_mode", False))
            token.full_sector_mode = bool(entry.get("full_sector_mode", False))
        elif not calcs:
            token.footprints = []
            token.footprint_visibility = {}
            token.from_mlp_mode = False
            token.full_sector_mode = False

    def _persist_active_footprint_state(self, token):
        mode_id = getattr(token, "active_footprint_mode", None)
        calcs = getattr(token, "footprint_calculations", None) or {}
        if not mode_id or mode_id not in calcs:
            return
        entry = calcs[mode_id]
        entry["footprints"] = token.footprints
        entry["footprint_visibility"] = dict(token.footprint_visibility)
        entry["from_mlp_mode"] = bool(token.from_mlp_mode)
        entry["full_sector_mode"] = bool(token.full_sector_mode)

    def _store_footprint_calculation(self, token, mode_id, footprints, *, from_mlp_mode=False, full_sector_mode=False, label=None):
        if not hasattr(token, "footprint_calculations") or token.footprint_calculations is None:
            token.footprint_calculations = {}
        entry_label = label or FOOTPRINT_CALC_MODES.get(mode_id, mode_id.replace("_", " ").title())
        token.footprint_calculations[mode_id] = {
            "label": entry_label,
            "det_label": self._footprint_det_label(token),
            "footprints": footprints,
            "footprint_visibility": {i: True for i in range(len(footprints))},
            "from_mlp_mode": bool(from_mlp_mode),
            "full_sector_mode": bool(full_sector_mode),
        }
        token.active_footprint_mode = mode_id
        self._apply_active_footprint_mode(token)

    def _switch_footprint_mode(self, token, mode_id):
        calcs = getattr(token, "footprint_calculations", None) or {}
        if mode_id not in calcs:
            return
        self._persist_active_footprint_state(token)
        token.active_footprint_mode = mode_id
        self._apply_active_footprint_mode(token)
        self._sync_footprint_ids_and_visibility(token, reset_visibility=False)
        self._update_token_draw(token)

    def _footprint_det_label(self, token):
        """Short label describing which detection cueing mode is active for footprints."""
        if token.use_det_range_override and token.det_range_override_km > 0:
            return f"Fixed det {token.det_range_override_km:g}km"
        if token.use_space_based_delay:
            return "Space-based"
        return "Native det"

    def _on_calc_mode_box_click(self, token, mode_id):
        self._switch_footprint_mode(token, mode_id)  # also redraws (rebuilds this box)

    def _draw_calc_mode_box(self, token):
        """Constant little box on the map, just north of the ILP, to switch calculation mode."""
        win_key = "calc_mode_win"
        modes = self._iter_footprint_calc_modes(token) if token.kind == "ilp" else []

        def _remove():
            if win_key in token.canvas_items:
                try:
                    self.map_canvas.delete(token.canvas_items[win_key])
                except tk.TclError:
                    pass
                del token.canvas_items[win_key]
            fr = getattr(token, "_calc_mode_frame", None)
            if fr is not None:
                try:
                    fr.destroy()
                except tk.TclError:
                    pass
            token._calc_mode_frame = None
            token._calc_mode_sig = None

        if len(modes) < 2:
            _remove()
            return

        active = getattr(token, "active_footprint_mode", None) or modes[0][0]
        sig = (tuple(m for m, _ in modes), active)
        x, y = self._latlon_to_xy(token.lat, token.lon)
        off = getattr(token, "_calc_mode_offset", None) or (0, -20)
        px, py = x + off[0], y + off[1]

        if getattr(token, "_calc_mode_sig", None) != sig or win_key not in token.canvas_items:
            _remove()
            fr = tk.Frame(self.map_canvas, bg=THEME["titlebar"],
                          highlightthickness=1, highlightbackground=THEME["accent"])
            grip = tk.Label(fr, text="⠿ Calculation mode", bg=THEME["titlebar"], fg=THEME["fg_muted"],
                            font=("TkDefaultFont", 8, "bold"), cursor="fleur")
            grip.pack(fill="x", padx=6, pady=(3, 1))
            grip.bind("<ButtonPress-1>", lambda e, t=token: self._calc_box_drag_start(t, e))
            grip.bind("<B1-Motion>", lambda e, t=token: self._calc_box_drag_move(t, e))
            for mode_id, entry in modes:
                is_active = (mode_id == active)
                n = len(entry.get("footprints") or [])
                det_note = entry.get("det_label") or ""
                label = (("● " if is_active else "○ ") + str(entry.get("label", mode_id))
                         + (f"  ({n})" if n else ""))
                if det_note:
                    label += f" · {det_note}"
                btn = tk.Button(
                    fr, text=label,
                    command=lambda mid=mode_id: self._on_calc_mode_box_click(token, mid),
                    bg=(THEME["accent"] if is_active else THEME["tool_btn_bg"]),
                    fg=("#ffffff" if is_active else THEME["tool_btn_fg"]),
                    activebackground=THEME["accent_dim"], activeforeground="#ffffff",
                    relief="flat", borderwidth=0, padx=8, pady=2, cursor="hand2",
                    anchor="w", font=("TkDefaultFont", 8, "bold"), highlightthickness=0,
                )
                btn.pack(fill="x", padx=4, pady=1)
            tk.Frame(fr, bg=THEME["titlebar"], height=3).pack()
            token.canvas_items[win_key] = self.map_canvas.create_window(
                px, py, window=fr, anchor="s", tags=(f"token:{token.token_id}",))
            token._calc_mode_frame = fr
            token._calc_mode_sig = sig
        else:
            self.map_canvas.coords(token.canvas_items[win_key], px, py)
        try:
            self.map_canvas.tag_raise(token.canvas_items[win_key])
        except tk.TclError:
            pass

    def _calc_box_drag_start(self, token, event):
        win = token.canvas_items.get("calc_mode_win")
        if win is None:
            return
        coords = self.map_canvas.coords(win)
        mx = event.x_root - self.map_canvas.winfo_rootx()
        my = event.y_root - self.map_canvas.winfo_rooty()
        self._calc_box_drag = {"dx": coords[0] - mx, "dy": coords[1] - my}

    def _calc_box_drag_move(self, token, event):
        win = token.canvas_items.get("calc_mode_win")
        if win is None or not hasattr(self, "_calc_box_drag"):
            return
        mx = event.x_root - self.map_canvas.winfo_rootx()
        my = event.y_root - self.map_canvas.winfo_rooty()
        nx = mx + self._calc_box_drag["dx"]
        ny = my + self._calc_box_drag["dy"]
        ilp_x, ilp_y = self._latlon_to_xy(token.lat, token.lon)
        token._calc_mode_offset = (nx - ilp_x, ny - ilp_y)  # follows the ILP on pan/zoom
        self.map_canvas.coords(win, nx, ny)

    def _fp_layer_visible(self, token, fp_id):
        return token.footprint_visibility.get(fp_id, True)

    def _draw_footprint(self, token):
        existing_keys = {k for k in token.canvas_items if k.startswith("footprint:")}
        active_keys = set()
        label_keys = {k for k in token.canvas_items if k.startswith("footprint_label:")}

        for fp_idx, fp in enumerate(token.footprints):
            fp_id = int(fp.get("fp_id", fp_idx))
            vis = self._fp_layer_visible(token, fp_id)
            latlon_points = []
            # A footprint may carry its own boresight (full-sector layers / already-rotated
            # intersection regions); otherwise it follows the token's effective direction.
            fp_bs = fp.get("boresight")
            theta = math.radians(fp_bs if fp_bs is not None else self._effective_center_dir(token))
            for x_local, y_local in fp["points"]:
                east = x_local * math.cos(theta) + y_local * math.sin(theta)
                north = -x_local * math.sin(theta) + y_local * math.cos(theta)
                lat, lon = self._offset_latlon(token.lat, token.lon, east, north)
                latlon_points.append((lat, lon))
            segments = self._split_latlon_at_dateline(latlon_points)

            token_tag = f"token:{token.token_id}"
            for seg_idx, seg in enumerate(segments):
                points = []
                for lat, lon in seg:
                    px, py = self._latlon_to_xy(lat, lon)
                    points.extend([px, py])
                key = f"footprint:{fp_id}:{seg_idx}"
                active_keys.add(key)
                poly_state = "normal" if vis else "hidden"
                if key not in token.canvas_items:
                    token.canvas_items[key] = self.map_canvas.create_polygon(
                        points,
                        outline=fp["color"],
                        fill="",
                        width=2,
                        dash=(4, 3),
                        joinstyle="round",
                        state=poly_state,
                        tags=(token_tag,),
                    )
                else:
                    self.map_canvas.coords(token.canvas_items[key], *points)
                    self.map_canvas.itemconfigure(
                        token.canvas_items[key],
                        state=poly_state,
                        outline=fp["color"],
                    )

            label_key = f"footprint_label:{fp_id}"
            label_keys.discard(label_key)
            display_label = fp.get("label") or self._missile_label(fp["missile_key"])
            lbl_state = "normal" if vis else "hidden"
            if points:
                label_x = points[0]
                label_y = points[1]
                if label_key not in token.canvas_items:
                    token.canvas_items[label_key] = self.map_canvas.create_text(
                        label_x + 6,
                        label_y + 6,
                        text=display_label,
                        anchor="nw",
                        fill=fp["color"],
                        font=("Helvetica", 8),
                        state=lbl_state,
                        tags=(token_tag,)
                    )
                else:
                    self.map_canvas.coords(token.canvas_items[label_key], label_x + 6, label_y + 6)
                    self.map_canvas.itemconfigure(
                        token.canvas_items[label_key],
                        text=display_label,
                        fill=fp["color"],
                        state=lbl_state,
                    )

        for key in existing_keys - active_keys:
            self.map_canvas.delete(token.canvas_items[key])
            del token.canvas_items[key]

        for key in label_keys:
            self.map_canvas.delete(token.canvas_items[key])
            del token.canvas_items[key]

    # ------------------------------------------------------------------ #
    # Threat-origin sampling arc                                          #
    # ------------------------------------------------------------------ #
    def _effective_center_dir(self, token):
        """Azimuth used to orient the footprint: the isolated launch point if one
        is selected, otherwise the sector boresight."""
        if getattr(token, "isolated_azimuth", None) is not None:
            return token.isolated_azimuth
        return token.center_dir

    def _cached_missile_range_km(self, token, m_key):
        """Threat missile range in km, honoring a per-site range override, cached
        so dragging/panning does not re-run the trajectory sim every frame."""
        if getattr(token, "use_range_override", False) and getattr(token, "range_override_km", 0) > 0:
            return float(token.range_override_km)
        if m_key in self._missile_range_cache:
            return self._missile_range_cache[m_key]
        range_km = self._get_missile_range_km(m_key)
        if range_km is not None:
            self._missile_range_cache[m_key] = range_km
        return range_km

    def _threat_arc_azimuths(self, token):
        """Sampled azimuths (deg) spanning the sector, centered on the boresight."""
        n = max(2, int(getattr(token, "threat_arc_samples", 9)))
        half = token.sector_width / 2.0
        start = token.center_dir - half
        step = token.sector_width / (n - 1)
        return [(start + step * i) % 360.0 for i in range(n)]

    def _draw_threat_arc(self, token):
        """Draw the curved sampling line of threat launch points as dots."""
        dot_keys = {k for k in token.canvas_items if k.startswith("threat_dot:")}
        line_keys = {k for k in token.canvas_items if k.startswith("threat_arc:")}
        token.threat_arc_dots = []

        if token.kind != "ilp" or not getattr(token, "show_threat_arc", True):
            for key in dot_keys | line_keys:
                self.map_canvas.delete(token.canvas_items[key])
                del token.canvas_items[key]
            return

        token_tag = f"token:{token.token_id}"
        azimuths = self._threat_arc_azimuths(token)
        iso_az = getattr(token, "isolated_azimuth", None)
        iso_mk = getattr(token, "isolated_missile_key", None)
        active_dot = set()
        active_line = set()

        for m_idx, m_key in enumerate(token.missile_keys):
            range_km = self._cached_missile_range_km(token, m_key)
            if not range_km:
                continue
            arc_latlon = []
            for a_idx, az in enumerate(azimuths):
                lat, lon = self._destination_latlon(token.lat, token.lon, az, range_km)
                arc_latlon.append((lat, lon))
                px, py = self._latlon_to_xy(lat, lon)
                is_iso = (iso_az is not None
                          and iso_mk == m_key
                          and abs(self._angle_diff(az, iso_az)) < 0.5)
                r = 6 if is_iso else 4
                outline = "#0f172a"
                fill = "#f97316" if is_iso else "#fbbf24"
                width = 3 if is_iso else 1
                key = f"threat_dot:{m_key}:{a_idx}"
                active_dot.add(key)
                if key not in token.canvas_items:
                    token.canvas_items[key] = self.map_canvas.create_oval(
                        px - r, py - r, px + r, py + r,
                        fill=fill, outline=outline, width=width,
                        tags=(token_tag, "threat_dot"),
                    )
                else:
                    self.map_canvas.coords(token.canvas_items[key], px - r, py - r, px + r, py + r)
                    self.map_canvas.itemconfigure(
                        token.canvas_items[key], fill=fill, outline=outline, width=width, state="normal"
                    )
                token.threat_arc_dots.append({
                    "item": token.canvas_items[key],
                    "azimuth": az,
                    "missile_key": m_key,
                    "lat": lat,
                    "lon": lon,
                })

            # thin curved line connecting the sampled launch points
            for seg_idx, seg in enumerate(self._split_latlon_at_dateline(arc_latlon)
                                          if len(arc_latlon) >= 3 else [arc_latlon]):
                pts = []
                for lat, lon in seg:
                    px, py = self._latlon_to_xy(lat, lon)
                    pts.extend([px, py])
                if len(pts) < 4:
                    continue
                key = f"threat_arc:{m_key}:{seg_idx}"
                active_line.add(key)
                if key not in token.canvas_items:
                    token.canvas_items[key] = self.map_canvas.create_line(
                        *pts, fill="#fbbf24", width=1, dash=(2, 3), tags=(token_tag,)
                    )
                else:
                    self.map_canvas.coords(token.canvas_items[key], *pts)
                    self.map_canvas.itemconfigure(token.canvas_items[key], state="normal")

        for key in dot_keys - active_dot:
            self.map_canvas.delete(token.canvas_items[key])
            del token.canvas_items[key]
        for key in line_keys - active_line:
            self.map_canvas.delete(token.canvas_items[key])
            del token.canvas_items[key]

        # keep dots on top of footprint polygons so they stay clickable
        self.map_canvas.tag_raise("threat_dot")

    def _hit_test_threat_dot(self, x, y):
        """Return (token, dot) for the threat-origin dot nearest (x, y), or (None, None).

        Picks the closest dot center rather than the first match so that clustered
        dots (short-range threats / zoomed-out map) still resolve to one the user aimed at.
        """
        items = set(self.map_canvas.find_overlapping(x - 6, y - 6, x + 6, y + 6))
        best = (None, None)
        best_d2 = None
        for token in self.tokens.values():
            if token.kind != "ilp":
                continue
            for dot in getattr(token, "threat_arc_dots", []):
                if dot["item"] not in items:
                    continue
                dx, dy = self._latlon_to_xy(dot["lat"], dot["lon"])
                d2 = (dx - x) ** 2 + (dy - y) ** 2
                if best_d2 is None or d2 < best_d2:
                    best_d2 = d2
                    best = (token, dot)
        return best

    def _isolate_threat_dot(self, token, dot):
        """Isolate a single launch point (toggle off if it is already isolated)."""
        already = (
            getattr(token, "isolated_azimuth", None) is not None
            and token.isolated_missile_key == dot["missile_key"]
            and abs(self._angle_diff(dot["azimuth"], token.isolated_azimuth)) < 0.5
        )
        if already:
            self._clear_isolation(token)
            return
        token.isolated_azimuth = dot["azimuth"]
        token.isolated_missile_key = dot["missile_key"]
        # Treat the isolated dot as a single fixed launch point: use the
        # from-launch-point solver (per-target range), oriented at the dot's azimuth.
        token._saved_mlp_latlon = getattr(token, "fixed_mlp_latlon", None)
        token.fixed_mlp_latlon = (dot["lat"], dot["lon"])
        token.center_dir = dot["azimuth"]
        self._select_token(token)
        self.status_var.set(
            f"Isolated launch point: az {dot['azimuth']:.1f}° · missile #{dot['missile_key']} (from-launch-point)"
        )
        self._apply_sidebar()
        thread = threading.Thread(target=self._compute_and_render, args=(token,), kwargs={"from_mlp": True})
        thread.daemon = True
        thread.start()

    def _clear_isolation(self, token, recompute=True):
        if getattr(token, "isolated_azimuth", None) is None:
            return
        token.isolated_azimuth = None
        token.isolated_missile_key = None
        # restore any launch point that existed before the dot was isolated
        token.fixed_mlp_latlon = getattr(token, "_saved_mlp_latlon", None)
        token._saved_mlp_latlon = None
        if recompute and token.footprints:
            self.status_var.set("Restored full-sector footprint")
            self._run_footprint()
        else:
            self._update_token_draw(token)

    def _draw_launch_box(self, token):
        """Draw the launch-sampling box (dotted) and its sampled MLP dots."""
        keys = {k for k in token.canvas_items if k.startswith("launch_box")}
        box = getattr(token, "launch_box", None)
        if token.kind != "ilp" or not box:
            for k in keys:
                self.map_canvas.delete(token.canvas_items[k])
                del token.canvas_items[k]
            return
        token_tag = f"token:{token.token_id}"
        # corners -> screen
        corners = [(box["lat1"], box["lon1"]), (box["lat1"], box["lon2"]),
                   (box["lat2"], box["lon2"]), (box["lat2"], box["lon1"])]
        pts = []
        for lat, lon in corners:
            x, y = self._latlon_to_xy(lat, lon)
            pts.extend([x, y])
        bk = "launch_box_rect"
        if bk not in token.canvas_items:
            token.canvas_items[bk] = self.map_canvas.create_polygon(
                pts, outline="#7c3aed", fill="", width=2, dash=(5, 4), tags=(token_tag,))
        else:
            self.map_canvas.coords(token.canvas_items[bk], *pts)
            self.map_canvas.itemconfigure(token.canvas_items[bk], state="normal")
        # sampled MLP dots
        active = {bk}
        for i, (mlat, mlon) in enumerate(self._launch_box_samples(box)):
            x, y = self._latlon_to_xy(mlat, mlon)
            k = f"launch_box_dot:{i}"
            active.add(k)
            if k not in token.canvas_items:
                token.canvas_items[k] = self.map_canvas.create_oval(
                    x - 3, y - 3, x + 3, y + 3, fill="#a78bfa", outline="#4c1d95",
                    width=1, tags=(token_tag,))
            else:
                self.map_canvas.coords(token.canvas_items[k], x - 3, y - 3, x + 3, y + 3)
                self.map_canvas.itemconfigure(token.canvas_items[k], state="normal")
        for k in keys - active:
            self.map_canvas.delete(token.canvas_items[k])
            del token.canvas_items[k]

    def _launch_box_samples(self, box):
        """Grid of (lat, lon) launch points sampled inside the box."""
        n = max(1, int(box.get("grid_n", 3)))
        lat1, lat2 = box["lat1"], box["lat2"]
        lon1, lon2 = box["lon1"], box["lon2"]
        out = []
        for i in range(n):
            fi = 0.5 if n == 1 else i / (n - 1)
            lat = lat1 + (lat2 - lat1) * fi
            for j in range(n):
                fj = 0.5 if n == 1 else j / (n - 1)
                lon = lon1 + (lon2 - lon1) * fj
                out.append((lat, lon))
        return out

    def _draw_fixed_mlp(self, token):
        """Draw the placed fixed launch point as an opaque yellow, draggable dot."""
        keys = {k for k in token.canvas_items if k.startswith("fixed_mlp")}
        latlon = getattr(token, "fixed_mlp_latlon", None)
        if token.kind != "ilp" or not latlon:
            for k in keys:
                self.map_canvas.delete(token.canvas_items[k])
                del token.canvas_items[k]
            return
        token_tag = f"token:{token.token_id}"
        mlp_x, mlp_y = self._latlon_to_xy(*latlon)
        ilp_x, ilp_y = self._latlon_to_xy(token.lat, token.lon)
        # connector line ILP -> MLP
        lk = "fixed_mlp_line"
        if lk not in token.canvas_items:
            token.canvas_items[lk] = self.map_canvas.create_line(
                ilp_x, ilp_y, mlp_x, mlp_y, fill="#eab308", width=1, dash=(4, 3), tags=(token_tag,)
            )
        else:
            self.map_canvas.coords(token.canvas_items[lk], ilp_x, ilp_y, mlp_x, mlp_y)
            self.map_canvas.itemconfigure(token.canvas_items[lk], state="normal")
        r = 8
        dk = "fixed_mlp_dot"
        if dk not in token.canvas_items:
            token.canvas_items[dk] = self.map_canvas.create_oval(
                mlp_x - r, mlp_y - r, mlp_x + r, mlp_y + r,
                fill="#facc15", outline="#1f2937", width=2, tags=(token_tag, "fixed_mlp"),
            )
        else:
            self.map_canvas.coords(token.canvas_items[dk], mlp_x - r, mlp_y - r, mlp_x + r, mlp_y + r)
            self.map_canvas.itemconfigure(token.canvas_items[dk], state="normal")
        tk_ = "fixed_mlp_label"
        if tk_ not in token.canvas_items:
            token.canvas_items[tk_] = self.map_canvas.create_text(
                mlp_x + 11, mlp_y - 11, text="launch pt", anchor="w",
                fill="#854d0e", font=("Helvetica", 8, "bold"), tags=(token_tag, "fixed_mlp"),
            )
        else:
            self.map_canvas.coords(token.canvas_items[tk_], mlp_x + 11, mlp_y - 11)
            self.map_canvas.itemconfigure(token.canvas_items[tk_], state="normal")
        self.map_canvas.tag_raise("fixed_mlp")

    def _draw_mlp_ranges(self, token):
        existing_keys = {k for k in token.canvas_items if k.startswith("mlp_range:")}
        active_keys = set()
        stats_lines = []

        for idx, entry in enumerate(token.missile_ranges):
            m_key = entry["missile_key"]
            radius_km = entry["range_km"]
            color = entry["color"]
            points = self._circle_points(token.lat, token.lon, radius_km, 72)
            segments = self._split_latlon_at_dateline(points)
            token_tag = f"token:{token.token_id}"
            for seg_idx, seg in enumerate(segments):
                screen_pts = []
                for lat, lon in seg:
                    px, py = self._latlon_to_xy(lat, lon)
                    screen_pts.extend([px, py])
                key = f"mlp_range:{m_key}:{seg_idx}"
                active_keys.add(key)
                if key not in token.canvas_items:
                    token.canvas_items[key] = self.map_canvas.create_polygon(
                        screen_pts,
                        outline=color,
                        fill="",
                        width=2,
                        dash=(5, 4),
                        joinstyle="round",
                        tags=(token_tag,),
                    )
                else:
                    self.map_canvas.coords(token.canvas_items[key], *screen_pts)
                    self.map_canvas.itemconfigure(token.canvas_items[key], state="normal", outline=color)
            stats_lines.append(f"#{m_key} · {radius_km:.0f} km")

        for key in existing_keys - active_keys:
            self.map_canvas.delete(token.canvas_items[key])
            del token.canvas_items[key]

        stats_text = " | ".join(stats_lines)
        stats_key = "mlp_stats"
        if stats_text:
            if stats_key not in token.canvas_items:
                token.canvas_items[stats_key] = self.map_canvas.create_text(
                    *self._latlon_to_xy(token.lat, token.lon),
                    text=stats_text,
                    anchor="n",
                    font=("Helvetica", 8),
                    tags=(f"token:{token.token_id}",)
                )
            else:
                self.map_canvas.coords(token.canvas_items[stats_key], *self._latlon_to_xy(token.lat, token.lon))
                self.map_canvas.itemconfigure(token.canvas_items[stats_key], text=stats_text, state="normal")
        else:
            if stats_key in token.canvas_items:
                self.map_canvas.itemconfigure(token.canvas_items[stats_key], state="hidden")

    def _run_footprint(self):
        token = self.selected_token
        if token is None:
            return
        self._apply_sidebar()
        det_note = self._footprint_det_label(token)
        self.status_var.set(f"Computing footprint ({det_note})...")
        thread = threading.Thread(target=self._compute_and_render, args=(token,))
        thread.daemon = True
        thread.start()

    def _run_footprint_from_mlp(self):
        token = self.selected_token
        if token is None or token.kind != "ilp":
            return
        self._apply_sidebar()
        if not getattr(token, "fixed_mlp_latlon", None):
            self.status_var.set("Place a launch point first (Set launch point, then click the map)")
            return
        # Orient the footprint/boresight toward the launch point it is computed against.
        mlp_lat, mlp_lon = token.fixed_mlp_latlon
        token.center_dir = self._bearing_between(token.lat, token.lon, mlp_lat, mlp_lon)
        token.isolated_azimuth = None
        token.isolated_missile_key = None
        self._refresh_sidebar_token(token)
        self.status_var.set("Computing footprint from launch point...")
        thread = threading.Thread(target=self._compute_and_render, args=(token,), kwargs={"from_mlp": True})
        thread.daemon = True
        thread.start()

    def _run_footprint_from_ranges(self):
        token = self.selected_token
        if token is None or token.kind != "ilp":
            return
        self._apply_sidebar()
        if not token.missile_keys:
            self.status_var.set("Select at least one threat missile first")
            return
        # Each missile's launch point is the central (boresight) dot on its own max-range
        # arc, so footprints stay oriented along the boresight.
        token.isolated_azimuth = None
        token.isolated_missile_key = None
        self.status_var.set("Computing footprints from each missile's max range...")
        thread = threading.Thread(target=self._compute_and_render, args=(token,), kwargs={"from_ranges": True})
        thread.daemon = True
        thread.start()

    def _run_footprint_full_sector(self):
        token = self.selected_token
        if token is None or token.kind != "ilp":
            return
        self._apply_sidebar()
        if not token.missile_keys:
            self.status_var.set("Select at least one threat missile first")
            return
        token.isolated_azimuth = None
        token.isolated_missile_key = None
        self.status_var.set("Computing All-Max-Distances footprints across the sector...")
        thread = threading.Thread(target=self._compute_and_render_full_sector, args=(token,))
        thread.daemon = True
        thread.start()

    def _compute_and_render_full_sector(self, token):
        """The standard (classic, max-range-everywhere) footprint fanned across a set of
        boresights spanning the sector. The classic shape is rotationally symmetric
        (independent of boresight, absent ground radars), so it is computed once per
        missile/interceptor and layered at each boresight azimuth (count = Threat dots
        per sector)."""
        try:
            footprints = []
            colors = ["red", "blue", "green", "orange", "purple", "brown", "cyan", "magenta"]
            interceptor_keys = token.interceptor_keys[:] if token.interceptor_keys else [int(token.config.get("itype", 11))]
            azimuths = self._threat_arc_azimuths(token)
            layer = 0
            for idx, m_key in enumerate(token.missile_keys):
                m_label = self._missile_label(m_key)
                for i_idx, i_key in enumerate(interceptor_keys):
                    i_label = self._interceptor_label(i_key)
                    # classic footprint (from_mlp=False): max-range threat to every target
                    xy = self._calculate_footprint_xy(token, m_key, i_type_override=i_key)
                    if not xy:
                        continue
                    pts = footprint_xy_to_meters(xy)
                    for b_idx, az in enumerate(azimuths):
                        footprints.append({
                            "missile_key": m_key,
                            "i_key": i_key,
                            "boresight": az,
                            "label": f"BS {az:.0f}° · {m_label} · {i_label}",
                            "color": colors[layer % len(colors)],
                            "points": pts,
                        })
                        layer += 1
            self._store_footprint_calculation(
                token, "full_sector", footprints,
                from_mlp_mode=False, full_sector_mode=True,
            )
        except Exception as exc:
            self._safe_status_update(f"Error: {exc}")
            return
        if not token.footprints:
            self.root.after(0, lambda: self._finish_footprint(token, "No footprint found for current settings"))
            return
        self.root.after(0, lambda: self._finish_footprint(token, f"Full-sector footprints: {len(token.footprints)} layers"))

    def _show_100pct_interceptable(self):
        token = self.selected_token
        if token is None or token.kind != "ilp":
            return
        full_entry = (getattr(token, "footprint_calculations", None) or {}).get("full_sector")
        if not full_entry or not full_entry.get("footprints"):
            self.status_var.set("Run 'All Max Distances, Full Sector' first")
            return
        self.status_var.set("Computing 100%-interceptable region(s)...")
        thread = threading.Thread(target=self._compute_100pct, args=(token, list(full_entry["footprints"])))
        thread.daemon = True
        thread.start()

    def _compute_100pct(self, token, full_fps):
        try:
            # group the full-sector layers by missile + interceptor
            groups = {}
            for fp in full_fps:
                groups.setdefault((fp["missile_key"], fp.get("i_key")), []).append(fp)
            colors = ["red", "blue", "green", "orange", "purple", "brown", "cyan", "magenta"]
            new_fps = []
            for gi, (key, fps) in enumerate(groups.items()):
                # transform each boresight footprint into the common east/north frame
                polys = []
                for fp in fps:
                    a = math.radians(fp.get("boresight", 0.0))
                    ca, sa = math.cos(a), math.sin(a)
                    polys.append([(x * ca + y * sa, -x * sa + y * ca) for x, y in fp["points"]])
                region = self._polygon_intersection_region(polys)
                if region:
                    m_label = self._missile_label(key[0])
                    i_label = self._interceptor_label(key[1]) if key[1] is not None else ""
                    new_fps.append({
                        "missile_key": key[0],
                        "i_key": key[1],
                        "boresight": 0.0,  # already in east/north
                        "label": f"100% defended · {m_label} · {i_label}",
                        "color": colors[gi % len(colors)],
                        "points": region,
                    })
            self._store_footprint_calculation(
                token, "pct100", new_fps,
                from_mlp_mode=False, full_sector_mode=False,
            )
        except Exception as exc:
            self._safe_status_update(f"Error: {exc}")
            return
        if not token.footprints:
            self.root.after(0, lambda: self._finish_footprint(token, "No region defended across 100% of boresights"))
            return
        self.root.after(0, lambda: self._finish_footprint(token, "Showing 100%-interceptable region(s)"))

    def _polygon_intersection_region(self, polys, n_grid=120, n_rays=120):
        """Intersection of several polygons (east/north metres). Rasterises the
        intersection onto a grid once, then traces a star-shaped boundary (list of
        (e, n)) from the intersection centroid against that grid, or returns []."""
        if not polys:
            return []
        es = [p[0] for poly in polys for p in poly]
        ns = [p[1] for poly in polys for p in poly]
        emin, emax, nmin, nmax = min(es), max(es), min(ns), max(ns)
        if emax - emin < 1.0 or nmax - nmin < 1.0:
            return []
        de = (emax - emin) / n_grid
        dn = (nmax - nmin) / n_grid
        grid = [[False] * (n_grid + 1) for _ in range(n_grid + 1)]
        sum_e = sum_n = 0.0
        count = 0
        for i in range(n_grid + 1):
            e = emin + i * de
            for j in range(n_grid + 1):
                n = nmin + j * dn
                if all(self._point_in_polygon(e, n, poly) for poly in polys):
                    grid[i][j] = True
                    sum_e += e
                    sum_n += n
                    count += 1
        if not count:
            return []
        cx = sum_e / count
        cy = sum_n / count

        def in_grid(e, n):
            i = int(round((e - emin) / de))
            j = int(round((n - nmin) / dn))
            if 0 <= i <= n_grid and 0 <= j <= n_grid:
                return grid[i][j]
            return False

        step = min(de, dn)
        max_r = math.hypot(emax - emin, nmax - nmin)
        boundary = []
        for k in range(n_rays):
            ang = 2.0 * math.pi * k / n_rays
            dx, dy = math.cos(ang), math.sin(ang)
            r = 0.0
            last = (cx, cy)
            while r <= max_r:
                e = cx + dx * r
                n = cy + dy * r
                if not in_grid(e, n):
                    break
                last = (e, n)
                r += step
            boundary.append(last)
        return boundary

    def _mlp_button_action(self, token):
        """Single stateful button: idle -> arm placement -> (place) -> proceed/compute."""
        self._apply_sidebar()
        state = getattr(token, "_mlp_btn_state", "idle")
        if state == "idle":
            token._mlp_btn_state = "placing"
            self.placing_fixed_mlp_for = token.token_id
            self.drawing_launch_box_for = None
            self.add_mode = None
            self.map_canvas.configure(cursor="crosshair")
            self.status_var.set("Click the map to place the launch point")
            self._build_sidebar(token)
        elif state == "placing":
            token._mlp_btn_state = "idle"
            self.placing_fixed_mlp_for = None
            self.map_canvas.configure(cursor="")
            self.status_var.set("Placement cancelled")
            self._build_sidebar(token)
        else:  # placed -> proceed
            token._mlp_btn_state = "idle"
            self.placing_fixed_mlp_for = None
            self.map_canvas.configure(cursor="")
            self._build_sidebar(token)
            self._run_footprint_from_mlp()

    def _box_button_action(self, token):
        """Single stateful button: idle -> arm box draw -> (draw) -> proceed/compute."""
        self._apply_sidebar()
        if not token.missile_keys:
            self.status_var.set("Select at least one threat missile first")
            return
        state = getattr(token, "_box_btn_state", "idle")
        if state == "idle":
            token._box_btn_state = "drawing"
            self.drawing_launch_box_for = token.token_id
            self.placing_fixed_mlp_for = None
            self.add_mode = None
            self.map_canvas.configure(cursor="crosshair")
            self.status_var.set("Drag a box on the map to sample launch points")
            self._build_sidebar(token)
        elif state == "drawing":
            token._box_btn_state = "idle"
            self.drawing_launch_box_for = None
            self.map_canvas.configure(cursor="")
            self.status_var.set("Box drawing cancelled")
            self._build_sidebar(token)
        else:  # drawn -> proceed
            token._box_btn_state = "idle"
            self.drawing_launch_box_for = None
            self.map_canvas.configure(cursor="")
            self._build_sidebar(token)
            if not getattr(token, "launch_box", None):
                self.status_var.set("No launch box to compute")
                return
            self.status_var.set("Computing launch-box footprints...")
            t = threading.Thread(target=self._compute_launch_box_footprints, args=(token,))
            t.daemon = True
            t.start()

    def _clear_launch_box_for_selected(self):
        token = self.selected_token
        if token is None or token.kind != "ilp":
            return
        self._destroy_launch_box_popup()
        token.launch_box = None
        token._box_btn_state = "idle"
        self.drawing_launch_box_for = None
        self.map_canvas.configure(cursor="")
        self._update_token_draw(token)
        self._build_sidebar(token)
        self.status_var.set("Launch box cleared")

    def _destroy_launch_box_popup(self):
        if self._launch_box_popup is not None:
            try:
                self._launch_box_popup.destroy()
            except tk.TclError:
                pass
            self._launch_box_popup = None

    def _show_launch_box_popup(self, token):
        """Small floating panel above the box to pick which threat missiles it fires."""
        self._destroy_launch_box_popup()
        box = getattr(token, "launch_box", None)
        if not box:
            return
        win = tk.Toplevel(self.root)
        self._launch_box_popup = win
        win.title("Launch box")
        win.configure(bg=THEME["panel_bg"])
        win.transient(self.root)
        outer = tk.Frame(win, bg=THEME["panel_bg"], padx=10, pady=8)
        outer.pack(fill="both", expand=True)
        tk.Label(outer, text="Missile types this box fires:", bg=THEME["panel_bg"],
                 fg=THEME["fg"], font=("TkDefaultFont", 9, "bold")).pack(anchor="w")
        firing = set(box.get("missile_keys", []))
        cb_vars = {}
        for m_key in token.missile_keys:
            v = tk.BooleanVar(value=(m_key in firing) or not firing)
            cb_vars[m_key] = v
            ttk.Checkbutton(outer, text=f"#{m_key} — {self._missile_label(m_key)}", variable=v).pack(anchor="w")
        grid_row = tk.Frame(outer, bg=THEME["panel_bg"])
        grid_row.pack(fill="x", pady=(6, 2))
        tk.Label(grid_row, text="Grid N×N:", bg=THEME["panel_bg"], fg=THEME["fg"]).pack(side="left")
        grid_var = tk.StringVar(value=str(box.get("grid_n", 3)))
        ttk.Entry(grid_row, textvariable=grid_var, width=5).pack(side="left", padx=4)

        def on_grid_change(*_a):
            try:
                box["grid_n"] = max(1, min(8, int(float(grid_var.get()))))
            except (ValueError, tk.TclError):
                return
            self._update_token_draw(token)
        grid_var.trace_add("write", on_grid_change)

        def apply_selection():
            sel = [k for k, v in cb_vars.items() if v.get()]
            if not sel:
                self.status_var.set("Select at least one missile for the box")
                return
            try:
                gn = max(1, min(8, int(float(grid_var.get()))))
            except (ValueError, tk.TclError):
                gn = 3
            box["missile_keys"] = sel
            box["grid_n"] = gn
            self._update_token_draw(token)
            self._destroy_launch_box_popup()
            self.status_var.set("Firing missiles set — press Proceed on the box button to compute")

        tk.Label(outer, text="Pick which missiles fire, then press Proceed on the box button.",
                 bg=THEME["panel_bg"], fg=THEME["fg_muted"], font=("TkDefaultFont", 9),
                 wraplength=240, justify="left").pack(anchor="w", pady=(6, 2))
        self._lib_btn(outer, "Apply firing selection", apply_selection, primary=True).pack(anchor="w", pady=(2, 0))
        win.bind("<Escape>", lambda e: self._destroy_launch_box_popup())
        win.protocol("WM_DELETE_WINDOW", self._destroy_launch_box_popup)
        win.update_idletasks()
        x1, y1 = self._latlon_to_xy(box["lat1"], box["lon1"])
        x2, y2 = self._latlon_to_xy(box["lat2"], box["lon2"])
        rx = self.map_canvas.winfo_rootx() + int((x1 + x2) / 2) - 90
        ry = self.map_canvas.winfo_rooty() + int(min(y1, y2)) - win.winfo_height() - 8
        win.geometry(f"+{max(0, rx)}+{max(0, ry)}")

    def _compute_launch_box_footprints(self, token):
        try:
            box = getattr(token, "launch_box", None)
            if not box:
                return
            firing = box.get("missile_keys") or list(token.missile_keys)
            samples = self._launch_box_samples(box)
            interceptor_keys = token.interceptor_keys[:] if token.interceptor_keys else [int(token.config.get("itype", 11))]
            loft = bool(getattr(token, "fixed_mlp_loft", False))
            colors = ["red", "blue", "green", "orange", "purple", "brown", "cyan", "magenta"]
            footprints = []
            layer = 0
            n_attempt = 0      # launch-point x missile attempts
            n_in_range = 0     # of those, with the launch point within the missile's reach
            for m_key in firing:
                missile_data = rd.missile(m_key, ROCKET_DATA_PATH)
                if missile_data is False:
                    continue
                if token.use_range_override and token.range_override_km > 0:
                    missile_data = dict(missile_data)
                    missile_data["range"] = str(token.range_override_km)
                tables = ss.build_mlp_tables(missile_data, loft=loft)  # once per missile, shared cache
                m_max = tables["max_range"]
                m_label = self._missile_label(m_key)
                for (mlat, mlon) in samples:
                    D = self._geodesic_distance_km(token.lat, token.lon, mlat, mlon) * 1000.0
                    if D <= 0:
                        continue
                    n_attempt += 1
                    if D < m_max:
                        n_in_range += 1
                    az = self._bearing_between(token.lat, token.lon, mlat, mlon)
                    for i_key in interceptor_keys:
                        xy = self._calculate_footprint_xy(
                            token, m_key, i_type_override=i_key, from_mlp=True,
                            mlp_dist_m_override=D, mlp_tables=tables)
                        if not xy:
                            continue
                        det_label = self._footprint_det_label(token)
                        footprints.append({
                            "missile_key": m_key, "i_key": i_key, "boresight": az,
                            "label": (
                                f"Box MLP({mlat:.2f},{mlon:.2f}) · {m_label} · "
                                f"{self._interceptor_label(i_key)} · {det_label}"
                            ),
                            "color": colors[layer % len(colors)],
                            "points": footprint_xy_to_meters(xy),
                        })
                        layer += 1
            self._store_footprint_calculation(
                token, "launch_box", footprints,
                from_mlp_mode=True, full_sector_mode=False,
            )
        except Exception as exc:
            self._safe_status_update(f"Error: {exc}")
            return
        if not token.footprints:
            # Empty is usually physics, not a range check: a launch point very short
            # vs the missile's max range forces a flat/depressed trajectory the interceptor
            # can't engage, and one near max range is degenerate (threat barely reaches).
            if n_in_range == 0:
                msg = f"Launch box: all {n_attempt} sampled points are beyond the missile's range — move the box closer."
            else:
                msg = (f"Launch box: {n_in_range}/{n_attempt} points within range but none yielded an "
                       f"interceptable footprint (very short range → flat/depressed trajectory, or near "
                       f"max range → degenerate). Try a box at an intermediate launch distance.")
            self.root.after(0, lambda m=msg: self._finish_footprint(token, m))
            return
        self.root.after(0, lambda: self._finish_footprint(token, f"Launch-box footprints: {len(token.footprints)} layers"))

    def _clear_fixed_mlp_for_selected(self):
        token = self.selected_token
        if token is None or token.kind != "ilp":
            return
        self.placing_fixed_mlp_for = None
        token.fixed_mlp_latlon = None
        token._mlp_btn_state = "idle"
        self.map_canvas.configure(cursor="")
        self._update_token_draw(token)
        self._build_sidebar(token)
        self.status_var.set("Launch point cleared")

    def _set_fixed_mlp(self, token, lat, lon):
        token.fixed_mlp_latlon = (clamp(lat, -85, 85), ((lon + 180) % 360) - 180)
        token.center_dir = self._bearing_between(
            token.lat, token.lon, token.fixed_mlp_latlon[0], token.fixed_mlp_latlon[1]
        )
        self._update_token_draw(token)
        self._refresh_sidebar_token(token)

    def _reset_isolation_for_selected(self):
        token = self.selected_token
        if token is None or token.kind != "ilp":
            return
        self._apply_sidebar()
        if getattr(token, "isolated_azimuth", None) is None:
            self.status_var.set("No launch point isolated")
            self._update_token_draw(token)
            return
        self._clear_isolation(token)

    def _run_mlp_ranges(self):
        token = self.selected_token
        if token is None:
            return
        self._apply_sidebar(skip_mlp_refresh=True)
        self.status_var.set("Computing missile ranges...")
        thread = threading.Thread(target=self._compute_mlp_ranges, args=(token,))
        thread.daemon = True
        thread.start()

    def _compute_mlp_ranges(self, token):
        try:
            ranges = []
            colors = ["red", "blue", "green", "orange", "purple", "brown"]
            for idx, m_key in enumerate(token.missile_keys):
                range_km = self._get_missile_range_km(m_key)
                if range_km is None:
                    continue
                ranges.append({
                    "missile_key": m_key,
                    "range_km": range_km,
                    "color": colors[idx % len(colors)]
                })
            token.missile_ranges = ranges
        except Exception as exc:
            self.status_var.set(f"Error: {exc}")
            return
        self.root.after(0, lambda: self._finish_footprint(token))

    def _compute_and_render(self, token, from_mlp=False, from_ranges=False):
        try:
            footprints = []
            colors = ["red", "blue", "green", "orange", "purple", "brown", "cyan", "magenta"]
            gbewr_tokens = [self.tokens[tid] for tid in (token.gbewr_assigned or []) if tid in self.tokens and self.tokens[tid].kind == "gbewr"]
            det_sweep = self._parse_km_sweep(token.det_sweep_km_csv) if token.use_det_sweep else []
            interceptor_keys = token.interceptor_keys[:] if token.interceptor_keys else [int(token.config.get("itype", 11))]
            # When a single launch point is isolated, compute only that missile and
            # don't clip to the sector (a single launch point is one azimuth, not a span).
            # Isolating an arc dot routes through the from-launch-point solver (from_mlp=True).
            # from_ranges: launch point per missile = the central (boresight) dot on that
            # missile's own max-range arc -- all computed at once via the same solver.
            eff_from_mlp = from_mlp or from_ranges
            isolated = (token.isolated_azimuth is not None) and not from_ranges
            restrict_missile = isolated and token.isolated_missile_key is not None
            loft_txt = "loft" if getattr(token, "fixed_mlp_loft", False) else "depr"
            if from_ranges:
                iso_prefix = ""  # built per-missile below (includes the range)
            elif from_mlp:
                iso_prefix = f"Isolated LP ({loft_txt}) · " if isolated else f"From MLP ({loft_txt}) · "
            elif isolated:
                iso_prefix = f"Isolated az {token.isolated_azimuth:.0f}° · "
            else:
                iso_prefix = ""
            missile_keys = [token.isolated_missile_key] if restrict_missile else token.missile_keys
            apply_sectoral = token.use_sectoral and not isolated and not eff_from_mlp
            for idx, m_key in enumerate(missile_keys):
                # Per-missile launch-point distance for "from ranges" = this missile's max range.
                mlp_override = None
                eff_prefix = iso_prefix
                if from_ranges:
                    rng_km = self._cached_missile_range_km(token, m_key)
                    if not rng_km:
                        continue
                    mlp_override = float(rng_km) * 1000.0
                    eff_prefix = f"From range {rng_km:.0f}km ({loft_txt}) · "
                for i_idx, i_key in enumerate(interceptor_keys):
                    i_label = self._interceptor_label(i_key)
                    m_label = self._missile_label(m_key)
                    base_color = colors[(idx + i_idx) % len(colors)]
                    footprint_xy = self._calculate_footprint_xy(token, m_key, gbewr_list=None, i_type_override=i_key, from_mlp=eff_from_mlp, mlp_dist_m_override=mlp_override)
                    if footprint_xy:
                        if apply_sectoral:
                            footprint_xy = self._filter_sector_points(footprint_xy, token.sector_width)
                        if footprint_xy:
                            base_label = self._footprint_det_label(token)
                            if getattr(token, "use_op_range_override", False) and getattr(token, "op_range_override_km", 0) > 0:
                                base_label += f" · op_cap={token.op_range_override_km:.0f}km*"
                            footprints.append({
                                "missile_key": m_key,
                                "label": f"{eff_prefix}{m_label} · {i_label} · {base_label}",
                                "color": base_color,
                                "points": footprint_xy_to_meters(footprint_xy)
                            })
                    for gb_idx, gb_tok in enumerate([] if eff_from_mlp else gbewr_tokens):
                        spec = gbewr.get_radar_spec(gb_tok.r_key) or {}
                        gb_list = [{"lat": gb_tok.lat, "lon": gb_tok.lon, "range_km": spec.get("range_km", 0)}]
                        footprint_xy = self._calculate_footprint_xy(token, m_key, gbewr_list=gb_list, i_type_override=i_key)
                        if footprint_xy:
                            if apply_sectoral:
                                footprint_xy = self._filter_sector_points(footprint_xy, token.sector_width)
                            if footprint_xy:
                                label = "GBEWR: " + (spec.get("type", f"r{gb_tok.r_key}")[:24])
                                if token.use_op_range_override and token.op_range_override_km > 0:
                                    label += f" · op_cap={token.op_range_override_km:.0f}km*"
                                footprints.append({
                                    "missile_key": m_key,
                                    "label": f"{eff_prefix}{m_label} · {i_label} · {label}",
                                    "color": colors[(idx + i_idx + 2 + gb_idx) % len(colors)],
                                    "points": footprint_xy_to_meters(footprint_xy)
                                })
                    for det_idx, det_km in enumerate(det_sweep):
                        footprint_xy = self._calculate_footprint_xy(
                            token,
                            m_key,
                            gbewr_list=None,
                            forced_det_range_km=det_km,
                            i_type_override=i_key,
                            from_mlp=eff_from_mlp,
                            mlp_dist_m_override=mlp_override,
                        )
                        if footprint_xy:
                            if apply_sectoral:
                                footprint_xy = self._filter_sector_points(footprint_xy, token.sector_width)
                            if footprint_xy:
                                label = f"Det {det_km:g} km"
                                if token.use_op_range_override and token.op_range_override_km > 0:
                                    label += f" · op_cap={token.op_range_override_km:.0f}km*"
                                footprints.append({
                                    "missile_key": m_key,
                                    "label": f"{eff_prefix}{m_label} · {i_label} · {label}",
                                    "color": colors[(idx + i_idx + 4 + det_idx) % len(colors)],
                                    "points": footprint_xy_to_meters(footprint_xy)
                                })
            if from_ranges:
                mode_id = "from_ranges"
            elif from_mlp and isolated:
                mode_id = "isolated_lp"
            elif from_mlp:
                mode_id = "from_mlp"
            else:
                mode_id = "classic"
            self._store_footprint_calculation(
                token, mode_id, footprints,
                from_mlp_mode=eff_from_mlp,
                full_sector_mode=False,
            )
        except Exception as exc:
            self.status_var.set(f"Error: {exc}")
            return
        if not token.footprints:
            self.root.after(0, lambda: self._finish_footprint(token, "No footprint found for current settings"))
            return
        self.root.after(0, lambda: self._finish_footprint(token))

    def _finish_footprint(self, token, message=None):
        self._sync_footprint_ids_and_visibility(token, reset_visibility=True)
        active = getattr(token, "active_footprint_mode", None)
        calcs = getattr(token, "footprint_calculations", None) or {}
        if active and active in calcs:
            calcs[active]["det_label"] = self._footprint_det_label(token)
        self.status_var.set(message or "Footprint updated")
        self._update_token_draw(token)

    def _destroy_footprint_layer_popup(self):
        if self._footprint_layer_popup is not None:
            try:
                self._footprint_layer_popup.destroy()
            except tk.TclError:
                pass
            self._footprint_layer_popup = None

    def _show_footprint_layer_popup(self, token):
        """Popup near the ILP to switch Calculation Mode and toggle footprint layers."""
        self._sync_footprint_ids_and_visibility(token, reset_visibility=False)
        self._destroy_footprint_layer_popup()

        win = tk.Toplevel(self.root)
        self._footprint_layer_popup = win
        win.title(f"Footprints — {token.name}")
        win.configure(bg=THEME["panel_bg"])
        win.transient(self.root)
        win.resizable(True, True)

        outer = tk.Frame(win, bg=THEME["panel_bg"], padx=14, pady=12)
        outer.pack(fill="both", expand=True)

        calc_modes = self._iter_footprint_calc_modes(token)
        mode_var = tk.StringVar(
            value=token.active_footprint_mode or (calc_modes[0][0] if calc_modes else "")
        )

        if len(calc_modes) > 1:
            tk.Label(
                outer,
                text="Calculation Mode",
                bg=THEME["panel_bg"],
                fg=THEME["fg"],
                font=("TkDefaultFont", 9, "bold"),
            ).pack(anchor="w")
            mode_frame = tk.Frame(outer, bg=THEME["panel_bg"])
            mode_frame.pack(fill="x", pady=(4, 8))
            for mode_id, entry in calc_modes:
                layer_count = len(entry.get("footprints") or [])
                suffix = f" ({layer_count} layer{'s' if layer_count != 1 else ''})" if layer_count else ""
                ttk.Radiobutton(
                    mode_frame,
                    text=f"{entry.get('label', mode_id)}{suffix}",
                    value=mode_id,
                    variable=mode_var,
                ).pack(anchor="w", pady=1)

        layers_heading = tk.Label(
            outer,
            text="Layers",
            bg=THEME["panel_bg"],
            fg=THEME["fg"],
            font=("TkDefaultFont", 9, "bold"),
        )
        # scrollable container so long layer lists (full-sector / launch-box) don't overflow
        checks_host = tk.Frame(outer, bg=THEME["panel_bg"])
        checks_canvas = tk.Canvas(checks_host, bg=THEME["panel_bg"], highlightthickness=0,
                                  width=380, height=300)
        checks_vbar = ttk.Scrollbar(checks_host, orient="vertical", command=checks_canvas.yview)
        checks_canvas.configure(yscrollcommand=checks_vbar.set)
        checks_vbar.pack(side="right", fill="y")
        checks_canvas.pack(side="left", fill="both", expand=True)
        checks_frame = tk.Frame(checks_canvas, bg=THEME["panel_bg"])
        _checks_win = checks_canvas.create_window((0, 0), window=checks_frame, anchor="nw")
        checks_frame.bind("<Configure>", lambda e: checks_canvas.configure(scrollregion=checks_canvas.bbox("all")))
        checks_canvas.bind("<Configure>", lambda e: checks_canvas.itemconfigure(_checks_win, width=e.width))
        checks_canvas.bind("<MouseWheel>", lambda e: checks_canvas.yview_scroll(int(-e.delta / 120), "units"))
        cb_vars = []

        def redraw():
            self._draw_footprint(token)

        def rebuild_layers():
            for child in checks_frame.winfo_children():
                child.destroy()
            cb_vars.clear()
            if len(token.footprints or []) <= 1:
                layers_heading.pack_forget()
                checks_host.pack_forget()
                return
            if len(calc_modes) > 1:
                layers_heading.pack(anchor="w", pady=(4, 0))
            else:
                layers_heading.pack_forget()
            checks_host.pack(fill="both", expand=True, pady=(4, 8))
            for fp in token.footprints:
                fp_id = int(fp.get("fp_id", 0))
                row = tk.Frame(checks_frame, bg=THEME["panel_bg"])
                row.pack(fill="x", pady=2)
                sw = tk.Canvas(row, width=14, height=14, highlightthickness=0, bg=THEME["panel_bg"])
                sw.create_rectangle(1, 1, 13, 13, outline=fp.get("color", "#888"), width=2, fill=THEME["panel_bg"])
                sw.pack(side="left", padx=(0, 6))
                display = fp.get("label") or self._missile_label(fp["missile_key"])
                var = tk.BooleanVar(value=self._fp_layer_visible(token, fp_id))
                cb_vars.append((fp_id, var))

                def on_toggle(fid=fp_id, v=var):
                    token.footprint_visibility[fid] = v.get()
                    self._persist_active_footprint_state(token)
                    redraw()

                ttk.Checkbutton(
                    row,
                    text=f"#{fp_id + 1} · m{fp['missile_key']} · {display}",
                    variable=var,
                    command=on_toggle,
                ).pack(side="left", fill="x", expand=True)

        def on_mode_change(*_args):
            self._switch_footprint_mode(token, mode_var.get())
            rebuild_layers()
            redraw()

        if len(calc_modes) > 1:
            mode_var.trace_add("write", on_mode_change)
            on_mode_change()
        else:
            if len(calc_modes) == 1:
                tk.Label(
                    outer,
                    text="Toggle overlays on the map (computed footprints are unchanged).",
                    bg=THEME["panel_bg"],
                    fg=THEME["fg_muted"],
                    font=("TkDefaultFont", 9),
                    wraplength=420,
                    justify="left",
                ).pack(anchor="w")
            rebuild_layers()

        btn_row = tk.Frame(outer, bg=THEME["panel_bg"])
        btn_row.pack(fill="x", pady=(6, 0))

        def show_all():
            for fid, v in cb_vars:
                v.set(True)
                token.footprint_visibility[fid] = True
            self._persist_active_footprint_state(token)
            redraw()

        def hide_all():
            for fid, v in cb_vars:
                v.set(False)
                token.footprint_visibility[fid] = False
            self._persist_active_footprint_state(token)
            redraw()

        def close():
            self._persist_active_footprint_state(token)
            self._destroy_footprint_layer_popup()

        if cb_vars:
            ttk.Button(btn_row, text="Show all", command=show_all).pack(side="left", padx=(0, 6))
            ttk.Button(btn_row, text="Hide all", command=hide_all).pack(side="left", padx=(0, 6))
        ttk.Button(btn_row, text="Close", command=close).pack(side="right")

        win.bind("<Escape>", lambda e: close())
        win.protocol("WM_DELETE_WINDOW", close)

        win.update_idletasks()
        cx, cy = self._latlon_to_xy(token.lat, token.lon)
        rx = self.map_canvas.winfo_rootx() + int(cx) + 18
        ry = self.map_canvas.winfo_rooty() + int(cy) + 18
        win.geometry(f"+{rx}+{ry}")

    def _set_status(self, message):
        self.status_var.set(message)

    def _on_status_change(self, *_args):
        """Run the busy spinner while a 'Computing…' status is showing. Marshalled to the
        main thread since status can be set from worker threads."""
        try:
            busy = self.status_var.get().strip().lower().startswith("computing")
        except tk.TclError:
            return
        self.root.after(0, self._spinner_start if busy else self._spinner_stop)

    def _spinner_start(self):
        if self._spin_job is not None:
            return
        try:
            self._spin_canvas.grid()
        except tk.TclError:
            return
        self._spin_step()

    def _spin_step(self):
        c = self._spin_canvas
        try:
            c.delete("all")
        except tk.TclError:
            self._spin_job = None
            return
        # faint full ring + a bright rotating arc
        c.create_oval(3, 3, 15, 15, outline=THEME["border"], width=2)
        c.create_arc(3, 3, 15, 15, start=self._spin_angle, extent=110,
                     style="arc", outline=THEME["success"], width=3)
        self._spin_angle = (self._spin_angle - 40) % 360
        self._spin_job = self.root.after(70, self._spin_step)

    def _spinner_stop(self):
        if self._spin_job is not None:
            try:
                self.root.after_cancel(self._spin_job)
            except tk.TclError:
                pass
            self._spin_job = None
        try:
            self._spin_canvas.delete("all")
            self._spin_canvas.grid_remove()
        except tk.TclError:
            pass

    def _safe_status_update(self, msg):
        """Thread-safe status bar update."""
        self.root.after(0, lambda: self.status_var.set(msg))

    def _open_builder(self):
        if not MPL_TK_AVAILABLE:
            messagebox.showerror("Builder", "matplotlib (TkAgg) is required for the Builder window.")
            return
        existing = getattr(self, "_builder_win", None)
        if existing is not None and getattr(existing, "win", None) is not None:
            try:
                existing.win.deiconify()
                existing.win.lift()
                return
            except tk.TclError:
                pass
        self._builder_win = BuilderWindow(self)

    def _calculate_footprint_xy(self, token, m_type, gbewr_list=None, forced_det_range_km=None, i_type_override=None, from_mlp=False, mlp_dist_m_override=None, mlp_tables=None):
        cfg = token.config
        sat_delay_prev = fcc_constants.sat_delay
        try:
            delay_s = float(token.space_based_delay_s)
        except (TypeError, ValueError):
            delay_s = float(cfg.get("set_sat_delay", sat_delay_prev))
        if token.use_space_based_delay:
            fcc_constants.sat_delay = max(0.0, delay_s)
        else:
            fcc_constants.sat_delay = 0.0
        i_type = int(i_type_override if i_type_override is not None else cfg.get("itype", 11))
        h_int_min = float(cfg.get("h_int_min", 0.0))
        h_discr = float(cfg.get("h_discr", 0.0))
        t_delay = float(cfg.get("t_delay", 0.0))
        mode2 = bool(cfg.get("fp_calc_mode", False))
        angle_step = float(cfg.get("angle_step", 5.0))
        num_steps_mode2 = int(cfg.get("num_steps_mode2", 15))
        acc = float(cfg.get("acc", 0.03))
        emode_sls = bool(cfg.get("set_shoot_look_shoot", False))
        emode_sls_timed = bool(cfg.get("set_sls_timed", False))
        sls_delay_s = max(0.0, float(cfg.get("sls_delay_s", 0.0)))

        try:
            missile_data = rd.missile(m_type, ROCKET_DATA_PATH)
        except (KeyError, AttributeError, TypeError, ValueError) as e:
            self._safe_status_update(f"Missile {m_type}: invalid data - {e}")
            return []
        if missile_data is False:
            self._safe_status_update(f"Missile {m_type} not found in rocket_data.json")
            return []
        if token.use_range_override and token.range_override_km > 0:
            missile_data = dict(missile_data)
            missile_data["range"] = str(token.range_override_km)
        trj = bm.balmisflight(missile_data)
        mrange = trj[len(trj) - 1, 2] * R_e
        if mrange > bm_range_limit:
            return []

        burn_time = sum(missile_data["t_bu"]) + sum(missile_data["t_delay"])
        t_int_lnc = float(burn_time + t_delay)

        interceptor_data = rd.interceptor(i_type, ROCKET_DATA_PATH)
        if interceptor_data is False:
            self._safe_status_update(f"Interceptor {i_type} not found in rocket_data.json")
            return []
        if h_discr < 1000:
            h_discr *= 1000
        if h_int_min < 500:
            h_int_min *= 1000
        h_int_min = max(h_int_min, interceptor_data.get("mpia", 0) * 1000)
        maxia = interceptor_data.get("maxia", 0) * 1000

        op_range = interceptor_data.get("op_range", 0)
        if op_range and op_range < 2000:
            op_range *= 1000
        if getattr(token, "use_op_range_override", False) and getattr(token, "op_range_override_km", 0) > 0:
            op_range = float(token.op_range_override_km) * 1000.0

        det_range = 0.0
        if forced_det_range_km is not None:
            det_range = max(0.0, float(forced_det_range_km)) * 1000.0
        elif token.use_space_based_delay:
            # Space-based override keeps det_range at 0 so sat-delay cueing is used.
            det_range = 0.0
        elif token.use_det_range_override and token.det_range_override_km > 0:
            det_range = token.det_range_override_km * 1000.0
        else:
            native_det = interceptor_data.get("det_range", 0.0)
            if isinstance(native_det, (list, tuple)):
                if len(native_det) > m_type:
                    det_range = float(native_det[m_type] or 0.0)
            elif isinstance(native_det, dict):
                key = str(m_type)
                if key in native_det:
                    det_range = float(native_det.get(key) or 0.0)
            elif native_det:
                det_range = float(native_det)
            if det_range and det_range < 11000:
                det_range *= 1000.0

        try:
            if mode2 and gbewr_list and len(gbewr_list) > 0:
                det_range = gbewr.make_det_range_func(
                    token.lat, token.lon, self._effective_center_dir(token),
                    det_range, trj, gbewr_list,
                    base_uses_sat_delay=token.use_space_based_delay
                )

            int_table = m.load_int_table(i_type, fcc_constants.psi_step, beta_step, ROCKET_DATA_PATH, "", cfg.get("set_keep_int_tables", True))

            if from_mlp:
                # Footprint defended against a threat from one FIXED launch point:
                # the MLP->target range (hence trajectory) changes per analyzed point.
                # Uses the mode-1 (fi/dist) parameterization regardless of fp_calc_mode.
                if mlp_dist_m_override is not None:
                    # explicit ILP->launch-point distance (e.g. this missile's own max range)
                    d_im = float(mlp_dist_m_override)
                else:
                    mlp_latlon = getattr(token, "fixed_mlp_latlon", None)
                    if not mlp_latlon:
                        return []
                    mlp_lat, mlp_lon = mlp_latlon
                    d_im = self._geodesic_distance_km(token.lat, token.lon, mlp_lat, mlp_lon) * 1000.0
                if d_im <= 0:
                    return []
                if emode_sls_timed:
                    base_search = ss.make_sls_timed_search(sls_delay_s)
                elif emode_sls:
                    base_search = ss.sls_search
                else:
                    base_search = ss.short_search
                mlp_search, fmlp_max_range = ss.make_fixed_mlp_search(
                    missile_data, d_im,
                    loft=bool(getattr(token, "fixed_mlp_loft", False)),
                    s_func=base_search,
                    tables=mlp_tables,
                )
                reach = d_im + fmlp_max_range
                fake_trj = np.array([[0.0, R_e, 0.0], [1.0, R_e, reach / R_e]])
                footprint_tab = fp.footprint_calc_v2(
                    mlp_search, fake_trj, int_table, h_int_min, maxia, t_int_lnc,
                    angle_step, op_range, det_range, t_delay, h_discr, acc, reach * dist_param
                )
                if not np.any(footprint_tab):
                    return []
                return [(row[3], row[4]) for row in footprint_tab]

            if not mode2:
                if emode_sls_timed:
                    search_func = ss.make_sls_timed_search(sls_delay_s)
                elif emode_sls:
                    search_func = ss.sls_search
                else:
                    search_func = ss.short_search
                dist = mrange * dist_param
                footprint_tab = fp.footprint_calc_v2(
                    search_func,
                    trj,
                    int_table,
                    h_int_min,
                    maxia,
                    t_int_lnc,
                    angle_step,
                    op_range,
                    det_range,
                    t_delay,
                    h_discr,
                    acc,
                    dist
                )
                if not np.any(footprint_tab):
                    return []
                return [(row[3], row[4]) for row in footprint_tab]

            if emode_sls_timed:
                search_func = ss.make_sls_timed_search2(sls_delay_s)
            elif emode_sls:
                search_func = ss.sls_search2
            else:
                search_func = ss.short_search2
            footprint_tab2 = ss.footprint_mode2(
                search_func,
                trj,
                int_table,
                h_int_min,
                maxia,
                t_int_lnc,
                angle_step,
                op_range,
                det_range,
                t_delay,
                h_discr,
                acc,
                num_steps_mode2
            )
            fp_m1 = isec.fprint_m2tom1(footprint_tab2)
            points = []
            for part in fp_m1:
                for row in part:
                    points.append((row[3], row[4]))  # x, y in km (same as Mode 1)
            return points
        finally:
            fcc_constants.sat_delay = sat_delay_prev

    def _apply_sidebar(self, skip_mlp_refresh=False):
        token = self.selected_token
        if token is None:
            return
        token.name = self.sidebar_vars["name"].get()
        token.lat = float(self.sidebar_vars["lat"].get())
        token.lon = float(self.sidebar_vars["lon"].get())
        if token.kind == "gbewr":
            radar_combo = self.sidebar_vars.get("gbewr_radar")
            if radar_combo is not None:
                radars = gbewr.load_gbewr_data()
                idx = radar_combo.current()
                if 0 <= idx < len(radars):
                    token.r_key = radars[idx].get("r_key")
        elif token.kind == "ilp":
            token.center_dir = float(self.sidebar_vars["center_dir"].get()) % 360
            token.sector_width = clamp(float(self.sidebar_vars["sector_width"].get()), 5.0, 180.0)
            token.show_center = bool(self.sidebar_vars["show_center"].get())
            token.show_sector = bool(self.sidebar_vars["show_sector"].get())
            token.use_sectoral = bool(self.sidebar_vars["use_sectoral"].get())
            raw_det_override = bool(self.sidebar_vars["use_det_range_override"].get())
            raw_space_override = bool(self.sidebar_vars["use_space_based_delay"].get())
            if raw_det_override and raw_space_override:
                prev_det = bool(token.use_det_range_override)
                prev_space = bool(token.use_space_based_delay)
                if raw_space_override != prev_space:
                    raw_det_override = False
                elif raw_det_override != prev_det:
                    raw_space_override = False
                else:
                    raw_space_override = False
                self.sidebar_vars["use_det_range_override"].set(raw_det_override)
                self.sidebar_vars["use_space_based_delay"].set(raw_space_override)
            token.use_det_range_override = raw_det_override
            token.det_range_override_km = float(self.sidebar_vars["det_range_override"].get() or 0.0)
            token.use_det_sweep = bool(self.sidebar_vars["use_det_sweep"].get())
            token.det_sweep_km_csv = str(self.sidebar_vars["det_sweep_km_csv"].get() or "").strip()
            token.use_space_based_delay = raw_space_override
            token.space_based_delay_s = max(0.0, float(self.sidebar_vars["space_based_delay_s"].get() or 0.0))
            token.config["set_sat_delay"] = token.space_based_delay_s
            token.use_range_override = bool(self.sidebar_vars["use_range_override"].get())
            token.range_override_km = float(self.sidebar_vars["range_override"].get() or 0.0)
            token.use_op_range_override = bool(self.sidebar_vars["use_op_range_override"].get())
            token.op_range_override_km = float(self.sidebar_vars["op_range_override_km"].get() or 0.0)
            if "show_threat_arc" in self.sidebar_vars:
                token.show_threat_arc = bool(self.sidebar_vars["show_threat_arc"].get())
            if "threat_arc_samples" in self.sidebar_vars:
                try:
                    token.threat_arc_samples = max(2, int(float(self.sidebar_vars["threat_arc_samples"].get())))
                except (ValueError, TypeError):
                    pass
            if "fixed_mlp_loft" in self.sidebar_vars:
                token.fixed_mlp_loft = bool(self.sidebar_vars["fixed_mlp_loft"].get())

            missile_list = self.sidebar_vars.get("missile_list")
            if missile_list is not None:
                missiles = self._get_missiles()
                selected = []
                for idx in missile_list.curselection():
                    if idx < len(missiles):
                        selected.append(int(missiles[idx]["m_key"]))
                if selected:
                    token.missile_keys = selected
                    token.config["mtype"] = selected[0]

            i_combo = self.sidebar_vars.get("ilp_itype_combo")
            i_keys = self.sidebar_vars.get("ilp_itype_keys", [])
            if i_combo is not None and i_keys:
                i_idx = i_combo.current()
                if 0 <= i_idx < len(i_keys):
                    token.config["itype"] = int(i_keys[i_idx])

            i_list = self.sidebar_vars.get("interceptor_list")
            if i_list is not None and i_keys:
                selected_i = [int(i_keys[i]) for i in i_list.curselection() if i < len(i_keys)]
                token.interceptor_keys = selected_i or [int(token.config.get("itype", 11))]

            gbewr_list = self.sidebar_vars.get("gbewr_list")
            gbewr_ids = self.sidebar_vars.get("gbewr_token_ids", [])
            if gbewr_list is not None and gbewr_ids:
                token.gbewr_assigned = [gbewr_ids[i] for i in gbewr_list.curselection() if i < len(gbewr_ids)]
        else:
            missile_list = self.sidebar_vars.get("mlp_missile_list")
            if missile_list is not None:
                missiles = self._get_missiles()
                selected = []
                for idx in missile_list.curselection():
                    if idx < len(missiles):
                        selected.append(int(missiles[idx]["m_key"]))
                if selected:
                    token.missile_keys = selected
                    token.config["mtype"] = selected[0]

        for key in list(token.config.keys()):
            field_id = f"cfg::{key}"
            var = self.sidebar_vars.get(field_id)
            if var is None:
                continue
            if isinstance(token.config[key], bool):
                token.config[key] = bool(var.get())
            else:
                raw = var.get()
                try:
                    if isinstance(token.config[key], int):
                        token.config[key] = int(float(raw))
                    elif isinstance(token.config[key], float):
                        token.config[key] = float(raw)
                    else:
                        token.config[key] = raw
                except ValueError:
                    token.config[key] = raw

        self._update_token_draw(token)
        if token.kind == "mlp" and not skip_mlp_refresh:
            self._run_mlp_ranges()

    def _delete_token(self):
        token = self.selected_token
        if token is None:
            return
        kind_label = (
            "Defense site" if token.kind == "ilp" else "Threat launch" if token.kind == "mlp" else "Radar site"
        )
        if not messagebox.askyesno(
            "Confirm Delete",
            f"Are you sure you would like to delete {token.name}?",
            icon="warning"
        ):
            return
        self._destroy_footprint_layer_popup()
        token_id = token.token_id
        token_tag = f"token:{token_id}"
        self.map_canvas.delete(token_tag)
        del self.tokens[token_id]
        if self.coverage_mode and (
            self.coverage_mode.get("ilp") == token or self.coverage_mode.get("mlp") == token
        ):
            self.coverage_mode = None
        if self.distance_mode and (
            self.distance_mode.get("point1", {}).get("token") == token
        ):
            self.distance_mode = None
        next_token = next(iter(self.tokens.values()), None) if self.tokens else None
        self._select_token(next_token)
        self._set_status(f"{kind_label} {token.name} deleted")

    def _refresh_sidebar_token(self, token):
        if self.selected_token != token:
            return
        self._suppress_sidebar_trace = True  # these are programmatic, not user edits
        try:
            self.sidebar_vars["lat"].set(f"{token.lat:.6f}")
            self.sidebar_vars["lon"].set(f"{token.lon:.6f}")
            if token.kind == "ilp":
                self.sidebar_vars["center_dir"].set(f"{token.center_dir:.1f}")
                self.sidebar_vars["sector_width"].set(f"{token.sector_width:.1f}")
                if "space_based_delay_s" in self.sidebar_vars:
                    self.sidebar_vars["space_based_delay_s"].set(f"{token.space_based_delay_s:.1f}")
        finally:
            self._suppress_sidebar_trace = False

    def _parse_km_sweep(self, raw):
        """Parse comma-separated detection-range list in km."""
        if not raw:
            return []
        out = []
        for part in str(raw).split(","):
            text = part.strip()
            if not text:
                continue
            try:
                val = float(text)
            except ValueError:
                continue
            if val > 0:
                out.append(val)
        # Remove duplicates while preserving order
        uniq = []
        seen = set()
        for v in out:
            key = round(v, 6)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(v)
        return uniq

    def _set_token_highlight(self, token, selected):
        color = "yellow" if selected else "white"
        self.map_canvas.itemconfigure(token.canvas_items["marker"], outline=color)

    def _start_coverage_check(self):
        if self.coverage_mode:
            self.coverage_mode = None
            self.distance_mode = None
            self._set_status("Coverage check canceled")
            return

        ilp_name = simpledialog.askstring("Select ILP", "ILP name (optional, otherwise click token):")
        ilp_token = self._find_token_by_name(ilp_name, kind="ilp") if ilp_name else None

        mlp_name = simpledialog.askstring("Select MLP", "MLP name (optional, otherwise click token):")
        mlp_token = self._find_token_by_name(mlp_name, kind="mlp") if mlp_name else None

        self.coverage_mode = {
            "step": "ilp" if ilp_token is None else "mlp" if mlp_token is None else "spot",
            "ilp": ilp_token,
            "mlp": mlp_token
        }

        if self.coverage_mode["step"] == "ilp":
            self._set_status("Select an ILP token on the map")
        elif self.coverage_mode["step"] == "mlp":
            self._set_status("Select an MLP token on the map")
        else:
            self._set_status("Click a spot on the map to check coverage")

    def _start_distance_measure(self):
        self._clear_distance_pins()
        if self.distance_mode:
            self.distance_mode = None
            self._set_status("Distance measure canceled")
            return
        self.distance_mode = {"step": "first", "point1": None}
        self._set_status("Select first point (defense site, threat launch, or any map location)")

    def _clear_distance_pins(self):
        self.map_canvas.delete("distance_pin")

    def _add_distance_map_pin(self, lat, lon, index, color):
        """Numbered map pin for distance measure (tags: distance_pin)."""
        x, y = self._latlon_to_xy(lat, lon)
        r = 8
        tag = "distance_pin"
        self.map_canvas.create_oval(
            x - r,
            y - r * 1.35,
            x + r,
            y + r * 0.15,
            fill=color,
            outline="#0f172a",
            width=2,
            tags=(tag,),
        )
        self.map_canvas.create_polygon(
            x - r * 0.35,
            y + r * 0.05,
            x + r * 0.35,
            y + r * 0.05,
            x,
            y + r * 1.45,
            fill=color,
            outline="#0f172a",
            width=1,
            joinstyle="round",
            tags=(tag,),
        )
        self.map_canvas.create_text(
            x,
            y - r * 0.48,
            text=str(index),
            fill="#ffffff",
            font=("Helvetica", 11, "bold"),
            tags=(tag,),
        )

    def _show_distance_result_dialog(self, p1_label, p2_label, dist_km):
        win = tk.Toplevel(self.root)
        win.title("Distance")
        win.configure(bg=THEME["panel_bg"])
        win.transient(self.root)
        win.resizable(False, False)
        win.grab_set()

        outer = tk.Frame(win, bg=THEME["panel_bg"], padx=28, pady=24)
        outer.pack(fill="both", expand=True)

        tk.Label(
            outer,
            text="Geodesic distance",
            bg=THEME["panel_bg"],
            fg=THEME["fg_muted"],
            font=("TkDefaultFont", 10),
        ).pack(anchor="w")
        tk.Label(
            outer,
            text=f"{dist_km:.3f} km",
            bg=THEME["panel_bg"],
            fg=THEME["accent"],
            font=("TkDefaultFont", 26, "bold"),
        ).pack(anchor="w", pady=(4, 18))

        tk.Label(
            outer,
            text=f"1 · {p1_label}",
            bg=THEME["panel_bg"],
            fg=THEME["fg"],
            font=("TkDefaultFont", 10),
            wraplength=440,
            justify="left",
        ).pack(anchor="w")
        tk.Label(
            outer,
            text=f"2 · {p2_label}",
            bg=THEME["panel_bg"],
            fg=THEME["fg"],
            font=("TkDefaultFont", 10),
            wraplength=440,
            justify="left",
        ).pack(anchor="w", pady=(8, 22))

        def close():
            win.destroy()
            self._clear_distance_pins()

        dbg = THEME["tool_btn_fg"]
        btn = tk.Button(
            outer,
            text="Done",
            command=close,
            bg=THEME["tool_btn_bg"],
            fg=dbg,
            activebackground=THEME["tool_btn_active_bg"],
            activeforeground=THEME["tool_btn_active_fg"],
            font=("TkDefaultFont", 10, "bold"),
            relief="flat",
            borderwidth=0,
            padx=22,
            pady=9,
            cursor="hand2",
            highlightthickness=0,
            highlightbackground=THEME["tool_btn_bg"],
            highlightcolor=THEME["tool_btn_bg"],
        )

        def d_enter(_e):
            btn.configure(bg=THEME["tool_btn_hover"], fg=dbg)

        def d_leave(_e):
            btn.configure(bg=THEME["tool_btn_bg"], fg=dbg)

        btn.bind("<Enter>", d_enter)
        btn.bind("<Leave>", d_leave)
        btn.pack(anchor="e")

        win.bind("<Return>", lambda e: close())
        win.bind("<Escape>", lambda e: close())
        win.protocol("WM_DELETE_WINDOW", close)

        win.update_idletasks()
        ww = win.winfo_width()
        wh = win.winfo_height()
        rx = self.root.winfo_rootx() + (self.root.winfo_width() - ww) // 2
        ry = self.root.winfo_rooty() + (self.root.winfo_height() - wh) // 2
        win.geometry(f"+{rx}+{ry}")

    def _destroy_interceptor_flight_preview_win(self):
        if self._interceptor_flight_preview_win is not None:
            try:
                self._interceptor_flight_preview_win.destroy()
            except tk.TclError:
                pass
            self._interceptor_flight_preview_win = None

    def _show_interceptor_flight_preview(self, token):
        """Run interceptor balmisflight; print to terminal and show summary + charts."""
        i_key = int(token.config.get("itype", 1))
        try:
            interceptor_data = rd.interceptor(i_key, ROCKET_DATA_PATH)
        except Exception as exc:
            self._set_status(f"Interceptor data error: {exc}")
            return

        capture = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = _TeeStream(old_stdout, capture)
        try:
            trj = bm.balmisflight(interceptor_data, trajectory_data=True, ind_flight_dataprint=True)
            if trj is False or trj is None:
                print("Interceptor flight calculation failed.")
                i_range = None
            else:
                if fcc_constants.check_fpa and interceptor_data.get("traj_type") != "bal_mis":
                    i_range = -2 * math.radians(trj[len(trj) - 1, 2]) * R_e
                else:
                    i_range = trj[len(trj) - 1, 2] * R_e
                print("Interceptor range = {:.0f} km".format(i_range / 1000))
        finally:
            sys.stdout = old_stdout

        text = capture.getvalue()
        if not text.endswith("\n"):
            text += "\n"

        self._destroy_interceptor_flight_preview_win()
        win = tk.Toplevel(self.root)
        self._interceptor_flight_preview_win = win
        r_name = interceptor_data.get("type", f"i{i_key}")
        win.title(f"Interceptor flight — {r_name}")
        win.configure(bg=THEME["panel_bg"])
        win.transient(self.root)
        win.resizable(True, True)

        outer = tk.Frame(win, bg=THEME["panel_bg"], padx=14, pady=12)
        outer.pack(fill="both", expand=True)

        tk.Label(
            outer,
            text=f"{r_name}  ·  i{i_key}",
            bg=THEME["panel_bg"],
            fg=THEME["accent"],
            font=("TkDefaultFont", 12, "bold"),
        ).pack(anchor="w")

        if i_range is not None:
            tk.Label(
                outer,
                text=f"Range: {i_range / 1000:.0f} km",
                bg=THEME["panel_bg"],
                fg=THEME["success"],
                font=("TkDefaultFont", 11, "bold"),
            ).pack(anchor="w", pady=(2, 8))

        text_frame = tk.Frame(outer, bg=THEME["panel_bg"])
        text_frame.pack(fill="both", expand=False, pady=(0, 8))

        text_scroll = tk.Scrollbar(text_frame, orient="vertical")
        text_widget = tk.Text(
            text_frame,
            height=11,
            width=92,
            wrap="none",
            font=("Menlo", 10) if self._is_macos else ("Courier", 10),
            bg=THEME["entry_bg"],
            fg=THEME["fg"],
            insertbackground=THEME["fg"],
            relief="flat",
            borderwidth=0,
            padx=8,
            pady=8,
            yscrollcommand=text_scroll.set,
        )
        text_scroll.config(command=text_widget.yview)
        text_scroll.pack(side="right", fill="y")
        text_widget.pack(side="left", fill="both", expand=True)
        text_widget.insert("1.0", text)
        text_widget.configure(state="disabled")

        chart_ok = (
            MATPLOTLIB_TK_AVAILABLE
            and trj is not False
            and trj is not None
            and len(trj) > 1
        )
        if chart_ok:
            try:
                fig = _build_flight_trajectory_figure(trj, f"{r_name} trajectory")
                canvas = FigureCanvasTkAgg(fig, master=outer)
                canvas.draw()
                canvas.get_tk_widget().pack(fill="both", expand=True, pady=(4, 8))
            except Exception as exc:
                tk.Label(
                    outer,
                    text=f"Chart unavailable: {exc}",
                    bg=THEME["panel_bg"],
                    fg=THEME["warning"],
                    font=("TkDefaultFont", 9),
                ).pack(anchor="w", pady=(0, 8))

        btn_row = tk.Frame(outer, bg=THEME["panel_bg"])
        btn_row.pack(fill="x")

        def close():
            self._destroy_interceptor_flight_preview_win()

        dbg = THEME["tool_btn_fg"]
        btn = tk.Button(
            btn_row,
            text="Close",
            command=close,
            bg=THEME["tool_btn_bg"],
            fg=dbg,
            activebackground=THEME["tool_btn_active_bg"],
            activeforeground=THEME["tool_btn_active_fg"],
            font=("TkDefaultFont", 10, "bold"),
            relief="flat",
            borderwidth=0,
            padx=18,
            pady=8,
            cursor="hand2",
            highlightthickness=0,
        )

        def d_enter(_e):
            btn.configure(bg=THEME["tool_btn_hover"], fg=dbg)

        def d_leave(_e):
            btn.configure(bg=THEME["tool_btn_bg"], fg=dbg)

        btn.bind("<Enter>", d_enter)
        btn.bind("<Leave>", d_leave)
        btn.pack(anchor="e")

        win.bind("<Escape>", lambda _e: close())
        win.protocol("WM_DELETE_WINDOW", close)

        win.geometry("900x680")
        win.update_idletasks()
        ww = win.winfo_width()
        wh = win.winfo_height()
        rx = self.root.winfo_rootx() + max(20, (self.root.winfo_width() - ww) // 2)
        ry = self.root.winfo_rooty() + max(20, (self.root.winfo_height() - wh) // 2)
        win.geometry(f"+{rx}+{ry}")

        status = f"Interceptor flight preview — {r_name}"
        if i_range is not None:
            status += f" ({i_range / 1000:.0f} km)"
        self._set_status(status)

    def _handle_distance_click(self, event):
        point = self._get_click_point(event)
        if point is None:
            return
        lat, lon, label, token = point
        step = self.distance_mode["step"]
        if step == "first":
            self.distance_mode["point1"] = {"lat": lat, "lon": lon, "label": label, "token": token}
            self.distance_mode["step"] = "second"
            self._add_distance_map_pin(lat, lon, 1, "#2563eb")
            self._set_status(f"First point set — choose second point")
            return
        p1 = self.distance_mode["point1"]
        dist_km = self._geodesic_distance_km(p1["lat"], p1["lon"], lat, lon)
        msg = f"{p1['label']} ↔ {label}: {dist_km:.2f} km"
        self._set_status(msg)
        self._add_distance_map_pin(lat, lon, 2, "#d97706")
        self.distance_mode = None
        self._show_distance_result_dialog(p1["label"], label, dist_km)

    def _get_click_point(self, event):
        """Return (lat, lon, label, token_or_none) for a click: token (ILP/MLP) or map point."""
        item = self.map_canvas.find_withtag("current")
        if item:
            tags = self.map_canvas.gettags(item)
            token_id = self._token_id_from_tags(tags)
            if token_id is not None and token_id in self.tokens:
                tok = self.tokens[token_id]
                if tok.kind in ("ilp", "mlp"):
                    return (tok.lat, tok.lon, tok.name, tok)
        lat, lon = self._xy_to_latlon(event.x, event.y)
        return (lat, lon, f"{lat:.4f}, {lon:.4f}", None)

    def _bearing_between(self, lat1, lon1, lat2, lon2):
        """Initial great-circle bearing (deg, from north) from point 1 to point 2."""
        if _geod is not None:
            return _geod.Inverse(lat1, lon1, lat2, lon2)["azi1"] % 360.0
        p1 = math.radians(lat1)
        p2 = math.radians(lat2)
        dl = math.radians(lon2 - lon1)
        y = math.sin(dl) * math.cos(p2)
        x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
        return math.degrees(math.atan2(y, x)) % 360.0

    def _geodesic_distance_km(self, lat1, lon1, lat2, lon2):
        """Geodesic distance in km between two points."""
        if _geod is not None:
            inv = _geod.Inverse(lat1, lon1, lat2, lon2)
            return inv["s12"] / 1000.0
        # Haversine fallback
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(min(1, a)))
        return R_e / 1000.0 * c

    def _handle_coverage_click(self, event):
        item = self.map_canvas.find_withtag("current")
        token_id = None
        if item:
            tags = self.map_canvas.gettags(item)
            token_id = self._token_id_from_tags(tags)

        step = self.coverage_mode["step"]
        if step == "ilp":
            if token_id is None or self.tokens[token_id].kind != "ilp":
                self._set_status("Click an ILP token")
                return
            self.coverage_mode["ilp"] = self.tokens[token_id]
            self.coverage_mode["step"] = "mlp"
            self._set_status("Select an MLP token on the map")
            return

        if step == "mlp":
            if token_id is None or self.tokens[token_id].kind != "mlp":
                self._set_status("Click an MLP token")
                return
            self.coverage_mode["mlp"] = self.tokens[token_id]
            self.coverage_mode["step"] = "spot"
            self._set_status("Click a spot on the map to check coverage")
            return

        if step == "spot":
            lat, lon = self._xy_to_latlon(event.x, event.y)
            ilp = self.coverage_mode["ilp"]
            mlp = self.coverage_mode["mlp"]
            missile_key = mlp.missile_keys[0] if mlp and mlp.missile_keys else int(ilp.config.get("mtype", 1))
            defendable = self._is_point_defendable(ilp, missile_key, lat, lon)
            status = "defendable" if defendable else "NOT defendable"
            label = f"{lat:.4f}, {lon:.4f} is {status} by {ilp.name} against MLP {mlp.name if mlp else 'N/A'} m{missile_key}"
            self._add_coverage_pin(lat, lon, label, "green" if defendable else "red")
            self._set_status(label)
            self.coverage_mode = None

    def _find_token_by_name(self, name, kind=None):
        if not name:
            return None
        for token in self.tokens.values():
            if kind and token.kind != kind:
                continue
            if token.name.strip().lower() == name.strip().lower():
                return token
        return None

    def _add_coverage_pin(self, lat, lon, text, color):
        x, y = self._latlon_to_xy(lat, lon)
        r = 7
        head = self.map_canvas.create_oval(
            x - r, y - r * 1.4, x + r, y + r * 0.2, fill=color, outline="#0f172a", width=2
        )
        stem = self.map_canvas.create_polygon(
            x - r * 0.35,
            y + r * 0.1,
            x + r * 0.35,
            y + r * 0.1,
            x,
            y + r * 1.5,
            fill=color,
            outline="#0f172a",
            width=1,
            joinstyle="round",
        )
        label = self.map_canvas.create_text(
            x + 10, y + 12, text=text, anchor="nw", font=("Helvetica", 8, "bold"), fill="#0f172a"
        )
        self.coverage_pins.extend([(head, stem, label)])

    def _is_point_defendable(self, ilp_token, missile_key, lat, lon):
        # Use existing footprint if available; otherwise compute.
        footprint_points = None
        for fp in ilp_token.footprints:
            if fp["missile_key"] == missile_key:
                footprint_points = fp["points"]
                break
        if footprint_points is None:
            footprint_xy = self._calculate_footprint_xy(ilp_token, missile_key)
            if not footprint_xy:
                return False
            if ilp_token.use_sectoral:
                footprint_xy = self._filter_sector_points(footprint_xy, ilp_token.sector_width)
            footprint_points = footprint_xy_to_meters(footprint_xy)

        # Convert lat/lon to ILP-local x,y (meters), then rotate to footprint frame.
        lat0 = ilp_token.lat
        lon0 = ilp_token.lon
        dlat = math.radians(lat - lat0)
        dlon = math.radians(lon - lon0)
        east = dlon * R_e * math.cos(math.radians(lat0))
        north = dlat * R_e
        theta = math.radians(ilp_token.center_dir)
        x_local = east * math.cos(theta) - north * math.sin(theta)
        y_local = east * math.sin(theta) + north * math.cos(theta)

        return self._point_in_polygon(x_local, y_local, footprint_points)

    def _point_in_polygon(self, x, y, polygon):
        if not polygon:
            return False
        inside = False
        n = len(polygon)
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            intersect = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi)
            if intersect:
                inside = not inside
            j = i
        return inside

    def _on_sector_field_change(self):
        """Live-apply boresight/sector-width edits so the sector and the threat-origin
        dots track the values as they are typed (no Apply needed)."""
        if self._suppress_sidebar_trace:
            return  # programmatic var update (e.g. drag refresh), not a user edit
        token = self.selected_token
        if token is None or token.kind != "ilp":
            return
        try:
            cd = float(self.sidebar_vars["center_dir"].get()) % 360.0
            sw = clamp(float(self.sidebar_vars["sector_width"].get()), 5.0, 180.0)
        except (ValueError, KeyError, tk.TclError):
            return  # mid-edit / partial input
        token.center_dir = cd
        token.sector_width = sw
        self._update_token_draw(token)

    def _on_missile_select(self):
        """Live sync when the threat-missile multi-select changes.

        Updates the selected threat set, and if the primary threat (the 'mtype'
        text entry in Solver parameters) is no longer among the selected missiles,
        adopts the first selected one and reflects it in that text entry.
        """
        token = self.selected_token
        if token is None or token.kind not in ("ilp", "mlp"):
            return
        lb_key = "missile_list" if token.kind == "ilp" else "mlp_missile_list"
        listbox = self.sidebar_vars.get(lb_key)
        if listbox is None:
            return
        missiles = self._get_missiles()
        selected = [int(missiles[i]["m_key"]) for i in listbox.curselection() if i < len(missiles)]
        if not selected:
            return
        token.missile_keys = selected
        try:
            primary = int(token.config.get("mtype", 0))
        except (TypeError, ValueError):
            primary = 0
        if primary not in selected:
            token.config["mtype"] = selected[0]
            mtype_var = self.sidebar_vars.get("cfg::mtype")
            if mtype_var is not None:
                mtype_var.set(str(selected[0]))
            self._set_status(f"Primary threat #{primary} deselected — now #{selected[0]}")
        self._update_token_draw(token)

    def _populate_missile_list(self, token, listbox):
        listbox.delete(0, tk.END)
        missiles = self._get_missiles()
        for idx, item in enumerate(missiles):
            nm = (item.get("type") or "").strip() or "unnamed"
            label = f"#{item['m_key']} — {nm}"
            listbox.insert(tk.END, label)
            if int(item["m_key"]) in token.missile_keys:
                listbox.selection_set(idx)

    def _populate_interceptor_list(self, token, listbox):
        listbox.delete(0, tk.END)
        interceptors = self._get_interceptors()
        active = set(token.interceptor_keys or [int(token.config.get("itype", 11))])
        for idx, item in enumerate(interceptors):
            i_key = int(item.get("i_key", item.get("m_key", 0)))
            nm = (item.get("type") or "").strip() or "unnamed"
            label = f"#{i_key} — {nm}"
            listbox.insert(tk.END, label)
            if i_key in active:
                listbox.selection_set(idx)

    def _config_tooltip(self, key):
        tips = {
            "mtype": "Index of the threat missile in the Missile types library.",
            "itype": "Interceptor design used for footprint and range tables.",
            "h_int_min": "Reject intercept solutions below this altitude.",
            "h_discr": "Altitude for discrimination / timeline assumptions.",
            "t_delay": "Seconds from cue to interceptor launch.",
            "fp_calc_mode": "Use alternate footprint solver (Mode 2) when enabled.",
            "acc": "Numerical cutoff for footprint boundary search.",
            "angle_step": "Azimuth sampling step for the fast footprint mode.",
            "num_steps_mode2": "Samples along trajectory in Mode 2.",
            "set_shoot_look_shoot": "Use shoot-look-shoot engagement logic when available.",
            "set_sls_timed": "Timed shoot-look-shoot: launch the second interceptor a fixed delay after the first, instead of waiting for the first to reach the target. Overrides plain Shoot-Look-Shoot when enabled.",
            "sls_delay_s": "Seconds between the first and second interceptor launches in timed shoot-look-shoot. Shorter delays enlarge the footprint, longer delays shrink it.",
            "set_keep_int_tables": "Keep generated interception tables on disk for debugging.",
            "set_sat_delay": "Latency for space-based cueing (seconds).",
        }
        return tips.get(key) or (
            f"Config field «{key.replace('_', ' ')}». Open the Field reference tab for the full variable glossary."
        )

    def _config_label(self, key):
        labels = {
            "m_key": "Missile catalog #",
            "i_key": "Interceptor catalog #",
            "type": "Type",
            "note": "Notes",
            "cd_type": "Drag coefficient model",
            "m_st": "Stage mass (kg)",
            "m_fu": "Fuel mass (kg)",
            "v_ex": "Exhaust velocity (m/s)",
            "t_bu": "Burn time (s)",
            "t_delay": "Interceptor launch delay (s)",
            "a_mid": "Midcourse drag area",
            "a_nz": "Nozzle area",
            "c_bal": "Ballistic coefficient",
            "m_warhead": "Warhead mass (kg)",
            "m_shroud": "Shroud mass (kg)",
            "t_shroud": "Shroud release time (s)",
            "m_pl": "Payload mass (kg)",
            "vert_launch_height": "Vertical launch height (m)",
            "grav_turn_angle": "Gravity turn angle (deg)",
            "traj_type": "Trajectory type",
            "flight_path_angle": "Flight path angle (deg)",
            "mpia": "Minimum possible intercept altitude (km)",
            "maxia": "Maximum intercept altitude (km)",
            "op_range": "Operational range (km)",
            "range": "Range (km)",
            "det_range": "Detection range by missile type (km)",
            "mtype": "Threat missile (catalog #)",
            "itype": "Interceptor (catalog #)",
            "h_int_min": "Min intercept altitude (km)",
            "h_discr": "Warhead discrimination altitude (km)",
            "h_int_min_list": "Min intercept altitude list (km)",
            "h_discr_list": "Discrimination altitude list (km)",
            "t_delay_list": "Launch delay list (s)",
            "op_range_list": "Interceptor op range list (km)",
            "maxia_list": "Max intercept altitude list (km)",
            "fp_calc_mode": "Footprint mode (Mode 2)",
            "acc": "Search cutoff threshold",
            "angle_step": "Mode 1 angle step (deg)",
            "num_steps_mode2": "Mode 2 steps",
            "set_shoot_look_shoot": "Shoot-Look-Shoot mode",
            "set_sls_timed": "Timed Shoot-Look-Shoot mode",
            "sls_delay_s": "SLS 2nd-shot delay (s)",
            "det_range_list": "Detection ranges list (km)",
            "mumi_list": "Multi-missile list",
            "muin_list": "Multi-interceptor list",
            "sect_angle_beg": "Probing angle begin (deg)",
            "sect_angle_end": "Probing angle end (deg)",
            "sect_angle_step": "Probing angle step (deg)",
            "sect_dist_beg": "Probing distance begin (km)",
            "sect_dist_num": "Probing distance steps",
            "gtheight_beg": "GT height begin (km)",
            "gtheight_end": "GT height end (km)",
            "gtangle_beg": "GT angle begin (deg)",
            "gtangle_end": "GT angle end (deg)",
            "maxrange_acc": "Max range accuracy",
            "set_mirror_segment": "Mirror probing segment",
            "plot_hit_charts": "Plot hit charts",
            "hit_chart_angle": "Hit chart angle (deg)",
            "set_keep_int_tables": "Keep interception tables",
            "set_keep_fp_chart": "Keep footprint charts",
            "set_keep_fp_data": "Keep footprint data",
            "set_keep_trj_data": "Keep trajectory data",
            "stdout_to_file": "Save stdout to file",
            "set_keep_stdout_file": "Keep stdout file",
            "set_time_stamp": "Timestamp output files",
            "set_int_table_samp_verify": "Verify sampled table",
            "sound_task_complete": "Sound on completion",
            "save_config_on_exit": "Save config on exit",
            "show_extra_param": "Show extra parameters",
            "show_extra_procs": "Show extra procedures",
            "show_ftprint_probe": "Show probing footprint",
            "show_chart_titles": "Show chart titles",
            "def_sector_step": "Default sector step (deg)",
            "set_sat_delay": "Satellite delay (s)",
            "set_psi_step": "Interceptor angle step (deg)",
            "no_atmosphere": "Disable atmosphere",
        }
        return labels.get(key, key.replace("_", " ").title())

    def _add_procedure_buttons(self, parent):
        btn_frame = tk.Frame(parent, bg=THEME["panel_inner"])
        btn_frame.pack(fill="x", padx=4, pady=4)
        buttons = [
            ("Footprint (standard)", lambda: self._call_fp_gui("footprint"), "Main single-footprint workflow."),
            ("Multi-footprint vs detection range", lambda: self._call_fp_gui("multi_detrange"), "Sweep detection assumptions."),
            ("Double footprint by intercept mode", lambda: self._call_fp_gui("double_footprint"), "Compare two engagement modes."),
            ("Multi-footprint by intercept params", lambda: self._call_fp_gui("param_footprint"), "Parameter sweep."),
            ("Multi-missile footprint", lambda: self._call_fp_gui("multi_missile"), "Several threat types at once."),
            ("Multi-interceptor footprint", lambda: self._call_fp_gui("multi_interceptor"), "Several interceptors."),
            ("Footprint by probing", lambda: self._call_fp_gui("probing"), "Probe sector / range grid."),
            ("Missile flight preview", lambda: self._call_fp_gui("missile_flight"), ""),
            ("Missile trajectory export", lambda: self._call_fp_gui("missile_traj"), ""),
            ("Missile trajectory charts", lambda: self._call_fp_gui("missile_charts"), ""),
            ("Missile max range estimate", lambda: self._call_fp_gui("missile_maxrange"), ""),
            ("Interceptor flight preview", lambda: self._call_fp_gui("interceptor_flight"), ""),
            ("Interceptor trajectory export", lambda: self._call_fp_gui("interceptor_traj"), ""),
            ("Interceptor trajectory charts", lambda: self._call_fp_gui("interceptor_charts"), ""),
            ("Interceptor max range estimate", lambda: self._call_fp_gui("interceptor_maxrange"), ""),
            ("Build interception table", lambda: self._call_fp_gui("interception_table"), "Generate lookup tables for offline use."),
        ]

        for label, action, tip in buttons:
            btn = self._lib_btn(btn_frame, label, action)
            btn.pack(fill="x", pady=3)
            if tip:
                ToolTip(btn, tip, delay_ms=500)

    @staticmethod
    def _star_xy(cx, cy, r_outer, r_inner, points=5):
        flat = []
        n = points * 2
        for i in range(n):
            ang = math.pi / 2 + i * math.pi / points
            r = r_outer if i % 2 == 0 else r_inner
            flat.append(cx + r * math.sin(ang))
            flat.append(cy - r * math.cos(ang))
        return flat

    @staticmethod
    def _hex_xy(cx, cy, r):
        flat = []
        for i in range(6):
            ang = math.pi / 2 + i * math.pi / 3
            flat.append(cx + r * math.sin(ang))
            flat.append(cy - r * math.cos(ang))
        return flat

    @staticmethod
    def _diamond_xy(cx, cy, r):
        return [cx, cy - r, cx + r, cy, cx, cy + r, cx - r, cy]

    @staticmethod
    def _radar_xy(cx, cy, r):
        """Rounded triangular radar icon."""
        return [
            cx, cy - r * 1.1,
            cx - r * 0.95, cy + r * 0.72,
            cx - r * 0.28, cy + r * 0.35,
            cx + r * 0.28, cy + r * 0.35,
            cx + r * 0.95, cy + r * 0.72,
        ]

    def _call_fp_gui(self, action):
        token = self.selected_token
        if token is None:
            self._set_status("Select an ILP token first")
            return
        if token.kind != "ilp":
            self._set_status("Procedures are available for ILP tokens only")
            return

        # Keep token.config (itype, mtype, etc.) in sync with the inspector before fp_gui reads it.
        self._apply_sidebar()

        if action == "interceptor_flight":
            self._show_interceptor_flight_preview(token)
            return

        try:
            import fp_gui as fpg
        except Exception as exc:
            self._set_status(f"fp_gui import error: {exc}")
            return

        self._ensure_fp_gui(fpg)
        self._sync_fp_gui_from_token(fpg, token)

        try:
            if action == "footprint":
                fpg.gui_footprint()
            elif action == "multi_detrange":
                fpg.gui_multi_detrange_footprint()
            elif action == "double_footprint":
                fpg.gui_double_footprint()
            elif action == "param_footprint":
                fpg.gui_param_footprint()
            elif action == "multi_missile":
                fpg.gui_multi_missile_footprint()
            elif action == "multi_interceptor":
                fpg.gui_multi_interceptor_footprint()
            elif action == "probing":
                fpg.gui_probing()
            elif action == "missile_flight":
                fpg.gui_balmis_flight()
            elif action == "missile_traj":
                fpg.gui_balmis_trajectory()
            elif action == "missile_charts":
                missile_data = rd.missile(int(token.config.get("mtype", 1)), ROCKET_DATA_PATH)
                fpg.gui_trajcharts(missile_data, True)
            elif action == "missile_maxrange":
                fpg.gui_missile_maxrange()
            elif action == "interceptor_traj":
                fpg.gui_interceptor_trajectory()
            elif action == "interceptor_charts":
                fpg.gui_interceptor_trajcharts()
            elif action == "interceptor_maxrange":
                fpg.gui_interceptor_maxrange()
            elif action == "interception_table":
                fpg.gui_interception_table()
            else:
                self._set_status("Unknown procedure")
        except Exception as exc:
            self._set_status(f"Procedure error: {exc}")

    def _ensure_fp_gui(self, fpg):
        if hasattr(fpg, "root") and getattr(fpg, "root") is not None:
            return

        fpg.root = self.root
        fpg.default_config()
        cfg = fpg.program_config
        fpg.busy_running = tk.BooleanVar(fpg.root, value=False)

        fpg.mtype_var = tk.IntVar(fpg.root, value=cfg["mtype"])
        fpg.itype_var = tk.IntVar(fpg.root, value=cfg["itype"])
        fpg.mia_var = tk.DoubleVar(fpg.root, value=cfg["h_int_min"])
        fpg.h_discr_var = tk.DoubleVar(fpg.root, value=cfg["h_discr"])
        fpg.t_delay_var = tk.DoubleVar(fpg.root, value=cfg["t_delay"])

        fpg.h_int_min_list_var = tk.StringVar(fpg.root, value=cfg["h_int_min_list"])
        fpg.h_discr_list_var = tk.StringVar(fpg.root, value=cfg["h_discr_list"])
        fpg.t_delay_list_var = tk.StringVar(fpg.root, value=cfg["t_delay_list"])
        fpg.op_range_list_var = tk.StringVar(fpg.root, value=cfg["op_range_list"])
        fpg.maxia_list_var = tk.StringVar(fpg.root, value=cfg["maxia_list"])

        fpg.fp_calc_mode_var = tk.BooleanVar(fpg.root, value=cfg["fp_calc_mode"])
        fpg.acc_var = tk.DoubleVar(fpg.root, value=cfg["acc"])
        fpg.angle_step_var = tk.DoubleVar(fpg.root, value=cfg["angle_step"])
        fpg.num_steps_mode2_var = tk.IntVar(fpg.root, value=cfg["num_steps_mode2"])
        fpg.emode_var = tk.BooleanVar(fpg.root, value=cfg["set_shoot_look_shoot"])

        fpg.det_range_list_var = tk.StringVar(fpg.root, value=str(cfg["det_range_list"]))
        fpg.mumi_list_var = tk.StringVar(fpg.root, value=str(cfg["mumi_list"]))
        fpg.muin_list_var = tk.StringVar(fpg.root, value=str(cfg["muin_list"]))

        fpg.sect_angle_beg_var = tk.DoubleVar(fpg.root, value=cfg["sect_angle_beg"])
        fpg.sect_angle_end_var = tk.DoubleVar(fpg.root, value=cfg["sect_angle_end"])
        fpg.sect_angle_step_var = tk.DoubleVar(fpg.root, value=cfg["sect_angle_step"])
        fpg.sect_dist_beg_var = tk.DoubleVar(fpg.root, value=cfg["sect_dist_beg"])
        fpg.sect_dist_num_var = tk.IntVar(fpg.root, value=cfg["sect_dist_num"])

        fpg.gtheight_beg_var = tk.DoubleVar(fpg.root, value=cfg["gtheight_beg"])
        fpg.gtheight_end_var = tk.DoubleVar(fpg.root, value=cfg["gtheight_end"])
        fpg.gtangle_beg_var = tk.DoubleVar(fpg.root, value=cfg["gtangle_beg"])
        fpg.gtangle_end_var = tk.DoubleVar(fpg.root, value=cfg["gtangle_end"])
        fpg.maxrange_acc_var = tk.DoubleVar(fpg.root, value=cfg["maxrange_acc"])

        fpg.set_mirror_segment_var = tk.BooleanVar(fpg.root, value=cfg["set_mirror_segment"])
        fpg.plot_hit_charts_var = tk.BooleanVar(fpg.root, value=cfg["plot_hit_charts"])
        fpg.hit_chart_angle_var = tk.DoubleVar(fpg.root, value=cfg["hit_chart_angle"])

        fpg.set_keep_int_tables_var = tk.BooleanVar(fpg.root, value=cfg["set_keep_int_tables"])
        fpg.set_keep_fp_chart_var = tk.BooleanVar(fpg.root, value=cfg["set_keep_fp_chart"])
        fpg.set_keep_fp_data_var = tk.BooleanVar(fpg.root, value=cfg["set_keep_fp_data"])
        fpg.set_keep_trj_data_var = tk.BooleanVar(fpg.root, value=cfg["set_keep_trj_data"])
        fpg.stdout_to_file_var = tk.BooleanVar(fpg.root, value=cfg["stdout_to_file"])
        fpg.set_keep_stdout_file_var = tk.BooleanVar(fpg.root, value=cfg["set_keep_stdout_file"])
        fpg.set_time_stamp_var = tk.BooleanVar(fpg.root, value=cfg["set_time_stamp"])
        fpg.set_int_table_samp_verify_var = tk.BooleanVar(fpg.root, value=cfg["set_int_table_samp_verify"])
        fpg.sound_task_complete_var = tk.BooleanVar(fpg.root, value=cfg["sound_task_complete"])
        fpg.save_config_on_exit_var = tk.BooleanVar(fpg.root, value=cfg["save_config_on_exit"])
        fpg.show_extra_param_var = tk.BooleanVar(fpg.root, value=cfg["show_extra_param"])
        fpg.show_extra_procs_var = tk.BooleanVar(fpg.root, value=cfg["show_extra_procs"])
        fpg.show_ftprint_probe_var = tk.BooleanVar(fpg.root, value=cfg["show_ftprint_probe"])
        fpg.show_chart_titles_var = tk.BooleanVar(fpg.root, value=cfg["show_chart_titles"])
        fpg.def_sector_step_var = tk.DoubleVar(fpg.root, value=cfg["def_sector_step"])
        fpg.set_sat_delay_var = tk.DoubleVar(fpg.root, value=cfg["set_sat_delay"])
        fpg.set_psi_step_var = tk.DoubleVar(fpg.root, value=cfg["set_psi_step"])
        fpg.no_atmosphere_var = tk.BooleanVar(fpg.root, value=cfg["no_atmosphere"])

    def _sync_fp_gui_from_token(self, fpg, token):
        cfg = token.config
        fpg.mtype_var.set(int(cfg.get("mtype", 1)))
        fpg.itype_var.set(int(cfg.get("itype", 11)))
        fpg.mia_var.set(float(cfg.get("h_int_min", 0.0)))
        fpg.h_discr_var.set(float(cfg.get("h_discr", 0.0)))
        fpg.t_delay_var.set(float(cfg.get("t_delay", 0.0)))

        fpg.h_int_min_list_var.set(str(cfg.get("h_int_min_list", "")))
        fpg.h_discr_list_var.set(str(cfg.get("h_discr_list", "")))
        fpg.t_delay_list_var.set(str(cfg.get("t_delay_list", "")))
        fpg.op_range_list_var.set(str(cfg.get("op_range_list", "")))
        fpg.maxia_list_var.set(str(cfg.get("maxia_list", "")))

        fpg.fp_calc_mode_var.set(bool(cfg.get("fp_calc_mode", False)))
        fpg.acc_var.set(float(cfg.get("acc", 0.03)))
        fpg.angle_step_var.set(float(cfg.get("angle_step", 5.0)))
        fpg.num_steps_mode2_var.set(int(cfg.get("num_steps_mode2", 15)))
        fpg.emode_var.set(bool(cfg.get("set_shoot_look_shoot", False)))

        fpg.det_range_list_var.set(str(cfg.get("det_range_list", "")))
        if token.missile_keys:
            fpg.mumi_list_var.set(", ".join(str(k) for k in token.missile_keys))
        else:
            fpg.mumi_list_var.set(str(cfg.get("mumi_list", "")))
        fpg.muin_list_var.set(str(cfg.get("muin_list", "")))

        fpg.sect_angle_beg_var.set(float(cfg.get("sect_angle_beg", 30.0)))
        fpg.sect_angle_end_var.set(float(cfg.get("sect_angle_end", 120.0)))
        fpg.sect_angle_step_var.set(float(cfg.get("sect_angle_step", 2.0)))
        fpg.sect_dist_beg_var.set(float(cfg.get("sect_dist_beg", 50.0)))
        fpg.sect_dist_num_var.set(int(cfg.get("sect_dist_num", 30)))

        fpg.gtheight_beg_var.set(float(cfg.get("gtheight_beg", 0.0)))
        fpg.gtheight_end_var.set(float(cfg.get("gtheight_end", 0.0)))
        fpg.gtangle_beg_var.set(float(cfg.get("gtangle_beg", 0.1)))
        fpg.gtangle_end_var.set(float(cfg.get("gtangle_end", 8.0)))
        fpg.maxrange_acc_var.set(float(cfg.get("maxrange_acc", 0.01)))

        fpg.set_mirror_segment_var.set(bool(cfg.get("set_mirror_segment", True)))
        fpg.plot_hit_charts_var.set(bool(cfg.get("plot_hit_charts", False)))
        fpg.hit_chart_angle_var.set(float(cfg.get("hit_chart_angle", 180.0)))

        fpg.set_keep_int_tables_var.set(bool(cfg.get("set_keep_int_tables", True)))
        fpg.set_keep_fp_chart_var.set(bool(cfg.get("set_keep_fp_chart", True)))
        fpg.set_keep_fp_data_var.set(bool(cfg.get("set_keep_fp_data", False)))
        fpg.set_keep_trj_data_var.set(bool(cfg.get("set_keep_trj_data", True)))
        fpg.stdout_to_file_var.set(bool(cfg.get("stdout_to_file", False)))
        fpg.set_keep_stdout_file_var.set(bool(cfg.get("set_keep_stdout_file", True)))
        fpg.set_time_stamp_var.set(bool(cfg.get("set_time_stamp", False)))
        fpg.set_int_table_samp_verify_var.set(bool(cfg.get("set_int_table_samp_verify", True)))
        fpg.sound_task_complete_var.set(bool(cfg.get("sound_task_complete", True)))
        fpg.save_config_on_exit_var.set(bool(cfg.get("save_config_on_exit", True)))
        fpg.show_extra_param_var.set(bool(cfg.get("show_extra_param", True)))
        fpg.show_extra_procs_var.set(bool(cfg.get("show_extra_procs", True)))
        fpg.show_ftprint_probe_var.set(bool(cfg.get("show_ftprint_probe", True)))
        fpg.show_chart_titles_var.set(bool(cfg.get("show_chart_titles", True)))
        fpg.def_sector_step_var.set(float(cfg.get("def_sector_step", 20.0)))
        fpg.set_sat_delay_var.set(float(cfg.get("set_sat_delay", 30.0)))
        fpg.set_psi_step_var.set(float(cfg.get("set_psi_step", 0.25)))
        fpg.no_atmosphere_var.set(bool(cfg.get("no_atmosphere", False)))

    def _get_missiles(self):
        if not self.rocket_data or len(self.rocket_data) < 1:
            return []
        return sorted(self.rocket_data[0], key=lambda m: int(m.get("m_key", 0)))

    def _get_interceptors(self):
        if not self.rocket_data or len(self.rocket_data) < 2:
            return []
        return sorted(self.rocket_data[1], key=lambda m: int(m.get("i_key", m.get("m_key", 0))))

    def _get_gbewr_radars(self):
        radars = gbewr.load_gbewr_data()
        return sorted(radars, key=lambda r: int(r.get("r_key", 0)))

    def _missile_label(self, m_key):
        missiles = self._get_missiles()
        for item in missiles:
            if int(item.get("m_key", -1)) == int(m_key):
                name = item.get("type", f"m{m_key}")
                return name
        return f"m{m_key}"

    def _interceptor_label(self, i_key):
        interceptors = self._get_interceptors()
        for item in interceptors:
            key = int(item.get("i_key", item.get("m_key", -1)))
            if key == int(i_key):
                return item.get("type", f"i{i_key}")
        return f"i{i_key}"

    def _get_missile_range_km(self, m_key):
        try:
            missile_data = rd.missile(m_key, ROCKET_DATA_PATH)
        except (KeyError, AttributeError, TypeError, ValueError) as e:
            self._safe_status_update(f"Missile {m_key}: invalid data - {e}")
            return None
        if missile_data is False:
            self._safe_status_update(f"Missile {m_key} not found in rocket_data.json")
            return None
        trj = bm.balmisflight(missile_data)
        mrange = trj[len(trj) - 1, 2] * R_e
        return mrange / 1000.0

    def _destination_latlon(self, lat, lon, bearing_deg, dist_km):
        """Geodesic destination point at the given bearing and distance."""
        dist_m = dist_km * 1000.0
        bearing_deg = bearing_deg % 360.0
        if _geod is not None:
            result = _geod.Direct(lat, lon, bearing_deg, dist_m)
            lon2 = ((result["lon2"] + 180.0) % 360.0) - 180.0
            return result["lat2"], lon2
        d = dist_m / R_e
        b = math.radians(bearing_deg)
        lat1 = math.radians(lat)
        lon1 = math.radians(lon)
        lat2 = math.asin(math.sin(lat1) * math.cos(d) + math.cos(lat1) * math.sin(d) * math.cos(b))
        lon2 = lon1 + math.atan2(math.sin(b) * math.sin(d) * math.cos(lat1),
                                 math.cos(d) - math.sin(lat1) * math.sin(lat2))
        return math.degrees(lat2), ((math.degrees(lon2) + 180) % 360) - 180

    def _circle_points(self, lat, lon, radius_km, steps):
        dist_m = radius_km * 1000.0
        points = []
        if _geod is not None:
            for i in range(steps):
                bearing_deg = 360.0 * i / steps
                result = _geod.Direct(lat, lon, bearing_deg, dist_m)
                lon2 = result["lon2"]
                if lon2 > 180:
                    lon2 -= 360
                elif lon2 <= -180:
                    lon2 += 360
                points.append((result["lat2"], lon2))
        else:
            for i in range(steps):
                bearing = 2 * math.pi * i / steps
                d = dist_m / R_e
                lat1 = math.radians(lat)
                lon1 = math.radians(lon)
                lat2 = math.asin(math.sin(lat1) * math.cos(d) + math.cos(lat1) * math.sin(d) * math.cos(bearing))
                lon2 = lon1 + math.atan2(math.sin(bearing) * math.sin(d) * math.cos(lat1),
                                          math.cos(d) - math.sin(lat1) * math.sin(lat2))
                points.append((math.degrees(lat2), ((math.degrees(lon2) + 180) % 360) - 180))
        return points

    def _filter_sector_points(self, points, sector_width):
        if not points:
            return []
        half = sector_width / 2.0
        kept = []
        for x, y in points:
            angle = (math.degrees(math.atan2(x, y)) + 360.0) % 360.0
            delta = (angle + 180.0) % 360.0 - 180.0
            if abs(delta) <= half:
                kept.append((x, y))
        return kept

    def _draw_base_map(self):
        self.map_canvas.delete("base")
        w = self.map_canvas.winfo_width() or 1000
        h = self.map_canvas.winfo_height() or 600
        self.map_canvas.config(scrollregion=(0, 0, w, h))

        tiles_available = os.path.isdir(TILE_ROOT)
        if tiles_available:
            self._draw_tile_map(w, h)
            self._draw_grid_map(w, h)
            self._refresh_tokens_after_map()
            return

        if CARTOPY_AVAILABLE:
            try:
                self._draw_cartopy_map(w, h)
                self._draw_grid_map(w, h)
                self._refresh_tokens_after_map()
                return
            except Exception:
                pass

        self._draw_grid_map(w, h)
        self._refresh_tokens_after_map()

    def _raise_map_overlays(self):
        try:
            self.map_canvas.tag_raise("distance_pin")
        except tk.TclError:
            pass

    def _refresh_tokens_after_map(self):
        if not self.tokens:
            self._raise_map_overlays()
            return
        for token in self.tokens.values():
            self._update_token_draw(token)
            token_tag = f"token:{token.token_id}"
            self.map_canvas.tag_raise(token_tag)
        self._raise_map_overlays()

    def _draw_grid_map(self, w, h):
        z, world_px, left, top, view_w_px, view_h_px = self._view_world_bounds()
        if view_w_px > 3000:
            step_px = 512
        elif view_w_px > 1500:
            step_px = 256
        elif view_w_px > 800:
            step_px = 128
        else:
            step_px = 64

        start_x = math.floor(left / step_px) * step_px
        end_x = left + view_w_px
        x = start_x
        while x <= end_x:
            lon, _ = self._world_px_to_latlon(x, top, z)
            screen_x = (x - left) / view_w_px * w
            self.map_canvas.create_line(screen_x, 0, screen_x, h, fill="#ddd", tags=("base",))
            self.map_canvas.create_text(screen_x + 2, 12, text=f"{lon:.1f}", fill="#888", anchor="nw", tags=("base",))
            x += step_px

        start_y = math.floor(top / step_px) * step_px
        end_y = top + view_h_px
        y = start_y
        while y <= end_y:
            _, lat = self._world_px_to_latlon(left, y, z)
            screen_y = (y - top) / view_h_px * h
            self.map_canvas.create_line(0, screen_y, w, screen_y, fill="#ddd", tags=("base",))
            self.map_canvas.create_text(6, screen_y + 2, text=f"{lat:.1f}", fill="#888", anchor="nw", tags=("base",))
            y += step_px

    def _draw_tile_map(self, w, h):
        z, world_px, left, top, view_w_px, view_h_px = self._view_world_bounds()
        tiles_per_axis = 2 ** z
        self.tile_images = []
        tile_screen_size = max(1, int(round(TILE_SIZE / view_w_px * w)))

        tile_left = int(math.floor(left / TILE_SIZE))
        tile_right = int(math.floor((left + view_w_px) / TILE_SIZE))
        tile_top = int(math.floor(top / TILE_SIZE))
        tile_bottom = int(math.floor((top + view_h_px) / TILE_SIZE))

        for ty in range(tile_top, tile_bottom + 1):
            if ty < 0 or ty >= tiles_per_axis:
                continue
            for tx in range(tile_left, tile_right + 1):
                wrapped_x = tx % tiles_per_axis
                tile_path = os.path.join(TILE_ROOT, str(z), str(wrapped_x), f"{ty}.png")
                if not os.path.exists(tile_path):
                    continue
                img = self._load_tile_image(tile_path, tile_screen_size)
                if img is None:
                    continue
                self.tile_images.append(img)
                tile_x = tx * TILE_SIZE
                tile_y = ty * TILE_SIZE
                x0 = (tile_x - left) / view_w_px * w
                y0 = (tile_y - top) / view_h_px * h
                self.map_canvas.create_image(x0, y0, image=img, anchor="nw", tags=("base",))

    def _load_tile_image(self, tile_path, target_size):
        cache_key = (tile_path, target_size)
        if cache_key in self.tile_cache:
            return self.tile_cache[cache_key]

        if PIL_AVAILABLE:
            try:
                img = Image.open(tile_path)
                if img.size[0] != target_size:
                    try:
                        resample = Image.Resampling.BILINEAR
                    except AttributeError:
                        resample = Image.BILINEAR
                    img = img.resize((target_size, target_size), resample)
                tk_img = ImageTk.PhotoImage(img)
                self.tile_cache[cache_key] = tk_img
                return tk_img
            except Exception:
                pass

        try:
            img = tk.PhotoImage(file=tile_path)
            if target_size != TILE_SIZE:
                if target_size > TILE_SIZE:
                    factor = max(1, int(round(target_size / TILE_SIZE)))
                    img = img.zoom(factor, factor)
                else:
                    factor = max(1, int(round(TILE_SIZE / target_size)))
                    img = img.subsample(factor, factor)
            self.tile_cache[cache_key] = img
            return img
        except Exception:
            return None

    def _schedule_map_render(self):
        if self.map_render_job is not None:
            self.root.after_cancel(self.map_render_job)
        self.map_render_job = self.root.after(72, self._draw_base_map)

    def _draw_cartopy_map(self, w, h):
        dpi = 100
        fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
        # Keep fallback map projection aligned with the Web Mercator math used by
        # _latlon_to_xy/_xy_to_latlon so overlays stay scale-consistent.
        ax = plt.axes(projection=ccrs.Mercator())
        z, world_px, left, top, view_w_px, view_h_px = self._view_world_bounds()
        lon_min, lat_max = self._world_px_to_latlon(left, top, z)
        lon_max, lat_min = self._world_px_to_latlon(left + view_w_px, top + view_h_px, z)
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        used_tiles = False

        try:
            import cartopy.io.img_tiles as cimgt
            try:
                tiler = cimgt.StadiaMapsTiles("stamen_terrain_background")
            except Exception:
                tiler = cimgt.Stamen("terrain-background")
            ax.add_image(tiler, 4)
            used_tiles = True
        except Exception:
            try:
                relief = cfeature.NaturalEarthFeature(
                    "physical",
                    "shaded_relief",
                    "110m"
                )
                ax.add_feature(relief)
            except Exception:
                ax.add_feature(cfeature.OCEAN, facecolor="#b3cde3")
                ax.add_feature(cfeature.LAND, facecolor="#ccebc5")

        try:
            ax.add_feature(cfeature.BORDERS, linewidth=0.6, edgecolor="black")
        except Exception:
            pass
        try:
            ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="black")
        except Exception:
            pass
        try:
            ax.outline_patch.set_visible(False)
        except Exception:
            pass
        ax.set_axis_off()
        fig.subplots_adjust(0, 0, 1, 1)

        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp_path = tmp_file.name
        tmp_file.close()
        fig.savefig(tmp_path, pad_inches=0)
        plt.close(fig)

        try:
            self.map_image = tk.PhotoImage(file=tmp_path)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        if self.map_image_item is None:
            self.map_image_item = self.map_canvas.create_image(0, 0, image=self.map_image, anchor="nw", tags=("base",))
        else:
            self.map_canvas.itemconfigure(self.map_image_item, image=self.map_image)

    def _latlon_to_xy(self, lat, lon):
        w = self.map_canvas.winfo_width() or 1000
        h = self.map_canvas.winfo_height() or 600
        z, world_px, left, top, view_w_px, view_h_px = self._view_world_bounds()
        world_x, world_y = self._latlon_to_world_px(lat, lon, z)
        x = (world_x - left) / view_w_px * w
        y = (world_y - top) / view_h_px * h
        return x, y

    def _xy_to_latlon(self, x, y):
        w = self.map_canvas.winfo_width() or 1000
        h = self.map_canvas.winfo_height() or 600
        z, world_px, left, top, view_w_px, view_h_px = self._view_world_bounds()
        world_x = left + (x / w) * view_w_px
        world_y = top + (y / h) * view_h_px
        lon, lat = self._world_px_to_latlon(world_x, world_y, z)
        return lat, lon

    def _point_from_bearing(self, x, y, bearing_deg, length):
        theta = math.radians(bearing_deg)
        dx = length * math.sin(theta)
        dy = -length * math.cos(theta)
        return x + dx, y + dy

    def _bearing_from_token(self, token, x, y):
        cx, cy = self._latlon_to_xy(token.lat, token.lon)
        dx = x - cx
        dy = cy - y
        bearing = math.degrees(math.atan2(dx, dy)) % 360
        return bearing

    def _angle_diff(self, a, b):
        diff = (a - b + 180) % 360 - 180
        return diff

    def _offset_latlon(self, lat, lon, east_m, north_m):
        dist = math.sqrt(east_m * east_m + north_m * north_m)
        if dist < 1.0:
            return lat, lon
        if _geod is not None:
            bearing = (math.degrees(math.atan2(east_m, north_m)) + 360.0) % 360.0
            result = _geod.Direct(lat, lon, bearing, dist)
            lon2 = ((result["lon2"] + 180.0) % 360.0) - 180.0
            return result["lat2"], lon2
        dlat = north_m / R_e
        dlon = east_m / (R_e * math.cos(math.radians(lat)))
        return lat + math.degrees(dlat), lon + math.degrees(dlon)

    def _view_world_bounds(self):
        z = int(clamp(round(math.log2(self.zoom) + 4), 4, 7))
        world_px = TILE_SIZE * (2 ** z)
        w = self.map_canvas.winfo_width() or 1000
        h = self.map_canvas.winfo_height() or 600
        view_w_px = world_px / self.zoom
        view_h_px = view_w_px * (h / w)
        center_x, center_y = self._latlon_to_world_px(self.view_center_lat, self.view_center_lon, z)
        left = center_x - view_w_px / 2.0
        top = center_y - view_h_px / 2.0

        if top < 0:
            top = 0
        if top + view_h_px > world_px:
            top = max(0, world_px - view_h_px)
        return z, world_px, left, top, view_w_px, view_h_px

    def _set_view_center(self, lat, lon):
        self.view_center_lat = clamp(lat, -85, 85)
        self.view_center_lon = ((lon + 180) % 360) - 180

    def _latlon_to_world_px(self, lat, lon, z):
        lat = clamp(lat, -85.05112878, 85.05112878)
        world_px = TILE_SIZE * (2 ** z)
        meters_per_px = 2 * math.pi * R_e / world_px
        lon_rad = math.radians(lon)
        lat_rad = math.radians(lat)
        x_m = R_e * lon_rad
        y_m = R_e * math.log(math.tan(math.pi / 4 + lat_rad / 2))
        x = (x_m + math.pi * R_e) / meters_per_px
        y = (math.pi * R_e - y_m) / meters_per_px
        return x, y

    def _world_px_to_latlon(self, x, y, z):
        world_px = TILE_SIZE * (2 ** z)
        meters_per_px = 2 * math.pi * R_e / world_px
        x_m = x * meters_per_px - math.pi * R_e
        y_m = math.pi * R_e - y * meters_per_px
        lon = math.degrees(x_m / R_e)
        lat = math.degrees(2 * math.atan(math.exp(y_m / R_e)) - math.pi / 2)
        return lon, lat


class BuilderWindow:
    """Interactive missile / interceptor builder: edit physics specs and see the
    trajectory, speed and ground-range update live (drag the trajectory to change the
    launch angle, or the impact marker to set the range). Autofill from a text
    description via AIP, then Save & Inject into the rocket library."""

    # per-stage columns (each becomes a comma-separated entry in the rocket data)
    STAGE_COLS = [
        ("m_st", "Mass kg"), ("m_fu", "Fuel kg"), ("v_ex", "V_exh m/s"),
        ("t_bu", "Burn s"), ("a_mid", "Body area"), ("t_delay", "Coast s"),
    ]
    # single-valued (non-stage) fields
    SINGLE_FIELDS = [
        ("cd_type", "Drag model (cd_type)"),
        ("a_nz", "Nozzle area (a_nz)"),
        ("c_bal", "Ballistic coeff (c_bal)"),
        ("m_pl", "Payload kg (m_pl)"),
        ("m_warhead", "Warhead kg (m_warhead)"),
        ("m_shroud", "Shroud kg (m_shroud)"),
        ("t_shroud", "Shroud release s (t_shroud)"),
        ("vert_launch_height", "Vert launch h m"),
        ("range", "Range label km (range)"),
    ]
    INTERCEPTOR_FIELDS = [
        ("op_range", "Op range km (op_range)"),
        ("mpia", "Min intercept alt km (mpia)"),
        ("maxia", "Max intercept alt km (maxia)"),
        ("det_range", "Detection range (det_range)"),
    ]
    STAGE_DEFAULTS = {"m_st": "1330", "m_fu": "7400", "v_ex": "2254.0",
                      "t_bu": "127.8", "a_mid": "d 1", "t_delay": ""}
    NEW_STAGE = {"m_st": "300", "m_fu": "1500", "v_ex": "2600", "t_bu": "40",
                 "a_mid": "d 0.6", "t_delay": "0"}
    DEFAULTS = {
        "type": "New missile", "cd_type": "v2", "a_nz": "0.135", "c_bal": "7320",
        "m_pl": "500.0", "m_warhead": "", "m_shroud": "", "t_shroud": "",
        "vert_launch_height": "100", "range": "1000",
        "grav_turn_angle": "0.7", "flight_path_angle": "40", "traj_type": "bal_mis",
        "op_range": "", "mpia": "", "maxia": "", "det_range": "",
    }
    INTERCEPTOR_DEFAULTS = {  # purpose-built interceptor defaults
        "type": "New interceptor", "cd_type": "ls", "a_nz": "0", "c_bal": "14300",
        "m_pl": "150.0", "m_warhead": "", "m_shroud": "0", "t_shroud": "0",
        "vert_launch_height": "0", "range": "100",
        "grav_turn_angle": "0.7", "flight_path_angle": "40", "traj_type": "int_endo",
        "op_range": "100", "mpia": "1", "maxia": "100", "det_range": "300",
    }
    INTERCEPTOR_STAGE = {"m_st": "0", "m_fu": "80", "v_ex": "2800",
                         "t_bu": "2", "a_mid": "0.042", "t_delay": ""}
    # atmospheric layers (alt_lo_km, alt_hi_km, name, fill color)
    ATM_LAYERS = [
        (0, 12, "Troposphere", "#cdebff"),
        (12, 50, "Stratosphere", "#dcd2ff"),
        (50, 85, "Mesosphere", "#ffe6bf"),
        (85, 600, "Thermosphere", "#ffd4d4"),
    ]
    KARMAN_KM = 100.0

    def __init__(self, app):
        self.app = app
        self.win = tk.Toplevel(app.root)
        self.win.title("Builder — missile / interceptor")
        self.win.configure(bg=THEME["panel_bg"])
        self.win.geometry("1180x760")
        self.vars = {}
        self._recompute_job = None
        self._drag = None        # active matplotlib drag: "angle" or "range"
        self._last_trj = None
        self._slider_guard = False
        self.win.protocol("WM_DELETE_WINDOW", self._close)
        self._build_ui()
        self._recompute()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        left = tk.Frame(self.win, bg=THEME["panel_bg"], width=360)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)
        right = tk.Frame(self.win, bg=THEME["panel_bg"])
        right.pack(side="left", fill="both", expand=True)

        tk.Label(left, text="Builder", bg=THEME["panel_bg"], fg=THEME["fg"],
                 font=("TkDefaultFont", 13, "bold")).pack(anchor="w", padx=10, pady=(10, 2))

        # kind + trajectory type
        kind_row = tk.Frame(left, bg=THEME["panel_bg"]); kind_row.pack(fill="x", padx=10, pady=2)
        self.kind_var = tk.StringVar(value="missile")
        for txt, val in (("Missile", "missile"), ("Interceptor", "interceptor")):
            ttk.Radiobutton(kind_row, text=txt, value=val, variable=self.kind_var,
                            command=lambda: self._on_kind_change(apply_defaults=True)).pack(side="left", padx=(0, 8))
        tt_row = tk.Frame(left, bg=THEME["panel_bg"]); tt_row.pack(fill="x", padx=10, pady=2)
        tk.Label(tt_row, text="Trajectory:", bg=THEME["panel_bg"], fg=THEME["fg"]).pack(side="left")
        self.vars["traj_type"] = tk.StringVar(value=self.DEFAULTS["traj_type"])
        self.tt_combo = ttk.Combobox(tt_row, values=["bal_mis"],
                                     textvariable=self.vars["traj_type"], state="readonly", width=12)
        self.tt_combo.pack(side="left", padx=6)
        self.tt_combo.bind("<<ComboboxSelected>>", lambda e: (self._sync_angle_label(), self._schedule_recompute()))

        # scrollable spec fields
        sf = ScrollableFrame(left, width=340)
        sf.pack(fill="both", expand=True, padx=4, pady=4)
        host = sf.inner
        # launch angle (slider + entry) -- the trajectory-shaping DOF
        self.vars["grav_turn_angle"] = tk.StringVar(value=self.DEFAULTS["grav_turn_angle"])
        self.vars["flight_path_angle"] = tk.StringVar(value=self.DEFAULTS["flight_path_angle"])
        ang_frame = tk.Frame(host, bg=THEME["panel_inner"]); ang_frame.pack(fill="x", padx=6, pady=3)
        self.angle_label = tk.Label(ang_frame, text="Launch angle", width=22, anchor="w",
                                    bg=THEME["panel_inner"], fg=THEME["fg"])
        self.angle_label.pack(side="left")
        self.angle_entry_var = tk.StringVar(value=self.DEFAULTS["grav_turn_angle"])
        ae = ttk.Entry(ang_frame, textvariable=self.angle_entry_var, width=8)
        ae.pack(side="left")
        self.angle_entry_var.trace_add("write", lambda *a: self._on_angle_entry())
        self.angle_slider = ttk.Scale(host, from_=0.2, to=80.0, orient="horizontal",
                                      command=self._on_angle_slider)
        self.angle_slider.pack(fill="x", padx=8, pady=(0, 6))
        self.angle_slider.set(float(self.DEFAULTS["grav_turn_angle"]))
        self._slider_guard = False

        # name
        self._add_field(host, "type", "Name")

        # ---- interactive multi-stage builder ----
        self._build_stage_section(host)

        # airframe / payload single fields
        self.airframe_header = tk.Label(host, text="— Airframe / payload —", bg=THEME["panel_inner"],
                                        fg=THEME["fg_muted"], font=("TkDefaultFont", 9, "bold"))
        self.airframe_header.pack(fill="x", padx=6, pady=(8, 2))
        for key, label in self.SINGLE_FIELDS:
            self._add_field(host, key, label)

        # interceptor-only interception parameters
        self.intercept_header = tk.Label(host, text="— Interception —", bg=THEME["panel_inner"],
                                         fg=THEME["fg_muted"], font=("TkDefaultFont", 9, "bold"))
        self._interceptor_field_frames = []
        for key, label in self.INTERCEPTOR_FIELDS:
            fr = self._add_field(host, key, label)
            self._interceptor_field_frames.append(fr)

        # AIP autofill
        tk.Label(left, text="Describe a missile → AIP autofill:", bg=THEME["panel_bg"],
                 fg=THEME["fg"], font=("TkDefaultFont", 9, "bold")).pack(anchor="w", padx=10, pady=(6, 0))
        self.aip_text = tk.Text(left, height=3, width=40, wrap="word")
        self.aip_text.pack(fill="x", padx=10, pady=2)
        btns = tk.Frame(left, bg=THEME["panel_bg"]); btns.pack(fill="x", padx=10, pady=4)
        self.app._lib_btn(btns, "Autofill from description", self._autofill_aip).pack(side="left", padx=(0, 6))
        self.app._lib_btn(btns, "Save & Inject", self._save_inject, primary=True).pack(side="left")

        self.status = tk.Label(left, text="", bg=THEME["panel_bg"], fg=THEME["fg_muted"],
                               wraplength=340, justify="left", font=("TkDefaultFont", 9))
        self.status.pack(anchor="w", padx=10, pady=(4, 8))

        # charts
        self.fig = Figure(figsize=(7.2, 7.0), facecolor="#f7f7fb")
        gs = self.fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
        self.ax_traj = self.fig.add_subplot(gs[0, 0])
        self.ax_speed = self.fig.add_subplot(gs[0, 1])
        self.ax_map = self.fig.add_subplot(gs[1, :])
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.mpl_connect("button_press_event", self._on_mpl_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_mpl_motion)
        self.canvas.mpl_connect("button_release_event", self._on_mpl_release)
        self._on_kind_change()
        self._sync_angle_label()

    def _add_field(self, host, key, label):
        fr = tk.Frame(host, bg=THEME["panel_inner"]); fr.pack(fill="x", padx=6, pady=2)
        tk.Label(fr, text=label, width=22, anchor="w", bg=THEME["panel_inner"], fg=THEME["fg"],
                 font=("TkDefaultFont", 9)).pack(side="left")
        var = tk.StringVar(value=self.DEFAULTS.get(key, ""))
        self.vars[key] = var
        ent = ttk.Entry(fr, textvariable=var, width=14); ent.pack(side="left", fill="x", expand=True)
        var.trace_add("write", lambda *a: self._schedule_recompute())
        return fr

    # ------------------------------------------------------------- stages
    def _build_stage_section(self, host):
        self.stage_vars = []          # list (per stage) of {col: StringVar}
        self._stage_defaults = dict(self.STAGE_DEFAULTS)
        hdr = tk.Frame(host, bg=THEME["panel_inner"]); hdr.pack(fill="x", padx=6, pady=(8, 0))
        tk.Label(hdr, text="Stages", bg=THEME["panel_inner"], fg=THEME["fg"],
                 font=("TkDefaultFont", 10, "bold")).pack(side="left")
        self.app._lib_btn(hdr, "−", self._remove_stage).pack(side="right", padx=2)
        self.stage_count_lbl = tk.Label(hdr, text="1", width=2, bg=THEME["panel_inner"], fg=THEME["fg"])
        self.stage_count_lbl.pack(side="right")
        self.app._lib_btn(hdr, "+", self._add_stage).pack(side="right", padx=2)
        # column headers
        self.stage_table = tk.Frame(host, bg=THEME["panel_inner"]); self.stage_table.pack(fill="x", padx=6)
        # generated rocket-data strings (read-only, shows exactly what gets saved)
        self.stage_summary = tk.Label(host, text="", bg=THEME["panel_inner"], fg=THEME["fg_muted"],
                                      font=("TkDefaultFont", 8), justify="left", anchor="w", wraplength=320)
        self.stage_summary.pack(fill="x", padx=6, pady=(2, 4))
        # seed one stage from defaults
        self.stage_vars.append({ck: tk.StringVar(value=self._stage_defaults.get(ck, "")) for ck, _ in self.STAGE_COLS})
        self._rebuild_stage_rows()

    def _rebuild_stage_rows(self):
        for w in self.stage_table.winfo_children():
            w.destroy()
        n = len(self.stage_vars)
        multi = n > 1
        # header
        head = tk.Frame(self.stage_table, bg=THEME["panel_inner"]); head.grid(row=0, column=0, columnspan=7, sticky="w")
        tk.Label(head, text="#", width=2, bg=THEME["panel_inner"], fg=THEME["fg_muted"],
                 font=("TkDefaultFont", 8)).grid(row=0, column=0)
        cols = self.STAGE_COLS if multi else [c for c in self.STAGE_COLS if c[0] != "t_delay"]
        for ci, (ck, clabel) in enumerate(cols):
            tk.Label(head, text=clabel, width=8, bg=THEME["panel_inner"], fg=THEME["fg_muted"],
                     font=("TkDefaultFont", 8)).grid(row=0, column=ci + 1)
        for si in range(n):
            row = tk.Frame(self.stage_table, bg=THEME["panel_inner"]); row.grid(row=si + 1, column=0, sticky="w", pady=1)
            tk.Label(row, text=str(si + 1), width=2, bg=THEME["panel_inner"], fg=THEME["fg"],
                     font=("TkDefaultFont", 8)).grid(row=0, column=0)
            for ci, (ck, _l) in enumerate(cols):
                var = self.stage_vars[si][ck]
                ent = ttk.Entry(row, textvariable=var, width=8)
                ent.grid(row=0, column=ci + 1, padx=1)
                if not getattr(var, "_traced", False):
                    var.trace_add("write", lambda *a: self._on_stage_change())
                    var._traced = True
        self.stage_count_lbl.config(text=str(n))
        self._on_stage_change()

    def _add_stage(self):
        if len(self.stage_vars) >= 4:
            return
        base = self.INTERCEPTOR_STAGE if self.kind_var.get() == "interceptor" else self.NEW_STAGE
        self.stage_vars.append({ck: tk.StringVar(value=base.get(ck, "")) for ck, _ in self.STAGE_COLS})
        self._rebuild_stage_rows()

    def _remove_stage(self):
        if len(self.stage_vars) <= 1:
            return
        self.stage_vars.pop()
        self._rebuild_stage_rows()

    def _on_stage_change(self):
        # refresh the generated rocket-data summary and recompute
        s = self._stage_strings()
        self.stage_summary.config(text="m_st: {m_st}\nm_fu: {m_fu}\nv_ex: {v_ex}\nt_bu: {t_bu}\n"
                                       "a_mid: {a_mid}".format(**s)
                                  + (("\nt_delay: " + s["t_delay"]) if len(self.stage_vars) > 1 else ""))
        self._schedule_recompute()

    def _stage_strings(self):
        """Build the comma-separated rocket-data strings from the stage table."""
        n = len(self.stage_vars)
        out = {}
        for ck, _ in self.STAGE_COLS:
            vals = [self.stage_vars[i][ck].get().strip() for i in range(n)]
            if ck == "t_delay":
                if n == 1 or all(v in ("", "0", "0.0") for v in vals[:-1]):
                    out[ck] = ""
                    continue
                vals = [v if v != "" else "0" for v in vals]
            else:
                vals = [v if v != "" else "0" for v in vals]
            out[ck] = vals[0] if n == 1 else ", ".join(vals)
        return out

    def _set_stages_from_strings(self, specs):
        """Parse comma-separated rocket-data strings (e.g. from AIP) back into stage rows."""
        def split(key):
            raw = str(specs.get(key, "")).split("#", 1)[0]
            return [p.strip() for p in raw.split(",")] if raw.strip() else []
        counts = [len(split(k)) for k in ("m_st", "m_fu", "v_ex", "t_bu") if split(k)]
        n = max(counts) if counts else 1
        n = max(1, min(4, n))
        self.stage_vars = []
        for i in range(n):
            d = {}
            for ck, _ in self.STAGE_COLS:
                parts = split(ck)
                d[ck] = tk.StringVar(value=(parts[i] if i < len(parts) else self.STAGE_DEFAULTS.get(ck, "")))
            self.stage_vars.append(d)
        self._rebuild_stage_rows()

    def _on_kind_change(self, apply_defaults=False):
        """Reconfigure the form for a purpose-built missile vs interceptor view."""
        is_int = self.kind_var.get() == "interceptor"
        # trajectory-type choices differ by kind
        self.tt_combo.config(values=(["int_endo", "int_exo"] if is_int else ["bal_mis"]))
        if is_int and self.vars["traj_type"].get() == "bal_mis":
            self.vars["traj_type"].set("int_endo")
        elif not is_int and self.vars["traj_type"].get() != "bal_mis":
            self.vars["traj_type"].set("bal_mis")
        # optionally reset spec defaults to the kind's purpose-built values
        if apply_defaults:
            dflt = self.INTERCEPTOR_DEFAULTS if is_int else self.DEFAULTS
            for k in self.vars:
                if k in dflt:
                    self.vars[k].set(dflt[k])
            base = self.INTERCEPTOR_STAGE if is_int else self.STAGE_DEFAULTS
            self.stage_vars = [{ck: tk.StringVar(value=base.get(ck, "")) for ck, _ in self.STAGE_COLS}]
            self._rebuild_stage_rows()
        # show/hide the interception section
        self.intercept_header.pack_forget()
        for fr in getattr(self, "_interceptor_field_frames", []):
            fr.pack_forget()
        if is_int:
            self.intercept_header.pack(fill="x", padx=6, pady=(8, 2))
            for fr in self._interceptor_field_frames:
                fr.pack(fill="x", padx=6, pady=2)
        self.win.title("Builder — " + ("interceptor" if is_int else "missile"))
        self._sync_angle_label()
        self._schedule_recompute()

    # --------------------------------------------------------------- angle
    def _angle_key(self):
        return "grav_turn_angle" if self.vars["traj_type"].get() == "bal_mis" else "flight_path_angle"

    def _sync_angle_label(self):
        ak = self._angle_key()
        self.angle_label.config(text="Gravity-turn angle°" if ak == "grav_turn_angle" else "Flight-path angle°")
        self._slider_guard = True
        try:
            self.angle_entry_var.set(self.vars[ak].get())
            self.angle_slider.set(max(0.2, min(80.0, float(self.vars[ak].get() or 0))))
        except (ValueError, tk.TclError):
            pass
        self._slider_guard = False

    def _on_angle_entry(self):
        if self._slider_guard:
            return
        ak = self._angle_key()
        self.vars[ak].set(self.angle_entry_var.get())
        try:
            self._slider_guard = True
            self.angle_slider.set(max(0.2, min(80.0, float(self.angle_entry_var.get() or 0))))
        except (ValueError, tk.TclError):
            pass
        finally:
            self._slider_guard = False
        self._schedule_recompute()

    def _on_angle_slider(self, val):
        if self._slider_guard:
            return
        ak = self._angle_key()
        sval = f"{float(val):.2f}"
        self._slider_guard = True
        self.angle_entry_var.set(sval)
        self._slider_guard = False
        self.vars[ak].set(sval)
        self._schedule_recompute()

    def _set_angle(self, deg):
        deg = max(0.2, min(80.0, float(deg)))
        ak = self._angle_key()
        self.vars[ak].set(f"{deg:.2f}")
        self._slider_guard = True
        self.angle_entry_var.set(f"{deg:.2f}")
        try:
            self.angle_slider.set(deg)
        except tk.TclError:
            pass
        self._slider_guard = False

    # --------------------------------------------------------------- compute
    def _string_specs(self):
        d = dict(self._stage_strings())
        d["type"] = self.vars["type"].get()
        for key, _ in self.SINGLE_FIELDS:
            d[key] = self.vars[key].get()
        d["traj_type"] = self.vars["traj_type"].get()
        d["grav_turn_angle"] = self.vars["grav_turn_angle"].get() or "0"
        d["flight_path_angle"] = self.vars["flight_path_angle"].get() or "0"
        d["note"] = ""
        if self.kind_var.get() == "interceptor":
            for key, _ in self.INTERCEPTOR_FIELDS:
                d[key] = self.vars[key].get()
        return d

    def _schedule_recompute(self):
        if self._recompute_job is not None:
            try:
                self.win.after_cancel(self._recompute_job)
            except tk.TclError:
                pass
        self._recompute_job = self.win.after(180, self._recompute)

    def _recompute(self):
        self._recompute_job = None
        specs = self._string_specs()
        try:
            numeric = dict(specs)
            numeric["m_key" if self.kind_var.get() == "missile" else "i_key"] = 0
            numeric = rd.str2num(numeric)
            trj = bm.balmisflight(numeric)
            if not np.any(trj):
                raise ValueError("no trajectory (check propulsion/angle)")
            self._last_trj = trj
            self._draw_charts(trj, numeric)
            rng = trj[len(trj) - 1, 2] * R_e / 1000.0
            apo = (max(trj[:, 1]) - R_e) / 1000.0
            vmax = max(trj[:, 3]) / 1000.0
            self.status.config(text=f"range {rng:.0f} km · apogee {apo:.0f} km · v_max {vmax:.2f} km/s · flight {trj[-1,0]:.0f} s",
                               fg=THEME["fg_muted"])
        except Exception as exc:
            self.status.config(text=f"Spec error: {exc}", fg="#dc2626")

    def _stage_burnout_times(self, numeric):
        tbu = numeric.get("t_bu", []); tdel = numeric.get("t_delay", [])
        if isinstance(tbu, (int, float)): tbu = [tbu]
        if isinstance(tdel, (int, float)): tdel = [tdel]
        times, acc = [], 0.0
        for i in range(len(tbu)):
            acc += tbu[i]
            times.append(acc)
            acc += tdel[i] if i < len(tdel) else 0.0
        return times

    def _draw_atmosphere(self, ax, ymax):
        for lo, hi, name, color in self.ATM_LAYERS:
            if lo > ymax:
                break
            top = min(hi, ymax)
            ax.axhspan(lo, top, color=color, alpha=0.45, zorder=0)
            if top - lo > ymax * 0.04:
                ax.text(0.012, (lo + top) / 2, name, transform=ax.get_yaxis_transform(),
                        fontsize=6.5, va="center", color="#444", zorder=1)
        if ymax >= self.KARMAN_KM * 0.6:
            ax.axhline(self.KARMAN_KM, color="#111", ls=":", lw=1, zorder=1)
            ax.text(0.99, self.KARMAN_KM, "Kármán 100 km", transform=ax.get_yaxis_transform(),
                    fontsize=6.5, ha="right", va="bottom", color="#111", zorder=1)

    def _draw_charts(self, trj, numeric):
        is_int = self.kind_var.get() == "interceptor"
        t = trj[:, 0]
        alt = (trj[:, 1] - R_e) / 1000.0
        grnd = trj[:, 2] * R_e / 1000.0
        spd = trj[:, 3] / 1000.0
        rng = grnd[-1]
        for ax in (self.ax_traj, self.ax_speed, self.ax_map):
            ax.clear()
        ymax = max(alt.max() * 1.1, 12)

        # --- trajectory + atmospheric layers ---
        self._draw_atmosphere(self.ax_traj, ymax)
        self.ax_traj.plot(grnd, alt, "-", color="#dc2626", lw=2, zorder=4)
        # interceptor: shade the intercept-altitude window (mpia..maxia)
        if is_int:
            try:
                mpia = float(self.vars["mpia"].get() or 0)
                maxia = float(self.vars["maxia"].get() or 0)
                if maxia > mpia >= 0:
                    self.ax_traj.axhspan(mpia, maxia, color="#16a34a", alpha=0.18, zorder=2)
                    self.ax_traj.text(0.5, (mpia + maxia) / 2, "intercept window",
                                      transform=self.ax_traj.get_yaxis_transform(), fontsize=6.5,
                                      ha="center", va="center", color="#166534", zorder=3)
            except ValueError:
                pass
        # staging markers
        burns = self._stage_burnout_times(numeric)
        for si, bt in enumerate(burns):
            k = int(np.searchsorted(t, bt))
            if 0 < k < len(grnd):
                self.ax_traj.plot([grnd[k]], [alt[k]], marker="^", ms=7, color="#111", zorder=5)
                if len(burns) > 1:
                    self.ax_traj.annotate(f"S{si+1}", (grnd[k], alt[k]), fontsize=6.5,
                                          textcoords="offset points", xytext=(3, 3), zorder=5)
        self.ax_traj.set_ylim(0, ymax)
        self.ax_traj.set_title("Trajectory + atmosphere (drag ↕ = launch angle)", fontsize=9)
        self.ax_traj.set_xlabel("ground range (km)", fontsize=8)
        self.ax_traj.set_ylabel("altitude (km)", fontsize=8)
        self.ax_traj.grid(alpha=0.25)

        # --- speed (+ staging) ---
        self.ax_speed.plot(t, spd, "-", color="#2563eb", lw=2)
        for si, bt in enumerate(burns):
            self.ax_speed.axvline(bt, color="#111", ls="--", lw=0.8, alpha=0.6)
            if len(burns) > 1:
                self.ax_speed.annotate(f"S{si+1}", (bt, spd.max()), fontsize=6.5,
                                       textcoords="offset points", xytext=(2, -2))
        self.ax_speed.set_title("Speed (dashed = staging)", fontsize=9)
        self.ax_speed.set_xlabel("time (s)", fontsize=8)
        self.ax_speed.set_ylabel("speed (km/s)", fontsize=8)
        self.ax_speed.grid(alpha=0.3)

        # --- top-down ground map (purpose-built per kind) ---
        th = np.linspace(0, 2 * np.pi, 120)
        self.ax_map.plot([0], [0], marker="*", ms=14, color="#16a34a")
        self.ax_map.plot(rng * np.cos(th), rng * np.sin(th), "--", color="#7c3aed", lw=1.2,
                         label="reach" if is_int else "max range")
        self.ax_map.plot([0, rng], [0, 0], "-", color="#444", lw=1.2)
        self._impact_pt, = self.ax_map.plot([rng], [0], marker="o", ms=10, color="#f59e0b",
                                            markeredgecolor="#7c2d12")
        rings_max = rng
        if is_int:
            for key, col, lbl in (("op_range", "#0891b2", "op range"), ("det_range", "#16a34a", "det range")):
                try:
                    raw = str(self.vars[key].get()).split(",")[0].split(":")[-1].strip()
                    val = float(raw)
                    if val <= 0:
                        continue
                    if val < 2000:  # km in the data convention
                        pass
                    self.ax_map.plot(val * np.cos(th), val * np.sin(th), ":", color=col, lw=1.4, label=lbl)
                    rings_max = max(rings_max, val)
                except (ValueError, IndexError):
                    pass
            self.ax_map.legend(loc="upper right", fontsize=6)
            self.ax_map.set_title(f"Interceptor reach {rng:.0f} km (drag ● = range)", fontsize=9)
        else:
            self.ax_map.set_title(f"Ground map (drag ● to set range) — range {rng:.0f} km", fontsize=9)
        self.ax_map.set_xlabel("km", fontsize=8)
        self.ax_map.set_aspect("equal", adjustable="datalim")
        lim = max(rings_max * 1.15, 10)
        self.ax_map.set_xlim(-lim, lim); self.ax_map.set_ylim(-lim, lim)
        self.ax_map.grid(alpha=0.3)
        self.canvas.draw_idle()

    # --------------------------------------------------------- mpl drag
    def _on_mpl_press(self, event):
        if event.inaxes is self.ax_traj:
            self._drag = ("angle", event.y)
        elif event.inaxes is self.ax_map and event.xdata is not None:
            self._drag = ("range", None)

    def _on_mpl_motion(self, event):
        if self._drag is None or event.inaxes is None:
            return
        kind = self._drag[0]
        if kind == "angle" and event.inaxes is self.ax_traj:
            # vertical pixel drag -> change launch angle (drag up = more lofted)
            dy = event.y - self._drag[1]
            self._drag = ("angle", event.y)
            ak = self._angle_key()
            try:
                cur = float(self.vars[ak].get() or 0)
            except ValueError:
                cur = 10.0
            # lofted means smaller grav_turn_angle (from vertical) / larger flight_path_angle (from horizontal)
            sign = -1.0 if ak == "grav_turn_angle" else 1.0
            self._set_angle(cur + sign * dy * 0.15)
            self._recompute()
        elif kind == "range" and event.inaxes is self.ax_map and event.xdata is not None:
            r = math.hypot(event.xdata, event.ydata)
            if self._impact_pt is not None:
                self._impact_pt.set_data([r], [0])
                self.ax_map.set_title(f"Ground map — target {r:.0f} km (release to solve)", fontsize=9)
                self.canvas.draw_idle()
            self._drag = ("range", r)

    def _on_mpl_release(self, event):
        if self._drag is not None and self._drag[0] == "range" and self._drag[1]:
            self._solve_angle_for_range(self._drag[1] * 1000.0)
        self._drag = None

    def _solve_angle_for_range(self, target_m):
        """Set the launch angle so the missile reaches target_m ground range, using the
        depressed-side launch-angle -> range table (same machinery as the footprint solver)."""
        specs = self._string_specs()
        kkey = "m_key" if self.kind_var.get() == "missile" else "i_key"
        try:
            s = dict(specs); s[kkey] = 0
            numeric = rd.str2num(s)
            tables = ss.build_mlp_tables(numeric, loft=False)
            tr, ta = tables["tbl_ranges"], tables["tbl_angles"]
            angle = float(np.interp(max(1.0, min(target_m, tables["max_range"])), tr, ta))
            self._set_angle(angle)
            self._recompute()
        except Exception as exc:
            self.status.config(text=f"Range solve failed: {exc}", fg="#dc2626")

    # --------------------------------------------------------- AIP
    def _autofill_aip(self):
        desc = self.aip_text.get("1.0", "end").strip()
        if not desc:
            self.status.config(text="Type a description first.", fg="#dc2626")
            return
        self.status.config(text="Querying AIP…", fg=THEME["fg_muted"])
        threading.Thread(target=self._aip_worker, args=(desc,), daemon=True).start()

    def _aip_worker(self, desc):
        try:
            import using_aip
            result = using_aip.query_missile(desc)
        except Exception as exc:
            self.win.after(0, lambda: self.status.config(text=f"AIP error: {exc}", fg="#dc2626"))
            return
        self.win.after(0, lambda: self._apply_aip(result))

    def _apply_aip(self, result):
        if not isinstance(result, dict):
            self.status.config(text=f"AIP returned: {result}", fg="#dc2626")
            return
        stage_keys = {ck for ck, _ in self.STAGE_COLS}
        applied = []
        # let traj_type apply first so kind-specific fields land correctly
        if result.get("traj_type"):
            tt = str(result["traj_type"])
            self.kind_var.set("interceptor" if tt.startswith("int") else "missile")
            self._on_kind_change(apply_defaults=False)
            self.vars["traj_type"].set(tt); applied.append("traj_type")
        stage_overrides = {k: v for k, v in result.items() if k in stage_keys}
        if stage_overrides:
            # apply AIP stage values onto stage 1 (AIP returns scalars)
            for k, v in stage_overrides.items():
                self.stage_vars[0][k].set(str(v)); applied.append(k)
            self._rebuild_stage_rows()
        for k, v in result.items():
            if k in ("traj_type",) or k in stage_keys:
                continue
            if k in self.vars:
                self.vars[k].set(str(v)); applied.append(k)
        self._sync_angle_label()
        self.status.config(text=f"AIP filled: {', '.join(applied) or 'nothing matched'}", fg=THEME["fg_muted"])
        self._recompute()

    # --------------------------------------------------------- save
    def _save_inject(self):
        is_missile = self.kind_var.get() == "missile"
        key_name = "m_key" if is_missile else "i_key"
        group = self.app.rocket_data[0] if is_missile else self.app.rocket_data[1]
        new_key = max([int(it.get(key_name, 0)) for it in group], default=0) + 1
        entry = {key_name: new_key}
        entry.update(self._stage_strings())          # m_st, m_fu, v_ex, t_bu, a_mid, t_delay
        entry["type"] = self.vars["type"].get()
        for key, _ in self.SINGLE_FIELDS:
            entry[key] = self.vars[key].get()
        entry["traj_type"] = self.vars["traj_type"].get()
        entry[self._angle_key()] = self.vars[self._angle_key()].get()
        entry["note"] = ""
        if not is_missile:
            for key, _ in self.INTERCEPTOR_FIELDS:
                entry[key] = self.vars[key].get()
        # validate it parses + flies before injecting
        try:
            test = dict(entry)
            num = rd.str2num(test)
            if not np.any(bm.balmisflight(num)):
                raise ValueError("trajectory failed")
        except Exception as exc:
            messagebox.showerror("Save & Inject", f"Specs don't produce a valid trajectory:\n{exc}")
            return
        group.append(entry)
        save_rocket_data(self.app.rocket_data)
        self.app._missile_range_cache.clear()
        kind = "missile" if is_missile else "interceptor"
        prefix = "m" if is_missile else "i"
        self.status.config(text=f"Injected {prefix}{new_key} '{entry['type']}' into the rocket library.",
                           fg="#16a34a")
        self.app._set_status(f"Builder: injected {kind} {prefix}{new_key} '{entry['type']}'")

    def _close(self):
        try:
            self.win.destroy()
        except tk.TclError:
            pass
        self.app._builder_win = None


if __name__ == "__main__":
    root = tk.Tk()
    app = MapApp(root)
    root.mainloop()
