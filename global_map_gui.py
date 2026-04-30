import json
import math
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


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "fcc_config.json")
TILE_ROOT = os.path.join(BASE_DIR, "world_tiles")
TILE_SIZE = 256
ROCKET_DATA_PATH = os.path.join(BASE_DIR, "rocket_data.json")
GBEWR_DATA_PATH = os.path.join(BASE_DIR, "gbewr_data.json")
MIN_ZOOM = 1.0
MAX_ZOOM = 128.0

THEME = {
    "root_bg": "#1a1b26",
    "panel_bg": "#24283b",
    "panel_inner": "#1f2335",
    "titlebar": "#2b3047",
    "toolbar_bg": "#24283b",
    "border": "#414868",
    "fg": "#c0caf5",
    "fg_muted": "#9aa5ce",
    "accent": "#7aa2f7",
    "accent_dim": "#5a7cc9",
    "success": "#9ece6a",
    "danger": "#f7768e",
    "warning": "#e0af68",
    "entry_bg": "#16161e",
    "list_bg": "#16161e",
    "list_fg": "#c0caf5",
    "select_bg": "#414868",
    "map_stripe": "#161620",
    # Light fills + dark labels: readable on macOS where tk.Button often renders as a light native chip
    "tool_btn_bg": "#c5d0e8",
    "tool_btn_fg": "#0f172a",
    "tool_btn_hover": "#aab9d9",
    "tool_btn_active_bg": "#8fa4c8",
    "tool_btn_active_fg": "#020617",
}

HELP = {
    "win_title": "Plan interceptor launch points (ILPs), missile launch points (MLPs), and ground-based early-warning radars on the world map.",
    "library": "Edit missile types, interceptor types, and radar presets used by tokens. Data is stored in rocket_data.json and gbewr_data.json.",
    "inspector": "When you select a unit on the map, its position, footprint options, and configuration appear here. Click Apply to save edits to the live token.",
    "check_coverage": "Walkthrough: pick an ILP and MLP (by name or by clicking the map), then click a target point. Shows whether that point lies inside the computed footprint.",
    "measure_distance": "Click two points (existing units or anywhere on the map) to measure geodesic distance in kilometers.",
    "map_pan": "Drag with the left mouse button on empty map to pan. Right-drag also pans. Drag units or handles to move or aim.",
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
    "det_override": "Use a fixed detection range (km) instead of the native interceptor detection range. Mutually exclusive with space-based override.",
    "space_delay": "Override native detection with space-based cueing. Set delay in seconds (default 30). Mutually exclusive with fixed detection-range override.",
    "range_override": "Override the threat’s maximum range from rocket data for footprint math only.",
    "op_range_override": "For footprint display only: replace the interceptor’s operational range from rocket_data (km). Use for terminal-phase or notional envelope studies; does not change the interceptor trajectory table.",
    "assign_gbewr": "Early-warning radars whose detection arcs feed this defense site in Mode 2 / combined solves.",
    "procedures": "Launch detailed analysis windows from fp_gui.py using this site’s configuration.",
    "footprint_cfg": "Advanced solver parameters mirrored from fcc_config.json for this token.",
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


def normalize_xy_to_meters(xy_points):
    if not xy_points:
        return []
    max_val = max(max(abs(x), abs(y)) for x, y in xy_points)
    if max_val < 1e5:
        return [(x * 1000.0, y * 1000.0) for x, y in xy_points]
    return xy_points


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
        self.use_det_range_override = False
        self.det_range_override_km = 0.0
        self.use_space_based_delay = False
        self.use_range_override = False
        self.range_override_km = 0.0
        self.use_op_range_override = False
        self.op_range_override_km = 0.0
        self.config = json.loads(json.dumps(base_config))
        self.space_based_delay_s = float(self.config.get("set_sat_delay", fcc_constants.sat_delay))
        self.missile_keys = [int(self.config.get("mtype", 1))]
        self.footprints = []
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

        self.inner.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )

        self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.v_scrollbar.set, xscrollcommand=self.h_scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.v_scrollbar.pack(side="right", fill="y")
        self.h_scrollbar.pack(side="bottom", fill="x")

        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Shift-MouseWheel>", self._on_shift_mousewheel)

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
        style.configure("TButton", padding=(10, 6))
        style.map(
            "TButton",
            background=[("active", THEME["titlebar"]), ("!disabled", THEME["toolbar_bg"])],
            foreground=[("!disabled", fg)],
        )
        style.configure("TNotebook", background=bg, borderwidth=0)
        style.configure("TNotebook.Tab", background=THEME["titlebar"], foreground=THEME["fg_muted"], padding=(12, 6))
        style.map("TNotebook.Tab", background=[("selected", THEME["accent_dim"])], foreground=[("selected", "#ffffff")])
        style.configure("TEntry", fieldbackground=THEME["entry_bg"], foreground=fg, insertcolor=fg)
        style.configure("TCheckbutton", background=bg, foreground=fg)
        style.configure(
            "TCombobox",
            fieldbackground=THEME["entry_bg"],
            background=THEME["entry_bg"],
            foreground=fg,
            arrowcolor=fg,
        )
        style.map("TCombobox", fieldbackground=[("readonly", THEME["entry_bg"])])
        style.configure("Vertical.TScrollbar", background=THEME["titlebar"], troughcolor=THEME["panel_inner"])
        style.configure("Horizontal.TScrollbar", background=THEME["titlebar"], troughcolor=THEME["panel_inner"])

    def _map_tool_btn(self, parent, text, command, tooltip=None):
        bg = THEME["tool_btn_bg"]
        hover = THEME["tool_btn_hover"]
        fg = THEME["tool_btn_fg"]
        b = tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg,
            fg=fg,
            activebackground=THEME["tool_btn_active_bg"],
            activeforeground=THEME["tool_btn_active_fg"],
            relief="flat",
            borderwidth=0,
            padx=14,
            pady=8,
            cursor="hand2",
            font=("TkDefaultFont", 10, "bold"),
            highlightthickness=0,
            highlightbackground=bg,
            highlightcolor=bg,
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
        self.map_toolbar.grid_columnconfigure(7, weight=1)

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

        self.status_var = tk.StringVar(value="Ready")
        self.status_label = tk.Label(
            self.map_toolbar,
            textvariable=self.status_var,
            bg=THEME["toolbar_bg"],
            fg=THEME["success"],
            font=("TkDefaultFont", 10),
        )
        self.status_label.grid(row=0, column=8, padx=16, pady=8, sticky="e")

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

    def _lib_btn(self, parent, text, command, fg_color=None, active_fg=None):
        bg = THEME["tool_btn_bg"]
        hover = THEME["tool_btn_hover"]
        fg = fg_color if fg_color is not None else THEME["tool_btn_fg"]
        act_fg = active_fg if active_fg is not None else THEME["tool_btn_active_fg"]
        b = tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg,
            fg=fg,
            activebackground=THEME["tool_btn_active_bg"],
            activeforeground=act_fg,
            relief="flat",
            borderwidth=0,
            padx=10,
            pady=6,
            cursor="hand2",
            font=("TkDefaultFont", 9, "bold"),
            highlightthickness=0,
            highlightbackground=bg,
            highlightcolor=bg,
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

    def _inspector_sep(self):
        ttk.Separator(self.scrollable.inner, orient="horizontal").pack(fill="x", padx=6, pady=8)

    def _inspector_heading(self, text, help_key=None):
        row = tk.Frame(self.scrollable.inner, bg=THEME["panel_inner"])
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
        if token is None:
            f = tk.Frame(inner, bg=THEME["panel_inner"])
            f.pack(anchor="w", padx=6, pady=12)
            tk.Label(
                f,
                text="Nothing selected",
                bg=THEME["panel_inner"],
                fg=THEME["fg_muted"],
                font=("TkDefaultFont", 11),
            ).pack(anchor="w")
            _help_icon(f, HELP["insp_empty"]).pack(anchor="w", pady=(8, 0))
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

        if token.kind == "ilp":
            show_center_var = tk.BooleanVar(value=token.show_center)
            show_sector_var = tk.BooleanVar(value=token.show_sector)
            use_sectoral_var = tk.BooleanVar(value=token.use_sectoral)
            use_det_override_var = tk.BooleanVar(value=token.use_det_range_override)
            use_space_delay_var = tk.BooleanVar(value=token.use_space_based_delay)
            use_range_override_var = tk.BooleanVar(value=token.use_range_override)
            use_op_range_override_var = tk.BooleanVar(value=token.use_op_range_override)
            self.sidebar_vars["show_center"] = show_center_var
            self.sidebar_vars["show_sector"] = show_sector_var
            self.sidebar_vars["use_sectoral"] = use_sectoral_var
            self.sidebar_vars["use_det_range_override"] = use_det_override_var
            self.sidebar_vars["use_space_based_delay"] = use_space_delay_var
            self.sidebar_vars["use_range_override"] = use_range_override_var
            self.sidebar_vars["use_op_range_override"] = use_op_range_override_var
            self._inspector_heading("Overlay & footprint options", None)
            self._insp_check(inner, "Show boresight line", show_center_var, HELP["show_center"], "show_center")
            self._insp_check(inner, "Show sector edges", show_sector_var, HELP["show_sector"], "show_sector")
            self._insp_check(inner, "Clip footprint to sector", use_sectoral_var, HELP["sectoral_fp"], "use_sectoral")
            self._insp_check(
                inner,
                "Override detection range (fixed km)",
                use_det_override_var,
                HELP["det_override"],
                "use_det_range_override",
            )
            self._add_entry("Detection range (km)", token.det_range_override_km, "det_range_override", HELP["det_override"])
            self._insp_check(
                inner,
                "Override with space-based detection",
                use_space_delay_var,
                HELP["space_delay"],
                "use_space_based_delay",
            )
            self._add_entry("Space-based delay (s)", token.space_based_delay_s, "space_based_delay_s", HELP["space_delay"])
            use_det_override_var.trace_add("write", lambda *_: self._on_det_mode_toggle("use_det_range_override"))
            use_space_delay_var.trace_add("write", lambda *_: self._on_det_mode_toggle("use_space_based_delay"))
            self._insp_check(
                inner,
                "Override threat max range",
                use_range_override_var,
                HELP["range_override"],
                "use_range_override",
            )
            self._add_entry("Threat range override (km)", token.range_override_km, "range_override", HELP["range_override"])
            self._insp_check(
                inner,
                "Override interceptor op range (footprint only)",
                use_op_range_override_var,
                HELP["op_range_override"],
                "use_op_range_override",
            )
            self._add_entry("Interceptor op range (km)", token.op_range_override_km, "op_range_override_km", HELP["op_range_override"])

            self._inspector_sep()
            self._inspector_heading("Linked radars", "assign_gbewr")
            gbewr_frame = ttk.Frame(inner)
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

        if token.kind == "ilp":
            self._inspector_sep()
            self._inspector_heading("Analysis tools (fp_gui)", "procedures")
            self._add_procedure_buttons(inner)

        if token.kind not in ("gbewr", "mlp"):
            self._inspector_sep()
            self._inspector_heading("Solver parameters", "footprint_cfg")

            for key, value in token.config.items():
                field_id = f"cfg::{key}"
                label = self._config_label(key)
                tip = self._config_tooltip(key)
                if isinstance(value, bool):
                    var = tk.BooleanVar(value=value)
                    self._insp_check(inner, label, var, tip, field_id)
                else:
                    self._add_entry(label, value, field_id, tip)

        self._inspector_sep()
        btn_row = tk.Frame(inner, bg=THEME["panel_inner"])
        btn_row.pack(fill="x", padx=6, pady=8)
        apply_b = self._lib_btn(btn_row, "Apply", self._apply_sidebar)
        apply_b.pack(side="left", padx=4)
        ToolTip(apply_b, HELP["apply"])
        if token.kind == "ilp":
            go_b = self._lib_btn(btn_row, "Compute footprint", self._run_footprint)
            go_b.pack(side="left", padx=4)
            ToolTip(go_b, HELP["go_footprint"])
        elif token.kind == "mlp":
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
        frame = tk.Frame(self.scrollable.inner, bg=THEME["panel_inner"])
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

        token_id, drag_tag = self._hit_test_map_token(event.x, event.y)
        if token_id is not None:
            self._select_token(self.tokens[token_id])
            self.drag_state = {"token_id": token_id, "tag": drag_tag}
            self.map_canvas.configure(cursor="hand2")
            return

        self._left_pan_armed = True
        self._left_pan_last = (event.x, event.y)

    def _on_map_drag(self, event):
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

        if self.drag_state:
            self.drag_state = None
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
        for tag in ("dir_handle", "left_handle", "right_handle", "token"):
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

        self._draw_footprint(token)

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

    def _draw_footprint(self, token):
        existing_keys = {k for k in token.canvas_items if k.startswith("footprint:")}
        active_keys = set()
        label_keys = {k for k in token.canvas_items if k.startswith("footprint_label:")}

        for fp in token.footprints:
            latlon_points = []
            theta = math.radians(token.center_dir)
            for x_local, y_local in fp["points"]:
                east = x_local * math.cos(theta) + y_local * math.sin(theta)
                north = -x_local * math.sin(theta) + y_local * math.cos(theta)
                lat, lon = self._offset_latlon(token.lat, token.lon, east, north)
                latlon_points.append((lat, lon))
            segments = self._split_latlon_at_dateline(latlon_points)

            token_tag = f"token:{token.token_id}"
            label = fp.get("label", "")
            for seg_idx, seg in enumerate(segments):
                points = []
                for lat, lon in seg:
                    px, py = self._latlon_to_xy(lat, lon)
                    points.extend([px, py])
                key = f"footprint:{fp['missile_key']}:{label}:{seg_idx}"
                active_keys.add(key)
                if key not in token.canvas_items:
                    token.canvas_items[key] = self.map_canvas.create_polygon(
                        points,
                        outline=fp["color"],
                        fill="",
                        width=2,
                        dash=(4, 3),
                        joinstyle="round",
                        tags=(token_tag,),
                    )
                else:
                    self.map_canvas.coords(token.canvas_items[key], *points)
                    self.map_canvas.itemconfigure(token.canvas_items[key], state="normal", outline=fp["color"])

            label_key = f"footprint_label:{fp['missile_key']}:{label}"
            label_keys.discard(label_key)
            display_label = fp.get("label") or self._missile_label(fp["missile_key"])
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
                        tags=(token_tag,)
                    )
                else:
                    self.map_canvas.coords(token.canvas_items[label_key], label_x + 6, label_y + 6)
                    self.map_canvas.itemconfigure(token.canvas_items[label_key], text=display_label, fill=fp["color"], state="normal")

        for key in existing_keys - active_keys:
            self.map_canvas.delete(token.canvas_items[key])
            del token.canvas_items[key]

        for key in label_keys:
            self.map_canvas.delete(token.canvas_items[key])
            del token.canvas_items[key]

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
        self.status_var.set("Computing footprint...")
        thread = threading.Thread(target=self._compute_and_render, args=(token,))
        thread.daemon = True
        thread.start()

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

    def _compute_and_render(self, token):
        try:
            footprints = []
            colors = ["red", "blue", "green", "orange", "purple", "brown", "cyan", "magenta"]
            gbewr_tokens = [self.tokens[tid] for tid in (token.gbewr_assigned or []) if tid in self.tokens and self.tokens[tid].kind == "gbewr"]
            for idx, m_key in enumerate(token.missile_keys):
                base_color = colors[idx % len(colors)]
                footprint_xy = self._calculate_footprint_xy(token, m_key, gbewr_list=None)
                if footprint_xy:
                    if token.use_sectoral:
                        footprint_xy = self._filter_sector_points(footprint_xy, token.sector_width)
                    if footprint_xy:
                        if token.use_det_range_override:
                            base_label = "Fixed det"
                        elif token.use_space_based_delay:
                            base_label = "Space-based"
                        else:
                            base_label = "Native det"
                        if getattr(token, "use_op_range_override", False) and getattr(token, "op_range_override_km", 0) > 0:
                            base_label += f" · op_cap={token.op_range_override_km:.0f}km*"
                        footprints.append({
                            "missile_key": m_key,
                            "label": base_label,
                            "color": base_color,
                            "points": normalize_xy_to_meters(footprint_xy)
                        })
                for gb_idx, gb_tok in enumerate(gbewr_tokens):
                    spec = gbewr.get_radar_spec(gb_tok.r_key) or {}
                    gb_list = [{"lat": gb_tok.lat, "lon": gb_tok.lon, "range_km": spec.get("range_km", 0)}]
                    footprint_xy = self._calculate_footprint_xy(token, m_key, gbewr_list=gb_list)
                    if footprint_xy:
                        if token.use_sectoral:
                            footprint_xy = self._filter_sector_points(footprint_xy, token.sector_width)
                        if footprint_xy:
                            label = "GBEWR: " + (spec.get("type", f"r{gb_tok.r_key}")[:24])
                            if token.use_op_range_override and token.op_range_override_km > 0:
                                label += f" · op_cap={token.op_range_override_km:.0f}km*"
                            footprints.append({
                                "missile_key": m_key,
                                "label": label,
                                "color": colors[(idx + 2 + gb_idx) % len(colors)],
                                "points": normalize_xy_to_meters(footprint_xy)
                            })
            token.footprints = footprints
        except Exception as exc:
            self.status_var.set(f"Error: {exc}")
            return
        if not token.footprints:
            self.root.after(0, lambda: self._finish_footprint(token, "No footprint found for current settings"))
            return
        self.root.after(0, lambda: self._finish_footprint(token))

    def _finish_footprint(self, token, message=None):
        self.status_var.set(message or "Footprint updated")
        self._update_token_draw(token)

    def _set_status(self, message):
        self.status_var.set(message)

    def _safe_status_update(self, msg):
        """Thread-safe status bar update."""
        self.root.after(0, lambda: self.status_var.set(msg))

    def _calculate_footprint_xy(self, token, m_type, gbewr_list=None):
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
        i_type = int(cfg.get("itype", 11))
        h_int_min = float(cfg.get("h_int_min", 0.0))
        h_discr = float(cfg.get("h_discr", 0.0))
        t_delay = float(cfg.get("t_delay", 0.0))
        mode2 = bool(cfg.get("fp_calc_mode", False))
        angle_step = float(cfg.get("angle_step", 5.0))
        num_steps_mode2 = int(cfg.get("num_steps_mode2", 15))
        acc = float(cfg.get("acc", 0.03))
        emode_sls = bool(cfg.get("set_shoot_look_shoot", False))

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
        if token.use_space_based_delay:
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
                    token.lat, token.lon, token.center_dir,
                    det_range, trj, gbewr_list,
                    base_uses_sat_delay=token.use_space_based_delay
                )

            int_table = m.load_int_table(i_type, fcc_constants.psi_step, beta_step, ROCKET_DATA_PATH, "", cfg.get("set_keep_int_tables", True))

            if not mode2:
                search_func = ss.sls_search if emode_sls else ss.short_search
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

            search_func = ss.sls_search2 if emode_sls else ss.short_search2
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
            token.use_space_based_delay = raw_space_override
            token.space_based_delay_s = max(0.0, float(self.sidebar_vars["space_based_delay_s"].get() or 0.0))
            token.config["set_sat_delay"] = token.space_based_delay_s
            token.use_range_override = bool(self.sidebar_vars["use_range_override"].get())
            token.range_override_km = float(self.sidebar_vars["range_override"].get() or 0.0)
            token.use_op_range_override = bool(self.sidebar_vars["use_op_range_override"].get())
            token.op_range_override_km = float(self.sidebar_vars["op_range_override_km"].get() or 0.0)

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
        self.sidebar_vars["lat"].set(f"{token.lat:.6f}")
        self.sidebar_vars["lon"].set(f"{token.lon:.6f}")
        if token.kind == "ilp":
            self.sidebar_vars["center_dir"].set(f"{token.center_dir:.1f}")
            self.sidebar_vars["sector_width"].set(f"{token.sector_width:.1f}")
            if "space_based_delay_s" in self.sidebar_vars:
                self.sidebar_vars["space_based_delay_s"].set(f"{token.space_based_delay_s:.1f}")

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
            footprint_points = normalize_xy_to_meters(footprint_xy)

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

    def _populate_missile_list(self, token, listbox):
        listbox.delete(0, tk.END)
        missiles = self._get_missiles()
        for idx, item in enumerate(missiles):
            nm = (item.get("type") or "").strip() or "unnamed"
            label = f"#{item['m_key']} — {nm}"
            listbox.insert(tk.END, label)
            if int(item["m_key"]) in token.missile_keys:
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
            elif action == "interceptor_flight":
                fpg.gui_interceptor_flight()
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
        ax = plt.axes(projection=ccrs.PlateCarree())
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
            return result["lat2"], result["lon2"]
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


if __name__ == "__main__":
    root = tk.Tk()
    app = MapApp(root)
    root.mainloop()
