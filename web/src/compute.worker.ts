/// <reference lib="webworker" />

type WorkerRequest =
  | { id: string; type: "init" }
  | { id: string; type: "computeRanges"; payload: { missileKeys: number[] } }
  | {
      id: string;
      type: "computeFootprint";
      payload: {
        token: {
          lat: number;
          lon: number;
          centerDir: number;
          sectorWidth: number;
          useSectoral: boolean;
          useRangeOverride?: boolean;
          rangeOverrideKm?: number;
          config: Record<string, unknown>;
        };
        missileKey: number;
      };
    };

type WorkerResponse =
  | { id: string; ok: true; type: "init"; payload: { ready: true } }
  | { id: string; ok: true; type: "computeRanges"; payload: Array<{ missileKey: number; rangeKm: number }> }
  | { id: string; ok: true; type: "computeFootprint"; payload: { points: Array<[number, number]> } }
  | { id: string; ok: false; error: string };

let pyodide: any = null;
let ready = false;

async function ensureReady() {
  if (ready) return;
  importScripts("https://cdn.jsdelivr.net/pyodide/v0.27.2/full/pyodide.js");
  pyodide = await (self as any).loadPyodide({ indexURL: "https://cdn.jsdelivr.net/pyodide/v0.27.2/full/" });
  await pyodide.loadPackage(["numpy"]);

  const files = [
    "main.py",
    "balmis.py",
    "footprintv2.py",
    "short_search.py",
    "intersection.py",
    "rocket_data.py",
    "ground_based_early_warning.py",
    "fcc_constants.py",
    "angle.py",
    "c_classes.py",
    "tables_v2b.py",
    "rocket_data.json",
    "fcc_config.json",
    "gbewr_data.json",
  ];

  pyodide.FS.mkdirTree("/app");
  pyodide.FS.chdir("/app");
  for (const name of files) {
    const txt = await fetch(`/py/${name}`).then((r) => r.text());
    pyodide.FS.writeFile(`/app/${name}`, txt);
  }

  await pyodide.runPythonAsync(`
import json
import math
import numpy as np
import main as m
import rocket_data as rd
import balmis as bm
import short_search as ss
import footprintv2 as fp
import intersection as isec
import fcc_constants
from fcc_constants import R_e, beta_step, dist_param, bm_range_limit

ROCKET_DATA_PATH = "/app/rocket_data.json"

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

def _normalize_xy_to_meters(xy_points):
    if not xy_points:
        return []
    max_val = max(max(abs(x), abs(y)) for x, y in xy_points)
    if max_val < 1e5:
        return [(x * 1000.0, y * 1000.0) for x, y in xy_points]
    return xy_points

def _filter_sector_points(points, sector_width):
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

def _get_missile_range_km(m_key):
    missile_data = rd.missile(m_key, ROCKET_DATA_PATH)
    if not missile_data:
        return 0
    trj = bm.balmisflight(missile_data)
    return float(trj[len(trj) - 1, 2] * R_e / 1000.0)

def compute_ranges(missile_keys):
    out = []
    for mk in missile_keys:
        try:
            out.append({"missileKey": int(mk), "rangeKm": _get_missile_range_km(int(mk))})
        except Exception:
            out.append({"missileKey": int(mk), "rangeKm": 0.0})
    return out

def _calculate_footprint_xy(token, m_type):
    cfg = dict(token.get("config", {}))
    i_type = int(cfg.get("itype", 11))
    h_int_min = float(cfg.get("h_int_min", 0.0))
    h_discr = float(cfg.get("h_discr", 0.0))
    t_delay = float(cfg.get("t_delay", 0.0))
    mode2 = bool(cfg.get("fp_calc_mode", False))
    angle_step = float(cfg.get("angle_step", 5.0))
    num_steps_mode2 = int(cfg.get("num_steps_mode2", 15))
    acc = float(cfg.get("acc", 0.03))
    emode_sls = bool(cfg.get("set_shoot_look_shoot", False))

    missile_data = rd.missile(m_type, ROCKET_DATA_PATH)
    if missile_data is False:
        return []
    if token.get("useRangeOverride") and float(token.get("rangeOverrideKm", 0.0)) > 0:
        missile_data = dict(missile_data)
        missile_data["range"] = str(token.get("rangeOverrideKm"))

    trj = bm.balmisflight(missile_data)
    mrange = trj[len(trj) - 1, 2] * R_e
    if mrange > bm_range_limit:
        return []

    burn_time = sum(missile_data["t_bu"]) + sum(missile_data["t_delay"])
    t_int_lnc = float(burn_time + t_delay)

    interceptor_data = rd.interceptor(i_type, ROCKET_DATA_PATH)
    if interceptor_data is False:
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
    det_range = 0.0

    int_table = m.load_int_table(
        i_type,
        fcc_constants.psi_step,
        beta_step,
        ROCKET_DATA_PATH,
        "",
        bool(cfg.get("set_keep_int_tables", True)),
    )

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
            dist,
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
        num_steps_mode2,
    )
    fp_m1 = isec.fprint_m2tom1(footprint_tab2)
    points = []
    for part in fp_m1:
        for row in part:
            points.append((row[3], row[4]))
    return points

def compute_footprint(payload):
    token = payload["token"]
    missile_key = int(payload["missileKey"])
    pts = _calculate_footprint_xy(token, missile_key)
    if token.get("useSectoral"):
        pts = _filter_sector_points(pts, float(token.get("sectorWidth", 60.0)))
    pts = _normalize_xy_to_meters(pts)
    return [(float(x), float(y)) for x, y in pts]
  `);

  ready = true;
}

self.onmessage = async (ev: MessageEvent<WorkerRequest>) => {
  const msg = ev.data;
  try {
    await ensureReady();

    if (msg.type === "init") {
      (self as unknown as { postMessage: (m: WorkerResponse) => void }).postMessage({
        id: msg.id,
        ok: true,
        type: "init",
        payload: { ready: true },
      });
      return;
    }

    if (msg.type === "computeRanges") {
      pyodide.globals.set("missile_keys_js", msg.payload.missileKeys);
      const out = pyodide.runPython(`compute_ranges(missile_keys_js)`).toJs({ dict_converter: Object.fromEntries });
      (self as unknown as { postMessage: (m: WorkerResponse) => void }).postMessage({
        id: msg.id,
        ok: true,
        type: "computeRanges",
        payload: out,
      });
      return;
    }

    if (msg.type === "computeFootprint") {
      pyodide.globals.set("payload_js", { token: msg.payload.token, missileKey: msg.payload.missileKey });
      const pts = pyodide.runPython(`compute_footprint(payload_js)`).toJs();
      (self as unknown as { postMessage: (m: WorkerResponse) => void }).postMessage({
        id: msg.id,
        ok: true,
        type: "computeFootprint",
        payload: { points: pts as Array<[number, number]> },
      });
      return;
    }
  } catch (err) {
    (self as unknown as { postMessage: (m: WorkerResponse) => void }).postMessage({
      id: msg.id,
      ok: false,
      error: err instanceof Error ? err.message : String(err),
    });
  }
};

