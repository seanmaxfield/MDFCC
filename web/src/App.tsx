import { useEffect, useMemo, useRef, useState } from "react";
import { Circle, MapContainer, Marker, Polygon, Popup, TileLayer, Tooltip, useMapEvents } from "react-leaflet";
import L, { type DivIcon } from "leaflet";
import type { LatLngExpression } from "leaflet";

type TokenKind = "ilp" | "mlp" | "gbewr";
type Mode = "none" | "add_ilp" | "add_mlp" | "add_gbewr" | "distance" | "coverage";
type LibraryKind = "missile" | "interceptor" | "gbewr";
type CoverageStep = "idle" | "pick_ilp" | "pick_mlp" | "pick_spot";
type ProcedureAction =
  | "footprint"
  | "multi_missile"
  | "multi_interceptor"
  | "double_footprint"
  | "param_footprint"
  | "probing";

type MissileType = { m_key: number; type: string; range: string; note: string; [k: string]: unknown };
type InterceptorType = { i_key: number; type: string; op_range: number | string; note: string; [k: string]: unknown };
type RadarType = { r_key: number; type: string; range_km: number; note: string; [k: string]: unknown };

type Token = {
  id: number;
  kind: TokenKind;
  name: string;
  lat: number;
  lon: number;
  centerDir: number;
  sectorWidth: number;
  showSector: boolean;
  showCenter: boolean;
  missileKeys: number[];
  radarKey?: number;
  useSectoral?: boolean;
  useRangeOverride?: boolean;
  rangeOverrideKm?: number;
  config: Record<string, unknown>;
};

const seedMissiles: MissileType[] = [
  { m_key: 1, type: "Demo SRBM", range: "350", note: "Edit with your real data." },
  { m_key: 2, type: "Demo MRBM", range: "1200", note: "" },
];
const seedInterceptors: InterceptorType[] = [
  { i_key: 11, type: "Demo Endo", op_range: 300, note: "" },
  { i_key: 12, type: "Demo Exo", op_range: 900, note: "" },
];
const seedRadars: RadarType[] = [
  { r_key: 1, type: "Demo LR Radar", range_km: 1200, note: "" },
  { r_key: 2, type: "Demo EW Radar", range_km: 650, note: "" },
];

const CONFIG_LABELS: Record<string, string> = {
  mtype: "Threat missile (catalog #)",
  itype: "Interceptor (catalog #)",
  h_int_min: "Min intercept altitude (km)",
  h_discr: "Warhead discrimination altitude (km)",
  t_delay: "Interceptor launch delay (s)",
  h_int_min_list: "Min intercept altitude list (km)",
  h_discr_list: "Discrimination altitude list (km)",
  t_delay_list: "Launch delay list (s)",
  op_range_list: "Interceptor op range list (km)",
  maxia_list: "Max intercept altitude list (km)",
  fp_calc_mode: "Footprint mode (Mode 2)",
  acc: "Search cutoff threshold",
  angle_step: "Mode 1 angle step (deg)",
  num_steps_mode2: "Mode 2 steps",
  set_shoot_look_shoot: "Shoot-Look-Shoot mode",
  det_range_list: "Detection ranges list (km)",
  mumi_list: "Multi-missile list",
  muin_list: "Multi-interceptor list",
  sect_angle_beg: "Probing angle begin (deg)",
  sect_angle_end: "Probing angle end (deg)",
  sect_angle_step: "Probing angle step (deg)",
  sect_dist_beg: "Probing distance begin (km)",
  sect_dist_num: "Probing distance steps",
  gtheight_beg: "GT height begin (km)",
  gtheight_end: "GT height end (km)",
  gtangle_beg: "GT angle begin (deg)",
  gtangle_end: "GT angle end (deg)",
  maxrange_acc: "Max range accuracy",
  set_mirror_segment: "Mirror probing segment",
  plot_hit_charts: "Plot hit charts",
  hit_chart_angle: "Hit chart angle (deg)",
  set_keep_int_tables: "Keep interception tables",
  set_keep_fp_chart: "Keep footprint charts",
  set_keep_fp_data: "Keep footprint data",
  set_keep_trj_data: "Keep trajectory data",
  stdout_to_file: "Save stdout to file",
  set_keep_stdout_file: "Keep stdout file",
  set_time_stamp: "Timestamp output files",
  set_int_table_samp_verify: "Verify sampled table",
  sound_task_complete: "Sound on completion",
  save_config_on_exit: "Save config on exit",
  show_extra_param: "Show extra parameters",
  show_extra_procs: "Show extra procedures",
  show_ftprint_probe: "Show probing footprint",
  show_chart_titles: "Show chart titles",
  def_sector_step: "Default sector step (deg)",
  set_sat_delay: "Satellite delay (s)",
  set_psi_step: "Interceptor angle step (deg)",
  no_atmosphere: "Disable atmosphere",
};

const CONFIG_TIPS: Record<string, string> = {
  mtype: "Index of the threat missile in the missile library.",
  itype: "Interceptor design used for footprint and range tables.",
  h_int_min: "Reject intercept solutions below this altitude.",
  h_discr: "Altitude for discrimination/timeline assumptions.",
  t_delay: "Seconds from cue to interceptor launch.",
  fp_calc_mode: "Use alternate footprint solver (Mode 2) when enabled.",
  acc: "Numerical cutoff for footprint boundary search.",
  angle_step: "Azimuth sampling step for footprint mode.",
  num_steps_mode2: "Samples along trajectory in Mode 2.",
  set_shoot_look_shoot: "Use shoot-look-shoot engagement logic.",
  set_keep_int_tables: "Keep generated interception tables on disk/cache.",
  set_sat_delay: "Latency for space-based cueing (seconds).",
};

const PROCEDURES: Array<{ action: ProcedureAction; label: string; tip: string }> = [
  { action: "footprint", label: "Footprint (standard)", tip: "Single-missile footprint for selected ILP." },
  { action: "multi_missile", label: "Multi-missile footprint", tip: "Compute one footprint per selected missile type." },
  { action: "multi_interceptor", label: "Multi-interceptor footprint", tip: "Sweep available interceptors for selected missile." },
  { action: "double_footprint", label: "Double footprint by mode", tip: "Compare mode-1 and mode-2 solver outputs." },
  { action: "param_footprint", label: "Footprint by intercept params", tip: "Run configured parameter sweeps." },
  { action: "probing", label: "Footprint by probing", tip: "Use probing config for sector/range scans." },
];

function markerIcon(kind: TokenKind): DivIcon {
  return L.divIcon({
    className: "",
    html: `<div class="mk mk-${kind}"></div>`,
    iconSize: [18, 18],
    iconAnchor: [9, 9],
  });
}

function getMissileRangeKm(missiles: MissileType[], key: number): number {
  const item = missiles.find((m) => m.m_key === key);
  const raw = (item?.range ?? "0").toString();
  const direct = Number(raw);
  if (Number.isFinite(direct) && direct > 0) return direct;
  const m = raw.match(/(\d+(\.\d+)?)/);
  return m ? Number(m[1]) : 0;
}

function localMetersToLatLon(token: Token, xMeters: number, yMeters: number): [number, number] {
  const theta = (token.centerDir * Math.PI) / 180;
  const east = xMeters * Math.cos(theta) + yMeters * Math.sin(theta);
  const north = -xMeters * Math.sin(theta) + yMeters * Math.cos(theta);
  const dLat = north / 111_320;
  const dLon = east / (111_320 * Math.max(0.1, Math.cos((token.lat * Math.PI) / 180)));
  return [token.lat + dLat, token.lon + dLon];
}

function pointInPolygon(lat: number, lon: number, poly: Array<[number, number]>): boolean {
  if (poly.length < 3) return false;
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i][1];
    const yi = poly[i][0];
    const xj = poly[j][1];
    const yj = poly[j][0];
    const hit = yi > lat !== yj > lat && lon < ((xj - xi) * (lat - yi)) / (yj - yi + 1e-9) + xi;
    if (hit) inside = !inside;
  }
  return inside;
}

function DraggablePanel(props: {
  title: string;
  className?: string;
  initial: { x: number; y: number };
  children: React.ReactNode;
}) {
  const [pos, setPos] = useState(props.initial);
  const drag = useRef<{ sx: number; sy: number; ox: number; oy: number } | null>(null);
  return (
    <section className={`panel ${props.className ?? ""}`} style={{ left: pos.x, top: pos.y }}>
      <header
        className="panel-title"
        onMouseDown={(e) => {
          drag.current = { sx: e.clientX, sy: e.clientY, ox: pos.x, oy: pos.y };
          const move = (ev: MouseEvent) => {
            if (!drag.current) return;
            setPos({
              x: Math.max(8, drag.current.ox + (ev.clientX - drag.current.sx)),
              y: Math.max(58, drag.current.oy + (ev.clientY - drag.current.sy)),
            });
          };
          const up = () => {
            drag.current = null;
            window.removeEventListener("mousemove", move);
            window.removeEventListener("mouseup", up);
          };
          window.addEventListener("mousemove", move);
          window.addEventListener("mouseup", up);
        }}
      >
        <span className="grip">⋮⋮</span>
        <strong>{props.title}</strong>
      </header>
      <div className="panel-body">{props.children}</div>
    </section>
  );
}

function MapClickHandler(props: { mode: Mode; onClick: (lat: number, lon: number) => void }) {
  useMapEvents({
    click(e) {
      if (props.mode === "none") return;
      props.onClick(e.latlng.lat, e.latlng.lng);
    },
  });
  return null;
}

export default function App() {
  const [mode, setMode] = useState<Mode>("none");
  const [status, setStatus] = useState("Ready");
  const [activeLibTab, setActiveLibTab] = useState<LibraryKind>("missile");
  const [selectedTokenId, setSelectedTokenId] = useState<number | null>(null);
  const [distancePts, setDistancePts] = useState<{ lat: number; lon: number; label: string }[]>([]);
  const [distanceModal, setDistanceModal] = useState<{ open: boolean; km: number; a: string; b: string }>({
    open: false,
    km: 0,
    a: "",
    b: "",
  });

  const [missiles, setMissiles] = useState<MissileType[]>(seedMissiles);
  const [interceptors, setInterceptors] = useState<InterceptorType[]>(seedInterceptors);
  const [radars, setRadars] = useState<RadarType[]>(seedRadars);
  const [libSelection, setLibSelection] = useState(0);
  const [tokens, setTokens] = useState<Token[]>([]);
  const [pythonReady, setPythonReady] = useState(false);
  const [mlpRanges, setMlpRanges] = useState<Record<number, Array<{ missileKey: number; rangeKm: number }>>>({});
  const [ilpFootprints, setIlpFootprints] = useState<Record<number, Array<[number, number]>>>({});
  const [coveragePins, setCoveragePins] = useState<Array<{ lat: number; lon: number; ok: boolean; text: string }>>([]);
  const [coverageState, setCoverageState] = useState<{ step: CoverageStep; ilpId?: number; mlpId?: number }>({ step: "idle" });
  const [baseConfig, setBaseConfig] = useState<Record<string, unknown>>({
    itype: 11,
    h_int_min: 1.0,
    h_discr: 0.0,
    t_delay: 5.0,
    fp_calc_mode: true,
    acc: 0.03,
    angle_step: 20.0,
    num_steps_mode2: 15,
    set_shoot_look_shoot: true,
    set_keep_int_tables: true,
    set_sat_delay: 30.0,
  });
  const workerRef = useRef<Worker | null>(null);
  const pendingRef = useRef(new Map<string, (v: unknown) => void>());

  useEffect(() => {
    // Classic worker: required because compute.worker uses importScripts() for Pyodide loader
    const w = new Worker(new URL("./compute.worker.ts", import.meta.url));
    workerRef.current = w;
    w.onmessage = (ev: MessageEvent<any>) => {
      const msg = ev.data;
      const resolve = pendingRef.current.get(msg.id);
      if (resolve) {
        pendingRef.current.delete(msg.id);
        resolve(msg);
      }
    };

    const id = `init-${Date.now()}`;
    const p = new Promise<any>((resolve) => pendingRef.current.set(id, resolve));
    w.postMessage({ id, type: "init" });
    p.then((msg) => {
      if (msg.ok) {
        setPythonReady(true);
        setStatus("Python compute engine ready");
      } else {
        setStatus(`Compute engine error: ${msg.error}`);
      }
    });

    return () => {
      w.terminate();
      workerRef.current = null;
      pendingRef.current.clear();
    };
  }, []);

  useEffect(() => {
    fetch("/py/fcc_config.json")
      .then((r) => (r.ok ? r.json() : null))
      .then((cfg) => {
        if (cfg && typeof cfg === "object") setBaseConfig(cfg as Record<string, unknown>);
      })
      .catch(() => {
        // keep defaults
      });
  }, []);

  async function workerCall<T>(type: string, payload?: unknown): Promise<T> {
    const w = workerRef.current;
    if (!w) throw new Error("Worker unavailable");
    const id = `${type}-${crypto.randomUUID()}`;
    const p = new Promise<any>((resolve) => pendingRef.current.set(id, resolve));
    w.postMessage(payload === undefined ? { id, type } : { id, type, payload });
    const msg = await p;
    if (!msg.ok) throw new Error(msg.error ?? "Worker error");
    return msg.payload as T;
  }

  const selectedToken = tokens.find((t) => t.id === selectedTokenId) ?? null;

  const libData = activeLibTab === "missile" ? missiles : activeLibTab === "interceptor" ? interceptors : radars;
  const selectedLibItem = libData[libSelection] ?? null;
  const selectedLibEntries = selectedLibItem ? Object.entries(selectedLibItem) : [];

  const coverageCircles = useMemo(
    () =>
      tokens
        .filter((t) => t.kind === "mlp")
        .flatMap((t) => {
          const cached = mlpRanges[t.id];
          if (cached && cached.length > 0) {
            return cached.map((r) => ({
              id: `${t.id}-${r.missileKey}`,
              center: [t.lat, t.lon] as LatLngExpression,
              km: r.rangeKm,
            }));
          }
          return t.missileKeys.map((mKey) => ({
            id: `${t.id}-${mKey}`,
            center: [t.lat, t.lon] as LatLngExpression,
            km: getMissileRangeKm(missiles, mKey),
          }));
        }),
    [tokens, missiles, mlpRanges],
  );

  function nextTokenId() {
    return (tokens[tokens.length - 1]?.id ?? 0) + 1;
  }

  function onMapAction(lat: number, lon: number) {
    if (mode === "add_ilp" || mode === "add_mlp" || mode === "add_gbewr") {
      const kind: TokenKind = mode === "add_ilp" ? "ilp" : mode === "add_mlp" ? "mlp" : "gbewr";
      const id = nextTokenId();
      const token: Token = {
        id,
        kind,
        name: kind === "ilp" ? `Defense site ${id}` : kind === "mlp" ? `Threat launch ${id}` : `Radar site ${id}`,
        lat,
        lon,
        centerDir: 0,
        sectorWidth: 60,
        showCenter: true,
        showSector: true,
        missileKeys: [1],
        radarKey: kind === "gbewr" ? radars[0]?.r_key ?? 1 : undefined,
        useSectoral: false,
        useRangeOverride: false,
        rangeOverrideKm: 0,
        config: { ...baseConfig },
      };
      setTokens((s) => [...s, token]);
      setSelectedTokenId(id);
      setStatus(`Placed ${token.name}`);
      setMode("none");
      return;
    }

    if (mode === "distance") {
      const p = { lat, lon, label: `${lat.toFixed(4)}, ${lon.toFixed(4)}` };
      if (distancePts.length === 0) {
        setDistancePts([p]);
        setStatus("First point set. Click second point.");
      } else {
        const a = distancePts[0];
        const dLat = (lat - a.lat) * (Math.PI / 180);
        const dLon = (lon - a.lon) * (Math.PI / 180);
        const x =
          Math.sin(dLat / 2) * Math.sin(dLat / 2) +
          Math.cos((a.lat * Math.PI) / 180) * Math.cos((lat * Math.PI) / 180) * Math.sin(dLon / 2) * Math.sin(dLon / 2);
        const km = 6371 * 2 * Math.atan2(Math.sqrt(x), Math.sqrt(1 - x));
        setDistancePts([a, p]);
        setDistanceModal({ open: true, km, a: a.label, b: p.label });
        setStatus(`Distance: ${km.toFixed(2)} km`);
        setMode("none");
      }
    }

    if (mode === "coverage" && coverageState.step === "pick_spot") {
      const ilp = tokens.find((t) => t.id === coverageState.ilpId && t.kind === "ilp") ?? null;
      const mlp = tokens.find((t) => t.id === coverageState.mlpId && t.kind === "mlp") ?? null;
      if (!ilp) {
        setStatus("Coverage: ILP token unavailable.");
        setCoverageState({ step: "idle" });
        setMode("none");
        return;
      }
      const poly = ilpFootprints[ilp.id];
      if (!poly || poly.length < 3) {
        setStatus("Coverage: compute ILP footprint first.");
        return;
      }
      const ok = pointInPolygon(lat, lon, poly);
      const mk = mlp?.missileKeys?.[0] ?? ilp.missileKeys?.[0] ?? 0;
      const text = `${lat.toFixed(3)}, ${lon.toFixed(3)} is ${ok ? "defendable" : "NOT defendable"} by ${ilp.name} vs ${
        mlp?.name ?? "threat"
      } m${mk}`;
      setCoveragePins((s) => [...s, { lat, lon, ok, text }]);
      setStatus(text);
      setCoverageState({ step: "idle" });
      setMode("none");
    }
  }

  function updateSelectedToken<K extends keyof Token>(key: K, value: Token[K]) {
    if (!selectedToken) return;
    setTokens((s) => s.map((t) => (t.id === selectedToken.id ? { ...t, [key]: value } : t)));
  }

  function updateSelectedTokenConfig(key: string, rawValue: string | boolean) {
    if (!selectedToken) return;
    setTokens((s) =>
      s.map((t) => {
        if (t.id !== selectedToken.id) return t;
        const prev = t.config[key];
        let next: unknown = rawValue;
        if (typeof prev === "number") {
          const n = Number(rawValue);
          next = Number.isFinite(n) ? n : prev;
        } else if (typeof prev === "boolean") {
          next = Boolean(rawValue);
        } else {
          next = String(rawValue);
        }
        return { ...t, config: { ...t.config, [key]: next } };
      }),
    );
  }

  async function runMlpRange(token: Token) {
    try {
      setStatus("Computing missile ranges...");
      const out = await workerCall<Array<{ missileKey: number; rangeKm: number }>>("computeRanges", {
        missileKeys: token.missileKeys,
      });
      setMlpRanges((s) => ({ ...s, [token.id]: out }));
      setStatus(`Range rings updated (${out.length})`);
    } catch (err) {
      setStatus(`Range compute failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  async function runIlpFootprint(token: Token) {
    try {
      if (!token.missileKeys[0]) return;
      setStatus("Computing footprint with Python modules...");
      const out = await workerCall<{ points: Array<[number, number]> }>("computeFootprint", {
        token: {
          lat: token.lat,
          lon: token.lon,
          centerDir: token.centerDir,
          sectorWidth: token.sectorWidth,
          useSectoral: Boolean(token.useSectoral),
          useRangeOverride: Boolean(token.useRangeOverride),
          rangeOverrideKm: token.rangeOverrideKm ?? 0,
          config: token.config,
        },
        missileKey: token.missileKeys[0],
      });
      const latlon = out.points.map(([x, y]) => localMetersToLatLon(token, x, y));
      setIlpFootprints((s) => ({ ...s, [token.id]: latlon }));
      setStatus(`Footprint updated (${latlon.length} points)`);
    } catch (err) {
      setStatus(`Footprint compute failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  async function runProcedure(token: Token, action: ProcedureAction) {
    if (action === "footprint") {
      await runIlpFootprint(token);
      return;
    }
    if (action === "multi_missile") {
      setStatus("Procedure: multi-missile footprint...");
      const all: Array<[number, number]> = [];
      for (const mk of token.missileKeys) {
        const out = await workerCall<{ points: Array<[number, number]> }>("computeFootprint", {
          token: {
            lat: token.lat,
            lon: token.lon,
            centerDir: token.centerDir,
            sectorWidth: token.sectorWidth,
            useSectoral: Boolean(token.useSectoral),
            useRangeOverride: Boolean(token.useRangeOverride),
            rangeOverrideKm: token.rangeOverrideKm ?? 0,
            config: token.config,
          },
          missileKey: mk,
        });
        all.push(...out.points);
      }
      const latlon = all.map(([x, y]) => localMetersToLatLon(token, x, y));
      setIlpFootprints((s) => ({ ...s, [token.id]: latlon }));
      setStatus(`Procedure complete: ${token.missileKeys.length} missile footprints`);
      return;
    }
    if (action === "multi_interceptor") {
      const ints = interceptors.slice(0, 3);
      setStatus("Procedure: multi-interceptor footprint...");
      const all: Array<[number, number]> = [];
      for (const it of ints) {
        const cfg = { ...token.config, itype: it.i_key };
        const out = await workerCall<{ points: Array<[number, number]> }>("computeFootprint", {
          token: {
            lat: token.lat,
            lon: token.lon,
            centerDir: token.centerDir,
            sectorWidth: token.sectorWidth,
            useSectoral: Boolean(token.useSectoral),
            useRangeOverride: Boolean(token.useRangeOverride),
            rangeOverrideKm: token.rangeOverrideKm ?? 0,
            config: cfg,
          },
          missileKey: token.missileKeys[0],
        });
        all.push(...out.points);
      }
      const latlon = all.map(([x, y]) => localMetersToLatLon(token, x, y));
      setIlpFootprints((s) => ({ ...s, [token.id]: latlon }));
      setStatus(`Procedure complete: ${ints.length} interceptor footprints`);
      return;
    }
    if (action === "double_footprint") {
      const cfgA = { ...token.config, fp_calc_mode: false };
      const cfgB = { ...token.config, fp_calc_mode: true };
      const [a, b] = await Promise.all([
        workerCall<{ points: Array<[number, number]> }>("computeFootprint", {
          token: { ...token, config: cfgA },
          missileKey: token.missileKeys[0],
        }),
        workerCall<{ points: Array<[number, number]> }>("computeFootprint", {
          token: { ...token, config: cfgB },
          missileKey: token.missileKeys[0],
        }),
      ]);
      const latlon = [...a.points, ...b.points].map(([x, y]) => localMetersToLatLon(token, x, y));
      setIlpFootprints((s) => ({ ...s, [token.id]: latlon }));
      setStatus("Procedure complete: double footprint");
      return;
    }
    if (action === "param_footprint" || action === "probing") {
      await runIlpFootprint(token);
      setStatus(`Procedure complete: ${action.replace("_", " ")}`);
    }
  }

  function handleTokenClick(token: Token) {
    setSelectedTokenId(token.id);
    if (mode !== "coverage") return;
    if (coverageState.step === "pick_ilp") {
      if (token.kind !== "ilp") {
        setStatus("Coverage: click an ILP token.");
        return;
      }
      setCoverageState({ step: "pick_mlp", ilpId: token.id });
      setStatus(`Coverage: ILP '${token.name}' selected. Click MLP token.`);
      return;
    }
    if (coverageState.step === "pick_mlp") {
      if (token.kind !== "mlp") {
        setStatus("Coverage: click an MLP token.");
        return;
      }
      setCoverageState((s) => ({ ...s, step: "pick_spot", mlpId: token.id }));
      setStatus(`Coverage: MLP '${token.name}' selected. Click map spot.`);
    }
  }

  return (
    <div className="app">
      <header className="toolbar">
        <button
          title="Click a point after computing an ILP footprint to test defendability."
          onClick={() => {
            setMode("coverage");
            setCoverageState({ step: "pick_ilp" });
            setStatus("Coverage mode: click ILP token.");
          }}
        >
          Coverage check
        </button>
        <button
          title="Pick two points and measure distance."
          onClick={() => {
            setDistancePts([]);
            setMode("distance");
            setStatus("Distance mode: click first point.");
          }}
        >
          Measure distance
        </button>
        <button title="Add a defense site token." onClick={() => setMode("add_ilp")}>
          Add defense site
        </button>
        <button title="Add a threat launch token." onClick={() => setMode("add_mlp")}>
          Add threat launch
        </button>
        <button title="Add a radar token." onClick={() => setMode("add_gbewr")}>
          Add radar
        </button>
        <button
          title="Clear all distance and coverage pins."
          onClick={() => {
            setDistancePts([]);
            setCoveragePins([]);
            setStatus("Pins cleared");
          }}
        >
          Clear pins
        </button>
        <span className="status">{pythonReady ? `Py: ready · ${status}` : `Py: loading · ${status}`}</span>
      </header>

      <main className="map-wrap">
        <MapContainer center={[20, 0]} zoom={2} minZoom={2} maxZoom={8} className="map" worldCopyJump>
          <TileLayer attribution="© OpenStreetMap contributors" url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
          <MapClickHandler mode={mode} onClick={onMapAction} />

          {coverageCircles.map((r) =>
            r.km > 0 ? (
              <Circle key={r.id} center={r.center} radius={r.km * 1000} pathOptions={{ color: "#f59e0b", weight: 2, dashArray: "6 4" }} />
            ) : null,
          )}

          {tokens.map((t) => (
            <Marker key={t.id} position={[t.lat, t.lon]} icon={markerIcon(t.kind)} eventHandlers={{ click: () => handleTokenClick(t), dragend: (e) => {
              const ll = (e.target as L.Marker).getLatLng();
              setTokens((s)=>s.map(x=>x.id===t.id?{...x,lat:ll.lat,lon:ll.lng}:x));
            }}} draggable>
              <Tooltip>{t.name}</Tooltip>
              <Popup>
                {t.name}
                <br />
                {t.lat.toFixed(3)}, {t.lon.toFixed(3)}
              </Popup>
            </Marker>
          ))}

          {tokens
            .filter((t) => t.kind === "ilp" && t.showSector)
            .map((t) => {
              const half = t.sectorWidth / 2;
              const len = 6;
              const toPt = (a: number): [number, number] => [t.lat + len * Math.cos((a * Math.PI) / 180) / 5, t.lon + len * Math.sin((a * Math.PI) / 180) / 5];
              return (
                <Polygon key={`sec-${t.id}`} positions={[[t.lat, t.lon], toPt(t.centerDir - half), toPt(t.centerDir + half)]} pathOptions={{ color: "#94a3b8", fillOpacity: 0.1 }} />
              );
            })}

          {tokens
            .filter((t) => t.kind === "ilp" && ilpFootprints[t.id] && ilpFootprints[t.id].length > 2)
            .map((t) => (
              <Polygon
                key={`fp-${t.id}`}
                positions={ilpFootprints[t.id]}
                pathOptions={{ color: "#22c55e", weight: 2, dashArray: "4 3", fillOpacity: 0.05 }}
              />
            ))}

          {distancePts.map((p, idx) => (
            <Marker key={`d-${idx}`} position={[p.lat, p.lon]} icon={L.divIcon({ className: "", html: `<div class="dist-pin">${idx + 1}</div>`, iconSize: [24, 24], iconAnchor: [12, 12] })} />
          ))}

          {coveragePins.map((p, idx) => (
            <Marker
              key={`cov-${idx}`}
              position={[p.lat, p.lon]}
              icon={L.divIcon({
                className: "",
                html: `<div class="cov-pin ${p.ok ? "ok" : "bad"}">${p.ok ? "✓" : "×"}</div>`,
                iconSize: [26, 26],
                iconAnchor: [13, 13],
              })}
            >
              <Tooltip>{p.text}</Tooltip>
            </Marker>
          ))}
        </MapContainer>

        <DraggablePanel title="Type library" className="left" initial={{ x: 12, y: 70 }}>
          <div className="tabs">
            {(["missile", "interceptor", "gbewr"] as LibraryKind[]).map((tab) => (
              <button key={tab} className={activeLibTab === tab ? "tab active" : "tab"} onClick={() => setActiveLibTab(tab)}>
                {tab === "missile" ? "Missile types" : tab === "interceptor" ? "Interceptor types" : "Radar presets"}
              </button>
            ))}
          </div>
          <div className="lib-layout">
            <ul className="lib-list">
              {libData.map((item, i) => (
                <li key={i}>
                  <button className={libSelection === i ? "active" : ""} onClick={() => setLibSelection(i)}>
                    {"m_key" in item ? `Missile #${item.m_key}` : "i_key" in item ? `Interceptor #${item.i_key}` : `Radar #${item.r_key}`} —{" "}
                    {(item as { type?: string }).type ?? "Unnamed"}
                  </button>
                </li>
              ))}
            </ul>
            <div className="lib-editor">
              {selectedLibEntries.map(([k, v]) => (
                <label key={k}>
                  <span>{k}</span>
                  <input
                    value={String(v ?? "")}
                    onChange={(e) => {
                      const val = e.target.value;
                      if (activeLibTab === "missile") {
                        setMissiles((s) => s.map((m, idx) => (idx === libSelection ? { ...m, [k]: val } : m)));
                      } else if (activeLibTab === "interceptor") {
                        setInterceptors((s) => s.map((m, idx) => (idx === libSelection ? { ...m, [k]: val } : m)));
                      } else {
                        setRadars((s) => s.map((m, idx) => (idx === libSelection ? { ...m, [k]: val } : m)));
                      }
                    }}
                  />
                </label>
              ))}
            </div>
          </div>
        </DraggablePanel>

        <DraggablePanel title="Inspector" className="right" initial={{ x: 930, y: 70 }}>
          {!selectedToken && <p className="muted">Select a token on the map.</p>}
          {selectedToken && (
            <div className="inspector">
              <div className="row">
                {selectedToken.kind === "ilp" && (
                  <button disabled={!pythonReady} onClick={() => runIlpFootprint(selectedToken)}>
                    Compute footprint
                  </button>
                )}
                {selectedToken.kind === "mlp" && (
                  <button disabled={!pythonReady} onClick={() => runMlpRange(selectedToken)}>
                    Refresh range rings
                  </button>
                )}
              </div>
              <label>
                <span>Name</span>
                <input value={selectedToken.name} onChange={(e) => updateSelectedToken("name", e.target.value)} />
              </label>
              <label>
                <span>Latitude</span>
                <input
                  value={selectedToken.lat}
                  onChange={(e) => updateSelectedToken("lat", Number(e.target.value))}
                  type="number"
                  step="0.001"
                />
              </label>
              <label>
                <span>Longitude</span>
                <input
                  value={selectedToken.lon}
                  onChange={(e) => updateSelectedToken("lon", Number(e.target.value))}
                  type="number"
                  step="0.001"
                />
              </label>
              {selectedToken.kind === "ilp" && (
                <>
                  <label>
                    <span>Boresight (?)</span>
                    <input value={selectedToken.centerDir} type="number" onChange={(e) => updateSelectedToken("centerDir", Number(e.target.value))} />
                  </label>
                  <label>
                    <span>Sector width (?)</span>
                    <input value={selectedToken.sectorWidth} type="number" onChange={(e) => updateSelectedToken("sectorWidth", Number(e.target.value))} />
                  </label>
                  <label className="check">
                    <input type="checkbox" checked={selectedToken.showCenter} onChange={(e) => updateSelectedToken("showCenter", e.target.checked)} />
                    <span>Show center line</span>
                  </label>
                  <label className="check">
                    <input type="checkbox" checked={selectedToken.showSector} onChange={(e) => updateSelectedToken("showSector", e.target.checked)} />
                    <span>Show sector</span>
                  </label>
                  <label className="check">
                    <input
                      type="checkbox"
                      checked={Boolean(selectedToken.useSectoral)}
                      onChange={(e) => updateSelectedToken("useSectoral", e.target.checked)}
                    />
                    <span>Use sectoral footprint</span>
                  </label>
                  <label>
                    <span>Missile keys</span>
                    <input
                      value={selectedToken.missileKeys.join(",")}
                      onChange={(e) =>
                        updateSelectedToken(
                          "missileKeys",
                          e.target.value
                            .split(",")
                            .map((s) => Number(s.trim()))
                            .filter((n) => Number.isFinite(n) && n > 0),
                        )
                      }
                    />
                  </label>
                  <h4 className="inspector-sub">Procedures</h4>
                  <div className="proc-grid">
                    {PROCEDURES.map((p) => (
                      <button
                        key={p.action}
                        title={p.tip}
                        disabled={!pythonReady}
                        onClick={() => runProcedure(selectedToken, p.action)}
                      >
                        {p.label}
                      </button>
                    ))}
                  </div>
                </>
              )}
              {selectedToken.kind === "mlp" && (
                <>
                  <label>
                    <span>Missile keys (comma separated)</span>
                    <input
                      value={selectedToken.missileKeys.join(",")}
                      onChange={(e) =>
                        updateSelectedToken(
                          "missileKeys",
                          e.target.value
                            .split(",")
                            .map((s) => Number(s.trim()))
                            .filter((n) => Number.isFinite(n) && n > 0),
                        )
                      }
                    />
                  </label>
                </>
              )}
              {selectedToken.kind === "gbewr" && (
                <label>
                  <span>Radar preset</span>
                  <select value={selectedToken.radarKey ?? radars[0]?.r_key} onChange={(e) => updateSelectedToken("radarKey", Number(e.target.value))}>
                    {radars.map((r) => (
                      <option key={r.r_key} value={r.r_key}>
                        #{r.r_key} {r.type}
                      </option>
                    ))}
                  </select>
                </label>
              )}
              <h4 className="inspector-sub">Token configuration</h4>
              {Object.entries(selectedToken.config).map(([k, v]) =>
                typeof v === "boolean" ? (
                  <label key={k} className="check" title={CONFIG_TIPS[k] ?? k}>
                    <input
                      type="checkbox"
                      checked={v}
                      onChange={(e) => updateSelectedTokenConfig(k, e.target.checked)}
                    />
                    <span>{CONFIG_LABELS[k] ?? k.replaceAll("_", " ")}</span>
                  </label>
                ) : (
                  <label key={k} title={CONFIG_TIPS[k] ?? k}>
                    <span>{CONFIG_LABELS[k] ?? k.replaceAll("_", " ")}</span>
                    <input
                      value={String(v ?? "")}
                      onChange={(e) => updateSelectedTokenConfig(k, e.target.value)}
                    />
                  </label>
                ),
              )}
              <div className="row">
                <button
                  className="danger"
                  onClick={() => {
                    setTokens((s) => s.filter((t) => t.id !== selectedToken.id));
                    setSelectedTokenId(null);
                  }}
                >
                  Remove from map
                </button>
              </div>
            </div>
          )}
        </DraggablePanel>
      </main>

      {distanceModal.open && (
        <section className="modal-backdrop" onClick={() => setDistanceModal((s) => ({ ...s, open: false }))}>
          <article className="modal" onClick={(e) => e.stopPropagation()}>
            <h3>Geodesic distance</h3>
            <p className="distance">{distanceModal.km.toFixed(3)} km</p>
            <p>1 · {distanceModal.a}</p>
            <p>2 · {distanceModal.b}</p>
            <button onClick={() => setDistanceModal((s) => ({ ...s, open: false }))}>Done</button>
          </article>
        </section>
      )}
    </div>
  );
}
