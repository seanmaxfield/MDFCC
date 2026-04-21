import { useMemo, useRef, useState } from "react";
import { Circle, MapContainer, Marker, Polygon, Popup, TileLayer, Tooltip, useMapEvents } from "react-leaflet";
import L, { type DivIcon } from "leaflet";
import type { LatLngExpression } from "leaflet";

type TokenKind = "ilp" | "mlp" | "gbewr";
type Mode = "none" | "add_ilp" | "add_mlp" | "add_gbewr" | "distance" | "coverage";
type LibraryKind = "missile" | "interceptor" | "gbewr";

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

  const selectedToken = tokens.find((t) => t.id === selectedTokenId) ?? null;

  const libData = activeLibTab === "missile" ? missiles : activeLibTab === "interceptor" ? interceptors : radars;
  const selectedLibItem = libData[libSelection] ?? null;
  const selectedLibEntries = selectedLibItem ? Object.entries(selectedLibItem) : [];

  const coverageCircles = useMemo(
    () =>
      tokens
        .filter((t) => t.kind === "mlp")
        .flatMap((t) =>
          t.missileKeys.map((mKey) => ({
            id: `${t.id}-${mKey}`,
            center: [t.lat, t.lon] as LatLngExpression,
            km: getMissileRangeKm(missiles, mKey),
          })),
        ),
    [tokens, missiles],
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
  }

  function updateSelectedToken<K extends keyof Token>(key: K, value: Token[K]) {
    if (!selectedToken) return;
    setTokens((s) => s.map((t) => (t.id === selectedToken.id ? { ...t, [key]: value } : t)));
  }

  return (
    <div className="app">
      <header className="toolbar">
        <button title="Coverage walkthrough tool." onClick={() => setMode("coverage")}>
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
        <span className="status">{status}</span>
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
            <Marker key={t.id} position={[t.lat, t.lon]} icon={markerIcon(t.kind)} eventHandlers={{ click: () => setSelectedTokenId(t.id), dragend: (e) => {
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

          {distancePts.map((p, idx) => (
            <Marker key={`d-${idx}`} position={[p.lat, p.lon]} icon={L.divIcon({ className: "", html: `<div class="dist-pin">${idx + 1}</div>`, iconSize: [24, 24], iconAnchor: [12, 12] })} />
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
                </>
              )}
              {selectedToken.kind === "mlp" && (
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
