import { createSignal, For } from 'solid-js';
import 'uplot/dist/uPlot.min.css';
import HydroChart from './HydroChart';
import networkSvg from './assets/network.svg';

function formatDate(date) {
  const d = date.getDate().toString().padStart(2, '0');
  const m = (date.getMonth() + 1).toString().padStart(2, '0');
  const y = date.getFullYear();
  return `${d}/${m}/${y}`;
}

function defaultDates() {
  const end = new Date();
  const start = new Date();
  start.setDate(start.getDate() - 7);
  return {
    start: start.toISOString().slice(0, 10),
    end: end.toISOString().slice(0, 10),
  };
}

const STATIONS = [
  { id: 'J700061001', label: 'Bourgon - Pont D 106 - (Vilaine)' },
  { id: 'J701064001', label: 'La Chapelle-Erbree - barrage Vilaine - (Vilaine)' },
  { id: 'J702401001', label: 'Erbree - Les Ravenieres - (Valiere)' },
  { id: 'J702403001', label: 'Erbree - retenue Valiere - (Valiere)' },
  { id: 'J702402001', label: 'Vitre - Chateau des Rochers - aval retenue - (Valiere)' },
  { id: 'J701061001', label: 'Vitre - Bas Pont - (Vilaine)' },
  { id: 'J704301001', label: 'Taillis - La Basse Moliere - (Cantache)' },
  { id: 'J705302001', label: 'Poce-les-Bois - barrage Villaumur - (Cantache)' },
  { id: 'J706062001', label: 'Chateaubourg - Bel Air - (Vilaine)' },
  { id: 'J708311001', label: 'La Bouexiere - Le Drugeon - (Veuvre/Chevre)' },
  { id: 'J709063002', label: 'Cesson-Sevigne - Pont Briand - (Vilaine)' },
];

const fontFamily = '"Inter", -apple-system, BlinkMacSystemFont, "SF Pro Display", sans-serif';

const inputStyle = {
  appearance: "none",
  "-webkit-appearance": "none",
  background: "#fff",
  border: "1px solid #d2d2d7",
  "border-radius": "10px",
  padding: "10px 14px",
  "font-size": "15px",
  "font-family": "inherit",
  color: "#1d1d1f",
  outline: "none",
};

const labelStyle = {
  "font-size": "12px",
  "font-weight": "600",
  "text-transform": "uppercase",
  "letter-spacing": "0.5px",
  color: "#86868b",
};

export default function App() {
  const defaults = defaultDates();
  const [stationId, setStationId] = createSignal('J706062001');
  const [startDate, setStartDate] = createSignal(defaults.start);
  const [endDate, setEndDate] = createSignal(defaults.end);
  const [loading, setLoading] = createSignal(false);
  const [error, setError] = createSignal(null);
  const [dataH, setDataH] = createSignal(null);
  const [dataQ, setDataQ] = createSignal(null);

  fetchData();

  async function fetchData() {
    setLoading(true);
    setError(null);
    setDataH(null);
    setDataQ(null);

    const start = formatDate(new Date(startDate()));
    const end = formatDate(new Date(endDate()));

    const buildUrl = (variable) => {
      const params = new URLSearchParams({ startAt: start, endAt: end, variable });
      return `/api/station/${stationId()}/series?${params}`;
    };

    try {
      const [resH, resQ] = await Promise.all([
        fetch(buildUrl('H')),
        fetch(buildUrl('Q')),
      ]);

      if (!resH.ok && !resQ.ok) throw new Error(`Erreur H:${resH.status} Q:${resQ.status}`);

      const jsonH = resH.ok ? await resH.json() : null;
      const jsonQ = resQ.ok ? await resQ.json() : null;

      setDataH(jsonH);
      setDataQ(jsonQ);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{
      "max-width": "960px",
      margin: "3rem auto",
      padding: "2rem",
      "font-family": fontFamily,
      background: "#fff",
      "border-radius": "20px",
      "box-shadow": "0 1px 3px rgba(0,0,0,0.06), 0 8px 24px rgba(0,0,0,0.06)",
    }}>
      <h1 style={{
        "font-weight": "700",
        "font-size": "28px",
        "letter-spacing": "-0.5px",
        color: "#1d1d1f",
        "margin-top": "0",
        "margin-bottom": "1.5rem",
      }}>Vigicrue</h1>

      <div style={{
        background: "#f5f5f7",
        "border-radius": "14px",
        padding: "1.25rem 1.5rem",
      }}>
        <div style={{ display: "flex", gap: "1rem", "flex-wrap": "wrap", "align-items": "end" }}>
          <label style={{ flex: "1 1 280px" }}>
            <div style={labelStyle}>Station</div>
            <select
              value={stationId()}
              onChange={(e) => { setStationId(e.target.value); fetchData(); }}
              style={{ ...inputStyle, width: "100%", cursor: "pointer" }}
            >
              <For each={STATIONS}>
                {(s) => <option value={s.id}>{s.label}</option>}
              </For>
            </select>
          </label>
          <label>
            <div style={labelStyle}>Debut</div>
            <input
              type="date"
              value={startDate()}
              onInput={(e) => setStartDate(e.target.value)}
              style={inputStyle}
            />
          </label>
          <label>
            <div style={labelStyle}>Fin</div>
            <input
              type="date"
              value={endDate()}
              onInput={(e) => setEndDate(e.target.value)}
              style={inputStyle}
            />
          </label>
          <button
            onClick={fetchData}
            disabled={loading()}
            style={{
              "align-self": "end",
              background: loading() ? "#a1a1aa" : "#1d1d1f",
              color: "#fff",
              border: "none",
              "border-radius": "10px",
              padding: "10px 20px",
              height: "41px",
              "font-weight": "600",
              "font-size": "15px",
              "font-family": "inherit",
              cursor: loading() ? "default" : "pointer",
              transition: "background 0.2s ease",
            }}
            onMouseEnter={(e) => { if (!loading()) e.target.style.background = "#48484a"; }}
            onMouseLeave={(e) => { if (!loading()) e.target.style.background = "#1d1d1f"; }}
          >
            {loading() ? 'Chargement...' : 'Recharger'}
          </button>
        </div>
      </div>

      {error() && <p style={{ color: "#ef4444", "font-size": "14px", "margin-top": "1rem" }}>Erreur : {error()}</p>}

      {dataH()?.series?.title && (
        <p style={{
          "font-size": "14px",
          color: "#86868b",
          "font-weight": "500",
          "margin-top": "1rem",
          "margin-bottom": "0",
          "text-align": "center",
        }}>
          {dataH().series.title.replace(/^.* - [A-Z]\d{3} \d{4} \d{2} - /, '')}
        </p>
      )}

      {(dataH() || dataQ()) && <HydroChart dataH={dataH()} dataQ={dataQ()} />}

      <details style={{
        "margin-top": "1.5rem",
        background: "#f5f5f7",
        "border-radius": "14px",
        overflow: "hidden",
      }}>
        <summary style={{
          padding: "0.75rem 1.25rem",
          cursor: "pointer",
          "font-size": "13px",
          "font-weight": "600",
          color: "#6e6e73",
          "letter-spacing": "0.3px",
          "list-style": "none",
          display: "flex",
          "align-items": "center",
          gap: "0.5rem",
          "user-select": "none",
        }}>
          <span style={{
            display: "inline-block",
            transition: "transform 0.2s ease",
            "font-size": "10px",
          }} class="details-arrow">&#9654;</span>
          Réseau hydrographique — Vilaine Amont
        </summary>
        <div style={{ padding: "0 1rem 1rem" }}>
          <img src={networkSvg} alt="Réseau hydrographique du bassin Vilaine Amont" style={{ width: "100%" }} />
        </div>
      </details>

      <p style={{
        "text-align": "center",
        "font-size": "11px",
        color: "#aeaeb2",
        "letter-spacing": "0.3px",
        "margin-top": "2rem",
      }}>
        v{import.meta.env.VITE_APP_VERSION || "dev"}
      </p>
    </div>
  );
}
