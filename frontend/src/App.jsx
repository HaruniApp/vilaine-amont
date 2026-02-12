import { createSignal, For } from 'solid-js';
import 'uplot/dist/uPlot.min.css';
import HydroChart from './HydroChart';

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
  { id: 'J709063002', label: 'Cesson-Sevigne - Pont Briand - (Vilaine)' },
  { id: 'J706062001', label: 'Chateaubourg - Bel Air - (Vilaine)' },
  { id: 'J705302001', label: 'Poce-les-Bois - barrage Villaumur - (Cantache)' },
  { id: 'J701061001', label: 'Vitre - Bas Pont - (Vilaine)' },
  { id: 'J704301001', label: 'Taillis - La Basse Moliere - (Cantache)' },
  { id: 'J702402001', label: 'Vitre - Le Chateau des Rochers - aval retenue Valiere - (Valiere)' },
  { id: 'J701064001', label: 'La Chapelle-Erbree - barrage Vilaine - (Vilaine)' },
  { id: 'J702401001', label: 'Erbree - Les Ravenieres - (Valiere)' },
];

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
    <div style={{ "max-width": "900px", margin: "2rem auto", "font-family": "sans-serif" }}>
      <h1>Vigicrue</h1>
      <div style={{ display: "flex", gap: "1rem", "flex-wrap": "wrap", "align-items": "end" }}>
        <label>
          Station
          <br />
          <select value={stationId()} onChange={(e) => { setStationId(e.target.value); fetchData(); }}>
            <For each={STATIONS}>
              {(s) => <option value={s.id}>{s.label}</option>}
            </For>
          </select>
        </label>
        <label>
          Debut
          <br />
          <input type="date" value={startDate()} onInput={(e) => setStartDate(e.target.value)} />
        </label>
        <label>
          Fin
          <br />
          <input type="date" value={endDate()} onInput={(e) => setEndDate(e.target.value)} />
        </label>
        <button onClick={fetchData} disabled={loading()}>
          {loading() ? 'Chargement...' : 'Recharger'}
        </button>
      </div>

      {error() && <p style={{ color: "red" }}>Erreur : {error()}</p>}

      {dataH()?.series?.title && (
        <p style={{ "font-size": "13px", color: "#666", "margin-top": "0.75rem" }}>
          {dataH().series.title.replace(/^.* - [A-Z]\d{3} \d{4} \d{2} - /, '')}
        </p>
      )}

      {(dataH() || dataQ()) && <HydroChart dataH={dataH()} dataQ={dataQ()} />}
    </div>
  );
}
