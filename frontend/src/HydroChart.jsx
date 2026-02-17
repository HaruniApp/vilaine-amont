import { createMemo, Show, For } from 'solid-js';
import uPlot from 'uplot';
import { SolidUplot, createPluginBus } from '@dschz/solid-uplot';
import { cursor, tooltip, focusSeries } from '@dschz/solid-uplot/plugins';

function Tooltip(props) {
  const idx = () => props.cursor.idx;
  return (
    <div style={{
      background: "rgba(255,255,255,0.92)",
      "backdrop-filter": "blur(8px)",
      "-webkit-backdrop-filter": "blur(8px)",
      padding: "10px 14px",
      border: "none",
      "border-radius": "12px",
      "font-size": "13px",
      "box-shadow": "0 4px 20px rgba(0,0,0,0.1)",
    }}>
      <div style={{ "margin-bottom": "4px", "font-weight": "bold" }}>
        {new Date(props.cursor.xValue * 1000).toLocaleString('fr-FR', { timeZone: 'Europe/Paris' })}
      </div>
      <For each={props.seriesData}>
        {(series) => {
          const value = () => props.u.data[series.seriesIdx]?.[idx()];
          return (
            <Show when={series.visible && value() != null}>
              <div style={{ color: series.stroke }}>
                {series.label}: {value()?.toFixed(2)}
              </div>
            </Show>
          );
        }}
      </For>
    </div>
  );
}

function extractPoints(apiResponse) {
  return apiResponse?.series?.data ?? null;
}

const THRESHOLD_COLORS = ['#ef4444', '#8b5cf6', '#f97316'];

function extractThresholds(apiResponse) {
  const raw = apiResponse?.thresholds;
  if (!Array.isArray(raw) || raw.length === 0) return null;
  return raw.map((t, i) => ({
    label: t.label.replace(/^[^(]*\((.+)\)$/, '$1'),
    value: t.v / 1000,
    color: THRESHOLD_COLORS[i % THRESHOLD_COLORS.length],
  }));
}

function createThresholdsPlugin(getThresholds) {
  return {
    hooks: {
      draw: [
        (u) => {
          const thresholds = getThresholds();
          if (!thresholds) return;

          const ctx = u.ctx;
          const { left, top, width, height } = u.bbox;
          const dpr = devicePixelRatio;
          const fontSize = 11 * dpr;
          const lineHeight = fontSize + 2 * dpr;

          ctx.save();
          ctx.beginPath();
          ctx.rect(left, top, width, height);
          ctx.clip();

          const visible = thresholds
            .filter(t => {
              const y = u.valToPos(t.value, 'H', true);
              return y >= top && y <= top + height;
            })
            .sort((a, b) => b.value - a.value);

          for (let i = 0; i < visible.length; i++) {
            const { value, color } = visible[i];
            const y = u.valToPos(value, 'H', true);

            ctx.strokeStyle = color;
            ctx.lineWidth = 1.5 * dpr;
            ctx.setLineDash([6 * dpr, 4 * dpr]);
            ctx.beginPath();
            ctx.moveTo(left, y);
            ctx.lineTo(left + width, y);
            ctx.stroke();

            ctx.setLineDash([]);
          }

          ctx.restore();
        },
      ],
    },
  };
}

function buildData(dataH, dataQ, forecast) {
  const pointsH = extractPoints(dataH);
  const pointsQ = extractPoints(dataQ);
  const timestamps = [];
  const valuesH = [];
  const valuesQ = [];

  if (pointsH && pointsH.length > 0) {
    for (const point of pointsH) {
      const ts = Math.floor(new Date(point.t).getTime() / 1000);
      timestamps.push(ts);
      valuesH.push(point.v / 1000);
    }

    if (pointsQ && pointsQ.length > 0) {
      const qMap = new Map();
      for (const point of pointsQ) {
        const ts = Math.floor(new Date(point.t).getTime() / 1000);
        qMap.set(ts, point.v);
      }
      for (const ts of timestamps) {
        const qv = qMap.get(ts);
        valuesQ.push(qv != null ? qv / 1000 : null);
      }
    } else {
      for (let i = 0; i < timestamps.length; i++) valuesQ.push(null);
    }
  } else if (pointsQ && pointsQ.length > 0) {
    for (const point of pointsQ) {
      timestamps.push(Math.floor(new Date(point.t).getTime() / 1000));
      valuesH.push(null);
      valuesQ.push(point.v / 1000);
    }
  }

  // Build forecast series
  const valuesForecast = new Array(timestamps.length).fill(null);

  if (forecast?.forecasts?.length && timestamps.length > 0) {
    // Start with the last observed H value for visual continuity
    const lastHVal = valuesH[valuesH.length - 1];
    const lastHTs = timestamps[timestamps.length - 1];
    if (lastHVal != null) {
      valuesForecast[valuesForecast.length - 1] = lastHVal;
    }

    // Add forecast points at future timestamps
    for (const f of forecast.forecasts) {
      const ts = Math.floor(new Date(f.t).getTime() / 1000);
      timestamps.push(ts);
      valuesH.push(null);
      valuesQ.push(null);
      valuesForecast.push(f.v);
    }
  }

  return [timestamps, valuesH, valuesQ, valuesForecast];
}

export default function HydroChart(props) {
  const bus = createPluginBus();

  const chartData = createMemo(() => buildData(props.dataH, props.dataQ, props.forecast));

  const thresholds = createMemo(() => extractThresholds(props.dataH));

  const hasData = createMemo(() => {
    const d = chartData();
    return d[0].length > 0;
  });

  const plugins = [
    cursor(),
    focusSeries({ pxThreshold: 15 }),
    tooltip(Tooltip, { placement: "top-right", zIndex: 20 }),
    createThresholdsPlugin(() => thresholds()),
  ];

  const series = [
    {},
    {
      label: "Hauteur (m)",
      stroke: "#3b82f6",
      width: 2.5,
      scale: "H",
      value: (u, v) => v == null ? "--" : v.toFixed(2),
    },
    {
      label: "Debit (m\u00b3/s)",
      stroke: "#f97316",
      width: 2.5,
      scale: "Q",
    },
    {
      label: "Prediction H (m)",
      stroke: "#0d9488",
      width: 2,
      dash: [6, 4],
      scale: "H",
      value: (u, v) => v == null ? "--" : v.toFixed(2),
    },
  ];

  let hRangeRatio = 1;

  const scales = {
    x: { time: true },
    H: {
      auto: true,
      side: 3,
      range: (u, min, max) => {
        const dataMin = Math.min(min, 0);
        const dataRange = max - dataMin;
        const t = thresholds();
        if (t) {
          for (const { value } of t) {
            if (value > max) max = value;
          }
        }
        const pad = (max - dataMin) * 0.05;
        const finalMin = dataMin;
        const finalMax = max + pad;
        hRangeRatio = dataRange > 0 ? (finalMax - finalMin) / dataRange : 1;
        return [finalMin, finalMax];
      },
    },
    Q: {
      auto: true,
      side: 1,
      range: (u, min, max) => {
        const pad = (max - min) * 0.05;
        const baseMin = min - pad;
        const baseRange = (max + pad) - baseMin;
        return [baseMin, baseMin + baseRange * hRangeRatio];
      },
    },
  };

  const tzDate = ts => uPlot.tzDate(new Date(ts * 1000), 'Europe/Paris');

  const axisFont = '12px "Inter", -apple-system, BlinkMacSystemFont, sans-serif';
  const axisLabelFont = '12px "Inter", -apple-system, BlinkMacSystemFont, sans-serif';

  const axes = [
    {
      font: axisFont,
      stroke: "#86868b",
      ticks: { stroke: "#e5e5ea", width: 1 },
      grid: { stroke: "#f0f0f2", width: 1 },
      values: [
        [3600*24*365, "{YYYY}",     null,                          null, null,               null, null, null, 1],
        [3600*24*28,  "{MMM}",      "\n{YYYY}",                    null, null,               null, null, null, 1],
        [3600*24,     "{DD}/{MM}",  "\n{YYYY}",                    null, null,               null, null, null, 1],
        [3600,        "{HH}h",      "\n{DD}/{MM}/{YY}",            null, "\n{DD}/{MM}",      null, null, null, 1],
        [60,          "{HH}:{mm}",  "\n{DD}/{MM}/{YY}",            null, "\n{DD}/{MM}",      null, null, null, 1],
        [1,           ":{ss}",      "\n{DD}/{MM}/{YY} {HH}:{mm}",  null, "\n{DD}/{MM} {HH}:{mm}", null, "\n{HH}:{mm}", null, 1],
      ],
    },
    {
      scale: "H",
      side: 3,
      label: "Hauteur (m)",
      labelFont: axisLabelFont,
      font: axisFont,
      stroke: "#3b82f6",
      ticks: { stroke: "#e5e5ea", width: 1 },
      grid: { show: true, stroke: "#f0f0f2", width: 1 },
      values: (u, vals) => vals.map(v => v.toFixed(2)),
    },
    {
      scale: "Q",
      side: 1,
      label: "Debit (m\u00b3/s)",
      labelFont: axisLabelFont,
      font: axisFont,
      stroke: "#f97316",
      ticks: { stroke: "#e5e5ea", width: 1 },
      grid: { show: false },
      splits: (u) => {
        const hSplits = u.axes[1]._splits;
        if (!hSplits || hSplits.length < 2) return;
        const hScale = u.scales.H;
        const qScale = u.scales.Q;
        const hMin = hScale.min;
        const hMax = hScale.max;
        const hRange = hMax - hMin;
        if (hRange <= 0) return;
        return hSplits
          .filter(v => v >= hMin - 1e-9 && v <= hMax + 1e-9)
          .map(v => qScale.min + (v - hMin) / hRange * (qScale.max - qScale.min));
      },
      values: (u, vals) => vals.map(v => v.toFixed(1)),
    },
  ];

  return (
    <Show when={hasData()} fallback={<p>Aucune donnee a afficher.</p>}>
      <div style={{ width: "100%", height: "400px", "margin-top": "1rem" }}>
        <SolidUplot
          data={chartData()}
          series={series}
          scales={scales}
          axes={axes}
          tzDate={tzDate}
          autoResize={true}
          plugins={plugins}
          pluginBus={bus}
          resetScales={true}
        />
      </div>
      <Show when={thresholds()}>
        <div style={{ display: "flex", "flex-wrap": "wrap", "justify-content": "center", gap: "0.75rem 1.5rem", "margin-top": "0.5rem", "font-size": "12px" }}>
          <For each={thresholds()}>
            {(t) => (
              <div style={{ display: "flex", "align-items": "center", gap: "6px" }}>
                <span style={{
                  display: "inline-block",
                  width: "24px",
                  height: "0",
                  "border-top": `2px dashed ${t.color}`,
                }} />
                <span style={{ color: "#86868b" }}>{t.label} ({t.value.toFixed(2)} m)</span>
              </div>
            )}
          </For>
        </div>
      </Show>
    </Show>
  );
}
