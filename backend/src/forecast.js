import { readFile } from 'fs/promises';
import { existsSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { cachedFetch } from './cache.js';

const TTL_30MIN = 30 * 60 * 1000;

const __dirname = dirname(fileURLToPath(import.meta.url));
const MODELS_DIR = join(__dirname, '..', 'models');

// Station coordinates for Open-Meteo precipitation
const STATION_COORDS = {
  J700061001: { lat: 48.187, lon: -1.058 },
  J701064001: { lat: 48.146, lon: -1.128 },
  J702401001: { lat: 48.109, lon: -1.127 },
  J702403001: { lat: 48.083, lon: -1.164 },
  J702402001: { lat: 48.077, lon: -1.166 },
  J701061001: { lat: 48.123, lon: -1.223 },
  J704301001: { lat: 48.163, lon: -1.225 },
  J705302001: { lat: 48.128, lon: -1.293 },
  J706062001: { lat: 48.108, lon: -1.404 },
  J708311001: { lat: 48.183, lon: -1.499 },
  J709063002: { lat: 48.124, lon: -1.564 },
};

// Stations without Q data (barrages + Taillis)
const STATIONS_NO_Q = new Set(['J701064001', 'J702403001', 'J705302001', 'J704301001']);

const STATION_CODES = [
  'J700061001', 'J701064001', 'J702401001', 'J702403001', 'J702402001',
  'J701061001', 'J704301001', 'J705302001', 'J706062001', 'J708311001',
  'J709063002',
];

const BARRAGE_CODES = new Set(['J701064001', 'J702403001', 'J705302001']);

// Clipping des outliers
const DH_CLIP = 100;
const DQ_CLIP = 2000;

// Vars per station (padded, uniform)
const VARS_PER_STATION = 5;

let session = null;
let meta = null;
let normParams = null;

async function init() {
  if (session) return;

  const onnxPath = join(MODELS_DIR, 'station_attn.onnx');
  if (!existsSync(onnxPath)) {
    throw Object.assign(new Error('model_not_found'), { code: 'MODEL_NOT_FOUND' });
  }

  const ort = await import('onnxruntime-node');
  session = await ort.InferenceSession.create(onnxPath);
  meta = JSON.parse(await readFile(join(MODELS_DIR, 'station_attn_meta.json'), 'utf-8'));
  normParams = JSON.parse(await readFile(join(MODELS_DIR, 'norm_params.json'), 'utf-8'));
}

function formatDateHydro(date) {
  const d = date.getDate().toString().padStart(2, '0');
  const m = (date.getMonth() + 1).toString().padStart(2, '0');
  const y = date.getFullYear();
  return `${d}/${m}/${y}`;
}

async function fetchHydroSeries(stationId, startAt, endAt, variable) {
  const params = new URLSearchParams({
    'hydro_series[startAt]': startAt,
    'hydro_series[endAt]': endAt,
    'hydro_series[variableType]': 'simple_and_interpolated_and_hourly_variable',
    'hydro_series[simpleAndInterpolatedAndHourlyVariable]': variable,
    'hydro_series[statusData]': 'most_valid',
    'hydro_series[threshold]': '1',
  });

  const url = `https://www.hydro.eaufrance.fr/stationhydro/ajax/${stationId}/series?${params}`;
  try {
    const data = await cachedFetch(url, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json, text/javascript, */*; q=0.01',
        'Referer': `https://www.hydro.eaufrance.fr/stationhydro/${stationId}/series`,
        'X-Requested-With': 'XMLHttpRequest',
      },
    }, TTL_30MIN);

    if (!data) {
      console.warn(`  Hydro ${stationId} ${variable}: API error`);
      return [];
    }
    const points = data?.series?.data ?? [];
    console.log(`  Hydro ${stationId} ${variable}: ${points.length} pts`);
    return points;
  } catch (err) {
    console.warn(`  Hydro ${stationId} ${variable}: ${err.message}`);
    return [];
  }
}

async function fetchPrecipitation(lat, lon, pastHours, forecastHours) {
  const url = `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&past_hours=${pastHours}&forecast_hours=${forecastHours}&hourly=precipitation&timezone=Europe%2FParis`;
  const data = await cachedFetch(url, {}, TTL_30MIN);
  if (!data) return [];
  const times = data?.hourly?.time ?? [];
  const precip = data?.hourly?.precipitation ?? [];
  return times.map((t, i) => ({ t, v: precip[i] ?? 0 }));
}

function roundToHour(date) {
  const d = new Date(date);
  d.setMinutes(0, 0, 0);
  return d;
}

function buildHourlyIndex(count, lastHour) {
  const timestamps = [];
  for (let i = count - 1; i >= 0; i--) {
    timestamps.push(new Date(lastHour.getTime() - i * 3600000));
  }
  return timestamps;
}

function alignToHourlyGrid(points, hourlyTimestamps) {
  const map = new Map();
  for (const p of points) {
    const ts = roundToHour(new Date(p.t)).getTime();
    map.set(ts, p.v);
  }

  const result = new Array(hourlyTimestamps.length);
  let lastVal = null;
  for (let i = 0; i < hourlyTimestamps.length; i++) {
    const key = hourlyTimestamps[i].getTime();
    const val = map.get(key);
    if (val != null) {
      result[i] = val;
      lastVal = val;
    } else {
      result[i] = lastVal;
    }
  }

  if (result[0] == null) {
    const firstKnown = result.find(v => v != null) ?? 0;
    for (let i = 0; i < result.length; i++) {
      if (result[i] == null) result[i] = firstKnown;
      else break;
    }
  }

  return result;
}

function alignPrecipToGrid(points, hourlyTimestamps) {
  const map = new Map();
  for (const p of points) {
    const ts = new Date(p.t).getTime();
    map.set(ts, p.v ?? 0);
  }

  const result = new Array(hourlyTimestamps.length);
  for (let i = 0; i < hourlyTimestamps.length; i++) {
    result[i] = map.get(hourlyTimestamps[i].getTime()) ?? 0;
  }
  return result;
}

function computeCentralDerivative(values) {
  const d = new Array(values.length);
  d[0] = 0;
  for (let i = 1; i < values.length - 1; i++) {
    const prev = values[i - 1] ?? 0;
    const next = values[i + 1] ?? 0;
    d[i] = (next - prev) / 2.0;
  }
  d[values.length - 1] = 0;
  return d;
}

function normalize(value, min, max) {
  if (max === min || min == null || max == null || isNaN(min) || isNaN(max)) return 0;
  return (value - min) / (max - min);
}

export async function forecast(stationId) {
  await init();

  const inputWindow = meta.input_window; // 72
  const futureHours = meta.future_precip_hours; // 6
  const nStations = meta.n_stations; // 11
  const forecastHorizons = meta.forecast_horizons;
  const maxHorizon = Math.max(...forecastHorizons);
  const now = new Date();
  const lastHour = roundToHour(now);
  const startDate = new Date(lastHour.getTime() - (inputWindow + 1) * 3600000);
  const hourlyTimestamps = buildHourlyIndex(inputWindow, lastHour);

  // Future timestamps for model input (6h)
  const futureTimestamps = [];
  for (let h = 1; h <= futureHours; h++) {
    futureTimestamps.push(new Date(lastHour.getTime() + h * 3600000));
  }
  // Display future timestamps matching max forecast horizon (24h)
  const displayFutureTimestamps = [];
  for (let h = 1; h <= maxHorizon; h++) {
    displayFutureTimestamps.push(new Date(lastHour.getTime() + h * 3600000));
  }
  // Combined timestamps for precip fetch (past + display future)
  const allPrecipTimestamps = [...hourlyTimestamps, ...displayFutureTimestamps];

  const startStr = formatDateHydro(startDate);
  const endStr = formatDateHydro(now);

  // Fetch hydro data
  console.log('Fetching hydro data...');
  const hydroResults = [];
  for (const code of STATION_CODES) {
    const h = await fetchHydroSeries(code, startStr, endStr, 'H');
    hydroResults.push(h);
    if (!STATIONS_NO_Q.has(code)) {
      const q = await fetchHydroSeries(code, startStr, endStr, 'Q');
      hydroResults.push(q);
    } else {
      hydroResults.push([]);
    }
  }

  // Fetch precipitation with future forecast
  console.log('Fetching precipitation (past + forecast)...');
  const precipResults = await Promise.all(
    STATION_CODES.map(code => {
      const { lat, lon } = STATION_COORDS[code];
      return fetchPrecipitation(lat, lon, inputWindow + 2, maxHorizon + 1);
    })
  );

  // Organize data
  const stationData = {};
  for (let i = 0; i < STATION_CODES.length; i++) {
    const code = STATION_CODES[i];
    const hArr = alignToHourlyGrid(hydroResults[i * 2], hourlyTimestamps);
    const qArr = alignToHourlyGrid(hydroResults[i * 2 + 1], hourlyTimestamps);
    const allPrecip = alignPrecipToGrid(precipResults[i], allPrecipTimestamps);
    const precipPast = allPrecip.slice(0, inputWindow);
    const precipFuture = allPrecip.slice(inputWindow, inputWindow + futureHours);
    const precipFutureDisplay = allPrecip.slice(inputWindow, inputWindow + maxHorizon);

    stationData[code] = { h: hArr, q: qArr, precip: precipPast, precipFuture, precipFutureDisplay };
  }

  // Compute derivatives (central) + clip
  for (const code of STATION_CODES) {
    stationData[code].dh = computeCentralDerivative(stationData[code].h)
      .map(v => Math.max(-DH_CLIP, Math.min(DH_CLIP, v)));
    stationData[code].dq = computeCentralDerivative(stationData[code].q)
      .map(v => Math.max(-DQ_CLIP, Math.min(DQ_CLIP, v)));
  }

  // Compute release feature for barrages
  for (const code of BARRAGE_CODES) {
    const dh = stationData[code].dh;
    const precip = stationData[code].precip;
    stationData[code].release = dh.map((d, i) => {
      const neg_dh = Math.max(0, -(d ?? 0));
      const p = Math.min(1, precip[i] ?? 0);
      return neg_dh * (1 - p);
    });
  }

  // --- Build past tensor (padded: n_stations * 5 vars) ---
  // Slot order per station: 0=h, 1=q, 2=precip, 3=dh, 4=dq/release
  const nFeaturesPadded = nStations * VARS_PER_STATION;
  const pastTensor = new Float32Array(inputWindow * nFeaturesPadded);

  for (let t = 0; t < inputWindow; t++) {
    for (let s = 0; s < STATION_CODES.length; s++) {
      const code = STATION_CODES[s];
      const base = s * VARS_PER_STATION;
      const sd = stationData[code];
      const np_h = normParams[`${code}_h`];
      const np_precip = normParams[`${code}_precip`];
      const np_dh = normParams[`${code}_dh`];

      pastTensor[t * nFeaturesPadded + base + 0] = normalize(sd.h[t] ?? 0, np_h?.min, np_h?.max);
      pastTensor[t * nFeaturesPadded + base + 2] = normalize(sd.precip[t] ?? 0, np_precip?.min, np_precip?.max);
      pastTensor[t * nFeaturesPadded + base + 3] = normalize(sd.dh[t] ?? 0, np_dh?.min, np_dh?.max);

      if (!STATIONS_NO_Q.has(code)) {
        // Normal station: slot 1=q, slot 4=dq
        const np_q = normParams[`${code}_q`];
        const np_dq = normParams[`${code}_dq`];
        pastTensor[t * nFeaturesPadded + base + 1] = normalize(sd.q[t] ?? 0, np_q?.min, np_q?.max);
        pastTensor[t * nFeaturesPadded + base + 4] = normalize(sd.dq[t] ?? 0, np_dq?.min, np_dq?.max);
      } else if (BARRAGE_CODES.has(code)) {
        // Barrage: slot 1=0, slot 4=release
        const np_release = normParams[`${code}_release`];
        pastTensor[t * nFeaturesPadded + base + 1] = 0;
        pastTensor[t * nFeaturesPadded + base + 4] = normalize(sd.release?.[t] ?? 0, np_release?.min, np_release?.max);
      }
      // else no_q non-barrage (Taillis): slots 1 and 4 stay 0
    }
  }

  // --- Build future precip tensor (n_stations * future_hours) ---
  const futurePrecipTensor = new Float32Array(nStations * futureHours);
  for (let s = 0; s < STATION_CODES.length; s++) {
    const code = STATION_CODES[s];
    const np_precip = normParams[`${code}_precip`];
    const pf = stationData[code].precipFuture;
    for (let h = 0; h < futureHours; h++) {
      futurePrecipTensor[s * futureHours + h] = normalize(pf[h] ?? 0, np_precip?.min, np_precip?.max);
    }
  }

  // --- Run inference ---
  const ort = await import('onnxruntime-node');
  const pastInput = new ort.Tensor('float32', pastTensor, [1, inputWindow, nFeaturesPadded]);
  const futureInput = new ort.Tensor('float32', futurePrecipTensor, [1, nStations * futureHours]);
  const results = await session.run({ past_input: pastInput, future_precip: futureInput });
  const predictions = results.predictions.data;

  // --- Denormalize predictions for all stations ---
  const outputMap = meta.output_map;
  const rmseData = meta.rmse || {};
  const allForecasts = {};

  for (const code of STATION_CODES) {
    const om = outputMap[code];
    const np_h = normParams[`${code}_h`];
    const hRange = (np_h?.max ?? 0) - (np_h?.min ?? 0);
    const sd = stationData[code];
    const lastHNorm = normalize(sd.h[inputWindow - 1] ?? 0, np_h?.min, np_h?.max);
    const stationRmse = rmseData[code];

    const hForecasts = forecastHorizons.map((h, j) => {
      const delta = predictions[om.h_start + j];
      const rawMm = (lastHNorm + delta) * hRange + (np_h?.min ?? 0);
      const meters = rawMm / 1000;
      const timestamp = new Date(lastHour.getTime() + h * 3600000);
      const point = { t: timestamp.toISOString(), v: meters, horizon: `t+${h}h` };
      if (stationRmse?.h?.[j] != null) {
        const rmseM = stationRmse.h[j] / 1000; // mm → m
        point.v_lower = meters - rmseM;
        point.v_upper = meters + rmseM;
      }
      return point;
    });

    const entry = { h: hForecasts };

    if (om.q_start != null) {
      const np_q = normParams[`${code}_q`];
      const qRange = (np_q?.max ?? 0) - (np_q?.min ?? 0);
      const lastQNorm = normalize(sd.q[inputWindow - 1] ?? 0, np_q?.min, np_q?.max);

      entry.q = forecastHorizons.map((h, j) => {
        const delta = predictions[om.q_start + j];
        const rawLs = (lastQNorm + delta) * qRange + (np_q?.min ?? 0);
        const m3s = rawLs / 1000;
        const timestamp = new Date(lastHour.getTime() + h * 3600000);
        const point = { t: timestamp.toISOString(), v: m3s, horizon: `t+${h}h` };
        if (stationRmse?.q?.[j] != null) {
          const rmseM3s = stationRmse.q[j] / 1000; // L/s → m³/s
          point.v_lower = m3s - rmseM3s;
          point.v_upper = m3s + rmseM3s;
        }
        return point;
      });
    }

    // Raw precipitation data (mm/h) for chart display
    entry.precip = hourlyTimestamps.map((ts, i) => ({
      t: ts.toISOString(),
      v: stationData[code].precip[i] ?? 0,
    }));
    entry.precipFuture = displayFutureTimestamps.map((ts, i) => ({
      t: ts.toISOString(),
      v: stationData[code].precipFutureDisplay[i] ?? 0,
    }));

    allForecasts[code] = entry;
  }

  // Backward-compatible response: forecasts array for the requested station
  const stationForecasts = allForecasts[stationId]?.h ?? allForecasts[STATION_CODES[0]]?.h ?? [];

  console.log(`Forecast for ${stationId}:`, stationForecasts.map(f => `${f.horizon}: ${f.v.toFixed(3)}m`).join(', '));

  return {
    forecasts: stationForecasts,
    target_station: stationId,
    model: 'station_attn',
    all_stations: allForecasts,
  };
}
