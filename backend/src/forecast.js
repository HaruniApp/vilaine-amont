import { readFile } from 'fs/promises';
import { existsSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

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

let session = null;
let meta = null;
let normParams = null;

async function init() {
  if (session) return;

  const onnxPath = join(MODELS_DIR, 'tft.onnx');
  if (!existsSync(onnxPath)) {
    throw Object.assign(new Error('model_not_found'), { code: 'MODEL_NOT_FOUND' });
  }

  const ort = await import('onnxruntime-node');
  session = await ort.InferenceSession.create(onnxPath);
  meta = JSON.parse(await readFile(join(MODELS_DIR, 'tft_meta.json'), 'utf-8'));
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
    const response = await fetch(url, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json, text/javascript, */*; q=0.01',
        'Referer': `https://www.hydro.eaufrance.fr/stationhydro/${stationId}/series`,
        'X-Requested-With': 'XMLHttpRequest',
      },
    });

    if (!response.ok) {
      console.warn(`  Hydro ${stationId} ${variable}: HTTP ${response.status}`);
      return [];
    }
    const data = await response.json();
    const points = data?.series?.data ?? [];
    console.log(`  Hydro ${stationId} ${variable}: ${points.length} pts`);
    return points;
  } catch (err) {
    console.warn(`  Hydro ${stationId} ${variable}: ${err.message}`);
    return [];
  }
}

async function fetchPrecipitation(lat, lon, pastHours) {
  const url = `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&past_hours=${pastHours}&forecast_hours=0&hourly=precipitation&timezone=Europe%2FParis`;
  const response = await fetch(url);
  if (!response.ok) return [];
  const data = await response.json();
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
  // Build a map from hour-rounded timestamp to value
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
      // Pad with last known value
      result[i] = lastVal;
    }
  }

  // Fill leading nulls with first known value
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
    // Open-Meteo returns "2024-01-01T00:00" format
    const ts = new Date(p.t).getTime();
    map.set(ts, p.v ?? 0);
  }

  const result = new Array(hourlyTimestamps.length);
  for (let i = 0; i < hourlyTimestamps.length; i++) {
    result[i] = map.get(hourlyTimestamps[i].getTime()) ?? 0;
  }
  return result;
}

function computeDerivative(values) {
  const d = new Array(values.length);
  d[0] = 0;
  for (let i = 1; i < values.length; i++) {
    d[i] = (values[i] ?? 0) - (values[i - 1] ?? 0);
  }
  return d;
}

function normalize(value, min, max) {
  if (max === min || isNaN(min) || isNaN(max)) return 0;
  return (value - min) / (max - min);
}

export async function forecast(stationId) {
  await init();

  if (meta.target_station !== stationId) {
    return { forecasts: [], target_station: meta.target_station, model: 'tft', info: 'predictions_only_for_target' };
  }

  const inputWindow = meta.input_window; // 72
  const now = new Date();
  const lastHour = roundToHour(now);
  const startDate = new Date(lastHour.getTime() - (inputWindow + 1) * 3600000);
  const hourlyTimestamps = buildHourlyIndex(inputWindow, lastHour);

  const startStr = formatDateHydro(startDate);
  const endStr = formatDateHydro(now);

  // Fetch hydro data station by station (avoid rate-limiting)
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

  // Fetch precipitation for all stations (Open-Meteo is more tolerant)
  console.log('Fetching precipitation...');
  const precipResults = await Promise.all(
    STATION_CODES.map(code => {
      const { lat, lon } = STATION_COORDS[code];
      return fetchPrecipitation(lat, lon, inputWindow + 2);
    })
  );

  // Organize hydro data: station -> { h: [], q: [] }
  const stationData = {};
  for (let i = 0; i < STATION_CODES.length; i++) {
    const code = STATION_CODES[i];
    stationData[code] = {
      h: alignToHourlyGrid(hydroResults[i * 2], hourlyTimestamps),
      q: alignToHourlyGrid(hydroResults[i * 2 + 1], hourlyTimestamps),
      precip: alignPrecipToGrid(precipResults[i], hourlyTimestamps),
    };
  }

  // Compute derivatives
  for (const code of STATION_CODES) {
    stationData[code].dh = computeDerivative(stationData[code].h);
    stationData[code].dq = computeDerivative(stationData[code].q);
  }

  // Compute temporal features
  const hourSin = new Array(inputWindow);
  const hourCos = new Array(inputWindow);
  const doySin = new Array(inputWindow);
  const doyCos = new Array(inputWindow);

  for (let i = 0; i < inputWindow; i++) {
    const dt = hourlyTimestamps[i];
    // Use UTC hours to match training data (prepare_dataset uses tz-naive UTC timestamps)
    const hour = dt.getUTCHours();
    const startOfYear = new Date(Date.UTC(dt.getUTCFullYear(), 0, 0));
    const doy = Math.floor((dt - startOfYear) / 86400000);
    hourSin[i] = Math.sin(2 * Math.PI * hour / 24);
    hourCos[i] = Math.cos(2 * Math.PI * hour / 24);
    doySin[i] = Math.sin(2 * Math.PI * doy / 365.25);
    doyCos[i] = Math.cos(2 * Math.PI * doy / 365.25);
  }

  // Build feature map
  const featureMap = {};
  for (const code of STATION_CODES) {
    featureMap[`${code}_h`] = stationData[code].h;
    featureMap[`${code}_q`] = stationData[code].q;
    featureMap[`${code}_precip`] = stationData[code].precip;
    featureMap[`${code}_dh`] = stationData[code].dh;
    featureMap[`${code}_dq`] = stationData[code].dq;
  }
  featureMap['hour_sin'] = hourSin;
  featureMap['hour_cos'] = hourCos;
  featureMap['doy_sin'] = doySin;
  featureMap['doy_cos'] = doyCos;

  // Build tensor: shape (1, 72, 59), ordered by feature_names
  const tensorData = new Float32Array(inputWindow * meta.n_features);
  for (let t = 0; t < inputWindow; t++) {
    for (let f = 0; f < meta.feature_names.length; f++) {
      const fname = meta.feature_names[f];
      const raw = featureMap[fname]?.[t] ?? 0;
      const np = normParams[fname];
      const normalized = normalize(raw, np?.min, np?.max);
      tensorData[t * meta.n_features + f] = normalized;
    }
  }

  // --- DEBUG: log target station raw values + last timestep features ---
  const targetCode = meta.target_station;
  const lastT = inputWindow - 1;
  const targetH = stationData[targetCode].h;
  const targetQ = stationData[targetCode].q;
  console.log('\n=== FORECAST DEBUG ===');
  console.log(`Grid: ${hourlyTimestamps[0].toISOString()} → ${hourlyTimestamps[lastT].toISOString()}`);
  console.log(`${targetCode}_h raw (last 6h):`, targetH.slice(-6).map(v => `${v} mm (${(v/1000).toFixed(3)} m)`));
  console.log(`${targetCode}_q raw (last 6h):`, targetQ.slice(-6));
  console.log(`Precip ${targetCode} (last 6h):`, stationData[targetCode].precip.slice(-6));
  console.log(`dH (last 6h):`, stationData[targetCode].dh.slice(-6));

  // Log normalized values for target_h at last timestep
  const targetHIdx = meta.feature_names.indexOf(`${targetCode}_h`);
  const normAtLast = tensorData[lastT * meta.n_features + targetHIdx];
  const np = normParams[`${targetCode}_h`];
  console.log(`\nNorm params ${targetCode}_h: min=${np.min}, max=${np.max}`);
  console.log(`Raw H at t=${lastT}: ${targetH[lastT]} mm → normalized: ${normAtLast.toFixed(6)}`);

  // Log all features at last timestep (name: raw → normalized)
  console.log(`\nAll features at last timestep (t=${lastT}):`);
  for (let f = 0; f < meta.feature_names.length; f++) {
    const fname = meta.feature_names[f];
    const raw = featureMap[fname]?.[lastT] ?? 0;
    const norm = tensorData[lastT * meta.n_features + f];
    console.log(`  ${fname}: raw=${raw} → norm=${norm.toFixed(6)}`);
  }
  // --- END DEBUG ---

  const ort = await import('onnxruntime-node');
  const inputTensor = new ort.Tensor('float32', tensorData, [1, inputWindow, meta.n_features]);
  const results = await session.run({ input: inputTensor });
  const predictions = results.predictions.data;

  // Denormalize: predictions are normalized target (J706062001_h)
  const targetNorm = normParams[`${meta.target_station}_h`];
  const targetMin = targetNorm.min;
  const targetMax = targetNorm.max;

  console.log(`\nRaw model output: [${Array.from(predictions).map(v => v.toFixed(6)).join(', ')}]`);

  const forecasts = meta.forecast_horizons.map((h, i) => {
    const rawMm = predictions[i] * (targetMax - targetMin) + targetMin;
    const meters = rawMm / 1000;
    const timestamp = new Date(lastHour.getTime() + h * 3600000);
    console.log(`  ${`t+${h}h`.padEnd(5)}: norm=${predictions[i].toFixed(6)} → ${rawMm.toFixed(1)} mm → ${meters.toFixed(3)} m`);
    return {
      t: timestamp.toISOString(),
      v: meters,
      horizon: `t+${h}h`,
    };
  });

  console.log('=== END DEBUG ===\n');

  return { forecasts, target_station: meta.target_station, model: 'tft' };
}
