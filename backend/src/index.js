import express from 'express';
import cors from 'cors';

const app = express();
const PORT = process.env.PORT || 3001;

app.use(cors());

app.get('/api/station/:stationId/series', async (req, res) => {
  const { stationId } = req.params;
  const { startAt, endAt, variable = 'H' } = req.query;

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
      return res.status(response.status).json({ error: `API returned ${response.status}` });
    }

    const data = await response.json();
    res.json(data);
  } catch (err) {
    console.error('Proxy error:', err.message);
    res.status(500).json({ error: err.message });
  }
});

app.listen(PORT, () => {
  console.log(`Backend running on http://localhost:${PORT}`);
});
