import { defineConfig } from 'vite'
import path from 'path'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import fs from 'fs'

// Plugin to serve local data files as JSON API
function localDataPlugin() {
  const f1Root = path.resolve(__dirname, '..');
  return {
    name: 'local-data-api',
    configureServer(server: any) {
      // Serve JSON files from /f1/data/other/openf1/ and /f1/data/other/jolpica/
      server.middlewares.use('/api/local', (req: any, res: any, next: any) => {
        const url = new URL(req.url, 'http://localhost');
        const filePath = url.pathname.replace(/^\//, '');

        // Map routes to files
        const routes: Record<string, string> = {
          'openf1/sessions': 'data/other/openf1/sessions.json',
          'openf1/laps': 'data/other/openf1/laps.json',
          'openf1/position': 'data/other/openf1/position.json',
          'openf1/weather': 'data/other/openf1/weather.json',
          'openf1/intervals': 'data/other/openf1/intervals.json',
          'openf1/pit': 'data/other/openf1/pit.json',
          'openf1/stints': 'data/other/openf1/stints.json',
          'openf1/drivers': 'data/other/openf1/mclaren_drivers_by_session.json',
          'openf1/overtakes': 'data/other/openf1/overtakes.json',
          'openf1/race_control': 'data/other/openf1/race_control.json',
          'openf1/championship_drivers': 'data/other/openf1/championship_drivers.json',
          'openf1/championship_teams': 'data/other/openf1/championship_teams.json',
          'jolpica/driver_standings': 'data/other/jolpica/driver_standings.json',
          'jolpica/constructor_standings': 'data/other/jolpica/constructor_standings.json',
          'jolpica/race_results': 'data/other/jolpica/race_results.json',
          'jolpica/qualifying': 'data/other/jolpica/qualifying.json',
          'jolpica/circuits': 'data/other/jolpica/circuits.json',
          'jolpica/pit_stops': 'data/other/jolpica/pit_stops.json',
          'jolpica/lap_times': 'data/other/jolpica/lap_times.json',
          'jolpica/drivers': 'data/other/jolpica/drivers.json',
          'jolpica/seasons': 'data/other/jolpica/seasons.json',
          // f1data pipeline results
          'pipeline/gdino': 'f1data/McMedia/gdino_results/gdino_results.json',
          'pipeline/fused': 'f1data/McMedia/fused_results/fused_results.json',
          'pipeline/minicpm': 'f1data/McMedia/minicpm_results/minicpm_results.json',
          'pipeline/videomae': 'f1data/McMedia/videomae_results/videomae_results.json',
          'pipeline/timesformer': 'f1data/McMedia/timesformer_results/timesformer_results.json',
          // PDF extraction intelligence
          'pipeline/intelligence': 'pipeline/output/intelligence.json',
          // Anomaly detection scores
          'pipeline/anomaly': 'pipeline/output/anomaly_scores.json',
        };

        // Static CSV routes (small summary files from data/other/Mccsv/)
        const csvRoutes: Record<string, string> = {
          'mccsv/driver_career': 'data/other/Mccsv/driver_career_at_mclaren.csv',
          'mccsv/circuit_performance': 'data/other/Mccsv/circuit_performance.csv',
          'mccsv/drivers': 'data/other/Mccsv/drivers.csv',
          'mccsv/season_summary': 'data/other/Mccsv/season_summary.csv',
          'mccsv/race_weekend_summary': 'data/other/Mccsv/race_weekend_summary.csv',
        };

        const mapped = routes[filePath];
        if (mapped) {
          const fullPath = path.join(f1Root, mapped);
          if (fs.existsSync(fullPath)) {
            res.setHeader('Content-Type', 'application/json');
            res.end(fs.readFileSync(fullPath, 'utf-8'));
            return;
          }
        }

        // Serve static CSV summary files
        const csvMapped = csvRoutes[filePath];
        if (csvMapped) {
          const fullPath = path.join(f1Root, csvMapped);
          if (fs.existsSync(fullPath)) {
            res.setHeader('Content-Type', 'text/csv');
            res.end(fs.readFileSync(fullPath, 'utf-8'));
            return;
          }
        }

        // Dynamic per-race CSV routes with sampling for large files
        // Patterns: mccar/{year}/{race}.csv, mcdriver/{year}/{race}.csv, mcracecontext/{year}/{file}.csv
        const dynamicCsvPatterns: Record<string, string> = {
          'mccar': 'f1data/McCar',
          'mcdriver': 'f1data/McDriver',
          'mcracecontext': 'f1data/McRaceContext',
        };

        const dynamicMatch = filePath.match(/^(mccar|mcdriver|mcracecontext)\/(.+\.csv)$/);
        if (dynamicMatch) {
          const baseDir = dynamicCsvPatterns[dynamicMatch[1]];
          const fullPath = path.join(f1Root, baseDir, dynamicMatch[2]);
          if (fs.existsSync(fullPath)) {
            res.setHeader('Content-Type', 'text/csv');
            const content = fs.readFileSync(fullPath, 'utf-8');
            const lines = content.split('\n');
            // Sample large CSVs to keep browser responsive (~50K rows max)
            const MAX_ROWS = 50000;
            if (lines.length > MAX_ROWS + 1) {
              const header = lines[0];
              const dataLines = lines.slice(1).filter(l => l.trim());
              const step = Math.ceil(dataLines.length / MAX_ROWS);
              const sampled = [header];
              for (let i = 0; i < dataLines.length; i += step) {
                sampled.push(dataLines[i]);
              }
              res.end(sampled.join('\n'));
            } else {
              res.end(content);
            }
            return;
          }
        }

        // Serve CSV files from f1data (legacy fallback)
        if (filePath.startsWith('f1data/') && filePath.endsWith('.csv')) {
          const fullPath = path.join(f1Root, filePath);
          if (fs.existsSync(fullPath)) {
            res.setHeader('Content-Type', 'text/csv');
            res.end(fs.readFileSync(fullPath, 'utf-8'));
            return;
          }
        }

        // Car telemetry summary endpoint: aggregates per-race stats server-side
        // GET /api/local/mccar-summary/{year}/{driver}
        const carSummaryMatch = filePath.match(/^mccar-summary\/(\d{4})\/(NOR|PIA)$/);
        if (carSummaryMatch) {
          const cy = carSummaryMatch[1];
          const cd = carSummaryMatch[2];
          const carDir = path.join(f1Root, 'f1data/McCar', cy);
          if (fs.existsSync(carDir)) {
            const files = fs.readdirSync(carDir).filter((f: string) => f.endsWith('.csv') && !f.startsWith('ALL'));
            const summary: any[] = [];
            for (const file of files) {
              const raceMatch = file.match(/^\d{4}_(.+)_Grand_Prix_Race\.csv$/);
              if (!raceMatch) continue;
              const raceName = raceMatch[1].replace(/_/g, ' ');
              const content = fs.readFileSync(path.join(carDir, file), 'utf-8');
              const lines = content.split('\n');
              if (lines.length < 2) continue;
              const headers = lines[0].split(',');
              const driverIdx = headers.indexOf('Driver');
              const speedIdx = headers.indexOf('Speed');
              const rpmIdx = headers.indexOf('RPM');
              const throttleIdx = headers.indexOf('Throttle');
              const brakeIdx = headers.indexOf('Brake');
              const drsIdx = headers.indexOf('DRS');
              const gearIdx = headers.indexOf('nGear');
              const compoundIdx = headers.indexOf('Compound');
              if (speedIdx === -1) continue;
              let speedSum = 0, speedMax = 0, rpmSum = 0, rpmMax = 0;
              let throttleSum = 0, brakeCount = 0, drsCount = 0, count = 0;
              let maxGear = 0;
              const compounds = new Set<string>();
              for (let i = 1; i < lines.length; i++) {
                const vals = lines[i].split(',');
                if (driverIdx >= 0 && vals[driverIdx] !== cd) continue;
                const spd = parseFloat(vals[speedIdx]);
                if (isNaN(spd)) continue;
                speedSum += spd;
                if (spd > speedMax) speedMax = spd;
                if (rpmIdx >= 0) { const r = parseFloat(vals[rpmIdx]) || 0; rpmSum += r; if (r > rpmMax) rpmMax = r; }
                if (throttleIdx >= 0) throttleSum += parseFloat(vals[throttleIdx]) || 0;
                if (brakeIdx >= 0 && (vals[brakeIdx] === 'True' || vals[brakeIdx] === '1')) brakeCount++;
                if (drsIdx >= 0 && parseFloat(vals[drsIdx]) >= 10) drsCount++;
                if (gearIdx >= 0) { const g = parseFloat(vals[gearIdx]) || 0; if (g > maxGear) maxGear = g; }
                if (compoundIdx >= 0 && vals[compoundIdx]) compounds.add(vals[compoundIdx]);
                count++;
              }
              if (count > 0) {
                summary.push({
                  race: raceName,
                  avgSpeed: Math.round(speedSum / count * 10) / 10,
                  topSpeed: Math.round(speedMax * 10) / 10,
                  avgRPM: Math.round(rpmSum / count),
                  maxRPM: Math.round(rpmMax),
                  avgThrottle: Math.round(throttleSum / count * 10) / 10,
                  brakePct: Math.round(brakeCount / count * 1000) / 10,
                  drsPct: Math.round(drsCount / count * 1000) / 10,
                  compounds: Array.from(compounds).filter(c => c.length > 0),
                  samples: count,
                });
              }
            }
            res.setHeader('Content-Type', 'application/json');
            res.end(JSON.stringify(summary));
            return;
          }
        }

        // Biometric summary endpoint: aggregates per-race stats server-side
        // GET /api/local/mcdriver-summary/{year}/{driver}
        const summaryMatch = filePath.match(/^mcdriver-summary\/(\d{4})\/(NOR|PIA)$/);
        if (summaryMatch) {
          const sy = summaryMatch[1];
          const sd = summaryMatch[2];
          const driverDir = path.join(f1Root, 'f1data/McDriver', sy);
          if (fs.existsSync(driverDir)) {
            const files = fs.readdirSync(driverDir).filter((f: string) => f.endsWith('_biometrics.csv'));
            const summary: any[] = [];
            for (const file of files) {
              const raceMatch = file.match(/^\d{4}_(.+)_Grand_Prix_Race_biometrics\.csv$/);
              if (!raceMatch) continue;
              const raceName = raceMatch[1].replace(/_/g, ' ');
              const content = fs.readFileSync(path.join(driverDir, file), 'utf-8');
              const lines = content.split('\n');
              if (lines.length < 2) continue;
              const headers = lines[0].split(',');
              const driverIdx = headers.indexOf('Driver');
              const hrIdx = headers.indexOf('HeartRate_bpm');
              const tempIdx = headers.indexOf('CockpitTemp_C');
              const battleIdx = headers.indexOf('BattleIntensity');
              const airIdx = headers.indexOf('AirTemp_C');
              const trackIdx = headers.indexOf('TrackTemp_C');
              if (hrIdx === -1) continue;
              let hrSum = 0, hrMax = 0, tempSum = 0, battleSum = 0, airTemp = 0, trackTemp = 0, count = 0;
              for (let i = 1; i < lines.length; i++) {
                const vals = lines[i].split(',');
                if (driverIdx >= 0 && vals[driverIdx] !== sd) continue;
                const hr = parseFloat(vals[hrIdx]);
                if (isNaN(hr)) continue;
                hrSum += hr;
                if (hr > hrMax) hrMax = hr;
                if (tempIdx >= 0) tempSum += parseFloat(vals[tempIdx]) || 0;
                if (battleIdx >= 0) battleSum += parseFloat(vals[battleIdx]) || 0;
                if (count === 0) {
                  if (airIdx >= 0) airTemp = parseFloat(vals[airIdx]) || 0;
                  if (trackIdx >= 0) trackTemp = parseFloat(vals[trackIdx]) || 0;
                }
                count++;
              }
              if (count > 0) {
                summary.push({
                  race: raceName,
                  avgHR: Math.round(hrSum / count * 10) / 10,
                  peakHR: Math.round(hrMax * 10) / 10,
                  avgTemp: Math.round(tempSum / count * 10) / 10,
                  battleIntensity: Math.round(battleSum / count * 1000) / 10,
                  airTemp: Math.round(airTemp * 10) / 10,
                  trackTemp: Math.round(trackTemp * 10) / 10,
                  samples: count,
                });
              }
            }
            res.setHeader('Content-Type', 'application/json');
            res.end(JSON.stringify(summary));
            return;
          }
        }

        res.statusCode = 404;
        res.end(JSON.stringify({ error: 'Not found' }));
      });

      // Serve annotated images from McMedia results
      server.middlewares.use('/media', (req: any, res: any, next: any) => {
        const filePath = path.join(f1Root, 'f1data/McMedia', req.url.replace(/^\//, ''));
        if (fs.existsSync(filePath)) {
          const ext = path.extname(filePath).toLowerCase();
          const mimeTypes: Record<string, string> = {
            '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png',
          };
          res.setHeader('Content-Type', mimeTypes[ext] || 'application/octet-stream');
          res.end(fs.readFileSync(filePath));
          return;
        }
        res.statusCode = 404;
        res.end('Not found');
      });
    },
  };
}

// ── GenUI Diagnostic Plugin — Vercel AI SDK + Groq ──────────────────
function genUIPlugin() {
  const f1Root = path.resolve(__dirname, '..');
  // Load GROQ_API_KEY from .env
  const envPath = path.join(f1Root, '.env');
  let groqApiKey = process.env.GROQ_API_KEY || '';
  if (!groqApiKey && fs.existsSync(envPath)) {
    const envContent = fs.readFileSync(envPath, 'utf-8');
    const match = envContent.match(/^GROQ_API_KEY=(.+)$/m);
    if (match) groqApiKey = match[1].trim();
  }

  const DIAGNOSE_SYSTEM_PROMPT = `You are the F1 OmniSense Diagnostic AI for McLaren fleet management.
You analyze telemetry and anomaly detection data for specific vehicle systems across a racing season.

You MUST use the provided tools to render structured diagnostic UI components. Do NOT just output plain text — call the tools to display your analysis visually.

Guidelines:
- Start with 2-3 metric_card calls for the most important stats (current health, trend, worst race)
- Then call sparkline to show the health trend across the season
- Then call text with a concise 2-3 sentence analysis
- End with 1-2 recommendation calls for actionable maintenance items
- Use comparison if cross-system or cross-race comparison would be insightful
- Be specific with numbers and race names
- Think like an F1 race engineer advising the team`;

  return {
    name: 'genui-diagnose',
    configureServer(server: any) {
      server.middlewares.use('/api/fleet/diagnose', async (req: any, res: any, next: any) => {
        if (req.method !== 'POST') { next(); return; }

        // Parse request body
        let body = '';
        for await (const chunk of req) body += chunk;
        let parsed: any;
        try { parsed = JSON.parse(body); } catch { res.statusCode = 400; res.end('Bad JSON'); return; }

        if (!groqApiKey) { res.statusCode = 500; res.end(JSON.stringify({ error: 'GROQ_API_KEY not set' })); return; }

        try {
          const { createGroq } = await import('@ai-sdk/groq');
          const { streamText, tool } = await import('ai');
          const { z } = await import('zod');

          const groq = createGroq({ apiKey: groqApiKey });

          const result = streamText({
            model: groq('llama-3.3-70b-versatile'),
            system: DIAGNOSE_SYSTEM_PROMPT,
            messages: parsed.messages ?? [],
            tools: {
              metric_card: tool({
                description: 'Display a key diagnostic metric with trend indicator. Use for health %, peak values, or comparisons.',
                parameters: z.object({
                  title: z.string().describe('Metric name, e.g. "Current Health" or "Peak RPM"'),
                  value: z.string().describe('Display value with unit, e.g. "65%" or "12,400 RPM"'),
                  trend: z.enum(['up', 'down', 'stable']).describe('Trend direction relative to previous races'),
                  severity: z.enum(['nominal', 'warning', 'critical']).describe('Health severity level'),
                  subtitle: z.string().optional().describe('Brief context, e.g. "Down 13% since Abu Dhabi"'),
                }),
              }),
              sparkline: tool({
                description: 'Show a mini line chart of a metric across races. Use for health trends, RPM trends, temperature trends.',
                parameters: z.object({
                  title: z.string().describe('Chart title'),
                  data: z.array(z.object({ race: z.string(), value: z.number() })).describe('Data points per race'),
                  unit: z.string().describe('Value unit, e.g. "%" or "RPM"'),
                  thresholds: z.object({
                    warning: z.number(),
                    critical: z.number(),
                  }).optional().describe('Threshold lines to draw on the chart'),
                }),
              }),
              comparison: tool({
                description: 'Show side-by-side bar comparison of values. Use for comparing metrics across races or systems.',
                parameters: z.object({
                  title: z.string(),
                  items: z.array(z.object({
                    label: z.string(),
                    value: z.number(),
                    max: z.number(),
                  })),
                }),
              }),
              recommendation: tool({
                description: 'Display an actionable maintenance recommendation with severity.',
                parameters: z.object({
                  title: z.string().describe('Short recommendation title'),
                  description: z.string().describe('Detailed explanation'),
                  severity: z.enum(['info', 'warning', 'critical']),
                  action: z.string().describe('Specific action to take'),
                }),
              }),
              text: tool({
                description: 'Display a paragraph of analysis text. Use for connecting the data points and providing engineering insight.',
                parameters: z.object({
                  content: z.string().describe('Analysis text (1-3 sentences)'),
                }),
              }),
            },
            temperature: 0.4,
            maxOutputTokens: 2048,
          });

          result.pipeTextStreamToResponse(res);
        } catch (err: any) {
          console.error('GenUI diagnose error:', err);
          res.statusCode = 500;
          res.end(JSON.stringify({ error: err.message }));
        }
      });
    },
  };
}

/** Fleet vehicles CRUD — dev middleware backed by MongoDB */
function fleetVehiclesPlugin() {
  return {
    name: 'fleet-vehicles-api',
    configureServer(server: any) {
      server.middlewares.use('/api/fleet-vehicles', async (req: any, res: any, next: any) => {
        // Dynamically import mongodb (already a project dependency)
        const { MongoClient } = await import('mongodb');

        // Load env
        const f1Root = path.resolve(__dirname, '..');
        const envPath = path.join(f1Root, '.env');
        let mongoUri = process.env.MONGODB_URI || '';
        let mongoDb = process.env.MONGODB_DB || 'marip_f1';
        if (!mongoUri && fs.existsSync(envPath)) {
          const envContent = fs.readFileSync(envPath, 'utf-8');
          const uriMatch = envContent.match(/^MONGODB_URI=(.+)$/m);
          if (uriMatch) mongoUri = uriMatch[1].trim();
          const dbMatch = envContent.match(/^MONGODB_DB=(.+)$/m);
          if (dbMatch) mongoDb = dbMatch[1].trim();
        }

        if (!mongoUri) {
          res.statusCode = 500;
          res.end(JSON.stringify({ error: 'MONGODB_URI not configured' }));
          return;
        }

        const client = new MongoClient(mongoUri);
        try {
          await client.connect();
          const col = client.db(mongoDb).collection('fleet_vehicles');

          if (req.method === 'GET') {
            const vehicles = await col.find({}, { projection: { _id: 0 } })
              .sort({ createdAt: -1 })
              .toArray();
            res.setHeader('Content-Type', 'application/json');
            res.end(JSON.stringify(vehicles));
            return;
          }

          if (req.method === 'POST') {
            let body = '';
            for await (const chunk of req) body += chunk;
            let parsed: any;
            try { parsed = JSON.parse(body); } catch { res.statusCode = 400; res.end('Bad JSON'); return; }

            const { model, driverName, driverNumber, driverCode, teamName, chassisId, engineSpec, season, notes } = parsed;
            if (!model || !driverName || !driverNumber || !driverCode) {
              res.statusCode = 400;
              res.end(JSON.stringify({ error: 'Missing required fields: model, driverName, driverNumber, driverCode' }));
              return;
            }

            const doc = {
              model: String(model),
              driverName: String(driverName),
              driverNumber: Number(driverNumber),
              driverCode: String(driverCode).toUpperCase().slice(0, 3),
              teamName: String(teamName || 'McLaren'),
              chassisId: String(chassisId || ''),
              engineSpec: String(engineSpec || ''),
              season: Number(season) || new Date().getFullYear(),
              notes: String(notes || ''),
              createdAt: new Date(),
            };

            await col.insertOne(doc);
            res.statusCode = 201;
            res.setHeader('Content-Type', 'application/json');
            const { _id, ...safe } = doc as any;
            res.end(JSON.stringify(safe));
            return;
          }

          res.statusCode = 405;
          res.end(JSON.stringify({ error: 'Method not allowed' }));
        } catch (err: any) {
          res.statusCode = 500;
          res.end(JSON.stringify({ error: err.message }));
        } finally {
          await client.close();
        }
      });
    },
  };
}

/** Cache large static assets (GLB models) — tells browser to keep them for 1 day */
function cacheGLBPlugin() {
  return {
    name: 'cache-glb',
    configureServer(server: any) {
      server.middlewares.use((req: any, res: any, next: any) => {
        if (req.url && /\.glb(\?|$)/.test(req.url)) {
          res.setHeader('Cache-Control', 'public, max-age=86400, immutable');
        }
        next();
      });
    },
  };
}

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
    localDataPlugin(),
    genUIPlugin(),
    fleetVehiclesPlugin(),
    cacheGLBPlugin(),
  ],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  assetsInclude: ['**/*.svg'],
  build: {
    target: 'es2022',
  },
  optimizeDeps: {
    exclude: ['three'],
    include: ['maplibre-gl', 'react-map-gl/maplibre'],
    esbuildOptions: {
      target: 'es2022',
    },
  },
  server: {
    proxy: {
      '/api/chat': {
        target: 'http://localhost:8100',
        changeOrigin: true,
        rewrite: (path: string) => path.replace(/^\/api\/chat/, '/chat'),
      },
      '/api/health': {
        target: 'http://localhost:8100',
        changeOrigin: true,
        rewrite: (path: string) => path.replace(/^\/api\/health/, '/health'),
      },
      '/api/visual-search': {
        target: 'http://localhost:8100',
        changeOrigin: true,
        rewrite: (path: string) => path.replace(/^\/api\/visual-search/, '/visual-search'),
      },
      '/api/visual-tags': {
        target: 'http://localhost:8100',
        changeOrigin: true,
        rewrite: (path: string) => path.replace(/^\/api\/visual-tags/, '/visual-tags'),
      },
      '/api/upload': {
        target: 'http://localhost:8100',
        changeOrigin: true,
        rewrite: (path: string) => path.replace(/^\/api\/upload/, '/upload'),
      },
      '/api/3d-gen': {
        target: 'http://localhost:8100',
        changeOrigin: true,
      },
      '/api/omni': {
        target: 'http://localhost:8100',
        changeOrigin: true,
      },
      '/api/jolpica': { target: 'http://localhost:8100', changeOrigin: true, rewrite: (path: string) => path.replace(/^\/api\//, '/api/local/') },
      '/api/openf1': { target: 'http://localhost:8100', changeOrigin: true, rewrite: (path: string) => path.replace(/^\/api\//, '/api/local/') },
      '/api/opponents': { target: 'http://localhost:8100', changeOrigin: true },
      '/api/driver_intel': { target: 'http://localhost:8100', changeOrigin: true, rewrite: (path: string) => path.replace(/^\/api\//, '/api/local/') },
      '/api/circuit_intel': { target: 'http://localhost:8100', changeOrigin: true, rewrite: (path: string) => path.replace(/^\/api\//, '/api/local/') },
      '/api/pipeline': { target: 'http://localhost:8100', changeOrigin: true, rewrite: (path: string) => path.replace(/^\/api\//, '/api/local/') },
      '/api/f1data': { target: 'http://localhost:8100', changeOrigin: true, rewrite: (path: string) => path.replace(/^\/api\//, '/api/local/') },
      '/api/mccar': { target: 'http://localhost:8100', changeOrigin: true, rewrite: (path: string) => path.replace(/^\/api\//, '/api/local/') },
      '/api/mcdriver': { target: 'http://localhost:8100', changeOrigin: true, rewrite: (path: string) => path.replace(/^\/api\//, '/api/local/') },
      '/api/mcracecontext': { target: 'http://localhost:8100', changeOrigin: true, rewrite: (path: string) => path.replace(/^\/api\//, '/api/local/') },
      '/3d-models': {
        target: 'http://localhost:8100',
        changeOrigin: true,
      },
    },
  },
})
