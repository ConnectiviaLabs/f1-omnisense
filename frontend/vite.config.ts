import { defineConfig } from 'vite'
import path from 'path'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import fs from 'fs'
import { execFile } from 'child_process'

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

        // These endpoints are served by the Python backend — let proxy handle them
        if (filePath.startsWith('strategy/')) { next(); return; }
        if (filePath.startsWith('mccar-summary/')) { next(); return; }
        if (filePath.startsWith('mccar-telemetry/')) { next(); return; }
        if (filePath.startsWith('mccar-race-telemetry/')) { next(); return; }
        if (filePath.startsWith('mccar-race-stints/')) { next(); return; }
        if (filePath.startsWith('mcdriver-summary/')) { next(); return; }
        if (filePath.startsWith('driver_intel/')) { next(); return; }
        if (filePath.startsWith('team_intel/')) { next(); return; }

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

      // Run GDino detection pipeline on a single video (spawns Python process)
      const activeGdinoJobs = new Map<string, { status: string; result?: any; error?: string }>();

      server.middlewares.use('/api/run-gdino', async (req: any, res: any) => {
        if (req.method !== 'POST') { res.statusCode = 405; res.end('Method not allowed'); return; }
        let body = '';
        req.on('data', (chunk: any) => { body += chunk; });
        req.on('end', () => {
          try {
            const { filename } = JSON.parse(body);
            if (!filename) { res.statusCode = 400; res.end(JSON.stringify({ error: 'filename required' })); return; }

            // Check if already running
            const existing = activeGdinoJobs.get(filename);
            if (existing?.status === 'running') {
              res.writeHead(200, { 'Content-Type': 'application/json' });
              res.end(JSON.stringify({ status: 'running', message: 'Already processing' }));
              return;
            }

            const scriptPath = path.join(f1Root, 'f1data/McMedia/test_gdino_local.py');
            activeGdinoJobs.set(filename, { status: 'running' });

            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ status: 'started', filename }));

            // Spawn async — result available via /api/gdino-status
            execFile('python3', [scriptPath, filename], {
              cwd: path.join(f1Root, 'f1data/McMedia'),
              timeout: 600000, // 10 min max
              env: { ...process.env, PYTORCH_CUDA_ALLOC_CONF: 'expandable_segments:True' },
            }, (err: any, stdout: string, stderr: string) => {
              if (err) {
                activeGdinoJobs.set(filename, { status: 'error', error: stderr || err.message });
              } else {
                activeGdinoJobs.set(filename, { status: 'done', result: stdout });
              }
            });
          } catch (e: any) {
            res.statusCode = 500;
            res.end(JSON.stringify({ error: e.message }));
          }
        });
      });

      server.middlewares.use('/api/gdino-status', (req: any, res: any) => {
        const url = new URL(req.url, 'http://localhost');
        const filename = url.searchParams.get('filename');
        res.writeHead(200, { 'Content-Type': 'application/json' });
        if (!filename) {
          // Return all statuses
          const all: Record<string, any> = {};
          activeGdinoJobs.forEach((v, k) => { all[k] = v; });
          res.end(JSON.stringify(all));
        } else {
          const job = activeGdinoJobs.get(filename);
          res.end(JSON.stringify(job || { status: 'idle' }));
        }
      });

      // Upload video — raw binary with X-Filename header
      server.middlewares.use('/api/upload-video', (req: any, res: any) => {
        if (req.method !== 'POST') { res.statusCode = 405; res.end('Method not allowed'); return; }
        const filename = req.headers['x-filename'];
        if (!filename) { res.statusCode = 400; res.end(JSON.stringify({ error: 'X-Filename header required' })); return; }
        const ext = path.extname(String(filename)).toLowerCase();
        if (!['.mp4', '.webm', '.mov'].includes(ext)) {
          res.statusCode = 400; res.end(JSON.stringify({ error: 'Only mp4/webm/mov allowed' })); return;
        }
        const safeName = String(filename).replace(/[/\\]/g, '_');
        const dest = path.join(f1Root, 'f1data/McMedia', safeName);
        const ws = fs.createWriteStream(dest);
        req.pipe(ws);
        ws.on('finish', () => {
          const stat = fs.statSync(dest);
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({
            filename: safeName,
            size_mb: Math.round(stat.size / 1024 / 1024 * 100) / 100,
            added: new Date().toISOString(),
            analyzed: false,
          }));
        });
        ws.on('error', (err: any) => { res.statusCode = 500; res.end(JSON.stringify({ error: err.message })); });
      });

      // VLM analysis — sends frames + model context to Ollama qwen3.5:2b
      server.middlewares.use('/api/vlm-analyze', async (req: any, res: any) => {
        if (req.method !== 'POST') { res.statusCode = 405; res.end('Method not allowed'); return; }
        let body = '';
        req.on('data', (chunk: any) => { body += chunk; });
        req.on('end', async () => {
          try {
            const { filename } = JSON.parse(body);
            const mcMediaDir = path.join(f1Root, 'f1data/McMedia');

            // Load model results
            const loadJson = (p: string) => { try { return JSON.parse(fs.readFileSync(p, 'utf-8')); } catch { return {}; } };
            const gdino = loadJson(path.join(mcMediaDir, 'gdino_results/gdino_results.json'));
            const fused = loadJson(path.join(mcMediaDir, 'fused_results/fused_results.json'));
            const videomae = loadJson(path.join(mcMediaDir, 'videomae_results/videomae_results.json'));
            const timesformer = loadJson(path.join(mcMediaDir, 'timesformer_results/timesformer_results.json'));

            // Get detection frames for this video
            const gdinoFrames = gdino[filename] || [];
            const fusedFrames = Array.isArray(fused[filename]) ? fused[filename] : [];

            // Pick up to 8 evenly spaced frame images
            const frameImages: string[] = [];
            const frameDir = path.join(mcMediaDir, 'gdino_results');
            const availableFrames = gdinoFrames.filter((f: any) => f.output_image && fs.existsSync(path.join(frameDir, f.output_image)));
            const step = Math.max(1, Math.floor(availableFrames.length / 8));
            const selected = availableFrames.filter((_: any, i: number) => i % step === 0).slice(0, 8);

            for (const frame of selected) {
              const imgPath = path.join(frameDir, frame.output_image);
              const imgBuf = fs.readFileSync(imgPath);
              frameImages.push(imgBuf.toString('base64'));
            }

            // Summarize detections across all frames
            const allDets: Record<string, number[]> = {};
            for (const src of [gdinoFrames, fusedFrames]) {
              for (const frame of src) {
                for (const det of (frame.detections || [])) {
                  if (!allDets[det.category]) allDets[det.category] = [];
                  allDets[det.category].push(det.score || 0);
                }
              }
            }
            const detSummary = Object.entries(allDets)
              .sort((a, b) => Math.max(...b[1]) - Math.max(...a[1]))
              .slice(0, 15)
              .map(([cat, scores]) => `  - ${cat}: seen ${scores.length}x, best ${Math.round(Math.max(...scores) * 100)}%, avg ${Math.round(scores.reduce((a, b) => a + b, 0) / scores.length * 100)}%`)
              .join('\n');

            // Summarize classifications
            const classLines: string[] = [];
            for (const [name, data] of [['VideoMAE', videomae[filename]], ['TimeSformer', timesformer[filename]]] as const) {
              const preds = (data as any)?.top_predictions?.slice(0, 3) || [];
              if (preds.length) {
                classLines.push(`  ${name}: ${preds.map((p: any) => `${p.label} (${Math.round(p.score * 100)}%)`).join(', ')}`);
              }
            }

            const prompt = `You are an F1 race strategy engineer at McLaren analyzing video: ${filename}\n\n` +
              `I'm showing you ${frameImages.length} evenly-spaced frames from this video.\n\n` +
              `OBJECT DETECTION RESULTS (GroundingDINO + SAM2):\n${detSummary || 'No objects detected.'}\n\n` +
              `ACTION CLASSIFICATION:\n${classLines.join('\n') || 'No classifications available.'}\n\n` +
              `Based on these ${frameImages.length} frames and the model results, respond with ONLY a JSON object (no markdown, no code fences):\n\n` +
              `{"scene":"1-2 sentence description","key_objects":["object1","object2"],"track_conditions":"dry/wet + observations","strategic_notes":"strategic implications","incidents":["any incidents observed"]}`;

            // Call Ollama
            const ollamaRes = await fetch('http://localhost:11434/api/chat', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                model: 'qwen3.5:2b',
                messages: [{ role: 'user', content: prompt, images: frameImages }],
                stream: false,
                options: { temperature: 0.3, num_predict: 512 },
              }),
            });

            const ollamaData = await ollamaRes.json();
            const rawAnalysis = ollamaData.message?.content || '';
            const tokens = ollamaData.eval_count || 0;
            const totalDur = (ollamaData.total_duration || 0) / 1e9;

            // Attempt to parse structured JSON from VLM response
            let structured = null;
            try {
              structured = JSON.parse(rawAnalysis);
            } catch {
              const jsonMatch = rawAnalysis.match(/\{[\s\S]*\}/);
              if (jsonMatch) { try { structured = JSON.parse(jsonMatch[0]); } catch {} }
            }

            res.setHeader('Content-Type', 'application/json');
            res.end(JSON.stringify({
              analysis: rawAnalysis,
              structured,
              tokens,
              time_s: Math.round(totalDur * 100) / 100,
              tok_per_s: totalDur > 0 ? Math.round(tokens / totalDur * 10) / 10 : 0,
              frames_analyzed: frameImages.length,
              model: 'qwen3.5:2b',
            }));
          } catch (e: any) {
            res.statusCode = 500;
            res.setHeader('Content-Type', 'application/json');
            res.end(JSON.stringify({ error: e.message }));
          }
        });
      });

      // Serve media files (images + videos) from McMedia results
      server.middlewares.use('/media', (req: any, res: any, next: any) => {
        const filePath = path.join(f1Root, 'f1data/McMedia', decodeURIComponent(req.url.replace(/^\//, '')));
        if (!fs.existsSync(filePath)) { res.statusCode = 404; res.end('Not found'); return; }
        const ext = path.extname(filePath).toLowerCase();
        const mimeTypes: Record<string, string> = {
          '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png',
          '.mp4': 'video/mp4', '.webm': 'video/webm', '.mov': 'video/quicktime',
        };
        const mime = mimeTypes[ext] || 'application/octet-stream';
        const stat = fs.statSync(filePath);

        // Stream videos with Range support for seeking
        if (ext === '.mp4' || ext === '.webm' || ext === '.mov') {
          const range = req.headers.range;
          if (range) {
            const parts = range.replace(/bytes=/, '').split('-');
            const start = parseInt(parts[0], 10);
            const end = parts[1] ? parseInt(parts[1], 10) : stat.size - 1;
            res.writeHead(206, {
              'Content-Range': `bytes ${start}-${end}/${stat.size}`,
              'Accept-Ranges': 'bytes',
              'Content-Length': end - start + 1,
              'Content-Type': mime,
            });
            fs.createReadStream(filePath, { start, end }).pipe(res);
          } else {
            res.writeHead(200, {
              'Content-Length': stat.size,
              'Content-Type': mime,
              'Accept-Ranges': 'bytes',
            });
            fs.createReadStream(filePath).pipe(res);
          }
          return;
        }

        // Images: read into memory (small files)
        res.setHeader('Content-Type', mime);
        res.end(fs.readFileSync(filePath));
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

  const DIAGNOSE_SYSTEM_BASE = `You are the F1 OmniSense Diagnostic AI for McLaren fleet management.
You analyze REAL telemetry and anomaly detection data for specific vehicle systems across a racing season.

CRITICAL: You have access to REAL data below. Use ONLY these numbers — NEVER invent or hallucinate statistics.
If the user asks about a driver not in the data, say you don't have data for that driver.

You MUST use the provided tools to render structured diagnostic UI components. Do NOT just output plain text — call the tools to display your analysis visually.

Guidelines:
- Start with 2-3 metric_card calls for the most important stats (current health, trend, worst race)
- Then call sparkline to show the health trend across the season
- Then call text with a concise 2-3 sentence analysis
- End with 1-2 recommendation calls for actionable maintenance items
- Use comparison if cross-system or cross-race comparison would be insightful
- Be specific with numbers and race names from the REAL DATA below
- Think like an F1 race engineer advising the team`;

  const API_BASE = 'http://127.0.0.1:8300';

  // Cache grounding data (refreshed every 5 minutes)
  let cachedSnapshot: any = null;
  let cachedPerf: any[] = [];
  let cachedProfiles: Record<string, any> = {};
  let cacheTime = 0;
  const CACHE_TTL = 5 * 60 * 1000;

  async function refreshCache() {
    if (Date.now() - cacheTime < CACHE_TTL && cachedSnapshot) return;
    try {
      const [snapRes, perfRes] = await Promise.all([
        fetch(`${API_BASE}/api/local/pipeline/anomaly`),
        fetch(`${API_BASE}/api/local/driver_intel/performance_markers`),
      ]);
      if (snapRes.ok) cachedSnapshot = await snapRes.json();
      if (perfRes.ok) cachedPerf = await perfRes.json();
      cacheTime = Date.now();
    } catch { /* keep stale cache */ }
  }

  /** Fetch VectorProfile narrative for a driver (lazy, per-driver cache) */
  async function getVectorNarrative(code: string): Promise<string | null> {
    if (cachedProfiles[code]) return cachedProfiles[code];
    try {
      const res = await fetch(`${API_BASE}/api/local/driver_intel/vector_profile/${code}`);
      if (!res.ok) return null;
      const doc = await res.json();
      cachedProfiles[code] = doc.narrative ?? null;
      return cachedProfiles[code];
    } catch { return null; }
  }

  /** Fetch similar drivers for a code */
  async function getSimilarDrivers(code: string, k = 5): Promise<any[]> {
    try {
      const res = await fetch(`${API_BASE}/api/local/driver_intel/similar/${code}?k=${k}`);
      if (!res.ok) return [];
      return await res.json();
    } catch { return []; }
  }

  /** Extract driver codes mentioned in the conversation */
  function extractDriverCodes(messages: any[]): string[] {
    const DRIVER_ALIASES: Record<string, string> = {
      norris: 'NOR', lando: 'NOR', piastri: 'PIA', oscar: 'PIA',
      verstappen: 'VER', max: 'VER', hamilton: 'HAM', lewis: 'HAM',
      leclerc: 'LEC', charles: 'LEC', sainz: 'SAI', carlos: 'SAI',
      russell: 'RUS', george: 'RUS', perez: 'PER', checo: 'PER',
      alonso: 'ALO', fernando: 'ALO', stroll: 'STR', lance: 'STR',
      gasly: 'GAS', pierre: 'GAS', ocon: 'OCO', esteban: 'OCO',
      tsunoda: 'TSU', yuki: 'TSU', ricciardo: 'RIC', daniel: 'RIC',
      hulkenberg: 'HUL', nico: 'HUL', magnussen: 'MAG', kevin: 'MAG',
      albon: 'ALB', alexander: 'ALB', bottas: 'BOT', valtteri: 'BOT',
      zhou: 'ZHO', guanyu: 'ZHO', sargeant: 'SAR', lawson: 'LAW',
      bearman: 'BEA', colapinto: 'COL', devries: 'DEV',
      mclaren: '__TEAM_MCL__', ferrari: '__TEAM_FER__',
      'red bull': '__TEAM_RBR__', mercedes: '__TEAM_MER__',
    };
    const text = messages.map((m: any) => {
      if (m.parts) return m.parts.filter((p: any) => p.type === 'text').map((p: any) => p.text).join(' ');
      return m.content ?? '';
    }).join(' ').toLowerCase();

    const codes = new Set<string>();
    // Check 3-letter codes directly
    const codeMatches = text.match(/\b[A-Z]{3}\b/gi) ?? [];
    for (const c of codeMatches) {
      if (cachedSnapshot?.drivers?.some((d: any) => d.code === c.toUpperCase())) {
        codes.add(c.toUpperCase());
      }
    }
    // Check aliases
    for (const [alias, code] of Object.entries(DRIVER_ALIASES)) {
      if (text.includes(alias)) {
        if (code.startsWith('__TEAM_')) {
          // Add all drivers from this team
          const teamMap: Record<string, string> = {
            '__TEAM_MCL__': 'McLaren', '__TEAM_FER__': 'Ferrari',
            '__TEAM_RBR__': 'Red Bull Racing', '__TEAM_MER__': 'Mercedes',
          };
          const team = teamMap[code];
          for (const d of cachedSnapshot?.drivers ?? []) {
            if (d.team?.includes(team)) codes.add(d.code);
          }
        } else {
          codes.add(code);
        }
      }
    }
    return [...codes];
  }

  /** Build grounding data for the system prompt */
  function buildGroundingData(mentionedCodes: string[]): string {
    const sections: string[] = [];
    const drivers = cachedSnapshot?.drivers ?? [];

    // Fleet summary — all drivers, one line each
    sections.push('## Fleet Health Summary (all drivers)');
    const sortedDrivers = [...drivers].sort((a: any, b: any) =>
      (a.overallHealth ?? a.overall_health ?? 0) - (b.overallHealth ?? b.overall_health ?? 0)
    );
    for (const d of sortedDrivers) {
      const health = d.overallHealth ?? d.overall_health ?? 'N/A';
      const level = d.level ?? d.overall_level ?? '?';
      sections.push(`- ${d.code} (${d.team ?? '?'}): ${health}/100 (${level})`);
    }

    // Detailed data for mentioned drivers only
    if (mentionedCodes.length > 0) {
      sections.push('\n## Detailed Driver Data');
      for (const code of mentionedCodes) {
        const d = drivers.find((x: any) => x.code === code);
        if (!d) continue;

        // Systems
        const systems = d.systems ?? [];
        const sysLines = systems.map((s: any) =>
          `  - ${s.name}: ${s.health}/100 (${s.level})${s.maintenanceAction && s.maintenanceAction !== 'none' ? ` [ACTION: ${s.maintenanceAction}]` : ''}`
        ).join('\n');

        // Last 10 races with system breakdowns
        const recentRaces = (d.races ?? []).slice(-10);
        const raceLines = recentRaces.map((r: any) => {
          const parts = Object.entries(r.systems ?? {}).map(([name, info]: [string, any]) =>
            `${name}: ${info.health}%`
          ).join(', ');
          return `  - ${r.race}: ${parts}`;
        }).join('\n');

        sections.push(`### ${d.code} (${d.team ?? '?'}) — Health: ${d.overallHealth ?? d.overall_health}/100
Last race: ${d.lastRace ?? d.last_race ?? 'N/A'}
${sysLines ? `Systems:\n${sysLines}` : ''}
Race-by-race (last ${recentRaces.length}):\n${raceLines}`);

        // Performance markers for this driver
        const perf = cachedPerf.find((m: any) => m.Driver === code);
        if (perf) {
          sections.push(`Performance metrics:
- Tyre degradation: ${perf.degradation_slope_s_per_lap?.toFixed(3) ?? 'N/A'} s/lap
- Late-race delta: ${perf.late_race_delta_s?.toFixed(2) ?? 'N/A'} s
- Lap consistency (std): ${perf.lap_time_consistency_std?.toFixed(3) ?? 'N/A'}
- Avg top speed: ${perf.avg_top_speed_kmh?.toFixed(1) ?? 'N/A'} km/h
- Throttle smoothness: ${perf.throttle_smoothness?.toFixed(1) ?? 'N/A'}%
- Late-race speed drop: ${perf.late_race_speed_drop_kmh?.toFixed(1) ?? 'N/A'} km/h`);
        }
      }
    }

    return sections.join('\n');
  }

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
          const { streamText, tool, convertToModelMessages } = await import('ai');
          const { z } = await import('zod');

          const groq = createGroq({ apiKey: groqApiKey });

          // Fetch real data from MongoDB via Python backend
          await refreshCache();
          const mentionedCodes = extractDriverCodes(parsed.messages ?? []);
          const groundingData = buildGroundingData(mentionedCodes);

          // Add VectorProfile narratives for mentioned drivers
          const narrativeSections: string[] = [];
          for (const code of mentionedCodes) {
            const narrative = await getVectorNarrative(code);
            if (narrative) narrativeSections.push(`### ${code} Full Profile\n${narrative}`);
          }
          const narrativeBlock = narrativeSections.length > 0
            ? `\n\n## Driver Intelligence Profiles\n${narrativeSections.join('\n\n')}`
            : '';

          const systemPrompt = `${DIAGNOSE_SYSTEM_BASE}\n\n--- REAL DATA ---\n${groundingData}${narrativeBlock}\n--- END DATA ---`;

          // Convert UIMessages (parts-based) to ModelMessages for streamText
          const modelMessages = await convertToModelMessages(
            parsed.messages ?? [],
          );

          const result = streamText({
            model: groq('llama-3.3-70b-versatile'),
            system: systemPrompt,
            messages: modelMessages,
            tools: {
              metric_card: tool({
                description: 'Display a key diagnostic metric with trend indicator. Use for health %, peak values, or comparisons.',
                inputSchema: z.object({
                  title: z.string().describe('Metric name, e.g. "Current Health" or "Peak RPM"'),
                  value: z.string().describe('Display value with unit, e.g. "65%" or "12,400 RPM"'),
                  trend: z.enum(['up', 'down', 'stable']).describe('Trend direction relative to previous races'),
                  severity: z.enum(['nominal', 'warning', 'critical']).describe('Health severity level'),
                  subtitle: z.string().optional().describe('Brief context, e.g. "Down 13% since Abu Dhabi"'),
                }),
              }),
              sparkline: tool({
                description: 'Show a mini line chart of a metric across races. Use for health trends, RPM trends, temperature trends.',
                inputSchema: z.object({
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
                inputSchema: z.object({
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
                inputSchema: z.object({
                  title: z.string().describe('Short recommendation title'),
                  description: z.string().describe('Detailed explanation'),
                  severity: z.enum(['info', 'warning', 'critical']),
                  action: z.string().describe('Specific action to take'),
                }),
              }),
              text: tool({
                description: 'Display a paragraph of analysis text. Use for connecting the data points and providing engineering insight.',
                inputSchema: z.object({
                  content: z.string().describe('Analysis text (1-3 sentences)'),
                }),
              }),
              similar_drivers: tool({
                description: 'Find and display drivers with similar performance profiles. Use when asked "who drives like X?" or "find comparable drivers". Shows similarity scores and team.',
                inputSchema: z.object({
                  driver_code: z.string().describe('3-letter driver code to find similar drivers for, e.g. "NOR"'),
                  count: z.number().optional().describe('Number of similar drivers to show (default 5)'),
                }),
                execute: async ({ driver_code, count }) => {
                  const results = await getSimilarDrivers(driver_code.toUpperCase(), count ?? 5);
                  return { driver_code: driver_code.toUpperCase(), similar: results };
                },
              }),
            },
            temperature: 0.4,
            maxOutputTokens: 2048,
          });

          result.pipeUIMessageStreamToResponse(res);
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
        target: 'http://127.0.0.1:8300',
        changeOrigin: true,
        rewrite: (path: string) => path.replace(/^\/api\/chat/, '/chat'),
      },
      '/api/health': {
        target: 'http://127.0.0.1:8300',
        changeOrigin: true,
        rewrite: (path: string) => path.replace(/^\/api\/health/, '/health'),
      },
      '/api/visual-search': {
        target: 'http://127.0.0.1:8300',
        changeOrigin: true,
        rewrite: (path: string) => path.replace(/^\/api\/visual-search/, '/visual-search'),
      },
      '/api/visual-tags': {
        target: 'http://127.0.0.1:8300',
        changeOrigin: true,
        rewrite: (path: string) => path.replace(/^\/api\/visual-tags/, '/visual-tags'),
      },
      '/api/upload': {
        target: 'http://127.0.0.1:8300',
        changeOrigin: true,
        rewrite: (path: string) => path.replace(/^\/api\/upload/, '/upload'),
      },
      '/api/3d-gen': {
        target: 'http://127.0.0.1:8300',
        changeOrigin: true,
      },
      '/api/omni': {
        target: 'http://127.0.0.1:8300',
        changeOrigin: true,
      },
      '/api/local/mccar-summary': { target: 'http://127.0.0.1:8300', changeOrigin: true },
      '/api/local/mccar-telemetry': { target: 'http://127.0.0.1:8300', changeOrigin: true },
      '/api/local/mccar-race-telemetry': { target: 'http://127.0.0.1:8300', changeOrigin: true },
      '/api/local/mccar-race-stints': { target: 'http://127.0.0.1:8300', changeOrigin: true },
      '/api/local/mcdriver-summary': { target: 'http://127.0.0.1:8300', changeOrigin: true },
      '/api/local/strategy': { target: 'http://127.0.0.1:8300', changeOrigin: true },
      '/api/local/driver_intel': { target: 'http://127.0.0.1:8300', changeOrigin: true },
      '/api/local/team_intel': { target: 'http://127.0.0.1:8300', changeOrigin: true },
      '/api/constructor_profiles': { target: 'http://127.0.0.1:8300', changeOrigin: true, rewrite: (path: string) => path.replace(/^\/api\//, '/api/local/') },
      '/api/jolpica': { target: 'http://127.0.0.1:8300', changeOrigin: true, rewrite: (path: string) => path.replace(/^\/api\//, '/api/local/') },
      '/api/openf1': { target: 'http://127.0.0.1:8300', changeOrigin: true, rewrite: (path: string) => path.replace(/^\/api\//, '/api/local/') },
      '/api/opponents': { target: 'http://127.0.0.1:8300', changeOrigin: true },
      '/api/driver_intel': { target: 'http://127.0.0.1:8300', changeOrigin: true, rewrite: (path: string) => path.replace(/^\/api\//, '/api/local/') },
      '/api/circuit_intel': { target: 'http://127.0.0.1:8300', changeOrigin: true, rewrite: (path: string) => path.replace(/^\/api\//, '/api/local/') },
      '/api/anomaly': { target: 'http://127.0.0.1:8300', changeOrigin: true, rewrite: (path: string) => path.replace(/^\/api\//, '/api/local/') },
      '/api/forecast': { target: 'http://127.0.0.1:8300', changeOrigin: true, rewrite: (path: string) => path.replace(/^\/api\//, '/api/local/') },
      '/api/pipeline': { target: 'http://127.0.0.1:8300', changeOrigin: true, rewrite: (path: string) => path.replace(/^\/api\//, '/api/local/') },
      '/api/f1data': { target: 'http://127.0.0.1:8300', changeOrigin: true, rewrite: (path: string) => path.replace(/^\/api\//, '/api/local/') },
      '/api/mccar': { target: 'http://127.0.0.1:8300', changeOrigin: true, rewrite: (path: string) => path.replace(/^\/api\//, '/api/local/') },
      '/api/mcdriver': { target: 'http://127.0.0.1:8300', changeOrigin: true, rewrite: (path: string) => path.replace(/^\/api\//, '/api/local/') },
      '/api/mcracecontext': { target: 'http://127.0.0.1:8300', changeOrigin: true, rewrite: (path: string) => path.replace(/^\/api\//, '/api/local/') },
      '/3d-models': {
        target: 'http://127.0.0.1:8300',
        changeOrigin: true,
      },
    },
  },
})
