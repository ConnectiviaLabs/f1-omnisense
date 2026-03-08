import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import {
  Brain, Video, Scan, Loader2, Search, Tag, ImageIcon, Film, Sparkles,
  Upload, ChevronDown,
} from 'lucide-react';
// ── Interfaces ──

interface GDinoFrame {
  frame_index: number;
  detections: { category: string; score: number; bbox: number[] }[];
  output_image: string;
}

interface NarrationFrame {
  frame_index: number;
  narration: string;
  tokens: number;
  time_s: number;
  tok_per_s: number;
}

interface VideoModelResult {
  total_frames: number;
  fps: number;
  inference_time_s: number;
  top_predictions: { label: string; score: number }[];
}

interface VisualSearchResult {
  path: string;
  score: number;
  auto_tags: { label: string; score: number }[];
  source_video: string;
  frame_index: number;
}

interface VisualTagImage {
  path: string;
  auto_tags: { label: string; score: number }[];
  source_video: string;
  frame_index: number;
}

interface MediaVideo {
  filename: string;
  size_mb: number;
  added: string;
  analyzed: boolean;
}

interface StructuredAnalysis {
  scene: string;
  key_objects: string[];
  track_conditions: string;
  strategic_notes: string;
  incidents: string[];
}

interface DetectionSummary {
  category: string;
  count: number;
  avgConfidence: number;
  bestConfidence: number;
}

// ── Helpers ──

function computeDetectionSummary(frames: GDinoFrame[]): DetectionSummary[] {
  const map: Record<string, { count: number; scores: number[] }> = {};
  for (const f of frames) {
    for (const d of (f.detections ?? [])) {
      if (!d.category) continue;
      if (!map[d.category]) map[d.category] = { count: 0, scores: [] };
      map[d.category].count++;
      map[d.category].scores.push(d.score ?? 0);
    }
  }
  return Object.entries(map)
    .map(([category, { count, scores }]) => ({
      category,
      count,
      avgConfidence: scores.reduce((a, b) => a + b, 0) / scores.length,
      bestConfidence: Math.max(...scores),
    }))
    .sort((a, b) => b.count - a.count);
}

// ── AnnotatedVideoPlayer ──

function AnnotatedVideoPlayer({
  videoSrc, frames, fps, narrations,
  confidenceThreshold, activeCategories,
}: {
  videoSrc: string;
  frames: GDinoFrame[];
  fps: number;
  narrations?: NarrationFrame[];
  confidenceThreshold: number;
  activeCategories: Set<string>;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [activeFrameIdx, setActiveFrameIdx] = useState(0);
  const hasFrames = frames.length > 0;

  const frameTimes = frames.map(f => f.frame_index / fps);

  const catColors: Record<string, string> = {};
  const palette = ['#FF8000', '#22c55e', '#3b82f6', '#a855f7', '#ef4444', '#eab308', '#06b6d4'];
  let colorIdx = 0;
  const getColor = (cat: string) => {
    if (!catColors[cat]) { catColors[cat] = palette[colorIdx % palette.length]; colorIdx++; }
    return catColors[cat];
  };

  const drawBoxes = useCallback((frameIdx: number) => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const rect = video.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const frame = frames[frameIdx];
    if (!frame?.detections?.length) return;

    const scaleX = rect.width / (video.videoWidth || 1);
    const scaleY = rect.height / (video.videoHeight || 1);

    const NMS_IOU = 0.45;
    const filtered = frame.detections
      .filter((d: any) =>
        d.bbox?.length >= 4 &&
        (d.score ?? 0) >= confidenceThreshold &&
        activeCategories.has(d.category)
      )
      .sort((a: any, b: any) => (b.score ?? 0) - (a.score ?? 0));

    // NMS
    const iou = (a: number[], b: number[]) => {
      const ix1 = Math.max(a[0], b[0]), iy1 = Math.max(a[1], b[1]);
      const ix2 = Math.min(a[2], b[2]), iy2 = Math.min(a[3], b[3]);
      const inter = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
      const areaA = (a[2] - a[0]) * (a[3] - a[1]);
      const areaB = (b[2] - b[0]) * (b[3] - b[1]);
      return inter / (areaA + areaB - inter + 1e-6);
    };
    const keep: typeof filtered = [];
    for (const det of filtered) {
      if (keep.some(k => iou(k.bbox, det.bbox) > NMS_IOU)) continue;
      keep.push(det);
    }

    const shortLabel = (cat: string) => {
      const words = cat.split(/[\s\-–]+/).filter(w => !['a','the','of','and','with','car','one'].includes(w.toLowerCase()));
      return (words[0] || cat).slice(0, 12);
    };

    const labelRects: { x: number; y: number; w: number; h: number }[] = [];
    const LABEL_H = 14;

    for (const det of keep) {
      const [x1, y1, x2, y2] = det.bbox;
      const sx1 = x1 * scaleX, sy1 = y1 * scaleY;
      const sw = (x2 - x1) * scaleX, sh = (y2 - y1) * scaleY;
      const color = getColor(det.category);

      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([]);
      ctx.strokeRect(sx1, sy1, sw, sh);

      const label = `${shortLabel(det.category)} ${((det.score ?? 0) * 100).toFixed(0)}%`;
      ctx.font = 'bold 9px system-ui, sans-serif';
      const textW = ctx.measureText(label).width + 6;

      let labelY = sy1 - 2;
      let attempts = 0;
      while (attempts < 8) {
        const candidate = { x: sx1, y: labelY - LABEL_H, w: textW, h: LABEL_H };
        const overlaps = labelRects.some(r =>
          candidate.x < r.x + r.w && candidate.x + candidate.w > r.x &&
          candidate.y < r.y + r.h && candidate.y + candidate.h > r.y
        );
        if (!overlaps) break;
        labelY -= LABEL_H + 1;
        attempts++;
      }
      if (labelY - LABEL_H < 0) labelY = sy1 + sh + LABEL_H + 2;

      labelRects.push({ x: sx1, y: labelY - LABEL_H, w: textW, h: LABEL_H });

      ctx.fillStyle = color;
      ctx.globalAlpha = 0.85;
      ctx.fillRect(sx1, labelY - LABEL_H, textW, LABEL_H);
      ctx.globalAlpha = 1.0;

      ctx.fillStyle = '#000';
      ctx.fillText(label, sx1 + 3, labelY - 3);
    }
  }, [frames, confidenceThreshold, activeCategories]);

  const onTimeUpdate = useCallback(() => {
    const video = videoRef.current;
    if (!video || !hasFrames) return;
    const t = video.currentTime;
    let best = 0;
    let bestDist = Math.abs(t - frameTimes[0]);
    for (let i = 1; i < frameTimes.length; i++) {
      const dist = Math.abs(t - frameTimes[i]);
      if (dist < bestDist) { best = i; bestDist = dist; }
    }
    if (best !== activeFrameIdx) {
      setActiveFrameIdx(best);
      drawBoxes(best);
    }
  }, [frameTimes, hasFrames, activeFrameIdx, drawBoxes]);

  const seekToFrame = (idx: number) => {
    setActiveFrameIdx(idx);
    drawBoxes(idx);
    if (videoRef.current) videoRef.current.currentTime = frameTimes[idx];
  };

  // Redraw on resize or filter change
  useEffect(() => {
    drawBoxes(activeFrameIdx);
  }, [activeFrameIdx, drawBoxes, confidenceThreshold, activeCategories]);

  useEffect(() => {
    const handleResize = () => drawBoxes(activeFrameIdx);
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [activeFrameIdx, drawBoxes]);

  const activeFrame = frames[activeFrameIdx];
  const activeNarration = narrations?.find(n => {
    if (!activeFrame) return false;
    return n.frame_index === activeFrame.frame_index || n.frame_index === activeFrameIdx + 1;
  });

  return (
    <div className="space-y-2">
      {/* Video + canvas overlay */}
      <div ref={containerRef} className="relative rounded-lg overflow-hidden border border-border">
        <video
          ref={videoRef}
          controls
          onTimeUpdate={onTimeUpdate}
          onPlay={() => drawBoxes(activeFrameIdx)}
          onSeeked={() => drawBoxes(activeFrameIdx)}
          className="w-full block"
          src={videoSrc}
        />
        <canvas
          ref={canvasRef}
          className="absolute top-0 left-0 w-full h-full pointer-events-none"
          style={{ height: 'calc(100% - 36px)' }}
        />
        {hasFrames && activeFrame && (
          <div className="absolute top-2 left-2 flex items-center gap-2 pointer-events-none">
            <span className="text-[10px] font-mono bg-black/70 text-[#FF8000] px-2 py-0.5 rounded">
              Frame {activeFrame.frame_index} — {frameTimes[activeFrameIdx].toFixed(1)}s
            </span>
            <span className="text-[10px] font-mono bg-black/70 text-green-400 px-2 py-0.5 rounded">
              {activeFrame.detections?.filter(d => (d.score ?? 0) >= confidenceThreshold && activeCategories.has(d.category)).length ?? 0} detections
            </span>
          </div>
        )}
      </div>

      {/* Detection timeline heatmap */}
      {hasFrames && (
        <div className="relative h-5 bg-background rounded overflow-hidden border border-border">
          {frames.map((f, i) => {
            const detCount = f.detections?.filter(d =>
              (d.score ?? 0) >= confidenceThreshold && activeCategories.has(d.category)
            ).length ?? 0;
            const maxDets = Math.max(...frames.map(fr => fr.detections?.length ?? 0), 1);
            const intensity = detCount / maxDets;
            return (
              <button
                key={f.frame_index}
                type="button"
                onClick={() => seekToFrame(i)}
                className="absolute top-0 bottom-0 hover:ring-1 hover:ring-[#FF8000]/50"
                style={{
                  left: `${(i / frames.length) * 100}%`,
                  width: `${Math.max(100 / frames.length, 2)}%`,
                  backgroundColor: `rgba(255, 128, 0, ${0.05 + intensity * 0.7})`,
                }}
                title={`${frameTimes[i].toFixed(1)}s — ${detCount} detections`}
              />
            );
          })}
          <div
            className="absolute top-0 bottom-0 w-0.5 bg-white/80 z-10 pointer-events-none"
            style={{ left: `${(activeFrameIdx / Math.max(frames.length - 1, 1)) * 100}%` }}
          />
        </div>
      )}

      {/* AI narration for frame */}
      {activeNarration && (
        <div className="bg-background border border-purple-500/20 rounded-lg p-2.5">
          <div className="flex items-center gap-2 mb-1">
            <Brain className="w-3 h-3 text-purple-400" />
            <span className="text-[10px] text-purple-400 tracking-wider font-mono">AI INSIGHT</span>
            <span className="text-[9px] text-muted-foreground ml-auto">{activeNarration.tokens} tok</span>
          </div>
          <p className="text-[11px] text-foreground/90 leading-relaxed">{activeNarration.narration}</p>
        </div>
      )}

      {/* Frame strip */}
      {hasFrames && (
        <div className="flex gap-1 overflow-x-auto pb-1">
          {frames.map((f, i) => {
            const detCount = f.detections?.filter(d =>
              (d.score ?? 0) >= confidenceThreshold && activeCategories.has(d.category)
            ).length ?? 0;
            return (
              <button
                key={f.frame_index}
                type="button"
                onClick={() => seekToFrame(i)}
                className={`shrink-0 rounded-md px-2 py-1 text-center transition-all border ${
                  i === activeFrameIdx
                    ? 'bg-[#FF8000]/15 border-[#FF8000] text-[#FF8000]'
                    : 'bg-background border-border text-muted-foreground hover:border-[#FF8000]/30'
                }`}
              >
                <div className="text-[9px] font-mono">{frameTimes[i].toFixed(1)}s</div>
                <div className="text-[8px]">{detCount} det</div>
              </button>
            );
          })}
        </div>
      )}

      {!hasFrames && (
        <div className="text-[11px] text-muted-foreground text-center py-3">
          No detection data — click <span className="text-[#FF8000]">Analyze</span> to run the pipeline
        </div>
      )}
    </div>
  );
}

// ── Main Component ──

export function MediaIntelligence() {
  const [gdinoData, setGdinoData] = useState<Record<string, GDinoFrame[]> | null>(null);
  // fusedData removed — gdinoData now loaded per-video from backend
  const [videomaeData, setVideomaeData] = useState<Record<string, VideoModelResult> | null>(null);
  const [timesformerData, setTimesformerData] = useState<Record<string, VideoModelResult> | null>(null);
  const [loading, setLoading] = useState(true);
  const [narrationData, setNarrationData] = useState<Record<string, NarrationFrame[]> | null>(null);

  // CLIP
  const [clipQuery, setClipQuery] = useState('');
  const [clipResults, setClipResults] = useState<VisualSearchResult[] | null>(null);
  const [clipSearching, setClipSearching] = useState(false);
  const [allTags, setAllTags] = useState<VisualTagImage[] | null>(null);
  const [selectedTag, setSelectedTag] = useState<string | null>(null);
  const [topTags, setTopTags] = useState<{ label: string; max_score: number }[]>([]);
  const [clipExpanded, setClipExpanded] = useState(false);

  // Video state
  const [videos, setVideos] = useState<MediaVideo[]>([]);
  const [selectedVideo, setSelectedVideo] = useState<string | null>(null);

  // Detection controls
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.35);
  const [activeCategories, setActiveCategories] = useState<Set<string>>(new Set());

  // Analysis pipeline
  const [analyzeProgress, setAnalyzeProgress] = useState<'idle' | 'detecting' | 'analyzing' | 'done' | 'error'>('idle');
  const [vlmAnalysis, setVlmAnalysis] = useState<{ analysis: string; structured?: StructuredAnalysis | null; tokens: number; time_s: number; tok_per_s: number; frames_analyzed: number; model: string } | null>(null);

  // Upload
  const [uploadDragging, setUploadDragging] = useState(false);

  // Load video list + CLIP tags on mount
  useEffect(() => {
    Promise.allSettled([
      fetch('/api/omni/vis/videos')
        .then(r => r.ok ? r.json() : null)
        .then(data => { if (data?.videos) setVideos(data.videos); }),
      fetch('/api/omni/vis/clip/tags')
        .then(r => r.ok ? r.json() : null)
        .then(data => {
          if (data) { setAllTags(data.images); setTopTags(data.tags || []); }
        }),
    ]).finally(() => setLoading(false));
  }, []);

  // Load per-video results when selection changes
  useEffect(() => {
    if (!selectedVideo) return;
    fetch(`/api/omni/vis/results/${encodeURIComponent(selectedVideo)}`)
      .then(r => r.ok ? r.json() : null)
      .then(data => {
        if (!data) return;
        // GDino frames
        if (data.gdino_frames?.length) {
          setGdinoData(prev => ({ ...prev, [data.filename]: data.gdino_frames }));
        }
        // VideoMAE
        if (data.videomae) {
          const v = data.videomae;
          setVideomaeData(prev => ({ ...prev, [data.filename]: { total_frames: v.total_frames, fps: v.fps, inference_time_s: v.inference_time_s ?? 0, top_predictions: v.top_predictions } }));
        }
        // TimeSformer
        if (data.timesformer) {
          const t = data.timesformer;
          setTimesformerData(prev => ({ ...prev, [data.filename]: { total_frames: t.total_frames, fps: t.fps, inference_time_s: t.inference_time_s ?? 0, top_predictions: t.top_predictions } }));
        }
        // MiniCPM narrations
        if (data.minicpm_narrations?.length) {
          setNarrationData(prev => ({ ...prev, [data.filename]: data.minicpm_narrations }));
        }
        // VLM analysis (if previously run)
        if (data.vlm_analysis) {
          setVlmAnalysis(data.vlm_analysis);
        }
      })
      .catch(() => {});
  }, [selectedVideo]);

  // Compute categories from all detection data
  const allFrames = useMemo(() =>
    gdinoData ? Object.values(gdinoData).flat().filter(Boolean) : [],
  [gdinoData]);

  const categories = useMemo(() =>
    [...new Set(allFrames.flatMap(f => (f.detections ?? []).map(d => d.category)).filter(Boolean))],
  [allFrames]);

  const totalDetections = allFrames.reduce((s, f) => s + (f.detections?.length ?? 0), 0);
  const totalFramesAnalyzed = allFrames.length;

  // Init categories filter when data loads
  useEffect(() => {
    if (categories.length > 0 && activeCategories.size === 0) {
      setActiveCategories(new Set(categories));
    }
  }, [categories]);

  // ── Handlers ──

  const analyzeVideo = async (filename: string) => {
    setAnalyzeProgress('detecting');
    setVlmAnalysis(null);

    try {
      // Step 1: Run GDino via backend
      const gdinoRes = await fetch('/api/omni/vis/analyze-video-by-name', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename }),
      });
      if (gdinoRes.ok) {
        const gdinoResult = await gdinoRes.json();
        if (gdinoResult?.frames?.length) {
          setGdinoData(prev => ({ ...prev, [filename]: gdinoResult.frames }));
        }
      }

      // Step 2: VLM analysis via Groq
      setAnalyzeProgress('analyzing');
      const res = await fetch('/api/omni/vis/vlm-analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename }),
      });

      if (res.ok) {
        const data = await res.json();
        setVlmAnalysis(data);
      }
      setAnalyzeProgress('done');
    } catch {
      setAnalyzeProgress('error');
    }
  };

  const handleUpload = async (file: File) => {
    const formData = new FormData();
    formData.append('video', file);
    const res = await fetch('/api/omni/vis/upload', {
      method: 'POST',
      body: formData,
    });
    if (res.ok) {
      const data = await res.json();
      setVideos(prev => [data, ...prev]);
    }
  };

  const doClipSearch = async (query: string) => {
    if (!query.trim()) { setClipResults(null); return; }
    setClipSearching(true);
    try {
      const res = await fetch(`/api/omni/vis/clip/search?q=${encodeURIComponent(query)}&k=12`);
      if (res.ok) setClipResults((await res.json()).results);
    } catch { /* */ }
    finally { setClipSearching(false); }
  };

  const filteredByTag = selectedTag && allTags
    ? allTags.filter(img => img.auto_tags.some(t => t.label === selectedTag))
    : null;

  // ── Render ──

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="w-6 h-6 text-[#FF8000] animate-spin" />
        <span className="ml-3 text-muted-foreground text-sm">Loading media pipeline...</span>
      </div>
    );
  }

  return (
    <div className="space-y-3">

      {/* ── HEADER BAR ── */}
      <div className="bg-card border border-border rounded-xl shadow-[0_1px_3px_rgba(0,0,0,0.3)] px-4 py-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-[#FF8000]/10 flex items-center justify-center">
              <Brain className="w-4 h-4 text-[#FF8000]" />
            </div>
            <div>
              <h3 className="text-sm text-foreground tracking-wide">OMNISENSE MEDIA INTELLIGENCE</h3>
              <div className="text-[11px] text-muted-foreground font-mono">
                GDino + SAM2 + VideoMAE + TimeSformer + CLIP + VLM
              </div>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-3 text-[11px] text-muted-foreground">
              <span>{videos.length} videos</span>
              <span className="text-[rgba(255,128,0,0.3)]">|</span>
              <span>{totalFramesAnalyzed} frames</span>
              <span className="text-[rgba(255,128,0,0.3)]">|</span>
              <span>{totalDetections} detections</span>
            </div>
            <label className="flex items-center gap-1.5 px-3 py-1.5 text-[11px] bg-[#FF8000]/15 text-[#FF8000] rounded-lg hover:bg-[#FF8000]/25 cursor-pointer transition-colors">
              <Upload className="w-3 h-3" />
              Upload
              <input
                type="file"
                accept=".mp4,.webm,.mov"
                className="hidden"
                onChange={e => { if (e.target.files?.[0]) handleUpload(e.target.files[0]); }}
              />
            </label>
          </div>
        </div>
      </div>

      {/* ── VIDEO STRIP ── */}
      <div
        className={`bg-card border rounded-xl shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-3 transition-colors ${
          uploadDragging ? 'border-[#FF8000]' : 'border-border'
        }`}
        onDragOver={e => { e.preventDefault(); setUploadDragging(true); }}
        onDragLeave={() => setUploadDragging(false)}
        onDrop={e => {
          e.preventDefault();
          setUploadDragging(false);
          const file = e.dataTransfer.files[0];
          if (file && /\.(mp4|webm|mov)$/i.test(file.name)) handleUpload(file);
        }}
      >
        <div className="flex gap-2 overflow-x-auto pb-1">
          {videos.map(v => {
            const isSelected = selectedVideo === v.filename;
            const hasGdino = !!gdinoData?.[v.filename]?.length;
            const thumbFrame = gdinoData?.[v.filename]?.[0]?.output_image;
            return (
              <button
                key={v.filename}
                type="button"
                onClick={() => {
                  setSelectedVideo(isSelected ? null : v.filename);
                  setAnalyzeProgress('idle');
                  setVlmAnalysis(null);
                }}
                className={`shrink-0 w-36 rounded-lg overflow-hidden border transition-all ${
                  isSelected
                    ? 'border-[#FF8000] ring-1 ring-[#FF8000]/30'
                    : 'border-border hover:border-[#FF8000]/30'
                }`}
              >
                <div className="h-20 bg-background relative">
                  {thumbFrame ? (
                    <img
                      src={`/media/gdino_results/${thumbFrame}`}
                      className="w-full h-full object-cover"
                      alt=""
                    />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center">
                      <Film className="w-5 h-5 text-muted-foreground/40" />
                    </div>
                  )}
                  <div className="absolute top-1 right-1">
                    {hasGdino ? (
                      <span className="text-[8px] bg-green-500/20 text-green-400 px-1.5 py-0.5 rounded font-medium">
                        {gdinoData![v.filename].length}f
                      </span>
                    ) : (
                      <span className="text-[8px] bg-[#FF8000]/20 text-[#FF8000] px-1.5 py-0.5 rounded">Pending</span>
                    )}
                  </div>
                </div>
                <div className="px-2 py-1.5 bg-background">
                  <div className="text-[10px] text-foreground truncate">{v.filename}</div>
                  <div className="text-[9px] text-muted-foreground">{v.size_mb} MB</div>
                </div>
              </button>
            );
          })}
          {videos.length === 0 && (
            <div className="text-[12px] text-muted-foreground py-6 text-center w-full">
              {uploadDragging ? 'Drop video here...' : 'No videos — drag & drop or click Upload'}
            </div>
          )}
        </div>
      </div>

      {/* ── MAIN CONTENT ── */}
      {selectedVideo && (() => {
        const resultData = gdinoData;
        const frames = resultData?.[selectedVideo] ?? [];
        const fps = videomaeData?.[selectedVideo]?.fps ?? timesformerData?.[selectedVideo]?.fps ?? 30;
        const hasGdino = frames.length > 0;
        const summary = computeDetectionSummary(frames);
        const vmae = videomaeData?.[selectedVideo];
        const tsf = timesformerData?.[selectedVideo];
        const maxCount = Math.max(...summary.map(s => s.count), 1);
        const structured = vlmAnalysis?.structured;

        return (
          <div className="grid grid-cols-12 gap-3">

            {/* LEFT — Video Player + Controls (col-span-8) */}
            <div className="col-span-8 space-y-3">
              {/* Controls bar — above player so it's always visible */}
              <div className="bg-card border border-border rounded-xl shadow-[0_1px_3px_rgba(0,0,0,0.3)] px-4 py-3">
                <div className="flex items-center gap-4 flex-wrap">
                  {/* Analyze button */}
                  <button
                    type="button"
                    onClick={() => analyzeVideo(selectedVideo)}
                    disabled={analyzeProgress === 'detecting' || analyzeProgress === 'analyzing'}
                    className="flex items-center gap-2 px-4 py-2 bg-[#FF8000]/20 text-[#FF8000] text-[12px] rounded-lg hover:bg-[#FF8000]/30 disabled:opacity-40 transition-colors"
                  >
                    {analyzeProgress === 'detecting' && <><Loader2 className="w-3.5 h-3.5 animate-spin" /> Detecting...</>}
                    {analyzeProgress === 'analyzing' && <><Loader2 className="w-3.5 h-3.5 animate-spin" /> AI Analyzing...</>}
                    {(analyzeProgress === 'idle' || analyzeProgress === 'done' || analyzeProgress === 'error') && (
                      <><Sparkles className="w-3.5 h-3.5" /> {hasGdino ? 'Re-Analyze' : 'Analyze'}</>
                    )}
                  </button>

                  {/* Confidence slider */}
                  <div className="flex items-center gap-2">
                    <span className="text-[10px] text-muted-foreground">Confidence</span>
                    <input
                      type="range"
                      min="0" max="100" step="1"
                      title="Detection confidence threshold"
                      value={Math.round(confidenceThreshold * 100)}
                      onChange={e => setConfidenceThreshold(Number(e.target.value) / 100)}
                      className="w-20 accent-[#FF8000] h-1"
                    />
                    <span className="text-[10px] font-mono text-[#FF8000] w-7">{Math.round(confidenceThreshold * 100)}%</span>
                  </div>

                  {/* Category toggles */}
                  <div className="flex items-center gap-1 flex-wrap">
                    {categories.slice(0, 12).map(cat => {
                      const active = activeCategories.has(cat);
                      return (
                        <button
                          key={cat}
                          type="button"
                          onClick={() => {
                            setActiveCategories(prev => {
                              const next = new Set(prev);
                              if (next.has(cat)) next.delete(cat); else next.add(cat);
                              return next;
                            });
                          }}
                          className={`text-[9px] px-1.5 py-0.5 rounded-full border transition-all ${
                            active
                              ? 'border-[#FF8000]/30 bg-[#FF8000]/15 text-[#FF8000]'
                              : 'border-border text-muted-foreground/40 line-through'
                          }`}
                        >
                          {(cat ?? '').split(/\s+/)[0]}
                        </button>
                      );
                    })}
                  </div>
                </div>

                {analyzeProgress === 'error' && (
                  <div className="mt-2 text-[11px] text-red-400 bg-red-500/10 rounded-lg px-3 py-2">
                    Analysis failed. Check that the backend server is running on port 8300.
                  </div>
                )}
              </div>

              {/* Video Player */}
              <div className="bg-card border border-border rounded-xl shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-4">
                <AnnotatedVideoPlayer
                  key={selectedVideo}
                  videoSrc={`/media/${encodeURIComponent(selectedVideo)}`}
                  frames={frames}
                  fps={fps}
                  narrations={narrationData?.[selectedVideo] ?? undefined}
                  confidenceThreshold={confidenceThreshold}
                  activeCategories={activeCategories}
                />
              </div>
            </div>

            {/* RIGHT — AI Insights + Detection Summary (col-span-4) */}
            <div className="col-span-4 space-y-3">

              {/* AI Insights */}
              <div className="bg-card border border-border rounded-xl shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-4">
                <h3 className="text-[11px] text-muted-foreground tracking-widest mb-3 flex items-center gap-2">
                  <Brain className="w-3 h-3" />
                  AI INSIGHTS
                </h3>

                {structured ? (
                  <div className="space-y-3">
                    <div>
                      <div className="text-[9px] text-[#FF8000] tracking-wider mb-1">SCENE</div>
                      <p className="text-[11px] text-foreground/90 leading-relaxed">{structured.scene}</p>
                    </div>
                    <div>
                      <div className="text-[9px] text-[#FF8000] tracking-wider mb-1">KEY OBJECTS</div>
                      <div className="flex flex-wrap gap-1">
                        {structured.key_objects?.map(obj => (
                          <span key={obj} className="text-[9px] px-1.5 py-0.5 rounded bg-[#FF8000]/10 text-[#FF8000]">{obj}</span>
                        ))}
                      </div>
                    </div>
                    <div>
                      <div className="text-[9px] text-cyan-400 tracking-wider mb-1">CONDITIONS</div>
                      <p className="text-[11px] text-foreground/90">{structured.track_conditions}</p>
                    </div>
                    <div>
                      <div className="text-[9px] text-purple-400 tracking-wider mb-1">STRATEGY</div>
                      <p className="text-[11px] text-foreground/90">{structured.strategic_notes}</p>
                    </div>
                    {structured.incidents?.length > 0 && structured.incidents[0] !== '' && (
                      <div>
                        <div className="text-[9px] text-red-400 tracking-wider mb-1">INCIDENTS</div>
                        {structured.incidents.map((inc, i) => (
                          <div key={i} className="text-[11px] text-foreground/90 flex items-start gap-1.5">
                            <span className="text-red-400 mt-0.5">-</span>
                            <span>{inc}</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                ) : vlmAnalysis ? (
                  <p className="text-[11px] text-foreground/90 leading-relaxed whitespace-pre-line">{vlmAnalysis.analysis}</p>
                ) : (
                  <div className="text-[11px] text-muted-foreground text-center py-6">
                    Click <span className="text-[#FF8000]">Analyze</span> to generate insights
                  </div>
                )}

                {vlmAnalysis && (
                  <div className="mt-2 pt-2 border-t border-border text-[9px] text-muted-foreground font-mono">
                    {vlmAnalysis.model} | {vlmAnalysis.frames_analyzed}f | {vlmAnalysis.tokens} tok | {vlmAnalysis.tok_per_s} tok/s
                  </div>
                )}
              </div>

              {/* Detection Summary */}
              {summary.length > 0 && (
                <div className="bg-card border border-border rounded-xl shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-4">
                  <h3 className="text-[11px] text-muted-foreground tracking-widest mb-3 flex items-center gap-2">
                    <Scan className="w-3 h-3" />
                    DETECTION SUMMARY
                  </h3>
                  <div className="space-y-2">
                    {summary.slice(0, 10).map(s => (
                      <div key={s.category}>
                        <div className="flex items-center justify-between mb-0.5">
                          <span className="text-[10px] text-foreground">{s.category}</span>
                          <span className="text-[9px] font-mono text-muted-foreground">
                            {s.count}x · {Math.round(s.avgConfidence * 100)}%
                          </span>
                        </div>
                        <div className="h-1 bg-background rounded-full overflow-hidden">
                          <div
                            className="h-full bg-[#FF8000] rounded-full transition-all"
                            style={{ width: `${(s.count / maxCount) * 100}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Classification */}
              {(vmae || tsf) && (
                <div className="bg-card border border-border rounded-xl shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-4">
                  <h3 className="text-[11px] text-muted-foreground tracking-widest mb-3 flex items-center gap-2">
                    <Video className="w-3 h-3" />
                    CLASSIFICATION
                  </h3>
                  <div className="space-y-2">
                    {vmae && (
                      <div>
                        <div className="text-[9px] text-purple-400 tracking-wider mb-1">VideoMAE</div>
                        {vmae.top_predictions.slice(0, 3).map((p, i) => (
                          <div key={i} className="flex items-center gap-2 text-[10px]">
                            <div className="w-10 h-1 bg-secondary rounded-full overflow-hidden">
                              <div className="h-full bg-purple-400 rounded-full" style={{ width: `${p.score * 100}%` }} />
                            </div>
                            <span className="font-mono text-foreground w-8">{(p.score * 100).toFixed(0)}%</span>
                            <span className="text-muted-foreground truncate">{p.label}</span>
                          </div>
                        ))}
                      </div>
                    )}
                    {tsf && (
                      <div className={vmae ? 'mt-2' : ''}>
                        <div className="text-[9px] text-cyan-400 tracking-wider mb-1">TimeSformer</div>
                        {tsf.top_predictions.slice(0, 3).map((p, i) => (
                          <div key={i} className="flex items-center gap-2 text-[10px]">
                            <div className="w-10 h-1 bg-secondary rounded-full overflow-hidden">
                              <div className="h-full bg-cyan-400 rounded-full" style={{ width: `${p.score * 100}%` }} />
                            </div>
                            <span className="font-mono text-foreground w-8">{(p.score * 100).toFixed(0)}%</span>
                            <span className="text-muted-foreground truncate">{p.label}</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        );
      })()}

      {/* ── CLIP VISUAL SEARCH (collapsible) ── */}
      <div className="bg-card border border-border rounded-xl shadow-[0_1px_3px_rgba(0,0,0,0.3)]">
        <button
          type="button"
          onClick={() => setClipExpanded(!clipExpanded)}
          className="w-full px-4 py-3 flex items-center justify-between"
        >
          <div className="flex items-center gap-3">
            <ImageIcon className="w-4 h-4 text-purple-400" />
            <span className="text-sm text-foreground">CLIP Visual Search</span>
            <span className="text-[11px] text-muted-foreground">ViT-B/32 · 512-dim</span>
          </div>
          <ChevronDown className={`w-4 h-4 text-muted-foreground transition-transform ${clipExpanded ? 'rotate-180' : ''}`} />
        </button>
        {clipExpanded && (
          <div className="px-4 pb-4 space-y-3">
            {/* Search bar */}
            <div className="flex items-center gap-2">
              <div className="flex-1 relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                <input
                  type="text"
                  placeholder='Search: "pit stop crew", "rain conditions", "cockpit view"...'
                  value={clipQuery}
                  onChange={e => setClipQuery(e.target.value)}
                  onKeyDown={e => { if (e.key === 'Enter') doClipSearch(clipQuery); }}
                  className="w-full bg-background border border-border rounded-lg pl-10 pr-4 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:border-purple-400/40"
                />
              </div>
              <button
                type="button"
                onClick={() => doClipSearch(clipQuery)}
                disabled={clipSearching || !clipQuery.trim()}
                className="px-4 py-2 bg-purple-500/20 text-purple-400 text-sm rounded-lg hover:bg-purple-500/30 disabled:opacity-30 transition-colors"
              >
                {clipSearching ? <Loader2 className="w-3 h-3 animate-spin" /> : 'Search'}
              </button>
            </div>

            {/* Tag chips */}
            {topTags.length > 0 && (
              <div className="flex items-center gap-1.5 flex-wrap">
                <Tag className="w-3 h-3 text-muted-foreground shrink-0" />
                <button
                  type="button"
                  onClick={() => { setSelectedTag(null); setClipResults(null); }}
                  className={`text-[11px] px-2 py-0.5 rounded-full transition-all ${
                    !selectedTag ? 'bg-purple-500/20 text-purple-400' : 'text-muted-foreground hover:bg-secondary'
                  }`}
                >
                  All
                </button>
                {topTags.slice(0, 12).map(tag => {
                  const shortLabel = tag.label.replace(/formula one |on a formula one car|of a formula one car/g, '').replace(/ car$/, '').trim();
                  return (
                    <button
                      key={tag.label}
                      type="button"
                      onClick={() => { setSelectedTag(selectedTag === tag.label ? null : tag.label); setClipResults(null); }}
                      className={`text-[11px] px-2 py-0.5 rounded-full transition-all ${
                        selectedTag === tag.label ? 'bg-purple-500/20 text-purple-400' : 'text-muted-foreground hover:bg-secondary'
                      }`}
                    >
                      {shortLabel}
                    </button>
                  );
                })}
              </div>
            )}

            {/* Search results */}
            {clipResults && (
              <div>
                <div className="text-[12px] text-muted-foreground mb-2">{clipResults.length} results for &quot;{clipQuery}&quot;</div>
                <div className="grid grid-cols-6 gap-2">
                  {clipResults.map((r, i) => (
                    <div key={r.path} className="relative rounded-lg overflow-hidden border border-border hover:border-purple-400/30 transition-all group">
                      <img src={`/media/${r.path}`} alt={r.path} className="w-full h-20 object-cover" />
                      <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                      <div className="absolute bottom-0 left-0 right-0 px-1.5 py-1 flex items-center justify-between">
                        <span className="text-[10px] text-purple-400 font-mono font-bold">{(r.score * 100).toFixed(1)}%</span>
                        <span className="text-[9px] text-muted-foreground">#{i + 1}</span>
                      </div>
                      <div className="absolute top-0 left-0 right-0 px-1.5 py-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
                        <span className="text-[9px] text-white/80">{r.source_video} f{r.frame_index}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Tag-filtered results */}
            {!clipResults && filteredByTag && (
              <div>
                <div className="text-[12px] text-muted-foreground mb-2">{filteredByTag.length} images tagged &quot;{selectedTag}&quot;</div>
                <div className="grid grid-cols-6 gap-2">
                  {filteredByTag.map(img => {
                    const tagScore = img.auto_tags.find(t => t.label === selectedTag)?.score ?? 0;
                    return (
                      <div key={img.path} className="relative rounded-lg overflow-hidden border border-border hover:border-purple-400/30 transition-all group">
                        <img src={`/media/${img.path}`} alt={img.path} className="w-full h-20 object-cover" />
                        <div className="absolute bottom-0 left-0 right-0 px-1.5 py-0.5 bg-black/70">
                          <span className="text-[10px] text-purple-400 font-mono">{(tagScore * 100).toFixed(1)}%</span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
