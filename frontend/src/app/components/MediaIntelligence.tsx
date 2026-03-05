import { useState, useEffect, useRef, useCallback } from 'react';
import {
  Brain, Video, Scan, Loader2, Search, Tag, ImageIcon, Play, Film,
} from 'lucide-react';
import { pipeline } from '../api/local';

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

/** Video player with bbox overlay drawn on canvas, synced to detection frames */
function AnnotatedVideoPlayer({
  videoSrc, frames, fps, narrations,
}: {
  videoSrc: string;
  frames: GDinoFrame[];
  fps: number;
  narrations?: NarrationFrame[];
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [activeFrameIdx, setActiveFrameIdx] = useState(0);
  const hasFrames = frames.length > 0;

  const frameTimes = frames.map(f => f.frame_index / fps);

  // Colors for different categories
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

    // Match canvas to displayed video size
    const rect = video.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const frame = frames[frameIdx];
    if (!frame?.detections?.length) return;

    // Scale from original video resolution to displayed size
    const scaleX = rect.width / (video.videoWidth || 1);
    const scaleY = rect.height / (video.videoHeight || 1);

    for (const det of frame.detections) {
      if (!det.bbox || det.bbox.length < 4) continue;
      const [x1, y1, x2, y2] = det.bbox;
      const sx1 = x1 * scaleX, sy1 = y1 * scaleY;
      const sw = (x2 - x1) * scaleX, sh = (y2 - y1) * scaleY;
      const color = getColor(det.category);

      // Draw box
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.strokeRect(sx1, sy1, sw, sh);

      // Draw label background
      const label = `${det.category} ${((det.score ?? 0) * 100).toFixed(0)}%`;
      ctx.font = 'bold 11px monospace';
      const textW = ctx.measureText(label).width;
      ctx.fillStyle = color;
      ctx.fillRect(sx1, sy1 - 16, textW + 8, 16);

      // Draw label text
      ctx.fillStyle = '#000';
      ctx.fillText(label, sx1 + 4, sy1 - 4);
    }
  }, [frames]);

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


  // Redraw on resize
  useEffect(() => {
    const handleResize = () => drawBoxes(activeFrameIdx);
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [activeFrameIdx, drawBoxes]);

  const activeFrame = frames[activeFrameIdx];

  // Find narration matching the active frame (by frame_index)
  const activeNarration = narrations?.find(n => {
    if (!activeFrame) return false;
    // Match by frame_index, or by 1-based index (frame_index in narrations is 1-based)
    return n.frame_index === activeFrame.frame_index
      || n.frame_index === activeFrameIdx + 1;
  });

  return (
    <div className="mt-3 pt-3 border-t border-[rgba(255,128,0,0.12)] space-y-3">
      {/* Video with canvas overlay for bounding boxes */}
      <div ref={containerRef} className="relative rounded-lg overflow-hidden border border-[rgba(255,128,0,0.12)]">
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
          style={{ height: 'calc(100% - 36px)' }}  /* avoid covering video controls */
        />
        {/* Frame info badge */}
        {hasFrames && activeFrame && (
          <div className="absolute top-2 left-2 flex items-center gap-2 pointer-events-none">
            <span className="text-[10px] font-mono bg-black/70 text-[#FF8000] px-2 py-0.5 rounded">
              Frame {activeFrame.frame_index} — {frameTimes[activeFrameIdx].toFixed(1)}s
            </span>
            <span className="text-[10px] font-mono bg-black/70 text-green-400 px-2 py-0.5 rounded">
              {activeFrame.detections?.length ?? 0} detections
            </span>
          </div>
        )}
      </div>

      {/* AI Insight — narration for active frame */}
      {activeNarration && (
        <div className="bg-[#0D1117] border border-purple-500/20 rounded-lg p-3">
          <div className="flex items-center gap-2 mb-1.5">
            <Brain className="w-3.5 h-3.5 text-purple-400" />
            <span className="text-[11px] text-purple-400 tracking-wider font-mono">AI INSIGHT</span>
            <span className="text-[10px] text-muted-foreground ml-auto">{activeNarration.tokens} tok · {activeNarration.time_s}s</span>
          </div>
          <p className="text-[12px] text-foreground/90 leading-relaxed">{activeNarration.narration}</p>
        </div>
      )}

      {hasFrames && (
        <>
          {/* Detection list for active frame */}
          {activeFrame?.detections?.length > 0 && (
            <div className="bg-[#0D1117] rounded-lg p-2 space-y-0.5">
              {activeFrame.detections.map((det, i) => (
                <div key={i} className="flex items-center gap-3 text-[11px]">
                  <span className="w-2 h-2 rounded-full shrink-0" style={{ background: getColor(det.category) }} />
                  <span className="text-[#FF8000] font-mono w-10">{((det.score ?? 0) * 100).toFixed(0)}%</span>
                  <span className="text-foreground">{det.category}</span>
                </div>
              ))}
            </div>
          )}

          {/* Frame timeline — click to seek */}
          <div className="flex gap-1 overflow-x-auto pb-1">
            {frames.map((f, i) => {
              const detCount = f.detections?.length ?? 0;
              const hasNarr = narrations?.some(n => n.frame_index === f.frame_index || n.frame_index === i + 1);
              return (
                <button
                  key={f.frame_index}
                  type="button"
                  onClick={() => seekToFrame(i)}
                  className={`shrink-0 rounded-md px-2.5 py-1.5 text-center transition-all border ${
                    i === activeFrameIdx
                      ? 'bg-[#FF8000]/15 border-[#FF8000] text-[#FF8000]'
                      : 'bg-[#0D1117] border-[rgba(255,128,0,0.12)] text-muted-foreground hover:border-[#FF8000]/30'
                  }`}
                >
                  <div className="text-[10px] font-mono">{frameTimes[i].toFixed(1)}s</div>
                  <div className="text-[9px]">
                    {detCount} det{hasNarr ? ' ✦' : ''}
                  </div>
                </button>
              );
            })}
          </div>
        </>
      )}

      {!hasFrames && (
        <div className="text-[12px] text-muted-foreground text-center py-2">
          No detection frames — click <span className="text-[#FF8000]">Analyze Video</span> to run the pipeline
        </div>
      )}
    </div>
  );
}

export function MediaIntelligence() {
  const [gdinoData, setGdinoData] = useState<Record<string, GDinoFrame[]> | null>(null);
  const [fusedData, setFusedData] = useState<Record<string, GDinoFrame[]> | null>(null);
  const [videomaeData, setVideomaeData] = useState<Record<string, VideoModelResult> | null>(null);
  const [timesformerData, setTimesformerData] = useState<Record<string, VideoModelResult> | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [clipQuery, setClipQuery] = useState('');
  const [clipResults, setClipResults] = useState<VisualSearchResult[] | null>(null);
  const [clipSearching, setClipSearching] = useState(false);
  const [allTags, setAllTags] = useState<VisualTagImage[] | null>(null);
  const [selectedTag, setSelectedTag] = useState<string | null>(null);
  const [topTags, setTopTags] = useState<{ label: string; max_score: number }[]>([]);
  const [narrationData, setNarrationData] = useState<Record<string, NarrationFrame[]> | null>(null);

  // Video picker state
  const [videos, setVideos] = useState<MediaVideo[]>([]);
  const [selectedVideo, setSelectedVideo] = useState<string | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [analyzeResult, setAnalyzeResult] = useState<any>(null);

  useEffect(() => {
    Promise.allSettled([
      pipeline.gdino().then(d => setGdinoData(d?.results?.length ? Object.fromEntries(d.results.filter((r: any) => r.filename && r.frames).map((r: any) => [r.filename, r.frames])) : null)),
      pipeline.fused().then(d => setFusedData(d?.results?.length ? Object.fromEntries(d.results.filter((r: any) => r.filename && r.frames).map((r: any) => [r.filename, r.frames])) : null)),
      pipeline.videomae().then(d => {
        if (d?.results) {
          const mapped: Record<string, VideoModelResult> = {};
          for (const r of d.results) {
            mapped[r.filename] = { total_frames: r.total_frames, fps: r.fps, inference_time_s: r.inference_time_s ?? 0, top_predictions: r.top_predictions };
          }
          setVideomaeData(mapped);
        } else {
          setVideomaeData(d);
        }
      }),
      pipeline.timesformer().then(d => {
        if (d?.results) {
          const mapped: Record<string, VideoModelResult> = {};
          for (const r of d.results) {
            mapped[r.filename] = { total_frames: r.total_frames, fps: r.fps, inference_time_s: r.inference_time_s ?? 0, top_predictions: r.top_predictions };
          }
          setTimesformerData(mapped);
        } else {
          setTimesformerData(d);
        }
      }),
      pipeline.videos().then(d => setVideos(d?.videos ?? [])),
      pipeline.minicpm().then(d => setNarrationData(d ?? null)),
    ]).finally(() => setLoading(false));

    fetch('/api/visual-tags')
      .then(r => r.ok ? r.json() : null)
      .then(data => {
        if (data) {
          setAllTags(data.images);
          setTopTags(data.tags || []);
        }
      })
      .catch(() => {});
  }, []);

  const doClipSearch = async (query: string) => {
    if (!query.trim()) { setClipResults(null); return; }
    setClipSearching(true);
    try {
      const res = await fetch(`/api/visual-search?q=${encodeURIComponent(query)}&k=12`);
      if (res.ok) {
        const data = await res.json();
        setClipResults(data.results);
      }
    } catch { /* server not running */ }
    finally { setClipSearching(false); }
  };

  const analyzeVideo = async (filename: string) => {
    setAnalyzing(true);
    setAnalyzeResult(null);
    try {
      const res = await fetch(`/api/omni/vis/analyze-video-by-name`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename, tasks: 'detect,classify' }),
      });
      if (res.ok) {
        const data = await res.json();
        setAnalyzeResult(data);
        // Refresh pipeline data after analysis
        pipeline.gdino().then(d => setGdinoData(d?.results ? Object.fromEntries(d.results.map((r: any) => [r.filename, r.frames])) : d));
        pipeline.videomae().then(d => {
          if (d?.results) {
            const mapped: Record<string, VideoModelResult> = {};
            for (const r of d.results) mapped[r.filename] = { total_frames: r.total_frames, fps: r.fps, inference_time_s: r.inference_time_s ?? 0, top_predictions: r.top_predictions };
            setVideomaeData(mapped);
          }
        });
      }
    } catch (e) {
      setAnalyzeResult({ error: String(e) });
    } finally {
      setAnalyzing(false);
    }
  };

  const filteredByTag = selectedTag && allTags
    ? allTags.filter(img => img.auto_tags.some(t => t.label === selectedTag))
    : null;

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="w-6 h-6 text-[#FF8000] animate-spin" />
        <span className="ml-3 text-muted-foreground text-sm">Loading media pipeline results...</span>
      </div>
    );
  }

  const allFrames = gdinoData ? Object.values(gdinoData).flat().filter(Boolean) : [];
  const totalDetections = allFrames.reduce((s, f) => s + (f.detections?.length ?? 0), 0);
  const totalFramesAnalyzed = allFrames.length;
  const categories = [...new Set(allFrames.flatMap(f => (f.detections ?? []).map(d => d.category)))];

  const pipelineStats = [
    { label: 'Videos Processed', value: gdinoData ? String(Object.keys(gdinoData).length) : '—' },
    { label: 'Frames Analyzed', value: String(totalFramesAnalyzed) },
    { label: 'Detections', value: String(totalDetections) },
    { label: 'VideoMAE Runs', value: videomaeData ? String(Object.keys(videomaeData).length) : '—' },
    { label: 'Videos Available', value: String(videos.length) },
  ];

  return (
    <div className="space-y-4">
      {/* Pipeline Status Bar */}
      <div className="bg-[#1A1F2E] border border-[rgba(255,128,0,0.12)] rounded-xl shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-4">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-[#FF8000]/10 flex items-center justify-center">
              <Brain className="w-4 h-4 text-[#FF8000]" />
            </div>
            <div>
              <h3 className="text-sm text-foreground">Media Analysis Pipeline</h3>
              <div className="text-[12px] text-muted-foreground">GroundingDINO + SAM2 + VideoMAE + TimeSformer + CLIP</div>
            </div>
          </div>
          <span className="flex items-center gap-1.5 text-[12px] text-green-400 bg-green-500/10 px-2 py-1 rounded-full">
            <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
            Results Loaded
          </span>
        </div>
        <div className="grid grid-cols-5 gap-3">
          {pipelineStats.map((stat) => (
            <div key={stat.label} className="bg-[#0D1117] rounded-lg p-2">
              <div className="text-[11px] text-muted-foreground tracking-wider mb-1">{stat.label}</div>
              <div className="text-sm font-mono text-foreground">{stat.value}</div>
            </div>
          ))}
        </div>
      </div>

      {/* CLIP Visual Search */}
      <div className="bg-[#1A1F2E] border border-[rgba(255,128,0,0.12)] rounded-xl shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-4">
        <div className="flex items-center gap-3 mb-3">
          <div className="w-8 h-8 rounded-lg bg-purple-500/10 flex items-center justify-center">
            <ImageIcon className="w-4 h-4 text-purple-400" />
          </div>
          <div className="flex-1">
            <h3 className="text-sm text-foreground">CLIP Visual Search</h3>
            <div className="text-[12px] text-muted-foreground">Search images by text description — CLIP ViT-B/32 (512-dim)</div>
          </div>
        </div>

        {/* Search Bar */}
        <div className="flex items-center gap-2 mb-3">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <input
              type="text"
              placeholder="Search: &quot;pit stop crew&quot;, &quot;rain conditions&quot;, &quot;cockpit view&quot;..."
              value={clipQuery}
              onChange={e => setClipQuery(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter') doClipSearch(clipQuery); }}
              className="w-full bg-[#0D1117] border border-[rgba(255,128,0,0.12)] rounded-lg pl-10 pr-4 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:border-purple-400/40"
            />
          </div>
          <button
            onClick={() => doClipSearch(clipQuery)}
            disabled={clipSearching || !clipQuery.trim()}
            className="px-4 py-2 bg-purple-500/20 text-purple-400 text-sm rounded-lg hover:bg-purple-500/30 disabled:opacity-30 transition-colors"
          >
            {clipSearching ? <Loader2 className="w-3 h-3 animate-spin" /> : 'Search'}
          </button>
        </div>

        {/* Auto-Tag Filter Chips */}
        {topTags.length > 0 && (
          <div className="flex items-center gap-1.5 flex-wrap mb-3">
            <Tag className="w-3 h-3 text-muted-foreground shrink-0" />
            <button
              onClick={() => { setSelectedTag(null); setClipResults(null); }}
              className={`text-[11px] px-2 py-0.5 rounded-full transition-all ${
                !selectedTag ? 'bg-purple-500/20 text-purple-400' : 'text-muted-foreground hover:bg-[#222838]'
              }`}
            >
              All
            </button>
            {topTags.slice(0, 12).map(tag => {
              const shortLabel = tag.label.replace(/formula one |on a formula one car|of a formula one car/g, '').replace(/ car$/, '').trim();
              return (
                <button
                  key={tag.label}
                  onClick={() => { setSelectedTag(selectedTag === tag.label ? null : tag.label); setClipResults(null); }}
                  className={`text-[11px] px-2 py-0.5 rounded-full transition-all ${
                    selectedTag === tag.label ? 'bg-purple-500/20 text-purple-400' : 'text-muted-foreground hover:bg-[#222838]'
                  }`}
                >
                  {shortLabel}
                </button>
              );
            })}
          </div>
        )}

        {/* Search Results */}
        {clipResults && (
          <div>
            <div className="text-[12px] text-muted-foreground mb-2">{clipResults.length} results for &quot;{clipQuery}&quot;</div>
            <div className="grid grid-cols-6 gap-2">
              {clipResults.map((r, i) => (
                <div key={r.path} className="relative rounded-lg overflow-hidden border border-[rgba(255,128,0,0.12)] hover:border-purple-400/30 transition-all group">
                  <img
                    src={`/media/${r.path}`}
                    alt={r.path}
                    className="w-full h-20 object-cover"
                  />
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

        {/* Tag-Filtered Results */}
        {!clipResults && filteredByTag && (
          <div>
            <div className="text-[12px] text-muted-foreground mb-2">{filteredByTag.length} images tagged &quot;{selectedTag}&quot;</div>
            <div className="grid grid-cols-6 gap-2">
              {filteredByTag.map(img => {
                const tagScore = img.auto_tags.find(t => t.label === selectedTag)?.score ?? 0;
                return (
                  <div key={img.path} className="relative rounded-lg overflow-hidden border border-[rgba(255,128,0,0.12)] hover:border-purple-400/30 transition-all group">
                    <img
                      src={`/media/${img.path}`}
                      alt={img.path}
                      className="w-full h-20 object-cover"
                    />
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

      <div className="grid grid-cols-12 gap-4">
        {/* Left — Results Feed */}
        <div className="col-span-7 space-y-4">
          {/* Detection Gallery */}
          {(fusedData || gdinoData) && (
            <div className="bg-[#1A1F2E] border border-[rgba(255,128,0,0.12)] rounded-xl shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-4">
              <h3 className="text-sm text-muted-foreground tracking-widest mb-3 flex items-center gap-2">
                <Scan className="w-3 h-3" />
                DETECTION GALLERY — {totalDetections} objects across {totalFramesAnalyzed} frames
              </h3>
              <div className="text-[12px] text-muted-foreground mb-3 flex items-center flex-wrap gap-1">
                <span>Categories:</span>
                {categories.map(c => (
                  <span key={c} className="inline-block px-1.5 py-0.5 rounded bg-[#FF8000]/10 text-[#FF8000]">{c}</span>
                ))}
              </div>
              <div className="grid grid-cols-4 gap-2">
                {Object.entries(fusedData ?? gdinoData!).flatMap(([video, frames]) =>
                  (frames ?? []).filter(f => f.output_image).map((frame) => ({ video, frame }))
                ).slice(0, 4).map(({ video, frame }) => (
                    <button
                      key={`${video}-${frame.frame_index}`}
                      type="button"
                      onClick={() => setSelectedImage(selectedImage === frame.output_image ? null : frame.output_image)}
                      className={`relative rounded-lg overflow-hidden border transition-all ${
                        selectedImage === frame.output_image ? 'border-[#FF8000]' : 'border-[rgba(255,128,0,0.12)] hover:border-[#FF8000]/30'
                      }`}
                    >
                      <img
                        src={`/media/gdino_results/${frame.output_image}`}
                        alt={frame.output_image}
                        className="w-full h-20 object-cover"
                        onError={(e) => { const img = e.target as HTMLImageElement; if (!img.dataset.fallback) { img.dataset.fallback = '1'; img.src = `/media/fused_results/${frame.output_image}`; } }}
                      />
                      <div className="absolute bottom-0 left-0 right-0 bg-black/70 px-1.5 py-0.5 flex items-center justify-between">
                        <span className="text-[10px] text-foreground font-mono">{frame.detections?.length ?? 0} det</span>
                        <span className="text-[9px] text-muted-foreground">f{frame.frame_index}</span>
                      </div>
                    </button>
                  ))}
              </div>
              {selectedImage && (
                <div className="mt-4 space-y-2">
                  <div className="rounded-lg overflow-hidden border border-[#FF8000]/20">
                    <img
                      src={`/media/gdino_results/${selectedImage}`}
                      alt="Detection detail"
                      className="w-full"
                      onError={(e) => { const img = e.target as HTMLImageElement; if (!img.dataset.fallback) { img.dataset.fallback = '1'; img.src = `/media/fused_results/${selectedImage}`; } }}
                    />
                  </div>
                  {(() => {
                    const frame = Object.values(fusedData ?? gdinoData!).flat().find(f => f.output_image === selectedImage);
                    if (!frame) return null;
                    return (
                      <div className="bg-[#0D1117] rounded-lg p-3 space-y-1">
                        <div className="text-[11px] text-muted-foreground tracking-wider mb-1">DETECTIONS — Frame {frame.frame_index}</div>
                        {(frame.detections ?? []).map((det, i) => (
                          <div key={i} className="flex items-center gap-3 text-[11px]">
                            <span className="text-[#FF8000] font-mono w-8">{(det.score * 100).toFixed(0)}%</span>
                            <span className="text-foreground">{det.category}</span>
                            <span className="text-muted-foreground font-mono text-[11px]">
                              [{det.bbox.join(', ')}]
                            </span>
                          </div>
                        ))}
                      </div>
                    );
                  })()}
                </div>
              )}
            </div>
          )}

          {/* Video Classification Results */}
          {(videomaeData || timesformerData) && (
            <div className="bg-[#1A1F2E] border border-[rgba(255,128,0,0.12)] rounded-xl shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-4">
              <h3 className="text-sm text-muted-foreground tracking-widest mb-3 flex items-center gap-2">
                <Video className="w-3 h-3" />
                VIDEO CLASSIFICATION
              </h3>
              <div className="space-y-3">
                {Object.keys(videomaeData ?? timesformerData ?? {}).slice(0, 4).map(video => {
                  const vmae = videomaeData?.[video];
                  const tsf = timesformerData?.[video];
                  return (
                    <div key={video} className="bg-[#0D1117] rounded-lg p-3">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm text-[#FF8000] font-mono">{video}</span>
                        <span className="text-[11px] text-muted-foreground">{vmae?.total_frames ?? tsf?.total_frames} frames @ {vmae?.fps ?? tsf?.fps}fps</span>
                      </div>
                      <div className="grid grid-cols-2 gap-3">
                        {vmae && (
                          <div>
                            <div className="text-[11px] text-purple-400 tracking-wider mb-1">VideoMAE</div>
                            {vmae.top_predictions.slice(0, 3).map((p, i) => (
                              <div key={i} className="flex items-center gap-2 text-[12px]">
                                <div className="w-16 h-1 bg-[#222838] rounded-full overflow-hidden">
                                  <div className="h-full bg-purple-400 rounded-full" style={{ width: `${p.score * 100}%` }} />
                                </div>
                                <span className="font-mono text-foreground">{(p.score * 100).toFixed(1)}%</span>
                                <span className="text-muted-foreground truncate">{p.label}</span>
                              </div>
                            ))}
                          </div>
                        )}
                        {tsf && (
                          <div>
                            <div className="text-[11px] text-cyan-400 tracking-wider mb-1">TimeSformer</div>
                            {tsf.top_predictions.slice(0, 3).map((p, i) => (
                              <div key={i} className="flex items-center gap-2 text-[12px]">
                                <div className="w-16 h-1 bg-[#222838] rounded-full overflow-hidden">
                                  <div className="h-full bg-cyan-400 rounded-full" style={{ width: `${p.score * 100}%` }} />
                                </div>
                                <span className="font-mono text-foreground">{(p.score * 100).toFixed(1)}%</span>
                                <span className="text-muted-foreground truncate">{p.label}</span>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>

        {/* Right Column — Video Picker */}
        <div className="col-span-5 space-y-4">
          <div className="bg-[#1A1F2E] border border-[rgba(255,128,0,0.12)] rounded-xl shadow-[0_1px_3px_rgba(0,0,0,0.3)] p-4">
            <h3 className="text-sm text-muted-foreground tracking-widest mb-3 flex items-center gap-2">
              <Film className="w-3 h-3" />
              VIDEO LIBRARY
            </h3>
            <div className="space-y-2">
              {videos.length === 0 && (
                <div className="text-[12px] text-muted-foreground py-4 text-center">No videos available</div>
              )}
              {videos.slice(0, 4).map(v => {
                const isSelected = selectedVideo === v.filename;
                const hasResults = gdinoData?.[v.filename] || videomaeData?.[v.filename];
                return (
                  <button
                    key={v.filename}
                    type="button"
                    onClick={() => setSelectedVideo(isSelected ? null : v.filename)}
                    className={`w-full text-left rounded-lg p-3 transition-all border ${
                      isSelected
                        ? 'bg-[#FF8000]/10 border-[#FF8000]/30'
                        : 'bg-[#0D1117] border-[rgba(255,128,0,0.12)] hover:border-[#FF8000]/20'
                    }`}
                  >
                    <div className="flex items-center gap-3">
                      <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${
                        hasResults ? 'bg-green-500/10' : 'bg-[#FF8000]/10'
                      }`}>
                        <Play className={`w-3.5 h-3.5 ${hasResults ? 'text-green-400' : 'text-[#FF8000]'}`} />
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="text-[12px] text-foreground truncate">{v.filename}</div>
                        <div className="text-[11px] text-muted-foreground flex items-center gap-2">
                          <span>{v.size_mb} MB</span>
                          {hasResults && (
                            <span className="text-green-400">analyzed</span>
                          )}
                        </div>
                      </div>
                    </div>
                  </button>
                );
              })}
            </div>

            {/* Annotated Video Player */}
            {selectedVideo && (() => {
              const resultData = fusedData ?? gdinoData;
              const frames = resultData?.[selectedVideo] ?? [];
              const fps = videomaeData?.[selectedVideo]?.fps ?? timesformerData?.[selectedVideo]?.fps ?? 30;
              return (
                <AnnotatedVideoPlayer
                  key={selectedVideo}
                  videoSrc={`/media/${encodeURIComponent(selectedVideo)}`}
                  frames={frames}
                  fps={fps}
                  narrations={narrationData?.[selectedVideo] ?? undefined}
                />
              );
            })()}

            {/* Analyze Button */}
            {selectedVideo && (
              <div className="mt-3 pt-3 border-t border-[rgba(255,128,0,0.12)]">
                <div className="text-[12px] text-muted-foreground mb-2">
                  Selected: <span className="text-[#FF8000] font-mono">{selectedVideo}</span>
                </div>
                <button
                  onClick={() => analyzeVideo(selectedVideo)}
                  disabled={analyzing}
                  className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-[#FF8000]/20 text-[#FF8000] text-sm rounded-lg hover:bg-[#FF8000]/30 disabled:opacity-40 transition-colors"
                >
                  {analyzing ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Analyzing on GPU...
                    </>
                  ) : (
                    <>
                      <Scan className="w-4 h-4" />
                      Analyze Video
                    </>
                  )}
                </button>
              </div>
            )}

            {/* Analysis Result */}
            {analyzeResult && (
              <div className="mt-3 pt-3 border-t border-[rgba(255,128,0,0.12)]">
                {analyzeResult.error ? (
                  <div className="text-[12px] text-red-400">{analyzeResult.error}</div>
                ) : (
                  <div className="space-y-2">
                    <div className="text-[11px] text-muted-foreground tracking-wider">ANALYSIS COMPLETE</div>
                    <div className="grid grid-cols-2 gap-2 text-[12px]">
                      <div className="bg-[#0D1117] rounded-lg p-2">
                        <div className="text-muted-foreground">Frames</div>
                        <div className="text-foreground font-mono">{analyzeResult.sampled_frames ?? '—'} / {analyzeResult.total_frames ?? '—'}</div>
                      </div>
                      <div className="bg-[#0D1117] rounded-lg p-2">
                        <div className="text-muted-foreground">FPS</div>
                        <div className="text-foreground font-mono">{analyzeResult.fps ?? '—'}</div>
                      </div>
                    </div>
                    {analyzeResult.classification && (
                      <div>
                        <div className="text-[11px] text-purple-400 tracking-wider mb-1">Classifications</div>
                        {analyzeResult.classification.slice(0, 3).map((c: any, i: number) => (
                          <div key={i} className="flex items-center gap-2 text-[12px]">
                            <div className="w-16 h-1 bg-[#222838] rounded-full overflow-hidden">
                              <div className="h-full bg-purple-400 rounded-full" style={{ width: `${(c.score ?? 0) * 100}%` }} />
                            </div>
                            <span className="font-mono text-foreground">{((c.score ?? 0) * 100).toFixed(1)}%</span>
                            <span className="text-muted-foreground truncate">{c.label}</span>
                          </div>
                        ))}
                      </div>
                    )}
                    {analyzeResult.gdino && (
                      <div className="text-[12px] text-muted-foreground">
                        {analyzeResult.gdino.length} frames with detections
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
