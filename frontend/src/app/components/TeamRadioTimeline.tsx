import { useState, useEffect, useRef, useCallback } from 'react';
import {
  Radio, Play, Pause, Filter, ChevronDown,
  AlertTriangle, Gauge, Wrench, Flag, Cloud, Heart, Users,
  Loader2,
} from 'lucide-react';

// ── Types ──────────────────────────────────────────────────────────────

interface RadioMessage {
  file_key: string;
  transcript: string;
  driver_name: string;
  driver_number: number;
  time: string;
  date: string;
  signal_tags: string[];
  sentiment_score: number;
  urgency_level: number;
  speaker: string;
  mentioned_components: string[];
  mentioned_corners: string[];
  tyre_compound_mentioned: string | null;
  confidence: number;
  sequence: number;
}

interface SessionOption {
  year: number;
  meeting: string;
  session: string;
  message_count: number;
  drivers: string[];
}

interface TimelineData {
  year: number;
  meeting: string;
  session: string;
  messages: RadioMessage[];
  count: number;
}

interface DriverProfile {
  driver_name: string;
  season: number;
  total_messages: number;
  messages_per_session_avg: number;
  sentiment_distribution: { negative: number; neutral: number; positive: number };
  top_signal_types: string[];
  signal_counts: Record<string, number>;
  complaint_frequency: number;
  frustration_rate_by_session_type: Record<string, number>;
}

interface RadioStats {
  total_tagged: number;
  signal_distribution: Record<string, number>;
  urgency_distribution: Record<number, number>;
}

// ── Signal styling ─────────────────────────────────────────────────────

const SIGNAL_CONFIG: Record<string, { label: string; color: string; bg: string; icon: React.ElementType }> = {
  reliability_flag:  { label: 'Reliability',   color: 'text-[#FB2C36]',  bg: 'bg-[#FB2C36]/12', icon: AlertTriangle },
  tyre_condition:    { label: 'Tyres',         color: 'text-[#F59E0B]',  bg: 'bg-[#F59E0B]/12', icon: Gauge },
  strategy_call:     { label: 'Strategy',      color: 'text-[#3B82F6]',  bg: 'bg-[#3B82F6]/12', icon: Flag },
  car_balance:       { label: 'Balance',       color: 'text-primary',     bg: 'bg-primary/12',   icon: Wrench },
  track_condition:   { label: 'Track',         color: 'text-[#8B5CF6]',  bg: 'bg-[#8B5CF6]/12', icon: Cloud },
  emotional_state:   { label: 'Emotion',       color: 'text-[#05DF72]',  bg: 'bg-[#05DF72]/12', icon: Heart },
  competitor_intel:  { label: 'Competitor',    color: 'text-[#EC4899]',  bg: 'bg-[#EC4899]/12', icon: Users },
};

const URGENCY_STYLES: Record<number, string> = {
  0: 'border-l-muted-foreground/20',
  1: 'border-l-[#3B82F6]/40',
  2: 'border-l-[#F59E0B]/60',
  3: 'border-l-[#FB2C36]/80',
};

function sentimentBar(score: number) {
  const pct = ((score + 1) / 2) * 100;
  const color = score < -0.3 ? '#FB2C36' : score > 0.3 ? '#05DF72' : '#64748B';
  return (
    <div className="w-16 h-1.5 rounded-full bg-secondary overflow-hidden" title={`Sentiment: ${score.toFixed(1)}`}>
      <div className="h-full rounded-full transition-all" style={{ width: `${pct}%`, background: color }} />
    </div>
  );
}

function formatMeetingName(name: string) {
  return name.replace(/_/g, ' ');
}

// ── Component ──────────────────────────────────────────────────────────

export function TeamRadioTimeline() {
  const [sessions, setSessions] = useState<SessionOption[]>([]);
  const [timeline, setTimeline] = useState<TimelineData | null>(null);
  const [stats, setStats] = useState<RadioStats | null>(null);
  const [profiles, setProfiles] = useState<DriverProfile[]>([]);
  const [loading, setLoading] = useState(false);
  const [statsLoading, setStatsLoading] = useState(true);

  // Filters
  const [selectedYear, setSelectedYear] = useState<number>(2024);
  const [selectedMeeting, setSelectedMeeting] = useState<string>('');
  const [selectedSession, setSelectedSession] = useState<string>('');
  const [filterDriver, setFilterDriver] = useState<string>('');
  const [filterSignal, setFilterSignal] = useState<string>('');
  const [filterMinUrgency, setFilterMinUrgency] = useState(0);
  const [showFilters, setShowFilters] = useState(false);

  // Audio
  const [playingKey, setPlayingKey] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  // Tab
  const [tab, setTab] = useState<'timeline' | 'profiles' | 'stats'>('timeline');

  // ── Fetch sessions ──
  useEffect(() => {
    fetch(`/api/radio/sessions?year=${selectedYear}`)
      .then(r => r.json())
      .then(data => {
        setSessions(data);
        if (data.length > 0 && !selectedMeeting) {
          // Default to first race session
          const race = data.find((s: SessionOption) => s.session === 'Race') || data[0];
          setSelectedMeeting(race.meeting);
          setSelectedSession(race.session);
        }
      })
      .catch(() => {});
  }, [selectedYear]);

  // ── Fetch timeline ──
  const loadTimeline = useCallback(() => {
    if (!selectedMeeting || !selectedSession) return;
    setLoading(true);
    const params = new URLSearchParams({
      year: String(selectedYear),
      meeting: selectedMeeting,
      session: selectedSession,
    });
    if (filterDriver) params.set('driver', filterDriver);
    if (filterSignal) params.set('signal_type', filterSignal);
    if (filterMinUrgency > 0) params.set('min_urgency', String(filterMinUrgency));

    fetch(`/api/radio/timeline?${params}`)
      .then(r => r.json())
      .then(data => setTimeline(data))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [selectedYear, selectedMeeting, selectedSession, filterDriver, filterSignal, filterMinUrgency]);

  useEffect(() => { loadTimeline(); }, [loadTimeline]);

  // ── Derive unique drivers from session data ──
  const allDrivers = [...new Set(sessions.flatMap(s => s.drivers))].sort();

  // ── Fetch stats + profiles for all drivers in this year ──
  useEffect(() => {
    if (allDrivers.length === 0) return;
    setStatsLoading(true);
    const profileFetches = allDrivers.map(d =>
      fetch(`/api/radio/driver-profile?driver=${encodeURIComponent(d)}&season=${selectedYear}`)
        .then(r => r.ok ? r.json() : [])
    );
    Promise.all([
      fetch(`/api/radio/stats?year=${selectedYear}`).then(r => r.json()),
      ...profileFetches,
    ]).then(([statsData, ...profileArrays]) => {
      setStats(statsData);
      setProfiles((profileArrays as DriverProfile[][]).flat());
    }).catch(() => {})
      .finally(() => setStatsLoading(false));
  }, [selectedYear, allDrivers.join(',')]);

  // ── Audio playback ──
  const toggleAudio = (fileKey: string) => {
    if (playingKey === fileKey) {
      audioRef.current?.pause();
      setPlayingKey(null);
      return;
    }
    if (audioRef.current) {
      audioRef.current.pause();
    }
    const audio = new Audio(`/api/radio/audio/${fileKey}`);
    audio.onended = () => setPlayingKey(null);
    audio.play();
    audioRef.current = audio;
    setPlayingKey(fileKey);
  };

  // ── Unique meetings for dropdown ──
  const meetings = [...new Set(sessions.map(s => s.meeting))];
  const sessionTypes = sessions.filter(s => s.meeting === selectedMeeting).map(s => s.session);

  // ── Render ──
  return (
    <div className="space-y-4">
      {/* Header bar */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Radio className="w-5 h-5 text-primary" />
          <h2 className="text-lg font-semibold text-foreground">Team Radio Intelligence</h2>
          {stats && (
            <span className="text-xs text-muted-foreground">
              {stats.total_tagged.toLocaleString()} tagged messages
            </span>
          )}
        </div>

        {/* Tabs */}
        <div className="flex gap-1 bg-secondary rounded-lg p-0.5">
          {(['timeline', 'profiles', 'stats'] as const).map(t => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`px-3 py-1 text-xs rounded-md transition-colors ${
                tab === t ? 'bg-background text-foreground shadow-sm' : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              {t === 'timeline' ? 'Timeline' : t === 'profiles' ? 'Driver Profiles' : 'Signal Stats'}
            </button>
          ))}
        </div>
      </div>

      {/* Session selector */}
      <div className="flex flex-wrap gap-2 items-center">
        <select
          value={selectedYear}
          onChange={e => { setSelectedYear(Number(e.target.value)); setSelectedMeeting(''); }}
          className="bg-secondary text-foreground text-xs rounded-md px-2 py-1.5 border border-border"
        >
          {[2024, 2023].map(y => <option key={y} value={y}>{y}</option>)}
        </select>

        <select
          value={selectedMeeting}
          onChange={e => {
            setSelectedMeeting(e.target.value);
            const first = sessions.find(s => s.meeting === e.target.value);
            if (first) setSelectedSession(first.session);
          }}
          className="bg-secondary text-foreground text-xs rounded-md px-2 py-1.5 border border-border max-w-[200px]"
        >
          <option value="">Select GP</option>
          {meetings.map(m => <option key={m} value={m}>{formatMeetingName(m)}</option>)}
        </select>

        <select
          value={selectedSession}
          onChange={e => setSelectedSession(e.target.value)}
          className="bg-secondary text-foreground text-xs rounded-md px-2 py-1.5 border border-border"
        >
          {sessionTypes.map(s => <option key={s} value={s}>{s.replace(/_/g, ' ')}</option>)}
        </select>

        <button
          onClick={() => setShowFilters(!showFilters)}
          className={`flex items-center gap-1 px-2 py-1.5 text-xs rounded-md border transition-colors ${
            showFilters ? 'border-primary text-primary bg-primary/5' : 'border-border text-muted-foreground hover:text-foreground'
          }`}
        >
          <Filter className="w-3 h-3" />
          Filters
          <ChevronDown className={`w-3 h-3 transition-transform ${showFilters ? 'rotate-180' : ''}`} />
        </button>
      </div>

      {/* Filters panel */}
      {showFilters && (
        <div className="flex flex-wrap gap-3 p-3 bg-secondary/50 rounded-lg border border-border">
          <div className="space-y-1">
            <label className="text-[10px] text-muted-foreground uppercase tracking-wider">Driver</label>
            <select
              value={filterDriver}
              onChange={e => setFilterDriver(e.target.value)}
              className="bg-background text-foreground text-xs rounded px-2 py-1 border border-border"
            >
              <option value="">All</option>
              {allDrivers.map(d => <option key={d} value={d}>{d}</option>)}
            </select>
          </div>
          <div className="space-y-1">
            <label className="text-[10px] text-muted-foreground uppercase tracking-wider">Signal</label>
            <select
              value={filterSignal}
              onChange={e => setFilterSignal(e.target.value)}
              className="bg-background text-foreground text-xs rounded px-2 py-1 border border-border"
            >
              <option value="">All</option>
              {Object.entries(SIGNAL_CONFIG).map(([k, v]) => (
                <option key={k} value={k}>{v.label}</option>
              ))}
            </select>
          </div>
          <div className="space-y-1">
            <label className="text-[10px] text-muted-foreground uppercase tracking-wider">Min Urgency</label>
            <select
              value={filterMinUrgency}
              onChange={e => setFilterMinUrgency(Number(e.target.value))}
              className="bg-background text-foreground text-xs rounded px-2 py-1 border border-border"
            >
              <option value={0}>Any</option>
              <option value={1}>1+ Notable</option>
              <option value={2}>2+ Important</option>
              <option value={3}>3 Critical</option>
            </select>
          </div>
        </div>
      )}

      {/* Tab content */}
      {tab === 'timeline' && (
        <TimelineView
          timeline={timeline}
          loading={loading}
          playingKey={playingKey}
          onToggleAudio={toggleAudio}
        />
      )}

      {tab === 'profiles' && (
        <ProfilesView profiles={profiles} loading={statsLoading} year={selectedYear} />
      )}

      {tab === 'stats' && (
        <StatsView stats={stats} loading={statsLoading} />
      )}
    </div>
  );
}

// ── Timeline View ──────────────────────────────────────────────────────

function TimelineView({
  timeline, loading, playingKey, onToggleAudio,
}: {
  timeline: TimelineData | null;
  loading: boolean;
  playingKey: string | null;
  onToggleAudio: (key: string) => void;
}) {
  if (loading) {
    return (
      <div className="flex items-center justify-center py-16">
        <Loader2 className="w-5 h-5 text-primary animate-spin" />
        <span className="ml-2 text-sm text-muted-foreground">Loading radio timeline...</span>
      </div>
    );
  }

  if (!timeline || timeline.messages.length === 0) {
    return (
      <div className="text-center py-16 text-muted-foreground text-sm">
        No radio messages found for this session.
      </div>
    );
  }

  return (
    <div className="space-y-1">
      <div className="text-xs text-muted-foreground mb-2">
        {formatMeetingName(timeline.meeting)} — {timeline.session.replace(/_/g, ' ')} — {timeline.count} messages
      </div>

      {timeline.messages.map((msg, i) => {
        const isPlaying = playingKey === msg.file_key;
        const urgencyBorder = URGENCY_STYLES[msg.urgency_level] || URGENCY_STYLES[0];

        return (
          <div
            key={msg.file_key || i}
            className={`group flex gap-3 p-2.5 rounded-lg border-l-2 ${urgencyBorder} bg-secondary/30 hover:bg-secondary/60 transition-colors`}
          >
            {/* Play button */}
            <button
              onClick={() => onToggleAudio(msg.file_key)}
              className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center transition-colors ${
                isPlaying
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-secondary hover:bg-primary/20 text-muted-foreground hover:text-primary'
              }`}
            >
              {isPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5 ml-0.5" />}
            </button>

            {/* Content */}
            <div className="flex-1 min-w-0 space-y-1">
              {/* Header row */}
              <div className="flex items-center gap-2 flex-wrap">
                <span className={`text-xs font-semibold ${
                  msg.driver_name === 'Norris' ? 'text-primary' : 'text-[#3B82F6]'
                }`}>
                  {msg.driver_name}
                </span>
                <span className="text-[10px] text-muted-foreground font-mono">{msg.time}</span>
                <span className="text-[10px] text-muted-foreground/60 capitalize">{msg.speaker}</span>

                {/* Signal badges */}
                {msg.signal_tags.map(tag => {
                  const cfg = SIGNAL_CONFIG[tag];
                  if (!cfg) return null;
                  const Icon = cfg.icon;
                  return (
                    <span key={tag} className={`inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded text-[10px] font-medium ${cfg.bg} ${cfg.color}`}>
                      <Icon className="w-2.5 h-2.5" />
                      {cfg.label}
                    </span>
                  );
                })}

                {msg.urgency_level >= 2 && (
                  <span className={`text-[10px] font-bold ${msg.urgency_level >= 3 ? 'text-[#FB2C36]' : 'text-[#F59E0B]'}`}>
                    URG {msg.urgency_level}
                  </span>
                )}
              </div>

              {/* Transcript */}
              <p className="text-sm text-foreground/90 leading-relaxed">
                &ldquo;{msg.transcript}&rdquo;
              </p>

              {/* Footer details */}
              <div className="flex items-center gap-3 text-[10px] text-muted-foreground/60">
                {sentimentBar(msg.sentiment_score)}
                {msg.mentioned_components.length > 0 && (
                  <span>{msg.mentioned_components.join(', ')}</span>
                )}
                {msg.mentioned_corners.length > 0 && (
                  <span>{msg.mentioned_corners.join(', ')}</span>
                )}
                {msg.tyre_compound_mentioned && (
                  <span className="capitalize">{msg.tyre_compound_mentioned}</span>
                )}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ── Profiles View ──────────────────────────────────────────────────────

function ProfilesView({ profiles, loading, year }: { profiles: DriverProfile[]; loading: boolean; year: number }) {
  if (loading) {
    return (
      <div className="flex items-center justify-center py-16">
        <Loader2 className="w-5 h-5 text-primary animate-spin" />
      </div>
    );
  }

  if (profiles.length === 0) {
    return <div className="text-center py-16 text-muted-foreground text-sm">No driver profiles found.</div>;
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {profiles.filter(p => p.season === year).map(profile => (
        <div key={`${profile.driver_name}-${profile.season}`} className="bg-secondary/30 rounded-lg p-4 space-y-3 border border-border/50">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-foreground">{profile.driver_name}</h3>
            <span className="text-xs text-muted-foreground">{profile.season}</span>
          </div>

          {/* Stats row */}
          <div className="grid grid-cols-3 gap-2">
            <div className="text-center">
              <div className="text-lg font-bold text-foreground">{profile.total_messages}</div>
              <div className="text-[10px] text-muted-foreground">Messages</div>
            </div>
            <div className="text-center">
              <div className="text-lg font-bold text-foreground">{profile.messages_per_session_avg}</div>
              <div className="text-[10px] text-muted-foreground">Per Session</div>
            </div>
            <div className="text-center">
              <div className="text-lg font-bold text-foreground">{(profile.complaint_frequency * 100).toFixed(0)}%</div>
              <div className="text-[10px] text-muted-foreground">Complaint Rate</div>
            </div>
          </div>

          {/* Sentiment bar */}
          <div>
            <div className="text-[10px] text-muted-foreground mb-1">Sentiment Distribution</div>
            <div className="flex h-2 rounded-full overflow-hidden">
              <div className="bg-[#FB2C36]/70" style={{ width: `${profile.sentiment_distribution.negative * 100}%` }} />
              <div className="bg-muted-foreground/30" style={{ width: `${profile.sentiment_distribution.neutral * 100}%` }} />
              <div className="bg-[#05DF72]/70" style={{ width: `${profile.sentiment_distribution.positive * 100}%` }} />
            </div>
            <div className="flex justify-between text-[9px] text-muted-foreground mt-0.5">
              <span>{(profile.sentiment_distribution.negative * 100).toFixed(0)}% neg</span>
              <span>{(profile.sentiment_distribution.neutral * 100).toFixed(0)}% neutral</span>
              <span>{(profile.sentiment_distribution.positive * 100).toFixed(0)}% pos</span>
            </div>
          </div>

          {/* Top signals */}
          <div>
            <div className="text-[10px] text-muted-foreground mb-1">Top Signal Types</div>
            <div className="flex flex-wrap gap-1">
              {profile.top_signal_types.map(tag => {
                const cfg = SIGNAL_CONFIG[tag];
                if (!cfg) return null;
                const count = profile.signal_counts[tag] || 0;
                return (
                  <span key={tag} className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] ${cfg.bg} ${cfg.color}`}>
                    {cfg.label} ({count})
                  </span>
                );
              })}
            </div>
          </div>

          {/* Frustration by session */}
          <div>
            <div className="text-[10px] text-muted-foreground mb-1">Frustration Rate by Session</div>
            <div className="space-y-0.5">
              {Object.entries(profile.frustration_rate_by_session_type)
                .sort(([,a], [,b]) => b - a)
                .slice(0, 5)
                .map(([session, rate]) => (
                  <div key={session} className="flex items-center gap-2">
                    <span className="text-[10px] text-muted-foreground w-24 truncate">{session.replace(/_/g, ' ')}</span>
                    <div className="flex-1 h-1.5 bg-secondary rounded-full overflow-hidden">
                      <div
                        className="h-full bg-[#FB2C36]/60 rounded-full"
                        style={{ width: `${rate * 100}%` }}
                      />
                    </div>
                    <span className="text-[10px] text-muted-foreground w-8 text-right">{(rate * 100).toFixed(0)}%</span>
                  </div>
                ))}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

// ── Stats View ─────────────────────────────────────────────────────────

function StatsView({ stats, loading }: { stats: RadioStats | null; loading: boolean }) {
  if (loading || !stats) {
    return (
      <div className="flex items-center justify-center py-16">
        <Loader2 className="w-5 h-5 text-primary animate-spin" />
      </div>
    );
  }

  const maxSignal = Math.max(...Object.values(stats.signal_distribution));

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {/* Signal distribution */}
      <div className="bg-secondary/30 rounded-lg p-4 border border-border/50">
        <h3 className="text-sm font-semibold text-foreground mb-3">Signal Distribution</h3>
        <div className="space-y-2">
          {Object.entries(stats.signal_distribution)
            .sort(([,a], [,b]) => b - a)
            .map(([signal, count]) => {
              const cfg = SIGNAL_CONFIG[signal];
              if (!cfg) return null;
              const Icon = cfg.icon;
              return (
                <div key={signal} className="flex items-center gap-2">
                  <Icon className={`w-3.5 h-3.5 ${cfg.color} flex-shrink-0`} />
                  <span className="text-xs text-foreground/80 w-20">{cfg.label}</span>
                  <div className="flex-1 h-2 bg-secondary rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full ${cfg.bg.replace('/12', '/50')}`}
                      style={{ width: `${(count / maxSignal) * 100}%` }}
                    />
                  </div>
                  <span className="text-xs text-muted-foreground w-10 text-right font-mono">{count}</span>
                </div>
              );
            })}
        </div>
      </div>

      {/* Urgency distribution */}
      <div className="bg-secondary/30 rounded-lg p-4 border border-border/50">
        <h3 className="text-sm font-semibold text-foreground mb-3">Urgency Distribution</h3>
        <div className="space-y-2">
          {Object.entries(stats.urgency_distribution)
            .sort(([a], [b]) => Number(a) - Number(b))
            .map(([level, count]) => {
              const labels = ['Routine', 'Notable', 'Important', 'Critical'];
              const colors = ['bg-muted-foreground/30', 'bg-[#3B82F6]/50', 'bg-[#F59E0B]/50', 'bg-[#FB2C36]/50'];
              const l = Number(level);
              return (
                <div key={level} className="flex items-center gap-2">
                  <span className="text-xs text-foreground/80 w-20">{labels[l] || `Level ${l}`}</span>
                  <div className="flex-1 h-2 bg-secondary rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full ${colors[l] || colors[0]}`}
                      style={{ width: `${(count / stats.total_tagged) * 100}%` }}
                    />
                  </div>
                  <span className="text-xs text-muted-foreground w-10 text-right font-mono">{count}</span>
                </div>
              );
            })}
        </div>
        <div className="mt-3 text-xs text-muted-foreground">
          Total: {stats.total_tagged.toLocaleString()} messages
        </div>
      </div>
    </div>
  );
}
