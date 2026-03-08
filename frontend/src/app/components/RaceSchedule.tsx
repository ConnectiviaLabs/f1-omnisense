import { useState, useEffect, useMemo } from 'react';
import { motion } from 'motion/react';
import { Calendar, MapPin, Clock, Zap, CheckCircle2, ChevronRight, Loader2, Flag } from 'lucide-react';

interface SessionDoc {
  session_key: number;
  meeting_key: number;
  session_name: string;
  session_type: string;
  circuit_short_name: string;
  country_name: string;
  country_code: string;
  location: string;
  date_start: string;
  date_end: string;
  year: number;
}

interface RaceWeekend {
  meeting_key: number;
  circuit: string;
  country: string;
  countryCode: string;
  location: string;
  dateStart: Date;
  raceDate: Date | null;
  sessions: string[];
  isSprint: boolean;
  round: number;
  gpName: string;
}

const COUNTRY_FLAGS: Record<string, string> = {
  AUS: '\u{1F1E6}\u{1F1FA}', CHN: '\u{1F1E8}\u{1F1F3}', JPN: '\u{1F1EF}\u{1F1F5}',
  BHR: '\u{1F1E7}\u{1F1ED}', SAU: '\u{1F1F8}\u{1F1E6}', USA: '\u{1F1FA}\u{1F1F8}',
  CAN: '\u{1F1E8}\u{1F1E6}', MCO: '\u{1F1F2}\u{1F1E8}', ESP: '\u{1F1EA}\u{1F1F8}',
  AUT: '\u{1F1E6}\u{1F1F9}', GBR: '\u{1F1EC}\u{1F1E7}', BEL: '\u{1F1E7}\u{1F1EA}',
  HUN: '\u{1F1ED}\u{1F1FA}', NLD: '\u{1F1F3}\u{1F1F1}', ITA: '\u{1F1EE}\u{1F1F9}',
  AZE: '\u{1F1E6}\u{1F1FF}', SGP: '\u{1F1F8}\u{1F1EC}', MEX: '\u{1F1F2}\u{1F1FD}',
  BRA: '\u{1F1E7}\u{1F1F7}', QAT: '\u{1F1F6}\u{1F1E6}', ARE: '\u{1F1E6}\u{1F1EA}',
};

const CIRCUIT_GP_NAMES: Record<string, string> = {
  Melbourne: 'Australian Grand Prix',
  Shanghai: 'Chinese Grand Prix',
  Suzuka: 'Japanese Grand Prix',
  Sakhir: 'Bahrain Grand Prix',
  Jeddah: 'Saudi Arabian Grand Prix',
  Miami: 'Miami Grand Prix',
  Montreal: 'Canadian Grand Prix',
  'Monte Carlo': 'Monaco Grand Prix',
  Catalunya: 'Barcelona Grand Prix',
  Spielberg: 'Austrian Grand Prix',
  Silverstone: 'British Grand Prix',
  'Spa-Francorchamps': 'Belgian Grand Prix',
  Hungaroring: 'Hungarian Grand Prix',
  Zandvoort: 'Dutch Grand Prix',
  Monza: 'Italian Grand Prix',
  Madring: 'Spanish Grand Prix',
  Baku: 'Azerbaijan Grand Prix',
  Singapore: 'Singapore Grand Prix',
  Austin: 'United States Grand Prix',
  'Mexico City': 'Mexico City Grand Prix',
  Interlagos: 'Brazilian Grand Prix',
  'Las Vegas': 'Las Vegas Grand Prix',
  Lusail: 'Qatar Grand Prix',
  'Yas Marina Circuit': 'Abu Dhabi Grand Prix',
};

function formatDate(d: Date): string {
  return d.toLocaleDateString('en-GB', { day: 'numeric', month: 'short' });
}

function formatFullDate(d: Date): string {
  return d.toLocaleDateString('en-GB', { weekday: 'short', day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit', hour12: false });
}

function getWeekendRange(start: Date, end: Date | null): string {
  const s = start.toLocaleDateString('en-GB', { day: 'numeric', month: 'short' });
  if (!end) return s;
  const e = end.toLocaleDateString('en-GB', { day: 'numeric', month: 'short' });
  if (start.getMonth() === end.getMonth()) {
    return `${start.getDate()}–${end.getDate()} ${start.toLocaleDateString('en-GB', { month: 'short' })}`;
  }
  return `${s} – ${e}`;
}

type RaceStatus = 'completed' | 'live' | 'next' | 'upcoming';

function getRaceStatus(weekend: RaceWeekend, now: Date): RaceStatus {
  const raceEnd = weekend.raceDate
    ? new Date(weekend.raceDate.getTime() + 2 * 60 * 60 * 1000)
    : new Date(weekend.dateStart.getTime() + 3 * 24 * 60 * 60 * 1000);

  if (now > raceEnd) return 'completed';

  const weekendStart = new Date(weekend.dateStart.getTime() - 2 * 60 * 60 * 1000);
  if (now >= weekendStart && now <= raceEnd) return 'live';

  return 'upcoming';
}

export function RaceSchedule() {
  const [sessions, setSessions] = useState<SessionDoc[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('/api/local/openf1/sessions?year=2026')
      .then(r => r.json())
      .then((data: SessionDoc[]) => {
        setSessions(Array.isArray(data) ? data : []);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  const weekends = useMemo(() => {
    if (!sessions.length) return [];

    const grouped = new Map<number, SessionDoc[]>();
    for (const s of sessions) {
      const arr = grouped.get(s.meeting_key) || [];
      arr.push(s);
      grouped.set(s.meeting_key, arr);
    }

    const results: RaceWeekend[] = [];
    for (const [mk, docs] of grouped) {
      const sessionNames = docs.map(d => d.session_name);
      // Skip pre-season testing
      if (sessionNames.every(n => n.startsWith('Day'))) continue;

      const sorted = [...docs].sort((a, b) => new Date(a.date_start).getTime() - new Date(b.date_start).getTime());
      const first = sorted[0];
      const raceSess = sorted.find(d => d.session_name === 'Race');

      results.push({
        meeting_key: mk,
        circuit: first.circuit_short_name,
        country: first.country_name,
        countryCode: first.country_code,
        location: first.location,
        dateStart: new Date(first.date_start),
        raceDate: raceSess ? new Date(raceSess.date_start) : null,
        sessions: sorted.map(d => d.session_name),
        isSprint: sessionNames.some(n => n === 'Sprint' || n === 'Sprint Qualifying'),
        round: 0,
        gpName: CIRCUIT_GP_NAMES[first.circuit_short_name] || `${first.country_name} Grand Prix`,
      });
    }

    results.sort((a, b) => a.dateStart.getTime() - b.dateStart.getTime());
    results.forEach((r, i) => { r.round = i + 1; });
    return results;
  }, [sessions]);

  const now = new Date();

  const nextIdx = useMemo(() => {
    const idx = weekends.findIndex(w => getRaceStatus(w, now) === 'live' || getRaceStatus(w, now) === 'upcoming');
    return idx >= 0 ? idx : weekends.length - 1;
  }, [weekends, now]);

  const completedCount = weekends.filter(w => getRaceStatus(w, now) === 'completed').length;
  const sprintCount = weekends.filter(w => w.isSprint).length;

  if (loading) {
    return (
      <div className="flex items-center justify-center py-32">
        <Loader2 className="w-6 h-6 text-[#FF8000] animate-spin" />
      </div>
    );
  }

  if (!weekends.length) {
    return (
      <div className="flex flex-col items-center justify-center py-32 text-muted-foreground">
        <Calendar className="w-8 h-8 mb-3 opacity-30" />
        <p className="text-sm">No 2026 schedule data available</p>
      </div>
    );
  }

  return (
    <div className="space-y-5 pt-4">
      {/* Season summary strip */}
      <div className="flex items-center gap-5 px-5 py-3 bg-card rounded-xl border border-border">
        <div className="flex items-center gap-2">
          <Flag className="w-4 h-4 text-[#FF8000]" />
          <span className="text-[13px] font-semibold text-foreground tracking-wide">2026 SEASON</span>
        </div>
        <div className="w-px h-5 bg-[rgba(255,128,0,0.12)]" />
        <span className="text-[12px] text-muted-foreground">
          <span className="font-mono text-foreground">{weekends.length}</span> rounds
        </span>
        <span className="text-[12px] text-muted-foreground">
          <span className="font-mono text-green-400">{completedCount}</span> completed
        </span>
        <span className="text-[12px] text-muted-foreground">
          <span className="font-mono text-[#FF8000]">{weekends.length - completedCount}</span> remaining
        </span>
        <span className="text-[12px] text-muted-foreground">
          <Zap className="w-3 h-3 inline text-yellow-400 mr-0.5" />
          <span className="font-mono text-yellow-400">{sprintCount}</span> sprints
        </span>
        {/* Progress bar */}
        <div className="flex-1 h-1.5 bg-background rounded-full overflow-hidden ml-2">
          <div
            className="h-full rounded-full bg-gradient-to-r from-[#FF8000] to-[#FF8000]/60 transition-all duration-700"
            style={{ width: `${(completedCount / weekends.length) * 100}%` }}
          />
        </div>
        <span className="text-[11px] font-mono text-muted-foreground">{Math.round((completedCount / weekends.length) * 100)}%</span>
      </div>

      {/* Featured next race */}
      {nextIdx >= 0 && nextIdx < weekends.length && (
        <NextRaceHero weekend={weekends[nextIdx]} status={getRaceStatus(weekends[nextIdx], now)} now={now} />
      )}

      {/* Full calendar grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
        {weekends.map((w, i) => {
          const status = getRaceStatus(w, now);
          return (
            <RaceCard
              key={w.meeting_key}
              weekend={w}
              status={status}
              index={i}
              isNext={i === nextIdx}
            />
          );
        })}
      </div>
    </div>
  );
}

/* ─── Featured Next Race ─── */

function NextRaceHero({ weekend: w, status, now }: { weekend: RaceWeekend; status: RaceStatus; now: Date }) {
  const isLive = status === 'live';
  const raceDate = w.raceDate || w.dateStart;
  const diff = raceDate.getTime() - now.getTime();
  const daysUntil = Math.max(0, Math.ceil(diff / (1000 * 60 * 60 * 24)));

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="relative bg-card border border-[rgba(255,128,0,0.15)] rounded-xl overflow-hidden"
    >
      {/* Top accent */}
      <div className="absolute top-0 left-0 right-0 h-[2px] bg-gradient-to-r from-[#FF8000] via-[#FF8000]/60 to-transparent" />
      {/* Background glow */}
      <div className="absolute inset-0 opacity-[0.03] bg-gradient-to-br from-[#FF8000] to-transparent pointer-events-none" />

      <div className="relative px-6 py-5 flex items-start gap-6">
        {/* Left: Round badge */}
        <div className="flex flex-col items-center gap-1 shrink-0">
          <span className="text-[10px] text-muted-foreground uppercase tracking-widest">Round</span>
          <span className="text-[32px] font-black font-mono text-[#FF8000] leading-none">{String(w.round).padStart(2, '0')}</span>
        </div>

        {/* Center: Race info */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            {isLive && (
              <span className="flex items-center gap-1 px-2 py-0.5 rounded bg-red-500/20 border border-red-500/30 text-[10px] font-bold text-red-400 uppercase tracking-wider">
                <span className="w-1.5 h-1.5 rounded-full bg-red-400 animate-pulse" />
                Live
              </span>
            )}
            {!isLive && (
              <span className="px-2 py-0.5 rounded bg-[#FF8000]/15 border border-[#FF8000]/20 text-[10px] font-bold text-[#FF8000] uppercase tracking-wider">
                Next Race
              </span>
            )}
            {w.isSprint && (
              <span className="flex items-center gap-1 px-2 py-0.5 rounded bg-yellow-500/15 border border-yellow-500/20 text-[10px] font-bold text-yellow-400 uppercase tracking-wider">
                <Zap className="w-2.5 h-2.5" /> Sprint
              </span>
            )}
          </div>

          <h2 className="text-lg font-bold text-foreground tracking-tight leading-tight">{w.gpName}</h2>

          <div className="flex items-center gap-3 mt-1.5 text-[12px] text-muted-foreground">
            <span className="flex items-center gap-1">
              {COUNTRY_FLAGS[w.countryCode] && <span className="text-base">{COUNTRY_FLAGS[w.countryCode]}</span>}
              {w.circuit}
            </span>
            <span className="flex items-center gap-1">
              <MapPin className="w-3 h-3" />
              {w.location}, {w.country}
            </span>
          </div>

          {/* Session timeline */}
          <div className="flex items-center gap-1 mt-3 flex-wrap">
            {w.sessions.map((s, i) => (
              <span
                key={i}
                className={`text-[9px] font-medium tracking-wide px-2 py-0.5 rounded-md border ${
                  s === 'Race'
                    ? 'bg-[#FF8000]/15 border-[#FF8000]/25 text-[#FF8000]'
                    : s === 'Sprint'
                    ? 'bg-yellow-500/10 border-yellow-500/20 text-yellow-400'
                    : 'bg-background border-border text-muted-foreground'
                }`}
              >
                {s}
              </span>
            ))}
          </div>
        </div>

        {/* Right: Countdown */}
        <div className="shrink-0 text-right">
          {!isLive && w.raceDate && (
            <>
              <div className="text-[10px] text-muted-foreground uppercase tracking-widest mb-1">Race Day</div>
              <div className="text-[13px] font-semibold text-foreground">{formatFullDate(w.raceDate)}</div>
              <div className="mt-2 flex items-center gap-1.5 justify-end">
                <Clock className="w-3 h-3 text-[#FF8000]" />
                <span className="text-[18px] font-black font-mono text-[#FF8000]">{daysUntil}</span>
                <span className="text-[11px] text-muted-foreground">{daysUntil === 1 ? 'day' : 'days'} away</span>
              </div>
            </>
          )}
          {isLive && (
            <div className="flex flex-col items-end gap-1">
              <span className="text-[11px] text-red-400 font-semibold">Race Weekend</span>
              <span className="text-[11px] text-muted-foreground">{getWeekendRange(w.dateStart, w.raceDate)}</span>
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
}

/* ─── Race Card ─── */

function RaceCard({ weekend: w, status, index, isNext }: { weekend: RaceWeekend; status: RaceStatus; index: number; isNext: boolean }) {
  const isCompleted = status === 'completed';
  const isLive = status === 'live';

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25, delay: Math.min(index * 0.025, 0.5) }}
      className={`relative bg-card border rounded-xl overflow-hidden transition-all group ${
        isNext
          ? 'border-[#FF8000]/30 ring-1 ring-[#FF8000]/10'
          : isLive
          ? 'border-red-500/30'
          : isCompleted
          ? 'border-[rgba(255,128,0,0.05)] opacity-60'
          : 'border-border hover:border-[rgba(255,128,0,0.18)]'
      }`}
    >
      {/* Top accent line */}
      <div className={`absolute top-0 left-0 right-0 h-[2px] ${
        isLive ? 'bg-red-500' : isNext ? 'bg-[#FF8000]' : isCompleted ? 'bg-green-500/40' : 'bg-[rgba(255,128,0,0.15)]'
      }`} />

      <div className="p-3.5">
        {/* Header row: round + status */}
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <span className="text-[18px] font-black font-mono leading-none text-[#FF8000]/30">
              {String(w.round).padStart(2, '0')}
            </span>
            {COUNTRY_FLAGS[w.countryCode] && (
              <span className="text-base leading-none">{COUNTRY_FLAGS[w.countryCode]}</span>
            )}
          </div>
          <div className="flex items-center gap-1.5">
            {w.isSprint && (
              <Zap className="w-3 h-3 text-yellow-400" />
            )}
            {isCompleted && <CheckCircle2 className="w-3.5 h-3.5 text-green-500/70" />}
            {isLive && (
              <span className="flex items-center gap-1 text-[9px] font-bold text-red-400">
                <span className="w-1.5 h-1.5 rounded-full bg-red-400 animate-pulse" />
                LIVE
              </span>
            )}
            {isNext && !isLive && (
              <ChevronRight className="w-3.5 h-3.5 text-[#FF8000]" />
            )}
          </div>
        </div>

        {/* GP Name */}
        <div className="text-[13px] font-bold text-foreground leading-tight mb-0.5 tracking-tight">
          {w.gpName.replace(' Grand Prix', '')}
        </div>
        <div className="text-[10px] text-muted-foreground/70 mb-2">{w.circuit}</div>

        {/* Date */}
        <div className="flex items-center justify-between text-[11px]">
          <span className="text-muted-foreground font-mono">
            {w.raceDate ? formatDate(w.raceDate) : formatDate(w.dateStart)}
          </span>
          {w.isSprint && (
            <span className="text-[9px] font-semibold text-yellow-400/80 tracking-wide">SPRINT</span>
          )}
        </div>
      </div>
    </motion.div>
  );
}
