interface StatusBadgeProps {
  status: 'nominal' | 'warning' | 'critical' | 'alert';
  size?: 'sm' | 'md';
}

const statusConfig = {
  nominal: { label: 'NOMINAL', bg: 'bg-green-500/15', text: 'text-green-400', dot: 'bg-green-400' },
  warning: { label: 'WARNING', bg: 'bg-amber-500/15', text: 'text-amber-400', dot: 'bg-amber-400' },
  alert: { label: 'ALERT', bg: 'bg-orange-500/15', text: 'text-orange-400', dot: 'bg-orange-400' },
  critical: { label: 'CRITICAL', bg: 'bg-red-500/15', text: 'text-red-400', dot: 'bg-red-400' },
};

export function StatusBadge({ status, size = 'sm' }: StatusBadgeProps) {
  const config = statusConfig[status];
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-md ${config.bg} ${config.text} ${
        size === 'sm' ? 'px-2 py-0.5 text-[12px]' : 'px-3 py-1 text-sm'
      } tracking-wider`}
    >
      <span className={`${config.dot} rounded-full ${size === 'sm' ? 'h-1.5 w-1.5' : 'h-2 w-2'}`} />
      {config.label}
    </span>
  );
}

export function SeverityBadge({ severity }: { severity: 'low' | 'medium' | 'high' | 'critical' }) {
  const config = {
    low: { bg: 'bg-blue-500/15', text: 'text-blue-400' },
    medium: { bg: 'bg-amber-500/15', text: 'text-amber-400' },
    high: { bg: 'bg-orange-500/15', text: 'text-orange-400' },
    critical: { bg: 'bg-red-500/15', text: 'text-red-400' },
  }[severity];
  return (
    <span className={`inline-flex items-center rounded-md px-2 py-0.5 text-[12px] tracking-wider uppercase ${config.bg} ${config.text}`}>
      {severity}
    </span>
  );
}
