interface StatusBadgeProps {
  status: 'nominal' | 'warning' | 'critical' | 'alert';
  size?: 'sm' | 'md';
}

const statusConfig = {
  nominal: { label: 'NOMINAL', bg: 'bg-success-soft', text: 'text-success-text', dot: 'bg-success' },
  warning: { label: 'WARNING', bg: 'bg-warning-soft', text: 'text-warning-text', dot: 'bg-warning' },
  alert: { label: 'ALERT', bg: 'bg-accent-soft', text: 'text-primary', dot: 'bg-primary' },
  critical: { label: 'CRITICAL', bg: 'bg-danger-soft', text: 'text-danger-text', dot: 'bg-danger' },
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
    low: { bg: 'bg-info-soft', text: 'text-info-text' },
    medium: { bg: 'bg-warning-soft', text: 'text-warning-text' },
    high: { bg: 'bg-accent-soft', text: 'text-primary' },
    critical: { bg: 'bg-danger-soft', text: 'text-danger-text' },
  }[severity];
  return (
    <span className={`inline-flex items-center rounded-md px-2 py-0.5 text-[12px] tracking-wider uppercase ${config.bg} ${config.text}`}>
      {severity}
    </span>
  );
}
