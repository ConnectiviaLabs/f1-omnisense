import { Loader2 } from 'lucide-react';

interface LoadingSpinnerProps {
  text?: string;
  className?: string;
}

export function LoadingSpinner({ text = 'Loading...', className = '' }: LoadingSpinnerProps) {
  return (
    <div className={`flex items-center gap-2 text-sm text-muted-foreground py-8 justify-center ${className}`}>
      <Loader2 className="w-4 h-4 animate-spin text-primary" />
      {text}
    </div>
  );
}
