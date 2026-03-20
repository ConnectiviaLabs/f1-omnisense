/**
 * ModelAgreement — visual indicator showing which anomaly ensemble models
 * flagged an anomaly vs. cleared it.
 *
 * Renders N dots (one per model), colored by flag status:
 *   - Flagged: red/orange/yellow depending on overall severity
 *   - Clear: green
 *
 * Also shows a consensus label: "Unanimous clear", "Split", "Majority flagged", etc.
 */
import { modelAgreementLabel, modelDotColor, type HealthLevel } from './anomalyHelpers';

const MODEL_NAMES = ['IF', 'SVM', 'KNN', 'PCA'];

interface ModelAgreementProps {
  voteCount: number;
  totalModels: number;
  level?: HealthLevel;
  compact?: boolean;
}

export function ModelAgreement({ voteCount, totalModels, level = 'nominal', compact = false }: ModelAgreementProps) {
  const n = Math.max(totalModels, 4);
  const label = modelAgreementLabel(voteCount, totalModels);

  return (
    <div className={`flex items-center gap-1.5 ${compact ? '' : 'mt-2'}`}>
      {/* Dots — one per model */}
      <div className="flex items-center gap-0.5">
        {Array.from({ length: n }, (_, i) => {
          const flagged = i < voteCount;
          const color = modelDotColor(flagged, level);
          const name = MODEL_NAMES[i] || `M${i + 1}`;
          return (
            <div
              key={i}
              title={`${name}: ${flagged ? 'Flagged' : 'Clear'}`}
              className="w-1.5 h-1.5 rounded-full transition-colors"
              style={{ backgroundColor: color }}
            />
          );
        })}
      </div>
      {/* Label */}
      <span className="text-[10px] text-muted-foreground font-mono">
        {label}
      </span>
    </div>
  );
}
