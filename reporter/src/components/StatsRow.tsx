import type { TrainingStats } from '../lib/types';

function Card({ label, value, detail, color }: { label: string; value: string; detail?: string; color?: string }) {
  return (
    <div className="bg-surface border border-border rounded-xl px-5 py-4">
      <div className="text-[0.75rem] uppercase tracking-widest text-muted font-medium mb-1">{label}</div>
      <div className="font-mono text-2xl font-semibold" style={{ color }}>
        {value}
      </div>
      {detail && <div className="text-sm text-muted mt-0.5">{detail}</div>}
    </div>
  );
}

export default function StatsRow({ stats }: { stats: TrainingStats }) {
  return (
    <div className="grid grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-3 max-w-[1200px] mx-auto px-6 mb-6">
      <Card
        label="Final Val Loss"
        value={formatNumber(stats.finalValLoss)}
        detail={`Perplexity ${formatNumber(stats.perplexity, 2)}`}
        color="var(--color-accent3)"
      />

      <Card
        label="Best Val Loss"
        value={formatNumber(stats.bestValLoss)}
        detail={`Step ${stats.bestValStep.toLocaleString()}`}
        color="var(--color-accent)"
      />

      <Card
        label="Training Time"
        value={formatTime(stats.wallClockSeconds)}
        detail={`${stats.wallClockSeconds.toLocaleString(undefined, { maximumFractionDigits: 0 })}s wall clock`}
      />

      <Card
        label="Total Steps"
        value={stats.totalSteps.toLocaleString()}
        detail={`${stats.epochs} epoch${stats.epochs > 1 ? 's' : ''}`}
      />

      {stats.phaseTransitionStep && (
        <Card
          label="Phase Transition"
          value={`~${(stats.phaseTransitionStep / 1000).toFixed(1)}K`}
          detail="Largest val loss drop"
          color="var(--color-accent2)"
        />
      )}
    </div>
  );
}

function formatNumber(n: number, d = 3) {
  return n.toFixed(d);
}

function formatTime(sec: number) {
  if (sec < 60) return `${sec.toFixed(0)}s`;
  if (sec < 3600) return `${(sec / 60).toFixed(1)}m`;
  return `${(sec / 3600).toFixed(1)}h`;
}
