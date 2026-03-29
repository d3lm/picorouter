import type { ReactNode } from 'react';

interface Props {
  title: string;
  dotColor: string;
  children: ReactNode;
  annotation?: ReactNode;
  tall?: boolean;
}

export default function ChartPanel({ title, dotColor, children, annotation, tall }: Props) {
  return (
    <div className="bg-surface border border-border rounded-xl p-5">
      <h2 className="text-base font-medium mb-4 flex items-center gap-2">
        <span className="w-2 h-2 rounded-full inline-block" style={{ background: dotColor }} />
        {title}
      </h2>
      <div className="relative w-full" style={{ height: tall ? 420 : 360 }}>
        {children}
      </div>
      {annotation && (
        <div className="bg-surface2 border border-border border-l-[3px] border-l-accent rounded-lg px-4 py-3 mt-4 text-sm text-muted leading-relaxed">
          {annotation}
        </div>
      )}
    </div>
  );
}
