import type { ChartOptions } from 'chart.js';
import { useEffect, useMemo, useState } from 'react';
import { Bar, Line } from 'react-chartjs-2';
import { cssVar, gridColor, tooltipStyle } from '../lib/chartDefaults';
import type { TrainingEntry } from '../lib/types';
import ChartPanel from './ChartPanel';

interface Props {
  data: TrainingEntry[];
}

function useChartColors() {
  const [themeVersion, setThemeVersion] = useState(0);

  useEffect(() => {
    const handleThemeChange = () => {
      setThemeVersion((version) => version + 1);
    };

    window.addEventListener('picorouter-theme-change', handleThemeChange);

    return () => {
      window.removeEventListener('picorouter-theme-change', handleThemeChange);
    };
  }, []);

  return useMemo(() => {
    const get = cssVar;

    return {
      accent: get('--color-accent'),
      accent2: get('--color-accent2'),
      accent3: get('--color-accent3'),
      accent4: get('--color-accent4'),
      muted: get('--color-muted'),
      grid: gridColor(),
      tt: tooltipStyle(),
    };
  }, [themeVersion]);
}

function LossChart({ data }: Props) {
  const chartColors = useChartColors();

  const valData = data.filter(({ val_loss }) => val_loss !== undefined);

  const chartData = {
    datasets: [
      {
        label: 'Train Loss',
        data: data.map((d) => ({ x: d.step, y: d.train_loss })),
        borderColor: chartColors.accent,
        backgroundColor: chartColors.accent + '1a',
        borderWidth: 1.5,
        pointRadius: 0,
        fill: true,
        tension: 0.3,
      },
      {
        label: 'Val Loss',
        data: valData.map((d) => ({ x: d.step, y: d.val_loss! })),
        borderColor: chartColors.accent2,
        backgroundColor: chartColors.accent2 + '1a',
        borderWidth: 2.5,
        pointRadius: 3,
        pointBackgroundColor: chartColors.accent2,
        fill: false,
        tension: 0.3,
      },
    ],
  };

  const options: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    scales: {
      x: {
        type: 'linear',
        title: { display: true, text: 'Step', color: chartColors.muted },
        grid: { color: chartColors.grid },
        ticks: { maxTicksLimit: 12 },
      },
      y: {
        type: 'logarithmic',
        title: { display: true, text: 'Loss (log scale)', color: chartColors.muted },
        grid: { color: chartColors.grid },
      },
    },
    plugins: {
      legend: { position: 'top', labels: { usePointStyle: true, padding: 20 } },
      tooltip: {
        ...chartColors.tt,
        callbacks: {
          label: (ctx) => `${ctx.dataset.label}: ${(ctx.parsed.y ?? 0).toFixed(4)}`,
        },
      },
    },
  };

  return (
    <ChartPanel title="Loss Curves (Full Run)" dotColor={chartColors.accent} tall>
      <Line data={chartData} options={options} />
    </ChartPanel>
  );
}

function ValLossLogChart({ data }: Props) {
  const chartColors = useChartColors();

  const valData = data.filter(({ val_loss }) => val_loss !== undefined);

  const chartData = {
    datasets: [
      {
        label: 'Val Loss',
        data: valData.map((d) => ({ x: d.step, y: d.val_loss! })),
        borderColor: chartColors.accent3,
        backgroundColor: chartColors.accent3 + '18',
        borderWidth: 2,
        pointRadius: 2.5,
        pointBackgroundColor: chartColors.accent3,
        fill: true,
        tension: 0.3,
      },
    ],
  };

  const options: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      x: {
        type: 'linear',
        title: { display: true, text: 'Step', color: chartColors.muted },
        grid: { color: chartColors.grid },
        ticks: { maxTicksLimit: 8 },
      },
      y: {
        type: 'logarithmic',
        title: { display: true, text: 'Val Loss (log)', color: chartColors.muted },
        grid: { color: chartColors.grid },
      },
    },
    plugins: {
      legend: { display: false },
      tooltip: {
        ...chartColors.tt,
        callbacks: {
          title: (items) => `Step ${items[0]?.parsed?.x?.toLocaleString() ?? ''}`,
          label: (ctx) => `Val Loss: ${(ctx.parsed.y ?? 0).toFixed(4)}`,
        },
      },
    },
  };

  return (
    <ChartPanel title="Validation Loss (Log Scale)" dotColor={chartColors.accent3}>
      <Line data={chartData} options={options} />
    </ChartPanel>
  );
}

function LRChart({ data }: Props) {
  const chartColors = useChartColors();

  const chartData = {
    datasets: [
      {
        label: 'Learning Rate',
        data: data.filter((_, i) => i % 2 === 0).map((d) => ({ x: d.step, y: d.lr })),
        borderColor: chartColors.accent4,
        borderWidth: 2,
        pointRadius: 0,
        fill: { target: 'origin' as const, above: chartColors.accent4 + '14' },
        tension: 0.3,
      },
    ],
  };

  const options: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      x: {
        type: 'linear',
        title: { display: true, text: 'Step', color: chartColors.muted },
        grid: { color: chartColors.grid },
        ticks: { maxTicksLimit: 8 },
      },
      y: {
        title: { display: true, text: 'LR', color: chartColors.muted },
        grid: { color: chartColors.grid },
        ticks: {
          callback: (v) => (Number(v) * 10000).toFixed(1) + 'e-4',
        },
      },
    },
    plugins: {
      legend: { display: false },
      tooltip: {
        ...chartColors.tt,
        callbacks: {
          title: (items) => `Step ${items[0]?.parsed?.x?.toLocaleString() ?? ''}`,
          label: (ctx) => `LR: ${(ctx.parsed.y ?? 0).toExponential(2)}`,
        },
      },
    },
  };

  return (
    <ChartPanel title="Learning Rate Schedule" dotColor={chartColors.accent4}>
      <Line data={chartData} options={options} />
    </ChartPanel>
  );
}

function ConvergenceChart({ data }: Props) {
  const chartColors = useChartColors();

  const totalSteps = data[data.length - 1]?.step ?? 0;
  const cutoff = Math.floor(totalSteps * 0.2);
  const convergenceData = data.filter(({ step }) => step >= cutoff);
  const convergenceVal = convergenceData.filter(({ val_loss }) => val_loss !== undefined);

  if (convergenceData.length < 2) {
    return null;
  }

  const chartData = {
    datasets: [
      {
        label: 'Train Loss',
        data: convergenceData.map((d) => ({ x: d.step, y: d.train_loss })),
        borderColor: chartColors.accent,
        backgroundColor: chartColors.accent + '10',
        borderWidth: 1.5,
        pointRadius: 0,
        fill: true,
        tension: 0.2,
      },
      {
        label: 'Val Loss',
        data: convergenceVal.map((d) => ({ x: d.step, y: d.val_loss! })),
        borderColor: chartColors.accent2,
        borderWidth: 2.5,
        pointRadius: 3,
        pointBackgroundColor: chartColors.accent2,
        fill: false,
        tension: 0.3,
      },
    ],
  };

  const options: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    scales: {
      x: {
        type: 'linear',
        title: { display: true, text: 'Step', color: chartColors.muted },
        grid: { color: chartColors.grid },
        ticks: { maxTicksLimit: 8 },
      },
      y: {
        title: { display: true, text: 'Loss', color: chartColors.muted },
        grid: { color: chartColors.grid },
        min: 0,
      },
    },
    plugins: {
      legend: {
        position: 'top',
        labels: { usePointStyle: true, padding: 16 },
      },
      tooltip: {
        ...chartColors.tt,
        callbacks: {
          label: (ctx) => `${ctx.dataset.label}: ${(ctx.parsed.y ?? 0).toFixed(4)}`,
        },
      },
    },
  };

  return (
    <ChartPanel title={`Convergence Phase (Steps ${(cutoff / 1000).toFixed(0)}K+)`} dotColor={chartColors.accent2}>
      <Line data={chartData} options={options} />
    </ChartPanel>
  );
}

function GapChart({ data }: Props) {
  const chartColors = useChartColors();

  const valData = data.filter(({ val_loss }) => val_loss !== undefined);
  const totalSteps = data[data.length - 1]?.step ?? 0;
  const cutoff = Math.floor(totalSteps * 0.2);

  const gapData = valData
    .filter(({ step }) => step >= cutoff)
    .map(({ step, val_loss }) => {
      const nearTrain = data.find(({ step: tStep }) => tStep === step);

      return { step, val: val_loss!, train: nearTrain?.train_loss ?? null };
    })
    .filter(({ train }) => train !== null);

  if (gapData.length < 2) {
    return null;
  }

  const chartData = {
    labels: gapData.map((d) => (d.step >= 1000 ? (d.step / 1000).toFixed(0) + 'K' : String(d.step))),
    datasets: [
      {
        label: 'Train Loss',
        data: gapData.map((d) => d.train!),
        backgroundColor: chartColors.accent + '99',
        borderColor: chartColors.accent,
        borderWidth: 1,
        borderRadius: 3,
      },
      {
        label: 'Val Loss',
        data: gapData.map((d) => d.val),
        backgroundColor: chartColors.accent2 + '99',
        borderColor: chartColors.accent2,
        borderWidth: 1,
        borderRadius: 3,
      },
    ],
  };

  const options: ChartOptions<'bar'> = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      x: {
        title: { display: true, text: 'Step', color: chartColors.muted },
        grid: { display: false },
        ticks: { maxRotation: 45, font: { size: 9 } },
      },
      y: {
        title: { display: true, text: 'Loss', color: chartColors.muted },
        grid: { color: chartColors.grid },
      },
    },
    plugins: {
      legend: {
        position: 'top',
        labels: { usePointStyle: true, padding: 16 },
      },
      tooltip: chartColors.tt,
    },
  };

  return (
    <ChartPanel title="Train vs Val Gap" dotColor={chartColors.accent}>
      <Bar data={chartData} options={options} />
    </ChartPanel>
  );
}

export default function Charts({ data }: Props) {
  return (
    <div className="max-w-[1200px] mx-auto px-6 pb-10 flex flex-col gap-5">
      <LossChart data={data} />
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        <ValLossLogChart data={data} />
        <LRChart data={data} />
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        <ConvergenceChart data={data} />
        <GapChart data={data} />
      </div>
    </div>
  );
}
