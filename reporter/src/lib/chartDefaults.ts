import {
  BarElement,
  CategoryScale,
  Chart as ChartJS,
  Filler,
  Legend,
  LinearScale,
  LineElement,
  LogarithmicScale,
  PointElement,
  Tooltip,
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  LogarithmicScale,
  PointElement,
  LineElement,
  BarElement,
  Filler,
  Tooltip,
  Legend,
);

export function cssVar(name: string) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export function tooltipStyle() {
  return {
    backgroundColor: cssVar('--color-surface2'),
    borderColor: cssVar('--color-border'),
    borderWidth: 1,
    titleColor: cssVar('--color-text'),
    bodyColor: cssVar('--color-text'),
  };
}

export function gridColor() {
  return cssVar('--chart-grid') || cssVar('--color-border') + '44';
}
