import { useCallback, useState } from 'react';
import { FiArrowLeft } from 'react-icons/fi';
import Charts from './components/Charts';
import FileUpload from './components/FileUpload';
import StatsRow from './components/StatsRow';
import ThemeToggle from './components/ThemeToggle';
import './lib/chartDefaults';
import { computeStats, downsample, parseFile } from './lib/parse';
import type { TrainingEntry, TrainingStats } from './lib/types';

export default function App() {
  const [data, setData] = useState<TrainingEntry[] | null>(null);
  const [stats, setStats] = useState<TrainingStats | null>(null);
  const [filename, setFilename] = useState('');
  const [error, setError] = useState<string | null>(null);

  const handleFile = useCallback((text: string, name: string) => {
    try {
      const parsed = parseFile(text, name);

      if (parsed.length === 0) {
        throw new Error('File contained no entries');
      }

      const sampled = downsample(parsed);

      setData(sampled);
      setStats(computeStats(parsed));
      setFilename(name);
      setError(null);
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Failed to parse file');
      setData(null);
      setStats(null);
    }
  }, []);

  const handleReset = useCallback(() => {
    setData(null);
    setStats(null);
    setFilename('');
    setError(null);
  }, []);

  if (!data || !stats) {
    return (
      <div className="relative">
        <div className="absolute top-6 right-6">
          <ThemeToggle />
        </div>
        <FileUpload onFileLoaded={handleFile} />
        {error && (
          <div className="fixed bottom-6 left-1/2 -translate-x-1/2 bg-accent2/20 border border-accent2/40 text-accent2 px-5 py-3 rounded-xl text-sm font-medium backdrop-blur">
            {error}
          </div>
        )}
      </div>
    );
  }

  return (
    <div>
      <header className="max-w-[1200px] mx-auto px-6 pt-10 pb-6">
        <div className="flex justify-between items-start gap-4">
          <div>
            <h1 className="text-3xl font-bold tracking-tight mb-1">
              Pico<span className="text-accent">Router</span> — Training Report
            </h1>
            <p className="text-muted text-sm">
              <button
                onClick={handleReset}
                className="inline-flex items-center gap-1 text-accent hover:underline mr-2 cursor-pointer"
              >
                <FiArrowLeft aria-hidden="true" className="text-base" />
                Upload new file
              </button>
              <span className="font-mono text-xs opacity-60">{filename}</span>
            </p>
          </div>
          <ThemeToggle />
        </div>
      </header>

      <StatsRow stats={stats} />

      <Charts data={data} />
    </div>
  );
}
