import type { TrainingEntry, TrainingStats } from './types';

export function parseJSONL(text: string): TrainingEntry[] {
  return text
    .split('\n')
    .filter((line) => line.trim())
    .map((line) => JSON.parse(line) as TrainingEntry);
}

export function parseJSON(text: string): TrainingEntry[] {
  const parsed = JSON.parse(text);

  if (Array.isArray(parsed)) {
    return parsed as TrainingEntry[];
  }

  throw new Error('Expected a JSON array of training entries');
}

export function parseFile(text: string, filename: string): TrainingEntry[] {
  if (filename.endsWith('.jsonl')) {
    return parseJSONL(text);
  }

  if (filename.endsWith('.json')) {
    return parseJSON(text);
  }

  try {
    return parseJSONL(text);
  } catch {
    return parseJSON(text);
  }
}

/**
 * Downsample to at most `maxPoints` entries, keeping every entry that has
 * val_loss and evenly spacing the rest.
 */
export function downsample(data: TrainingEntry[], maxPoints = 400): TrainingEntry[] {
  if (data.length <= maxPoints) {
    return data;
  }

  const valEntries = new Set(data.filter(({ val_loss }) => val_loss !== undefined).map(({ step }) => step));

  const stride = Math.ceil(data.length / maxPoints);

  const result: TrainingEntry[] = [];

  for (let i = 0; i < data.length; i++) {
    if (i % stride === 0 || valEntries.has(data[i].step)) {
      result.push(data[i]);
    }
  }

  return result;
}

export function computeStats(data: TrainingEntry[]): TrainingStats {
  const valEntries = data.filter(({ val_loss }) => val_loss !== undefined);

  const lastValue = valEntries[valEntries.length - 1];

  const bestValue = valEntries.reduce(
    (best, entry) => (entry.val_loss! < best.val_loss! ? entry : best),
    valEntries[0],
  );

  const last = data[data.length - 1];
  const epochs = new Set(data.map(({ epoch }) => epoch)).size;

  let phaseTransitionStep: number | null = null;
  let maxDrop = 0;

  for (let i = 1; i < valEntries.length; i++) {
    const drop = valEntries[i - 1].val_loss! - valEntries[i].val_loss!;

    if (drop > maxDrop) {
      maxDrop = drop;
      phaseTransitionStep = valEntries[i].step;
    }
  }

  return {
    totalSteps: last.step,
    epochs,
    finalValLoss: lastValue?.val_loss ?? last.train_loss,
    bestValLoss: bestValue?.val_loss ?? last.train_loss,
    bestValStep: bestValue?.step ?? last.step,
    perplexity: Math.exp(lastValue?.val_loss ?? last.train_loss),
    wallClockSeconds: last.wall_clock,
    phaseTransitionStep,
  };
}
