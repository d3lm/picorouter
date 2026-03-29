export interface TrainingEntry {
  step: number;
  epoch: number;
  train_loss: number;
  lr: number;
  wall_clock: number;
  val_loss?: number;
}

export interface TrainingStats {
  totalSteps: number;
  epochs: number;
  finalValLoss: number;
  bestValLoss: number;
  bestValStep: number;
  perplexity: number;
  wallClockSeconds: number;
  phaseTransitionStep: number | null;
}
