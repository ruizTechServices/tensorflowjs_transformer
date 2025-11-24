
export interface ModelConfig {
  vocabSize: number;
  dModel: number;
  numHeads: number;
  dff: number; // Feed forward dimension
  numLayers: number;
  maxLen: number;
  dropoutRate: number;
}

export interface AttentionState {
  attentionScores: number[][]; // Visualizing the weights
  headIndex: number;
}

export enum RunStatus {
  IDLE = 'IDLE',
  INITIALIZING = 'INITIALIZING',
  THINKING = 'THINKING', // Computing forward pass
  TRAINING = 'TRAINING',
  GENERATING = 'GENERATING',
  READY = 'READY',
}

// Colors for visualization
export const HEATMAP_COLORS = {
  LOW: '#0a0a0a',
  HIGH: '#00f3ff',
};

export interface GenerationConfig {
  maxNewTokens: number;
  temperature: number;
  topK: number;
}

export interface TrainingConfig {
  batchSize: number;
  epochs: number;
  learningRate: number;
  seqLen: number;
}
