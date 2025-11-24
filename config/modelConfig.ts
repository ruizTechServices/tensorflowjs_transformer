
import { ModelConfig } from '../types';

export type ModelSize = 'tiny' | 'small' | 'medium';

// Extending the base config to include display name
export interface PresetConfig extends Omit<ModelConfig, 'vocabSize'> {
  name: string;
}

export const MODEL_PRESETS: Record<ModelSize, PresetConfig> = {
  tiny: {
    name: 'Tiny (Playground)',
    dModel: 64,
    numHeads: 4,
    numLayers: 2,
    dff: 256, // usually 4x dModel
    maxLen: 64,
    dropoutRate: 0.1,
  },
  small: {
    name: 'Small (Learner)',
    dModel: 128,
    numHeads: 4,
    numLayers: 4,
    dff: 512,
    maxLen: 128,
    dropoutRate: 0.1,
  },
  medium: {
    name: 'Medium (Heavy)',
    dModel: 256,
    numHeads: 8,
    numLayers: 6,
    dff: 1024,
    maxLen: 128,
    dropoutRate: 0.1,
  },
};

export function buildModelConfig(
  size: ModelSize,
  vocabSize: number
): ModelConfig {
  // Destructure to remove the 'name' property, returning a pure ModelConfig
  const { name, ...base } = MODEL_PRESETS[size];
  return {
    ...base,
    vocabSize,
  };
}
