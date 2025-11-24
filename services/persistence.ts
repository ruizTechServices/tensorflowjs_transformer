
import * as tf from '@tensorflow/tfjs';
import { SimpleTokenizer, SerializedTokenizer } from './tokenizer';
import { TransformerModel } from './transformer';
import { ModelConfig } from '../types';

export interface SerializedWeights {
  [name: string]: {
    shape: number[];
    data: number[];
  };
}

export interface SavedModelState {
  config: ModelConfig;
  tokenizer: SerializedTokenizer;
  weights: SerializedWeights;
  timestamp: number;
}

export async function exportModelState(
  model: TransformerModel,
  tokenizer: SimpleTokenizer
): Promise<SavedModelState> {
  const vars = model.getTrainableVariables();
  const weights: SerializedWeights = {};

  // Extract all weights from GPU/Tensors
  for (const [name, variable] of Object.entries(vars)) {
    const data = await variable.data();
    weights[name] = {
      shape: variable.shape,
      data: Array.from(data as Float32Array),
    };
  }

  // Serialize Tokenizer
  const serializedTokenizer: SerializedTokenizer = {
    charToId: Object.fromEntries(tokenizer.vocab),
    idToChar: Object.fromEntries(tokenizer.inverseVocab),
    vocabSize: tokenizer.vocabSize,
  };

  return {
    config: model.config,
    tokenizer: serializedTokenizer,
    weights,
    timestamp: Date.now(),
  };
}

export async function importModelState(
  model: TransformerModel,
  state: SavedModelState
): Promise<void> {
  const vars = model.getTrainableVariables();

  tf.tidy(() => {
    for (const [name, spec] of Object.entries(state.weights)) {
      const v = vars[name];
      if (!v) {
        console.warn(`[persistence] No variable named "${name}" found in model architecture. Skipping.`);
        continue;
      }

      // Safety check for shape mismatch
      if (JSON.stringify(v.shape) !== JSON.stringify(spec.shape)) {
         console.warn(`[persistence] Shape mismatch for "${name}". Expected ${v.shape}, got ${spec.shape}. Skipping.`);
         continue;
      }

      const tensor = tf.tensor(spec.data, spec.shape);
      v.assign(tensor);
    }
  });
}

export function importTokenizer(serialized: SerializedTokenizer): SimpleTokenizer {
  return new SimpleTokenizer(serialized);
}
