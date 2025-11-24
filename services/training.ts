
import * as tf from '@tensorflow/tfjs';
import { SimpleTokenizer } from './tokenizer';
import { TransformerModel } from './transformer';
import { TrainingConfig } from '../types';

interface Dataset {
  inputs: tf.Tensor2D;
  targets: tf.Tensor2D;
}

/**
 * Creates sliding window dataset:
 * Text: "HELLO"
 * SeqLen: 3
 * Input: "HEL", Target: "ELL"
 * Input: "ELL", Target: "LLO"
 */
export function buildDataset(
  text: string,
  tokenizer: SimpleTokenizer,
  seqLen: number,
  batchSize: number
): Dataset {
  return tf.tidy(() => {
    const ids = tokenizer.encode(text);
    const inputIndices: number[][] = [];
    const targetIndices: number[][] = [];

    // Create windows
    for (let i = 0; i <= ids.length - seqLen - 1; i++) {
      const inputWindow = ids.slice(i, i + seqLen);
      const targetWindow = ids.slice(i + 1, i + seqLen + 1);
      inputIndices.push(inputWindow);
      targetIndices.push(targetWindow);
    }

    // Truncate to fit batch size perfectly (simplifies logic)
    const numSamples = Math.floor(inputIndices.length / batchSize) * batchSize;
    
    if (numSamples === 0) {
      throw new Error("Text too short for current sequence length and batch size.");
    }

    const slicedInputs = inputIndices.slice(0, numSamples);
    const slicedTargets = targetIndices.slice(0, numSamples);

    return {
      inputs: tf.tensor2d(slicedInputs, [numSamples, seqLen], 'int32'),
      targets: tf.tensor2d(slicedTargets, [numSamples, seqLen], 'int32'),
    };
  });
}

export async function trainModel(
  model: TransformerModel,
  tokenizer: SimpleTokenizer,
  corpus: string,
  config: TrainingConfig,
  onLog: (msg: string) => void
) {
  onLog(`Preparing dataset from ${corpus.length} characters...`);
  
  let dataset: Dataset;
  try {
    dataset = buildDataset(corpus, tokenizer, config.seqLen, config.batchSize);
  } catch (e) {
    onLog(`Error: ${(e as Error).message}`);
    return;
  }

  const { inputs, targets } = dataset;
  const numSamples = inputs.shape[0];
  const stepsPerEpoch = numSamples / config.batchSize;

  onLog(`Dataset created. Samples: ${numSamples}. Batches per epoch: ${stepsPerEpoch}`);

  const optimizer = tf.train.adam(config.learningRate);

  for (let epoch = 1; epoch <= config.epochs; epoch++) {
    let totalLoss = 0;
    
    for (let step = 0; step < stepsPerEpoch; step++) {
      const start = step * config.batchSize;
      
      // Slice batch
      const batchInputs = inputs.slice([start, 0], [config.batchSize, config.seqLen]);
      const batchTargets = targets.slice([start, 0], [config.batchSize, config.seqLen]);

      // Optimization step
      const lossVal = tf.tidy(() => {
        const { value, grads } = tf.variableGrads(() => {
            // Forward pass
            const { logits } = model.forward(batchInputs, true); 
            // logits: [batch, seq, vocab]
            // targets: [batch, seq]
            
            // Calculate Loss
            const oneHotLabels = tf.oneHot(batchTargets, model.config.vocabSize);
            const loss = tf.losses.softmaxCrossEntropy(
                oneHotLabels, 
                logits
            );
            
            return loss.mean() as tf.Scalar;
        });

        // Cast to any to bypass NamedTensorMap vs NamedVariableMap typing issues in TFJS
        optimizer.applyGradients(grads as any);
        return value.dataSync()[0];
      });

      totalLoss += lossVal;
      
      // Cleanup batch tensors
      batchInputs.dispose();
      batchTargets.dispose();
      
      // Yield to UI occasionally
      if (step % 5 === 0) await new Promise(r => setTimeout(r, 0));
    }
    
    const avgLoss = totalLoss / stepsPerEpoch;
    onLog(`Epoch ${epoch}/${config.epochs} - Loss: ${avgLoss.toFixed(4)}`);
  }

  // Cleanup
  inputs.dispose();
  targets.dispose();
  onLog("Training complete. Weights updated.");
}
