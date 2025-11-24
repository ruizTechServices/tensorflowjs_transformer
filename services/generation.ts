
import * as tf from '@tensorflow/tfjs';
import { SimpleTokenizer } from './tokenizer';
import { TransformerModel } from './transformer';
import { GenerationConfig } from '../types';

/**
 * Samples a token index from logits using Temperature and Top-K.
 */
function sampleFromLogits(logits: tf.Tensor1D, temperature: number, topK: number): number {
  return tf.tidy(() => {
    // 1. Apply Temperature
    // Higher temp = more random, Lower temp = more deterministic
    let adjusted = logits.div(tf.scalar(Math.max(temperature, 1e-5)));

    // 2. Top-K Filtering
    if (topK > 0 && topK < logits.shape[0]) {
      const { values, indices } = tf.topk(adjusted, topK);
      
      // We sample directly from the Top-K values
      const topKProbs = tf.softmax(values);
      const topKProbsData = topKProbs.dataSync();
      
      const r = Math.random();
      let acc = 0;
      let selectedIndexInTopK = 0;
      for(let i=0; i<topKProbsData.length; i++) {
        acc += topKProbsData[i];
        if(r <= acc) {
          selectedIndexInTopK = i;
          break;
        }
      }
      
      // Return the original index from the indices tensor
      return indices.dataSync()[selectedIndexInTopK];
    }

    // 3. Standard Softmax + Sampling
    const probs = tf.softmax(adjusted);
    const probsData = probs.dataSync();

    const r = Math.random();
    let acc = 0;
    for (let i = 0; i < probsData.length; i++) {
      acc += probsData[i];
      if (r <= acc) return i;
    }
    return probsData.length - 1;
  });
}

export async function generateText(
  model: TransformerModel,
  tokenizer: SimpleTokenizer,
  prompt: string,
  config: GenerationConfig,
  onStep?: (token: string) => void
): Promise<string> {
  let tokenIds = tokenizer.encode(prompt);
  let resultText = "";

  // Safety check
  if (tokenIds.length === 0) tokenIds = [tokenizer.vocab.get('<UNK>') || 1];

  for (let i = 0; i < config.maxNewTokens; i++) {
    // Context Window Clipping
    // If sequence is too long, take the last (maxLen) tokens
    const contextLen = Math.min(tokenIds.length, model.config.maxLen);
    const contextIds = tokenIds.slice(-contextLen);

    const inputTensor = tf.tensor2d([contextIds], [1, contextIds.length], 'int32');
    
    const { logits } = model.forward(inputTensor, false);
    
    // Get logits for the last token: [batch=1, seqLen, vocabSize] -> [vocabSize]
    const lastLogits = tf.tidy(() => {
        const seqLen = logits.shape[1];
        return logits.slice([0, seqLen - 1, 0], [1, 1, -1]).squeeze() as tf.Tensor1D;
    });

    const nextId = sampleFromLogits(lastLogits, config.temperature, config.topK);
    
    inputTensor.dispose();
    logits.dispose();
    lastLogits.dispose();

    tokenIds.push(nextId);
    const nextChar = tokenizer.decode([nextId]);
    resultText += nextChar;

    if (onStep) onStep(nextChar);

    // Yield to UI loop briefly to allow rendering
    await new Promise(r => setTimeout(r, 0));
  }

  return resultText;
}
