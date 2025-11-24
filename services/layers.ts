
import * as tf from '@tensorflow/tfjs';
import { gelu, dense } from './tf-utils';

/**
 * 1. Token Embedding
 * Maps integer tokens to dModel-dimensional vectors.
 */
export class TokenEmbedding {
  vocabSize: number;
  dModel: number;
  embedding: tf.Variable;

  constructor(vocabSize: number, dModel: number) {
    this.vocabSize = vocabSize;
    this.dModel = dModel;
    this.embedding = tf.variable(tf.randomNormal([vocabSize, dModel], 0, 0.02));
  }

  forward(tokenIds: tf.Tensor): tf.Tensor {
    return tf.gather(this.embedding, tokenIds);
  }
}

/**
 * 2. Positional Encoding
 * Adds sinusoidal signals to embeddings to give order information.
 * Calculated on CPU to avoid WebGL shader precision/compilation issues.
 */
export const getPositionalEncoding = (maxLen: number, dModel: number): tf.Tensor => {
  return tf.tidy(() => {
    const buffer = tf.buffer([maxLen, dModel]);
    for (let pos = 0; pos < maxLen; pos++) {
      for (let i = 0; i < dModel; i++) {
        // PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        // PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        const angle = pos / Math.pow(10000, (2 * Math.floor(i / 2)) / dModel);
        if (i % 2 === 0) {
          buffer.set(Math.sin(angle), pos, i);
        } else {
          buffer.set(Math.cos(angle), pos, i);
        }
      }
    }
    return buffer.toTensor();
  });
};

/**
 * 3. Scaled Dot-Product Attention (The Soul)
 */
export const scaledDotProductAttention = (q: tf.Tensor, k: tf.Tensor, v: tf.Tensor, mask?: tf.Tensor) => {
  return tf.tidy(() => {
    const dk = tf.scalar(k.shape[k.shape.length - 1] as number);
    
    // (batch, heads, seq_len_q, depth) * (batch, heads, depth, seq_len_k)
    // -> (batch, heads, seq_len_q, seq_len_k)
    let scores = tf.matMul(q, k, false, true);
    scores = tf.div(scores, tf.sqrt(dk));

    if (mask) {
      // mask is 0 for keep, 1 for discard. Multiply by -1e9
      // We expand dims to match heads: [seq, seq] -> [1, 1, seq, seq]
      const maskExpanded = mask.expandDims(0).expandDims(0);
      scores = tf.add(scores, tf.mul(maskExpanded, -1e9));
    }

    const attentionWeights = tf.softmax(scores); // axis -1 default
    const output = tf.matMul(attentionWeights, v);

    return { output, attentionWeights };
  });
};

/**
 * 4. Multi-Head Attention Class
 */
export class MultiHeadAttention {
  dModel: number;
  numHeads: number;
  depth: number;
  wq: tf.Variable;
  wk: tf.Variable;
  wv: tf.Variable;
  wo: tf.Variable;

  constructor(dModel: number, numHeads: number) {
    this.dModel = dModel;
    this.numHeads = numHeads;
    this.depth = Math.floor(dModel / numHeads);

    this.wq = tf.variable(tf.randomNormal([dModel, dModel], 0, 0.02));
    this.wk = tf.variable(tf.randomNormal([dModel, dModel], 0, 0.02));
    this.wv = tf.variable(tf.randomNormal([dModel, dModel], 0, 0.02));
    this.wo = tf.variable(tf.randomNormal([dModel, dModel], 0, 0.02));
  }

  splitHeads(x: tf.Tensor, batchSize: number) {
    // Split the last dimension into (numHeads, depth)
    // Transpose to (batch, numHeads, seqLen, depth)
    const reshaped = tf.reshape(x, [batchSize, -1, this.numHeads, this.depth]);
    return tf.transpose(reshaped, [0, 2, 1, 3]);
  }

  forward(v: tf.Tensor, k: tf.Tensor, q: tf.Tensor, mask?: tf.Tensor) {
    // No tf.tidy here to preserve gradient path for Variables
    const batchSize = q.shape[0] as number;

    // Use safe 'dense' op instead of raw matMul to avoid BatchMatMul gradient errors
    const Q = this.splitHeads(dense(q, this.wq), batchSize);
    const K = this.splitHeads(dense(k, this.wk), batchSize);
    const V = this.splitHeads(dense(v, this.wv), batchSize);

    const { output, attentionWeights } = scaledDotProductAttention(Q, K, V, mask);

    // Transpose and concat back
    // (batch, numHeads, seqLen, depth) -> (batch, seqLen, numHeads, depth)
    const transposed = tf.transpose(output, [0, 2, 1, 3]);
    const concat = tf.reshape(transposed, [batchSize, -1, this.dModel]);

    // Final linear layer
    const finalOutput = dense(concat, this.wo);

    return { finalOutput, attentionWeights };
  }
}

/**
 * 5. Pointwise Feed Forward Network
 */
export class FeedForward {
  w1: tf.Variable;
  w2: tf.Variable;

  constructor(dModel: number, dff: number) {
    this.w1 = tf.variable(tf.randomNormal([dModel, dff], 0, 0.02));
    this.w2 = tf.variable(tf.randomNormal([dff, dModel], 0, 0.02));
  }

  forward(x: tf.Tensor) {
    // Linear -> GELU -> Linear
    // Use safe dense op
    const layer1 = gelu(dense(x, this.w1));
    return dense(layer1, this.w2);
  }
}

/**
 * 6. Layer Normalization
 */
export class LayerNormalization {
  gamma: tf.Variable;
  beta: tf.Variable;
  epsilon: number = 1e-6;

  constructor(dModel: number) {
    this.gamma = tf.variable(tf.ones([dModel]));
    this.beta = tf.variable(tf.zeros([dModel]));
  }

  forward(x: tf.Tensor) {
    return tf.tidy(() => {
      const mean = tf.mean(x, -1, true);
      const variance = tf.mean(tf.square(tf.sub(x, mean)), -1, true);
      
      const normalized = tf.div(tf.sub(x, mean), tf.sqrt(tf.add(variance, tf.scalar(this.epsilon))));
      
      return tf.add(tf.mul(normalized, this.gamma), this.beta);
    });
  }
}
