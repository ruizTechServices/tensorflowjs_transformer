
import * as tf from '@tensorflow/tfjs';
import { TokenEmbedding, getPositionalEncoding, MultiHeadAttention, FeedForward, LayerNormalization } from './layers';
import { createLookAheadMask, dense } from './tf-utils';
import { ModelConfig } from '../types';

/**
 * Encoder Block: 
 * x -> LayerNorm(x + MHA(x)) -> LayerNorm(x + FFN(x))
 */
export class EncoderBlock {
  mha: MultiHeadAttention;
  ffn: FeedForward;
  layernorm1: LayerNormalization;
  layernorm2: LayerNormalization;

  constructor(dModel: number, numHeads: number, dff: number) {
    this.mha = new MultiHeadAttention(dModel, numHeads);
    this.ffn = new FeedForward(dModel, dff);
    this.layernorm1 = new LayerNormalization(dModel);
    this.layernorm2 = new LayerNormalization(dModel);
  }

  forward(x: tf.Tensor, training: boolean, mask?: tf.Tensor) {
    // Note: We don't use tf.tidy here because this is called inside the main model's tf.tidy or training loop
    // 1. MHA + Residual
    const { finalOutput: attnOutput, attentionWeights } = this.mha.forward(x, x, x, mask);
    const out1 = this.layernorm1.forward(tf.add(x, attnOutput));

    // 2. FFN + Residual
    const ffnOutput = this.ffn.forward(out1);
    const out2 = this.layernorm2.forward(tf.add(out1, ffnOutput));

    return { output: out2, attentionWeights };
  }
}

/**
 * The MiniGPT (Encoder-Only for simplicity in this demo, acting as a generator)
 */
export class TransformerModel {
  embedding: TokenEmbedding;
  encoderBlocks: EncoderBlock[];
  finalLayer: tf.Variable; // Projection to vocab
  config: ModelConfig;

  constructor(config: ModelConfig) {
    this.config = config;
    this.embedding = new TokenEmbedding(config.vocabSize, config.dModel);
    this.encoderBlocks = [];
    
    for (let i = 0; i < config.numLayers; i++) {
      this.encoderBlocks.push(new EncoderBlock(config.dModel, config.numHeads, config.dff));
    }

    // Final projection back to vocab size
    this.finalLayer = tf.variable(tf.randomNormal([config.dModel, config.vocabSize]));
  }

  /**
   * Main Forward Pass
   * @param inputIds Tensor of shape [batch_size, seq_len]
   * @param training boolean
   * @returns { logits: Tensor, attentionWeights: Tensor }
   */
  forward(inputIds: tf.Tensor, training: boolean = false) {
    // We expect the caller (training loop or predict function) to handle tf.tidy if needed
    
    const seqLen = inputIds.shape[1] as number;

    // 1. Embeddings + Positional
    let x = this.embedding.forward(inputIds);
    x = tf.mul(x, tf.scalar(Math.sqrt(this.config.dModel)));
    
    // Create pos encoding: [seqLen, dModel] -> broadcast to [batch, seqLen, dModel]
    const posEncoding = getPositionalEncoding(seqLen, this.config.dModel);
    x = tf.add(x, posEncoding);

    // 2. Create Lookahead Mask (Critical for GPT)
    // shape: [seqLen, seqLen]
    const mask = createLookAheadMask(seqLen);

    // 3. Encoder Blocks
    let lastAttnWeights = null;
    
    for (const block of this.encoderBlocks) {
      const result = block.forward(x, training, mask);
      x = result.output;
      lastAttnWeights = result.attentionWeights;
    }

    // 4. Final Projection
    // Use safe dense op to avoid BatchMatMul gradient errors
    const logits = dense(x, this.finalLayer);
    
    return { logits, attentionWeights: lastAttnWeights };
  }

  /**
   * Helper for single sequence inference (used in UI)
   */
  predict(tokenIds: number[]) {
    return tf.tidy(() => {
      if (tokenIds.length === 0) return { predictedId: 0, attentionWeights: [] };

      const tensorIds = tf.tensor2d([tokenIds], [1, tokenIds.length], 'int32');
      const { logits, attentionWeights } = this.forward(tensorIds, false);
      
      // Get logits for the last token
      const lastTokenLogits = logits.slice([0, tokenIds.length - 1, 0], [1, 1, -1]).squeeze();
      
      const predictedId = lastTokenLogits.argMax().dataSync()[0];
      const attnData = attentionWeights ? attentionWeights.arraySync() as number[][][][] : [];

      return { predictedId, attentionWeights: attnData };
    });
  }

  /**
   * Retrieves all trainable variables in a flattened map for serialization.
   */
  getTrainableVariables(): { [name: string]: tf.Variable } {
    const vars: { [name: string]: tf.Variable } = {};
    
    // Embeddings
    vars['emb_embedding'] = this.embedding.embedding;
    
    // Blocks
    this.encoderBlocks.forEach((block, i) => {
      const prefix = `blk${i}`;
      
      // MHA
      vars[`${prefix}_mha_wq`] = block.mha.wq;
      vars[`${prefix}_mha_wk`] = block.mha.wk;
      vars[`${prefix}_mha_wv`] = block.mha.wv;
      vars[`${prefix}_mha_wo`] = block.mha.wo;
      
      // FFN
      vars[`${prefix}_ffn_w1`] = block.ffn.w1;
      vars[`${prefix}_ffn_w2`] = block.ffn.w2;
      
      // LN
      vars[`${prefix}_ln1_g`] = block.layernorm1.gamma;
      vars[`${prefix}_ln1_b`] = block.layernorm1.beta;
      vars[`${prefix}_ln2_g`] = block.layernorm2.gamma;
      vars[`${prefix}_ln2_b`] = block.layernorm2.beta;
    });

    // Final
    vars['final_proj'] = this.finalLayer;

    return vars;
  }
}
