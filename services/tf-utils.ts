
import * as tf from '@tensorflow/tfjs';

// Set backend to WebGL for performance, fallback to CPU
export const initBackend = async () => {
  await tf.ready();
  // If WebGL is available, use it.
  if (tf.findBackend('webgl')) {
    await tf.setBackend('webgl');
  } else {
    await tf.setBackend('cpu');
  }
  console.log(`Backend set to: ${tf.getBackend()}`);
};

// Create a look-ahead mask (upper triangular matrix with 1s)
// Used to mask out future tokens. 
// Returns a matrix where Future positions are 1 and Past/Present are 0.
// Calculated on CPU for stability.
export const createLookAheadMask = (size: number) => {
  return tf.tidy(() => {
    const buffer = tf.buffer([size, size]);
    for (let i = 0; i < size; i++) {
      for (let j = 0; j < size; j++) {
        // i is current position, j is attended position.
        // We want to mask out cases where j > i (future)
        if (j > i) {
          buffer.set(1, i, j);
        } else {
          buffer.set(0, i, j);
        }
      }
    }
    // Force to float to prevent dtype errors in backend operations
    return buffer.toTensor().toFloat();
  });
};

export const gelu = (x: tf.Tensor) => {
  return tf.tidy(() => {
    const c = tf.scalar(0.044715);
    const sqrt2OverPi = tf.scalar(Math.sqrt(2 / Math.PI));
    
    const fit = tf.add(
      x,
      tf.mul(c, tf.pow(x, 3))
    );
    
    return tf.mul(
      tf.mul(tf.scalar(0.5), x),
      tf.add(tf.scalar(1), tf.tanh(tf.mul(sqrt2OverPi, fit)))
    );
  });
};

/**
 * Safe Dense Layer operation that handles rank-3 inputs (Batch, Seq, Dim)
 * by flattening to rank-2 for the MatMul, then reshaping back.
 * This avoids "Error in gradient for op BatchMatMul" in TFJS.
 */
export const dense = (x: tf.Tensor, w: tf.Tensor | tf.Variable) => {
  // We don't use tf.tidy here because it might be called inside a training loop 
  // where intermediate tensors need to be tracked by the gradient tape.
  
  const inputShape = x.shape;
  
  // If input is 3D or higher [Batch, Seq, Dim] and Weight is 2D [Dim, Out]
  if (x.rank > 2) {
    const inputDim = w.shape[0];
    // Flatten: [Batch * Seq, Dim]
    const xFlat = tf.reshape(x, [-1, inputDim]);
    
    // MatMul: [Batch * Seq, Out]
    const outFlat = tf.matMul(xFlat, w);
    
    // Reconstruct: [Batch, Seq, Out]
    const outputDim = w.shape[1];
    const newShape = [...inputShape.slice(0, -1), outputDim];
    
    return tf.reshape(outFlat, newShape);
  }
  
  // Standard 2D case
  return tf.matMul(x, w);
};
