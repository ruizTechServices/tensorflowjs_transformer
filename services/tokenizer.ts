
export interface SerializedTokenizer {
  charToId: Record<string, number>;
  idToChar: Record<number, string>;
  vocabSize: number;
}

export class SimpleTokenizer {
  vocab: Map<string, number>;
  inverseVocab: Map<number, string>;
  vocabSize: number;

  // Constructor can now take a string (corpus) OR serialized data
  constructor(input: string | SerializedTokenizer) {
    this.vocab = new Map();
    this.inverseVocab = new Map();
    this.vocabSize = 0;

    if (typeof input === 'string') {
      // Initialize from Corpus
      const chars = Array.from(new Set(input.split(''))).sort();
      
      // Reserve 0 for padding, 1 for unknown
      this.vocab.set('<PAD>', 0);
      this.vocab.set('<UNK>', 1);
      this.inverseVocab.set(0, '<PAD>');
      this.inverseVocab.set(1, '<UNK>');

      let idx = 2;
      chars.forEach(char => {
        this.vocab.set(char, idx);
        this.inverseVocab.set(idx, char);
        idx++;
      });
      
      this.vocabSize = idx;
    } else {
      // Hydrate from Serialized Data
      if (input && input.charToId && input.idToChar) {
        Object.entries(input.charToId).forEach(([char, id]) => {
          this.vocab.set(char, id);
        });
        Object.entries(input.idToChar).forEach(([id, char]) => {
          this.inverseVocab.set(Number(id), char);
        });
        this.vocabSize = input.vocabSize;
      }
    }
  }

  encode(text: string): number[] {
    return text.split('').map(char => this.vocab.get(char) || 1);
  }

  decode(ids: number[]): string {
    return ids.map(id => this.inverseVocab.get(id) || '').join('');
  }

  getVocabSize(): number {
    return this.vocabSize;
  }
}

export const DEFAULT_CORPUS = "The quick brown fox jumps over the lazy dog. Transformers are cool. TensorFlow is fun.";