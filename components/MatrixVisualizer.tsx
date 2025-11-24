import React from 'react';

interface MatrixVisualizerProps {
  data: number[][][];
  tokens: string[];
  headIndex: number;
}

const MatrixVisualizer: React.FC<MatrixVisualizerProps> = ({ data, tokens, headIndex }) => {
  if (!data || data.length === 0) return (
    <div className="flex items-center justify-center h-64 border border-cyber-gray rounded-lg text-gray-600 font-mono text-sm">
      NO DATA DETECTED // WAITING FOR SIGNAL
    </div>
  );

  // Data shape: [num_heads, seq_len, seq_len]
  // We render the specific head
  const matrix = data[headIndex]; 
  if(!matrix) return null;

  return (
    <div className="flex flex-col gap-4 animate-fade-in">
      <div className="flex justify-between items-end">
        <h3 className="text-cyber-accent font-mono text-sm font-bold tracking-wider">
          ATTENTION_HEAD_{headIndex}
        </h3>
        <span className="text-xs text-gray-500 font-mono">
          {matrix.length}x{matrix.length} TENSOR
        </span>
      </div>

      <div className="relative overflow-x-auto border border-cyber-gray rounded-lg p-4 bg-cyber-black/50 backdrop-blur-sm">
        <div 
          className="grid gap-px bg-cyber-gray/30"
          style={{
            gridTemplateColumns: `auto repeat(${tokens.length}, minmax(2rem, 1fr))`,
          }}
        >
          {/* Header Row */}
          <div className="h-8 w-8"></div>
          {tokens.map((tok, i) => (
            <div key={`col-${i}`} className="flex items-center justify-center font-mono text-xs text-gray-400 h-8 rotate-45 origin-bottom-left">
              {tok === ' ' ? '␣' : tok}
            </div>
          ))}

          {/* Rows */}
          {matrix.map((row, i) => (
            <React.Fragment key={`row-${i}`}>
              {/* Row Label */}
              <div className="flex items-center justify-end pr-2 font-mono text-xs text-gray-400">
                {tokens[i] === ' ' ? '␣' : tokens[i]}
              </div>
              
              {/* Cells */}
              {row.map((val, j) => {
                // Normalize opacity
                const opacity = Math.min(Math.max(val, 0), 1);
                return (
                  <div
                    key={`cell-${i}-${j}`}
                    className="aspect-square w-full transition-all duration-300 hover:scale-110 z-10"
                    style={{
                      backgroundColor: `rgba(0, 243, 255, ${opacity})`,
                      boxShadow: opacity > 0.5 ? `0 0 ${opacity * 10}px #00f3ff` : 'none'
                    }}
                    title={`Attention: ${val.toFixed(4)}`}
                  />
                );
              })}
            </React.Fragment>
          ))}
        </div>
      </div>
    </div>
  );
};

export default MatrixVisualizer;