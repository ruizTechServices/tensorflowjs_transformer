import React, { useEffect, useRef } from 'react';

interface TerminalLogProps {
  logs: string[];
}

const TerminalLog: React.FC<TerminalLogProps> = ({ logs }) => {
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  return (
    <div className="bg-cyber-black border border-cyber-gray p-4 rounded-lg h-48 overflow-y-auto font-mono text-xs">
      {logs.map((log, i) => (
        <div key={i} className="mb-1">
          <span className="text-cyber-purple mr-2">➜</span>
          <span className="text-cyber-text">{log}</span>
        </div>
      ))}
      <div ref={endRef} />
    </div>
  );
};

export default TerminalLog;