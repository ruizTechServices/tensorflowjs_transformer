
import React, { useState, useEffect, useCallback, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import { initBackend } from './services/tf-utils';
import { SimpleTokenizer, DEFAULT_CORPUS } from './services/tokenizer';
import { TransformerModel } from './services/transformer';
import { generateText } from './services/generation';
import { trainModel } from './services/training';
import { exportModelState, importModelState, importTokenizer, SavedModelState } from './services/persistence';
import { downloadJson, pickJsonFile } from './utils/file';
import { MODEL_PRESETS, ModelSize, buildModelConfig } from './config/modelConfig';
import MatrixVisualizer from './components/MatrixVisualizer';
import TerminalLog from './components/TerminalLog';
import { RunStatus, TrainingConfig, GenerationConfig } from './types';

// Default Configs
const DEFAULT_TRAINING_CONFIG: TrainingConfig = {
  batchSize: 4,
  epochs: 10,
  learningRate: 0.005,
  seqLen: 32,
};

const DEFAULT_GEN_CONFIG: GenerationConfig = {
  maxNewTokens: 100,
  temperature: 0.8,
  topK: 5,
};

const TECH_CORPUS = `The sky above the port was the color of television, tuned to a dead channel. It's not like I'm using, Case heard someone say, as he shouldered his way through the crowd around the door of the Chat. It's like my body's developed this massive drug deficiency.`;

const App: React.FC = () => {
  // System State
  const [status, setStatus] = useState<RunStatus>(RunStatus.INITIALIZING);
  const [logs, setLogs] = useState<string[]>(['Initializing system...', 'Loading TensorFlow.js backend...']);
  const [activeTab, setActiveTab] = useState<'interact' | 'train'>('interact');
  
  // Model Configuration
  const [modelSize, setModelSize] = useState<ModelSize>('tiny');
  
  // Interactive State
  const [inputSequence, setInputSequence] = useState<string>('The sky');
  const [generatedOutput, setGeneratedOutput] = useState<string>('');
  const [attentionData, setAttentionData] = useState<number[][][]>([]); 
  const [activeHead, setActiveHead] = useState<number>(0);
  
  // Training State
  const [trainingCorpus, setTrainingCorpus] = useState<string>(TECH_CORPUS);
  const [trainConfig, setTrainConfig] = useState<TrainingConfig>(DEFAULT_TRAINING_CONFIG);
  const [genConfig, setGenConfig] = useState<GenerationConfig>(DEFAULT_GEN_CONFIG);

  // Refs
  const tokenizerRef = useRef<SimpleTokenizer | null>(null);
  const modelRef = useRef<TransformerModel | null>(null);

  const addLog = (msg: string) => setLogs(prev => [...prev, msg]);

  const initSystem = useCallback(async () => {
    try {
      await initBackend();
      addLog('Backend loaded: WebGL/CPU ready.');
      
      addLog('Building Tokenizer...');
      const tokenizer = new SimpleTokenizer(DEFAULT_CORPUS + TECH_CORPUS); 
      tokenizerRef.current = tokenizer;
      addLog(`Tokenizer ready. Vocab size: ${tokenizer.getVocabSize()}`);

      addLog(`Initializing [${modelSize.toUpperCase()}] Transformer...`);
      const modelConfig = buildModelConfig(modelSize, tokenizer.getVocabSize());
      
      // Dispose previous model if exists
      if (modelRef.current) {
        // Ideally we dispose old tensors, but JS GC handles variables attached to class usually
      }

      const model = new TransformerModel(modelConfig);
      modelRef.current = model;
      
      addLog('Weights initialized (Random Normal). System Ready.');
      setStatus(RunStatus.READY);
    } catch (e) {
      console.error(e);
      addLog(`CRITICAL ERROR: ${(e as Error).message}`);
    }
  }, [modelSize]);

  // Initial Load
  useEffect(() => {
    initSystem();
  }, [initSystem]);

  // Handlers
  const handleGenerate = async () => {
    if (!modelRef.current || !tokenizerRef.current) return;
    
    try {
      setStatus(RunStatus.GENERATING);
      addLog(`Generating text... Temp: ${genConfig.temperature}`);
      setGeneratedOutput('');

      // Run visualization on the prompt first
      const promptIds = tokenizerRef.current.encode(inputSequence);
      const { attentionWeights } = modelRef.current.predict(promptIds);
      if (attentionWeights && attentionWeights.length > 0) {
        setAttentionData(attentionWeights[0]); 
      }

      // Run generation loop
      const text = await generateText(
        modelRef.current,
        tokenizerRef.current,
        inputSequence,
        genConfig,
        (char) => setGeneratedOutput(prev => prev + char)
      );

      addLog('Generation complete.');
    } catch (e) {
      addLog(`Generation Error: ${(e as Error).message}`);
    } finally {
      setStatus(RunStatus.READY);
    }
  };

  const handleTrain = async () => {
    if (!modelRef.current || !tokenizerRef.current) return;
    
    try {
      setStatus(RunStatus.TRAINING);
      addLog('Starting training session...');
      
      await trainModel(
        modelRef.current,
        tokenizerRef.current,
        trainingCorpus,
        trainConfig,
        (msg) => addLog(msg)
      );
    } catch (e) {
      addLog(`Training Error: ${(e as Error).message}`);
    } finally {
      setStatus(RunStatus.READY);
    }
  };

  const handleSave = async () => {
    if (!modelRef.current || !tokenizerRef.current) return;
    addLog('Serializing model state...');
    try {
      const state = await exportModelState(modelRef.current, tokenizerRef.current);
      downloadJson(`transformer_${modelSize}_${Date.now()}.json`, state);
      addLog('Model state downloaded.');
    } catch (e) {
      addLog(`Save Error: ${(e as Error).message}`);
    }
  };

  const handleLoad = async () => {
    try {
      addLog('Waiting for file selection...');
      const jsonStr = await pickJsonFile();
      const state = JSON.parse(jsonStr) as SavedModelState;
      
      addLog(`Loading state from ${new Date(state.timestamp).toLocaleTimeString()}...`);
      
      // Reconstruct Tokenizer
      tokenizerRef.current = importTokenizer(state.tokenizer);
      addLog(`Tokenizer restored. Vocab: ${tokenizerRef.current.getVocabSize()}`);

      // Reconstruct Model
      // We must match the config of the saved model
      const model = new TransformerModel(state.config);
      await importModelState(model, state);
      modelRef.current = model;
      
      // Update UI model size to match loaded model (visual only)
      // setModelSize would trigger re-init, so we might skip it or handle carefully
      // For now, just log it.
      addLog(`Restored config: ${state.config.dModel} dModel / ${state.config.numLayers} Layers`);
      
      addLog('Weights restored successfully.');
      setStatus(RunStatus.READY);
    } catch (e) {
      addLog(`Load Error: ${(e as Error).message}`);
    }
  };

  const reset = () => {
    setGeneratedOutput('');
    setAttentionData([]);
    addLog('Visual context cleared.');
  };

  const fullTokens = tokenizerRef.current 
    ? [...inputSequence.split('')] 
    : [];

  return (
    <div className="min-h-screen bg-cyber-black text-cyber-text p-4 md:p-8 font-sans selection:bg-cyber-purple selection:text-white">
      <header className="mb-8 flex flex-col md:flex-row justify-between items-end border-b border-cyber-gray pb-4 gap-4">
        <div>
          <h1 className="text-3xl md:text-4xl font-bold tracking-tighter text-white mb-1">
            TRANSFORMER<span className="text-cyber-accent">.JS</span>
          </h1>
          <div className="flex items-center gap-4">
             <p className="text-cyber-purple font-mono text-xs uppercase tracking-widest">
              Browser-Based Deep Learning
            </p>
            <select 
              value={modelSize}
              onChange={(e) => setModelSize(e.target.value as ModelSize)}
              className="bg-cyber-black border border-cyber-gray text-xs font-mono text-cyber-accent px-2 py-1 rounded focus:outline-none hover:border-cyber-accent cursor-pointer"
            >
              {Object.entries(MODEL_PRESETS).map(([key, cfg]) => (
                <option key={key} value={key}>{cfg.name}</option>
              ))}
            </select>
          </div>
        </div>
        <div className="flex flex-col items-end gap-2">
           <div className="flex gap-2">
             <button onClick={handleSave} className="text-[10px] font-mono border border-cyber-gray px-2 py-1 rounded text-gray-400 hover:text-white hover:border-white transition-colors">SAVE STATE</button>
             <button onClick={handleLoad} className="text-[10px] font-mono border border-cyber-gray px-2 py-1 rounded text-gray-400 hover:text-white hover:border-white transition-colors">LOAD STATE</button>
           </div>
          <div className="flex gap-2">
            <button 
              onClick={() => setActiveTab('interact')}
              className={`px-4 py-1 font-mono text-xs border rounded ${activeTab === 'interact' ? 'border-cyber-accent text-cyber-accent bg-cyber-accent/10' : 'border-cyber-gray text-gray-500 hover:text-white'}`}
            >
              INTERACT
            </button>
            <button 
              onClick={() => setActiveTab('train')}
              className={`px-4 py-1 font-mono text-xs border rounded ${activeTab === 'train' ? 'border-cyber-accent text-cyber-accent bg-cyber-accent/10' : 'border-cyber-gray text-gray-500 hover:text-white'}`}
            >
              TRAIN
            </button>
          </div>
          <div className={`text-sm font-bold font-mono ${status !== RunStatus.READY ? 'text-cyber-accent animate-pulse' : 'text-cyber-success'}`}>
            STATUS: {status}
          </div>
        </div>
      </header>

      <main className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* LEFT PANEL: INPUT & CONTROLS */}
        <div className="lg:col-span-4 flex flex-col gap-6">
          
          {activeTab === 'interact' ? (
            /* INTERACTION PANEL */
            <div className="bg-cyber-dark border border-cyber-gray rounded-xl p-6 shadow-lg shadow-cyber-black/50 animate-fade-in">
              <div className="mb-4">
                <label className="block text-xs font-mono text-cyber-accent mb-2 uppercase">Input Prompt</label>
                <input 
                  type="text" 
                  value={inputSequence}
                  onChange={(e) => setInputSequence(e.target.value)}
                  className="w-full bg-cyber-black border border-cyber-gray rounded p-3 text-white font-mono focus:outline-none focus:border-cyber-accent focus:ring-1 focus:ring-cyber-accent transition-all text-sm"
                />
              </div>
              
              <div className="grid grid-cols-2 gap-4 mb-4">
                <div>
                  <label className="block text-xs font-mono text-gray-500 mb-1">Temp</label>
                  <input 
                    type="number" step="0.1"
                    value={genConfig.temperature}
                    onChange={(e) => setGenConfig({...genConfig, temperature: parseFloat(e.target.value)})}
                    className="w-full bg-cyber-black border border-cyber-gray rounded p-2 text-xs font-mono text-white"
                  />
                </div>
                <div>
                  <label className="block text-xs font-mono text-gray-500 mb-1">Max Tokens</label>
                  <input 
                    type="number"
                    value={genConfig.maxNewTokens}
                    onChange={(e) => setGenConfig({...genConfig, maxNewTokens: parseInt(e.target.value)})}
                    className="w-full bg-cyber-black border border-cyber-gray rounded p-2 text-xs font-mono text-white"
                  />
                </div>
              </div>

              <div className="flex gap-3">
                <button 
                  onClick={handleGenerate}
                  disabled={status !== RunStatus.READY}
                  className="flex-1 bg-cyber-accent/10 hover:bg-cyber-accent/20 text-cyber-accent border border-cyber-accent/50 font-bold py-3 px-4 rounded transition-all uppercase tracking-wider font-mono text-xs disabled:opacity-50 disabled:cursor-not-allowed flex justify-center items-center gap-2"
                >
                   GENERATE
                </button>
                <button 
                  onClick={reset}
                  className="px-4 border border-cyber-gray text-gray-400 hover:text-white hover:border-white rounded transition-all font-mono text-xs"
                >
                  CLR
                </button>
              </div>
            </div>
          ) : (
            /* TRAINING PANEL */
            <div className="bg-cyber-dark border border-cyber-gray rounded-xl p-6 shadow-lg shadow-cyber-black/50 animate-fade-in">
              <div className="mb-4">
                <label className="block text-xs font-mono text-cyber-purple mb-2 uppercase">Training Corpus</label>
                <textarea 
                  value={trainingCorpus}
                  onChange={(e) => setTrainingCorpus(e.target.value)}
                  className="w-full h-32 bg-cyber-black border border-cyber-gray rounded p-3 text-white font-mono text-xs focus:outline-none focus:border-cyber-purple focus:ring-1 focus:ring-cyber-purple transition-all resize-none"
                />
              </div>

              <div className="grid grid-cols-2 gap-4 mb-4">
                <div>
                  <label className="block text-xs font-mono text-gray-500 mb-1">Epochs</label>
                  <input 
                    type="number"
                    value={trainConfig.epochs}
                    onChange={(e) => setTrainConfig({...trainConfig, epochs: parseInt(e.target.value)})}
                    className="w-full bg-cyber-black border border-cyber-gray rounded p-2 text-xs font-mono text-white"
                  />
                </div>
                <div>
                  <label className="block text-xs font-mono text-gray-500 mb-1">LR</label>
                  <input 
                    type="number" step="0.001"
                    value={trainConfig.learningRate}
                    onChange={(e) => setTrainConfig({...trainConfig, learningRate: parseFloat(e.target.value)})}
                    className="w-full bg-cyber-black border border-cyber-gray rounded p-2 text-xs font-mono text-white"
                  />
                </div>
              </div>

              <button 
                onClick={handleTrain}
                disabled={status !== RunStatus.READY}
                className="w-full bg-cyber-purple/10 hover:bg-cyber-purple/20 text-cyber-purple border border-cyber-purple/50 font-bold py-3 px-4 rounded transition-all uppercase tracking-wider font-mono text-xs disabled:opacity-50 disabled:cursor-not-allowed flex justify-center items-center gap-2"
              >
                 START TRAINING
              </button>
            </div>
          )}

          <TerminalLog logs={logs} />
        </div>

        {/* RIGHT PANEL: VISUALIZATION */}
        <div className="lg:col-span-8 flex flex-col gap-6">
          
          {/* Output Area */}
          <div className="bg-cyber-dark border border-cyber-gray rounded-xl p-6 relative overflow-hidden min-h-[160px]">
            <div className="absolute top-0 left-0 w-1 h-full bg-cyber-accent"></div>
            <h3 className="text-xs font-mono text-cyber-accent mb-2 uppercase">
              {status === RunStatus.TRAINING ? 'Training Output' : 'Sequence Output'}
            </h3>
            
            {status === RunStatus.TRAINING ? (
               <div className="flex items-center justify-center h-full text-cyber-purple animate-pulse font-mono text-sm">
                 OPTIMIZING NEURAL WEIGHTS...
               </div>
            ) : (
              <div className="font-mono text-lg leading-relaxed break-words">
                <span className="text-gray-500">{inputSequence}</span>
                <span className="text-cyber-text">{generatedOutput}</span>
                <span className="inline-block w-2 h-5 bg-cyber-accent ml-1 animate-pulse align-middle"></span>
              </div>
            )}
          </div>

          {/* Attention Visualization */}
          <div className="bg-cyber-dark border border-cyber-gray rounded-xl p-6 flex-1 min-h-[400px] flex flex-col">
            <div className="flex justify-between items-center mb-6">
              <h3 className="text-xs font-mono text-cyber-accent uppercase flex items-center gap-2">
                <span className={`w-2 h-2 rounded-full bg-cyber-accent ${status !== RunStatus.READY ? 'animate-pulse' : ''}`}></span>
                Attention Heatmap (Last Block)
              </h3>
              
              <div className="flex gap-1">
                {Array.from({ length: modelRef.current?.config.numHeads || 4 }).map((_, i) => (
                  <button
                    key={i}
                    onClick={() => setActiveHead(i)}
                    className={`h-6 w-6 flex items-center justify-center text-[10px] font-mono rounded border ${activeHead === i ? 'border-cyber-accent text-cyber-black bg-cyber-accent' : 'border-cyber-gray text-gray-500 hover:border-gray-400'}`}
                  >
                    {i}
                  </button>
                ))}
              </div>
            </div>

            <div className="flex-1 flex items-center justify-center bg-cyber-black/30 rounded-lg border border-dashed border-cyber-gray/50 p-4 overflow-auto">
              {attentionData.length > 0 ? (
                <MatrixVisualizer 
                  data={attentionData} 
                  tokens={fullTokens} 
                  headIndex={activeHead} 
                />
              ) : (
                <div className="text-center opacity-50">
                  <p className="text-gray-600 font-mono text-sm mb-2">NO TENSOR DATA</p>
                </div>
              )}
            </div>
          </div>

        </div>
      </main>
    </div>
  );
};

export default App;
