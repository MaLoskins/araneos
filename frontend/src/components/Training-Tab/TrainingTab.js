import React, { useState, useEffect, useRef, useMemo } from 'react';
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels';
import { trainModel } from '../../api';
import MetricsVisualizer from './MetricsVisualizer';
import { Link } from 'react-router-dom';
import WorkflowStepper from '../layout/WorkflowStepper';
import { useGraphData, useGraphActions } from '../../context/GraphDataContext';

const STEPS = ['Review', 'Configure', 'Train', 'Results'];
const MODELS = [
  { value: 'GCN', label: 'GCN' },
  { value: 'ResidualGCN', label: 'Residual GCN' },
  { value: 'GraphSAGE', label: 'GraphSAGE' },
  { value: 'GAT', label: 'GAT' },
  { value: 'GIN', label: 'GIN' },
  { value: 'ChebConv', label: 'ChebConv' },
  { value: 'NaiveBayes', label: 'Naive Bayes' },
];

const RESULT_TABS = ['Metrics', 'Charts', 'Report'];

const TrainingTab = () => {
  const graphContext = useGraphData();
  const graphActions = useGraphActions();

  const graphStats = useMemo(() => {
    const nodes = graphContext.nodes || [];
    const edges = graphContext.edges || [];
    const labelNodes = nodes.filter(n => (n.label != null && n.label !== '') || (n.features?.label != null && n.features?.label !== ''));
    return {
      nodes: nodes.length,
      edges: edges.length,
      hasLabels: labelNodes.length > 0,
      uniqueLabels: [...new Set(labelNodes.map(n => n.label ?? n.features?.label))],
    };
  }, [graphContext.nodes, graphContext.edges]);

  const trainingData = useMemo(() => {
    if (!graphContext.nodes?.length || !graphContext.edges?.length) return null;
    return { nodes: graphContext.nodes, links: graphContext.edges };
  }, [graphContext.nodes, graphContext.edges]);

  const isValidForTraining = graphStats.nodes > 0 && graphStats.edges > 0 && graphStats.hasLabels;

  const [modelConfig, setModelConfig] = useState({
    model_name: 'GCN', hidden_channels: 64, learning_rate: 0.01,
    epochs: 200, dropout: 0.3, heads: 8, K: 3,
  });

  const [isTraining, setIsTraining] = useState(false);
  const [metrics, setMetrics] = useState(null);
  const [trainingError, setTrainingError] = useState(null);
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [resultTab, setResultTab] = useState(0);
  const trainingRequestRef = useRef(null);
  const logsContainerRef = useRef(null);
  const trainingLogs = useMemo(() => graphContext.trainingLogs || [], [graphContext.trainingLogs]);
  const isNaiveBayes = modelConfig.model_name === 'NaiveBayes';

  useEffect(() => {
    graphActions.clearTrainingLogs();
    setMetrics(null);
    setTrainingError(null);
    setCurrentEpoch(0);
  }, [modelConfig.model_name, graphActions]);

  useEffect(() => {
    if (logsContainerRef.current) logsContainerRef.current.scrollTop = logsContainerRef.current.scrollHeight;
  }, [trainingLogs]);

  const workflowStep = useMemo(() => {
    if (!trainingData) return 0;
    if (!isTraining && !metrics) return 1;
    if (isTraining) return 2;
    return 3;
  }, [trainingData, isTraining, metrics]);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    let parsed = value;
    if (['hidden_channels', 'epochs', 'heads', 'K'].includes(name)) {
      parsed = parseInt(value, 10);
      if (isNaN(parsed) || parsed <= 0) return;
    } else if (['learning_rate', 'dropout'].includes(name)) {
      parsed = parseFloat(value);
      if (isNaN(parsed) || parsed < 0) return;
      if (name === 'dropout' && parsed > 1) return;
    }
    setModelConfig(prev => ({ ...prev, [name]: parsed }));
  };

  const handleStartTraining = () => {
    if (!trainingData) return;
    graphActions.clearTrainingLogs();
    setMetrics(null);
    setTrainingError(null);
    setIsTraining(true);
    setCurrentEpoch(0);

    const configToSend = {
      model_name: modelConfig.model_name,
      hidden_channels: modelConfig.hidden_channels,
      lr: modelConfig.learning_rate,
      epochs: modelConfig.epochs,
      dropout: modelConfig.dropout,
      ...(modelConfig.model_name === 'GAT' && { extra_params: { heads: modelConfig.heads } }),
      ...(modelConfig.model_name === 'ChebConv' && { extra_params: { K: modelConfig.K } }),
    };

    const handleMessage = (message) => {
      if (message.status === 'started') {
        graphActions.appendTrainingLog({ type: 'log', message: message.message || 'Training started...', timestamp: new Date().toISOString() });
      } else if (message.epoch && !message.status) {
        setCurrentEpoch(message.epoch);
        graphActions.appendTrainingLog({
          type: 'log',
          message: `Epoch ${message.epoch}/${message.total_epochs} — loss: ${message.train_loss?.toFixed(4)} — val_loss: ${message.val_loss?.toFixed(4)} — val_acc: ${(message.val_accuracy * 100)?.toFixed(1)}%`,
          timestamp: new Date().toISOString(),
        });
      } else if (message.status === 'completed') {
        graphActions.appendTrainingLog({
          type: 'log',
          message: `Complete — test accuracy: ${(message.test_accuracy * 100)?.toFixed(2)}%`,
          timestamp: new Date().toISOString(),
        });
        setMetrics({ test_accuracy: message.test_accuracy, best_val_loss: message.best_val_loss, ...message });
        setIsTraining(false);
        setResultTab(0);
      }
    };

    try {
      trainingRequestRef.current = trainModel(trainingData, configToSend, handleMessage, (error) => {
        setTrainingError(error?.message || 'Training error');
        setIsTraining(false);
      });
    } catch (error) {
      setTrainingError(error?.message || 'Training error');
      setIsTraining(false);
    }
  };

  const handleStopTraining = () => {
    trainingRequestRef.current?.cancel?.();
    setIsTraining(false);
    graphActions.appendTrainingLog({ type: 'log', message: 'Training canceled', timestamp: new Date().toISOString() });
  };

  // No graph data
  if (!graphStats.nodes || !graphStats.edges) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
        <WorkflowStepper steps={STEPS} currentStep={0} />
        <div className="stats-panel-empty">
          <div style={{ textAlign: 'center' }}>
            <h3 style={{ color: 'var(--text-secondary)', marginBottom: 12 }}>No Graph Data</h3>
            <p style={{ color: 'var(--text-muted)', marginBottom: 16 }}>Build and process a graph first.</p>
            <Link to="/" className="btn btn-primary">Go to Graph Builder</Link>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <WorkflowStepper steps={STEPS} currentStep={workflowStep} />

      <PanelGroup direction="horizontal" style={{ flex: 1 }}>
        {/* Left: Config */}
        <Panel defaultSize={22} minSize={18} maxSize={35}>
          <div className="panel training-config">
            <div className="panel-header"><h3>Configuration</h3></div>
            <div className="panel-body">
              <div className="graph-mini-stats">
                <div className="mini-stat">
                  <div className="mini-stat-value">{graphStats.nodes}</div>
                  <div className="mini-stat-label">Nodes</div>
                </div>
                <div className="mini-stat">
                  <div className="mini-stat-value">{graphStats.edges}</div>
                  <div className="mini-stat-label">Edges</div>
                </div>
                <div className="mini-stat">
                  <div className="mini-stat-value">{graphStats.hasLabels ? graphStats.uniqueLabels.length : '—'}</div>
                  <div className="mini-stat-label">Classes</div>
                </div>
              </div>

              <div className="form-group">
                <label>Model:
                  <select value={modelConfig.model_name} onChange={e => setModelConfig(prev => ({ ...prev, model_name: e.target.value }))}>
                    {MODELS.map(m => <option key={m.value} value={m.value}>{m.label}</option>)}
                  </select>
                </label>
              </div>

              <div className="form-section">
                <h4>Hyperparameters</h4>
                <div className="form-group">
                  <label>Hidden Channels: <input type="number" name="hidden_channels" value={modelConfig.hidden_channels} onChange={handleInputChange} min="8" max="256" disabled={isNaiveBayes} /></label>
                </div>
                <div className="form-group">
                  <label>Learning Rate: <input type="number" name="learning_rate" value={modelConfig.learning_rate} onChange={handleInputChange} step="0.001" min="0.0001" max="0.1" disabled={isNaiveBayes} /></label>
                </div>
                <div className="form-group">
                  <label>Epochs: <input type="number" name="epochs" value={modelConfig.epochs} onChange={handleInputChange} min="1" max="20000" disabled={isNaiveBayes} /></label>
                </div>
                <div className="form-group">
                  <label>Dropout: <input type="number" name="dropout" value={modelConfig.dropout} onChange={handleInputChange} step="0.1" min="0" max="0.9" disabled={isNaiveBayes} /></label>
                </div>
                {modelConfig.model_name === 'GAT' && (
                  <div className="form-group"><label>Heads: <input type="number" name="heads" value={modelConfig.heads} onChange={handleInputChange} min="1" max="16" /></label></div>
                )}
                {modelConfig.model_name === 'ChebConv' && (
                  <div className="form-group"><label>K: <input type="number" name="K" value={modelConfig.K} onChange={handleInputChange} min="1" max="10" /></label></div>
                )}
              </div>

              {!isValidForTraining && (
                <div className="validation-warning">
                  Missing labels, nodes, or edges. <Link to="/">Fix in Graph Builder</Link>
                </div>
              )}
              {trainingError && <div className="training-error">{trainingError}</div>}
            </div>

            <div className="panel-footer">
              <div className="training-controls">
                <button className="btn btn-success" onClick={handleStartTraining} disabled={isTraining || !isValidForTraining}>
                  {isTraining ? 'Training...' : 'Start'}
                </button>
                <button className="btn btn-danger" onClick={handleStopTraining} disabled={!isTraining}>Stop</button>
              </div>
            </div>
          </div>
        </Panel>

        <PanelResizeHandle />

        {/* Center: Logs */}
        <Panel defaultSize={38} minSize={20}>
          <div className="panel">
            <div className="panel-header">
              <h3>Training Log</h3>
              {isTraining && <span style={{ fontSize: 'var(--text-xs)', color: 'var(--accent)' }}>Epoch {currentEpoch}/{modelConfig.epochs}</span>}
            </div>
            <div className="log-terminal" ref={logsContainerRef}>
              {trainingLogs.length === 0 ? (
                <p className="empty-logs">Logs will appear here when training starts...</p>
              ) : (
                trainingLogs.map((log, i) => (
                  <div key={i} className="log-entry">
                    <span className="log-time">{log.timestamp ? new Date(log.timestamp).toLocaleTimeString() : ''}</span>
                    <span className="log-message">{log.message}</span>
                  </div>
                ))
              )}
            </div>
            {isTraining && (
              <div className="training-progress">
                <div className="progress-bar">
                  <div className="progress-fill" style={{ width: `${(currentEpoch / modelConfig.epochs) * 100}%` }} />
                </div>
              </div>
            )}
          </div>
        </Panel>

        <PanelResizeHandle />

        {/* Right: Results */}
        <Panel defaultSize={40} minSize={25}>
          <div className="panel">
            <div className="tab-bar">
              {RESULT_TABS.map((label, i) => (
                <button key={label} className={`tab-btn ${resultTab === i ? 'active' : ''}`} onClick={() => setResultTab(i)}>{label}</button>
              ))}
            </div>
            <div style={{ flex: 1, overflow: 'auto', padding: 'var(--sp-12)' }}>
              {!metrics ? (
                <div className="stats-panel-empty"><p>Train a model to see results.</p></div>
              ) : (
                <MetricsVisualizer metrics={metrics} view={RESULT_TABS[resultTab].toLowerCase()} />
              )}
            </div>
          </div>
        </Panel>
      </PanelGroup>
    </div>
  );
};

export default TrainingTab;
