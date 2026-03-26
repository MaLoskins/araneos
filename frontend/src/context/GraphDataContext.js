import React, { createContext, useContext, useReducer, useCallback, useMemo } from 'react';

const initialGraphState = {
  sessionId: null,
  nodes: [],       // Lightweight viz nodes (no embeddings)
  edges: [],       // Lightweight viz edges
  directed: false,
  stats: null,     // { node_count, edge_count, label_count, unique_labels, has_embeddings, ... }
  trainingLogs: [],
  lastUpdated: null,
};

function graphReducer(state, action) {
  const now = Date.now();
  switch (action.type) {
    case 'SET_GRAPH': {
      const { sessionId, graph, stats } = action.payload;
      return {
        ...state,
        sessionId,
        nodes: graph?.nodes || [],
        edges: graph?.edges || [],
        directed: graph?.directed || false,
        stats: stats || null,
        lastUpdated: now,
      };
    }
    case 'SET_STATS':
      return { ...state, stats: action.payload, lastUpdated: now };
    case 'RESET_GRAPH':
      return { ...initialGraphState, lastUpdated: now };
    case 'APPEND_TRAINING_LOG':
      return { ...state, trainingLogs: [...state.trainingLogs, action.payload], lastUpdated: now };
    case 'CLEAR_TRAINING_LOGS':
      return { ...state, trainingLogs: [], lastUpdated: now };
    default:
      throw new Error(`Unhandled action type: ${action.type}`);
  }
}

const GraphDataContext = createContext();
const GraphDispatchContext = createContext();

export function GraphDataProvider({ children }) {
  const [state, dispatch] = useReducer(graphReducer, initialGraphState);

  const setGraph = useCallback((payload) => dispatch({ type: 'SET_GRAPH', payload }), []);
  const setStats = useCallback((stats) => dispatch({ type: 'SET_STATS', payload: stats }), []);
  const resetGraph = useCallback(() => dispatch({ type: 'RESET_GRAPH' }), []);
  const appendTrainingLog = useCallback((logEntry) => dispatch({ type: 'APPEND_TRAINING_LOG', payload: logEntry }), []);
  const clearTrainingLogs = useCallback(() => dispatch({ type: 'CLEAR_TRAINING_LOGS' }), []);

  const dataValue = useMemo(() => ({
    ...state,
    ready: true,
  }), [state]);

  const actionsValue = useMemo(() => ({
    setGraph, setStats, resetGraph, appendTrainingLog, clearTrainingLogs,
  }), [setGraph, setStats, resetGraph, appendTrainingLog, clearTrainingLogs]);

  return (
    <GraphDataContext.Provider value={dataValue}>
      <GraphDispatchContext.Provider value={actionsValue}>
        {children}
      </GraphDispatchContext.Provider>
    </GraphDataContext.Provider>
  );
}

export function useGraphData() {
  const ctx = useContext(GraphDataContext);
  if (!ctx) throw new Error('useGraphData must be used within GraphDataProvider');
  return ctx;
}

export function useGraphActions() {
  const ctx = useContext(GraphDispatchContext);
  if (!ctx) throw new Error('useGraphActions must be used within GraphDataProvider');
  return ctx;
}
