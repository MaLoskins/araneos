import React, { createContext, useContext, useReducer, useCallback, useMemo } from 'react';

const initialGraphState = {
  nodes: [],
  edges: [],
  trainingLogs: [],
  validationWarning: null,
  lastUpdated: null,
};

function graphReducer(state, action) {
  const now = Date.now();
  switch (action.type) {
    case 'SET_GRAPH': {
      const { nodes = [], edges = [], links = [], ...rest } = action.payload;
      return {
        ...state,
        ...rest,
        nodes,
        edges: edges.length > 0 ? edges : links,
        lastUpdated: now,
      };
    }
    case 'UPDATE_NODES':
      return { ...state, nodes: action.payload, lastUpdated: now };
    case 'UPDATE_EDGES':
      return { ...state, edges: action.payload, lastUpdated: now };
    case 'RESET_GRAPH':
      return { ...initialGraphState, lastUpdated: now };
    case 'APPEND_TRAINING_LOG':
      return { ...state, trainingLogs: [...state.trainingLogs, action.payload], lastUpdated: now };
    case 'CLEAR_TRAINING_LOGS':
      return { ...state, trainingLogs: [], lastUpdated: now };
    case 'SET_VALIDATION_WARNING':
      return { ...state, validationWarning: action.payload, lastUpdated: now };
    default:
      throw new Error(`Unhandled action type: ${action.type}`);
  }
}

const GraphDataContext = createContext();
const GraphDispatchContext = createContext();

export function GraphDataProvider({ children }) {
  const [state, dispatch] = useReducer(graphReducer, initialGraphState);

  // Clear any stale localStorage from previous versions
  React.useEffect(() => {
    try {
      localStorage.removeItem('graphData');
      localStorage.removeItem('reactFlowConfig');
    } catch {}
  }, []);

  const setGraph = useCallback((graph) => dispatch({ type: 'SET_GRAPH', payload: graph }), []);
  const updateNodes = useCallback((nodes) => dispatch({ type: 'UPDATE_NODES', payload: nodes }), []);
  const updateEdges = useCallback((edges) => dispatch({ type: 'UPDATE_EDGES', payload: edges }), []);
  const resetGraph = useCallback(() => dispatch({ type: 'RESET_GRAPH' }), []);
  const appendTrainingLog = useCallback((logEntry) => dispatch({ type: 'APPEND_TRAINING_LOG', payload: logEntry }), []);
  const clearTrainingLogs = useCallback(() => dispatch({ type: 'CLEAR_TRAINING_LOGS' }), []);
  const setValidationWarning = useCallback((warning) => dispatch({ type: 'SET_VALIDATION_WARNING', payload: warning }), []);

  const validateState = useCallback(() => {
    return Array.isArray(state.nodes) && state.nodes.length > 0
      && Array.isArray(state.edges) && state.edges.length > 0;
  }, [state.nodes, state.edges]);

  const dataValue = useMemo(() => ({
    ...state,
    ready: true,
    validateState,
  }), [state, validateState]);

  const actionsValue = useMemo(() => ({
    setGraph, updateNodes, updateEdges, resetGraph,
    appendTrainingLog, clearTrainingLogs, setValidationWarning,
  }), [setGraph, updateNodes, updateEdges, resetGraph, appendTrainingLog, clearTrainingLogs, setValidationWarning]);

  return (
    <GraphDataContext.Provider value={dataValue}>
      <GraphDispatchContext.Provider value={actionsValue}>
        {children}
      </GraphDispatchContext.Provider>
    </GraphDataContext.Provider>
  );
}

export function useGraphData() {
  const context = useContext(GraphDataContext);
  if (context === undefined) throw new Error('useGraphData must be used within a GraphDataProvider');
  return context;
}

export function useGraphActions() {
  const context = useContext(GraphDispatchContext);
  if (context === undefined) throw new Error('useGraphActions must be used within a GraphDataProvider');
  return context;
}
