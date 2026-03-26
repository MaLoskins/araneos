import React, { createContext, useContext, useState, useCallback } from 'react';

const GraphBuilderContext = createContext();

export function GraphBuilderProvider({ children }) {
  const [csvData, setCsvData] = useState([]);
  const [columns, setColumns] = useState([]);
  const [config, setConfig] = useState({
    nodes: [], relationships: [], graph_type: 'directed', features: [],
  });
  const [loading, setLoading] = useState(false);
  const [graphError, setGraphError] = useState(null);
  const [useFeatureSpace, setUseFeatureSpace] = useState(false);
  const [featureConfigs, setFeatureConfigs] = useState([]);

  // ReactFlow nodes/edges stored as plain arrays (not useNodesState)
  // because useNodesState is a hook that must be in a component
  const [flowNodes, setFlowNodes] = useState([]);
  const [flowEdges, setFlowEdges] = useState([]);

  const resetBuilder = useCallback(() => {
    setConfig({ nodes: [], relationships: [], graph_type: 'directed', features: [] });
    setFlowNodes([]);
    setFlowEdges([]);
    setUseFeatureSpace(false);
    setFeatureConfigs([]);
    setGraphError(null);
  }, []);

  const value = {
    csvData, setCsvData, columns, setColumns,
    config, setConfig, loading, setLoading,
    graphError, setGraphError,
    useFeatureSpace, setUseFeatureSpace, toggleFeatureSpace: useCallback(() => setUseFeatureSpace(p => !p), []),
    featureConfigs, setFeatureConfigs,
    flowNodes, setFlowNodes, flowEdges, setFlowEdges,
    resetBuilder,
  };

  return (
    <GraphBuilderContext.Provider value={value}>
      {children}
    </GraphBuilderContext.Provider>
  );
}

export function useGraphBuilder() {
  const ctx = useContext(GraphBuilderContext);
  if (!ctx) throw new Error('useGraphBuilder must be used within GraphBuilderProvider');
  return ctx;
}
