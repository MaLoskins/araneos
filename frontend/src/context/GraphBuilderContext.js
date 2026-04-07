import React, { createContext, useContext, useState, useCallback, useMemo } from 'react';

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
  const [labelColumn, setLabelColumn] = useState('');

  const [flowNodes, setFlowNodes] = useState([]);
  const [flowEdges, setFlowEdges] = useState([]);

  const resetBuilder = useCallback(() => {
    setConfig({ nodes: [], relationships: [], graph_type: 'directed', features: [] });
    setFlowNodes([]);
    setFlowEdges([]);
    setUseFeatureSpace(false);
    setFeatureConfigs([]);
    setLabelColumn('');
    setGraphError(null);
  }, []);

  const toggleFeatureSpace = useCallback(() => setUseFeatureSpace(p => !p), []);

  const value = useMemo(() => ({
    csvData, setCsvData, columns, setColumns,
    config, setConfig, loading, setLoading,
    graphError, setGraphError,
    useFeatureSpace, setUseFeatureSpace, toggleFeatureSpace,
    featureConfigs, setFeatureConfigs,
    labelColumn, setLabelColumn,
    flowNodes, setFlowNodes, flowEdges, setFlowEdges,
    resetBuilder,
  }), [
    csvData, columns, config, loading, graphError,
    useFeatureSpace, featureConfigs, labelColumn,
    flowNodes, flowEdges, resetBuilder, toggleFeatureSpace,
  ]);

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
