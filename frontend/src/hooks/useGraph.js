import { useCallback, useMemo, useState } from 'react';
import { processData } from '../api';
import { addEdge } from 'react-flow-renderer';
import { useGraphData, useGraphActions } from '../context/GraphDataContext';
import { useGraphBuilder } from '../context/GraphBuilderContext';

const useGraph = () => {
  const graphData = useGraphData();
  const graphActions = useGraphActions();
  const builder = useGraphBuilder();

  const [nodeEditModalIsOpen, setNodeEditModalIsOpen] = useState(false);
  const [currentNode, setCurrentNode] = useState(null);
  const [relationshipModalIsOpen, setRelationshipModalIsOpen] = useState(false);
  const [currentEdge, setCurrentEdge] = useState(null);

  const handleFileDrop = useCallback((data, fields) => {
    builder.setCsvData(data);
    builder.setColumns(fields);
    builder.resetBuilder();
    graphActions.resetGraph();
  }, [builder, graphActions]);

  const handleSelectNode = useCallback((column) => {
    const alreadySelected = builder.config.nodes.find(n => n.id === column);
    if (alreadySelected) {
      builder.setConfig(prev => ({ ...prev, nodes: prev.nodes.filter(n => n.id !== column) }));
      builder.setFlowNodes(ns => ns.filter(n => n.id !== column));
      builder.setFlowEdges(es => es.filter(e => e.source !== column && e.target !== column));
    } else {
      builder.setConfig(prev => ({
        ...prev,
        nodes: [...prev.nodes, { id: column, type: 'default', features: {} }],
      }));
      builder.setFlowNodes(ns => [
        ...ns,
        { id: column, type: 'default', data: { label: column }, position: { x: Math.random() * 300, y: Math.random() * 300 }, features: {} },
      ]);
    }
  }, [builder]);

  const onConnectHandler = useCallback((connection) => {
    setCurrentEdge(connection);
    setRelationshipModalIsOpen(true);
  }, []);

  const onSaveRelationship = useCallback(({ relationshipType }) => {
    if (!currentEdge) return;
    builder.setFlowEdges(eds => addEdge({ ...currentEdge, label: relationshipType, type: 'smoothstep' }, eds));
    builder.setConfig(prev => ({
      ...prev,
      relationships: [...prev.relationships, { source: currentEdge.source, target: currentEdge.target, type: relationshipType || 'default' }],
    }));
    setRelationshipModalIsOpen(false);
    setCurrentEdge(null);
  }, [currentEdge, builder]);

  const onNodeClickHandler = useCallback((event, node) => {
    setCurrentNode(node);
    setNodeEditModalIsOpen(true);
  }, []);

  const handleSaveNodeEdit = useCallback(({ nodeType, nodeFeatures }) => {
    builder.setConfig(prev => ({
      ...prev,
      nodes: prev.nodes.map(n => n.id === currentNode.id ? { ...n, type: nodeType || 'default', features: nodeFeatures || {} } : n),
    }));
    builder.setFlowNodes(ns =>
      ns.map(nd => nd.id === currentNode.id
        ? { ...nd, type: nodeType || 'default', data: { ...nd.data, label: `${nd.id} (${nodeType || 'default'})` }, features: nodeFeatures || {} }
        : nd
      )
    );
    setNodeEditModalIsOpen(false);
    setCurrentNode(null);
  }, [currentNode, builder]);

  const handleSubmit = useCallback(async (labelColumn) => {
    if (labelColumn) localStorage.setItem('selectedLabelColumn', labelColumn);
    builder.setLoading(true);
    builder.setGraphError(null);
    try {
      const extendedConfig = {
        ...builder.config,
        features: builder.featureConfigs,
        use_feature_space: builder.useFeatureSpace,
        feature_space_config: builder.useFeatureSpace ? { features: builder.featureConfigs } : {},
        label_column: labelColumn || '',
      };
      const response = await processData(builder.csvData, extendedConfig);
      if (response.session_id && response.graph) {
        // Store session ID + lightweight viz data in context
        graphActions.setGraph({
          sessionId: response.session_id,
          graph: response.graph,
          stats: response.stats,
        });
        return true;
      }
      builder.setGraphError('No valid graph data returned');
      return false;
    } catch (err) {
      builder.setGraphError(err.message || 'Error processing data.');
      return false;
    } finally {
      builder.setLoading(false);
    }
  }, [builder, graphActions]);

  const graphStats = useMemo(() => {
    // Use server-provided stats if available
    if (graphData.stats) {
      return {
        nodeCount: graphData.stats.node_count,
        edgeCount: graphData.stats.edge_count,
        hasLabels: graphData.stats.labeled_nodes > 0,
        uniqueLabels: graphData.stats.unique_labels || [],
      };
    }
    return { nodeCount: 0, edgeCount: 0, hasLabels: false, uniqueLabels: [] };
  }, [graphData.stats]);

  const isValidForTraining = graphData.sessionId && graphStats.nodeCount > 0 && graphStats.edgeCount > 0 && graphStats.hasLabels;

  return {
    csvData: builder.csvData,
    columns: builder.columns,
    config: builder.config,
    loading: builder.loading,
    graphError: builder.graphError,
    nodes: builder.flowNodes,
    edges: builder.flowEdges,
    setNodes: builder.setFlowNodes,
    setEdges: builder.setFlowEdges,
    nodeEditModalIsOpen, currentNode,
    relationshipModalIsOpen, currentEdge,
    handleFileDrop, handleSelectNode, handleSubmit,
    onConnectHandler, onNodeClickHandler, onSaveRelationship,
    setNodeEditModalIsOpen, setRelationshipModalIsOpen, handleSaveNodeEdit,
    useFeatureSpace: builder.useFeatureSpace,
    toggleFeatureSpace: builder.toggleFeatureSpace,
    featureConfigs: builder.featureConfigs,
    setFeatureConfigs: builder.setFeatureConfigs,
    graphStats, isValidForTraining,
  };
};

export default useGraph;
