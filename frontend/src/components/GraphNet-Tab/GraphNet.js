import React, { useState, useCallback, useMemo } from 'react';
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels';
import FileUploader from './FileUploader';
import ConfigurationPanel from './ConfigurationPanel';
import ReactFlowWrapper from './ReactFlowWrapper';
import GraphVisualizer from './GraphVisualizer';
import GraphStatsPanel from './GraphStatsPanel';
import NodeEditModal from './NodeEditModal';
import RelationshipModal from './RelationshipModal';
import WorkflowStepper from '../layout/WorkflowStepper';
import useGraph from '../../hooks/useGraph';
import { useGraphData } from '../../context/GraphDataContext';

const STEPS = ['Upload', 'Nodes', 'Relationships', 'Features', 'Process'];

const MemoizedReactFlowWrapper = React.memo(ReactFlowWrapper);
const MemoizedGraphStatsPanel = React.memo(GraphStatsPanel);
const MemoizedFileUploader = React.memo(FileUploader);

function GraphNet() {
  const graph = useGraph();
  const graphData = useGraphData();
  const hasGraphData = graphData.nodes?.length > 0;
  const [rightTab, setRightTab] = useState(0);

  const handleSubmitWithView = useCallback(async (labelCol) => {
    const result = await graph.handleSubmit(labelCol);
    if (result) setRightTab(1);
  }, [graph.handleSubmit]);

  const currentStep = useMemo(() => {
    if (graph.columns.length === 0) return 0;
    if (graph.config.nodes.length === 0) return 1;
    if (graph.edges.length === 0) return 2;
    if (graph.loading) return 4;
    if (hasGraphData) return 5;
    return 3;
  }, [graph.columns.length, graph.config.nodes.length, graph.edges.length, graph.loading, hasGraphData]);

  const selectedNodes = useMemo(
    () => graph.config?.nodes?.map(n => n.id) || [],
    [graph.config?.nodes]
  );

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <WorkflowStepper steps={STEPS} currentStep={currentStep} />

      <PanelGroup direction="horizontal" style={{ flex: 1 }}>
        <Panel defaultSize={30} minSize={20} maxSize={45}>
          <div className="panel">
            <div className="panel-header"><h3>Configuration</h3></div>
            <div className="panel-body">
              <MemoizedFileUploader onFileDrop={graph.handleFileDrop} hasFile={graph.columns.length > 0} />
              {graph.columns.length > 0 && (
                <ConfigurationPanel
                  columns={graph.columns}
                  onSelectNode={graph.handleSelectNode}
                  onSubmit={handleSubmitWithView}
                  loading={graph.loading}
                  selectedNodes={selectedNodes}
                  useFeatureSpace={graph.useFeatureSpace}
                  onToggleFeatureSpace={graph.toggleFeatureSpace}
                  featureConfigs={graph.featureConfigs}
                  setFeatureConfigs={graph.setFeatureConfigs}
                  labelColumn={graph.labelColumn}
                  setLabelColumn={graph.setLabelColumn}
                />
              )}
            </div>
            {graph.columns.length > 0 && (
              <div className="panel-footer">
                <button
                  className="btn btn-primary btn-block"
                  onClick={() => handleSubmitWithView(graph.labelColumn || '')}
                  disabled={graph.loading || selectedNodes.length === 0}
                >
                  {graph.loading ? 'Processing...' : 'Process Graph'}
                </button>
                {graph.graphError && <div className="training-error" style={{ marginTop: 8 }}>{graph.graphError}</div>}
              </div>
            )}
          </div>
        </Panel>

        <PanelResizeHandle />

        <Panel defaultSize={70} minSize={40}>
          <div className="panel">
            <div className="tab-bar">
              {['Schema', 'Graph', 'Statistics'].map((label, i) => (
                <button key={label} className={`tab-btn ${rightTab === i ? 'active' : ''}`} onClick={() => setRightTab(i)}>{label}</button>
              ))}
            </div>

            <div style={{ flex: 1, overflow: 'hidden', display: rightTab === 0 ? 'flex' : 'none', flexDirection: 'column' }}>
              <MemoizedReactFlowWrapper
                nodes={graph.nodes}
                edges={graph.edges}
                setNodes={graph.setNodes}
                setEdges={graph.setEdges}
                onConnect={graph.onConnectHandler}
                onNodeClick={graph.onNodeClickHandler}
              />
            </div>
            <div style={{ flex: 1, overflow: 'hidden', display: rightTab === 1 ? 'flex' : 'none', flexDirection: 'column' }}>
              {hasGraphData
                ? <GraphVisualizer graphData={graphData} />
                : <div className="stats-panel-empty"><p>Process a graph to see the visualization.</p></div>
              }
            </div>
            <div style={{ flex: 1, overflow: 'hidden', display: rightTab === 2 ? 'flex' : 'none', flexDirection: 'column' }}>
              <MemoizedGraphStatsPanel />
            </div>
          </div>
        </Panel>
      </PanelGroup>

      {graph.currentNode && (
        <NodeEditModal
          isOpen={graph.nodeEditModalIsOpen}
          onRequestClose={() => graph.setNodeEditModalIsOpen(false)}
          node={graph.currentNode}
          onSaveNodeEdit={graph.handleSaveNodeEdit}
        />
      )}
      {graph.currentEdge && graph.relationshipModalIsOpen && (
        <RelationshipModal
          isOpen={graph.relationshipModalIsOpen}
          onRequestClose={() => graph.setRelationshipModalIsOpen(false)}
          onSaveRelationship={graph.onSaveRelationship}
        />
      )}
    </div>
  );
}

export default GraphNet;
