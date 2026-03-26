import React, { useCallback, useMemo } from 'react';
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  applyNodeChanges,
  applyEdgeChanges,
  MarkerType,
} from 'react-flow-renderer';

const nodeColor = (node) => {
  switch (node.type) {
    case 'input': return '#3fb950';
    case 'output': return '#f85149';
    default: return '#58a6ff';
  }
};

const defaultEdgeOptions = {
  type: 'smoothstep',
  animated: true,
  style: { stroke: '#58a6ff', strokeWidth: 2, opacity: 0.6 },
  markerEnd: { type: MarkerType.ArrowClosed, color: '#58a6ff', width: 16, height: 16 },
  labelStyle: { fill: '#8b949e', fontSize: 11, fontWeight: 500, fontFamily: '-apple-system, BlinkMacSystemFont, sans-serif' },
  labelBgStyle: { fill: '#161b22', stroke: '#30363d', strokeWidth: 1, rx: 4, ry: 4 },
  labelBgPadding: [6, 4],
};

const proOptions = { hideAttribution: true };

const ReactFlowWrapper = ({ nodes, edges, setNodes, setEdges, onConnect, onNodeClick }) => {
  const onNodesChange = useCallback(
    (changes) => setNodes((nds) => applyNodeChanges(changes, nds)),
    [setNodes]
  );

  const onEdgesChange = useCallback(
    (changes) => setEdges((eds) => applyEdgeChanges(changes, eds)),
    [setEdges]
  );

  const styledNodes = useMemo(() =>
    nodes.map((node) => ({
      ...node,
      style: {
        background: '#161b22',
        color: '#e6edf3',
        border: '1.5px solid #30363d',
        borderRadius: '6px',
        padding: '8px 14px',
        fontSize: '0.8125rem',
        fontFamily: '-apple-system, BlinkMacSystemFont, sans-serif',
        fontWeight: 500,
        boxShadow: '0 3px 8px rgba(0,0,0,0.3)',
        minWidth: '100px',
        textAlign: 'center',
        ...node.style,
      },
    })),
    [nodes]
  );

  return (
    <div className="flow-wrapper">
      <ReactFlow
        nodes={styledNodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onNodeClick={onNodeClick}
        defaultEdgeOptions={defaultEdgeOptions}
        proOptions={proOptions}
        fitView
        fitViewOptions={{ padding: 0.3 }}
        snapToGrid
        snapGrid={[16, 16]}
        connectionLineStyle={{ stroke: '#58a6ff', strokeWidth: 2 }}
        connectionLineType="smoothstep"
        style={{ width: '100%', height: '100%' }}
      >
        <Background color="#21262d" gap={20} size={1} variant="dots" />
        <Controls className="flow-controls" />
        <MiniMap
          nodeColor={nodeColor}
          nodeStrokeWidth={2}
          maskColor="rgba(13, 17, 23, 0.8)"
          style={{
            background: '#161b22',
            border: '1px solid #30363d',
            borderRadius: '6px',
          }}
        />
      </ReactFlow>
    </div>
  );
};

export default ReactFlowWrapper;
