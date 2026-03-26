import React, { useCallback } from 'react';
import ReactFlow, { Background, Controls, applyNodeChanges, applyEdgeChanges } from 'react-flow-renderer';

const ReactFlowWrapper = ({ nodes, edges, setNodes, setEdges, onConnect, onNodeClick }) => {
  const onNodesChange = useCallback(
    (changes) => setNodes((nds) => applyNodeChanges(changes, nds)),
    [setNodes]
  );

  const onEdgesChange = useCallback(
    (changes) => setEdges((eds) => applyEdgeChanges(changes, eds)),
    [setEdges]
  );

  return (
    <div className="flow-wrapper">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onNodeClick={onNodeClick}
        fitView
        style={{ width: '100%', height: '100%' }}
      >
        <Background color="#333" gap={20} />
        <Controls />
      </ReactFlow>
    </div>
  );
};

export default ReactFlowWrapper;
