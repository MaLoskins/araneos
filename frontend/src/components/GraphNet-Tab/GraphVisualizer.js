import React, { useRef, useEffect, useState, useMemo } from 'react';
import ForceGraph2D from 'react-force-graph-2d';

const GraphVisualizer = ({ graphData }) => {
  const fgRef = useRef();
  const containerRef = useRef(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
  const [hoverNode, setHoverNode] = useState(null);

  const processedGraphData = useMemo(() => {
    if (!graphData?.nodes?.length) return null;
    const rawLinks = graphData.edges || graphData.links || [];
    if (!rawLinks.length) return null;

    const nodes = JSON.parse(JSON.stringify(graphData.nodes));
    const nodeMap = new Map(nodes.map(n => [String(n.id), n]));
    const links = [];
    const degrees = {};

    for (const link of rawLinks) {
      const src = link.source;
      const tgt = link.target;
      const srcId = String(typeof src === 'object' && src !== null ? (src.id ?? src) : src);
      const tgtId = String(typeof tgt === 'object' && tgt !== null ? (tgt.id ?? tgt) : tgt);
      if (!nodeMap.has(srcId) || !nodeMap.has(tgtId)) continue;
      links.push({ source: srcId, target: tgtId, type: link.type || link.key });
      degrees[srcId] = (degrees[srcId] || 0) + 1;
      degrees[tgtId] = (degrees[tgtId] || 0) + 1;
    }

    nodes.forEach(node => {
      node.val = Math.max(1, Math.sqrt(degrees[node.id] || 1));
      node.type = node.type || 'default';
      if (node.label === undefined) node.label = node.id;
    });

    return { nodes, links, directed: graphData.directed };
  }, [graphData]);

  useEffect(() => {
    if (!fgRef.current || !processedGraphData) return;
    const fg = fgRef.current;
    const nc = processedGraphData.nodes.length;
    // Gentler forces: enough to separate but not scatter
    const charge = nc > 500 ? -30 : nc > 100 ? -80 : -150;
    const linkDist = nc > 500 ? 10 : nc > 100 ? 20 : 40;
    fg.d3Force('charge').strength(charge);
    fg.d3Force('link').distance(linkDist);
    // Zoom to fit after layout settles
    setTimeout(() => fg.zoomToFit(400, 60), nc > 500 ? 3000 : 1500);
  }, [processedGraphData]);

  useEffect(() => {
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) setDimensions({ width: entry.contentRect.width, height: entry.contentRect.height });
    });
    const node = containerRef.current;
    if (node) observer.observe(node);
    return () => { if (node) observer.unobserve(node); };
  }, []);

  const nc = processedGraphData?.nodes?.length || 0;
  const isLarge = nc > 500;
  const isMedium = nc > 100;

  return (
    <div className="graph-viz-container" ref={containerRef}>
      {processedGraphData && (
        <>
          <div className="graph-viz-badge">
            <span className="stat-chip">{processedGraphData.nodes.length} nodes</span>
            <span className="stat-chip">{processedGraphData.links.length} edges</span>
          </div>
          <ForceGraph2D
            ref={fgRef}
            graphData={processedGraphData}
            nodeAutoColorBy="type"
            nodeLabel={n => `${n.id}${n.label && n.label !== n.id ? ` (${n.label})` : ''}`}
            nodeVal={n => n.val || 1}
            nodeRelSize={isLarge ? 2 : isMedium ? 4 : 6}
            onNodeHover={setHoverNode}
            onBackgroundClick={() => setHoverNode(null)}
            linkDirectionalArrowLength={graphData?.directed ? 4 : 0}
            linkDirectionalArrowRelPos={0.5}
            linkWidth={isLarge ? 0.3 : isMedium ? 0.5 : 1}
            linkColor={() => 'rgba(88,166,255,0.2)'}
            width={dimensions.width}
            height={dimensions.height}
            backgroundColor="#0d1117"
            cooldownTicks={isLarge ? 300 : isMedium ? 200 : 100}
            warmupTicks={isLarge ? 100 : isMedium ? 50 : 0}
          />
        </>
      )}
      {hoverNode && (
        <div className="node-tooltip">
          <strong>{hoverNode.id}</strong>
          {hoverNode.label && hoverNode.label !== hoverNode.id && <div>Label: {hoverNode.label}</div>}
          <div>Type: {hoverNode.type || 'default'}</div>
          {hoverNode.features && Object.keys(hoverNode.features).length > 0 && (
            <div style={{ marginTop: 4 }}>
              {Object.entries(hoverNode.features).map(([key, value]) => (
                <div key={key}><strong>{key}:</strong> {Array.isArray(value) ? `[${value.length}d vector]` : String(value)}</div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default GraphVisualizer;
