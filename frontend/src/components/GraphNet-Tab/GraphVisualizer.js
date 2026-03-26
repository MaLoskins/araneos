import React, { useRef, useEffect, useState, useMemo, useCallback } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { forceCollide } from 'd3-force-3d';

const GraphVisualizer = ({ graphData }) => {
  const fgRef = useRef();
  const containerRef = useRef(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
  const [hoverNode, setHoverNode] = useState(null);
  const hasInitialized = useRef(false);

  const dataKey = graphData?.nodes?.length + '-' + graphData?.edges?.length + '-' + (graphData?.lastUpdated || '');

  const processedGraphData = useMemo(() => {
    if (!graphData?.nodes?.length) return null;
    const rawLinks = graphData.edges || graphData.links || [];
    if (!rawLinks.length) return null;

    const nodeIds = new Set(graphData.nodes.map(n => String(n.id)));
    const degrees = {};
    const links = [];

    for (const link of rawLinks) {
      const src = link.source;
      const tgt = link.target;
      const srcId = String(typeof src === 'object' && src !== null ? (src.id ?? src) : src);
      const tgtId = String(typeof tgt === 'object' && tgt !== null ? (tgt.id ?? tgt) : tgt);
      if (!nodeIds.has(srcId) || !nodeIds.has(tgtId)) continue;
      links.push({ source: srcId, target: tgtId });
      degrees[srcId] = (degrees[srcId] || 0) + 1;
      degrees[tgtId] = (degrees[tgtId] || 0) + 1;
    }

    const maxDeg = Math.max(1, ...Object.values(degrees));
    const nodeSet = new Set();
    const nodes = [];

    for (const n of graphData.nodes) {
      const id = String(n.id);
      if (nodeSet.has(id)) continue;
      nodeSet.add(id);
      const deg = degrees[id] || 0;
      // val drives rendered area: range 3 (isolated) to 30 (max hub)
      const val = 3 + (deg / maxDeg) * 27;
      nodes.push({
        id,
        label: n.label ?? n.features?.label ?? id,
        type: n.type || 'default',
        features: n.features,
        val,
        deg,
      });
    }

    return { nodes, links, directed: graphData.directed };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dataKey]);

  useEffect(() => {
    if (!fgRef.current || !processedGraphData) return;
    const fg = fgRef.current;
    const nc = processedGraphData.nodes.length;
    const relSize = nc > 500 ? 3 : nc > 100 ? 4 : 5;

    // Charge: repel enough to separate clusters
    fg.d3Force('charge').strength(nc > 1000 ? -40 : nc > 500 ? -60 : nc > 100 ? -100 : -200);
    fg.d3Force('link').distance(nc > 1000 ? 15 : nc > 500 ? 20 : nc > 100 ? 30 : 50);

    // Collision: prevent overlap. Radius = sqrt(val) * relSize (matches how force-graph sizes nodes)
    fg.d3Force('collide', forceCollide(node => Math.sqrt(node.val || 3) * relSize + 1).iterations(2));

    if (!hasInitialized.current) {
      hasInitialized.current = true;
      setTimeout(() => {
        if (fgRef.current) fgRef.current.zoomToFit(400, 30);
      }, nc > 500 ? 3000 : 1500);
    }
  }, [processedGraphData]);

  // Debounced resize
  useEffect(() => {
    let tid;
    const observer = new ResizeObserver((entries) => {
      clearTimeout(tid);
      tid = setTimeout(() => {
        for (const entry of entries) {
          const { width, height } = entry.contentRect;
          if (width > 0 && height > 0) setDimensions({ width, height });
        }
      }, 100);
    });
    const node = containerRef.current;
    if (node) observer.observe(node);
    return () => { clearTimeout(tid); if (node) observer.unobserve(node); };
  }, []);

  const handleHover = useCallback((node) => setHoverNode(node), []);
  const handleBgClick = useCallback(() => setHoverNode(null), []);

  if (!processedGraphData) return <div className="graph-viz-container" ref={containerRef} />;

  const nc = processedGraphData.nodes.length;

  return (
    <div className="graph-viz-container" ref={containerRef}>
      <div className="graph-viz-badge">
        <span className="stat-chip">{nc} nodes</span>
        <span className="stat-chip">{processedGraphData.links.length} edges</span>
      </div>
      <ForceGraph2D
        ref={fgRef}
        graphData={processedGraphData}
        nodeAutoColorBy="type"
        nodeLabel={n => `${n.id} (deg: ${n.deg})${n.label !== n.id ? ' — ' + n.label : ''}`}
        nodeVal="val"
        nodeRelSize={nc > 500 ? 3 : nc > 100 ? 4 : 5}
        onNodeHover={handleHover}
        onBackgroundClick={handleBgClick}
        linkDirectionalArrowLength={graphData?.directed ? 4 : 0}
        linkDirectionalArrowRelPos={0.5}
        linkWidth={nc > 500 ? 0.5 : nc > 100 ? 0.8 : 1.5}
        linkColor={() => 'rgba(88,166,255,0.35)'}
        width={dimensions.width}
        height={dimensions.height}
        backgroundColor="#0d1117"
        cooldownTicks={nc > 500 ? 250 : nc > 100 ? 150 : 80}
        warmupTicks={nc > 500 ? 80 : nc > 100 ? 40 : 0}
        enableNodeDrag={nc < 2000}
        enablePointerInteraction={nc < 5000}
      />
      {hoverNode && (
        <div className="node-tooltip">
          <strong>{hoverNode.id}</strong>
          {hoverNode.label && hoverNode.label !== hoverNode.id && <div>Label: {hoverNode.label}</div>}
          <div>Degree: {hoverNode.deg}</div>
          {hoverNode.features && Object.keys(hoverNode.features).length > 0 && (
            <div style={{ marginTop: 4 }}>
              {Object.entries(hoverNode.features).map(([key, value]) => (
                <div key={key}><strong>{key}:</strong> {Array.isArray(value) ? `[${value.length}d]` : String(value)}</div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default React.memo(GraphVisualizer);
