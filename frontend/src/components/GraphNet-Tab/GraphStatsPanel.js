import React, { useMemo } from 'react';
import { Chart as ChartJS, BarElement, CategoryScale, LinearScale, Title, Tooltip, Legend } from 'chart.js';
import { Bar } from 'react-chartjs-2';
import { useGraphData } from '../../context/GraphDataContext';

ChartJS.register(BarElement, CategoryScale, LinearScale, Title, Tooltip, Legend);

function GraphStatsPanel() {
  const graphData = useGraphData();
  const nodeCount = graphData.nodes?.length || 0;
  const edgeCount = graphData.edges?.length || 0;

  const degreeData = useMemo(() => {
    if (!graphData.nodes?.length || !graphData.edges?.length) return null;
    const degrees = {};
    for (const node of graphData.nodes) degrees[node.id] = 0;
    for (const edge of graphData.edges) {
      const src = typeof edge.source === 'object' ? edge.source.id : edge.source;
      const tgt = typeof edge.target === 'object' ? edge.target.id : edge.target;
      degrees[src] = (degrees[src] || 0) + 1;
      degrees[tgt] = (degrees[tgt] || 0) + 1;
    }
    const freq = {};
    Object.values(degrees).forEach(d => { freq[d] = (freq[d] || 0) + 1; });
    const sorted = Object.keys(freq).map(Number).sort((a, b) => a - b);
    return {
      labels: sorted.map(String),
      values: sorted.map(d => freq[d]),
      avgDegree: (Object.values(degrees).reduce((a, b) => a + b, 0) / Object.values(degrees).length).toFixed(1),
      maxDegree: Math.max(...Object.values(degrees)),
    };
  }, [graphData.nodes, graphData.edges]);

  if (!nodeCount) {
    return <div className="stats-panel-empty"><p>Process a graph to see statistics.</p></div>;
  }

  return (
    <div className="stats-panel">
      <div className="stats-cards">
        <div className="stats-card">
          <div className="stats-card-value">{nodeCount}</div>
          <div className="stats-card-label">Nodes</div>
        </div>
        <div className="stats-card">
          <div className="stats-card-value">{edgeCount}</div>
          <div className="stats-card-label">Edges</div>
        </div>
        <div className="stats-card">
          <div className="stats-card-value">{degreeData?.avgDegree || '—'}</div>
          <div className="stats-card-label">Avg Degree</div>
        </div>
        <div className="stats-card">
          <div className="stats-card-value">{degreeData?.maxDegree || '—'}</div>
          <div className="stats-card-label">Max Degree</div>
        </div>
      </div>

      {degreeData && (
        <div className="stats-chart">
          <h4>Degree Distribution</h4>
          <div className="stats-chart-container">
            <Bar
              data={{
                labels: degreeData.labels,
                datasets: [{
                  label: 'Nodes',
                  data: degreeData.values,
                  backgroundColor: 'rgba(74, 144, 226, 0.6)',
                  borderColor: 'rgba(74, 144, 226, 1)',
                  borderWidth: 1,
                }],
              }}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                  x: { title: { display: true, text: 'Degree', color: '#b0b0b0' }, ticks: { color: '#b0b0b0' }, grid: { color: '#333' } },
                  y: { title: { display: true, text: 'Count', color: '#b0b0b0' }, ticks: { color: '#b0b0b0' }, grid: { color: '#333' } },
                },
              }}
            />
          </div>
        </div>
      )}
    </div>
  );
}

export default GraphStatsPanel;
