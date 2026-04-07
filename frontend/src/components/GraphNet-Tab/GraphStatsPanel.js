import React, { useState, useEffect, useMemo } from 'react';
import { Chart as ChartJS, BarElement, CategoryScale, LinearScale, Title, Tooltip, Legend } from 'chart.js';
import { Bar } from 'react-chartjs-2';
import { useGraphData } from '../../context/GraphDataContext';
import { fetchGraphStats } from '../../api';

ChartJS.register(BarElement, CategoryScale, LinearScale, Title, Tooltip, Legend);

const chartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: { legend: { display: false } },
  scales: {
    x: { title: { display: true, text: 'Degree', color: '#8b949e' }, ticks: { color: '#8b949e' }, grid: { color: '#21262d' } },
    y: { title: { display: true, text: 'Count', color: '#8b949e' }, ticks: { color: '#8b949e' }, grid: { color: '#21262d' } },
  },
};

function GraphStatsPanel() {
  const graphData = useGraphData();
  const [serverStats, setServerStats] = useState(null);

  useEffect(() => {
    if (!graphData.sessionId) { setServerStats(null); return; }
    let cancelled = false;
    fetchGraphStats(graphData.sessionId)
      .then(data => { if (!cancelled) setServerStats(data); })
      .catch(() => {});
    return () => { cancelled = true; };
  }, [graphData.sessionId]);

  const stats = serverStats || graphData.stats;

  const chartData = useMemo(() => {
    const degDist = serverStats?.degree_distribution;
    if (!degDist || Object.keys(degDist).length === 0) return null;
    return {
      labels: Object.keys(degDist),
      datasets: [{
        label: 'Nodes',
        data: Object.values(degDist),
        backgroundColor: 'rgba(88, 166, 255, 0.6)',
        borderColor: 'rgba(88, 166, 255, 1)',
        borderWidth: 1,
      }],
    };
  }, [serverStats?.degree_distribution]);

  if (!stats) return <div className="stats-panel-empty"><p>Process a graph to see statistics.</p></div>;

  return (
    <div className="stats-panel">
      <div className="stats-cards">
        <div className="stats-card">
          <div className="stats-card-value">{stats.node_count}</div>
          <div className="stats-card-label">Nodes</div>
        </div>
        <div className="stats-card">
          <div className="stats-card-value">{stats.edge_count}</div>
          <div className="stats-card-label">Edges</div>
        </div>
        <div className="stats-card">
          <div className="stats-card-value">{stats.avg_degree ?? '—'}</div>
          <div className="stats-card-label">Avg Degree</div>
        </div>
        <div className="stats-card">
          <div className="stats-card-value">{stats.max_degree ?? '—'}</div>
          <div className="stats-card-label">Max Degree</div>
        </div>
        <div className="stats-card">
          <div className="stats-card-value">{stats.label_count}</div>
          <div className="stats-card-label">Classes</div>
        </div>
        <div className="stats-card">
          <div className="stats-card-value">{stats.has_embeddings ? 'Yes' : 'No'}</div>
          <div className="stats-card-label">Embeddings</div>
        </div>
      </div>

      {chartData && (
        <div className="stats-chart">
          <h4>Degree Distribution</h4>
          <div className="stats-chart-container">
            <Bar data={chartData} options={chartOptions} />
          </div>
        </div>
      )}
    </div>
  );
}

export default React.memo(GraphStatsPanel);
