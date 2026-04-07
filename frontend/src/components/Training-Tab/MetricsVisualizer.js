import React, { useMemo } from 'react';
import {
  Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement,
  BarElement, Title, Tooltip, Legend, Filler, ArcElement,
} from 'chart.js';
import { Line, Pie } from 'react-chartjs-2';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, BarElement, Title, Tooltip, Legend, Filler, ArcElement);

const baseChartOpts = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: { legend: { position: 'top', labels: { color: '#8b949e', boxWidth: 12 } }, tooltip: { mode: 'index', intersect: false } },
  scales: {
    x: { ticks: { color: '#6e7681' }, grid: { color: '#21262d' } },
    y: { ticks: { color: '#6e7681' }, grid: { color: '#21262d' } },
  },
};

const accChartOpts = {
  ...baseChartOpts,
  scales: {
    ...baseChartOpts.scales,
    y: { ...baseChartOpts.scales.y, min: 0, max: 1, ticks: { ...baseChartOpts.scales.y.ticks, callback: v => (v * 100).toFixed(0) + '%' } },
  },
};

const pieOpts = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: { legend: { position: 'right', labels: { color: '#8b949e', boxWidth: 12 } } },
};

const MetricsVisualizer = ({ metrics, view = 'metrics' }) => {
  if (!metrics) return null;

  if (view === 'metrics') return <MetricsCards metrics={metrics} />;
  if (view === 'charts') return <MetricsCharts metrics={metrics} />;
  if (view === 'report') return <MetricsReport metrics={metrics} />;
  return null;
};

function MetricsCards({ metrics }) {
  const cards = useMemo(() => [
    { label: 'Test Accuracy', value: `${(metrics.test_accuracy * 100).toFixed(2)}%` },
    metrics.f1_score != null && { label: 'F1 Score', value: metrics.f1_score.toFixed(4) },
    metrics.precision != null && { label: 'Precision', value: metrics.precision.toFixed(4) },
    metrics.recall != null && { label: 'Recall', value: metrics.recall.toFixed(4) },
    metrics.training_time != null && { label: 'Training Time', value: `${metrics.training_time.toFixed(1)}s` },
    metrics.best_val_loss != null && { label: 'Best Val Loss', value: metrics.best_val_loss.toFixed(4) },
  ].filter(Boolean), [metrics]);

  return (
    <div className="metrics-grid">
      {cards.map(({ label, value }) => (
        <div key={label} className="metric-card">
          <div className="metric-card-value">{value}</div>
          <div className="metric-card-label">{label}</div>
        </div>
      ))}
    </div>
  );
}

function MetricsCharts({ metrics }) {
  const history = metrics.history || {};
  const hasLoss = history.loss?.length > 0;
  const hasAcc = history.accuracy?.length > 0;

  const lossData = useMemo(() => {
    if (!hasLoss) return null;
    const epochs = Array.from({ length: history.loss.length }, (_, i) => i + 1);
    return {
      labels: epochs,
      datasets: [
        { label: 'Train Loss', data: history.loss, borderColor: '#f85149', backgroundColor: 'rgba(248,81,73,0.1)', fill: true, tension: 0.4 },
        { label: 'Val Loss', data: history.val_loss || [], borderColor: '#58a6ff', backgroundColor: 'rgba(88,166,255,0.1)', fill: true, tension: 0.4 },
      ],
    };
  }, [history.loss, history.val_loss, hasLoss]);

  const accData = useMemo(() => {
    if (!hasAcc) return null;
    const epochs = Array.from({ length: history.accuracy.length }, (_, i) => i + 1);
    return {
      labels: epochs,
      datasets: [
        { label: 'Train Acc', data: history.accuracy, borderColor: '#3fb950', backgroundColor: 'rgba(63,185,80,0.1)', fill: true, tension: 0.4 },
        { label: 'Val Acc', data: history.val_accuracy || [], borderColor: '#d2a8ff', backgroundColor: 'rgba(210,168,255,0.1)', fill: true, tension: 0.4 },
      ],
    };
  }, [history.accuracy, history.val_accuracy, hasAcc]);

  if (!hasLoss && !hasAcc) {
    return <div className="stats-panel-empty"><p>No training history available for charts.</p></div>;
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 24, height: '100%' }}>
      {lossData && (
        <div style={{ flex: 1, minHeight: 200 }}>
          <h4 style={{ fontSize: 'var(--text-sm)', color: 'var(--text-secondary)', marginBottom: 8 }}>Loss Curve</h4>
          <div className="chart-wrapper">
            <Line data={lossData} options={baseChartOpts} />
          </div>
        </div>
      )}
      {accData && (
        <div style={{ flex: 1, minHeight: 200 }}>
          <h4 style={{ fontSize: 'var(--text-sm)', color: 'var(--text-secondary)', marginBottom: 8 }}>Accuracy Curve</h4>
          <div className="chart-wrapper">
            <Line data={accData} options={accChartOpts} />
          </div>
        </div>
      )}
    </div>
  );
}

function MetricsReport({ metrics }) {
  const hasClassDist = metrics.class_distribution && Object.keys(metrics.class_distribution).length > 0;

  const pieData = useMemo(() => {
    if (!hasClassDist) return null;
    const labels = Object.keys(metrics.class_distribution);
    return {
      labels,
      datasets: [{
        data: Object.values(metrics.class_distribution),
        backgroundColor: labels.map((_, i) => `hsla(${(i * 137) % 360}, 70%, 60%, 0.7)`),
        borderWidth: 0,
      }],
    };
  }, [metrics.class_distribution, hasClassDist]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {metrics.class_report && (
        <div>
          <h4 style={{ fontSize: 'var(--text-sm)', color: 'var(--text-secondary)', marginBottom: 8 }}>Classification Report</h4>
          <pre className="metric-report">{metrics.class_report}</pre>
        </div>
      )}
      {pieData && (
        <div>
          <h4 style={{ fontSize: 'var(--text-sm)', color: 'var(--text-secondary)', marginBottom: 8 }}>Class Distribution</h4>
          <div style={{ height: 300 }}>
            <Pie data={pieData} options={pieOpts} />
          </div>
        </div>
      )}
      {!metrics.class_report && !hasClassDist && (
        <div className="stats-panel-empty"><p>No detailed report available.</p></div>
      )}
    </div>
  );
}

export default React.memo(MetricsVisualizer);
