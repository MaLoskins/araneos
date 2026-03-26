import React from 'react';
import { NavLink } from 'react-router-dom';
import { FiShare2, FiCpu, FiDownload } from 'react-icons/fi';
import { useGraphData } from '../../context/GraphDataContext';
import logo from '../../assets/logo.svg';

function CompactHeader() {
  const graphData = useGraphData();
  const nodeCount = graphData.stats?.node_count || 0;
  const edgeCount = graphData.stats?.edge_count || 0;

  const handleDownloadJSON = () => {
    if (!nodeCount) return;
    const blob = new Blob([JSON.stringify({ nodes: graphData.nodes, edges: graphData.edges }, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'graph_data.json'; a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <header className="compact-header">
      <div className="header-left">
        <img src={logo} alt="" className="header-logo" />
        <span className="header-title">Araneos</span>
      </div>

      <nav className="header-nav">
        <NavLink to="/" end className={({ isActive }) => `header-tab ${isActive ? 'active' : ''}`}>
          <FiShare2 size={14} />
          <span>Graph Builder</span>
        </NavLink>
        <NavLink to="/train" className={({ isActive }) => `header-tab ${isActive ? 'active' : ''}`}>
          <FiCpu size={14} />
          <span>Model Training</span>
        </NavLink>
      </nav>

      <div className="header-right">
        {nodeCount > 0 && (
          <div className="header-stats">
            <span className="stat-chip">{nodeCount} nodes</span>
            <span className="stat-chip">{edgeCount} edges</span>
          </div>
        )}
        {nodeCount > 0 && (
          <button className="header-icon-btn" onClick={handleDownloadJSON} title="Download graph JSON">
            <FiDownload size={14} />
          </button>
        )}
      </div>
    </header>
  );
}

export default CompactHeader;
