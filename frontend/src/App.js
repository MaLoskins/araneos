import React from 'react';
import { Routes, Route } from 'react-router-dom';
import CompactHeader from './components/layout/CompactHeader';
import GraphNet from './components/GraphNet-Tab/GraphNet';
import TrainingTab from './components/Training-Tab/TrainingTab';

function App() {
  return (
    <div className="app-container">
      <CompactHeader />
      <div className="workspace">
        <Routes>
          <Route path="/" element={<GraphNet />} />
          <Route path="/train" element={<TrainingTab />} />
        </Routes>
      </div>
    </div>
  );
}

export default App;
