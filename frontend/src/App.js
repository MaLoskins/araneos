import React, { Suspense, lazy } from 'react';
import { Routes, Route } from 'react-router-dom';
import CompactHeader from './components/layout/CompactHeader';
import ErrorBoundary from './components/ErrorBoundary';

const GraphNet = lazy(() => import('./components/GraphNet-Tab/GraphNet'));
const TrainingTab = lazy(() => import('./components/Training-Tab/TrainingTab'));

const Loading = () => (
  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--text-muted)' }}>
    Loading...
  </div>
);

function App() {
  return (
    <div className="app-container">
      <CompactHeader />
      <div className="workspace">
        <ErrorBoundary>
          <Suspense fallback={<Loading />}>
            <Routes>
              <Route path="/" element={<GraphNet />} />
              <Route path="/train" element={<TrainingTab />} />
            </Routes>
          </Suspense>
        </ErrorBoundary>
      </div>
    </div>
  );
}

export default App;
