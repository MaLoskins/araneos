import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import './styles/main.css';
import App from './App';
import { GraphDataProvider } from './context/GraphDataContext';
import { GraphBuilderProvider } from './context/GraphBuilderContext';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <BrowserRouter>
      <GraphDataProvider>
        <GraphBuilderProvider>
          <App />
        </GraphBuilderProvider>
      </GraphDataProvider>
    </BrowserRouter>
  </React.StrictMode>
);
