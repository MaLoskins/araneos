import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import './styles/main.css';
import App from './App';
import { GraphDataProvider } from './context/GraphDataContext';
import { GraphBuilderProvider } from './context/GraphBuilderContext';
import { NotificationProvider } from './context/NotificationContext';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <BrowserRouter>
      <NotificationProvider>
        <GraphDataProvider>
          <GraphBuilderProvider>
            <App />
          </GraphBuilderProvider>
        </GraphDataProvider>
      </NotificationProvider>
    </BrowserRouter>
  </React.StrictMode>
);
