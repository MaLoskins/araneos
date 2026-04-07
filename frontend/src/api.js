import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

async function withRetry(fn, retries = 2, delay = 1000) {
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      const status = error?.response?.status;
      const isRetryable = !status || status >= 500;
      if (attempt < retries && isRetryable) {
        await new Promise(r => setTimeout(r, delay * (attempt + 1)));
        continue;
      }
      throw error;
    }
  }
}

export const processData = (data, config) =>
  withRetry(() => axios.post(`${API_BASE_URL}/process-data`, { data, config }).then(r => r.data));

export const fetchGraphStats = (sessionId) =>
  withRetry(() => axios.get(`${API_BASE_URL}/graph/${sessionId}/stats`).then(r => r.data));

export const trainModel = (sessionId, modelConfig, onMessage, onError) => {
  if (!sessionId) {
    const err = new Error('No session ID. Please process a graph first.');
    onError(err);
    return Promise.reject(err);
  }

  if (!modelConfig?.model_name) {
    const err = new Error('Invalid model configuration: Missing model_name');
    onError(err);
    return Promise.reject(err);
  }

  let lastProcessedIndex = 0;

  return axios({
    url: `${API_BASE_URL}/train-gnn`,
    method: 'POST',
    data: { session_id: sessionId, configuration: modelConfig },
    responseType: 'text',
    onDownloadProgress: (progressEvent) => {
      const rawText = progressEvent.event?.target?.responseText
        || progressEvent.currentTarget?.responseText
        || progressEvent.currentTarget?.response
        || '';
      if (!rawText) return;

      const messages = rawText.split('\n').filter(line => line.trim());
      const newMessages = messages.slice(lastProcessedIndex);
      lastProcessedIndex = messages.length;

      newMessages.forEach(msg => {
        try { onMessage(JSON.parse(msg)); } catch {}
      });
    },
  }).catch(error => {
    onError(error);
    throw error;
  });
};
