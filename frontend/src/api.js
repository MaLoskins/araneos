import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

export const processData = async (data, config) => {
  const response = await axios.post(`${API_BASE_URL}/process-data`, { data, config });
  return response.data;
};

export const trainModel = (graph, modelConfig, onMessage, onError) => {
  if (!graph?.nodes || !graph?.links) {
    const err = new Error('Invalid graph data: Must contain nodes and links');
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
    data: { graph, configuration: modelConfig },
    responseType: 'text',
    onDownloadProgress: (progressEvent) => {
      // In browser, the raw response text is on the XMLHttpRequest target
      const rawText = progressEvent.event?.target?.responseText
        || progressEvent.currentTarget?.responseText
        || progressEvent.currentTarget?.response
        || '';
      if (!rawText) return;

      const messages = rawText.split('\n').filter(line => line.trim());
      const newMessages = messages.slice(lastProcessedIndex);
      lastProcessedIndex = messages.length;

      newMessages.forEach(msg => {
        try {
          onMessage(JSON.parse(msg));
        } catch {
          // Skip incomplete JSON chunks
        }
      });
    },
  }).catch(error => {
    onError(error);
    throw error;
  });
};
