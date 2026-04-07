// Import the function to test
import { trainModel } from '../api';

// Create a mock for axios
jest.mock('axios', () => {
  const mockAxios = jest.fn(() => {
    return {
      catch: jest.fn().mockImplementation(fn => {
        mockAxios.errorCallback = fn;
        return Promise.resolve();
      })
    };
  });

  mockAxios.mockClear = jest.fn();
  mockAxios.post = jest.fn();
  mockAxios.get = jest.fn();

  return mockAxios;
});

// Import axios again after mocking
const axios = require('axios');

describe('trainModel function', () => {
  let mockOnMessage;
  let mockOnError;
  let validSessionId;
  let validModelConfig;

  beforeEach(() => {
    jest.clearAllMocks();

    mockOnMessage = jest.fn();
    mockOnError = jest.fn();

    validSessionId = 'abc12345';

    validModelConfig = {
      model_name: 'GCN',
      epochs: 100,
      learning_rate: 0.01
    };
  });

  // #1 - Test input validation
  describe('Input validation', () => {
    test('rejects when sessionId is missing', async () => {
      await expect(trainModel(null, validModelConfig, mockOnMessage, mockOnError))
        .rejects.toThrow('No session ID');

      expect(mockOnError).toHaveBeenCalled();
      expect(axios).not.toHaveBeenCalled();
    });

    test('rejects when sessionId is empty string', async () => {
      await expect(trainModel('', validModelConfig, mockOnMessage, mockOnError))
        .rejects.toThrow('No session ID');

      expect(mockOnError).toHaveBeenCalled();
      expect(axios).not.toHaveBeenCalled();
    });

    test('rejects when modelConfig is missing model_name', async () => {
      const invalidConfig = { epochs: 100 };

      await expect(trainModel(validSessionId, invalidConfig, mockOnMessage, mockOnError))
        .rejects.toThrow('Invalid model configuration');

      expect(mockOnError).toHaveBeenCalled();
      expect(axios).not.toHaveBeenCalled();
    });
  });

  // #2 - Test request configuration
  describe('Request configuration', () => {
    test('creates correct request configuration', async () => {
      axios.mockImplementation(() => Promise.resolve({}));

      await trainModel(validSessionId, validModelConfig, mockOnMessage, mockOnError);

      expect(axios).toHaveBeenCalledWith(expect.objectContaining({
        url: 'http://localhost:8000/train-gnn',
        method: 'POST',
        data: {
          session_id: validSessionId,
          configuration: validModelConfig
        },
        responseType: 'text',
      }));

      const config = axios.mock.calls[0][0];
      expect(config).toHaveProperty('onDownloadProgress');
      expect(typeof config.onDownloadProgress).toBe('function');
    });
  });

  // #3 - Test streaming response handling
  describe('Streaming response handling', () => {
    test('processes streaming data correctly', async () => {
      axios.mockImplementation(() => Promise.resolve({}));

      await trainModel(validSessionId, validModelConfig, mockOnMessage, mockOnError);

      const config = axios.mock.calls[0][0];
      const onDownloadProgress = config.onDownloadProgress;

      const progressEvent = {
        currentTarget: {
          response: '{"epoch":1,"loss":0.5}\n{"epoch":2,"loss":0.3}'
        }
      };

      onDownloadProgress(progressEvent);

      expect(mockOnMessage).toHaveBeenCalledTimes(2);
      expect(mockOnMessage).toHaveBeenNthCalledWith(1, { epoch: 1, loss: 0.5 });
      expect(mockOnMessage).toHaveBeenNthCalledWith(2, { epoch: 2, loss: 0.3 });
    });

    test('handles empty response', async () => {
      axios.mockImplementation(() => Promise.resolve({}));

      await trainModel(validSessionId, validModelConfig, mockOnMessage, mockOnError);

      const config = axios.mock.calls[0][0];
      const onDownloadProgress = config.onDownloadProgress;

      const progressEvent = {
        currentTarget: {
          response: ''
        }
      };

      onDownloadProgress(progressEvent);

      expect(mockOnMessage).not.toHaveBeenCalled();
    });

    test('processes only new messages since last update', async () => {
      axios.mockImplementation(() => Promise.resolve({}));

      await trainModel(validSessionId, validModelConfig, mockOnMessage, mockOnError);

      const config = axios.mock.calls[0][0];
      const onDownloadProgress = config.onDownloadProgress;

      const initialProgressEvent = {
        currentTarget: {
          response: '{"epoch":1,"loss":0.5}\n{"epoch":2,"loss":0.3}'
        },
      };

      onDownloadProgress(initialProgressEvent);
      expect(mockOnMessage).toHaveBeenCalledTimes(2);

      mockOnMessage.mockClear();

      const updatedProgressEvent = {
        currentTarget: {
          response: '{"epoch":1,"loss":0.5}\n{"epoch":2,"loss":0.3}\n{"epoch":3,"loss":0.2}'
        },
      };

      onDownloadProgress(updatedProgressEvent);

      expect(mockOnMessage).toHaveBeenCalledTimes(1);
      expect(mockOnMessage).toHaveBeenCalledWith({ epoch: 3, loss: 0.2 });
    });

    test('handles invalid JSON in the response', async () => {
      const originalConsoleWarn = console.warn;
      console.warn = jest.fn();

      try {
        axios.mockImplementation(() => Promise.resolve({}));

        await trainModel(validSessionId, validModelConfig, mockOnMessage, mockOnError);

        const config = axios.mock.calls[0][0];
        const onDownloadProgress = config.onDownloadProgress;

        const progressEvent = {
          currentTarget: {
            response: '{"epoch":1,"loss":0.5}\nNOT JSON\n{"epoch":3,"loss":0.2}'
          }
        };

        onDownloadProgress(progressEvent);

        // Valid JSON lines are parsed, invalid ones are silently skipped
        expect(mockOnMessage).toHaveBeenCalledTimes(2);
        expect(mockOnMessage).toHaveBeenNthCalledWith(1, { epoch: 1, loss: 0.5 });
        expect(mockOnMessage).toHaveBeenNthCalledWith(2, { epoch: 3, loss: 0.2 });
      } finally {
        console.warn = originalConsoleWarn;
      }
    });
  });

  // #4 - Test error handling
  describe('Error handling', () => {
    test('handles network errors', async () => {
      const networkError = new Error('Network Error');

      axios.mockImplementation(() => Promise.reject(networkError));

      await expect(trainModel(validSessionId, validModelConfig, mockOnMessage, mockOnError))
        .rejects.toThrow('Network Error');

      expect(mockOnError).toHaveBeenCalledWith(networkError);
    });

    test('handles server errors (4xx/5xx)', async () => {
      const serverError = new Error('Internal Server Error');
      serverError.response = { status: 500, data: { message: 'Server failed' } };

      axios.mockImplementation(() => Promise.reject(serverError));

      await expect(trainModel(validSessionId, validModelConfig, mockOnMessage, mockOnError))
        .rejects.toThrow('Internal Server Error');

      expect(mockOnError).toHaveBeenCalledWith(serverError);
    });

    test('handles synchronous errors during request setup', () => {
      const syncError = new Error('Synchronous Error');
      axios.mockImplementation(() => { throw syncError; });

      // Sync throw from axios propagates directly since trainModel doesn't wrap in try/catch
      expect(() => trainModel(validSessionId, validModelConfig, mockOnMessage, mockOnError))
        .toThrow('Synchronous Error');
    });
  });
});
