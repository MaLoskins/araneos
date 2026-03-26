const sectionsInfo = {
  sidebar: {
    title: "Sidebar",
    description: "<p>Navigate between tabs and view graph statistics.</p>",
  },
  configurationPanel: {
    title: "Configuration Panel",
    description: `
      <p>Select which columns become <strong>nodes</strong> in the graph, then draw edges in React Flow to define relationships.</p>
      <p>Optionally enable advanced feature creation (BERT, GloVe, Word2Vec embeddings).</p>
      <p>When ready, click "Process Graph" to build your network.</p>
    `,
  },
  featureColumns: {
    title: "Feature Columns",
    description: `
      <p>Define how embeddings are generated for each feature:</p>
      <ul>
        <li><strong>Node ID Column:</strong> Identifies which node each feature belongs to.</li>
        <li><strong>Column Name:</strong> The text or numeric column to embed.</li>
        <li><strong>Embedding Method:</strong> BERT, GloVe, or Word2Vec for text features.</li>
      </ul>
      <p>These features attach to existing nodes in the final graph.</p>
    `,
  },
  fileUploader: {
    title: "CSV File Uploader",
    description: "<p>Drag & drop a CSV file or click to browse. The CSV must have a header row with column names.</p>",
  },
  graphFlow: {
    title: "React Flow Configuration",
    description: `
      <p>Visualize and arrange your nodes interactively. Draw edges between nodes to define relationships.</p>
      <p>Click a node to edit its properties.</p>
    `,
  },
  processGraph: {
    title: "Process Graph",
    description: "<p>Send your configuration to the backend to build the graph and generate embeddings.</p>",
  },
  graphVisualization: {
    title: "Graph Visualization",
    description: `
      <p>Interactive 2D force-directed graph showing your network. Hover over nodes to see their features.</p>
      <p>Directed graphs show arrows on edges.</p>
    `,
  },
};

export default sectionsInfo;
