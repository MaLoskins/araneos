import React, { useState } from 'react';
import { FiList, FiSettings } from 'react-icons/fi';
import CollapsibleSection from '../layout/CollapsibleSection';

function isFeatureComplete(f) {
  if (!f.node_id_column || !f.column_name || !f.type) return false;
  if (f.type === 'text' && !f.embedding_dim) return false;
  return true;
}

const DEFAULT_FEATURE = {
  node_id_column: '', column_name: '', type: 'text',
  embedding_method: 'bert', embedding_dim: 768, additional_params: {},
  data_type: 'float', processing: 'none', projection: {},
};

const ConfigurationPanel = ({
  columns, onSelectNode, onSubmit, loading, selectedNodes,
  useFeatureSpace, onToggleFeatureSpace, featureConfigs, setFeatureConfigs,
}) => {
  const [expandedIndices, setExpandedIndices] = useState([]);
  const [labelColumn, setLabelColumn] = useState(() => localStorage.getItem('selectedLabelColumn') || '');

  React.useEffect(() => { localStorage.setItem('selectedLabelColumn', labelColumn); }, [labelColumn]);

  const toggleExpand = (i) => setExpandedIndices(prev => { const u = [...prev]; u[i] = !prev[i]; return u; });
  const addFeature = () => { setFeatureConfigs(prev => [...prev, { ...DEFAULT_FEATURE }]); setExpandedIndices(prev => [...prev, true]); };
  const removeFeature = (i) => { setFeatureConfigs(prev => prev.filter((_, j) => j !== i)); setExpandedIndices(prev => prev.filter((_, j) => j !== i)); };
  const updateFeature = (i, key, val) => setFeatureConfigs(prev => prev.map((f, j) => j === i ? { ...f, [key]: val } : f));

  return (
    <>
      <CollapsibleSection title="Node Selection" icon={FiList} defaultExpanded>
        <div className="node-selection">
          {columns.map(col => (
            <div key={col} className="node-selector">
              <input type="checkbox" id={`node-${col}`} checked={selectedNodes.includes(col)} onChange={() => onSelectNode(col)} />
              <label htmlFor={`node-${col}`}>{col}</label>
            </div>
          ))}
        </div>

        <hr className="config-divider" />

        <div className="config-option-row">
          <label>
            Label Column:
            <select value={labelColumn} onChange={e => setLabelColumn(e.target.value)} style={{ marginLeft: 8, flex: 1 }}>
              <option value="">--None--</option>
              {columns.map(c => <option key={c} value={c}>{c}</option>)}
            </select>
          </label>
        </div>

        <div className="config-option-row">
          <label>
            <input type="checkbox" checked={useFeatureSpace} onChange={onToggleFeatureSpace} />
            Advanced embeddings (BERT/GloVe/Word2Vec)
          </label>
        </div>
      </CollapsibleSection>

      {useFeatureSpace && (
        <CollapsibleSection title="Feature Configuration" icon={FiSettings} defaultExpanded>
          <div className="feature-grid">
            {featureConfigs.map((feature, i) => {
              const complete = isFeatureComplete(feature);
              const expanded = expandedIndices[i];
              return (
                <div key={i} className="feature-config-item">
                  <button type="button" className="remove-feature-btn" onClick={() => removeFeature(i)}>Remove</button>
                  {complete && !expanded ? (
                    <>
                      <FeatureSummary feature={feature} index={i} />
                      <button onClick={() => toggleExpand(i)} className="edit-feature-btn">Edit</button>
                    </>
                  ) : (
                    <>
                      {complete && <div className="feature-summary-container"><FeatureSummary feature={feature} index={i} /></div>}
                      <FeatureEditForm feature={feature} index={i} columns={columns} updateFeature={updateFeature} />
                      <button onClick={() => toggleExpand(i)} className="toggle-feature-btn">{complete ? 'Close' : 'Done'}</button>
                    </>
                  )}
                </div>
              );
            })}
          </div>
          <button type="button" className="add-feature-btn" onClick={addFeature}>+ Add Feature</button>
        </CollapsibleSection>
      )}
    </>
  );
};

function FeatureSummary({ feature, index }) {
  return (
    <div className="feature-summary">
      <strong>Feature {index + 1}:</strong> {feature.column_name || '(unnamed)'}
      {feature.type === 'text' && ` — ${feature.embedding_method} (${feature.embedding_dim}D)`}
      {feature.type === 'numeric' && ` — ${feature.processing}`}
    </div>
  );
}

function FeatureEditForm({ feature, index, columns, updateFeature }) {
  return (
    <div className="feature-edit-form">
      <div className="form-group">
        <label>Node ID Column:
          <select value={feature.node_id_column} onChange={e => updateFeature(index, 'node_id_column', e.target.value)}>
            <option value="">--Select--</option>
            {columns.map(c => <option key={c} value={c}>{c}</option>)}
          </select>
        </label>
      </div>
      <div className="form-group">
        <label>Feature Column:
          <select value={feature.column_name} onChange={e => updateFeature(index, 'column_name', e.target.value)}>
            <option value="">--Select--</option>
            {columns.map(c => <option key={c} value={c}>{c}</option>)}
          </select>
        </label>
      </div>
      <div className="form-group">
        <label>Type:
          <select value={feature.type} onChange={e => updateFeature(index, 'type', e.target.value)}>
            <option value="text">Text</option>
            <option value="numeric">Numeric</option>
          </select>
        </label>
      </div>
      {feature.type === 'text' && <TextFields feature={feature} index={index} updateFeature={updateFeature} />}
      {feature.type === 'numeric' && <NumericFields feature={feature} index={index} updateFeature={updateFeature} />}
    </div>
  );
}

function TextFields({ feature, index, updateFeature }) {
  return (
    <>
      <div className="form-group">
        <label>Method:
          <select value={feature.embedding_method} onChange={e => updateFeature(index, 'embedding_method', e.target.value)}>
            <option value="bert">BERT</option><option value="glove">GloVe</option><option value="word2vec">Word2Vec</option>
          </select>
        </label>
      </div>
      <div className="form-group">
        <label>Dimension:
          <input type="number" value={feature.embedding_dim || ''} onChange={e => updateFeature(index, 'embedding_dim', parseInt(e.target.value) || 0)} min="50" max="2048" />
        </label>
      </div>
      {feature.embedding_method === 'bert' && (
        <div className="form-group">
          <label>BERT Model:
            <input type="text" value={feature.additional_params?.bert_model_name || 'bert-base-uncased'} onChange={e => updateFeature(index, 'additional_params', { ...feature.additional_params, bert_model_name: e.target.value })} />
          </label>
        </div>
      )}
    </>
  );
}

function NumericFields({ feature, index, updateFeature }) {
  return (
    <>
      <div className="form-group">
        <label>Processing:
          <select value={feature.processing} onChange={e => updateFeature(index, 'processing', e.target.value)}>
            <option value="none">None</option><option value="standardize">Standardize</option><option value="normalize">Normalize</option>
          </select>
        </label>
      </div>
      <div className="form-group">
        <label>Projection:
          <select value={feature.projection?.method || 'none'} onChange={e => updateFeature(index, 'projection', { ...feature.projection, method: e.target.value })}>
            <option value="none">None</option><option value="linear">Linear</option>
          </select>
        </label>
      </div>
    </>
  );
}

export default ConfigurationPanel;
