import React, { useState, useEffect } from 'react';
import { Dialog, DialogTitle, DialogContent, DialogActions, Button } from '@mui/material';

const NodeEditModal = ({ isOpen, onRequestClose, node, onSaveNodeEdit }) => {
  const [nodeType, setNodeType] = useState(node.type || '');
  const [nodeFeatures, setNodeFeatures] = useState(node.features || []);

  useEffect(() => {
    if (isOpen) {
      setNodeType(node.type || '');
      setNodeFeatures(node.features || []);
    }
  }, [isOpen, node]);

  const handleSubmit = (e) => {
    e.preventDefault();
    onSaveNodeEdit({ nodeType, nodeFeatures });
  };

  return (
    <Dialog open={isOpen} onClose={onRequestClose} maxWidth="xs" fullWidth>
      <DialogTitle>Edit Node: {node.id}</DialogTitle>
      <form onSubmit={handleSubmit}>
        <DialogContent>
          <div className="form-group">
            <label htmlFor="node-type">Type:</label>
            <input type="text" id="node-type" value={nodeType} onChange={e => setNodeType(e.target.value)} placeholder="e.g., User, Post" />
          </div>
        </DialogContent>
        <DialogActions>
          <Button type="button" onClick={onRequestClose} sx={{ color: 'var(--text-secondary)' }}>Cancel</Button>
          <Button type="submit" variant="contained" sx={{ bgcolor: 'var(--button-background)', '&:hover': { bgcolor: 'var(--button-hover)' } }}>Save</Button>
        </DialogActions>
      </form>
    </Dialog>
  );
};

export default NodeEditModal;
