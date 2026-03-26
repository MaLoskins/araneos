import React, { useState } from 'react';
import { Dialog, DialogTitle, DialogContent, DialogActions, Button } from '@mui/material';

const RelationshipModal = ({ isOpen, onRequestClose, onSaveRelationship }) => {
  const [relationshipType, setRelationshipType] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    onSaveRelationship({ relationshipType });
  };

  return (
    <Dialog open={isOpen} onClose={onRequestClose} maxWidth="xs" fullWidth>
      <DialogTitle>Define Relationship</DialogTitle>
      <form onSubmit={handleSubmit}>
        <DialogContent>
          <div className="form-group">
            <label htmlFor="rel-type">Relationship Type:</label>
            <input type="text" id="rel-type" value={relationshipType} onChange={e => setRelationshipType(e.target.value)} placeholder="e.g., connects, influences" />
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

export default RelationshipModal;
