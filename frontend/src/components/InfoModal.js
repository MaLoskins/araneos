import React from 'react';
import { Dialog, DialogTitle, DialogContent, DialogActions, Button } from '@mui/material';

const InfoModal = ({ isOpen, onRequestClose, title, description }) => (
  <Dialog open={isOpen} onClose={onRequestClose} maxWidth="sm" fullWidth>
    <DialogTitle>{title}</DialogTitle>
    <DialogContent>
      <div dangerouslySetInnerHTML={{ __html: description }} />
    </DialogContent>
    <DialogActions>
      <Button onClick={onRequestClose} variant="contained" size="small"
        sx={{ bgcolor: 'var(--button-background)', '&:hover': { bgcolor: 'var(--button-hover)' } }}>
        Close
      </Button>
    </DialogActions>
  </Dialog>
);

export default InfoModal;
