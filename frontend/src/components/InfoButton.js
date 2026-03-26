import React, { useState } from 'react';
import { FiInfo } from 'react-icons/fi';
import InfoModal from './InfoModal';

const InfoButton = ({ title, description }) => {
  const [open, setOpen] = useState(false);
  return (
    <span className="info-button-container">
      <FiInfo onClick={() => setOpen(true)} className="info-icon" />
      <InfoModal isOpen={open} onRequestClose={() => setOpen(false)} title={title} description={description} />
    </span>
  );
};

export default InfoButton;
