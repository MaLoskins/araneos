import React, { useState } from 'react';
import { FiChevronDown } from 'react-icons/fi';

function CollapsibleSection({ title, icon: Icon, defaultExpanded = true, children }) {
  const [expanded, setExpanded] = useState(defaultExpanded);

  return (
    <div className={`collapsible-section ${expanded ? 'expanded' : ''}`}>
      <button className="collapsible-header" onClick={() => setExpanded(!expanded)}>
        <div className="collapsible-title">
          {Icon && <Icon size={14} className="collapsible-icon" />}
          <span>{title}</span>
        </div>
        <FiChevronDown size={16} className={`collapsible-chevron ${expanded ? 'rotated' : ''}`} />
      </button>
      {expanded && <div className="collapsible-body">{children}</div>}
    </div>
  );
}

export default CollapsibleSection;
