import React from 'react';
import { FiCheck } from 'react-icons/fi';

function WorkflowStepper({ steps, currentStep }) {
  return (
    <div className="workflow-stepper">
      {steps.map((label, i) => {
        const isCompleted = i < currentStep;
        const isCurrent = i === currentStep;
        return (
          <React.Fragment key={i}>
            {i > 0 && <div className={`stepper-line ${isCompleted ? 'completed' : ''}`} />}
            <div className={`stepper-step ${isCompleted ? 'completed' : ''} ${isCurrent ? 'current' : ''}`}>
              <div className="stepper-circle">
                {isCompleted ? <FiCheck size={12} /> : <span>{i + 1}</span>}
              </div>
              <span className="stepper-label">{label}</span>
            </div>
          </React.Fragment>
        );
      })}
    </div>
  );
}

export default WorkflowStepper;
