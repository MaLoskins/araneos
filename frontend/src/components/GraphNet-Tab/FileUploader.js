import React, { useState } from 'react';
import { useDropzone } from 'react-dropzone';
import Papa from 'papaparse';
import { FiUpload, FiFile } from 'react-icons/fi';

const FileUploader = ({ onFileDrop, hasFile }) => {
  const [fileName, setFileName] = useState(null);

  const onDrop = (acceptedFiles) => {
    if (!acceptedFiles.length) return;
    const file = acceptedFiles[0];
    setFileName(file.name);

    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      complete: (results) => onFileDrop(results.data, results.meta.fields),
      error: (error) => {
        console.error('Error parsing CSV:', error);
        alert('Error parsing CSV file.');
      },
    });
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'text/csv': ['.csv'] },
    multiple: false,
  });

  if (hasFile && fileName) {
    return (
      <div className="file-uploader">
        <div className="file-uploader-compact" {...getRootProps()}>
          <input {...getInputProps()} data-testid="file-input" />
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <FiFile size={14} />
            <span className="file-name">{fileName}</span>
          </div>
          <span style={{ color: 'var(--text-muted)', fontSize: 'var(--text-xs)' }}>Click to change</span>
        </div>
      </div>
    );
  }

  return (
    <div className="file-uploader">
      <div {...getRootProps()} className={`dropzone ${isDragActive ? 'active' : ''}`}>
        <input {...getInputProps()} data-testid="file-input" />
        <FiUpload size={20} style={{ marginBottom: 8, opacity: 0.5 }} />
        <p>{isDragActive ? 'Drop CSV here...' : 'Drop CSV file or click to browse'}</p>
      </div>
    </div>
  );
};

export default FileUploader;
