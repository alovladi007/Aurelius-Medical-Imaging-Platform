import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, Loader, CheckCircle, AlertCircle, Download, Trash2 } from 'lucide-react';
import { api } from '../services/api';

const BatchProcessing = () => {
  const [files, setFiles] = useState([]);
  const [processing, setProcessing] = useState(false);
  const [results, setResults] = useState(null);
  const [progress, setProgress] = useState(0);

  const onDrop = useCallback((acceptedFiles) => {
    const newFiles = acceptedFiles.map((file) => ({
      file,
      id: Date.now() + Math.random(),
      status: 'pending',
    }));
    setFiles((prev) => [...prev, ...newFiles]);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.dcm'],
      'application/dicom': ['.dcm'],
    },
    multiple: true,
  });

  const removeFile = (id) => {
    setFiles((prev) => prev.filter((f) => f.id !== id));
  };

  const clearAll = () => {
    setFiles([]);
    setResults(null);
    setProgress(0);
  };

  const processBatch = async () => {
    if (files.length === 0) return;

    setProcessing(true);
    setProgress(0);

    try {
      const imageFiles = files.map((f) => f.file);
      const batchResults = await api.batchPredict(imageFiles);

      setResults(batchResults.results);
      setProgress(100);
    } catch (error) {
      console.error('Batch processing error:', error);
      alert('Batch processing failed. Please try again.');
    } finally {
      setProcessing(false);
    }
  };

  const exportResults = () => {
    if (!results) return;

    const exportData = {
      timestamp: new Date().toISOString(),
      totalFiles: files.length,
      results: results,
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `batch-results-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const exportCSV = () => {
    if (!results) return;

    const headers = ['Filename', 'Cancer Type', 'Confidence', 'Risk Score', 'Status'];
    const rows = results.map((r) => [
      r.filename,
      r.prediction?.cancer_type || 'Error',
      r.prediction?.confidence || 'N/A',
      r.prediction?.risk_score || 'N/A',
      r.error ? 'Failed' : 'Success',
    ]);

    const csv = [headers, ...rows].map((row) => row.join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `batch-results-${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="max-w-6xl mx-auto space-y-6 animate-fade-in">
      {/* Info Banner */}
      <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
        <h3 className="font-semibold text-purple-900 mb-2">Batch Processing</h3>
        <p className="text-sm text-purple-800">
          Upload multiple medical images for simultaneous AI analysis. This feature is
          perfect for processing large datasets or multiple patient scans at once.
        </p>
      </div>

      {/* Upload Zone */}
      <div className="card">
        <h2 className="text-lg font-semibold text-gray-800 mb-4">Upload Images</h2>

        <div
          {...getRootProps()}
          className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-all ${
            isDragActive
              ? 'border-purple-500 bg-purple-50'
              : 'border-gray-300 hover:border-purple-400 hover:bg-gray-50'
          }`}
        >
          <input {...getInputProps()} />
          <Upload className="w-16 h-16 mx-auto text-gray-400 mb-3" />
          <p className="text-lg font-medium text-gray-700">
            Drop images here or click to browse
          </p>
          <p className="text-sm text-gray-500 mt-1">
            You can select multiple files at once
          </p>
        </div>
      </div>

      {/* File List */}
      {files.length > 0 && (
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-gray-800">
              Selected Files ({files.length})
            </h2>
            <button onClick={clearAll} className="text-red-600 hover:text-red-700 text-sm flex items-center">
              <Trash2 className="w-4 h-4 mr-1" />
              Clear All
            </button>
          </div>

          <div className="max-h-96 overflow-y-auto space-y-2">
            {files.map((item) => (
              <div
                key={item.id}
                className="flex items-center justify-between p-3 bg-gray-50 rounded-lg"
              >
                <div className="flex items-center space-x-3 flex-1">
                  <CheckCircle className="w-5 h-5 text-green-600" />
                  <div>
                    <p className="font-medium text-gray-800">{item.file.name}</p>
                    <p className="text-sm text-gray-500">
                      {(item.file.size / 1024 / 1024).toFixed(2)} MB
                    </p>
                  </div>
                </div>
                <button
                  onClick={() => removeFile(item.id)}
                  className="p-2 text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                  disabled={processing}
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            ))}
          </div>

          {/* Progress Bar */}
          {processing && (
            <div className="mt-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-600">Processing...</span>
                <span className="text-sm font-medium text-gray-800">{progress}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className="bg-purple-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex justify-end space-x-3 mt-4">
            <button
              onClick={clearAll}
              className="btn btn-secondary"
              disabled={processing}
            >
              Clear
            </button>
            <button
              onClick={processBatch}
              className="btn btn-primary flex items-center"
              disabled={processing || files.length === 0}
            >
              {processing ? (
                <>
                  <Loader className="w-5 h-5 mr-2 animate-spin" />
                  Processing {files.length} files...
                </>
              ) : (
                <>
                  <Upload className="w-5 h-5 mr-2" />
                  Process Batch
                </>
              )}
            </button>
          </div>
        </div>
      )}

      {/* Results */}
      {results && (
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-gray-800">
              Processing Results ({results.length})
            </h2>
            <div className="flex space-x-2">
              <button onClick={exportCSV} className="btn btn-secondary text-sm">
                <Download className="w-4 h-4 mr-1" />
                Export CSV
              </button>
              <button onClick={exportResults} className="btn btn-primary text-sm">
                <Download className="w-4 h-4 mr-1" />
                Export JSON
              </button>
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-50 border-b border-gray-200">
                <tr>
                  <th className="px-4 py-3 text-left text-sm font-semibold text-gray-700">
                    File
                  </th>
                  <th className="px-4 py-3 text-left text-sm font-semibold text-gray-700">
                    Cancer Type
                  </th>
                  <th className="px-4 py-3 text-left text-sm font-semibold text-gray-700">
                    Confidence
                  </th>
                  <th className="px-4 py-3 text-left text-sm font-semibold text-gray-700">
                    Risk Score
                  </th>
                  <th className="px-4 py-3 text-left text-sm font-semibold text-gray-700">
                    Status
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {results.map((result, index) => (
                  <tr key={index} className="hover:bg-gray-50">
                    <td className="px-4 py-3 text-sm text-gray-800">
                      {result.filename}
                    </td>
                    {result.error ? (
                      <>
                        <td colSpan="3" className="px-4 py-3 text-sm text-red-600">
                          <div className="flex items-center">
                            <AlertCircle className="w-4 h-4 mr-2" />
                            {result.error}
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <span className="badge badge-danger">Failed</span>
                        </td>
                      </>
                    ) : (
                      <>
                        <td className="px-4 py-3 text-sm font-medium text-gray-900">
                          {result.prediction?.cancer_type}
                        </td>
                        <td className="px-4 py-3 text-sm text-gray-700">
                          {(result.prediction?.confidence * 100).toFixed(1)}%
                        </td>
                        <td className="px-4 py-3 text-sm text-gray-700">
                          {(result.prediction?.risk_score * 100).toFixed(1)}%
                        </td>
                        <td className="px-4 py-3">
                          <span className="badge badge-success flex items-center w-fit">
                            <CheckCircle className="w-3 h-3 mr-1" />
                            Success
                          </span>
                        </td>
                      </>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Summary Statistics */}
          <div className="mt-6 grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="bg-blue-50 rounded-lg p-4">
              <p className="text-sm text-blue-600 mb-1">Total Processed</p>
              <p className="text-2xl font-bold text-blue-900">{results.length}</p>
            </div>
            <div className="bg-green-50 rounded-lg p-4">
              <p className="text-sm text-green-600 mb-1">Successful</p>
              <p className="text-2xl font-bold text-green-900">
                {results.filter((r) => !r.error).length}
              </p>
            </div>
            <div className="bg-red-50 rounded-lg p-4">
              <p className="text-sm text-red-600 mb-1">Failed</p>
              <p className="text-2xl font-bold text-red-900">
                {results.filter((r) => r.error).length}
              </p>
            </div>
            <div className="bg-purple-50 rounded-lg p-4">
              <p className="text-sm text-purple-600 mb-1">Success Rate</p>
              <p className="text-2xl font-bold text-purple-900">
                {((results.filter((r) => !r.error).length / results.length) * 100).toFixed(1)}%
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default BatchProcessing;
