import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, FileImage, X, AlertCircle, Loader, CheckCircle } from 'lucide-react';
import { api } from '../services/api';
import useStore from '../store/useStore';
import PredictionResults from '../components/PredictionResults';

const NewPrediction = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const { addPrediction } = useStore();

  const [clinicalData, setClinicalData] = useState({
    clinical_notes: '',
    patient_age: '',
    patient_gender: '',
    smoking_history: false,
    family_history: false,
  });

  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
      setSelectedFile(file);
      setError(null);
      setResult(null);

      // Create preview for images
      if (file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = (e) => setPreview(e.target.result);
        reader.readAsDataURL(file);
      } else {
        setPreview(null);
      }
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.dcm'],
      'application/dicom': ['.dcm'],
    },
    multiple: false,
  });

  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target;
    setClinicalData((prev) => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!selectedFile) {
      setError('Please select an image file');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const prediction = await api.predict(selectedFile, {
        ...clinicalData,
        patient_age: parseInt(clinicalData.patient_age) || 0,
      });

      setResult({
        ...prediction,
        fileName: selectedFile.name,
        clinicalData,
      });

      addPrediction({
        ...prediction,
        fileName: selectedFile.name,
        clinicalData,
      });
    } catch (err) {
      setError(err.response?.data?.detail || 'Prediction failed. Please try again.');
      console.error('Prediction error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    setClinicalData({
      clinical_notes: '',
      patient_age: '',
      patient_gender: '',
      smoking_history: false,
      family_history: false,
    });
  };

  if (result) {
    return (
      <div className="animate-fade-in">
        <PredictionResults result={result} onReset={handleReset} />
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto space-y-6 animate-fade-in">
      {/* Instructions */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <h3 className="font-semibold text-blue-900 mb-2">How to use:</h3>
        <ol className="text-sm text-blue-800 space-y-1 list-decimal list-inside">
          <li>Upload a medical image (DICOM, NIfTI, or standard image formats)</li>
          <li>Fill in patient clinical information (optional but recommended)</li>
          <li>Click "Analyze Image" to get AI-powered predictions</li>
        </ol>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Image Upload */}
        <div className="card">
          <h2 className="text-lg font-semibold text-gray-800 mb-4">
            Upload Medical Image
          </h2>

          <div
            {...getRootProps()}
            className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-all ${
              isDragActive
                ? 'border-primary-500 bg-primary-50'
                : 'border-gray-300 hover:border-primary-400 hover:bg-gray-50'
            }`}
          >
            <input {...getInputProps()} />
            {preview ? (
              <div className="space-y-4">
                <img
                  src={preview}
                  alt="Preview"
                  className="max-h-64 mx-auto rounded-lg shadow-md"
                />
                <div className="flex items-center justify-center space-x-2 text-green-600">
                  <CheckCircle className="w-5 h-5" />
                  <span className="font-medium">{selectedFile.name}</span>
                </div>
              </div>
            ) : selectedFile ? (
              <div className="space-y-2">
                <FileImage className="w-16 h-16 mx-auto text-primary-600" />
                <p className="font-medium text-gray-800">{selectedFile.name}</p>
                <p className="text-sm text-gray-500">
                  {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                </p>
              </div>
            ) : (
              <div className="space-y-3">
                <Upload className="w-16 h-16 mx-auto text-gray-400" />
                <div>
                  <p className="text-lg font-medium text-gray-700">
                    Drop medical image here or click to browse
                  </p>
                  <p className="text-sm text-gray-500 mt-1">
                    Supports DICOM, NIfTI, PNG, JPG, TIFF formats
                  </p>
                </div>
              </div>
            )}
          </div>

          {selectedFile && (
            <button
              type="button"
              onClick={() => {
                setSelectedFile(null);
                setPreview(null);
              }}
              className="mt-3 text-sm text-red-600 hover:text-red-700 flex items-center"
            >
              <X className="w-4 h-4 mr-1" />
              Remove file
            </button>
          )}
        </div>

        {/* Clinical Data */}
        <div className="card">
          <h2 className="text-lg font-semibold text-gray-800 mb-4">
            Clinical Information
            <span className="text-sm font-normal text-gray-500 ml-2">(Optional)</span>
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Age */}
            <div>
              <label className="label">Patient Age</label>
              <input
                type="number"
                name="patient_age"
                value={clinicalData.patient_age}
                onChange={handleInputChange}
                className="input"
                placeholder="e.g., 55"
                min="0"
                max="120"
              />
            </div>

            {/* Gender */}
            <div>
              <label className="label">Gender</label>
              <select
                name="patient_gender"
                value={clinicalData.patient_gender}
                onChange={handleInputChange}
                className="input"
              >
                <option value="">Select gender</option>
                <option value="male">Male</option>
                <option value="female">Female</option>
                <option value="other">Other</option>
              </select>
            </div>

            {/* Smoking History */}
            <div className="flex items-center">
              <input
                type="checkbox"
                id="smoking_history"
                name="smoking_history"
                checked={clinicalData.smoking_history}
                onChange={handleInputChange}
                className="w-4 h-4 text-primary-600 border-gray-300 rounded focus:ring-primary-500"
              />
              <label htmlFor="smoking_history" className="ml-2 text-sm text-gray-700">
                Smoking History
              </label>
            </div>

            {/* Family History */}
            <div className="flex items-center">
              <input
                type="checkbox"
                id="family_history"
                name="family_history"
                checked={clinicalData.family_history}
                onChange={handleInputChange}
                className="w-4 h-4 text-primary-600 border-gray-300 rounded focus:ring-primary-500"
              />
              <label htmlFor="family_history" className="ml-2 text-sm text-gray-700">
                Family History of Cancer
              </label>
            </div>
          </div>

          {/* Clinical Notes */}
          <div className="mt-4">
            <label className="label">Clinical Notes</label>
            <textarea
              name="clinical_notes"
              value={clinicalData.clinical_notes}
              onChange={handleInputChange}
              className="input"
              rows="4"
              placeholder="Enter any relevant clinical observations or patient history..."
            />
          </div>
        </div>

        {/* Error Message */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-start">
            <AlertCircle className="w-5 h-5 text-red-600 mr-3 mt-0.5" />
            <div>
              <p className="font-medium text-red-800">Prediction Error</p>
              <p className="text-sm text-red-600">{error}</p>
            </div>
          </div>
        )}

        {/* Submit Button */}
        <div className="flex justify-end space-x-3">
          <button
            type="button"
            onClick={handleReset}
            className="btn btn-secondary"
            disabled={loading}
          >
            Reset
          </button>
          <button
            type="submit"
            className="btn btn-primary flex items-center"
            disabled={loading || !selectedFile}
          >
            {loading ? (
              <>
                <Loader className="w-5 h-5 mr-2 animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                <Upload className="w-5 h-5 mr-2" />
                Analyze Image
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  );
};

export default NewPrediction;
