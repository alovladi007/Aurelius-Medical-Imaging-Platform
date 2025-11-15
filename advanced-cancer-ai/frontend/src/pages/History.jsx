import React, { useState } from 'react';
import { Search, Trash2, Download, Calendar, Filter, ChevronDown } from 'lucide-react';
import useStore from '../store/useStore';
import { format } from 'date-fns';

const History = () => {
  const { predictions, deletePrediction, clearPredictions } = useStore();
  const [searchTerm, setSearchTerm] = useState('');
  const [filterType, setFilterType] = useState('all');
  const [sortBy, setSortBy] = useState('date');
  const [showFilters, setShowFilters] = useState(false);

  // Get unique cancer types
  const cancerTypes = [...new Set(predictions.map((p) => p.cancer_type))];

  // Filter and sort predictions
  const filteredPredictions = predictions
    .filter((pred) => {
      const matchesSearch =
        pred.cancer_type?.toLowerCase().includes(searchTerm.toLowerCase()) ||
        pred.fileName?.toLowerCase().includes(searchTerm.toLowerCase());
      const matchesType = filterType === 'all' || pred.cancer_type === filterType;
      return matchesSearch && matchesType;
    })
    .sort((a, b) => {
      switch (sortBy) {
        case 'date':
          return new Date(b.timestamp) - new Date(a.timestamp);
        case 'confidence':
          return b.confidence - a.confidence;
        case 'risk':
          return b.risk_score - a.risk_score;
        default:
          return 0;
      }
    });

  const exportHistory = () => {
    const exportData = {
      exportDate: new Date().toISOString(),
      totalPredictions: predictions.length,
      predictions: predictions,
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `prediction-history-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleClearAll = () => {
    if (
      window.confirm(
        'Are you sure you want to clear all prediction history? This action cannot be undone.'
      )
    ) {
      clearPredictions();
    }
  };

  return (
    <div className="max-w-7xl mx-auto space-y-6 animate-fade-in">
      {/* Header Actions */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-800">Prediction History</h1>
          <p className="text-gray-600">{predictions.length} total predictions</p>
        </div>
        <div className="flex space-x-2">
          <button onClick={exportHistory} className="btn btn-secondary text-sm">
            <Download className="w-4 h-4 mr-1" />
            Export
          </button>
          <button onClick={handleClearAll} className="btn btn-danger text-sm">
            <Trash2 className="w-4 h-4 mr-1" />
            Clear All
          </button>
        </div>
      </div>

      {/* Search and Filters */}
      <div className="card">
        <div className="flex flex-col sm:flex-row gap-4">
          {/* Search */}
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
            <input
              type="text"
              placeholder="Search predictions..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="input pl-10"
            />
          </div>

          {/* Filter Toggle */}
          <button
            onClick={() => setShowFilters(!showFilters)}
            className="btn btn-secondary flex items-center justify-center"
          >
            <Filter className="w-4 h-4 mr-2" />
            Filters
            <ChevronDown
              className={`w-4 h-4 ml-2 transition-transform ${
                showFilters ? 'rotate-180' : ''
              }`}
            />
          </button>
        </div>

        {/* Filter Options */}
        {showFilters && (
          <div className="mt-4 pt-4 border-t border-gray-200 grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <label className="label">Filter by Cancer Type</label>
              <select
                value={filterType}
                onChange={(e) => setFilterType(e.target.value)}
                className="input"
              >
                <option value="all">All Types</option>
                {cancerTypes.map((type) => (
                  <option key={type} value={type}>
                    {type}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="label">Sort By</label>
              <select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value)}
                className="input"
              >
                <option value="date">Date (Newest First)</option>
                <option value="confidence">Confidence (Highest First)</option>
                <option value="risk">Risk Score (Highest First)</option>
              </select>
            </div>
          </div>
        )}
      </div>

      {/* Predictions List */}
      {filteredPredictions.length > 0 ? (
        <div className="space-y-4">
          {filteredPredictions.map((prediction) => (
            <div key={prediction.id} className="card hover:shadow-lg transition-shadow">
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center space-x-3 mb-2">
                    <h3 className="text-lg font-semibold text-gray-800">
                      {prediction.cancer_type}
                    </h3>
                    <span
                      className={`badge ${
                        prediction.risk_score > 0.7
                          ? 'badge-danger'
                          : prediction.risk_score > 0.4
                          ? 'badge-warning'
                          : 'badge-success'
                      }`}
                    >
                      Risk: {(prediction.risk_score * 100).toFixed(0)}%
                    </span>
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-3">
                    <div>
                      <p className="text-xs text-gray-500">File Name</p>
                      <p className="text-sm font-medium text-gray-800">
                        {prediction.fileName || 'N/A'}
                      </p>
                    </div>
                    <div>
                      <p className="text-xs text-gray-500">Confidence</p>
                      <p className="text-sm font-medium text-gray-800">
                        {(prediction.confidence * 100).toFixed(1)}%
                      </p>
                    </div>
                    <div>
                      <p className="text-xs text-gray-500">Uncertainty</p>
                      <p className="text-sm font-medium text-gray-800">
                        {prediction.uncertainty?.toFixed(3) || 'N/A'}
                      </p>
                    </div>
                    <div>
                      <p className="text-xs text-gray-500">Date</p>
                      <p className="text-sm font-medium text-gray-800">
                        {format(new Date(prediction.timestamp), 'MMM dd, yyyy HH:mm')}
                      </p>
                    </div>
                  </div>

                  {/* Clinical Data */}
                  {prediction.clinicalData && (
                    <div className="bg-gray-50 rounded-lg p-3 mb-3">
                      <p className="text-xs text-gray-500 mb-2">Clinical Information</p>
                      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-sm">
                        {prediction.clinicalData.patient_age && (
                          <div>
                            <span className="text-gray-600">Age:</span>{' '}
                            <span className="font-medium">
                              {prediction.clinicalData.patient_age}
                            </span>
                          </div>
                        )}
                        {prediction.clinicalData.patient_gender && (
                          <div>
                            <span className="text-gray-600">Gender:</span>{' '}
                            <span className="font-medium capitalize">
                              {prediction.clinicalData.patient_gender}
                            </span>
                          </div>
                        )}
                        {prediction.clinicalData.smoking_history && (
                          <div>
                            <span className="text-gray-600">Smoking:</span>{' '}
                            <span className="font-medium">Yes</span>
                          </div>
                        )}
                        {prediction.clinicalData.family_history && (
                          <div>
                            <span className="text-gray-600">Family History:</span>{' '}
                            <span className="font-medium">Yes</span>
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Recommendations */}
                  {prediction.recommendations && prediction.recommendations.length > 0 && (
                    <div className="bg-blue-50 rounded-lg p-3">
                      <p className="text-xs text-blue-600 font-medium mb-2">
                        Recommendations
                      </p>
                      <ul className="space-y-1">
                        {prediction.recommendations.slice(0, 2).map((rec, idx) => (
                          <li key={idx} className="text-sm text-blue-800 flex items-start">
                            <span className="inline-block w-1.5 h-1.5 bg-blue-600 rounded-full mt-1.5 mr-2 flex-shrink-0" />
                            {rec}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>

                {/* Delete Button */}
                <button
                  onClick={() => {
                    if (window.confirm('Delete this prediction?')) {
                      deletePrediction(prediction.id);
                    }
                  }}
                  className="ml-4 p-2 text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                >
                  <Trash2 className="w-5 h-5" />
                </button>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="card text-center py-12">
          <Calendar className="w-16 h-16 mx-auto text-gray-400 mb-4" />
          <p className="text-gray-600 text-lg mb-2">No predictions found</p>
          <p className="text-gray-500 text-sm">
            {searchTerm || filterType !== 'all'
              ? 'Try adjusting your filters'
              : 'Start by making your first prediction'}
          </p>
        </div>
      )}
    </div>
  );
};

export default History;
