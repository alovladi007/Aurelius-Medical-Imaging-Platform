import React from 'react';
import {
  AlertTriangle,
  CheckCircle,
  TrendingUp,
  Activity,
  FileText,
  ArrowLeft,
  Download,
} from 'lucide-react';
import {
  RadialBarChart,
  RadialBar,
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Legend,
  Tooltip,
} from 'recharts';

const PredictionResults = ({ result, onReset }) => {
  const getRiskLevel = (score) => {
    if (score > 0.7) return { label: 'High Risk', color: 'red', icon: AlertTriangle };
    if (score > 0.4) return { label: 'Moderate Risk', color: 'yellow', icon: Activity };
    return { label: 'Low Risk', color: 'green', icon: CheckCircle };
  };

  const risk = getRiskLevel(result.risk_score);
  const RiskIcon = risk.icon;

  // Prepare probability data for chart
  const probabilityData = result.all_probabilities
    ? Object.entries(result.all_probabilities).map(([name, value]) => ({
        name,
        value: (value * 100).toFixed(2),
        fullValue: value,
      }))
    : [];

  const COLORS = ['#ef4444', '#f59e0b', '#10b981', '#3b82f6', '#8b5cf6'];

  // Confidence data for radial chart
  const confidenceData = [
    {
      name: 'Confidence',
      value: result.confidence * 100,
      fill: result.confidence > 0.7 ? '#10b981' : result.confidence > 0.4 ? '#f59e0b' : '#ef4444',
    },
  ];

  const handleExport = () => {
    const exportData = {
      timestamp: new Date().toISOString(),
      fileName: result.fileName,
      prediction: result.cancer_type,
      confidence: result.confidence,
      riskScore: result.risk_score,
      uncertainty: result.uncertainty,
      probabilities: result.all_probabilities,
      recommendations: result.recommendations,
      clinicalData: result.clinicalData,
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `prediction-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between">
        <button onClick={onReset} className="btn btn-secondary flex items-center">
          <ArrowLeft className="w-5 h-5 mr-2" />
          New Prediction
        </button>
        <button onClick={handleExport} className="btn btn-primary flex items-center">
          <Download className="w-5 h-5 mr-2" />
          Export Results
        </button>
      </div>

      {/* Main Result Card */}
      <div className="card bg-gradient-to-r from-primary-50 to-blue-50 border-2 border-primary-200">
        <div className="text-center">
          <h2 className="text-2xl font-bold text-gray-800 mb-2">Prediction Result</h2>
          <div className="inline-flex items-center space-x-3 bg-white px-6 py-3 rounded-full shadow-md">
            <div className={`w-3 h-3 bg-${risk.color}-500 rounded-full animate-pulse`}></div>
            <span className="text-3xl font-bold text-gray-900">{result.cancer_type}</span>
          </div>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Confidence */}
        <div className="card text-center">
          <Activity className="w-12 h-12 mx-auto mb-3 text-primary-600" />
          <p className="text-sm text-gray-600 mb-1">Confidence</p>
          <p className="text-4xl font-bold text-primary-600">
            {(result.confidence * 100).toFixed(1)}%
          </p>
        </div>

        {/* Risk Score */}
        <div className={`card text-center border-2 border-${risk.color}-200 bg-${risk.color}-50`}>
          <RiskIcon className={`w-12 h-12 mx-auto mb-3 text-${risk.color}-600`} />
          <p className="text-sm text-gray-600 mb-1">Risk Assessment</p>
          <p className={`text-4xl font-bold text-${risk.color}-600`}>{risk.label}</p>
          <p className="text-sm text-gray-600 mt-1">
            Score: {(result.risk_score * 100).toFixed(1)}%
          </p>
        </div>

        {/* Uncertainty */}
        <div className="card text-center">
          <TrendingUp className="w-12 h-12 mx-auto mb-3 text-purple-600" />
          <p className="text-sm text-gray-600 mb-1">Uncertainty</p>
          <p className="text-4xl font-bold text-purple-600">
            {result.uncertainty ? result.uncertainty.toFixed(3) : 'N/A'}
          </p>
        </div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Confidence Gauge */}
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-800 mb-4 text-center">
            Confidence Level
          </h3>
          <ResponsiveContainer width="100%" height={250}>
            <RadialBarChart
              cx="50%"
              cy="50%"
              innerRadius="60%"
              outerRadius="90%"
              data={confidenceData}
              startAngle={180}
              endAngle={0}
            >
              <RadialBar
                minAngle={15}
                background
                clockWise
                dataKey="value"
                cornerRadius={10}
              />
              <text
                x="50%"
                y="50%"
                textAnchor="middle"
                dominantBaseline="middle"
                className="text-3xl font-bold"
              >
                {confidenceData[0].value.toFixed(1)}%
              </text>
              <Tooltip />
            </RadialBarChart>
          </ResponsiveContainer>
        </div>

        {/* Probability Distribution */}
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-800 mb-4 text-center">
            Cancer Type Probabilities
          </h3>
          <ResponsiveContainer width="100%" height={250}>
            <PieChart>
              <Pie
                data={probabilityData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, value }) => `${name}: ${value}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {probabilityData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip formatter={(value) => `${value}%`} />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Detailed Probabilities */}
      {probabilityData.length > 0 && (
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">
            Detailed Probability Breakdown
          </h3>
          <div className="space-y-3">
            {probabilityData
              .sort((a, b) => b.fullValue - a.fullValue)
              .map((item, index) => (
                <div key={index}>
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-sm font-medium text-gray-700">{item.name}</span>
                    <span className="text-sm font-bold text-gray-900">{item.value}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className="h-2 rounded-full transition-all duration-500"
                      style={{
                        width: `${item.value}%`,
                        backgroundColor: COLORS[index % COLORS.length],
                      }}
                    />
                  </div>
                </div>
              ))}
          </div>
        </div>
      )}

      {/* Recommendations */}
      <div className="card bg-blue-50 border border-blue-200">
        <div className="flex items-start">
          <FileText className="w-6 h-6 text-blue-600 mr-3 mt-1" />
          <div className="flex-1">
            <h3 className="text-lg font-semibold text-blue-900 mb-3">
              Clinical Recommendations
            </h3>
            <ul className="space-y-2">
              {result.recommendations?.map((rec, index) => (
                <li key={index} className="flex items-start">
                  <span className="inline-block w-2 h-2 bg-blue-600 rounded-full mt-2 mr-3 flex-shrink-0" />
                  <span className="text-blue-800">{rec}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>

      {/* Disclaimer */}
      <div className="card bg-red-50 border-2 border-red-200">
        <div className="flex items-start">
          <AlertTriangle className="w-6 h-6 text-red-600 mr-3 mt-1" />
          <div>
            <h3 className="font-semibold text-red-900 mb-2">Medical Disclaimer</h3>
            <p className="text-sm text-red-800">
              This AI system is for research and educational purposes only. Results should
              NOT be used for clinical diagnosis or treatment decisions without proper
              validation by qualified healthcare professionals. Always consult with a
              licensed medical practitioner for health-related concerns.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PredictionResults;
