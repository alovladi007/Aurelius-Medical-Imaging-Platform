import React, { useMemo } from 'react';
import { TrendingUp, PieChart, BarChart3, Activity } from 'lucide-react';
import useStore from '../store/useStore';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  PieChart as RPieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from 'recharts';
import { format, subDays, parseISO } from 'date-fns';

const Analytics = () => {
  const { predictions } = useStore();

  // Calculate analytics
  const analytics = useMemo(() => {
    if (predictions.length === 0) return null;

    // Cancer type distribution
    const typeDistribution = predictions.reduce((acc, pred) => {
      const type = pred.cancer_type || 'Unknown';
      acc[type] = (acc[type] || 0) + 1;
      return acc;
    }, {});

    // Risk distribution
    const riskDistribution = {
      'Low Risk': predictions.filter((p) => p.risk_score <= 0.4).length,
      'Moderate Risk': predictions.filter((p) => p.risk_score > 0.4 && p.risk_score <= 0.7)
        .length,
      'High Risk': predictions.filter((p) => p.risk_score > 0.7).length,
    };

    // Confidence distribution
    const confidenceRanges = {
      '0-20%': predictions.filter((p) => p.confidence <= 0.2).length,
      '20-40%': predictions.filter((p) => p.confidence > 0.2 && p.confidence <= 0.4).length,
      '40-60%': predictions.filter((p) => p.confidence > 0.4 && p.confidence <= 0.6).length,
      '60-80%': predictions.filter((p) => p.confidence > 0.6 && p.confidence <= 0.8).length,
      '80-100%': predictions.filter((p) => p.confidence > 0.8).length,
    };

    // Timeline data (last 7 days)
    const timelineData = [];
    for (let i = 6; i >= 0; i--) {
      const date = subDays(new Date(), i);
      const dateStr = format(date, 'yyyy-MM-dd');
      const count = predictions.filter((p) => {
        const predDate = format(parseISO(p.timestamp), 'yyyy-MM-dd');
        return predDate === dateStr;
      }).length;

      timelineData.push({
        date: format(date, 'MMM dd'),
        predictions: count,
      });
    }

    // Average metrics
    const avgConfidence =
      predictions.reduce((sum, p) => sum + p.confidence, 0) / predictions.length;
    const avgRisk =
      predictions.reduce((sum, p) => sum + p.risk_score, 0) / predictions.length;
    const avgUncertainty =
      predictions.reduce((sum, p) => sum + (p.uncertainty || 0), 0) / predictions.length;

    // Performance by cancer type
    const performanceByType = Object.keys(typeDistribution).map((type) => {
      const typePredictions = predictions.filter((p) => p.cancer_type === type);
      const avgConf =
        typePredictions.reduce((sum, p) => sum + p.confidence, 0) / typePredictions.length;
      return {
        type,
        avgConfidence: avgConf * 100,
        count: typePredictions.length,
      };
    });

    return {
      typeDistribution,
      riskDistribution,
      confidenceRanges,
      timelineData,
      avgConfidence,
      avgRisk,
      avgUncertainty,
      performanceByType,
    };
  }, [predictions]);

  if (!analytics || predictions.length === 0) {
    return (
      <div className="card text-center py-12 animate-fade-in">
        <BarChart3 className="w-16 h-16 mx-auto text-gray-400 mb-4" />
        <p className="text-gray-600 text-lg mb-2">No analytics data available</p>
        <p className="text-gray-500 text-sm">
          Make some predictions to see analytics and insights
        </p>
      </div>
    );
  }

  const COLORS = ['#ef4444', '#f59e0b', '#10b981', '#3b82f6', '#8b5cf6'];

  // Prepare chart data
  const typeData = Object.entries(analytics.typeDistribution).map(([name, value]) => ({
    name,
    value,
  }));

  const riskData = Object.entries(analytics.riskDistribution).map(([name, value]) => ({
    name,
    value,
  }));

  const confidenceData = Object.entries(analytics.confidenceRanges).map(([name, value]) => ({
    name,
    value,
  }));

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 mb-1">Total Predictions</p>
              <p className="text-3xl font-bold text-gray-900">{predictions.length}</p>
            </div>
            <Activity className="w-12 h-12 text-primary-600 opacity-20" />
          </div>
        </div>

        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 mb-1">Avg Confidence</p>
              <p className="text-3xl font-bold text-green-600">
                {(analytics.avgConfidence * 100).toFixed(1)}%
              </p>
            </div>
            <TrendingUp className="w-12 h-12 text-green-600 opacity-20" />
          </div>
        </div>

        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 mb-1">Avg Risk Score</p>
              <p className="text-3xl font-bold text-orange-600">
                {(analytics.avgRisk * 100).toFixed(1)}%
              </p>
            </div>
            <BarChart3 className="w-12 h-12 text-orange-600 opacity-20" />
          </div>
        </div>

        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 mb-1">Avg Uncertainty</p>
              <p className="text-3xl font-bold text-purple-600">
                {analytics.avgUncertainty.toFixed(3)}
              </p>
            </div>
            <PieChart className="w-12 h-12 text-purple-600 opacity-20" />
          </div>
        </div>
      </div>

      {/* Timeline Chart */}
      <div className="card">
        <h2 className="text-lg font-semibold text-gray-800 mb-4">
          Prediction Activity (Last 7 Days)
        </h2>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={analytics.timelineData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line
              type="monotone"
              dataKey="predictions"
              stroke="#0ea5e9"
              strokeWidth={2}
              dot={{ fill: '#0ea5e9', r: 4 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Distribution Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Cancer Type Distribution */}
        <div className="card">
          <h2 className="text-lg font-semibold text-gray-800 mb-4">
            Cancer Type Distribution
          </h2>
          <ResponsiveContainer width="100%" height={300}>
            <RPieChart>
              <Pie
                data={typeData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                outerRadius={100}
                fill="#8884d8"
                dataKey="value"
              >
                {typeData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
            </RPieChart>
          </ResponsiveContainer>
        </div>

        {/* Risk Distribution */}
        <div className="card">
          <h2 className="text-lg font-semibold text-gray-800 mb-4">Risk Distribution</h2>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={riskData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="value" fill="#0ea5e9" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Confidence Distribution */}
        <div className="card">
          <h2 className="text-lg font-semibold text-gray-800 mb-4">
            Confidence Distribution
          </h2>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={confidenceData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="value" fill="#10b981" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Performance Radar */}
        <div className="card">
          <h2 className="text-lg font-semibold text-gray-800 mb-4">
            Performance by Cancer Type
          </h2>
          <ResponsiveContainer width="100%" height={300}>
            <RadarChart data={analytics.performanceByType}>
              <PolarGrid />
              <PolarAngleAxis dataKey="type" />
              <PolarRadiusAxis angle={90} domain={[0, 100]} />
              <Radar
                name="Avg Confidence %"
                dataKey="avgConfidence"
                stroke="#8884d8"
                fill="#8884d8"
                fillOpacity={0.6}
              />
              <Tooltip />
              <Legend />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Detailed Statistics Table */}
      <div className="card">
        <h2 className="text-lg font-semibold text-gray-800 mb-4">
          Detailed Statistics by Cancer Type
        </h2>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50 border-b border-gray-200">
              <tr>
                <th className="px-4 py-3 text-left text-sm font-semibold text-gray-700">
                  Cancer Type
                </th>
                <th className="px-4 py-3 text-left text-sm font-semibold text-gray-700">
                  Count
                </th>
                <th className="px-4 py-3 text-left text-sm font-semibold text-gray-700">
                  Avg Confidence
                </th>
                <th className="px-4 py-3 text-left text-sm font-semibold text-gray-700">
                  Percentage
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {analytics.performanceByType.map((item, index) => (
                <tr key={index} className="hover:bg-gray-50">
                  <td className="px-4 py-3 text-sm font-medium text-gray-900">
                    {item.type}
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-700">{item.count}</td>
                  <td className="px-4 py-3 text-sm text-gray-700">
                    {item.avgConfidence.toFixed(1)}%
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-700">
                    {((item.count / predictions.length) * 100).toFixed(1)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default Analytics;
