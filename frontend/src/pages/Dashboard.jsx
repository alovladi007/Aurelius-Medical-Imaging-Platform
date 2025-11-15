import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  TrendingUp,
  AlertCircle,
  CheckCircle,
  Activity,
  Clock,
  FileText,
} from 'lucide-react';
import { api } from '../services/api';
import useStore from '../store/useStore';
import {
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

const Dashboard = () => {
  const { predictions, getStatistics } = useStore();
  const [systemHealth, setSystemHealth] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    checkSystemHealth();
  }, []);

  const checkSystemHealth = async () => {
    try {
      const health = await api.healthCheck();
      setSystemHealth(health);
    } catch (error) {
      console.error('Health check failed:', error);
      setSystemHealth({ status: 'error' });
    } finally {
      setLoading(false);
    }
  };

  const stats = getStatistics();
  const recentPredictions = predictions.slice(0, 5);

  // Prepare chart data
  const typeData = Object.entries(stats.byType).map(([name, value]) => ({
    name,
    value,
  }));

  const COLORS = ['#ef4444', '#f59e0b', '#10b981', '#3b82f6', '#8b5cf6'];

  const StatCard = ({ icon: Icon, title, value, subtitle, color, trend }) => (
    <div className="card card-hover">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm text-gray-600 mb-1">{title}</p>
          <p className={`text-3xl font-bold ${color}`}>{value}</p>
          {subtitle && <p className="text-sm text-gray-500 mt-1">{subtitle}</p>}
          {trend && (
            <div className="flex items-center mt-2 text-sm text-green-600">
              <TrendingUp className="w-4 h-4 mr-1" />
              {trend}
            </div>
          )}
        </div>
        <div className={`p-4 rounded-full ${color.replace('text', 'bg').replace('600', '100')}`}>
          <Icon className={`w-8 h-8 ${color}`} />
        </div>
      </div>
    </div>
  );

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="spinner"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6 animate-fade-in">
      {/* System Status Alert */}
      {systemHealth?.status === 'healthy' ? (
        <div className="bg-green-50 border border-green-200 rounded-lg p-4 flex items-center">
          <CheckCircle className="w-5 h-5 text-green-600 mr-3" />
          <div>
            <p className="font-medium text-green-800">System Operational</p>
            <p className="text-sm text-green-600">
              Model loaded and ready for predictions
            </p>
          </div>
        </div>
      ) : (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-center">
          <AlertCircle className="w-5 h-5 text-red-600 mr-3" />
          <div>
            <p className="font-medium text-red-800">System Error</p>
            <p className="text-sm text-red-600">
              Unable to connect to the prediction server
            </p>
          </div>
        </div>
      )}

      {/* Statistics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          icon={FileText}
          title="Total Predictions"
          value={stats.total}
          subtitle="All time"
          color="text-primary-600"
        />
        <StatCard
          icon={Activity}
          title="Avg Confidence"
          value={`${(stats.avgConfidence * 100).toFixed(1)}%`}
          subtitle="Model confidence"
          color="text-green-600"
        />
        <StatCard
          icon={AlertCircle}
          title="High Risk Cases"
          value={stats.highRiskCount}
          subtitle="Risk score > 0.7"
          color="text-red-600"
        />
        <StatCard
          icon={Clock}
          title="Recent Activity"
          value={recentPredictions.length}
          subtitle="Last 24 hours"
          color="text-blue-600"
        />
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Cancer Type Distribution */}
        <div className="card">
          <h2 className="text-lg font-semibold text-gray-800 mb-4">
            Cancer Type Distribution
          </h2>
          {typeData.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={typeData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) =>
                    `${name}: ${(percent * 100).toFixed(0)}%`
                  }
                  outerRadius={100}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {typeData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-64 flex items-center justify-center text-gray-500">
              No data available
            </div>
          )}
        </div>

        {/* Prediction Trends */}
        <div className="card">
          <h2 className="text-lg font-semibold text-gray-800 mb-4">
            Prediction Trends
          </h2>
          {typeData.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={typeData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" angle={-45} textAnchor="end" height={100} />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="value" fill="#0ea5e9" />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-64 flex items-center justify-center text-gray-500">
              No data available
            </div>
          )}
        </div>
      </div>

      {/* Recent Predictions */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-gray-800">Recent Predictions</h2>
          <Link to="/history" className="text-primary-600 hover:text-primary-700 text-sm font-medium">
            View All
          </Link>
        </div>
        {recentPredictions.length > 0 ? (
          <div className="space-y-3">
            {recentPredictions.map((prediction) => (
              <div
                key={prediction.id}
                className="flex items-center justify-between p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
              >
                <div className="flex-1">
                  <p className="font-medium text-gray-800">{prediction.cancer_type}</p>
                  <p className="text-sm text-gray-500">
                    {new Date(prediction.timestamp).toLocaleString()}
                  </p>
                </div>
                <div className="flex items-center space-x-4">
                  <div>
                    <span className="text-sm text-gray-600">Confidence: </span>
                    <span className="font-medium">
                      {(prediction.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
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
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-12 text-gray-500">
            <FileText className="w-12 h-12 mx-auto mb-3 opacity-50" />
            <p>No predictions yet</p>
            <Link to="/predict" className="text-primary-600 hover:text-primary-700 text-sm font-medium mt-2 inline-block">
              Make your first prediction
            </Link>
          </div>
        )}
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Link to="/predict" className="card card-hover text-center">
          <div className="p-6">
            <div className="w-16 h-16 bg-primary-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <Activity className="w-8 h-8 text-primary-600" />
            </div>
            <h3 className="font-semibold text-gray-800 mb-2">New Prediction</h3>
            <p className="text-sm text-gray-600">
              Upload medical image for AI analysis
            </p>
          </div>
        </Link>

        <Link to="/batch" className="card card-hover text-center">
          <div className="p-6">
            <div className="w-16 h-16 bg-purple-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <FileText className="w-8 h-8 text-purple-600" />
            </div>
            <h3 className="font-semibold text-gray-800 mb-2">Batch Processing</h3>
            <p className="text-sm text-gray-600">
              Process multiple images at once
            </p>
          </div>
        </Link>

        <Link to="/analytics" className="card card-hover text-center">
          <div className="p-6">
            <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <TrendingUp className="w-8 h-8 text-green-600" />
            </div>
            <h3 className="font-semibold text-gray-800 mb-2">View Analytics</h3>
            <p className="text-sm text-gray-600">
              Analyze prediction trends and insights
            </p>
          </div>
        </Link>
      </div>
    </div>
  );
};

export default Dashboard;
