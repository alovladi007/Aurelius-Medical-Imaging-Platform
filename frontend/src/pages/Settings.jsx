import React, { useState, useEffect } from 'react';
import {
  Settings as SettingsIcon,
  Save,
  RefreshCw,
  Bell,
  Database,
  Shield,
  Info,
} from 'lucide-react';
import useStore from '../store/useStore';
import { api } from '../services/api';

const Settings = () => {
  const { settings, updateSettings } = useStore();
  const [localSettings, setLocalSettings] = useState(settings);
  const [modelInfo, setModelInfo] = useState(null);
  const [loading, setLoading] = useState(false);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    loadModelInfo();
  }, []);

  const loadModelInfo = async () => {
    try {
      const info = await api.getModelInfo();
      setModelInfo(info);
    } catch (error) {
      console.error('Failed to load model info:', error);
    }
  };

  const handleChange = (key, value) => {
    setLocalSettings((prev) => ({
      ...prev,
      [key]: value,
    }));
    setSaved(false);
  };

  const handleSave = () => {
    setLoading(true);
    setTimeout(() => {
      updateSettings(localSettings);
      setSaved(true);
      setLoading(false);
      setTimeout(() => setSaved(false), 3000);
    }, 500);
  };

  const handleReset = () => {
    if (window.confirm('Reset all settings to defaults?')) {
      const defaultSettings = {
        theme: 'light',
        notifications: true,
        autoSave: true,
        confidenceThreshold: 0.7,
      };
      setLocalSettings(defaultSettings);
      updateSettings(defaultSettings);
    }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6 animate-fade-in">
      {/* System Information */}
      <div className="card">
        <div className="flex items-center space-x-3 mb-4">
          <Info className="w-6 h-6 text-primary-600" />
          <h2 className="text-lg font-semibold text-gray-800">System Information</h2>
        </div>

        {modelInfo ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-gray-50 rounded-lg p-4">
              <p className="text-sm text-gray-600 mb-1">Model Framework</p>
              <p className="text-lg font-semibold text-gray-900">{modelInfo.framework}</p>
            </div>
            <div className="bg-gray-50 rounded-lg p-4">
              <p className="text-sm text-gray-600 mb-1">Model Path</p>
              <p className="text-sm font-medium text-gray-900 truncate">
                {modelInfo.model_path}
              </p>
            </div>
            <div className="bg-gray-50 rounded-lg p-4">
              <p className="text-sm text-gray-600 mb-1">Input Shape</p>
              <p className="text-lg font-semibold text-gray-900">
                {modelInfo.input_shape?.join(' × ')}
              </p>
            </div>
            <div className="bg-gray-50 rounded-lg p-4">
              <p className="text-sm text-gray-600 mb-1">Classes</p>
              <p className="text-lg font-semibold text-gray-900">
                {modelInfo.class_names?.length || 0}
              </p>
            </div>
          </div>
        ) : (
          <div className="text-center py-6 text-gray-500">
            <div className="spinner mx-auto mb-3"></div>
            <p>Loading model information...</p>
          </div>
        )}

        {modelInfo?.class_names && (
          <div className="mt-4">
            <p className="text-sm text-gray-600 mb-2">Supported Cancer Types:</p>
            <div className="flex flex-wrap gap-2">
              {modelInfo.class_names.map((name, index) => (
                <span key={index} className="badge badge-info">
                  {name}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Application Settings */}
      <div className="card">
        <div className="flex items-center space-x-3 mb-4">
          <SettingsIcon className="w-6 h-6 text-primary-600" />
          <h2 className="text-lg font-semibold text-gray-800">Application Settings</h2>
        </div>

        <div className="space-y-6">
          {/* Theme */}
          <div>
            <label className="label">Theme</label>
            <select
              value={localSettings.theme}
              onChange={(e) => handleChange('theme', e.target.value)}
              className="input"
            >
              <option value="light">Light</option>
              <option value="dark">Dark</option>
              <option value="auto">Auto</option>
            </select>
            <p className="text-sm text-gray-500 mt-1">
              Choose your preferred color theme
            </p>
          </div>

          {/* Confidence Threshold */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="label mb-0">Confidence Threshold</label>
              <span className="text-sm font-medium text-primary-600">
                {(localSettings.confidenceThreshold * 100).toFixed(0)}%
              </span>
            </div>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={localSettings.confidenceThreshold}
              onChange={(e) => handleChange('confidenceThreshold', parseFloat(e.target.value))}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-primary-600"
            />
            <p className="text-sm text-gray-500 mt-1">
              Minimum confidence level for predictions (50-95% recommended)
            </p>
          </div>

          {/* Notifications */}
          <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
            <div className="flex items-center space-x-3">
              <Bell className="w-5 h-5 text-gray-600" />
              <div>
                <p className="font-medium text-gray-800">Enable Notifications</p>
                <p className="text-sm text-gray-500">
                  Get notified about prediction results
                </p>
              </div>
            </div>
            <label className="relative inline-flex items-center cursor-pointer">
              <input
                type="checkbox"
                checked={localSettings.notifications}
                onChange={(e) => handleChange('notifications', e.target.checked)}
                className="sr-only peer"
              />
              <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-primary-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary-600"></div>
            </label>
          </div>

          {/* Auto Save */}
          <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
            <div className="flex items-center space-x-3">
              <Database className="w-5 h-5 text-gray-600" />
              <div>
                <p className="font-medium text-gray-800">Auto-save Predictions</p>
                <p className="text-sm text-gray-500">
                  Automatically save predictions to history
                </p>
              </div>
            </div>
            <label className="relative inline-flex items-center cursor-pointer">
              <input
                type="checkbox"
                checked={localSettings.autoSave}
                onChange={(e) => handleChange('autoSave', e.target.checked)}
                className="sr-only peer"
              />
              <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-primary-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary-600"></div>
            </label>
          </div>
        </div>
      </div>

      {/* Data & Privacy */}
      <div className="card">
        <div className="flex items-center space-x-3 mb-4">
          <Shield className="w-6 h-6 text-primary-600" />
          <h2 className="text-lg font-semibold text-gray-800">Data & Privacy</h2>
        </div>

        <div className="space-y-4">
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <p className="text-sm text-blue-800">
              <strong>Privacy Notice:</strong> All data is stored locally in your browser.
              No patient information is transmitted to external servers except for the AI
              prediction API endpoint.
            </p>
          </div>

          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
            <p className="text-sm text-yellow-800">
              <strong>HIPAA Compliance:</strong> This application is designed with
              healthcare data privacy in mind. Ensure your deployment follows all
              applicable regulations for your jurisdiction.
            </p>
          </div>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex justify-between items-center">
        <button onClick={handleReset} className="btn btn-secondary flex items-center">
          <RefreshCw className="w-4 h-4 mr-2" />
          Reset to Defaults
        </button>

        <button
          onClick={handleSave}
          className="btn btn-primary flex items-center"
          disabled={loading}
        >
          {loading ? (
            <>
              <div className="spinner-small mr-2"></div>
              Saving...
            </>
          ) : saved ? (
            <>
              <Save className="w-4 h-4 mr-2" />
              Saved!
            </>
          ) : (
            <>
              <Save className="w-4 h-4 mr-2" />
              Save Settings
            </>
          )}
        </button>
      </div>

      {/* Version Info */}
      <div className="card bg-gray-50 text-center text-sm text-gray-600">
        <p className="mb-1">Advanced Cancer AI Dashboard v1.0.0</p>
        <p className="text-xs">For Research and Educational Use Only</p>
      </div>
    </div>
  );
};

export default Settings;
