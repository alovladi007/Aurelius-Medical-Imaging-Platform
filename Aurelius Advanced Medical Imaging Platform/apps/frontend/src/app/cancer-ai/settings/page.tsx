'use client'

import { useState } from 'react'
import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Badge } from '@/components/ui/badge'
import {
  Settings,
  Save,
  RotateCcw,
  Bell,
  Shield,
  Zap,
  Database,
  Mail,
  Sliders,
  CheckCircle
} from 'lucide-react'

export default function SettingsPage() {
  const [confidenceThreshold, setConfidenceThreshold] = useState(85)
  const [autoReview, setAutoReview] = useState(true)
  const [emailNotifications, setEmailNotifications] = useState(true)
  const [savedSuccess, setSavedSuccess] = useState(false)

  const handleSave = () => {
    // In production, this would save to API
    setSavedSuccess(true)
    setTimeout(() => setSavedSuccess(false), 3000)
  }

  const handleReset = () => {
    setConfidenceThreshold(85)
    setAutoReview(true)
    setEmailNotifications(true)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-purple-50 to-slate-50 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900 p-6">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-slate-900 dark:text-white mb-2">
          Cancer AI Settings
        </h1>
        <p className="text-slate-600 dark:text-slate-400">
          Configure preferences and thresholds for AI analysis
        </p>
      </div>

      {/* Save Success Banner */}
      {savedSuccess && (
        <div className="mb-6 p-4 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg flex items-center gap-3">
          <CheckCircle className="h-5 w-5 text-green-600" />
          <p className="text-green-800 dark:text-green-200 font-medium">
            Settings saved successfully!
          </p>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main Settings */}
        <div className="lg:col-span-2 space-y-6">
          {/* Detection Settings */}
          <Card className="p-6 bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm">
            <div className="flex items-center gap-3 mb-6">
              <div className="p-2 bg-purple-100 dark:bg-purple-900/50 rounded-lg">
                <Sliders className="h-5 w-5 text-purple-600" />
              </div>
              <div>
                <h3 className="text-lg font-bold text-slate-900 dark:text-white">
                  Detection Settings
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                  Configure AI detection parameters
                </p>
              </div>
            </div>

            <div className="space-y-6">
              {/* Confidence Threshold */}
              <div>
                <div className="flex items-center justify-between mb-3">
                  <Label className="text-sm font-medium">
                    Confidence Threshold
                  </Label>
                  <Badge variant="outline" className="bg-purple-50 dark:bg-purple-900/50 text-purple-700 dark:text-purple-300">
                    {confidenceThreshold}%
                  </Badge>
                </div>
                <input
                  type="range"
                  min="50"
                  max="99"
                  value={confidenceThreshold}
                  onChange={(e) => setConfidenceThreshold(Number(e.target.value))}
                  className="w-full h-2 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-purple-600"
                />
                <p className="text-xs text-slate-500 dark:text-slate-400 mt-2">
                  Minimum confidence level required for positive detection
                </p>
              </div>

              {/* Auto Review Toggle */}
              <div className="flex items-center justify-between p-4 bg-slate-50 dark:bg-slate-900/50 rounded-lg">
                <div className="flex items-center gap-3">
                  <Zap className="h-5 w-5 text-orange-600" />
                  <div>
                    <p className="font-medium text-slate-900 dark:text-white">Auto-Review Mode</p>
                    <p className="text-sm text-slate-600 dark:text-slate-400">
                      Automatically flag cases below threshold for manual review
                    </p>
                  </div>
                </div>
                <button
                  onClick={() => setAutoReview(!autoReview)}
                  className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                    autoReview ? 'bg-purple-600' : 'bg-slate-300 dark:bg-slate-600'
                  }`}
                >
                  <span
                    className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                      autoReview ? 'translate-x-6' : 'translate-x-1'
                    }`}
                  />
                </button>
              </div>

              {/* Model Selection */}
              <div>
                <Label className="text-sm font-medium mb-3 block">
                  Default Model Version
                </Label>
                <select className="w-full px-3 py-2 bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-lg text-slate-900 dark:text-white">
                  <option value="latest">Latest (v2.3.1)</option>
                  <option value="stable">Stable (v2.3.0)</option>
                  <option value="legacy">Legacy (v2.2.5)</option>
                </select>
                <p className="text-xs text-slate-500 dark:text-slate-400 mt-2">
                  Model version to use for new analyses
                </p>
              </div>
            </div>
          </Card>

          {/* Notification Settings */}
          <Card className="p-6 bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm">
            <div className="flex items-center gap-3 mb-6">
              <div className="p-2 bg-blue-100 dark:bg-blue-900/50 rounded-lg">
                <Bell className="h-5 w-5 text-blue-600" />
              </div>
              <div>
                <h3 className="text-lg font-bold text-slate-900 dark:text-white">
                  Notifications
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                  Manage alert preferences
                </p>
              </div>
            </div>

            <div className="space-y-4">
              {/* Email Notifications */}
              <div className="flex items-center justify-between p-4 bg-slate-50 dark:bg-slate-900/50 rounded-lg">
                <div className="flex items-center gap-3">
                  <Mail className="h-5 w-5 text-blue-600" />
                  <div>
                    <p className="font-medium text-slate-900 dark:text-white">Email Notifications</p>
                    <p className="text-sm text-slate-600 dark:text-slate-400">
                      Receive email alerts for high-confidence detections
                    </p>
                  </div>
                </div>
                <button
                  onClick={() => setEmailNotifications(!emailNotifications)}
                  className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                    emailNotifications ? 'bg-blue-600' : 'bg-slate-300 dark:bg-slate-600'
                  }`}
                >
                  <span
                    className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                      emailNotifications ? 'translate-x-6' : 'translate-x-1'
                    }`}
                  />
                </button>
              </div>

              {/* Email Input */}
              {emailNotifications && (
                <div>
                  <Label className="text-sm font-medium mb-2 block">
                    Notification Email
                  </Label>
                  <Input
                    type="email"
                    placeholder="email@example.com"
                    defaultValue="dr.johnson@hospital.com"
                    className="w-full"
                  />
                </div>
              )}
            </div>
          </Card>

          {/* Data & Privacy */}
          <Card className="p-6 bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm">
            <div className="flex items-center gap-3 mb-6">
              <div className="p-2 bg-green-100 dark:bg-green-900/50 rounded-lg">
                <Shield className="h-5 w-5 text-green-600" />
              </div>
              <div>
                <h3 className="text-lg font-bold text-slate-900 dark:text-white">
                  Data & Privacy
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                  Control data usage and retention
                </p>
              </div>
            </div>

            <div className="space-y-4">
              <div>
                <Label className="text-sm font-medium mb-3 block">
                  Data Retention Period
                </Label>
                <select className="w-full px-3 py-2 bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-lg text-slate-900 dark:text-white">
                  <option value="90">90 days</option>
                  <option value="180">180 days</option>
                  <option value="365">1 year</option>
                  <option value="-1">Indefinitely</option>
                </select>
              </div>

              <div className="flex items-center justify-between p-4 bg-slate-50 dark:bg-slate-900/50 rounded-lg">
                <div className="flex items-center gap-3">
                  <Database className="h-5 w-5 text-green-600" />
                  <div>
                    <p className="font-medium text-slate-900 dark:text-white">Anonymous Analytics</p>
                    <p className="text-sm text-slate-600 dark:text-slate-400">
                      Help improve models by sharing anonymized data
                    </p>
                  </div>
                </div>
                <button className="relative inline-flex h-6 w-11 items-center rounded-full bg-green-600">
                  <span className="inline-block h-4 w-4 transform rounded-full bg-white translate-x-6" />
                </button>
              </div>
            </div>
          </Card>
        </div>

        {/* Sidebar */}
        <div className="space-y-6">
          {/* Quick Actions */}
          <Card className="p-6 bg-gradient-to-br from-purple-500 to-pink-500 text-white">
            <Settings className="h-8 w-8 mb-4 opacity-80" />
            <h3 className="text-lg font-bold mb-2">Configuration</h3>
            <p className="text-sm text-purple-100 mb-6">
              Your settings are automatically synced across all devices and sessions
            </p>
            <div className="flex gap-2">
              <Button
                onClick={handleSave}
                className="flex-1 bg-white text-purple-600 hover:bg-purple-50"
              >
                <Save className="h-4 w-4 mr-2" />
                Save
              </Button>
              <Button
                onClick={handleReset}
                variant="outline"
                className="border-white/30 text-white hover:bg-white/10"
              >
                <RotateCcw className="h-4 w-4" />
              </Button>
            </div>
          </Card>

          {/* Current Configuration Summary */}
          <Card className="p-6 bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm">
            <h3 className="text-sm font-bold text-slate-900 dark:text-white mb-4">
              Current Configuration
            </h3>
            <div className="space-y-3">
              <div className="flex justify-between text-sm">
                <span className="text-slate-600 dark:text-slate-400">Threshold</span>
                <span className="font-medium text-slate-900 dark:text-white">{confidenceThreshold}%</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-slate-600 dark:text-slate-400">Auto-Review</span>
                <Badge className={autoReview ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'}>
                  {autoReview ? 'Enabled' : 'Disabled'}
                </Badge>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-slate-600 dark:text-slate-400">Email Alerts</span>
                <Badge className={emailNotifications ? 'bg-blue-100 text-blue-800' : 'bg-gray-100 text-gray-800'}>
                  {emailNotifications ? 'On' : 'Off'}
                </Badge>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-slate-600 dark:text-slate-400">Model Version</span>
                <span className="font-medium text-slate-900 dark:text-white">v2.3.1</span>
              </div>
            </div>
          </Card>

          {/* Help & Documentation */}
          <Card className="p-6 bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm">
            <h3 className="text-sm font-bold text-slate-900 dark:text-white mb-4">
              Need Help?
            </h3>
            <p className="text-sm text-slate-600 dark:text-slate-400 mb-4">
              Check our documentation for detailed information about each setting
            </p>
            <Button variant="outline" size="sm" className="w-full">
              View Documentation
            </Button>
          </Card>
        </div>
      </div>
    </div>
  )
}
