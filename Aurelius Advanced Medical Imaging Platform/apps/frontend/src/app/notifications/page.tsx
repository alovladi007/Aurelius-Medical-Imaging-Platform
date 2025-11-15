'use client'

import { Bell, CheckCircle, AlertTriangle, Info, X, Trash2, Clock } from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { useState } from 'react'

export default function NotificationsPage() {
  const [notifications, setNotifications] = useState([
    {
      id: 1,
      type: 'success',
      title: 'Cancer AI Analysis Complete',
      message: 'Your batch processing of 45 images has been completed successfully.',
      time: '5 minutes ago',
      read: false,
      icon: CheckCircle,
      iconColor: 'text-green-600',
      iconBg: 'bg-green-50'
    },
    {
      id: 2,
      type: 'warning',
      title: 'Review Required',
      message: 'Study #STD-2024-089 requires manual review. High uncertainty detected.',
      time: '12 minutes ago',
      read: false,
      icon: AlertTriangle,
      iconColor: 'text-orange-600',
      iconBg: 'bg-orange-50'
    },
    {
      id: 3,
      type: 'info',
      title: 'System Update Available',
      message: 'Cancer AI model v2.4.0 is available. New improvements in detection accuracy.',
      time: '1 hour ago',
      read: false,
      icon: Info,
      iconColor: 'text-blue-600',
      iconBg: 'bg-blue-50'
    },
    {
      id: 4,
      type: 'success',
      title: 'Study Uploaded',
      message: 'New DICOM study uploaded successfully - 250 images processed.',
      time: '2 hours ago',
      read: true,
      icon: CheckCircle,
      iconColor: 'text-green-600',
      iconBg: 'bg-green-50'
    },
    {
      id: 5,
      type: 'info',
      title: 'Collaboration Invitation',
      message: 'Dr. Sarah Johnson invited you to join research project "Brain Tumor Analysis".',
      time: '3 hours ago',
      read: true,
      icon: Info,
      iconColor: 'text-purple-600',
      iconBg: 'bg-purple-50'
    },
    {
      id: 6,
      type: 'success',
      title: 'Report Generated',
      message: 'Monthly analytics report for January 2024 is ready for download.',
      time: '5 hours ago',
      read: true,
      icon: CheckCircle,
      iconColor: 'text-green-600',
      iconBg: 'bg-green-50'
    }
  ])

  const markAsRead = (id: number) => {
    setNotifications(notifications.map(n =>
      n.id === id ? { ...n, read: true } : n
    ))
  }

  const deleteNotification = (id: number) => {
    setNotifications(notifications.filter(n => n.id !== id))
  }

  const markAllAsRead = () => {
    setNotifications(notifications.map(n => ({ ...n, read: true })))
  }

  const unreadCount = notifications.filter(n => !n.read).length

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-purple-50 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900 p-8">
      <div className="max-w-4xl mx-auto space-y-8">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-2">
              Notifications
            </h1>
            <p className="text-slate-600 dark:text-slate-400">
              Stay updated with your platform activity
            </p>
          </div>
          {unreadCount > 0 && (
            <Button onClick={markAllAsRead} variant="outline">
              <CheckCircle className="w-4 h-4 mr-2" />
              Mark all as read
            </Button>
          )}
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-slate-600">
                Total Notifications
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">{notifications.length}</div>
            </CardContent>
          </Card>
          <Card className="border-2 border-blue-200">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-slate-600">
                Unread
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-blue-600">{unreadCount}</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-slate-600">
                This Week
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">{notifications.length}</div>
            </CardContent>
          </Card>
        </div>

        {/* Notifications List */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="flex items-center gap-2">
                  <Bell className="w-5 h-5 text-blue-600" />
                  All Notifications
                </CardTitle>
                <CardDescription>Recent activity and system alerts</CardDescription>
              </div>
              {unreadCount > 0 && (
                <Badge className="bg-blue-600">{unreadCount} new</Badge>
              )}
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {notifications.map((notification) => (
                <div
                  key={notification.id}
                  className={`flex items-start gap-4 p-4 rounded-lg border transition-all ${
                    !notification.read
                      ? 'bg-blue-50/50 dark:bg-blue-950/20 border-blue-200 dark:border-blue-800'
                      : 'bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700'
                  }`}
                >
                  <div className={`p-2 rounded-lg ${notification.iconBg}`}>
                    <notification.icon className={`w-5 h-5 ${notification.iconColor}`} />
                  </div>

                  <div className="flex-1 min-w-0">
                    <div className="flex items-start justify-between gap-2">
                      <h3 className={`font-semibold ${!notification.read ? 'text-slate-900 dark:text-white' : 'text-slate-700 dark:text-slate-300'}`}>
                        {notification.title}
                      </h3>
                      {!notification.read && (
                        <div className="w-2 h-2 bg-blue-600 rounded-full flex-shrink-0 mt-2" />
                      )}
                    </div>
                    <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">
                      {notification.message}
                    </p>
                    <div className="flex items-center gap-2 mt-2">
                      <Clock className="w-3 h-3 text-slate-400" />
                      <span className="text-xs text-slate-500">{notification.time}</span>
                    </div>
                  </div>

                  <div className="flex gap-1">
                    {!notification.read && (
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => markAsRead(notification.id)}
                        className="h-8 px-2"
                      >
                        <CheckCircle className="w-4 h-4" />
                      </Button>
                    )}
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => deleteNotification(notification.id)}
                      className="h-8 px-2 text-red-600 hover:text-red-700 hover:bg-red-50"
                    >
                      <Trash2 className="w-4 h-4" />
                    </Button>
                  </div>
                </div>
              ))}

              {notifications.length === 0 && (
                <div className="text-center py-12 text-slate-500">
                  <Bell className="w-12 h-12 mx-auto mb-4 opacity-50" />
                  <p>No notifications</p>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
