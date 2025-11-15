'use client'

import { LucideIcon } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { cn } from '@/lib/utils'

interface MetricsCardProps {
  title: string
  value: string | number
  change?: string
  changeType?: 'positive' | 'negative' | 'neutral'
  icon: LucideIcon
  description?: string
  trend?: number[]
}

export function MetricsCard({
  title,
  value,
  change,
  changeType = 'neutral',
  icon: Icon,
  description,
  trend
}: MetricsCardProps) {
  const changeColor = {
    positive: 'text-green-600 dark:text-green-400',
    negative: 'text-red-600 dark:text-red-400',
    neutral: 'text-slate-600 dark:text-slate-400'
  }[changeType]

  return (
    <Card className="relative overflow-hidden hover:shadow-lg transition-all duration-300 border-l-4 border-l-primary">
      <CardHeader className="flex flex-row items-center justify-between pb-2 space-y-0">
        <CardTitle className="text-sm font-medium text-slate-600 dark:text-slate-400">
          {title}
        </CardTitle>
        <div className="p-2 bg-primary/10 rounded-lg">
          <Icon className="h-4 w-4 text-primary" />
        </div>
      </CardHeader>
      <CardContent>
        <div className="text-3xl font-bold text-slate-900 dark:text-white">
          {value}
        </div>
        {description && (
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
            {description}
          </p>
        )}
        {change && (
          <p className={cn("text-xs mt-2 font-medium", changeColor)}>
            {change}
          </p>
        )}
        {trend && trend.length > 0 && (
          <div className="mt-4 h-8">
            <svg className="w-full h-full" viewBox="0 0 100 30" preserveAspectRatio="none">
              <polyline
                points={trend.map((v, i) => `${(i / (trend.length - 1)) * 100},${30 - (v / Math.max(...trend)) * 25}`).join(' ')}
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                className="text-primary opacity-50"
              />
            </svg>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
