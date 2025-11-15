'use client'

import { useState } from 'react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import {
  LayoutDashboard,
  Database,
  Brain,
  Search,
  Users,
  Settings,
  FileText,
  BarChart3,
  Microscope,
  Flask,
  BookOpen,
  Bell,
  ChevronLeft,
  ChevronRight,
  Activity
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'

const navigation = [
  {
    name: 'Dashboard',
    href: '/',
    icon: LayoutDashboard,
    badge: null
  },
  {
    name: 'Studies',
    href: '/studies',
    icon: Database,
    badge: '1.2k'
  },
  {
    name: 'DICOM Viewer',
    href: '/viewer',
    icon: Microscope,
    badge: null
  },
  {
    name: 'AI & ML',
    href: '/ml',
    icon: Brain,
    badge: 'New'
  },
  {
    name: 'Cancer AI',
    href: '/cancer-ai',
    icon: Activity,
    badge: 'Beta'
  },
  {
    name: 'Research',
    href: '/research',
    icon: Flask,
    badge: null
  },
  {
    name: 'Analytics',
    href: '/analytics',
    icon: BarChart3,
    badge: null
  },
  {
    name: 'Search',
    href: '/search',
    icon: Search,
    badge: null
  },
  {
    name: 'Worklists',
    href: '/worklists',
    icon: FileText,
    badge: '12'
  },
  {
    name: 'Collaborations',
    href: '/collaborations',
    icon: Users,
    badge: '3'
  },
  {
    name: 'Publications',
    href: '/publications',
    icon: BookOpen,
    badge: null
  }
]

const bottomNavigation = [
  {
    name: 'Notifications',
    href: '/notifications',
    icon: Bell,
    badge: '5'
  },
  {
    name: 'Settings',
    href: '/settings',
    icon: Settings,
    badge: null
  }
]

export function Sidebar() {
  const pathname = usePathname()
  const [collapsed, setCollapsed] = useState(false)

  return (
    <div
      className={cn(
        'flex flex-col h-screen bg-slate-900 dark:bg-slate-950 text-white transition-all duration-300 border-r border-slate-800',
        collapsed ? 'w-16' : 'w-64'
      )}
    >
      {/* Logo */}
      <div className="flex items-center justify-between p-4 border-b border-slate-800">
        {!collapsed && (
          <div className="flex items-center gap-2">
            <Activity className="h-8 w-8 text-primary" />
            <div>
              <h1 className="text-lg font-bold">Aurelius</h1>
              <p className="text-xs text-slate-400">Research Platform</p>
            </div>
          </div>
        )}
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setCollapsed(!collapsed)}
          className={cn("hover:bg-slate-800", collapsed && "mx-auto")}
        >
          {collapsed ? (
            <ChevronRight className="h-4 w-4" />
          ) : (
            <ChevronLeft className="h-4 w-4" />
          )}
        </Button>
      </div>

      {/* Main Navigation */}
      <nav className="flex-1 overflow-y-auto p-2 space-y-1">
        {navigation.map((item) => {
          const isActive = pathname === item.href
          return (
            <Link
              key={item.name}
              href={item.href}
              className={cn(
                'flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors',
                isActive
                  ? 'bg-primary text-white'
                  : 'text-slate-300 hover:bg-slate-800 hover:text-white',
                collapsed && 'justify-center'
              )}
            >
              <item.icon className="h-5 w-5 flex-shrink-0" />
              {!collapsed && (
                <>
                  <span className="flex-1">{item.name}</span>
                  {item.badge && (
                    <span className={cn(
                      "px-2 py-0.5 text-xs rounded-full",
                      isActive
                        ? "bg-white/20"
                        : "bg-slate-800 text-slate-300"
                    )}>
                      {item.badge}
                    </span>
                  )}
                </>
              )}
            </Link>
          )
        })}
      </nav>

      {/* Bottom Navigation */}
      <div className="p-2 space-y-1 border-t border-slate-800">
        {bottomNavigation.map((item) => {
          const isActive = pathname === item.href
          return (
            <Link
              key={item.name}
              href={item.href}
              className={cn(
                'flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors',
                isActive
                  ? 'bg-primary text-white'
                  : 'text-slate-300 hover:bg-slate-800 hover:text-white',
                collapsed && 'justify-center'
              )}
            >
              <item.icon className="h-5 w-5 flex-shrink-0" />
              {!collapsed && (
                <>
                  <span className="flex-1">{item.name}</span>
                  {item.badge && (
                    <span className="px-2 py-0.5 text-xs rounded-full bg-red-500 text-white">
                      {item.badge}
                    </span>
                  )}
                </>
              )}
            </Link>
          )
        })}

        {/* User Profile */}
        {!collapsed && (
          <div className="mt-4 p-3 bg-slate-800 rounded-lg">
            <div className="flex items-center gap-3">
              <div className="h-10 w-10 rounded-full bg-primary flex items-center justify-center text-sm font-bold">
                DJ
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium truncate">Dr. Johnson</p>
                <p className="text-xs text-slate-400 truncate">Researcher</p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
