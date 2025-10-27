/**
 * Tenant Administration Dashboard
 * 
 * Provides:
 * - Tenant overview and settings
 * - User management (invite, remove, change roles)
 * - Usage monitoring and quota visualization
 * - Billing and subscription management
 */

import React, { useState, useEffect } from 'react';

interface Tenant {
  id: string;
  name: string;
  slug: string;
  tier: 'free' | 'starter' | 'professional' | 'enterprise';
  status: 'active' | 'trial' | 'suspended' | 'cancelled';
  quota_api_calls: number;
  quota_storage_gb: number;
  quota_gpu_hours: number;
  trial_ends_at?: string;
  created_at: string;
}

interface Usage {
  api_calls: number;
  storage_gb: number;
  gpu_hours: number;
  quota_api_calls: number;
  quota_storage_gb: number;
  quota_gpu_hours: number;
  percentage_used: {
    api_calls: number;
    storage_gb: number;
    gpu_hours: number;
  };
}

interface User {
  user_id: string;
  email: string;
  role: 'owner' | 'admin' | 'member' | 'viewer';
  joined_at: string;
}

interface BillingSummary {
  tenant: {
    name: string;
    tier: string;
    status: string;
  };
  current_usage: {
    api_calls: number;
    storage_gb: number;
    gpu_hours: number;
    estimated_cost: number;
  };
  quotas: {
    api_calls: number;
    storage_gb: number;
    gpu_hours: number;
  };
  recent_invoices: Array<{
    invoice_number: string;
    amount: number;
    status: string;
    period: {
      start: string;
      end: string;
    };
  }>;
}

export default function TenantAdminDashboard() {
  const [tenant, setTenant] = useState<Tenant | null>(null);
  const [usage, setUsage] = useState<Usage | null>(null);
  const [users, setUsers] = useState<User[]>([]);
  const [billing, setBilling] = useState<BillingSummary | null>(null);
  const [activeTab, setActiveTab] = useState<'overview' | 'users' | 'usage' | 'billing'>('overview');
  const [loading, setLoading] = useState(true);
  
  // Invite modal state
  const [showInviteModal, setShowInviteModal] = useState(false);
  const [inviteEmail, setInviteEmail] = useState('');
  const [inviteRole, setInviteRole] = useState<'admin' | 'member' | 'viewer'>('member');

  useEffect(() => {
    loadTenantData();
  }, []);

  const loadTenantData = async () => {
    setLoading(true);
    try {
      // Load tenant info
      const tenantRes = await fetch('/api/v1/tenants/current');
      const tenantData = await tenantRes.json();
      setTenant(tenantData);

      // Load usage
      const usageRes = await fetch(`/api/v1/tenants/${tenantData.id}/usage`);
      const usageData = await usageRes.json();
      setUsage(usageData);

      // Load users
      const usersRes = await fetch(`/api/v1/tenants/${tenantData.id}/users`);
      const usersData = await usersRes.json();
      setUsers(usersData);

      // Load billing
      const billingRes = await fetch(`/api/v1/tenants/${tenantData.id}/billing`);
      const billingData = await billingRes.json();
      setBilling(billingData);
    } catch (error) {
      console.error('Error loading tenant data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleInviteUser = async () => {
    try {
      const res = await fetch(`/api/v1/tenants/${tenant?.id}/invitations`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          email: inviteEmail,
          role: inviteRole,
        }),
      });

      if (res.ok) {
        alert('Invitation sent!');
        setShowInviteModal(false);
        setInviteEmail('');
        loadTenantData();
      } else {
        alert('Failed to send invitation');
      }
    } catch (error) {
      console.error('Error inviting user:', error);
    }
  };

  const handleRemoveUser = async (userId: string) => {
    if (!confirm('Are you sure you want to remove this user?')) return;

    try {
      const res = await fetch(`/api/v1/tenants/${tenant?.id}/users/${userId}`, {
        method: 'DELETE',
      });

      if (res.ok) {
        alert('User removed');
        loadTenantData();
      } else {
        alert('Failed to remove user');
      }
    } catch (error) {
      console.error('Error removing user:', error);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-xl">Loading...</div>
      </div>
    );
  }

  return (
    <div className="container mx-auto p-6">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">{tenant?.name}</h1>
        <div className="flex items-center gap-4">
          <span className={`px-3 py-1 rounded-full text-sm font-semibold ${
            tenant?.status === 'active' ? 'bg-green-100 text-green-800' :
            tenant?.status === 'trial' ? 'bg-blue-100 text-blue-800' :
            'bg-gray-100 text-gray-800'
          }`}>
            {tenant?.status}
          </span>
          <span className="px-3 py-1 rounded-full text-sm font-semibold bg-purple-100 text-purple-800">
            {tenant?.tier}
          </span>
        </div>
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-200 mb-6">
        <nav className="flex gap-8">
          {(['overview', 'users', 'usage', 'billing'] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`pb-4 px-1 border-b-2 font-medium text-sm ${
                activeTab === tab
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              {tab.charAt(0).toUpperCase() + tab.slice(1)}
            </button>
          ))}
        </nav>
      </div>

      {/* Tab Content */}
      {activeTab === 'overview' && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Stats Cards */}
          <div className="bg-white p-6 rounded-lg shadow">
            <div className="text-sm text-gray-500 mb-1">API Calls</div>
            <div className="text-3xl font-bold">{usage?.api_calls.toLocaleString()}</div>
            <div className="text-sm text-gray-500 mt-2">
              {usage?.percentage_used.api_calls.toFixed(1)}% of quota
            </div>
          </div>

          <div className="bg-white p-6 rounded-lg shadow">
            <div className="text-sm text-gray-500 mb-1">Storage</div>
            <div className="text-3xl font-bold">{usage?.storage_gb.toFixed(1)} GB</div>
            <div className="text-sm text-gray-500 mt-2">
              {usage?.percentage_used.storage_gb.toFixed(1)}% of quota
            </div>
          </div>

          <div className="bg-white p-6 rounded-lg shadow">
            <div className="text-sm text-gray-500 mb-1">GPU Hours</div>
            <div className="text-3xl font-bold">{usage?.gpu_hours.toFixed(1)}</div>
            <div className="text-sm text-gray-500 mt-2">
              {usage?.percentage_used.gpu_hours.toFixed(1)}% of quota
            </div>
          </div>

          {/* Team Members */}
          <div className="bg-white p-6 rounded-lg shadow md:col-span-2">
            <h3 className="text-lg font-semibold mb-4">Team Members</h3>
            <div className="text-3xl font-bold">{users.length}</div>
            <div className="text-sm text-gray-500 mt-2">
              {users.filter(u => u.role === 'owner').length} owners, 
              {users.filter(u => u.role === 'admin').length} admins
            </div>
          </div>

          {/* Trial Info */}
          {tenant?.status === 'trial' && tenant.trial_ends_at && (
            <div className="bg-blue-50 p-6 rounded-lg border border-blue-200">
              <h3 className="text-lg font-semibold mb-2">Trial Period</h3>
              <div className="text-sm">
                Ends {new Date(tenant.trial_ends_at).toLocaleDateString()}
              </div>
              <button className="mt-4 bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700">
                Upgrade Now
              </button>
            </div>
          )}
        </div>
      )}

      {activeTab === 'users' && (
        <div>
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-xl font-semibold">Team Members ({users.length})</h2>
            <button
              onClick={() => setShowInviteModal(true)}
              className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
            >
              Invite User
            </button>
          </div>

          <div className="bg-white rounded-lg shadow overflow-hidden">
            <table className="w-full">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Email</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Role</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Joined</th>
                  <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {users.map((user) => (
                  <tr key={user.user_id}>
                    <td className="px-6 py-4 whitespace-nowrap">{user.email}</td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`px-2 py-1 rounded text-xs font-semibold ${
                        user.role === 'owner' ? 'bg-purple-100 text-purple-800' :
                        user.role === 'admin' ? 'bg-blue-100 text-blue-800' :
                        'bg-gray-100 text-gray-800'
                      }`}>
                        {user.role}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {new Date(user.joined_at).toLocaleDateString()}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-right">
                      {user.role !== 'owner' && (
                        <button
                          onClick={() => handleRemoveUser(user.user_id)}
                          className="text-red-600 hover:text-red-800 text-sm"
                        >
                          Remove
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {activeTab === 'usage' && usage && (
        <div>
          <h2 className="text-xl font-semibold mb-6">Current Usage</h2>
          
          {/* Usage Bars */}
          <div className="space-y-6">
            <UsageBar
              label="API Calls"
              current={usage.api_calls}
              quota={usage.quota_api_calls}
              percentage={usage.percentage_used.api_calls}
            />
            <UsageBar
              label="Storage (GB)"
              current={usage.storage_gb}
              quota={usage.quota_storage_gb}
              percentage={usage.percentage_used.storage_gb}
            />
            <UsageBar
              label="GPU Hours"
              current={usage.gpu_hours}
              quota={usage.quota_gpu_hours}
              percentage={usage.percentage_used.gpu_hours}
            />
          </div>

          {/* Estimated Cost */}
          {billing && (
            <div className="mt-8 bg-white p-6 rounded-lg shadow">
              <h3 className="text-lg font-semibold mb-4">Estimated Monthly Cost</h3>
              <div className="text-4xl font-bold text-blue-600">
                ${billing.current_usage.estimated_cost.toFixed(2)}
              </div>
              <div className="text-sm text-gray-500 mt-2">
                Based on current usage this month
              </div>
            </div>
          )}
        </div>
      )}

      {activeTab === 'billing' && billing && (
        <div>
          <h2 className="text-xl font-semibold mb-6">Billing & Invoices</h2>

          {/* Subscription Info */}
          <div className="bg-white p-6 rounded-lg shadow mb-6">
            <h3 className="text-lg font-semibold mb-4">Current Plan</h3>
            <div className="flex justify-between items-center">
              <div>
                <div className="text-2xl font-bold capitalize">{billing.tenant.tier} Plan</div>
                <div className="text-sm text-gray-500 mt-1">
                  Status: {billing.tenant.status}
                </div>
              </div>
              <button className="bg-blue-600 text-white px-6 py-2 rounded hover:bg-blue-700">
                Change Plan
              </button>
            </div>
          </div>

          {/* Recent Invoices */}
          <div className="bg-white rounded-lg shadow">
            <div className="px-6 py-4 border-b">
              <h3 className="text-lg font-semibold">Recent Invoices</h3>
            </div>
            <div className="divide-y divide-gray-200">
              {billing.recent_invoices.map((invoice) => (
                <div key={invoice.invoice_number} className="px-6 py-4 flex justify-between items-center">
                  <div>
                    <div className="font-medium">{invoice.invoice_number}</div>
                    <div className="text-sm text-gray-500">
                      {new Date(invoice.period.start).toLocaleDateString()} - 
                      {new Date(invoice.period.end).toLocaleDateString()}
                    </div>
                  </div>
                  <div className="flex items-center gap-4">
                    <span className={`px-2 py-1 rounded text-xs font-semibold ${
                      invoice.status === 'paid' ? 'bg-green-100 text-green-800' :
                      invoice.status === 'draft' ? 'bg-gray-100 text-gray-800' :
                      'bg-red-100 text-red-800'
                    }`}>
                      {invoice.status}
                    </span>
                    <div className="font-semibold">${invoice.amount.toFixed(2)}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Invite User Modal */}
      {showInviteModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center">
          <div className="bg-white rounded-lg p-6 w-full max-w-md">
            <h3 className="text-lg font-semibold mb-4">Invite User</h3>
            
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Email Address
              </label>
              <input
                type="email"
                value={inviteEmail}
                onChange={(e) => setInviteEmail(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded focus:ring-blue-500 focus:border-blue-500"
                placeholder="user@example.com"
              />
            </div>

            <div className="mb-6">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Role
              </label>
              <select
                value={inviteRole}
                onChange={(e) => setInviteRole(e.target.value as any)}
                className="w-full px-3 py-2 border border-gray-300 rounded focus:ring-blue-500 focus:border-blue-500"
              >
                <option value="viewer">Viewer (Read-only)</option>
                <option value="member">Member (Read & Write)</option>
                <option value="admin">Admin (Full Access)</option>
              </select>
            </div>

            <div className="flex gap-3">
              <button
                onClick={() => setShowInviteModal(false)}
                className="flex-1 px-4 py-2 border border-gray-300 rounded hover:bg-gray-50"
              >
                Cancel
              </button>
              <button
                onClick={handleInviteUser}
                disabled={!inviteEmail}
                className="flex-1 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
              >
                Send Invitation
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// Usage Bar Component
function UsageBar({ label, current, quota, percentage }: {
  label: string;
  current: number;
  quota: number;
  percentage: number;
}) {
  const isUnlimited = quota === -1;
  const color = percentage > 90 ? 'bg-red-500' : percentage > 70 ? 'bg-yellow-500' : 'bg-green-500';

  return (
    <div className="bg-white p-6 rounded-lg shadow">
      <div className="flex justify-between items-center mb-2">
        <span className="font-medium">{label}</span>
        <span className="text-sm text-gray-500">
          {current.toLocaleString()} / {isUnlimited ? '∞' : quota.toLocaleString()}
        </span>
      </div>
      <div className="w-full bg-gray-200 rounded-full h-2.5">
        <div
          className={`h-2.5 rounded-full ${color}`}
          style={{ width: `${Math.min(percentage, 100)}%` }}
        />
      </div>
      <div className="text-sm text-gray-500 mt-2">
        {percentage.toFixed(1)}% used
      </div>
    </div>
  );
}
