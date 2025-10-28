// Core Types for Aurelius Medical Imaging Platform

export interface User {
  id: string;
  email: string;
  name: string;
  role: 'owner' | 'admin' | 'member' | 'viewer';
  tenantId: string;
  avatar?: string;
  createdAt: string;
}

export interface Tenant {
  id: string;
  name: string;
  subdomain: string;
  tier: 'free' | 'starter' | 'pro' | 'enterprise';
  status: 'active' | 'suspended' | 'cancelled';
  usageQuota: {
    apiCalls: number;
    storage: number;
    gpuHours: number;
    users: number;
    studies: number;
  };
  usageCurrent: {
    apiCalls: number;
    storage: number;
    gpuHours: number;
    users: number;
    studies: number;
  };
  billingCycle: 'monthly' | 'annual';
  createdAt: string;
}

export interface Patient {
  id: string;
  patientId: string;
  name: string;
  dateOfBirth: string;
  gender: 'M' | 'F' | 'O';
  email?: string;
  phone?: string;
  address?: string;
  medicalRecordNumber?: string;
  createdAt: string;
  updatedAt: string;
}

export interface Study {
  id: string;
  studyInstanceUid: string;
  patientId: string;
  patient?: Patient;
  studyDate: string;
  studyTime?: string;
  studyDescription: string;
  modality: string;
  accessionNumber?: string;
  referringPhysician?: string;
  seriesCount: number;
  instanceCount: number;
  status: 'pending' | 'available' | 'processing' | 'error';
  thumbnailUrl?: string;
  createdAt: string;
  updatedAt: string;
}

export interface Series {
  id: string;
  seriesInstanceUid: string;
  studyId: string;
  seriesNumber: number;
  seriesDescription: string;
  modality: string;
  instanceCount: number;
  bodyPart?: string;
  thumbnailUrl?: string;
}

export interface MLModel {
  id: string;
  name: string;
  version: string;
  description: string;
  modelType: 'detection' | 'segmentation' | 'classification' | 'generation';
  modality: string[];
  accuracy?: number;
  status: 'active' | 'training' | 'deprecated';
  createdAt: string;
  tags: string[];
}

export interface MLInference {
  id: string;
  studyId: string;
  modelId: string;
  model?: MLModel;
  status: 'pending' | 'running' | 'completed' | 'failed';
  confidence?: number;
  results: any;
  processingTime?: number;
  createdAt: string;
  completedAt?: string;
}

export interface Worklist {
  id: string;
  name: string;
  description?: string;
  priority: 'low' | 'normal' | 'high' | 'urgent';
  status: 'active' | 'completed' | 'archived';
  assignedTo?: string;
  itemCount: number;
  completedCount: number;
  createdAt: string;
  dueDate?: string;
}

export interface WorklistItem {
  id: string;
  worklistId: string;
  studyId: string;
  study?: Study;
  status: 'pending' | 'in_progress' | 'completed' | 'cancelled';
  priority: 'low' | 'normal' | 'high' | 'urgent';
  assignedTo?: string;
  notes?: string;
  createdAt: string;
  completedAt?: string;
}

export interface Annotation {
  id: string;
  studyId: string;
  seriesId?: string;
  userId: string;
  annotationType: 'measurement' | 'region' | 'point' | 'text' | 'arrow';
  data: any;
  description?: string;
  createdAt: string;
  updatedAt: string;
}

export interface MetricsData {
  timestamp: string;
  value: number;
  label?: string;
}

export interface SystemMetrics {
  apiRequests: MetricsData[];
  responseTime: MetricsData[];
  errorRate: MetricsData[];
  activeUsers: number;
  storageUsed: number;
  storageTotal: number;
  cpuUsage: number;
  memoryUsage: number;
  gpuUsage?: number;
}

export interface ApiResponse<T> {
  data: T;
  message?: string;
  error?: string;
  pagination?: {
    page: number;
    pageSize: number;
    total: number;
    totalPages: number;
  };
}

export interface PaginationParams {
  page?: number;
  pageSize?: number;
  sortBy?: string;
  sortOrder?: 'asc' | 'desc';
}

export interface FilterParams {
  search?: string;
  status?: string;
  dateFrom?: string;
  dateTo?: string;
  modality?: string;
  [key: string]: any;
}
