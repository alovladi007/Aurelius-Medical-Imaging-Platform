"use client";

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Database,
  Zap,
  Server,
  Activity,
  FileText,
  CheckCircle,
  Clock,
  TrendingUp,
  AlertCircle,
  Wifi,
  Heart,
  TestTube
} from 'lucide-react';

export default function Layer1Page() {
  const [pipelines, setPipelines] = useState([
    { id: 1, name: 'HL7 v2 ADT Stream', type: 'HL7', status: 'running', messagesProcessed: 12483, latency: 45 },
    { id: 2, name: 'FHIR R4 Bulk Export', type: 'FHIR', status: 'running', messagesProcessed: 8921, latency: 120 },
    { id: 3, name: 'DICOMweb Imaging', type: 'DICOM', status: 'running', messagesProcessed: 4521, latency: 230 },
    { id: 4, name: 'Lab Middleware ORU', type: 'HL7', status: 'running', messagesProcessed: 6782, latency: 38 },
    { id: 5, name: 'Wearables MQTT', type: 'IoT', status: 'running', messagesProcessed: 45123, latency: 12 }
  ]);

  const [mappingStats, setMappingStats] = useState({
    snomedMapped: 98.5,
    loincMapped: 97.2,
    rxnormMapped: 99.1,
    icd10Mapped: 96.8
  });

  const connectors = [
    { name: 'HL7 v2', protocols: 'ADT, ORU, ORM', status: 'active', count: 3 },
    { name: 'FHIR R4/R5', protocols: 'REST + Bulk', status: 'active', count: 2 },
    { name: 'DICOMweb', protocols: 'WADO, STOW, QIDO', status: 'active', count: 1 },
    { name: 'Lab Middleware', protocols: 'HL7 LIS', status: 'active', count: 1 },
    { name: 'Bedside Devices', protocols: 'IEEE 11073', status: 'active', count: 4 },
    { name: 'Wearables', protocols: 'BLE → MQTT', status: 'active', count: 1 }
  ];

  return (
    <div className="p-8 space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <Database className="h-8 w-8 text-primary" />
            Layer 1: Clinical Data Ingestion
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            HL7, FHIR, DICOM harmonization into queryable Delta Lake
          </p>
        </div>
        <div className="flex gap-3">
          <Button variant="outline">Add Connector</Button>
          <Button><Zap className="h-4 w-4 mr-2" />New Pipeline</Button>
        </div>
      </div>

      {/* Active Pipelines */}
      <div>
        <h2 className="text-2xl font-bold mb-4">Active Data Pipelines</h2>
        <div className="space-y-3">
          {pipelines.map((pipeline) => (
            <Card key={pipeline.id}>
              <CardContent className="pt-6">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <div className="w-10 h-10 bg-green-100 dark:bg-green-950 rounded-full flex items-center justify-center">
                      <Activity className="h-5 w-5 text-green-600 animate-pulse" />
                    </div>
                    <div>
                      <p className="font-semibold">{pipeline.name}</p>
                      <p className="text-sm text-gray-600 dark:text-gray-400">Type: {pipeline.type}</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-6">
                    <div className="text-center">
                      <p className="text-2xl font-bold">{pipeline.messagesProcessed.toLocaleString()}</p>
                      <p className="text-xs text-gray-600">Messages</p>
                    </div>
                    <div className="text-center">
                      <p className="text-2xl font-bold">{pipeline.latency}ms</p>
                      <p className="text-xs text-gray-600">Latency</p>
                    </div>
                    <Badge variant="default" className="bg-green-600">
                      <Activity className="h-3 w-3 mr-1" />
                      {pipeline.status}
                    </Badge>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>

      {/* Data Connectors */}
      <Card>
        <CardHeader>
          <CardTitle>Configured Connectors</CardTitle>
          <CardDescription>Multi-source clinical data integration</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {connectors.map((connector, index) => (
              <div key={index} className="p-4 border rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <p className="font-semibold">{connector.name}</p>
                  <Badge variant={connector.status === 'active' ? 'default' : 'secondary'}>
                    {connector.status}
                  </Badge>
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">{connector.protocols}</p>
                <p className="text-xs text-gray-500">{connector.count} active pipeline{connector.count > 1 ? 's' : ''}</p>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Terminology Mapping */}
      <Card>
        <CardHeader>
          <CardTitle>Terminology Normalization</CardTitle>
          <CardDescription>SNOMED CT, LOINC, RxNorm, ICD-10-CM mapping rates</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span>SNOMED CT</span>
                <span className="font-mono">{mappingStats.snomedMapped}%</span>
              </div>
              <div className="w-full bg-gray-200 dark:bg-gray-800 rounded-full h-2">
                <div className="bg-blue-600 h-2 rounded-full" style={{ width: `${mappingStats.snomedMapped}%` }} />
              </div>
            </div>
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span>LOINC (Labs)</span>
                <span className="font-mono">{mappingStats.loincMapped}%</span>
              </div>
              <div className="w-full bg-gray-200 dark:bg-gray-800 rounded-full h-2">
                <div className="bg-green-600 h-2 rounded-full" style={{ width: `${mappingStats.loincMapped}%` }} />
              </div>
            </div>
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span>RxNorm (Medications)</span>
                <span className="font-mono">{mappingStats.rxnormMapped}%</span>
              </div>
              <div className="w-full bg-gray-200 dark:bg-gray-800 rounded-full h-2">
                <div className="bg-purple-600 h-2 rounded-full" style={{ width: `${mappingStats.rxnormMapped}%` }} />
              </div>
            </div>
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span>ICD-10-CM (Diagnoses)</span>
                <span className="font-mono">{mappingStats.icd10Mapped}%</span>
              </div>
              <div className="w-full bg-gray-200 dark:bg-gray-800 rounded-full h-2">
                <div className="bg-orange-600 h-2 rounded-full" style={{ width: `${mappingStats.icd10Mapped}%` }} />
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* De-identification Modes */}
      <Card>
        <CardHeader>
          <CardTitle>De-identification & Privacy</CardTitle>
          <CardDescription>Configurable modes based on purpose-of-use</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-4 border rounded-lg">
              <CheckCircle className="h-6 w-6 text-green-600 mb-2" />
              <p className="font-semibold mb-1">Safe Harbor</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">Research use, HIPAA compliant de-identification</p>
            </div>
            <div className="p-4 border rounded-lg">
              <FileText className="h-6 w-6 text-blue-600 mb-2" />
              <p className="font-semibold mb-1">Limited Data Set</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">Partial identifiers for approved studies</p>
            </div>
            <div className="p-4 border rounded-lg">
              <Heart className="h-6 w-6 text-red-600 mb-2" />
              <p className="font-semibold mb-1">Fully Identified</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">Care operations, authorized access only</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Data Quality */}
      <Card>
        <CardHeader>
          <CardTitle>Data Quality Monitoring</CardTitle>
          <CardDescription>Schema validation, drift detection, cohort building</CardDescription>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="validation">
            <TabsList>
              <TabsTrigger value="validation">Validation</TabsTrigger>
              <TabsTrigger value="drift">Drift Detection</TabsTrigger>
              <TabsTrigger value="missingness">Missingness</TabsTrigger>
            </TabsList>

            <TabsContent value="validation" className="space-y-3">
              <div className="flex items-center justify-between p-3 bg-green-50 dark:bg-green-950 rounded">
                <span className="text-sm">Schema Validation</span>
                <Badge variant="default" className="bg-green-600">Passing</Badge>
              </div>
              <div className="flex items-center justify-between p-3 bg-green-50 dark:bg-green-950 rounded">
                <span className="text-sm">Semantic Validation</span>
                <Badge variant="default" className="bg-green-600">Passing</Badge>
              </div>
              <div className="flex items-center justify-between p-3 bg-green-50 dark:bg-green-950 rounded">
                <span className="text-sm">Unit Canonicalization (UCUM)</span>
                <Badge variant="default" className="bg-green-600">Active</Badge>
              </div>
            </TabsContent>

            <TabsContent value="drift">
              <p className="text-sm text-gray-600 dark:text-gray-400">
                No significant drift detected in the last 24 hours.
              </p>
              <div className="mt-3 text-xs text-gray-500">
                Monitored metrics: field distributions, value ranges, vocabulary coverage
              </div>
            </TabsContent>

            <TabsContent value="missingness">
              <div className="space-y-2 text-sm">
                <p><strong>Patient Demographics:</strong> 0.2% missing</p>
                <p><strong>Lab Results:</strong> 1.5% missing</p>
                <p><strong>Medications:</strong> 0.8% missing</p>
                <p><strong>Imaging Reports:</strong> 3.2% missing</p>
              </div>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
}
