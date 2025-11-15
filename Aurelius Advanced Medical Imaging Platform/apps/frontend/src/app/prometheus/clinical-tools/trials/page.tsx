"use client";

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import {
  Microscope,
  Search,
  CheckCircle,
  XCircle,
  AlertCircle,
  MapPin,
  Calendar,
  Users,
  FileText,
  TrendingUp,
  Filter
} from 'lucide-react';

export default function TrialMatchingPage() {
  const [selectedTrial, setSelectedTrial] = useState<any>(null);

  const patientProfile = {
    id: 'P-2024-001',
    age: 67,
    gender: 'Male',
    diagnoses: ['Non-Small Cell Lung Cancer', 'Stage IIIB', 'EGFR+'],
    priorTreatments: ['Platinum-based chemotherapy', 'Radiation therapy'],
    ecog: 1,
    labs: {
      wbc: 6.5,
      platelets: 180,
      creatinine: 1.1,
      alt: 32,
      ast: 28
    },
    location: 'Boston, MA'
  };

  const trials = [
    {
      id: 'NCT04567890',
      title: 'Phase III Study of Osimertinib vs Platinum-Pemetrexed in EGFR+ NSCLC',
      phase: 'Phase III',
      sponsor: 'AstraZeneca',
      condition: 'Non-Small Cell Lung Cancer',
      status: 'Recruiting',
      eligibility: {
        match: 95,
        met: ['EGFR+ NSCLC', 'Stage IIIB-IV', 'Prior platinum therapy', 'ECOG 0-1', 'Adequate organ function'],
        notMet: [],
        unclear: ['PD-L1 status']
      },
      sites: [
        { name: 'Dana-Farber Cancer Institute', location: 'Boston, MA', distance: 2.3, status: 'Recruiting' },
        { name: 'Mass General Hospital', location: 'Boston, MA', distance: 3.1, status: 'Recruiting' }
      ],
      arms: [
        'Osimertinib 80mg once daily',
        'Platinum-based chemotherapy + Pemetrexed'
      ],
      primaryEndpoint: 'Progression-free survival',
      estimatedEnrollment: 450,
      estimatedCompletion: '2026-12'
    },
    {
      id: 'NCT03456789',
      title: 'Combination Immunotherapy for Advanced NSCLC',
      phase: 'Phase II',
      sponsor: 'Bristol Myers Squibb',
      condition: 'Non-Small Cell Lung Cancer',
      status: 'Recruiting',
      eligibility: {
        match: 88,
        met: ['NSCLC Stage III-IV', 'ECOG 0-2', 'Prior systemic therapy'],
        notMet: ['Requires PD-L1 >50%'],
        unclear: []
      },
      sites: [
        { name: 'Dana-Farber Cancer Institute', location: 'Boston, MA', distance: 2.3, status: 'Recruiting' }
      ],
      arms: [
        'Nivolumab + Ipilimumab',
        'Nivolumab monotherapy'
      ],
      primaryEndpoint: 'Overall response rate',
      estimatedEnrollment: 120,
      estimatedCompletion: '2025-08'
    },
    {
      id: 'NCT02345678',
      title: 'Novel TKI for EGFR+ NSCLC with Brain Metastases',
      phase: 'Phase I/II',
      sponsor: 'Takeda',
      condition: 'Non-Small Cell Lung Cancer',
      status: 'Recruiting',
      eligibility: {
        match: 78,
        met: ['EGFR+ NSCLC', 'Brain metastases allowed', 'Prior TKI failure'],
        notMet: ['Requires T790M mutation'],
        unclear: ['Leptomeningeal disease status']
      },
      sites: [
        { name: 'Mass General Hospital', location: 'Boston, MA', distance: 3.1, status: 'Recruiting' },
        { name: 'Yale Cancer Center', location: 'New Haven, CT', distance: 115, status: 'Recruiting' }
      ],
      arms: [
        'Experimental TKI dose escalation'
      ],
      primaryEndpoint: 'Maximum tolerated dose & CNS response rate',
      estimatedEnrollment: 60,
      estimatedCompletion: '2025-06'
    }
  ];

  const eligibilityCheck = (criteria: any) => {
    const total = criteria.met.length + criteria.notMet.length + criteria.unclear.length;
    const matched = criteria.met.length;
    return {
      percentage: Math.round((matched / total) * 100),
      status: criteria.notMet.length === 0 ? 'eligible' : 'partial'
    };
  };

  return (
    <div className="p-8 space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <Microscope className="h-8 w-8 text-primary" />
            Clinical Trial Matching
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            UCG-based eligibility filtering with automated pre-screening
          </p>
        </div>
        <div className="flex gap-3">
          <Button variant="outline">
            <Filter className="h-4 w-4 mr-2" />
            Filter Trials
          </Button>
          <Button>
            <FileText className="h-4 w-4 mr-2" />
            Export Pre-Screen
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Patient Profile */}
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Patient Profile</CardTitle>
              <CardDescription>Automated eligibility matching</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <p className="text-sm font-semibold mb-2">Demographics</p>
                <div className="space-y-1 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Age:</span>
                    <span className="font-semibold">{patientProfile.age}y</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Gender:</span>
                    <span className="font-semibold">{patientProfile.gender}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Location:</span>
                    <span className="font-semibold text-xs">{patientProfile.location}</span>
                  </div>
                </div>
              </div>

              <div>
                <p className="text-sm font-semibold mb-2">Diagnoses</p>
                <div className="flex flex-wrap gap-1">
                  {patientProfile.diagnoses.map((dx, i) => (
                    <Badge key={i} variant="destructive" className="text-xs">{dx}</Badge>
                  ))}
                </div>
              </div>

              <div>
                <p className="text-sm font-semibold mb-2">Prior Treatments</p>
                <div className="flex flex-wrap gap-1">
                  {patientProfile.priorTreatments.map((tx, i) => (
                    <Badge key={i} variant="secondary" className="text-xs">{tx}</Badge>
                  ))}
                </div>
              </div>

              <div>
                <p className="text-sm font-semibold mb-2">Performance Status</p>
                <Badge variant="outline">ECOG {patientProfile.ecog}</Badge>
              </div>

              <div>
                <p className="text-sm font-semibold mb-2">Recent Labs</p>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div className="p-2 border rounded">
                    <p className="text-gray-600">WBC</p>
                    <p className="font-semibold">{patientProfile.labs.wbc} K/µL</p>
                  </div>
                  <div className="p-2 border rounded">
                    <p className="text-gray-600">Platelets</p>
                    <p className="font-semibold">{patientProfile.labs.platelets} K/µL</p>
                  </div>
                  <div className="p-2 border rounded">
                    <p className="text-gray-600">Creatinine</p>
                    <p className="font-semibold">{patientProfile.labs.creatinine} mg/dL</p>
                  </div>
                  <div className="p-2 border rounded">
                    <p className="text-gray-600">ALT</p>
                    <p className="font-semibold">{patientProfile.labs.alt} U/L</p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Matching Summary */}
          <Card className="border-green-200 dark:border-green-900">
            <CardHeader>
              <CardTitle className="text-lg">Matching Summary</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <div className="text-center p-4 bg-green-50 dark:bg-green-950 rounded-lg">
                  <p className="text-4xl font-bold text-green-600">{trials.length}</p>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">Eligible Trials Found</p>
                </div>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Phase III:</span>
                    <span className="font-semibold">1</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Phase II:</span>
                    <span className="font-semibold">1</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Phase I/II:</span>
                    <span className="font-semibold">1</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Local (< 10mi):</span>
                    <span className="font-semibold">2 trials</span>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Trials List */}
        <div className="lg:col-span-2 space-y-4">
          {trials.map((trial) => {
            const check = eligibilityCheck(trial.eligibility);
            return (
              <Card
                key={trial.id}
                className={`cursor-pointer transition-all ${
                  selectedTrial?.id === trial.id ? 'border-primary border-2' : 'hover:shadow-lg'
                }`}
                onClick={() => setSelectedTrial(trial)}
              >
                <CardHeader>
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-2">
                        <Badge>{trial.phase}</Badge>
                        <Badge variant="outline" className="text-xs">{trial.id}</Badge>
                        <Badge className={trial.status === 'Recruiting' ? 'bg-green-600' : 'bg-gray-600'}>
                          {trial.status}
                        </Badge>
                      </div>
                      <CardTitle className="text-lg">{trial.title}</CardTitle>
                      <CardDescription className="mt-2">
                        {trial.sponsor} • {trial.condition}
                      </CardDescription>
                    </div>
                    <div className="text-right ml-4">
                      <div className="w-16 h-16 rounded-full border-4 border-green-500 flex items-center justify-center">
                        <p className="text-2xl font-bold text-green-600">{check.percentage}%</p>
                      </div>
                      <p className="text-xs text-gray-600 mt-1">Match</p>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  {/* Eligibility Criteria */}
                  <div className="space-y-3">
                    {/* Met Criteria */}
                    {trial.eligibility.met.length > 0 && (
                      <div>
                        <p className="text-sm font-semibold mb-2 flex items-center gap-2">
                          <CheckCircle className="h-4 w-4 text-green-600" />
                          Criteria Met ({trial.eligibility.met.length})
                        </p>
                        <div className="flex flex-wrap gap-1">
                          {trial.eligibility.met.map((criterion, idx) => (
                            <Badge key={idx} variant="outline" className="bg-green-50 text-green-800 dark:bg-green-950 text-xs">
                              ✓ {criterion}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Not Met Criteria */}
                    {trial.eligibility.notMet.length > 0 && (
                      <div>
                        <p className="text-sm font-semibold mb-2 flex items-center gap-2">
                          <XCircle className="h-4 w-4 text-red-600" />
                          Criteria Not Met ({trial.eligibility.notMet.length})
                        </p>
                        <div className="flex flex-wrap gap-1">
                          {trial.eligibility.notMet.map((criterion, idx) => (
                            <Badge key={idx} variant="outline" className="bg-red-50 text-red-800 dark:bg-red-950 text-xs">
                              ✗ {criterion}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Unclear Criteria */}
                    {trial.eligibility.unclear.length > 0 && (
                      <div>
                        <p className="text-sm font-semibold mb-2 flex items-center gap-2">
                          <AlertCircle className="h-4 w-4 text-yellow-600" />
                          Needs Verification ({trial.eligibility.unclear.length})
                        </p>
                        <div className="flex flex-wrap gap-1">
                          {trial.eligibility.unclear.map((criterion, idx) => (
                            <Badge key={idx} variant="outline" className="bg-yellow-50 text-yellow-800 dark:bg-yellow-950 text-xs">
                              ? {criterion}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Trial Sites */}
                  <div className="mt-4">
                    <p className="text-sm font-semibold mb-2 flex items-center gap-2">
                      <MapPin className="h-4 w-4" />
                      Nearby Sites ({trial.sites.length})
                    </p>
                    <div className="space-y-2">
                      {trial.sites.map((site, idx) => (
                        <div key={idx} className="flex items-center justify-between p-2 bg-gray-50 dark:bg-gray-900 rounded text-sm">
                          <div className="flex-1">
                            <p className="font-semibold">{site.name}</p>
                            <p className="text-xs text-gray-600">{site.location} • {site.distance} miles</p>
                          </div>
                          <Badge className="bg-green-600">{site.status}</Badge>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Study Details */}
                  <div className="mt-4 grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <p className="text-gray-600">Primary Endpoint:</p>
                      <p className="font-semibold">{trial.primaryEndpoint}</p>
                    </div>
                    <div>
                      <p className="text-gray-600">Enrollment:</p>
                      <p className="font-semibold">{trial.estimatedEnrollment} patients</p>
                    </div>
                    <div>
                      <p className="text-gray-600">Treatment Arms:</p>
                      <p className="font-semibold">{trial.arms.length} arms</p>
                    </div>
                    <div>
                      <p className="text-gray-600">Est. Completion:</p>
                      <p className="font-semibold">{trial.estimatedCompletion}</p>
                    </div>
                  </div>

                  <div className="flex gap-2 mt-4">
                    <Button className="flex-1" size="sm">
                      <FileText className="h-4 w-4 mr-1" />
                      View Full Protocol
                    </Button>
                    <Button variant="outline" className="flex-1" size="sm">
                      <Users className="h-4 w-4 mr-1" />
                      Contact PI
                    </Button>
                  </div>
                </CardContent>
              </Card>
            );
          })}

          {trials.length === 0 && (
            <Card>
              <CardContent className="py-12">
                <div className="text-center text-gray-500">
                  <Search className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p>No eligible trials found for this patient</p>
                  <Button variant="outline" className="mt-4">
                    Expand Search Criteria
                  </Button>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
