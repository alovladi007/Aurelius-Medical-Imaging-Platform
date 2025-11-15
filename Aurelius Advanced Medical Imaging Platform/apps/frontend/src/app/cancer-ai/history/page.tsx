"use client";

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import {
  Search,
  Filter,
  Download,
  Eye,
  Calendar,
  TrendingUp,
  AlertCircle
} from 'lucide-react';

export default function HistoryPage() {
  const [predictions, setPredictions] = useState([
    {
      id: 1,
      patientId: 'P-2024-001',
      imageName: 'chest_ct_001.dcm',
      prediction: 'Lung Cancer',
      confidence: 94.5,
      risk: 'High',
      date: '2025-11-15 10:30',
      modality: 'CT',
      clinician: 'Dr. Smith'
    },
    {
      id: 2,
      patientId: 'P-2024-002',
      imageName: 'mammo_002.dcm',
      prediction: 'Breast Cancer',
      confidence: 89.2,
      risk: 'Medium',
      date: '2025-11-15 09:15',
      modality: 'Mammography',
      clinician: 'Dr. Johnson'
    },
    {
      id: 3,
      patientId: 'P-2024-003',
      imageName: 'prostate_mri_003.dcm',
      prediction: 'No Cancer',
      confidence: 96.8,
      risk: 'Low',
      date: '2025-11-14 16:45',
      modality: 'MRI',
      clinician: 'Dr. Williams'
    },
    {
      id: 4,
      patientId: 'P-2024-004',
      imageName: 'colon_ct_004.dcm',
      prediction: 'Colorectal Cancer',
      confidence: 91.3,
      risk: 'High',
      date: '2025-11-14 14:20',
      modality: 'CT',
      clinician: 'Dr. Brown'
    },
    {
      id: 5,
      patientId: 'P-2024-005',
      imageName: 'lung_xray_005.png',
      prediction: 'No Cancer',
      confidence: 88.7,
      risk: 'Low',
      date: '2025-11-14 11:10',
      modality: 'X-Ray',
      clinician: 'Dr. Davis'
    }
  ]);

  const [filteredPredictions, setFilteredPredictions] = useState(predictions);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterType, setFilterType] = useState('all');
  const [selectedPrediction, setSelectedPrediction] = useState(null);

  useEffect(() => {
    let filtered = predictions;

    if (searchTerm) {
      filtered = filtered.filter(p =>
        p.patientId.toLowerCase().includes(searchTerm.toLowerCase()) ||
        p.imageName.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    if (filterType !== 'all') {
      filtered = filtered.filter(p => p.risk.toLowerCase() === filterType);
    }

    setFilteredPredictions(filtered);
  }, [searchTerm, filterType, predictions]);

  const getRiskBadge = (risk: string) => {
    switch (risk) {
      case 'High':
        return <Badge variant="destructive">{risk}</Badge>;
      case 'Medium':
        return <Badge className="bg-orange-600">{risk}</Badge>;
      case 'Low':
        return <Badge className="bg-green-600">{risk}</Badge>;
      default:
        return <Badge variant="outline">{risk}</Badge>;
    }
  };

  return (
    <div className="p-8 space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Prediction History</h1>
          <p className="text-gray-600 dark:text-gray-400">View and manage past cancer AI predictions</p>
        </div>
        <Button>
          <Download className="h-4 w-4 mr-2" />
          Export All
        </Button>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Total Predictions</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold">{predictions.length}</p>
            <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">All time</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Cancer Detected</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold text-red-600">
              {predictions.filter(p => p.prediction !== 'No Cancer').length}
            </p>
            <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
              {((predictions.filter(p => p.prediction !== 'No Cancer').length / predictions.length) * 100).toFixed(1)}% positive
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Avg Confidence</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold text-primary">
              {(predictions.reduce((acc, p) => acc + p.confidence, 0) / predictions.length).toFixed(1)}%
            </p>
            <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">Across all predictions</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">High Risk Cases</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold text-orange-600">
              {predictions.filter(p => p.risk === 'High').length}
            </p>
            <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">Require followup</p>
          </CardContent>
        </Card>
      </div>

      {/* Filters */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex gap-4">
            <div className="flex-1">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
                <Input
                  placeholder="Search by patient ID or image name..."
                  className="pl-10"
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                />
              </div>
            </div>
            <Select value={filterType} onValueChange={setFilterType}>
              <SelectTrigger className="w-48">
                <Filter className="h-4 w-4 mr-2" />
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Risk Levels</SelectItem>
                <SelectItem value="high">High Risk</SelectItem>
                <SelectItem value="medium">Medium Risk</SelectItem>
                <SelectItem value="low">Low Risk</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      {/* Predictions Table */}
      <Card>
        <CardHeader>
          <CardTitle>All Predictions ({filteredPredictions.length})</CardTitle>
          <CardDescription>Complete history of cancer AI analyses</CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Patient ID</TableHead>
                <TableHead>Image</TableHead>
                <TableHead>Prediction</TableHead>
                <TableHead>Confidence</TableHead>
                <TableHead>Risk</TableHead>
                <TableHead>Date</TableHead>
                <TableHead>Clinician</TableHead>
                <TableHead>Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filteredPredictions.map((pred) => (
                <TableRow key={pred.id} className="cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-900">
                  <TableCell className="font-medium">{pred.patientId}</TableCell>
                  <TableCell>
                    <div>
                      <p className="text-sm">{pred.imageName}</p>
                      <Badge variant="outline" className="mt-1">{pred.modality}</Badge>
                    </div>
                  </TableCell>
                  <TableCell>
                    <Badge variant={pred.prediction === 'No Cancer' ? 'default' : 'destructive'}>
                      {pred.prediction}
                    </Badge>
                  </TableCell>
                  <TableCell>
                    <div className="flex items-center gap-2">
                      <div className="w-16 bg-gray-200 dark:bg-gray-800 rounded-full h-2">
                        <div
                          className="bg-primary h-2 rounded-full"
                          style={{ width: `${pred.confidence}%` }}
                        />
                      </div>
                      <span className="text-sm font-mono">{pred.confidence}%</span>
                    </div>
                  </TableCell>
                  <TableCell>{getRiskBadge(pred.risk)}</TableCell>
                  <TableCell className="text-sm text-gray-600">{pred.date}</TableCell>
                  <TableCell className="text-sm">{pred.clinician}</TableCell>
                  <TableCell>
                    <Button variant="ghost" size="sm" onClick={() => setSelectedPrediction(pred)}>
                      <Eye className="h-4 w-4" />
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      {/* Detail Modal */}
      {selectedPrediction && (
        <Card className="border-primary">
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle>Prediction Details</CardTitle>
              <Button variant="ghost" size="sm" onClick={() => setSelectedPrediction(null)}>✕</Button>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">Patient ID</p>
                <p className="font-semibold">{selectedPrediction.patientId}</p>
              </div>
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">Image</p>
                <p className="font-semibold">{selectedPrediction.imageName}</p>
              </div>
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">Prediction</p>
                <Badge variant={selectedPrediction.prediction === 'No Cancer' ? 'default' : 'destructive'}>
                  {selectedPrediction.prediction}
                </Badge>
              </div>
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">Confidence</p>
                <p className="font-semibold text-primary">{selectedPrediction.confidence}%</p>
              </div>
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">Risk Level</p>
                {getRiskBadge(selectedPrediction.risk)}
              </div>
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">Modality</p>
                <p className="font-semibold">{selectedPrediction.modality}</p>
              </div>
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">Date & Time</p>
                <p className="font-semibold">{selectedPrediction.date}</p>
              </div>
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">Clinician</p>
                <p className="font-semibold">{selectedPrediction.clinician}</p>
              </div>
            </div>
            <div className="flex gap-3 pt-4">
              <Button className="flex-1">View Full Report</Button>
              <Button variant="outline" className="flex-1">Download PDF</Button>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
