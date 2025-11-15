"use client";

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import {
  Upload,
  X,
  Loader2,
  CheckCircle,
  AlertTriangle,
  ArrowLeft,
  Brain
} from 'lucide-react';

interface PredictionResult {
  cancer_type: string;
  risk_score: number;
  confidence: number;
  uncertainty: number;
  recommendations: string[];
  all_probabilities?: Record<string, number>;
}

export default function CancerAIPrediction() {
  const router = useRouter();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Form data
  const [clinicalNotes, setClinicalNotes] = useState('');
  const [patientAge, setPatientAge] = useState('');
  const [patientGender, setPatientGender] = useState('');
  const [smokingHistory, setSmokingHistory] = useState(false);
  const [familyHistory, setFamilyHistory] = useState(false);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setError(null);

      // Create preview if image
      if (file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onloadend = () => {
          setPreview(reader.result as string);
        };
        reader.readAsDataURL(file);
      } else {
        setPreview(null);
      }
    }
  };

  const handleRemoveFile = () => {
    setSelectedFile(null);
    setPreview(null);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile) {
      setError('Please select an image file');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('image', selectedFile);
      formData.append('clinical_notes', clinicalNotes);
      formData.append('patient_age', patientAge || '0');
      formData.append('patient_gender', patientGender);
      formData.append('smoking_history', smokingHistory.toString());
      formData.append('family_history', familyHistory.toString());

      const response = await fetch('/api/cancer-ai/predict', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error('Prediction failed');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const getRiskColor = (score: number) => {
    if (score < 0.3) return 'text-green-600';
    if (score < 0.7) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getRiskLabel = (score: number) => {
    if (score < 0.3) return 'Low Risk';
    if (score < 0.7) return 'Moderate Risk';
    return 'High Risk';
  };

  return (
    <div className="p-8 max-w-6xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <Button
          variant="ghost"
          onClick={() => router.push('/cancer-ai')}
          className="mb-4"
        >
          <ArrowLeft className="h-4 w-4 mr-2" />
          Back to Dashboard
        </Button>
        <h1 className="text-4xl font-bold mb-2">Cancer AI Prediction</h1>
        <p className="text-gray-600 dark:text-gray-400">
          Upload a medical image for AI-powered cancer detection analysis
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Input Form */}
        <div>
          <form onSubmit={handleSubmit} className="space-y-6">
            {/* File Upload */}
            <Card>
              <CardHeader>
                <CardTitle>Medical Image</CardTitle>
                <CardDescription>
                  Upload DICOM, PNG, JPG, or other medical imaging formats
                </CardDescription>
              </CardHeader>
              <CardContent>
                {!selectedFile ? (
                  <label className="flex flex-col items-center justify-center w-full h-64 border-2 border-dashed rounded-lg cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors">
                    <div className="flex flex-col items-center justify-center pt-5 pb-6">
                      <Upload className="h-12 w-12 text-gray-400 mb-4" />
                      <p className="mb-2 text-sm text-gray-500">
                        <span className="font-semibold">Click to upload</span> or drag and drop
                      </p>
                      <p className="text-xs text-gray-500">
                        DICOM, PNG, JPG, TIFF (MAX. 50MB)
                      </p>
                    </div>
                    <input
                      type="file"
                      className="hidden"
                      onChange={handleFileChange}
                      accept="image/*,.dcm"
                    />
                  </label>
                ) : (
                  <div className="relative">
                    {preview && (
                      <img
                        src={preview}
                        alt="Preview"
                        className="w-full h-64 object-contain rounded-lg bg-gray-100 dark:bg-gray-800"
                      />
                    )}
                    {!preview && (
                      <div className="flex items-center justify-center h-64 bg-gray-100 dark:bg-gray-800 rounded-lg">
                        <p className="text-sm text-gray-600">{selectedFile.name}</p>
                      </div>
                    )}
                    <Button
                      type="button"
                      variant="destructive"
                      size="sm"
                      className="absolute top-2 right-2"
                      onClick={handleRemoveFile}
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Clinical Information */}
            <Card>
              <CardHeader>
                <CardTitle>Clinical Information (Optional)</CardTitle>
                <CardDescription>
                  Additional data helps improve prediction accuracy
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <Label htmlFor="age">Patient Age</Label>
                    <Input
                      id="age"
                      type="number"
                      placeholder="e.g., 55"
                      value={patientAge}
                      onChange={(e) => setPatientAge(e.target.value)}
                    />
                  </div>
                  <div>
                    <Label htmlFor="gender">Gender</Label>
                    <select
                      id="gender"
                      className="w-full p-2 border rounded-md"
                      value={patientGender}
                      onChange={(e) => setPatientGender(e.target.value)}
                    >
                      <option value="">Select...</option>
                      <option value="M">Male</option>
                      <option value="F">Female</option>
                      <option value="O">Other</option>
                    </select>
                  </div>
                </div>

                <div>
                  <Label htmlFor="notes">Clinical Notes</Label>
                  <Textarea
                    id="notes"
                    placeholder="Enter any relevant clinical observations..."
                    rows={3}
                    value={clinicalNotes}
                    onChange={(e) => setClinicalNotes(e.target.value)}
                  />
                </div>

                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      id="smoking"
                      checked={smokingHistory}
                      onChange={(e) => setSmokingHistory(e.target.checked)}
                      className="w-4 h-4"
                    />
                    <Label htmlFor="smoking" className="cursor-pointer">
                      Smoking History
                    </Label>
                  </div>
                  <div className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      id="family"
                      checked={familyHistory}
                      onChange={(e) => setFamilyHistory(e.target.checked)}
                      className="w-4 h-4"
                    />
                    <Label htmlFor="family" className="cursor-pointer">
                      Family History of Cancer
                    </Label>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Submit Button */}
            <Button
              type="submit"
              size="lg"
              className="w-full"
              disabled={!selectedFile || loading}
            >
              {loading ? (
                <>
                  <Loader2 className="h-5 w-5 mr-2 animate-spin" />
                  Analyzing Image...
                </>
              ) : (
                <>
                  <Brain className="h-5 w-5 mr-2" />
                  Analyze with AI
                </>
              )}
            </Button>
          </form>
        </div>

        {/* Results */}
        <div>
          {error && (
            <Card className="border-red-200 bg-red-50 dark:bg-red-950">
              <CardHeader>
                <div className="flex items-center gap-2">
                  <AlertTriangle className="h-5 w-5 text-red-600" />
                  <CardTitle className="text-red-900 dark:text-red-100">
                    Error
                  </CardTitle>
                </div>
              </CardHeader>
              <CardContent className="text-sm text-red-800 dark:text-red-200">
                {error}
              </CardContent>
            </Card>
          )}

          {result && (
            <div className="space-y-6">
              {/* Main Result */}
              <Card>
                <CardHeader>
                  <div className="flex items-center gap-2">
                    <CheckCircle className="h-6 w-6 text-green-600" />
                    <CardTitle>Prediction Result</CardTitle>
                  </div>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <div className="text-sm text-gray-600 mb-1">Detected Type</div>
                    <div className="text-2xl font-bold">{result.cancer_type}</div>
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <div className="text-sm text-gray-600 mb-1">Confidence</div>
                      <div className="text-xl font-semibold">
                        {(result.confidence * 100).toFixed(1)}%
                      </div>
                    </div>
                    <div>
                      <div className="text-sm text-gray-600 mb-1">Risk Level</div>
                      <div className={`text-xl font-semibold ${getRiskColor(result.risk_score)}`}>
                        {getRiskLabel(result.risk_score)}
                      </div>
                    </div>
                  </div>

                  <div>
                    <div className="text-sm text-gray-600 mb-2">Risk Score</div>
                    <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3">
                      <div
                        className={`h-3 rounded-full ${
                          result.risk_score < 0.3
                            ? 'bg-green-500'
                            : result.risk_score < 0.7
                            ? 'bg-yellow-500'
                            : 'bg-red-500'
                        }`}
                        style={{ width: `${result.risk_score * 100}%` }}
                      />
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Recommendations */}
              <Card>
                <CardHeader>
                  <CardTitle>Clinical Recommendations</CardTitle>
                </CardHeader>
                <CardContent>
                  <ul className="space-y-2">
                    {result.recommendations.map((rec, idx) => (
                      <li key={idx} className="flex items-start gap-2">
                        <div className="mt-1 text-blue-600">•</div>
                        <div className="text-sm">{rec}</div>
                      </li>
                    ))}
                  </ul>
                </CardContent>
              </Card>

              {/* All Probabilities */}
              {result.all_probabilities && (
                <Card>
                  <CardHeader>
                    <CardTitle>Detailed Probabilities</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    {Object.entries(result.all_probabilities).map(([type, prob]) => (
                      <div key={type}>
                        <div className="flex justify-between text-sm mb-1">
                          <span>{type}</span>
                          <span className="font-semibold">{(prob * 100).toFixed(1)}%</span>
                        </div>
                        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                          <div
                            className="bg-blue-600 h-2 rounded-full"
                            style={{ width: `${prob * 100}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </CardContent>
                </Card>
              )}

              {/* Actions */}
              <div className="flex gap-4">
                <Button
                  variant="outline"
                  onClick={() => {
                    setResult(null);
                    setSelectedFile(null);
                    setPreview(null);
                  }}
                >
                  New Prediction
                </Button>
                <Button variant="outline">Save Result</Button>
                <Button variant="outline">Export PDF</Button>
              </div>
            </div>
          )}

          {!result && !error && (
            <Card>
              <CardContent className="flex flex-col items-center justify-center h-96 text-center">
                <Brain className="h-16 w-16 text-gray-300 mb-4" />
                <p className="text-gray-500">
                  Upload an image and submit to see AI prediction results
                </p>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
