"use client";

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  GraduationCap,
  BookOpen,
  Play,
  CheckCircle,
  XCircle,
  Award,
  TrendingUp,
  Brain,
  Stethoscope,
  Clock,
  Target,
  Star
} from 'lucide-react';

export default function EducationPortalPage() {
  const [selectedCase, setSelectedCase] = useState<any>(null);
  const [quizAnswers, setQuizAnswers] = useState<Record<number, number>>({});

  const studentProfile = {
    name: 'Medical Student',
    level: 'MS3',
    completedCases: 67,
    accuracy: 84.2,
    streak: 12,
    badges: ['Cardiology Expert', 'Quick Learner', 'Critical Thinker']
  };

  const caseLibrary = [
    {
      id: 1,
      title: 'Chest Pain in 67yo Male',
      specialty: 'Cardiology',
      difficulty: 'Intermediate',
      duration: '15 min',
      learningObjectives: [
        'Calculate and interpret HEART score',
        'Differentiate ACS from other causes of chest pain',
        'Apply evidence-based management pathways'
      ],
      completed: false,
      score: null
    },
    {
      id: 2,
      title: 'Septic Shock Management',
      specialty: 'Critical Care',
      difficulty: 'Advanced',
      duration: '20 min',
      learningObjectives: [
        'Recognize sepsis criteria (Sepsis-3)',
        'Apply Surviving Sepsis Campaign bundle',
        'Manage vasopressors and fluid resuscitation'
      ],
      completed: true,
      score: 92
    },
    {
      id: 3,
      title: 'Type 2 Diabetes Optimization',
      specialty: 'Endocrinology',
      difficulty: 'Beginner',
      duration: '10 min',
      learningObjectives: [
        'Interpret HbA1c and glucose patterns',
        'Select appropriate oral agents',
        'Apply ADA treatment algorithm'
      ],
      completed: true,
      score: 88
    }
  ];

  const caseStem = {
    presentation: `67-year-old male presents to ED with chest pain that started 2 hours ago. He describes it as "pressure-like" and radiating to his left arm. He has a history of hypertension, diabetes, and hyperlipidemia. He takes aspirin, metoprolol, metformin, and atorvastatin.

Vitals: BP 145/92, HR 88, RR 18, SpO2 96% on RA, Temp 98.6°F

Physical Exam: Uncomfortable-appearing male. Cardiovascular: Regular rate and rhythm, no murmurs. Lungs: Clear bilaterally. No peripheral edema.`,
    images: [
      { type: 'ECG', finding: 'Sinus rhythm, no acute ST changes' },
      { type: 'CXR', finding: 'No acute cardiopulmonary process' }
    ],
    labs: [
      { test: 'Troponin I', value: '0.04 ng/mL', flag: 'normal' },
      { test: 'Glucose', value: '156 mg/dL', flag: 'high' },
      { test: 'Creatinine', value: '1.1 mg/dL', flag: 'normal' }
    ]
  };

  const vivaQuestions = [
    {
      id: 1,
      question: 'What is this patient\'s HEART score?',
      options: ['2', '3', '4', '5'],
      correct: 2, // index 2 = "4"
      explanation: 'HEART score is 4: History (moderately suspicious = 1), ECG (no ST changes = 0), Age (≥65 = 2), Risk factors (≥3 = 2), Troponin (normal = 0). Total = 4.',
      reference: 'Six AJ et al. Neth Heart J. 2008;16(6):191-196.'
    },
    {
      id: 2,
      question: 'What is the most appropriate next step in management?',
      options: [
        'Discharge home with outpatient follow-up',
        'Serial troponins at 3 hours',
        'Immediate cardiac catheterization',
        'Start thrombolytics'
      ],
      correct: 1,
      explanation: 'HEART score 4-6 indicates moderate risk. Serial troponins at 0h and 3h are recommended per ESC guidelines.',
      reference: 'ESC Guidelines for ACS 2023'
    },
    {
      id: 3,
      question: 'Which medication should be added at this time?',
      options: [
        'Nitroglycerin sublingual',
        'Heparin infusion',
        'Clopidogrel loading dose',
        'Morphine IV'
      ],
      correct: 0,
      explanation: 'Nitroglycerin sublingual is appropriate for symptomatic relief and can help differentiate cardiac from non-cardiac pain. Definitive antiplatelet/anticoagulation awaits serial troponins.',
      reference: 'ACC/AHA Guidelines'
    }
  ];

  const performanceMetrics = [
    { category: 'Cardiology', cases: 23, accuracy: 87.5, avgTime: '12 min' },
    { category: 'Pulmonology', cases: 15, accuracy: 82.1, avgTime: '14 min' },
    { category: 'Gastroenterology', cases: 12, accuracy: 91.2, avgTime: '11 min' },
    { category: 'Endocrinology', cases: 17, accuracy: 78.9, avgTime: '13 min' }
  ];

  const checkAnswer = (questionId: number, answerIndex: number) => {
    setQuizAnswers({ ...quizAnswers, [questionId]: answerIndex });
  };

  return (
    <div className="p-8 space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <GraduationCap className="h-8 w-8 text-primary" />
            Education Portal
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            Case-based teaching, viva prompts, adaptive quizzes from real cases
          </p>
        </div>
        <div className="flex gap-3">
          <Button variant="outline">
            <TrendingUp className="h-4 w-4 mr-2" />
            My Progress
          </Button>
          <Button>
            <Award className="h-4 w-4 mr-2" />
            Achievements
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Student Profile */}
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Student Profile</CardTitle>
              <CardDescription>{studentProfile.level}</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="text-center p-4 bg-primary/5 rounded-lg">
                <p className="text-4xl font-bold text-primary">{studentProfile.completedCases}</p>
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">Cases Completed</p>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="text-center p-3 border rounded-lg">
                  <p className="text-2xl font-bold text-green-600">{studentProfile.accuracy}%</p>
                  <p className="text-xs text-gray-600">Accuracy</p>
                </div>
                <div className="text-center p-3 border rounded-lg">
                  <p className="text-2xl font-bold text-orange-600">{studentProfile.streak}</p>
                  <p className="text-xs text-gray-600">Day Streak</p>
                </div>
              </div>

              <div>
                <p className="text-sm font-semibold mb-2">Earned Badges</p>
                <div className="flex flex-wrap gap-2">
                  {studentProfile.badges.map((badge, idx) => (
                    <Badge key={idx} className="bg-yellow-600">
                      <Award className="h-3 w-3 mr-1" />
                      {badge}
                    </Badge>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Performance by Specialty</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {performanceMetrics.map((metric, idx) => (
                  <div key={idx} className="space-y-1">
                    <div className="flex items-center justify-between text-sm">
                      <span className="font-semibold">{metric.category}</span>
                      <Badge variant="outline">{metric.accuracy}%</Badge>
                    </div>
                    <div className="w-full bg-gray-200 dark:bg-gray-800 rounded-full h-2">
                      <div
                        className="bg-primary h-2 rounded-full"
                        style={{ width: `${metric.accuracy}%` }}
                      />
                    </div>
                    <div className="flex items-center justify-between text-xs text-gray-600">
                      <span>{metric.cases} cases</span>
                      <span>{metric.avgTime}</span>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          <Card className="border-blue-200 dark:border-blue-900">
            <CardHeader>
              <CardTitle className="text-sm flex items-center gap-2">
                <Target className="h-4 w-4 text-blue-600" />
                Learning Goals
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2 text-xs">
                <div className="flex items-center gap-2">
                  <CheckCircle className="h-4 w-4 text-green-600" />
                  <span>Complete 5 cardiology cases this week (5/5)</span>
                </div>
                <div className="flex items-center gap-2">
                  <Clock className="h-4 w-4 text-yellow-600" />
                  <span>Achieve 85% accuracy overall (3/5)</span>
                </div>
                <div className="flex items-center gap-2">
                  <Clock className="h-4 w-4 text-gray-400" />
                  <span>Earn "Rapid Responder" badge (0/10)</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Main Content */}
        <div className="lg:col-span-2 space-y-6">
          <Tabs defaultValue="cases">
            <TabsList className="w-full">
              <TabsTrigger value="cases" className="flex-1">Case Library</TabsTrigger>
              <TabsTrigger value="active" className="flex-1">Active Case</TabsTrigger>
            </TabsList>

            {/* Case Library */}
            <TabsContent value="cases" className="space-y-4 mt-6">
              {caseLibrary.map((case_) => (
                <Card
                  key={case_.id}
                  className="cursor-pointer hover:shadow-lg transition-all"
                  onClick={() => setSelectedCase(case_)}
                >
                  <CardHeader>
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-2">
                          <Badge>{case_.specialty}</Badge>
                          <Badge variant={
                            case_.difficulty === 'Beginner' ? 'default' :
                            case_.difficulty === 'Intermediate' ? 'secondary' :
                            'destructive'
                          }>
                            {case_.difficulty}
                          </Badge>
                          <Badge variant="outline">
                            <Clock className="h-3 w-3 mr-1" />
                            {case_.duration}
                          </Badge>
                        </div>
                        <CardTitle className="text-lg">{case_.title}</CardTitle>
                      </div>
                      {case_.completed ? (
                        <div className="text-right ml-4">
                          <CheckCircle className="h-6 w-6 text-green-600 mb-1" />
                          <p className="text-sm font-semibold text-green-600">{case_.score}%</p>
                        </div>
                      ) : (
                        <Button size="sm" className="ml-4">
                          <Play className="h-4 w-4 mr-1" />
                          Start
                        </Button>
                      )}
                    </div>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm font-semibold mb-2">Learning Objectives:</p>
                    <ul className="text-sm space-y-1">
                      {case_.learningObjectives.map((obj, idx) => (
                        <li key={idx} className="flex items-start gap-2">
                          <span className="text-primary">•</span>
                          <span className="text-gray-700 dark:text-gray-300">{obj}</span>
                        </li>
                      ))}
                    </ul>
                  </CardContent>
                </Card>
              ))}
            </TabsContent>

            {/* Active Case */}
            <TabsContent value="active" className="space-y-6 mt-6">
              <Card>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle>Case Presentation</CardTitle>
                    <Badge>Cardiology • Intermediate</Badge>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="prose dark:prose-invert max-w-none">
                    <p className="text-sm whitespace-pre-wrap">{caseStem.presentation}</p>
                  </div>

                  <div className="mt-6 grid grid-cols-2 gap-4">
                    {caseStem.images.map((img, idx) => (
                      <div key={idx} className="p-4 border rounded-lg">
                        <div className="flex items-center gap-2 mb-2">
                          <Badge variant="outline">{img.type}</Badge>
                        </div>
                        <div className="h-32 bg-gray-100 dark:bg-gray-900 rounded flex items-center justify-center mb-2">
                          <p className="text-gray-500 text-sm">[{img.type} Image]</p>
                        </div>
                        <p className="text-sm text-gray-600 dark:text-gray-400">{img.finding}</p>
                      </div>
                    ))}
                  </div>

                  <div className="mt-6">
                    <p className="text-sm font-semibold mb-3">Initial Laboratory Results:</p>
                    <div className="grid grid-cols-3 gap-3">
                      {caseStem.labs.map((lab, idx) => (
                        <div key={idx} className="p-3 border rounded-lg">
                          <p className="text-xs text-gray-600 mb-1">{lab.test}</p>
                          <p className={`font-semibold ${lab.flag === 'high' ? 'text-red-600' : 'text-green-600'}`}>
                            {lab.value}
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Viva Questions */}
              <Card>
                <CardHeader>
                  <CardTitle>Viva Voce Questions</CardTitle>
                  <CardDescription>Test your clinical reasoning</CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  {vivaQuestions.map((q, idx) => (
                    <div key={q.id} className="space-y-3">
                      <p className="font-semibold">Question {idx + 1}: {q.question}</p>
                      <div className="space-y-2">
                        {q.options.map((option, optIdx) => {
                          const isSelected = quizAnswers[q.id] === optIdx;
                          const isCorrect = optIdx === q.correct;
                          const showResult = quizAnswers[q.id] !== undefined;

                          return (
                            <div
                              key={optIdx}
                              className={`p-3 border-2 rounded-lg cursor-pointer transition-all ${
                                showResult
                                  ? isCorrect
                                    ? 'border-green-500 bg-green-50 dark:bg-green-950'
                                    : isSelected
                                    ? 'border-red-500 bg-red-50 dark:bg-red-950'
                                    : 'border-gray-200 dark:border-gray-800'
                                  : isSelected
                                  ? 'border-primary bg-primary/5'
                                  : 'border-gray-200 dark:border-gray-800 hover:border-primary/50'
                              }`}
                              onClick={() => !showResult && checkAnswer(q.id, optIdx)}
                            >
                              <div className="flex items-center justify-between">
                                <span className="text-sm">{option}</span>
                                {showResult && isCorrect && <CheckCircle className="h-5 w-5 text-green-600" />}
                                {showResult && isSelected && !isCorrect && <XCircle className="h-5 w-5 text-red-600" />}
                              </div>
                            </div>
                          );
                        })}
                      </div>

                      {quizAnswers[q.id] !== undefined && (
                        <div className="p-4 bg-blue-50 dark:bg-blue-950 rounded-lg border border-blue-200 dark:border-blue-900">
                          <div className="flex items-start gap-2">
                            <Brain className="h-5 w-5 text-blue-600 flex-shrink-0 mt-0.5" />
                            <div>
                              <p className="font-semibold text-sm mb-1">Explanation:</p>
                              <p className="text-sm mb-2">{q.explanation}</p>
                              <p className="text-xs text-gray-600 dark:text-gray-400">
                                <BookOpen className="h-3 w-3 inline mr-1" />
                                Reference: {q.reference}
                              </p>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  ))}

                  {Object.keys(quizAnswers).length === vivaQuestions.length && (
                    <div className="p-4 bg-green-50 dark:bg-green-950 rounded-lg border border-green-200 dark:border-green-900 text-center">
                      <Award className="h-8 w-8 text-green-600 mx-auto mb-2" />
                      <p className="font-bold text-lg">Case Complete!</p>
                      <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                        Score: {Math.round((vivaQuestions.filter((q, idx) => quizAnswers[q.id] === q.correct).length / vivaQuestions.length) * 100)}%
                      </p>
                      <Button className="mt-4">
                        Next Case
                      </Button>
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </div>
  );
}
