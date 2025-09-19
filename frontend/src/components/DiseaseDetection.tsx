import React, { useState } from "react"
import { Card } from "./ui/card"
import { Button } from "./ui/Button"
import { Input } from "./ui/Input"
import { Brain, AlertTriangle, CheckCircle, Loader } from "lucide-react"
import { DiseaseDetectionAPI, type PatientData, type DetectionResult } from "../services/api"

export const DiseaseDetection: React.FC = () => {
  const [patientData, setPatientData] = useState<PatientData>({
    age: "",
    gender: "",
    symptoms: "",
    medicalHistory: "",
    testResults: ""
  })
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [result, setResult] = useState<DetectionResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleInputChange = (field: string, value: string) => {
    setPatientData(prev => ({ ...prev, [field]: value }))
  }

  const handleDetection = async () => {
    setIsAnalyzing(true)
    setError(null)
    setResult(null)

    try {
      const data = await DiseaseDetectionAPI.detectDisease(patientData)
      setResult(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
    } finally {
      setIsAnalyzing(false)
    }
  }

  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel.toLowerCase()) {
      case 'high': return 'text-destructive'
      case 'medium': return 'text-secondary'
      case 'low': return 'text-chart-4'
      default: return 'text-muted-foreground'
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-3xl font-bold text-foreground mb-2 flex items-center justify-center gap-2">
          <Brain className="h-8 w-8 text-primary" />
          Early Disease Detection
        </h1>
        <p className="text-muted-foreground">
          AI-powered analysis for early disease detection and risk assessment
        </p>
      </div>

      {/* Input Form */}
      <Card className="p-6">
        <h2 className="text-xl font-semibold mb-4">Patient Information</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium mb-2">Age</label>
            <Input
              type="number"
              placeholder="Enter age"
              value={patientData.age}
              onChange={(e) => handleInputChange('age', e.target.value)}
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-2">Gender</label>
            <select
              className="w-full px-3 py-2 border border-border rounded-md bg-background"
              value={patientData.gender}
              onChange={(e) => handleInputChange('gender', e.target.value)}
            >
              <option value="">Select gender</option>
              <option value="male">Male</option>
              <option value="female">Female</option>
              <option value="other">Other</option>
            </select>
          </div>
          <div className="md:col-span-2">
            <label className="block text-sm font-medium mb-2">Symptoms</label>
            <textarea
              className="w-full px-3 py-2 border border-border rounded-md bg-background min-h-[100px]"
              placeholder="Describe current symptoms..."
              value={patientData.symptoms}
              onChange={(e) => handleInputChange('symptoms', e.target.value)}
            />
          </div>
          <div className="md:col-span-2">
            <label className="block text-sm font-medium mb-2">Medical History</label>
            <textarea
              className="w-full px-3 py-2 border border-border rounded-md bg-background min-h-[100px]"
              placeholder="Previous medical conditions, family history..."
              value={patientData.medicalHistory}
              onChange={(e) => handleInputChange('medicalHistory', e.target.value)}
            />
          </div>
          <div className="md:col-span-2">
            <label className="block text-sm font-medium mb-2">Test Results (Optional)</label>
            <textarea
              className="w-full px-3 py-2 border border-border rounded-md bg-background min-h-[100px]"
              placeholder="Lab results, imaging reports, etc..."
              value={patientData.testResults}
              onChange={(e) => handleInputChange('testResults', e.target.value)}
            />
          </div>
        </div>
        
        <div className="mt-6 flex justify-center">
          <Button
            onClick={handleDetection}
            disabled={isAnalyzing || !patientData.age || !patientData.gender || !patientData.symptoms}
            className="px-8 py-3"
          >
            {isAnalyzing ? (
              <>
                <Loader className="mr-2 h-4 w-4 animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                <Brain className="mr-2 h-4 w-4" />
                Analyze for Disease Risk
              </>
            )}
          </Button>
        </div>
      </Card>

      {/* Results */}
      {error && (
        <Card className="p-6 border-destructive">
          <div className="flex items-center gap-2 text-destructive">
            <AlertTriangle className="h-5 w-5" />
            <h3 className="font-semibold">Analysis Error</h3>
          </div>
          <p className="mt-2 text-sm">{error}</p>
        </Card>
      )}

      {result && (
        <Card className="p-6">
          <div className="flex items-center gap-2 mb-4">
            <CheckCircle className="h-5 w-5 text-chart-4" />
            <h3 className="text-xl font-semibold">Analysis Results</h3>
          </div>
          
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="p-4 bg-muted rounded-lg">
                <h4 className="font-medium text-sm text-muted-foreground">Detected Disease</h4>
                <p className="text-lg font-semibold">{result.disease}</p>
              </div>
              <div className="p-4 bg-muted rounded-lg">
                <h4 className="font-medium text-sm text-muted-foreground">Confidence</h4>
                <p className="text-lg font-semibold">{result.confidence}%</p>
              </div>
              <div className="p-4 bg-muted rounded-lg">
                <h4 className="font-medium text-sm text-muted-foreground">Risk Level</h4>
                <p className={`text-lg font-semibold ${getRiskColor(result.risk_level)}`}>
                  {result.risk_level}
                </p>
              </div>
            </div>
            
            <div>
              <h4 className="font-medium mb-2">Recommendations</h4>
              <ul className="space-y-1">
                {result.recommendations.map((rec, index) => (
                  <li key={index} className="text-sm text-muted-foreground flex items-start gap-2">
                    <span className="text-primary mt-1">•</span>
                    {rec}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </Card>
      )}
    </div>
  )
}
