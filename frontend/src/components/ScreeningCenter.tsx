import React, { useState } from "react"
import { Card } from "./ui/card"
import { Button } from "./ui/Button"
import { Brain, CheckCircle, AlertTriangle, Loader } from "lucide-react"
import { DiseaseDetectionAPI, type ScreeningResult } from "../services/api"

export const ScreeningCenter: React.FC = () => {
  const [selectedTest, setSelectedTest] = useState("")
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [result, setResult] = useState<ScreeningResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  const testTypes = [
    { id: "cardiovascular", name: "Cardiovascular Risk Assessment", description: "Heart disease and stroke risk analysis" },
    { id: "diabetes", name: "Diabetes Screening", description: "Blood sugar and insulin resistance analysis" },
    { id: "cancer", name: "Cancer Risk Assessment", description: "Oncological risk evaluation" },
    { id: "general", name: "General Health Screening", description: "Comprehensive health assessment" }
  ]

  const handleScreening = async () => {
    if (!selectedTest) return

    setIsAnalyzing(true)
    setError(null)
    setResult(null)

    try {
      const data = await DiseaseDetectionAPI.runScreening({ test_type: selectedTest })
      setResult(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
    } finally {
      setIsAnalyzing(false)
    }
  }

  const getRiskColor = (score: number) => {
    if (score >= 80) return 'text-destructive'
    if (score >= 50) return 'text-secondary'
    return 'text-chart-4'
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-3xl font-bold text-foreground mb-2 flex items-center justify-center gap-2">
          <Brain className="h-8 w-8 text-primary" />
          AI Screening Center
        </h1>
        <p className="text-muted-foreground">
          Automated health screening and risk assessment
        </p>
      </div>

      {/* Test Selection */}
      <Card className="p-6">
        <h2 className="text-xl font-semibold mb-4">Select Screening Test</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {testTypes.map((test) => (
            <div
              key={test.id}
              className={`p-4 border rounded-lg cursor-pointer transition-colors ${
                selectedTest === test.id
                  ? 'border-primary bg-primary/5'
                  : 'border-border hover:border-primary/50'
              }`}
              onClick={() => setSelectedTest(test.id)}
            >
              <h3 className="font-medium">{test.name}</h3>
              <p className="text-sm text-muted-foreground mt-1">{test.description}</p>
            </div>
          ))}
        </div>
        
        <div className="mt-6 flex justify-center">
          <Button
            onClick={handleScreening}
            disabled={isAnalyzing || !selectedTest}
            className="px-8 py-3"
          >
            {isAnalyzing ? (
              <>
                <Loader className="mr-2 h-4 w-4 animate-spin" />
                Running Screening...
              </>
            ) : (
              <>
                <Brain className="mr-2 h-4 w-4" />
                Start Screening
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
            <h3 className="font-semibold">Screening Error</h3>
          </div>
          <p className="mt-2 text-sm">{error}</p>
        </Card>
      )}

      {result && (
        <Card className="p-6">
          <div className="flex items-center gap-2 mb-4">
            <CheckCircle className="h-5 w-5 text-chart-4" />
            <h3 className="text-xl font-semibold">Screening Results</h3>
          </div>
          
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="p-4 bg-muted rounded-lg">
                <h4 className="font-medium text-sm text-muted-foreground">Test Type</h4>
                <p className="text-lg font-semibold capitalize">{result.test_type.replace('_', ' ')}</p>
              </div>
              <div className="p-4 bg-muted rounded-lg">
                <h4 className="font-medium text-sm text-muted-foreground">Status</h4>
                <p className="text-lg font-semibold capitalize">{result.status}</p>
              </div>
              <div className="p-4 bg-muted rounded-lg">
                <h4 className="font-medium text-sm text-muted-foreground">Risk Score</h4>
                <p className={`text-lg font-semibold ${getRiskColor(result.risk_score)}`}>
                  {result.risk_score}%
                </p>
              </div>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h4 className="font-medium mb-2">Key Findings</h4>
                <ul className="space-y-1">
                  {result.findings.map((finding, index) => (
                    <li key={index} className="text-sm text-muted-foreground flex items-start gap-2">
                      <span className="text-primary mt-1">•</span>
                      {finding}
                    </li>
                  ))}
                </ul>
              </div>
              
              <div>
                <h4 className="font-medium mb-2">Recommended Next Steps</h4>
                <ul className="space-y-1">
                  {result.next_steps.map((step, index) => (
                    <li key={index} className="text-sm text-muted-foreground flex items-start gap-2">
                      <span className="text-chart-4 mt-1">•</span>
                      {step}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        </Card>
      )}
    </div>
  )
}