// API service for disease detection system
import { config } from '../config/env'

const API_BASE_URL = config.API_BASE_URL

export interface PatientData {
  age: string
  gender: string
  symptoms: string
  medicalHistory: string
  testResults: string
}

export interface DetectionResult {
  disease: string
  confidence: number
  risk_level: string
  recommendations: string[]
}

export interface ScreeningRequest {
  test_type: string
}

export interface ScreeningResult {
  test_type: string
  status: string
  risk_score: number
  findings: string[]
  next_steps: string[]
}

export class DiseaseDetectionAPI {
  static async detectDisease(data: PatientData): Promise<DetectionResult> {
    const response = await fetch(`${API_BASE_URL}/api/detect-disease`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data)
    })

    if (!response.ok) {
      throw new Error(`API Error: ${response.status} - ${response.statusText}`)
    }

    return response.json()
  }

  static async runScreening(data: ScreeningRequest): Promise<ScreeningResult> {
    const response = await fetch(`${API_BASE_URL}/api/screening`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data)
    })

    if (!response.ok) {
      throw new Error(`API Error: ${response.status} - ${response.statusText}`)
    }

    return response.json()
  }

  static async getHealthStatus(): Promise<any> {
    const response = await fetch(`${API_BASE_URL}/api/health`)
    
    if (!response.ok) {
      throw new Error(`API Error: ${response.status} - ${response.statusText}`)
    }

    return response.json()
  }
}
