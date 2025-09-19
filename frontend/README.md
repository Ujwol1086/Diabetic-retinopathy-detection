# Early Disease Detection Frontend

A lightweight React frontend for the Early Disease Detection System that integrates with backend APIs.

## Features

- **Disease Detection**: AI-powered analysis of patient data for early disease detection
- **Screening Center**: Automated health screening and risk assessment
- **API Integration**: Clean, typed API service for backend communication

## API Endpoints

The frontend expects the following backend API endpoints:

### Disease Detection
- **POST** `/api/detect-disease`
  - **Body**: `PatientData`
  - **Response**: `DetectionResult`

### Screening
- **POST** `/api/screening`
  - **Body**: `ScreeningRequest`
  - **Response**: `ScreeningResult`

### Health Check
- **GET** `/api/health`
  - **Response**: Health status

## Data Types

### PatientData
```typescript
{
  age: string
  gender: string
  symptoms: string
  medicalHistory: string
  testResults: string
}
```

### DetectionResult
```typescript
{
  disease: string
  confidence: number
  risk_level: string
  recommendations: string[]
}
```

### ScreeningRequest
```typescript
{
  test_type: string
}
```

### ScreeningResult
```typescript
{
  test_type: string
  status: string
  risk_score: number
  findings: string[]
  next_steps: string[]
}
```

## Environment Variables

Set the backend API URL:
```bash
VITE_API_URL=http://localhost:5000
```

## Development

```bash
npm install
npm run dev
```

## Build

```bash
npm run build
```

## Project Structure

```
src/
├── components/
│   ├── DiseaseDetection.tsx    # Main disease detection interface
│   ├── ScreeningCenter.tsx     # Health screening interface
│   ├── Navigation.tsx          # Simple navigation
│   └── ui/                     # Reusable UI components
├── services/
│   └── api.ts                  # API service layer
├── App.tsx                     # Main application
└── main.tsx                    # Application entry point
```