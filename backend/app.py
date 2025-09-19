from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_restful import Api, Resource
import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)
api = Api(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthCheck(Resource):
    """Health check endpoint"""
    def get(self):
        return {
            "status": "healthy",
            "service": "Diabetic Retinopathy Detection API",
            "version": "1.0.0"
        }

class DiseaseDetection(Resource):
    """Disease detection endpoint for patient data analysis"""
    def post(self):
        try:
            data = request.get_json()
            
            # Validate required fields
            required_fields = ['age', 'gender', 'symptoms', 'medicalHistory']
            for field in required_fields:
                if field not in data or not data[field]:
                    return {"error": f"Missing required field: {field}"}, 400
            
            # Process the data (placeholder for actual ML model)
            result = self.process_patient_data(data)
            
            return result, 200
            
        except Exception as e:
            logger.error(f"Error in disease detection: {str(e)}")
            return {"error": "Internal server error"}, 500
    
    def process_patient_data(self, data):
        """Process patient data and return detection results"""
        # This is a placeholder - replace with actual ML model
        return {
            "disease": "Diabetic Retinopathy",
            "confidence": 85.5,
            "risk_level": "Medium",
            "recommendations": [
                "Schedule an eye examination with an ophthalmologist",
                "Monitor blood sugar levels regularly",
                "Maintain a healthy diet and exercise routine",
                "Consider annual retinal screening"
            ]
        }

class ScreeningCenter(Resource):
    """Screening center endpoint for health screening tests"""
    def post(self):
        try:
            data = request.get_json()
            
            if 'test_type' not in data:
                return {"error": "Missing test_type field"}, 400
            
            # Process the screening test
            result = self.process_screening_test(data['test_type'])
            
            return result, 200
            
        except Exception as e:
            logger.error(f"Error in screening: {str(e)}")
            return {"error": "Internal server error"}, 500
    
    def process_screening_test(self, test_type):
        """Process screening test and return results"""
        # This is a placeholder - replace with actual ML model
        test_results = {
            "cardiovascular": {
                "test_type": "cardiovascular",
                "status": "completed",
                "risk_score": 65,
                "findings": [
                    "Mild retinal vessel changes detected",
                    "No significant microaneurysms found",
                    "Normal optic disc appearance"
                ],
                "next_steps": [
                    "Follow up in 6 months",
                    "Continue regular monitoring",
                    "Maintain current treatment plan"
                ]
            },
            "diabetes": {
                "test_type": "diabetes",
                "status": "completed", 
                "risk_score": 78,
                "findings": [
                    "Early signs of diabetic retinopathy",
                    "Microaneurysms present in temporal region",
                    "Mild retinal thickening detected"
                ],
                "next_steps": [
                    "Urgent ophthalmologist consultation",
                    "Consider laser treatment",
                    "Strict blood sugar control required"
                ]
            },
            "cancer": {
                "test_type": "cancer",
                "status": "completed",
                "risk_score": 25,
                "findings": [
                    "No signs of retinal tumors",
                    "Normal retinal pigment epithelium",
                    "No suspicious lesions detected"
                ],
                "next_steps": [
                    "Continue routine screening",
                    "No immediate action required",
                    "Annual follow-up recommended"
                ]
            },
            "general": {
                "test_type": "general",
                "status": "completed",
                "risk_score": 45,
                "findings": [
                    "Overall retinal health appears stable",
                    "Minor age-related changes present",
                    "Good vascular integrity"
                ],
                "next_steps": [
                    "Maintain regular eye exams",
                    "Continue healthy lifestyle",
                    "Monitor for any changes"
                ]
            }
        }
        
        return test_results.get(test_type, {
            "test_type": test_type,
            "status": "error",
            "risk_score": 0,
            "findings": ["Invalid test type"],
            "next_steps": ["Please select a valid test type"]
        })

# Add resources to API
api.add_resource(HealthCheck, '/api/health')
api.add_resource(DiseaseDetection, '/api/detect-disease')
api.add_resource(ScreeningCenter, '/api/screening')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    logger.info(f"Starting Diabetic Retinopathy Detection API on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
