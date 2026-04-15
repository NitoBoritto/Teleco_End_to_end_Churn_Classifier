"""
FastTAPI + Streamlit serving application - Production-Ready ML Model Serving
========================================================================

This application provides a complete serving solution for the Telco Customer Churn model
with both programmatic API access and a user-friendly web interface.

Architecture:
- FastAPI: High-performance REST API with automatic OpenAPI documentation
- Pydantic: Data validation and automatic API documentation
"""

from fastapi import FastAPI
from pydantic import BaseModel
from src.serving.inference import predict

app = FastAPI(
    title = 'Teleco Customer Churn Prediction API',
    description = 'ML API for predicting customer churn in telecom industry',
    version = '1.0.0'
)


# Health check
@app.get('/')
def root():
    """
    Health check endpoint for monitoring and load balancer health checks
    
    """
    return {'status': 'ok'}

# Request data schema
# Pydantic model for automatic validation and API documentation
class customerdata(BaseModel):
    """
    Customer data schema for churn prediction.
    
    This schema defines the exact 18 features required for churn prediction.
    All features match the original dataset structure for consistency.
    
    """
    # Demographics
    gender: str                # "Male" or "Female"
    SeniorCitizen: int         # Binary "1" or "0" - is Senior
    Partner: str               # "Yes" or "No" - has partner
    Dependents: str            # "Yes" or "No" - has dependents
    
    # Phone services
    PhoneService: str          # "Yes" or "No"
    MultipleLines: str         # "Yes", "No", or "No phone service"
    
    # Internet services  
    InternetService: str       # "DSL", "Fiber optic", or "No"
    OnlineSecurity: str        # "Yes", "No", or "No internet service"
    OnlineBackup: str          # "Yes", "No", or "No internet service"
    DeviceProtection: str      # "Yes", "No", or "No internet service"
    TechSupport: str           # "Yes", "No", or "No internet service"
    StreamingTV: str           # "Yes", "No", or "No internet service"
    StreamingMovies: str       # "Yes", "No", or "No internet service"
    
    # Account information
    Contract: str              # "Month-to-month", "One year", "Two year"
    PaperlessBilling: str      # "Yes" or "No"
    PaymentMethod: str         # "Electronic check", "Mailed check", etc.
    
    # Numeric features
    tenure: int                # Number of months with company
    MonthlyCharges: float      # Monthly charges in dollars
    TotalCharges: float        # Total charges to date
    
# Main prediction API
@app.post('/predict')
def get_prediction(data: customerdata):
    """
    Main prediction endpoint for customer churn prediction.
    
    This endpoint:
    1. Receives validated customer data via Pydantic model
    2. Calls the inference pipeline to transform features and predict
    3. Returns churn prediction in JSON format
    
    Expected Response:
    - {"prediction": "Likely to churn"} or {"prediction": "Not likely to churn"}
    - {"error": "error_message"} if prediction fails
    """
    try:
        
        input_dict = data.model_dump()
        # Converting Pydantic model to dict and call inference pipeline
        result = predict(input_dict)
        return result
    except Exception as e:
        # Tracing errors
        import traceback
        print('--- Backend Traceback ---')
        traceback.print_exc()
        print('-------------------------')
        # Return error details for debugging (consider logging in production)
        return {"error": str(e)}