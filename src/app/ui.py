import streamlit as st
import numpy as np
import requests

# Page configuration
st.set_page_config(page_title = 'Churn Portal', layout = 'wide')

# Global Style & Background
custom_css = """
    <style>
        /* --- 1. Background & Base Layout (Gemini-Slate) --- */
        [data-testid="stAppViewContainer"] {
            background-color: #131314 !important;
            background-image: radial-gradient(circle at 50% 50%, #1e1e20 0%, #131314 100%) !important;
        }

        [data-testid="stHeader"] {
            background-color: rgba(0,0,0,0) !important;
        }

        h1, h2, h3, label, .stMarkdown {
            color: #e3e3e3 !important;
            font-family: 'Inter', sans-serif;
        }

        .stTitle {
            color: #ffffff !important;
            text-shadow: 0 0 15px rgba(255, 7, 58, 0.6) !important;
            font-weight: 700 !important;
        }

        /* --- 2. The Form Engine Card --- */
        [data-testid="stForm"] {
            background-color: #1e1e20 !important;
            border-radius: 16px !important;
            padding: 35px !important;
            border: 1px solid rgba(255, 7, 58, 0.4) !important;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5) !important;
        }

        /* --- 3. UNIFIED INPUT LOGIC (Categorical + Numerical) --- */
        /* Targets the outer wrapper of all main input types */
        [data-testid="stSelectbox"] > div, 
        [data-testid="stNumberInput"] > div,
        [data-testid="stMultiSelect"] > div {
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }

        /* Unified Hover: The "High-Lift" (-4px) and Glow */
        [data-testid="stSelectbox"]:hover > div, 
        [data-testid="stNumberInput"]:hover > div,
        [data-testid="stMultiSelect"]:hover > div {
            transform: translateY(-4px) !important;
            border-color: #ff073a !important;
            box-shadow: 0 8px 25px rgba(255, 7, 58, 0.3) !important;
        }

        /* Inner Box Styling (Dark Chrome look) */
        div[data-baseweb="select"] > div, 
        div[data-baseweb="input"] > div {
            background-color: #2c2c2e !important;
            border: 1px solid #3c3c3e !important;
            color: white !important;
            border-radius: 10px !important;
        }

        /* Unified Focus: Intense Glow when active */
        [data-testid="stSelectbox"] div[data-baseweb="select"]:focus-within, 
        [data-testid="stNumberInput"] div[data-baseweb="input"]:focus-within {
            border-color: #ff073a !important;
            box-shadow: 0 0 20px rgba(255, 7, 58, 0.5) !important;
            background-color: #242427 !important;
            transform: scale(1.01) !important;
        }

        /* --- 4. Interactive "Analyze" Button --- */
        .stButton>button {
            background-image: linear-gradient(135deg, #ff073a 0%, #dc143c 100%) !important;
            color: white !important;
            border-radius: 10px !important;
            border: none !important;
            box-shadow: 0 4px 15px rgba(255, 7, 58, 0.3) !important;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            font-weight: 800 !important;
            height: 3.8em !important;
            width: 100% !important;
            margin-top: 25px;
        }

        .stButton>button:hover {
            transform: translateY(-4px) scale(1.02) !important;
            box-shadow: 0 12px 25px rgba(255, 7, 58, 0.6) !important;
            background-image: linear-gradient(135deg, #ff2e5b 0%, #ff073a 100%) !important;
        }

        .stButton>button:active {
            transform: translateY(2px) scale(0.97) !important;
            transition: 0.05s !important;
        }

        /* --- 5. Prediction Result Card --- */
        .prediction-card {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(12px);
            border-radius: 16px;
            padding: 24px;
            border: 1px solid rgba(255, 7, 58, 0.5);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.8);
            margin-top: 25px;
        }

        .status-stay { color: #ffffff; text-shadow: 0 0 10px rgba(255, 255, 255, 0.3); font-size: 22px; font-weight: 800; }
        .status-churn { color: #ff073a; text-shadow: 0 0 15px rgba(255, 7, 58, 0.8); font-size: 22px; font-weight: 800; }

        .stProgress > div > div > div > div {
            background-image: linear-gradient(to right, #4a0000, #ff073a, #ffb300) !important;
        }
    </style>
"""

st.markdown(custom_css, unsafe_allow_html = True)

# Main UI
st.title('Telco Churn Predictor Portal')
st.markdown('---')

st.subheader('Churn parameters')

# --- Row 1: Identity (Full Width or Split)
st.subheader("Personal Information")
col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox('Gender', options=['Male', 'Female'])
    senior_selection = st.selectbox('Are They a Senior?', options=['Yes', 'No'])
with col2:
    partner = st.selectbox('Do They Have a Partner?', options=['Yes', 'No'])
    dependents = st.selectbox('Any Dependents?', options=['Yes', 'No'])
    
senior = 0 if senior_selection == 'No' else 1

st.markdown("---")

# --- Row 2: The "Master" Row (Full Width) 
internet_selection = st.selectbox('🌐 Internet Service Type', 
                                 options=['DSL', 'Fiber optic', 'No internet service'],
                                 help="This is the primary driver for service-related features.")

internet = 'No' if internet_selection == 'No internet service' else internet_selection
is_disabled = (internet_selection == 'No internet service')
dep_options = ['No internet service'] if is_disabled else ['Yes', 'No']

st.markdown("---")

# --- Row 3: Dependent Services (4 Columns)
st.subheader("Add-on Services")
c1, c2, c3, c4 = st.columns(4)
with c1:
    security = st.selectbox('Online Security?', options=dep_options, disabled=is_disabled)
    backup = st.selectbox('Online Backup?', options=dep_options, disabled=is_disabled)
with c2:
    device = st.selectbox('Device Protection?', options=dep_options, disabled=is_disabled)
    techsupport = st.selectbox('Tech Support?', options=dep_options, disabled=is_disabled)
with c3:
    tv = st.selectbox('Streaming TV?', options=dep_options, disabled=is_disabled)
    movies = st.selectbox('Streaming Movies?', options=dep_options, disabled=is_disabled)
with c4:
    phone = st.selectbox('Phone Service?', options=['Yes', 'No'])
    lines = st.selectbox('Multiple Lines?', options=['Yes', 'No'])

st.markdown("---")

# --- Row 4: Contract & Financials (Full Width or Split) ---
st.subheader("Billing Details")
col_a, col_b, col_c, col_d = st.columns(4)
with col_a:
    contract = st.selectbox('Contract Type', options=['Month-to-month', 'One year', 'Two year'])
with col_b:
    payment = st.selectbox('Payment Method', options=['Electronic check',
                                                      'Mailed check',
                                                      'Bank transfer (automatic)',
                                                      'Credit card (automatic)'])
with col_c:
    tenure = st.number_input('Tenure (Months)',
                             min_value=0,
                             max_value=120,
                             value=30)

with col_d:
    billing = st.selectbox('Paperless Billing', options = ['Yes', 'No'])

monthly_charges = st.number_input('Monthly Charges ($)',
                                  min_value = 0,
                                  max_value = 500,
                                  value = 0)

total_charges = st.number_input('Total Account Charges ($)',
                                  min_value = 0,
                                  value = 50)

st.markdown("---")

st.caption('Check if your customers are going to churn or not!')
prediction_button = st.button('⚙️ Analyze Churn Risk')

if prediction_button:
    input = {
        "gender": gender,
        "SeniorCitizen": int(senior), # We already mapped this to 1/0
        "Partner": partner,
        "Dependents": dependents,
        "tenure": int(tenure),
        "PhoneService": phone,
        "MultipleLines": lines,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": device,
        "TechSupport": techsupport,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": billing,
        "PaymentMethod": payment,
        "MonthlyCharges": float(monthly_charges), # Ensure these exist in your UI
        "TotalCharges": float(total_charges)
    }
    
    try:
        # Send the POST request
        response = requests.post(
            'http://127.0.0.1:8000/predict',
            json = input
        )
        
        # Output results
        if response.status_code == 200:
            result = response.json()
            
            if "error" in result:
                st.error(f"❌ **Model Error:** {result['error']}")
            else:
                prediction = result['prediction']
                probability = result['probability']
                
                st.markdown("---")
                
                if prediction == 'Likely to churn':
                    status_class = "status-churn"
                    display_label = "⚠️ High Churn Risk Detected"
                    confidence = probability
                    bar_color = "#ff4757"
                else:
                    status_class = "status-stay"
                    display_label = "✅ Won't Churn"
                    confidence = 100 - probability
                    bar_color = "#00d2ff"

                # Custom HTML Card
                st.markdown(f"""
                    <div class="prediction-card">
                        <div class="{status_class}">{display_label}</div>
                        <p style="color: #a4b0be; margin-bottom: 5px;">Model Confidence Score</p>
                        <h2 style="color: white; margin-top: 0px;">{confidence:.1f}%</h2>
                    </div>
                """, unsafe_allow_html=True)
                
                # Progress bar with matching color
                st.progress(confidence / 100)
                st.caption(f"Strategy: {'Immediate retention offer recommended' if prediction == 'Likely to churn' else 'Maintain current service level'}")
        else:
            st.error(f'Error from backend: {response.status_code}')
    except requests.exceptions.ConnectionError:
        st.error('Could not connect to FastAPI. Check server connection.')