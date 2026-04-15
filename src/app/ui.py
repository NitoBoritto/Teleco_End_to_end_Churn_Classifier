import streamlit as st
import numpy as np
import requests

# Page configuration
st.set_page_config(page_title = 'Churn Portal', layout = 'wide')

# Global Style & Background
custom_css = """
    <style>
        /* Main Container - Mid-Tone Matte Slate */
        [data-testid="stAppViewContainer"] {
            background-color: #576574 !important;
        }

        /* Top Header Area - Invisible */
        [data-testid="stHeader"] {
            background-color: rgba(0,0,0,0) !important;
        }

        /* The Form "Card" - Light Steel Blue/Grey */
        [data-testid="stForm"] {
            background-color: #f1f2f6 !important;
            border-radius: 12px !important;
            padding: 35px !important;
            border: 2px solid #2f3542 !important;
            box-shadow: 0 10px 20px rgba(0,0,0,0.3) !important;
        }

        /* Sidebar - Deep Gunmetal */
        [data-testid="stSidebar"] {
            background-color: #2f3542 !important;
            border-right: 1px solid #1e2124;
        }

        /* Typography - Deep Navy for readability on light cards */
        h1, h2, h3, label {
            color: #2f3542 !important;
            font-family: 'Inter', 'Segoe UI', sans-serif;
            font-weight: 600 !important;
        }

        /* Main Title override (to show on the dark background) */
        .stTitle {
            color: #ffffff !important;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }

        /* Button - Solid Navy/Slate */
        .stButton>button {
            background-color: #2f3542 !important;
            color: #ffffff !important;
            border-radius: 6px !important;
            border: none !important;
            height: 3.5em !important;
            width: 100% !important;
            font-weight: bold !important;
            transition: 0.3s;
        }

        .stButton>button:hover {
            background-color: #576574 !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.4) !important;
        }
    </style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Main UI
st.title('Telco Churn Predictor Portal')
st.markdown('---')


"""
Selection boxes and numerical boxes to use as input data for prediction

"""

st.subheader('Churn parameters')
st.caption('Check if your customers are going to churn or not!')


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

monthly_charges = st.number_input('Monthly Charges',
                                  min_value = 0,
                                  max_value = 200,
                                  value = 0)

total_charges = st.number_input('Total Account Charges',
                                  min_value = 0,
                                  value = 50)

st.markdown("---")

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
            
            if 'error' in result:
                st.error(f"❌ **Model Error:** {result['error']}")
            else:
                prediction = result['prediction']
                probability = result['probability']
                
                st.markdown('### Analysis Result')
                if prediction == 'Likely to churn':
                    st.error(f'**{prediction}**')
                    st.write(f'The model detected high-risk patterns for this customer with **{probability:.1f}%** confidence')
                    st.progress(probability / 100)
                else:
                    stay_conf = 100 - probability
                    st.success(f'**{prediction}**')
                    st.write(f'The model is **{stay_conf:.1f}%** confident this customer will remain')
                    st.progress(stay_conf / 100)
        else:
            st.error(f'Error from backend: {response.status_code}')
    except requests.exceptions.ConnectionError:
        st.error('Could not connect to FastAPI. Check server connection.')