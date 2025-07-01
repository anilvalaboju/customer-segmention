
import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the pre-trained model and scaler
try:
    with open('logistic_regression_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    st.success("Model and scaler loaded successfully.")
except FileNotFoundError:
    st.error("Model or scaler file not found. Please make sure 'logistic_regression_model.pkl' and 'scaler.pkl' are in the same directory.")
    model = None
    scaler = None

st.title('Customer Complaint Prediction')

if model and scaler:
    st.write("Enter the customer's details to predict if they will complain.")

    # Get input features from the user
    income = st.number_input('Income', min_value=0.0, format="%.2f")
    kidhome = st.slider('Number of Kids at Home', 0, 2, 0)
    teenhome = st.slider('Number of Teens at Home', 0, 2, 0)
    recency = st.number_input('Days since last purchase', min_value=0, format="%d")
    mntwines = st.number_input('Amount spent on Wine', min_value=0, format="%d")
    mntfruits = st.number_input('Amount spent on Fruits', min_value=0, format="%d")
    mntmeatproducts = st.number_input('Amount spent on Meat Products', min_value=0, format="%d")
    mntfishproducts = st.number_input('Amount spent on Fish Products', min_value=0, format="%d")
    mntsweetproducts = st.number_input('Amount spent on Sweet Products', min_value=0, format="%d")
    mntgoldprods = st.number_input('Amount spent on Gold Products', min_value=0, format="%d")
    numdealpurchases = st.number_input('Number of purchases with discount', min_value=0, format="%d")
    numwebpurchases = st.number_input('Number of purchases made through company website', min_value=0, format="%d")
    numcatalogpurchases = st.number_input('Number of purchases made using catalogue', min_value=0, format="%d")
    numstorepurchases = st.number_input('Number of purchases made directly in store', min_value=0, format="%d")
    numwebvisitsmonth = st.number_input('Number of visits to company website in the last month', min_value=0, format="%d")
    acceptedcmp3 = st.selectbox('Accepted Campaign 3', [0, 1])
    acceptedcmp4 = st.selectbox('Accepted Campaign 4', [0, 1])
    acceptedcmp5 = st.selectbox('Accepted Campaign 5', [0, 1])
    acceptedcmp1 = st.selectbox('Accepted Campaign 1', [0, 1])
    acceptedcmp2 = st.selectbox('Accepted Campaign 2', [0, 1])
    response = st.selectbox('Accepted last campaign (Response)', [0, 1])
    # Assuming Education and Marital_Status were encoded. Need to get the original labels.
    # For simplicity here, we'll assume the label encoding order was preserved.
    # In a real app, you'd need to load the LabelEncoders or map inputs to encoded values.
    education_mapping = {'Basic': 0, '2n Cycle': 1, 'Graduation': 2, 'Master': 3, 'PhD': 4} # Example mapping based on previous code
    marital_status_mapping = {'Single': 0, 'Together': 1, 'Married': 2, 'Divorced': 3, 'Widow': 4, 'Alone': 5, 'Absurd': 6, 'YOLO': 7} # Example mapping

    education = st.selectbox('Education Level', list(education_mapping.keys()))
    marital_status = st.selectbox('Marital Status', list(marital_status_mapping.keys()))

    # Convert selected labels to encoded values
    education_encoded = education_mapping[education]
    marital_status_encoded = marital_status_mapping[marital_status]

    # Create a DataFrame from user inputs
    input_data = pd.DataFrame({
        'Income': [income],
        'Kidhome': [kidhome],
        'Teenhome': [teenhome],
        'Recency': [recency],
        'MntWines': [mntwines],
        'MntFruits': [mntfruits],
        'MntMeatProducts': [mntmeatproducts],
        'MntFishProducts': [mntfishproducts],
        'MntSweetProducts': [mntsweetproducts],
        'MntGoldProds': [mntgoldprods],
        'NumDealsPurchases': [numdealpurchases],
        'NumWebPurchases': [numwebpurchases],
        'NumCatalogPurchases': [numcatalogpurchases],
        'NumStorePurchases': [numstorepurchases],
        'NumWebVisitsMonth': [numwebvisitsmonth],
        'AcceptedCmp3': [acceptedcmp3],
        'AcceptedCmp4': [acceptedcmp4],
        'AcceptedCmp5': [acceptedcmp5],
        'AcceptedCmp1': [acceptedcmp1],
        'AcceptedCmp2': [acceptedcmp2],
        'Response': [response],
        'Education': [education_encoded],
        'Marital_Status': [marital_status_encoded]
    })

    # Scale the input data
    scaled_input_data = scaler.transform(input_data)

    if st.button('Predict'):
        prediction = model.predict(scaled_input_data)
        if prediction[0] == 1:
            st.warning('Prediction: The customer is likely to complain.')
        else:
            st.success('Prediction: The customer is unlikely to complain.')

else:
    st.warning("Model not loaded. Please ensure model and scaler files are available.")

