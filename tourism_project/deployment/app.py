import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

MODEL_REPO = "Sagarikak17/tourism-model"   # <-- change this to your HF username

@st.cache_resource
def load_model():
    path = hf_hub_download(repo_id=MODEL_REPO, filename="best_model.joblib", repo_type="model")
    return joblib.load(path)

model = load_model()

st.title("Wellness Tourism Package — Purchase Predictor")
st.write("Enter customer details to predict the likelihood of purchase.")

age = st.number_input("Age", 18, 100, 35)
typeof_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
city_tier = st.selectbox("City Tier", [1, 2, 3])
duration_of_pitch = st.number_input("Duration of Pitch (minutes)", 1, 60, 10)
occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
gender = st.selectbox("Gender", ["Male", "Female"])
num_persons = st.number_input("Number of Persons Visiting", 1, 10, 2)
num_followups = st.number_input("Number of Followups", 0, 10, 3)
product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
preferred_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
num_trips = st.number_input("Number of Trips per Year", 0, 20, 2)
passport = st.selectbox("Passport", [0, 1])
pitch_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
own_car = st.selectbox("Own Car", [0, 1])
num_children = st.number_input("Number of Children Visiting", 0, 5, 0)
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
monthly_income = st.number_input("Monthly Income", 1000, 100000, 20000)

if st.button("Predict"):
    input_df = pd.DataFrame([{
        "Age": age, "TypeofContact": typeof_contact, "CityTier": city_tier,
        "DurationOfPitch": duration_of_pitch, "Occupation": occupation, "Gender": gender,
        "NumberOfPersonVisiting": num_persons, "NumberOfFollowups": num_followups,
        "ProductPitched": product_pitched, "PreferredPropertyStar": preferred_star,
        "MaritalStatus": marital_status, "NumberOfTrips": num_trips, "Passport": passport,
        "PitchSatisfactionScore": pitch_score, "OwnCar": own_car,
        "NumberOfChildrenVisiting": num_children, "Designation": designation,
        "MonthlyIncome": monthly_income,
    }])
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]
    if pred == 1:
        st.success(f"✅ Likely to purchase the package (probability: {prob:.2%})")
    else:
        st.warning(f"❌ Unlikely to purchase the package (probability: {prob:.2%})")
