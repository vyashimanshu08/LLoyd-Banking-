import streamlit as st
import joblib
import pandas as pd

# ── Load Model ────────────────────────────────────────────────────────────────
model = joblib.load('churn_model.pkl')

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔮",
    layout="centered"
)

st.title("🔮 Customer Churn Predictor")
st.markdown("Fill in the customer details below to predict whether they will churn.")
st.divider()

# ── Personal Information ──────────────────────────────────────────────────────
st.subheader("👤 Personal Information")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    gender = st.selectbox("Gender", ["M", "F"])
    marital_status = st.selectbox(
        "Marital Status",
        ["Single", "Married", "Divorced", "Widowed"]
    )

with col2:
    income_level = st.selectbox("Income Level", ["Low", "Medium", "High"])
    service_usage = st.selectbox(
        "Service Usage",
        ["Website", "Mobile App", "Online Banking"]
    )
    login_frequency = st.number_input(
        "Login Frequency (per month)", min_value=0, max_value=200, value=10
    )

st.divider()

# ── Transaction Information ───────────────────────────────────────────────────
st.subheader("🛍️ Transaction Information")
col3, col4 = st.columns(2)

with col3:
    product_category = st.selectbox(
        "Product Category",
        ["Electronics", "Clothing", "Furniture", "Groceries", "Books"]
    )
    amount_spent = st.number_input(
        "Amount Spent ($)", min_value=0.0, max_value=10000.0, value=200.0, step=10.0
    )

with col4:
    interaction_type = st.selectbox(
        "Interaction Type",
        ["Inquiry", "Complaint", "Feedback", "NoInteraction"]
    )
    resolution_status = st.selectbox(
        "Resolution Status",
        ["Resolved", "Unresolved", "NoInteraction"]
    )

st.divider()

# ── Interaction Information ───────────────────────────────────────────────────
st.subheader("📅 Interaction Information")

days_since_last_interaction = st.number_input(
    "Days Since Last Interaction",
    min_value=0,
    max_value=1000,
    value=30
)

st.divider()

# ── Predict Button ────────────────────────────────────────────────────────────
if st.button("🔍 Predict Churn", use_container_width=True):

    input_data = pd.DataFrame([{
        'Age'                     : age,
        'Gender'                  : gender,
        'MaritalStatus'           : marital_status,
        'IncomeLevel'             : income_level,
        'LoginFrequency'          : login_frequency,
        'ServiceUsage'            : service_usage,
        'AmountSpent'             : amount_spent,
        'ProductCategory'         : product_category,
        'InteractionType'         : interaction_type,
        'ResolutionStatus'        : resolution_status,
        'DaysSinceLastInteraction': days_since_last_interaction,
    }])

    prediction  = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.divider()
    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.error("⚠️ This customer is **likely to CHURN**")
        st.metric(label="Churn Probability", value=f"{probability:.2%}")
        st.markdown("""
        **Suggested Actions:**
        - 📞 Reach out with a retention offer
        - 🎁 Provide a personalized discount
        - 🔧 Address any unresolved complaints
        """)
    else:
        st.success("✅ This customer is **likely to STAY**")
        st.metric(label="Churn Probability", value=f"{probability:.2%}")
        st.markdown("""
        **Suggested Actions:**
        - 🌟 Continue engagement with loyalty rewards
        - 📧 Send personalized product recommendations
        """)

    # Show input summary
    with st.expander("📋 View Input Summary"):
        st.dataframe(input_data.T.rename(columns={0: 'Value'}))