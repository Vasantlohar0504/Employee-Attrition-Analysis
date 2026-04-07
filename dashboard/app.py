import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="Employee Attrition Analytics",
    layout="wide",
    page_icon="📊"
)

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv")

df = load_data()

# -------------------------------------------------
# SIDEBAR FILTERS
# -------------------------------------------------

st.sidebar.title("Filter Employees")
st.sidebar.divider()

department = st.sidebar.multiselect(
    "Department",
    options=sorted(df["Department"].unique()),
    placeholder="Select Department"
)

gender = st.sidebar.multiselect(
    "Gender",
    options=sorted(df["Gender"].unique()),
    placeholder="Select Gender"
)

job_role = st.sidebar.multiselect(
    "Job Role",
    options=sorted(df["JobRole"].unique()),
    placeholder="Select Job Role"
)

education = st.sidebar.multiselect(
    "Education Field",
    options=sorted(df["EducationField"].unique()),
    placeholder="Select Education Field"
)

overtime = st.sidebar.multiselect(
    "OverTime",
    options=sorted(df["OverTime"].unique()),
    placeholder="Select OverTime"
)

filtered_df = df.copy()

if department:
    filtered_df = filtered_df[filtered_df["Department"].isin(department)]

if gender:
    filtered_df = filtered_df[filtered_df["Gender"].isin(gender)]

if job_role:
    filtered_df = filtered_df[filtered_df["JobRole"].isin(job_role)]

if education:
    filtered_df = filtered_df[filtered_df["EducationField"].isin(education)]

if overtime:
    filtered_df = filtered_df[filtered_df["OverTime"].isin(overtime)]

df = filtered_df

# -------------------------------------------------
# TITLE
# -------------------------------------------------

st.title("📊 Employee Attrition Analytics Dashboard")
st.markdown("Analyze workforce attrition trends and predict employee churn.")

# -------------------------------------------------
# KPI CARDS
# -------------------------------------------------

total_emp = df.shape[0]

if total_emp == 0:
    attrition_rate = 0
    avg_salary = 0
    avg_years = 0
    st.warning("No data available for selected filters.")
else:
    attrition_rate = round(
        df[df["Attrition"] == "Yes"].shape[0] / total_emp * 100,2
    )
    avg_salary = f"${int(df['MonthlyIncome'].mean()):,}"
    avg_years = round(df["YearsAtCompany"].mean(),1)

st.markdown("""
<style>
.kpi-card{
    background-color:#ffffff;
    padding:20px;
    border-radius:12px;
    box-shadow:0px 4px 10px rgba(0,0,0,0.1);
    text-align:center;
    border-left:6px solid #4CAF50;
}
.kpi-title{
    font-size:16px;
    color:gray;
}
.kpi-value{
    font-size:30px;
    font-weight:bold;
    color:#1f77b4;
}
</style>
""", unsafe_allow_html=True)

col1,col2,col3,col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">👥 Total Employees</div>
        <div class="kpi-value">{total_emp}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">⚠ Attrition Rate</div>
        <div class="kpi-value">{attrition_rate}%</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">💰 Average Monthly Income</div>
        <div class="kpi-value">{avg_salary}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">📅 Avg Years at Company</div>
        <div class="kpi-value">{avg_years}</div>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------
# ATTRITION DISTRIBUTION
# -------------------------------------------------

col1,col2 = st.columns(2)

fig = px.pie(
    df,
    names="Attrition",
    title="Attrition Distribution",
    color_discrete_sequence=px.colors.qualitative.Set3
)

col1.plotly_chart(fig, width="stretch")

dept_attrition = df.groupby("Department")["Attrition"].value_counts().unstack()

fig = px.bar(
    dept_attrition,
    title="Attrition by Department",
    color_discrete_sequence=px.colors.qualitative.Bold
)

col2.plotly_chart(fig, width="stretch")

# -------------------------------------------------
# JOB ROLE ATTRITION
# -------------------------------------------------

fig = px.histogram(
    df,
    x="JobRole",
    color="Attrition",
    title="Attrition by Job Role",
    color_discrete_sequence=px.colors.qualitative.Pastel
)

st.plotly_chart(fig, width="stretch")

# -------------------------------------------------
# AGE DISTRIBUTION
# -------------------------------------------------

fig = px.histogram(
    df,
    x="Age",
    color="Attrition",
    title="Age Distribution vs Attrition",
    color_discrete_sequence=px.colors.qualitative.Set2
)

st.plotly_chart(fig, width="stretch")

# -------------------------------------------------
# SALARY VS ATTRITION
# -------------------------------------------------

fig = px.box(
    df,
    x="Attrition",
    y="MonthlyIncome",
    title="Salary vs Attrition",
    color="Attrition"
)

st.plotly_chart(fig, width="stretch")

# -------------------------------------------------
# WORK LIFE BALANCE
# -------------------------------------------------

fig = px.histogram(
    df,
    x="WorkLifeBalance",
    color="Attrition",
    title="Work Life Balance Impact",
    color_discrete_sequence=px.colors.qualitative.Safe
)

st.plotly_chart(fig, width="stretch")

# -------------------------------------------------
# OVERTIME IMPACT
# -------------------------------------------------

fig = px.histogram(
    df,
    x="OverTime",
    color="Attrition",
    title="Overtime Impact",
    color_discrete_sequence=px.colors.qualitative.Prism
)

st.plotly_chart(fig, width="stretch")

# -------------------------------------------------
# EDUCATION FIELD
# -------------------------------------------------

fig = px.histogram(
    df,
    x="EducationField",
    color="Attrition",
    title="Education Field vs Attrition",
    color_discrete_sequence=px.colors.qualitative.Dark2
)

st.plotly_chart(fig, width="stretch")

# -------------------------------------------------
# CORRELATION HEATMAP
# -------------------------------------------------

st.subheader("Top Feature Correlation with Attrition")

corr_df = df.copy()

le = LabelEncoder()

for col in corr_df.select_dtypes(include="object").columns:
    corr_df[col] = le.fit_transform(corr_df[col])

corr_matrix = corr_df.corr()

attr_corr = corr_matrix["Attrition"].sort_values(ascending=False).head(10)

fig,ax = plt.subplots(figsize=(6,4))

sns.heatmap(
    attr_corr.to_frame(),
    annot=True,
    cmap="RdYlBu",
    linewidths=0.5,
    ax=ax
)

st.pyplot(fig)

# -------------------------------------------------
# TENURE ANALYSIS
# -------------------------------------------------

fig = px.histogram(
    df,
    x="YearsAtCompany",
    color="Attrition",
    title="Years at Company vs Attrition",
    color_discrete_sequence=px.colors.qualitative.Set1
)

st.plotly_chart(fig, width="stretch")

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------

model = joblib.load("models/attrition_prediction_model.pkl")

# -------------------------------------------------
# ATTRITION PREDICTOR
# -------------------------------------------------

st.header("🤖 Attrition Risk Predictor")

col1,col2,col3 = st.columns(3)

age = col1.slider("Age",18,60,30)

income = col2.number_input("Monthly Income",1000,20000,5000)

years = col3.slider("Years at Company",0,40,5)

overtime_val = st.selectbox("OverTime",["Yes","No"])

overtime_val = 1 if overtime_val=="Yes" else 0

input_df = pd.DataFrame({
    "Age":[age],
    "MonthlyIncome":[income],
    "YearsAtCompany":[years],
    "OverTime":[overtime_val]
})

if st.button("Predict Attrition Risk"):

    prob = model.predict_proba(input_df)[0][1]

    risk = round(prob*100,2)

    st.metric("Attrition Risk Score",f"{risk}%")

    if risk>70:
        st.error("High Attrition Risk")
    elif risk>40:
        st.warning("Medium Attrition Risk")
    else:
        st.success("Low Attrition Risk")

# -------------------------------------------------
# FEATURE IMPORTANCE
# -------------------------------------------------

st.header("📈 Top Factors Driving Attrition")

encoded_df = df.copy()

encoded_df["Attrition"] = encoded_df["Attrition"].map({"Yes":1,"No":0})

for col in encoded_df.select_dtypes(include="object").columns:
    encoded_df[col] = le.fit_transform(encoded_df[col])

X = encoded_df.drop("Attrition",axis=1)

importances = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature":X.columns,
    "Importance":importances
}).sort_values(by="Importance",ascending=False)

fig = px.bar(
    importance_df.head(10),
    x="Importance",
    y="Feature",
    orientation="h",
    color="Importance",
    color_continuous_scale="viridis"
)

st.plotly_chart(fig, width="stretch")

# -------------------------------------------------
# HIGH RISK EMPLOYEES
# -------------------------------------------------

st.header("⚠ High Risk Employees")

risk_scores = model.predict_proba(X)[:,1]

encoded_df["RiskScore"] = risk_scores

high_risk = encoded_df.sort_values(
    by="RiskScore",
    ascending=False
).head(10)

st.dataframe(high_risk)

# -------------------------------------------------
# DOWNLOAD REPORT
# -------------------------------------------------

st.download_button(
    "⬇ Download Attrition Dataset",
    df.to_csv(index=False),
    "attrition_report.csv",
    "text/csv"
)

# -------------------------------------------------
# HR INSIGHTS
# -------------------------------------------------

st.header("💡 Key HR Insights")

st.markdown("""
• Employees working **overtime** show higher attrition rates.

• Employees with **lower salaries** are more likely to leave.

• Most attrition occurs within the **first 3 years**.

• **Sales department** has relatively higher turnover.

• Employees with **low work-life balance** leave more frequently.
""")