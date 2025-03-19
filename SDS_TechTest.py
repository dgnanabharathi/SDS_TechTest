import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt 
import seaborn as sns 
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

# Connect to SQLite database that is stored locally
db_path = "C:\\Users\\deepi\\Downloads\\Concert ai\\SDS_techTest_Feb2025\\SDS_techTest_Oct2024\\techTest.db"
conn = sqlite3.connect(db_path)

# Load all tables into DataFrames
df_patient = pd.DataFrame(pd.read_sql("SELECT * FROM patient", conn))
df_diagnosis = pd.DataFrame(pd.read_sql("SELECT * FROM diagnosis", conn))
df_biomarker = pd.DataFrame(pd.read_sql("SELECT * FROM biomarker", conn))

# Close the database connection
conn.close()

# Convert dates to datetime format, handle errors, and extract only the date part  
df_patient["dob"] = pd.to_datetime(df_patient["dob"], errors="coerce").dt.date
df_patient["dod"] = pd.to_datetime(df_patient["dod"], errors="coerce").dt.date
df_diagnosis["diagnosis_date"] = pd.to_datetime(df_diagnosis["diagnosis_date"], errors="coerce").dt.date
df_biomarker["test_date"] = pd.to_datetime(df_biomarker["test_date"], errors="coerce").dt.date

# Display the first few rows to verify
print(df_patient.head())
print(df_diagnosis.head())
print(df_biomarker.head())


#Merge the dataframes on 'patient_id'
df_merged = df_patient.merge(df_diagnosis, on='patient_id', how='left') \
                    .merge(df_biomarker, on='patient_id', how='left')


print(df_merged.info())

# Filter patients with diagnosis codes starting with "C50" (Breast Cancer)
bc_patients = df_merged[df_merged["diagnosis_code"].str.match(r"^C50.*")]

initial_diagnosis = bc_patients.groupby("patient_id")["diagnosis_date"].min().reset_index()
initial_diagnosis.rename(columns={"diagnosis_date": "initial_diagnosis_date"}, inplace=True)

# Last encounter from diagnosis table
last_diagnosis = bc_patients.groupby("patient_id")["diagnosis_date"].max().reset_index()

# Last encounter from biomarker table
last_biomarker = bc_patients.groupby("patient_id")["test_date"].max().reset_index()

# Merge both tables and take the maximum date
last_encounter = pd.merge(last_diagnosis, last_biomarker, on="patient_id", how="outer")

# Get the latest encounter date from either diagnosis or biomarker
last_encounter["last_encounter_date"] = last_encounter[["diagnosis_date", "test_date"]].max(axis=1)

# Keep only relevant columns
last_encounter = last_encounter[["patient_id", "last_encounter_date"]]

# Merge initial diagnosis and last encounter dates
follow_up_data = pd.merge(initial_diagnosis, last_encounter, on="patient_id", how="left")

# Compute follow-up time in years (No need for .dt)
follow_up_data["follow_up_time_years"] = ((follow_up_data["last_encounter_date"] - follow_up_data["initial_diagnosis_date"]).apply(lambda x: x.days) / 365).round(2)


# Summary statistics
# Remove rows where follow-up time is 0 years
follow_up_data_dropped = follow_up_data[follow_up_data["follow_up_time_years"] > 0]
summary_stats = follow_up_data_dropped["follow_up_time_years"].describe()
print("Summary Statistics for Follow-Up Time (Years):")
print(summary_stats)

# Boxplot for follow-up time distribution
plt.figure(figsize=(8, 4))
sns.boxplot(x=follow_up_data_dropped["follow_up_time_years"])
plt.xlabel("Follow-Up Time (Years)")
plt.title("Boxplot of Follow-Up Time for BC Patients")
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.show()

# Histogram with KDE for distribution
plt.figure(figsize=(10, 5))
sns.histplot(follow_up_data_dropped["follow_up_time_years"], bins=30, kde=True)
plt.xlabel("Follow-Up Time (Years)")
plt.ylabel("Frequency")
plt.title("Distribution of Follow-Up Time for BC Patients")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# Merge BC patients' initial diagnosis date with patient dob
bc_age_data = pd.merge(initial_diagnosis, df_patient[["patient_id", "dob"]], on="patient_id", how="left")

# Calculate age at initial diagnosis
bc_age_data["age_at_diagnosis"] = ((bc_age_data["initial_diagnosis_date"] - bc_age_data["dob"]).apply(lambda x: x.days) / 365).round(2)

# Summary statistics
summary_stats_age = bc_age_data["age_at_diagnosis"].describe()
print("Summary Statistics for Age at Initial Diagnosis:")
print(summary_stats_age)

# Boxplot for age distribution
plt.figure(figsize=(8, 4))
sns.boxplot(x=bc_age_data["age_at_diagnosis"])
plt.xlabel("Age at Initial Diagnosis (Years)")
plt.title("Boxplot of Age at Initial Diagnosis for BC Patients")
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.show()

# Histogram with KDE for age distribution
plt.figure(figsize=(10, 5))
sns.histplot(bc_age_data["age_at_diagnosis"], bins=30, kde=True)
plt.xlabel("Age at Initial Diagnosis (Years)")
plt.ylabel("Frequency")
plt.title("Distribution of Age at Initial Diagnosis for BC Patients")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# Get unique patient IDs for breast cancer patients  
bc_patient_ids = bc_patients["patient_id"].unique() 

# Filter records where the biomarker tested is HER2  
her2_tests = bc_patients[bc_patients["biomarker_name"] == "HER2"]

# Count the number of unique patients who were tested for HER2  
patients_tested_for_her2 = her2_tests["patient_id"].nunique()

# Calculate the intent-to-test rate as the percentage of breast cancer patients tested for HER2  
intent_to_test_rate = (patients_tested_for_her2 / len(bc_patient_ids)) * 100
intent_to_test_rate

# Define valid test results (exclude NA, Unknown, Inconclusive)
valid_results = ["Positive", "Negative"]
her2_valid_tests = her2_tests[her2_tests["test_result"].isin(valid_results)]

# Tested Rate: % of HER2-tested patients with a valid test result
tested_rate = (her2_valid_tests["patient_id"].nunique() / patients_tested_for_her2) * 100

# Negativity Rate: % of HER2-tested patients with a negative result
her2_negative_tests = her2_valid_tests[her2_valid_tests["test_result"] == "Negative"]
negativity_rate = (her2_negative_tests["patient_id"].nunique() / patients_tested_for_her2) * 100

# Print Results
print(f"Intent to Test Rate for HER2: {intent_to_test_rate:.2f}%")
print(f"Tested Rate for HER2: {tested_rate:.2f}%")
print(f"Negativity Rate for HER2: {negativity_rate:.2f}%")


# Step 1: Merge with bc_patients, keeping only the earliest diagnosis per patient
bc_patients = bc_patients.merge(initial_diagnosis, on="patient_id", how="left")

# Keep only the earliest diagnosis per patient  
bc_patients = bc_patients.sort_values("initial_diagnosis_date").groupby("patient_id", as_index=False).first()

# Step 2: Compute age at initial diagnosis
bc_patients["age_at_diagnosis"] = ((bc_patients["initial_diagnosis_date"] - bc_patients["dob"]).apply(lambda x: x.days) / 365).round(2)

# Step 3: Define today's date for censoring
today = datetime.date.today()  

# Step 4: Compute survival time (in years), handling censoring
bc_patients["survival_time"] = ((bc_patients["dod"].fillna(today) - bc_patients["initial_diagnosis_date"]).apply(lambda x: x.days) / 365).round(2)
bc_patients = bc_patients[bc_patients["survival_time"] > 0]  # Keep only positive survival times

# Step 5: Identify whether the event (death) was observed
bc_patients["event_observed"] = bc_patients["dod"].notna() & (bc_patients["dod"] <= today)

# Step 6: Stratify into age groups (<60 and 60+)
bc_patients["age_group"] = np.where(bc_patients["age_at_diagnosis"] < 60, "<60", "60+")

# Step 7: Perform Log-Rank Test (Survival Difference)
group1 = bc_patients[bc_patients["age_group"] == "<60"]
group2 = bc_patients[bc_patients["age_group"] == "60+"]

logrank_result = logrank_test(
    group1["survival_time"],  
    group2["survival_time"],  
    event_observed_A=group1["event_observed"],  
    event_observed_B=group2["event_observed"]  
)

# Print Summary Statistics & Log-Rank Test Results
summary_stats = bc_patients.groupby("age_group")["survival_time"].describe().round(2)
print("Summary Statistics for Survival Time by Age Group:")
print(summary_stats)
print("\nLog-Rank Test P-Value:", logrank_result.p_value)

# Step 8: Plot Kaplan-Meier Survival Curve
kmf1 = KaplanMeierFitter()
kmf2 = KaplanMeierFitter()

plt.figure(figsize=(10, 6))

# Fit & Plot for Age < 60
kmf1.fit(group1["survival_time"], event_observed=group1["event_observed"], label="Age < 60")
kmf1.plot_survival_function(ci_show=False)

# Fit & Plot for Age 60+
kmf2.fit(group2["survival_time"], event_observed=group2["event_observed"], label="Age 60+")
kmf2.plot_survival_function(ci_show=False)

plt.xlabel("Survival Time (Years)")
plt.ylabel("Survival Probability")
plt.title("Kaplan-Meier Survival Curve (Stratified by Age at Diagnosis)")
plt.legend()
plt.grid()
plt.show()