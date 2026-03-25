import pandas as pd

df = pd.read_excel("data\\nsclc_ctg-studies.xlsx", engine='openpyxl')
df.dropna(how='all', inplace=True)
df.columns = df.columns.str.strip()
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
df.replace("", pd.NA, inplace=True)
df = df[df['NCT Number'].astype(str).str.strip() != ""]
df.drop_duplicates(subset=['NCT Number'], inplace=True)

df["Enrollment"] = df["Enrollment"].astype(str).str.strip()
df["Enrollment"] = pd.to_numeric(df["Enrollment"], errors='coerce')

date_columns = ['Start Date', 'Completion Date', 'Primary Completion Date', 'Results First Posted']
for col in date_columns:
    df[col] = pd.to_datetime(df[col], errors='coerce')

df["start_year"] = df["Start Date"].dt.year
df["completion_year"] = df["Completion Date"].dt.year
df = df.fillna('')


def create_summary(row):
    summary = str(row["Brief Summary"])

    if len(summary.split()) < 10:
        
        fallback = (
            row["Study Title"] + ", studying "
            + str(row["Conditions"])
            + ". Phase: "
            + str(row["Phases"])
            + ". Current status: "
            + str(row["Study Status"])
            + ". Expected enrollment: "
            + str(row["Enrollment"])
            + "."
        )
        return fallback

    return summary

df["summary_fixed"] = df.apply(create_summary, axis=1)

df["combined_text"] = (
    "NCT Number: " + df['NCT Number'].astype(str) + ". " +
    "Study Title: " + df['Study Title'].astype(str) + ". " +
    "Sponsor: " + df['Sponsor'].astype(str) + ". " +
    "Condition or Disease: " + df['Conditions'].astype(str) + ". " +
    "Intervention or Drug or Treatment: " + df['Interventions'].astype(str) + ". " +
    "Phase: " + df['Phases'].astype(str) + ". " +
    "Study Status: " + df['Study Status'].astype(str) + ". " +
    "Enrollment: " + df['Enrollment'].astype(str) + ". " +
    "Summary: " + df['summary_fixed'].astype(str) + ". "
)

columns_to_keep = [
    "NCT Number",
    "Study Title",
    "Study Status",
    "Conditions",
    "Interventions",
    "Phases",
    "Enrollment",
    "Sex",
    "Age",
    "Sponsor",
    "Locations",
    "Start Date",
    "start_year",
    "Completion Date",
    "completion_year",
    "combined_text"
]

processed_df = df[columns_to_keep]

processed_df.to_csv("data\\processed_trials.csv", index=False)
print("Rows after cleaning:", len(processed_df))