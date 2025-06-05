import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# For summarization
from transformers import pipeline

# Load BART summarizer
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Read the meeting data
def read_data(filepath):
    df = pd.read_csv(filepath)
    df['Join Time'] = pd.to_datetime(df['Join Time'])
    df['Leave Time'] = pd.to_datetime(df['Leave Time'])
    df['Duration'] = pd.to_timedelta(df['Duration'])
    df['Attendance Duration'] = df['Leave Time'] - df['Join Time']
    df['Presence %'] = (df['Attendance Duration'] / df['Duration']) * 100
    return df

# Plot presence distribution
def plot_presence(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Presence %'], bins=10, kde=True)
    plt.title('Attendance Presence Distribution')
    plt.xlabel('Presence %')
    plt.ylabel('Number of Participants')
    plt.tight_layout()
    plt.savefig("presence_distribution.png")
    plt.close()

# Identify major time gaps
def compute_gaps(df):
    df_sorted = df.sort_values(by='Join Time')
    df_sorted['Gap'] = df_sorted['Join Time'].diff().fillna(pd.Timedelta(seconds=0))
    return df_sorted[['Full Name', 'Join Time', 'Gap']]

# Dummy summary example (you'll replace this with actual transcript text if available)
def generate_summary(text):
    if len(text.split()) < 50:
        return "Text too short for summarization."
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# Generate engagement metrics
def engagement_metrics(df):
    metrics = {
        "Average Presence %": df['Presence %'].mean(),
        "Max Presence %": df['Presence %'].max(),
        "Min Presence %": df['Presence %'].min(),
        "Total Participants": df['Full Name'].nunique()
    }
    return metrics

if __name__ == "__main__":
    # Path to your CSV file
    filepath = "oopat.csv"

    print("Reading data...")
    df = read_data(filepath)

    print("Computing engagement metrics...")
    metrics = engagement_metrics(df)
    for key, value in metrics.items():
        print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")

    print("Plotting presence distribution...")
    plot_presence(df)

    print("Calculating join time gaps...")
    gaps_df = compute_gaps(df)
    print(gaps_df.head())

    # Dummy long text to demonstrate summarization
    dummy_text = "Today we discussed the upcoming roadmap for Q3, including the migration to the new cloud system. All teams shared updates and committed to weekly progress tracking..." * 5
    print("Generating summary of meeting...")
    print(generate_summary(dummy_text))
