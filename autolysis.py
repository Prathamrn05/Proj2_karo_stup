import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import json
import openai
from pathlib import Path
import math

# Ensure seaborn styling
sns.set(style="whitegrid")

# Check command line argument
if len(sys.argv) != 2:
    print("Usage: uv run autolysis.py media.csv")
    sys.exit(1)

filename = "C:/Users/prath/Desktop/karo stup/proj2/data/media.csv"

if not os.path.exists(filename):
    print(f"Error: File '{filename}' not found.")
    sys.exit(1)

# Setup OpenAI client with AI Proxy
if "AIPROXY_TOKEN" not in os.environ:
    print("Error: AIPROXY_TOKEN not set in environment.")
    sys.exit(1)

openai.api_base = "https://api.aiproxy.io/v1"
openai.api_key = os.environ["AIPROXY_TOKEN"]

# Load dataset
df = pd.read_csv("C:/Users/prath/Desktop/karo stup/proj2/data/media.csv", encoding="ISO-8859-1")


# Output directory
output_dir = Path(".")
image_files = []

# ========== Basic EDA ========== 
summary = {
    "shape": df.shape,
    "columns": list(df.columns),
    "dtypes": df.dtypes.astype(str).to_dict(),
    "missing_values": df.isnull().sum().to_dict(),
    "numeric_summary": df.describe(include=[np.number]).to_dict(),
    "categorical_summary": df.describe(include=[object, "category"]).to_dict(),
    "sample_rows": df.sample(min(len(df), 5)).to_dict(orient="records"),
}

# ========== Visualization ========== 
def plot_correlation_heatmap(df, output_path):
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return
    corr = numeric_df.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    image_files.append(output_path.name)

def plot_numeric_histograms(df, output_path):
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] == 0:
        return
    
    # Dynamically calculate the layout
    num_columns = numeric_df.shape[1]
    ncols = min(num_columns, 3)  # Limit to 3 columns per row
    nrows = math.ceil(num_columns / ncols)  # Calculate the required number of rows
    
    # Plot the histograms
    plt.figure(figsize=(12, 4 * nrows))
    numeric_df.hist(bins=20, layout=(nrows, ncols), figsize=(12, 4 * nrows))
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    image_files.append(output_path.name)

plot_correlation_heatmap(df, output_dir / "correlation.png")
plot_numeric_histograms(df, output_dir / "histograms.png")

# ========== GPT-4o-Mini Prompting ========== 
def ask_gpt(prompt, system=None):
    messages = [{"role": "user", "content": prompt}]
    if system:
        messages.insert(0, {"role": "system", "content": system})
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7,
        max_tokens=1024
    )
    return response["choices"][0]["message"]["content"]

# LLM analysis summary
llm_context = f"""
You are given a CSV file with the following characteristics:
- Shape: {summary['shape']}
- Columns and Types: {summary['dtypes']}
- Missing Values: {summary['missing_values']}
- Numeric Summary (sample): {json.dumps({k: v for k, v in summary['numeric_summary'].items() if isinstance(v, dict)})}
- Sample Rows: {json.dumps(summary['sample_rows'][:2])}

Based on this data:
1. Summarize what this dataset appears to be about.
2. List 3 interesting insights or patterns.
3. Suggest 1–2 visualizations (we already created correlation heatmap and histograms).
4. Provide a compelling, readable narrative summarizing the dataset and our findings.
"""

gpt_summary = ask_gpt(llm_context)

# ========== Write README.md ========== 
with open("README.md", "w", encoding="utf-8") as f:
    f.write("# Automated Data Analysis Report\n\n")
    f.write(f"**Input File:** `{filename}`\n\n")
    f.write("##  GPT-4o-Mini Insights\n")
    f.write(gpt_summary + "\n\n")
    f.write("##  Visualizations\n")
    for img in image_files:
        f.write(f"![{img}](./{img})\n")

print("Analysis complete. Generated:")
print("- README.md")
for img in image_files:
    print(f"- {img}")
