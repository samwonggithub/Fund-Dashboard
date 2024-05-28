import pandas as pd
import numpy as np
import commonfunction as cf

# Load data from Excel
df = pd.read_excel("Fund and Benchmark Returns_Test Series_20240521.xlsx", sheet_name='Sheet1', index_col='Dates')

# Extract fund and benchmark returns
df_fund = df.iloc[:, 0].copy().to_frame()
df_benchmark1 = df.iloc[:, 1].copy().to_frame()
df_benchmark2 = df.iloc[:, 2].copy().to_frame() if df.shape[1] > 2 else pd.DataFrame()
df_benchmark3 = df.iloc[:, 3].copy().to_frame() if df.shape[1] > 3 else pd.DataFrame()

# Initialize an empty dictionary to store capture ratios
capture_ratios = {
    "Upside Capture Ratio": [],
    "Downside Capture Ratio": []
}

benchmarks = []

'''
# Calculate capture ratios for Benchmark 1
if not df_benchmark1.empty:
    bench1_upside_capture_ratio, bench1_downside_capture_ratio = calculate_capture_ratios(df_fund.iloc[:, 0], df_benchmark1.iloc[:, 0])
    capture_ratios["Upside Capture Ratio"].append(bench1_upside_capture_ratio)
    capture_ratios["Downside Capture Ratio"].append(bench1_downside_capture_ratio)
    benchmarks.append(df_benchmark1.columns[0])

# Check and calculate for Benchmark 2 if it exists
if not df_benchmark2.empty:
    bench2_upside_capture_ratio, bench2_downside_capture_ratio = calculate_capture_ratios(df_fund.iloc[:, 0], df_benchmark2.iloc[:, 0])
    capture_ratios["Upside Capture Ratio"].append(bench2_upside_capture_ratio)
    capture_ratios["Downside Capture Ratio"].append(bench2_downside_capture_ratio)
    benchmarks.append(df_benchmark2.columns[0])

# Check and calculate for Benchmark 3 if it exists
if not df_benchmark3.empty:
    bench3_upside_capture_ratio, bench3_downside_capture_ratio = calculate_capture_ratios(df_fund.iloc[:, 0], df_benchmark3.iloc[:, 0])
    capture_ratios["Upside Capture Ratio"].append(bench3_upside_capture_ratio)
    capture_ratios["Downside Capture Ratio"].append(bench3_downside_capture_ratio)
    benchmarks.append(df_benchmark3.columns[0])

# Create the DataFrame
capture_ratio_df = pd.DataFrame(capture_ratios, index=benchmarks)
capture_ratio_df = capture_ratio_df.T
'''
# Initialize dictionaries to store capture ratios
bench1_capture_ratios = {
    "Upside Capture Ratio": [],
    "Downside Capture Ratio": []
}

bench2_capture_ratios = {
    "Upside Capture Ratio": [],
    "Downside Capture Ratio": []
}

bench3_capture_ratios = {
    "Upside Capture Ratio": [],
    "Downside Capture Ratio": []
}

# Calculate capture ratios for Benchmark 1
if not df_benchmark1.empty:
    bench1_upside_capture_ratio, bench1_downside_capture_ratio = cf.calculate_capture_ratios(df_fund.iloc[:, 0], df_benchmark1.iloc[:, 0])
    bench1_capture_ratios["Upside Capture Ratio"].append(bench1_upside_capture_ratio)
    bench1_capture_ratios["Downside Capture Ratio"].append(bench1_downside_capture_ratio)
    benchmark1 = df_benchmark1.columns[0]

# Check and calculate for Benchmark 2 if it exists
if not df_benchmark2.empty:
    bench2_upside_capture_ratio, bench2_downside_capture_ratio = cf.calculate_capture_ratios(df_fund.iloc[:, 0], df_benchmark2.iloc[:, 0])
    bench2_capture_ratios["Upside Capture Ratio"].append(bench2_upside_capture_ratio)
    bench2_capture_ratios["Downside Capture Ratio"].append(bench2_downside_capture_ratio)
    benchmark2 = df_benchmark2.columns[0]

# Check and calculate for Benchmark 3 if it exists
if not df_benchmark3.empty:
    bench3_upside_capture_ratio, bench3_downside_capture_ratio = cf.calculate_capture_ratios(df_fund.iloc[:, 0], df_benchmark3.iloc[:, 0])
    bench3_capture_ratios["Upside Capture Ratio"].append(bench3_upside_capture_ratio)
    bench3_capture_ratios["Downside Capture Ratio"].append(bench3_downside_capture_ratio)
    benchmark3 = df_benchmark3.columns[0]

# Create separate DataFrames for each benchmark
if not df_benchmark1.empty:
    capture_ratio_df_benchmark1 = pd.DataFrame(bench1_capture_ratios, index=[benchmark1])

if not df_benchmark2.empty:
    capture_ratio_df_benchmark2 = pd.DataFrame(bench2_capture_ratios, index=[benchmark2])

if not df_benchmark3.empty:
    capture_ratio_df_benchmark3 = pd.DataFrame(bench3_capture_ratios, index=[benchmark3])

