import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import PercentFormatter
from sklearn.linear_model import LinearRegression
import commonfunction as cf  # Importing your common functions
import plotly.graph_objects as go



df = pd.read_excel("Fund and Benchmark Returns_Test Series_20240521.xlsx", sheet_name='Sheet1', index_col='Dates')
df_fund = df.iloc[:, 0].copy().to_frame().set_index(df.index)
df_benchmark1 = df.iloc[:, 1].copy().to_frame().set_index(df.index)
df_benchmark2 = df.iloc[:, 2].copy().to_frame().set_index(df.index) if df.shape[1] > 2 else pd.DataFrame()
df_benchmark3 = df.iloc[:, 3].copy().to_frame().set_index(df.index) if df.shape[1] > 3 else pd.DataFrame()

bench1_upside_capture_ratio, bench1_downside_capture_ratio = cf.calculate_capture_ratios(df_fund.iloc[:, 0], df_benchmark1.iloc[:, 0])
bench2_upside_capture_ratio, bench2_downside_capture_ratio = cf.calculate_capture_ratios(df_fund.iloc[:, 0], df_benchmark2.iloc[:, 0])
bench3_upside_capture_ratio, bench3_downside_capture_ratio = cf.calculate_capture_ratios(df_fund.iloc[:, 0], df_benchmark3.iloc[:, 0])

print(df)

print("Fund Returns:")
print(df_fund.head())
print("\nBenchmark 1 Returns:")
print(df_benchmark1.head())
print("\nBenchmark 2 Returns:")
print(df_benchmark2.head())
print("\nBenchmark 3 Returns:")
print(df_benchmark3.head())

'''
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(rolling_ir_df_fund_bench1)
'''

# Initialize an empty list to store capture ratios
capture_ratios = {
    "Upside Capture Ratio": [],
    "Downside Capture Ratio": []
}

benchmarks = []

# Calculate capture ratios for Benchmark 1
if not df_benchmark1.empty:
    bench1_upside_capture_ratio, bench1_downside_capture_ratio = cf.calculate_capture_ratios(df_fund.iloc[:, 0], df_benchmark1.iloc[:, 0])
    print("Benchmark 1 Upside Capture Ratio:", bench1_upside_capture_ratio)
    print("Benchmark 1 Downside Capture Ratio:", bench1_downside_capture_ratio)
    capture_ratios["Upside Capture Ratio"].append(bench1_upside_capture_ratio)
    capture_ratios["Downside Capture Ratio"].append(bench1_downside_capture_ratio)
    benchmarks.append(df_benchmark1.columns[0])

'''
# Check and calculate for Benchmark 2 if it exists
if df.shape[1] > 2:
    df_benchmark2 = df.iloc[:, 2].copy().to_frame().set_index(df.index)
    if not df_benchmark2.empty:
        bench2_upside_capture_ratio, bench2_downside_capture_ratio = cf.calculate_capture_ratios(df_fund.iloc[:, 0], df_benchmark2.iloc[:, 0])
        capture_ratios["Upside Capture Ratio"].append(bench2_upside_capture_ratio)
        capture_ratios["Downside Capture Ratio"].append(bench2_downside_capture_ratio)
        benchmarks.append(df_benchmark2.columns[0])

# Check and calculate for Benchmark 3 if it exists
if df.shape[1] > 3:
    df_benchmark3 = df.iloc[:, 3].copy().to_frame().set_index(df.index)
    if not df_benchmark3.empty:
        bench3_upside_capture_ratio, bench3_downside_capture_ratio = cf.calculate_capture_ratios(df_fund.iloc[:, 0], df_benchmark3.iloc[:, 0])
        capture_ratios["Upside Capture Ratio"].append(bench3_upside_capture_ratio)
        capture_ratios["Downside Capture Ratio"].append(bench3_downside_capture_ratio)
        benchmarks.append(df_benchmark3.columns[0])
'''

# Create the DataFrame
capture_ratio_df = pd.DataFrame(capture_ratios, columns=benchmarks, index=["Upside Capture Ratio", "Downside Capture Ratio"])

print(capture_ratio_df)
print(type(bench1_upside_capture_ratio))