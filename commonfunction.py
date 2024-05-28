import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import PercentFormatter
from sklearn.linear_model import LinearRegression



def calculate_drawdown(return_df):

    fund_col = return_df.columns[0]
    cumulative_return = (1 + return_df[fund_col]).cumprod()
    previous_peaks = cumulative_return.expanding(min_periods=1).max()
    fund_drawdown = (cumulative_return - previous_peaks) / previous_peaks

    cumulative_returns = pd.DataFrame({fund_col: cumulative_return})
    drawdowns_df = pd.DataFrame({fund_col: fund_drawdown})

    if len(return_df.columns) >= 2:
        for col in return_df.columns[1:]:
            cumulative_return_benchmark = (1 + return_df[col]).cumprod()
            previous_peaks = cumulative_return_benchmark.expanding(min_periods=1).max()
            benchmark_drawdown = (cumulative_return_benchmark - previous_peaks) / previous_peaks
            drawdowns_df[col] = benchmark_drawdown
            cumulative_returns[col] = cumulative_return_benchmark

    return drawdowns_df

def sort_returns_by_benchmark(fund_returns, benchmark_returns):
    # Sort the DataFrame by benchmark returns in ascending order
    sorted_df = pd.concat([fund_returns, benchmark_returns], axis=1).sort_values(by=benchmark_returns.columns[0])

    sorted_df_Fund = sorted_df[[sorted_df.columns[0]]].copy()
    sorted_df_Benchmark = sorted_df[[sorted_df.columns[1]]].copy()

    sorted_df_Fund['Fund or Benchmark'] = 'Fund'
    sorted_df_Benchmark['Fund or Benchmark'] = 'Benchmark'

    sorted_df_Fund.rename(columns={sorted_df_Fund.columns[0]: 'Return'}, inplace=True)
    sorted_df_Benchmark.rename(columns={sorted_df_Benchmark.columns[0]: 'Return'}, inplace=True)

    sorted_df_Fund['Row Number'] = range(len(sorted_df_Fund))
    sorted_df_Benchmark['Row Number'] = range(len(sorted_df_Benchmark))

    sorted_df_new = pd.concat([sorted_df_Fund, sorted_df_Benchmark], axis=0)
    sorted_df_new['Row Number'] = sorted_df_new['Row Number'].astype(int)
    sorted_df_new = sorted_df_new.sort_values(by=['Row Number', 'Fund or Benchmark'], ascending=[True, False])

    return sorted_df_new

def plot_sorted_monthly_returns_with_benchmark(portfolio_returns, benchmark_returns):
    # Sort benchmark returns in ascending order
    sorted_benchmark_returns = benchmark_returns.sort_values()

    # Create subplots
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot benchmark returns as bar chart
    ax.bar(range(len(sorted_benchmark_returns)), sorted_benchmark_returns, color='orange', label='Benchmark Returns', width=0.05)

    # Plot corresponding portfolio returns as bar chart next to each benchmark bar
    for i, benchmark_date in enumerate(sorted_benchmark_returns.index):
        portfolio_return = portfolio_returns[benchmark_date]
        ax.bar(i + 0.1, portfolio_return, color='skyblue', label='Portfolio Returns' if i == 0 else None, width=0.05)

    ax.set_title('Sorted Monthly Returns with Benchmark')
    ax.set_xlabel('Month')
    ax.set_ylabel('Return')
    ax.set_xticks(range(len(sorted_benchmark_returns)))
    ax.set_xticklabels(sorted_benchmark_returns.index.strftime('%Y-%m'), rotation=45)
    ax.legend()
    ax.grid(axis='y')
    plt.tight_layout()

    # Return the plot object
    return plt.gca()



def plot_return_distribution_matplotlib(return_series):
    # Calculate minimum and maximum return values
    min_return = return_series.min()
    max_return = return_series.max()

    # Define the bin width
    bin_width = 0.005

    # Determine the number of bins based on the bin width
    num_bins = int((max_return - min_return) / bin_width) + 1

    # Calculate the adjusted maximum return value to ensure all bins have the same width
    max_return_adj = min_return + num_bins * bin_width

    # Round down the leftmost bin edge and round up the rightmost bin edge
    leftmost_edge = math.floor(min_return * 1000) / 1000
    rightmost_edge = math.ceil(max_return_adj * 1000) / 1000

    # Define the bin edges with fixed width
    bins = np.linspace(leftmost_edge, rightmost_edge, num_bins + 1)

    # Insert 0 as a bin edge if necessary
    if min_return < 0 and max_return > 0:
        bins = np.insert(bins, np.searchsorted(bins, 0), 0)

    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create lists to store positive and negative return counts
    positive_counts = []
    negative_counts = []

    # Set background color based on the sign of return values
    ax.set_facecolor('lightgray')  # Set the background color of the plot

    # Create the histogram with transparent bars
    counts, _, patches = ax.hist(return_series, bins=bins, edgecolor='black', alpha=0.5)

    # Calculate counts of positive and negative returns
    for i in range(len(bins) - 1):
        if bins[i] < 0:
            negative_counts.append(int(counts[i]))
        else:
            positive_counts.append(int(counts[i]))

        # Set color of each bar based on return value
        color = 'lightcoral' if bins[i] < 0 else 'lightgreen'
        patches[i].set_facecolor(color)

        # Add count labels on top of each bar
        ax.text(patches[i].get_x() + patches[i].get_width() / 2, patches[i].get_height() + 0.02,
                str(int(counts[i])), ha='center', va='bottom')

    # Create a combined legend for positive and negative returns
    ax.legend([f'Positive Returns: {sum(positive_counts)}\nNegative Returns: {sum(negative_counts)}'],
              loc='upper right', handlelength=0)

    # Add labels and title
    ax.set_title('Monthly Return Distribution')
    ax.set_xlabel('Return')
    ax.set_ylabel('Count')

    # Format return axis labels as percentages
    formatter = FuncFormatter(lambda x, _: f'{x * 100:.1f}%')
    ax.xaxis.set_major_formatter(formatter)

    # Adjust layout to prevent overlap of axis labels
    plt.tight_layout()

    # Return the figure
    return fig




def annualized_return(returns):
    total_return = (1 + returns).prod() - 1
    num_months = len(returns)
    annualized_return = (1 + total_return) ** (12 / num_months) - 1
    return annualized_return


def rolling_annualized_returns(monthly_returns, periods=[12, 36, 60]):
    rolling_returns = pd.DataFrame(index=monthly_returns.index)
    for period in periods:
        rolling_returns[f'{period}-Month Return'] = monthly_returns.rolling(window=period).apply(annualized_return,
                                                                                                 raw=True)

    positive_counts = rolling_returns.apply(lambda x: (x > 0).sum())
    negative_counts = rolling_returns.apply(lambda x: (x < 0).sum())

    return positive_counts, negative_counts, rolling_returns

def annualized_volatility(returns):
    annual_volatility = returns.std() * np.sqrt(12)
    return annual_volatility

def rolling_annualized_volatility(monthly_returns, periods=[12, 36, 60]):
    rolling_volatility = pd.DataFrame(index=monthly_returns.index)
    for period in periods:
        rolling_volatility[f'{period}-Month Volatility'] = monthly_returns.rolling(window=period).apply(annualized_volatility, raw=True)

    return rolling_volatility


def annualized_information_ratio(window_fund_returns, window_benchmark_returns):

    excess_returns = window_fund_returns - window_benchmark_returns
    tracking_error = excess_returns.std()
    mean_excess_return = excess_returns.mean()

    annual_tracking_error = tracking_error * np.sqrt(12)  # Assuming monthly data
    annual_mean_excess_return = mean_excess_return * 12  # Assuming monthly data

    if annual_tracking_error == 0:
        return np.nan
    return annual_mean_excess_return / annual_tracking_error


def rolling_ann_information_ratio(fund_returns, benchmark_returns, periods=[12, 36, 60]):
    rolling_ir_df = pd.DataFrame(index=fund_returns.index)
    for window_size in periods:
        rolling_information_ratio = (
            fund_returns.rolling(window=window_size).apply(
                lambda x: annualized_information_ratio(x, benchmark_returns[x.index]), raw=False
            )
        )
        rolling_ir_df[f'{window_size}-Month Annualized Information Ratio'] = rolling_information_ratio

    return rolling_ir_df



def plot_rolling_annualized_returns(monthly_returns, periods=[12, 36, 60]):
    # Calculate rolling annualized returns for different periods
    rolling_annualized = rolling_annualized_returns(monthly_returns, periods)

    # Plot rolling annualized returns
    fig, ax = plt.subplots(figsize=(10, 6))
    rolling_annualized.plot(ax=ax)

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))

    # Add a bold horizontal line at 0% annualized return
    ax.axhline(y=0, color='black', linestyle='-', linewidth=2)

    # Highlight the year period in the title
    start_year = rolling_annualized.index[0].year
    end_year = rolling_annualized.index[-1].year
    title = f'Rolling Annualized Returns ({start_year}-{end_year})'
    plt.title(title)

    # Add labels and legend
    plt.xlabel('Date')
    plt.ylabel('Annualized Return')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Rolling Period', loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()

    return fig, ax


def plot_rolling_return_occasions(positive_counts, negative_counts):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    labels = ['Positive', 'Negative']

    for i, (positive, negative) in enumerate(zip(positive_counts, negative_counts)):
        ax = axs[i]
        counts = [positive, negative]
        ax.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'lightcoral'])
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

        if i == 0:
            ax.set_title('1-Year Rolling Returns')
        elif i == 1:
            ax.set_title('3-Year Rolling Returns')
        elif i == 2:
            ax.set_title('5-Year Rolling Returns')

    plt.tight_layout()
    return fig


def count_up_down_months(monthly_returns):
    up_months = (monthly_returns > 0).sum()
    up_months = up_months.item()
    down_months = (monthly_returns <= 0).sum()
    down_months = down_months.item()
    return up_months, down_months


def plot_up_down_months(up_months, down_months):
    total_months = up_months + down_months
    up_percentage = up_months / total_months * 100
    down_percentage = down_months / total_months * 100

    labels = [f'Up Months\n({up_months})', f'Down Months\n({down_months})']
    sizes = [up_percentage, down_percentage]
    colors = ['lightgreen', 'lightcoral']
    explode = (0.1, 0)  # explode the 1st slice

    fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    ax.set_title('Percentage and Number of Up and Down Months')
    return fig

def average_up_magnitude(portfolio_returns, benchmark_returns=None):
    # Calculate the average up magnitude for the portfolio
    portfolio_up = portfolio_returns[portfolio_returns > 0]
    portfolio_avg_up_magnitude = (portfolio_up.mean(axis=0)).item()
    # portfolio_avg_up_magnitude = '{:.2f}%'.format(portfolio_avg_up_magnitude)

    # Calculate the average up magnitude for the benchmark if provided
    if benchmark_returns is not None:
        benchmark_up = benchmark_returns[benchmark_returns > 0]
        benchmark_avg_up_magnitude = (benchmark_up.mean(axis=0)).item()
        # benchmark_avg_up_magnitude = '{:.2f}%'.format(benchmark_avg_up_magnitude)
    else:
        benchmark_avg_up_magnitude = None

    return portfolio_avg_up_magnitude, benchmark_avg_up_magnitude

def plot_average_up_magnitude(portfolio_avg_up, benchmark_avg_up):
    labels = ['Portfolio', 'Benchmark']
    values = [portfolio_avg_up, benchmark_avg_up]

    fig, ax = plt.subplots()
    bars = ax.bar(labels, values, color=['blue', 'green'])
    ax.set_xlabel('Asset')
    ax.set_ylabel('Average Up Magnitude (%)')
    ax.set_title('Average Up Magnitude Comparison')

    # Adding data labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, '{:.2f}%'.format(height), ha='center', va='bottom')

    return fig

def average_down_magnitude(portfolio_returns, benchmark_returns=None):
    # Calculate the average up magnitude for the portfolio
    portfolio_down = portfolio_returns[portfolio_returns <= 0]
    portfolio_avg_down_magnitude = (portfolio_down.mean(axis=0)).item()

    # Calculate the average up magnitude for the benchmark if provided
    if benchmark_returns is not None:
        benchmark_down = benchmark_returns[benchmark_returns <= 0]
        benchmark_avg_down_magnitude = (benchmark_down.mean(axis=0)).item()
        # benchmark_avg_down_magnitude = '{:.2f}%'.format(benchmark_avg_down_magnitude)
    else:
        benchmark_avg_down_magnitude = None

    return portfolio_avg_down_magnitude, benchmark_avg_down_magnitude

def plot_average_down_magnitude(portfolio_avg_down, benchmark_avg_down):
    labels = ['Portfolio', 'Benchmark']
    values = [portfolio_avg_down, benchmark_avg_down]

    fig, ax = plt.subplots()
    bars = ax.bar(labels, values, color=['blue', 'green'])
    ax.set_xlabel('Asset')
    ax.set_ylabel('Average Down Magnitude (%)')
    ax.set_title('Average Down Magnitude Comparison')

    # Adding data labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, '{:.2f}%'.format(height), ha='center', va='bottom')

    return fig


def calculate_yearly_returns(monthly_returns):
    # Initialize an empty list to store returns
    returns_list = []

    # Calculate returns for each year
    for year in monthly_returns.index.year.unique():
        year_returns = monthly_returns[monthly_returns.index.year == year]
        annual_return = ((1 + year_returns).prod() - 1) * 100
        returns_list.append({'Year': year, monthly_returns.columns[0]: annual_return.values.item()})

    # Convert the list of dictionaries to a DataFrame
    returns_df = pd.DataFrame(returns_list)
    returns_df['Year'] = returns_df['Year'].astype(int)
    returns_df.set_index('Year', inplace=True)


    # Calculate YTD return
    current_year = monthly_returns.index[-1].year
    ytd_returns = monthly_returns.iloc[monthly_returns.index.year == current_year, 0]
    ytd_return = ((1 + ytd_returns).prod() - 1) * 100

    # Latest date
    latest_date = monthly_returns.index[-1]

    # Calculate 1-year return if there are enough data points
    if len(monthly_returns) >= 12:
        one_year_ago = latest_date - pd.DateOffset(months=11)
        one_year_monthly_return_series = monthly_returns.loc[one_year_ago:latest_date]
        one_year_return = ((1 + one_year_monthly_return_series).prod() - 1) * 100
        one_year_return = one_year_return.item()
    else:
        one_year_return = "N.A."

    # Calculate 3-year return if there are enough data points
    if len(monthly_returns) >= 36:
        three_years_ago = latest_date - pd.DateOffset(months=35)
        three_year_monthly_return_series = monthly_returns.loc[three_years_ago:latest_date]
        three_year_return = (((1 + three_year_monthly_return_series).prod()) ** (1 / 3) - 1) * 100
        three_year_return = three_year_return.item()
    else:
        three_year_return = "N.A."

    # Calculate 5-year return if there are enough data points
    if len(monthly_returns) >= 60:
        five_years_ago = latest_date - pd.DateOffset(months=59)
        five_year_monthly_return_series = monthly_returns.loc[five_years_ago:latest_date]
        five_year_return = (((1 + five_year_monthly_return_series).prod()) ** (1 / 5) - 1) * 100
        five_year_return = five_year_return.item()
    else:
        five_year_return = "N.A."

    # Create a DataFrame to store the summary returns
    summary_df = pd.DataFrame({
        '1-Year Return': one_year_return,
        '3-Year Return': three_year_return,
        '5-Year Return': five_year_return,
        'YTD Return': ytd_return
    }, index=[monthly_returns.columns[0]])

    # Format numeric values as percentages
    # summary_df = summary_df.applymap(lambda x: '{:.2f}%'.format(x) if isinstance(x, (int, float)) else x)
    summary_df = summary_df.map(lambda x: '{:.2f}%'.format(x) if isinstance(x, (int, float)) else x)

    # Convert annual returns to percentage numbers
    returns_df[monthly_returns.columns[0]] = returns_df[monthly_returns.columns[0]].apply(lambda x: '{:.2f}%'.format(x))
    # returns_df['Return'] = returns_df['Return'] / 100

    return returns_df, summary_df

def calculate_annualized_volatiity(monthly_returns):

    # Latest date
    latest_date = monthly_returns.index[-1]

    # Calculate 1-year return if there are enough data points
    if len(monthly_returns) >= 12:
        one_year_ago = latest_date - pd.DateOffset(months=11)
        one_year_monthly_return_series = monthly_returns.loc[one_year_ago:latest_date]
        one_year_volatility = one_year_monthly_return_series.std() * math.sqrt(12) * 100
        # one_year_return = one_year_return.item()
    else:
        one_year_volatility = "N.A."

    # Calculate 3-year return if there are enough data points
    if len(monthly_returns) >= 36:
        three_years_ago = latest_date - pd.DateOffset(months=35)
        three_year_monthly_return_series = monthly_returns.loc[three_years_ago:latest_date]
        three_year_volatility = three_year_monthly_return_series.std() * math.sqrt(12) * 100
        # three_year_return = three_year_return.item()
    else:
        three_year_volatility = "N.A."

    # Calculate 5-year return if there are enough data points
    if len(monthly_returns) >= 60:
        five_years_ago = latest_date - pd.DateOffset(months=59)
        five_year_monthly_return_series = monthly_returns.loc[five_years_ago:latest_date]
        five_year_volatility = five_year_monthly_return_series.std() * math.sqrt(12) * 100
        # five_year_return = five_year_return.item()
    else:
        five_year_volatility = "N.A."

    # Create a DataFrame to store the summary returns
    vol_summary_df = pd.DataFrame({
        '1-Year Volatility': one_year_volatility,
        '3-Year Volatility': three_year_volatility,
        '5-Year Volatility': five_year_volatility
    }, index=[monthly_returns.columns[0]])

    # Format numeric values as percentages
    # summary_df = summary_df.applymap(lambda x: '{:.2f}%'.format(x) if isinstance(x, (int, float)) else x)
    vol_summary_df = vol_summary_df.map(lambda x: '{:.2f}%'.format(x) if isinstance(x, (int, float)) else x)

    return vol_summary_df

def calculate_rolling_correlation(fund_returns, benchmark_returns):
    # Take the shorter return series as the basis for correlation calculation
    min_length = min(len(fund_returns), len(benchmark_returns))
    fund_returns = fund_returns[-min_length:]
    benchmark_returns = benchmark_returns[-min_length:]

    # Calculate rolling correlations
    rolling_12m_correlation = fund_returns.iloc[:, 0].rolling(window=12).corr(benchmark_returns.iloc[:, 0])
    rolling_36m_correlation = fund_returns.iloc[:, 0].rolling(window=36).corr(benchmark_returns.iloc[:, 0])
    rolling_60m_correlation = fund_returns.iloc[:, 0].rolling(window=60).corr(benchmark_returns.iloc[:, 0])

    rolling_12m_correlation = rolling_12m_correlation.to_frame()
    rolling_12m_correlation.rename(columns={0: 'Rolling 12M Correlation'}, inplace=True)
    rolling_36m_correlation = rolling_36m_correlation.to_frame()
    rolling_36m_correlation.rename(columns={0: 'Rolling 36M Correlation'}, inplace=True)
    rolling_60m_correlation = rolling_60m_correlation.to_frame()
    rolling_60m_correlation.rename(columns={0: 'Rolling 60M Correlation'}, inplace=True)

    rolling_correlation = pd.concat([rolling_12m_correlation, rolling_36m_correlation, rolling_60m_correlation], axis=1)

    '''
    # Plot the rolling correlations
    plt.figure(figsize=(10, 6))
    plt.plot(rolling_12m_correlation, label='12-Month Rolling Correlation')
    plt.plot(rolling_36m_correlation, label='36-Month Rolling Correlation')
    plt.plot(rolling_60m_correlation, label='60-Month Rolling Correlation')
    plt.xlabel('Date')
    plt.ylabel('Correlation')
    plt.title('Rolling Correlation between Fund Returns and Benchmark Returns')
    plt.ylim(-1, 1)  # Set y-axis limits to -1 and 1
    plt.legend()
    plt.grid(True)
    plt.show()

    # Return the plot
    return plt
    '''

    return rolling_correlation

def calculate_rolling_beta(fund_returns, benchmark_returns):
    # Find the common index range
    common_index = fund_returns.index.intersection(benchmark_returns.index)
    fund_returns = fund_returns.loc[common_index]
    benchmark_returns = benchmark_returns.loc[common_index]

    rolling_12m_beta = []
    rolling_36m_beta = []
    rolling_60m_beta = []
    dates = []

    # Calculate rolling beta using linear regression
    for i in range(len(common_index) - 12 + 1):  # Iterate until the last possible 12-month window
        end_date = common_index[i + 11]  # End date of the rolling window

        X = benchmark_returns[i:i + 12].values.reshape(-1, 1)
        y = fund_returns[i:i + 12].values.reshape(-1, 1)  # Use 12-month fund returns for each window

        model_12m = LinearRegression().fit(X, y)  # Fit model with 12-month window
        rolling_12m_beta.append(model_12m.coef_[0][0])

        if i >= 24:  # Check if there are enough data points for 36-month window
            X_36m = benchmark_returns[i - 24:i + 12].values.reshape(-1, 1)
            y_36m = fund_returns[i - 24:i + 12].values.reshape(-1, 1)
            model_36m = LinearRegression().fit(X_36m, y_36m)  # Fit model with 36-month window
            rolling_36m_beta.append(model_36m.coef_[0][0])
        else:
            rolling_36m_beta.append(np.nan)  # If not enough data points, add NaN to the list

        if i >= 48:  # Check if there are enough data points for 60-month window
            X_60m = benchmark_returns[i - 48:i + 12].values.reshape(-1, 1)
            y_60m = fund_returns[i - 48:i + 12].values.reshape(-1, 1)
            model_60m = LinearRegression().fit(X_60m, y_60m)  # Fit model with 60-month window
            rolling_60m_beta.append(model_60m.coef_[0][0])
        else:
            rolling_60m_beta.append(np.nan)  # If not enough data points, add NaN to the list

        # Store the date for each window
        dates.append(end_date)
        # dates.append(str(end_date))  # Convert date to string format

    rolling_12m_beta_df = pd.DataFrame(rolling_12m_beta, index=dates)
    rolling_36m_beta_df = pd.DataFrame(rolling_36m_beta, index=dates)
    rolling_60m_beta_df = pd.DataFrame(rolling_60m_beta, index=dates)

    rolling_12m_beta_df.index.name = 'Dates'
    rolling_36m_beta_df.index.name = 'Dates'
    rolling_60m_beta_df.index.name = 'Dates'

    rolling_12m_beta_df.columns = ['Rolling 12M Beta']
    rolling_36m_beta_df.columns = ['Rolling 36M Beta']
    rolling_60m_beta_df.columns = ['Rolling 60M Beta']

    rolling_beta_df = pd.concat([rolling_12m_beta_df, rolling_36m_beta_df, rolling_60m_beta_df], axis=1)

    '''
    # Plot the rolling betas
    plt.figure(figsize=(10, 6))
    plt.plot(dates, rolling_12m_beta, label='12-Month Rolling Beta')
    plt.plot(dates, rolling_36m_beta, label='36-Month Rolling Beta')
    plt.plot(dates, rolling_60m_beta, label='60-Month Rolling Beta')
    plt.xlabel('End Date of Rolling Window')
    plt.ylabel('Beta')
    plt.title('Rolling Beta between Fund Returns and Benchmark Returns')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.show()

    return plt
    '''

    return rolling_beta_df

def calculate_rolling_alpha(fund_returns, benchmark_returns):
    # Find the common index range
    common_index = fund_returns.index.intersection(benchmark_returns.index)
    fund_returns = fund_returns.loc[common_index]
    benchmark_returns = benchmark_returns.loc[common_index]

    rolling_12m_alpha = []
    rolling_36m_alpha = []
    rolling_60m_alpha = []
    dates = []

    # Calculate rolling alpha using linear regression
    for i in range(len(common_index) - 12 + 1):  # Iterate until the last possible 12-month window
        end_date = common_index[i + 11]  # End date of the rolling window

        X = benchmark_returns[i:i + 12].values.reshape(-1, 1)
        y = fund_returns[i:i + 12].values.reshape(-1, 1)  # Use 12-month fund returns for each window

        model_12m = LinearRegression().fit(X, y)  # Fit model with 12-month window
        rolling_12m_alpha.append(model_12m.intercept_[0])

        if i >= 24:  # Check if there are enough data points for 36-month window
            X_36m = benchmark_returns[i - 24:i + 12].values.reshape(-1, 1)
            y_36m = fund_returns[i - 24:i + 12].values.reshape(-1, 1)
            model_36m = LinearRegression().fit(X_36m, y_36m)  # Fit model with 36-month window
            rolling_36m_alpha.append(model_36m.intercept_[0])
        else:
            rolling_36m_alpha.append(np.nan)  # If not enough data points, add NaN to the list

        if i >= 48:  # Check if there are enough data points for 60-month window
            X_60m = benchmark_returns[i - 48:i + 12].values.reshape(-1, 1)
            y_60m = fund_returns[i - 48:i + 12].values.reshape(-1, 1)
            model_60m = LinearRegression().fit(X_60m, y_60m)  # Fit model with 60-month window
            rolling_60m_alpha.append(model_60m.intercept_[0])
        else:
            rolling_60m_alpha.append(np.nan)  # If not enough data points, add NaN to the list

        # Store the date for each window
        dates.append(end_date)  # Convert date to string format

    rolling_12m_alpha_df = pd.DataFrame(rolling_12m_alpha, index=dates)
    rolling_36m_alpha_df = pd.DataFrame(rolling_36m_alpha, index=dates)
    rolling_60m_alpha_df = pd.DataFrame(rolling_60m_alpha, index=dates)

    rolling_12m_alpha_df.index.name = 'Dates'
    rolling_36m_alpha_df.index.name = 'Dates'
    rolling_60m_alpha_df.index.name = 'Dates'

    rolling_12m_alpha_df.columns = ['Rolling 12M alpha']
    rolling_36m_alpha_df.columns = ['Rolling 36M alpha']
    rolling_60m_alpha_df.columns = ['Rolling 60M alpha']

    rolling_alpha_df = pd.concat([rolling_12m_alpha_df, rolling_36m_alpha_df, rolling_60m_alpha_df], axis=1)

    '''
    # Plot the rolling alphas
    plt.figure(figsize=(10, 6))
    plt.plot(dates, rolling_12m_alpha, label='12-Month Rolling Alpha')
    plt.plot(dates, rolling_36m_alpha, label='36-Month Rolling Alpha')
    plt.plot(dates, rolling_60m_alpha, label='60-Month Rolling Alpha')
    plt.xlabel('End Date of Rolling Window')
    plt.ylabel('Alpha')
    plt.title('Rolling Alpha between Fund Returns and Benchmark Returns')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.show()

    return plt
    '''
    return rolling_alpha_df


def scenario_analysis_cumulative_return(fund_returns, benchmark_returns1=None, benchmark_returns2=None,
                                        benchmark_returns3=None, start_date=None, end_date=None, output_excel=None):
    # Define scenarios with month-end start dates
    scenarios = {
        'Asian Financial Crisis (Jun1997 - Nov1997)': ('1997-07-31', '1997-11-30'),
        'Russian Financial Crisis (Aug1998)': ('1998-08-31', '1998-08-31'),
        'Dot-Com Bubble Burst (Mar2000 - Oct2002)': ('2000-03-31', '2002-10-31'),
        'Global Financial Crisis (Aug2007 - Mar2009)': ('2007-08-31', '2009-03-31'),
        'European Sovereign Debt Crisis (Jan2009 - Dec2012)': ('2009-01-31', '2012-12-31'),
        'COVID-19 Pandemic Economic Crisis (Mar2020)': ('2020-03-31', '2020-03-31')  # End date equal to start date
    }

    # Add tailored scenario if start_date and end_date are provided
    if start_date and end_date:
        scenarios['Tailored Scenario'] = (start_date, end_date)

    cumulative_returns = pd.DataFrame(columns=[fund_returns.name, benchmark_returns1.name, benchmark_returns2.name, benchmark_returns3.name],
                                      index=list(scenarios.keys()))

    for i, (scenario, (scenario_start, scenario_end)) in enumerate(scenarios.items()):
        if fund_returns.index[0] <= pd.Timestamp(scenario_start) and fund_returns.index[-1] >= pd.Timestamp(scenario_end):
            relevant_fund_returns = fund_returns.loc[scenario_start:scenario_end]
            fund_cumulative_return = (1 + relevant_fund_returns).cumprod() - 1
            cumulative_returns.loc[scenario, fund_returns.name] = fund_cumulative_return.iloc[-1]
            if benchmark_returns1 is not None:
                if benchmark_returns1.index[0] <= pd.Timestamp(scenario_start) and benchmark_returns1.index[-1] >= pd.Timestamp(scenario_end):
                    relevant_benchmark_returns1 = benchmark_returns1.loc[scenario_start:scenario_end]
                    benchmark1_cumulative_return = (1 + relevant_benchmark_returns1).cumprod() - 1
                    cumulative_returns.loc[scenario, benchmark_returns1.name] = benchmark1_cumulative_return.iloc[-1]
            if benchmark_returns2 is not None:
                if benchmark_returns2.index[0] <= pd.Timestamp(scenario_start) and benchmark_returns2.index[-1] >= pd.Timestamp(scenario_end):
                    relevant_benchmark_returns2 = benchmark_returns2.loc[scenario_start:scenario_end]
                    benchmark2_cumulative_return = (1 + relevant_benchmark_returns2).cumprod() - 1
                    cumulative_returns.loc[scenario, benchmark_returns2.name] = benchmark2_cumulative_return.iloc[-1]
            if benchmark_returns3 is not None:
                if benchmark_returns3.index[0] <= pd.Timestamp(scenario_start) and benchmark_returns3.index[-1] >= pd.Timestamp(scenario_end):
                    relevant_benchmark_returns3 = benchmark_returns3.loc[scenario_start:scenario_end]
                    benchmark3_cumulative_return = (1 + relevant_benchmark_returns3).cumprod() - 1
                    cumulative_returns.loc[scenario, benchmark_returns3.name] = benchmark3_cumulative_return.iloc[-1]

    '''
    # Output cumulative_returns as Excel if specified
    if output_excel:
        cumulative_returns.to_excel(output_excel)

    # Plot the cumulative returns
    plt.figure(figsize=(12, 8))

    bar_width = 0.08  # Bar width
    index = np.arange(len(cumulative_returns.index))

    for i, column in enumerate(cumulative_returns.columns):
        plt.bar([x + i * bar_width for x in index], cumulative_returns[column], width=bar_width, label=column)

    plt.xlabel('Scenario')
    plt.ylabel('Cumulative Return')
    plt.title('Cumulative Returns for Various Scenarios')

    # Format y-axis as percentage
    formatter = FuncFormatter(lambda y, _: '{:.0%}'.format(y))
    plt.gca().yaxis.set_major_formatter(formatter)

    plt.xticks(index + (len(cumulative_returns.columns) - 1) * bar_width / 2,
               cumulative_returns.index)  # Setting category names as tick labels
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()
    '''

    return cumulative_returns


def calculate_capture_ratios(fund_returns, benchmark_returns):

    # Align the indices of fund_returns and benchmark_returns to their intersection
    common_index = fund_returns.index.intersection(benchmark_returns.index)
    fund_returns = fund_returns.loc[common_index]
    benchmark_returns = benchmark_returns.loc[common_index]

    # Upside capture ratio
    positive_benchmark = benchmark_returns[benchmark_returns > 0]
    if not positive_benchmark.empty:
        upside_capture_ratio = (fund_returns[benchmark_returns > 0].mean() / positive_benchmark.mean()) * 100
    else:
        upside_capture_ratio = None

    # Downside capture ratio
    negative_benchmark = benchmark_returns[benchmark_returns <= 0]
    if not negative_benchmark.empty:
        downside_capture_ratio = (fund_returns[benchmark_returns <= 0].mean() / negative_benchmark.mean()) * 100
    else:
        downside_capture_ratio = None

    return float(upside_capture_ratio), float(downside_capture_ratio)
