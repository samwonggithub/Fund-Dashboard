import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
import math
import commonfunction as cf  # Importing your common functions
import plotly.express as px  # Importing Plotly Express
import plotly.figure_factory as ff
import plotly.graph_objects as go
import pdfkit
import dash_renderer

# Initialize Dash app
app = dash.Dash(__name__)

# Define layout for input
app.layout = html.Div(
    children=[
        html.H1("Fund Analysis Dashboard (by Sam CS Wong)"),
        html.Div([
            html.Label("Select Fund and Benchmark Return File:"),
            dcc.Upload(
                id='upload-file',
                children=html.Button('Upload File')
            ),
            html.Div(id='filename'),
            html.Button('Submit', id='submit-button'),
            # html.Button('Export to PDF', id='export-pdf-button'),
            # html.Div(id='pdf-export-message'),
            html.Div(id='fund-name-subtitle',
                     style={'font-size': '60px', 'margin-top': '10px', 'font-weight': 'bold'}),
            html.Div(id='data-analysis-period',
                     style={'font-size': '50px', 'margin-top': '10px', 'font-weight': 'italic'})
        ]),
        html.Div(id='output-charts', style={'margin-top': '20px'})
    ]
)

# Function to read uploaded Excel file and return dataframe
def read_excel_to_dataframe(filename):
    if filename:
        df = pd.read_excel(filename, index_col='Dates')
        return df
    else:
        return None

# Callback to store uploaded file name
@app.callback(
    Output('filename', 'children'),
    [Input('upload-file', 'filename')]
)
def display_filename(filename):
    return filename

# Callback to generate dataframes and charts
@app.callback(
    [Output('output-charts', 'children'),
     Output('fund-name-subtitle', 'children'),
     Output('data-analysis-period', 'children')],
     # Output('pdf-export-message', 'children'),
    [Input('submit-button', 'n_clicks')],
     # Input('export-pdf-button', 'n_clicks')],
    [dash.dependencies.State('filename', 'children')]
)

def generate_dataframes_and_charts(n_clicks, filename):
    if filename:

        df = pd.read_excel(filename, sheet_name='Sheet1', index_col='Dates')
        df_fund = df.iloc[:, 0].copy().to_frame().set_index(df.index)
        df_benchmark1 = df.iloc[:, 1].copy().to_frame().set_index(df.index)
        df_benchmark2 = pd.DataFrame(index=df.index, columns=['Benchmark 2'])
        df_benchmark3 = pd.DataFrame(index=df.index, columns=['Benchmark 3'])

        if df.shape[1] == 3:
            df_benchmark2 = df.iloc[:, 2].copy().to_frame().set_index(df.index)
        elif df.shape[1] == 4:
            df_benchmark2 = df.iloc[:, 2].copy().to_frame().set_index(df.index)
            df_benchmark3 = df.iloc[:, 3].copy().to_frame().set_index(df.index)

        fund_name = df_fund.columns[0]
        fund_analysis_start_date = df_fund.index[0]
        fund_analysis_end_date = df_fund.index[-1]
        fund_analysis_start_date_month = fund_analysis_start_date.strftime('%b')
        fund_analysis_start_date_year = fund_analysis_start_date.year
        fund_analysis_end_date_month = fund_analysis_end_date.strftime('%b')
        fund_analysis_end_date_year = fund_analysis_end_date.year

        # Calculate drawdown
        df_drawdown = cf.calculate_drawdown(df)

        # Calculate Annualized Returns and Yearly Returns
        df_fund_ann_return, df_fund_summary = cf.calculate_yearly_returns(df_fund)
        df_bench1_ann_return, df_bench1_summary = cf.calculate_yearly_returns(df_benchmark1)
        df_bench2_ann_return, df_bench2_summary = cf.calculate_yearly_returns(df_benchmark2)
        df_bench3_ann_return, df_bench3_summary = cf.calculate_yearly_returns(df_benchmark3)

        df_ann_return = pd.concat(
            [df_fund_ann_return, df_bench1_ann_return, df_bench2_ann_return, df_bench3_ann_return], axis=1)
        if df_benchmark3.isna().all().all():
            df_ann_return = df_ann_return.drop(columns=df_ann_return.columns[-1])
            if df_benchmark2.isna().all().all():
                df_ann_return = df_ann_return.drop(columns=df_ann_return.columns[-1])

        df_ann_summary = pd.concat(
            [df_fund_summary, df_bench1_summary, df_bench2_summary, df_bench3_summary], axis=0)
        df_ann_summary = df_ann_summary.T

        if df_benchmark3.isna().all().all():
            df_ann_summary = df_ann_summary.drop(columns=df_ann_summary.columns[-1])
            if df_benchmark2.isna().all().all():
                df_ann_summary = df_ann_summary.drop(columns=df_ann_summary.columns[-1])

        df_fund_vol_summary = cf.calculate_annualized_volatiity(df_fund)
        df_bench1_vol_summary = cf.calculate_annualized_volatiity(df_benchmark1)
        df_bench2_vol_summary = cf.calculate_annualized_volatiity(df_benchmark2)
        df_bench3_vol_summary = cf.calculate_annualized_volatiity(df_benchmark3)

        df_vol_summary = pd.concat(
            [df_fund_vol_summary, df_bench1_vol_summary, df_bench2_vol_summary, df_bench3_vol_summary], axis=0)

        df_vol_summary = df_vol_summary.T

        if df_benchmark3.isna().all().all():
            df_vol_summary = df_vol_summary.drop(columns=df_vol_summary.columns[-1])
            if df_benchmark2.isna().all().all():
                df_vol_summary = df_vol_summary.drop(columns=df_vol_summary.columns[-1])

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
            bench1_upside_capture_ratio, bench1_downside_capture_ratio = cf.calculate_capture_ratios(df_fund.iloc[:, 0],
                                                                                                     df_benchmark1.iloc[
                                                                                                     :, 0])
            bench1_capture_ratios["Upside Capture Ratio"].append(bench1_upside_capture_ratio)
            bench1_capture_ratios["Downside Capture Ratio"].append(bench1_downside_capture_ratio)
            benchmark1 = df_benchmark1.columns[0]

        # Check and calculate for Benchmark 2 if it exists
        if df.shape[1] == 3:
            bench2_upside_capture_ratio, bench2_downside_capture_ratio = cf.calculate_capture_ratios(df_fund.iloc[:, 0],
                                                                                                     df_benchmark2.iloc[
                                                                                                     :, 0])
            bench2_capture_ratios["Upside Capture Ratio"].append(bench2_upside_capture_ratio)
            bench2_capture_ratios["Downside Capture Ratio"].append(bench2_downside_capture_ratio)
            benchmark2 = df_benchmark2.columns[0]

        # Check and calculate for Benchmark 3 if it exists
        if df.shape[1] == 4:
            bench2_upside_capture_ratio, bench2_downside_capture_ratio = cf.calculate_capture_ratios(df_fund.iloc[:, 0],
                                                                                                     df_benchmark2.iloc[
                                                                                                     :, 0])
            bench2_capture_ratios["Upside Capture Ratio"].append(bench2_upside_capture_ratio)
            bench2_capture_ratios["Downside Capture Ratio"].append(bench2_downside_capture_ratio)
            benchmark2 = df_benchmark2.columns[0]

            bench3_upside_capture_ratio, bench3_downside_capture_ratio = cf.calculate_capture_ratios(df_fund.iloc[:, 0],
                                                                                                     df_benchmark3.iloc[
                                                                                                     :, 0])
            bench3_capture_ratios["Upside Capture Ratio"].append(bench3_upside_capture_ratio)
            bench3_capture_ratios["Downside Capture Ratio"].append(bench3_downside_capture_ratio)
            benchmark3 = df_benchmark3.columns[0]

        # Create separate DataFrames for each benchmark
        if df.shape[1] == 2:
            capture_ratio_df_benchmark1 = pd.DataFrame(bench1_capture_ratios, index=[benchmark1])
        elif df.shape[1] == 3:
            capture_ratio_df_benchmark1 = pd.DataFrame(bench1_capture_ratios, index=[benchmark1])
            capture_ratio_df_benchmark2 = pd.DataFrame(bench2_capture_ratios, index=[benchmark2])
        elif df.shape[1] == 4:
            capture_ratio_df_benchmark1 = pd.DataFrame(bench1_capture_ratios, index=[benchmark1])
            capture_ratio_df_benchmark2 = pd.DataFrame(bench2_capture_ratios, index=[benchmark2])
            capture_ratio_df_benchmark3 = pd.DataFrame(bench3_capture_ratios, index=[benchmark3])


        # Calculate rolling information ratios
        rolling_ir_df_fund_bench1 = cf.rolling_ann_information_ratio(df_fund.iloc[:, 0], df_benchmark1.iloc[:, 0],
                                                                     periods=[12, 36, 60])
        rolling_ir_df_fund_bench2 = cf.rolling_ann_information_ratio(df_fund.iloc[:, 0], df_benchmark2.iloc[:, 0],
                                                                     periods=[12, 36, 60])
        rolling_ir_df_fund_bench3 = cf.rolling_ann_information_ratio(df_fund.iloc[:, 0], df_benchmark3.iloc[:, 0],
                                                                     periods=[12, 36, 60])


        # Avg Up Magnitude calculations
        df_fund_avg_up_magnitude, df_benchmark_avg_up_magnitude = cf.average_up_magnitude(df_fund)
        df_bench1_avg_up_magnitude, df_benchmark_avg_up_magnitude = cf.average_up_magnitude(df_benchmark1)
        df_bench2_avg_up_magnitude, df_benchmark_avg_up_magnitude = cf.average_up_magnitude(df_benchmark2)
        df_bench3_avg_up_magnitude, df_benchmark_avg_up_magnitude = cf.average_up_magnitude(df_benchmark3)

        # Avg Down Magnitude calculations
        df_fund_avg_down_magnitude, df_benchmark_avg_down_magnitude = cf.average_down_magnitude(df_fund)
        df_bench1_avg_down_magnitude, df_benchmark_avg_down_magnitude = cf.average_down_magnitude(df_benchmark1)
        df_bench2_avg_down_magnitude, df_benchmark_avg_down_magnitude = cf.average_down_magnitude(df_benchmark2)
        df_bench3_avg_down_magnitude, df_benchmark_avg_down_magnitude = cf.average_down_magnitude(df_benchmark3)

        avg_up_down_magnitude_fund = pd.DataFrame({
            'Average Up Magnitude': [df_fund_avg_up_magnitude],
            'Average Down Magnitude': [df_fund_avg_down_magnitude],
        })
        avg_up_down_magnitude_bench1 = pd.DataFrame({
            'Average Up Magnitude': [df_bench1_avg_up_magnitude],
            'Average Down Magnitude': [df_bench1_avg_down_magnitude],
        })
        avg_up_down_magnitude_bench2 = pd.DataFrame({
            'Average Up Magnitude': [df_bench2_avg_up_magnitude],
            'Average Down Magnitude': [df_bench2_avg_down_magnitude],
        })
        avg_up_down_magnitude_bench3 = pd.DataFrame({
            'Average Up Magnitude': [df_bench3_avg_up_magnitude],
            'Average Down Magnitude': [df_bench3_avg_down_magnitude],
        })

        avg_up_down_magnitude_fund['Average Up Magnitude'] = pd.to_numeric(
            avg_up_down_magnitude_fund['Average Up Magnitude'])
        avg_up_down_magnitude_fund['Average Down Magnitude'] = pd.to_numeric(
            avg_up_down_magnitude_fund['Average Down Magnitude'])
        avg_up_down_magnitude_bench1['Average Up Magnitude'] = pd.to_numeric(
            avg_up_down_magnitude_bench1['Average Up Magnitude'])
        avg_up_down_magnitude_bench1['Average Down Magnitude'] = pd.to_numeric(
            avg_up_down_magnitude_bench1['Average Down Magnitude'])
        avg_up_down_magnitude_bench2['Average Up Magnitude'] = pd.to_numeric(
            avg_up_down_magnitude_bench2['Average Up Magnitude'])
        avg_up_down_magnitude_bench2['Average Down Magnitude'] = pd.to_numeric(
            avg_up_down_magnitude_bench2['Average Down Magnitude'])
        avg_up_down_magnitude_bench3['Average Up Magnitude'] = pd.to_numeric(
            avg_up_down_magnitude_bench3['Average Up Magnitude'])
        avg_up_down_magnitude_bench3['Average Down Magnitude'] = pd.to_numeric(
            avg_up_down_magnitude_bench3['Average Down Magnitude'])

        '''
        avg_up_magnitude = pd.DataFrame({'Average Up Magnitude': [df_fund_avg_up_magnitude, df_bench1_avg_up_magnitude,
                                                                  df_bench2_avg_up_magnitude,
                                                                  df_bench3_avg_up_magnitude]},
                                        index=df.columns)
        avg_up_magnitude['Average Up Magnitude'] = pd.to_numeric(avg_up_magnitude['Average Up Magnitude'])
        avg_up_magnitude_in_percentage = avg_up_magnitude['Average Up Magnitude'] * 100

        avg_down_magnitude = pd.DataFrame({'Average Down Magnitude': [df_fund_avg_down_magnitude, df_bench1_avg_down_magnitude,
                                                                  df_bench2_avg_down_magnitude,
                                                                  df_bench3_avg_down_magnitude]},
                                        index=df.columns)
        avg_down_magnitude['Average Down Magnitude'] = pd.to_numeric(avg_down_magnitude['Average Down Magnitude'])
        avg_down_magnitude_in_percentage = avg_down_magnitude['Average Down Magnitude'] * 100
        '''

        # Up Down Percentage Ratio
        df_fund_up_months, df_fund_down_months = cf.count_up_down_months(df_fund)
        df_bench1_up_months, df_bench1_down_months = cf.count_up_down_months(df_benchmark1)
        df_bench2_up_months, df_bench2_down_months = cf.count_up_down_months(df_benchmark2)
        df_bench3_up_months, df_bench3_down_months = cf.count_up_down_months(df_benchmark3)

        df_fund_up_down_months = pd.DataFrame({
            'UP and DOWN Months': [df_fund_up_months, df_fund_down_months]},
            index=['Up Months', 'Down Months'])

        df_bench1_up_down_months = pd.DataFrame({
            'UP and DOWN Months': [df_bench1_up_months, df_bench1_down_months]},
            index=['Up Months', 'Down Months'])

        df_bench2_up_down_months = pd.DataFrame({
            'UP and DOWN Months': [df_bench2_up_months, df_bench2_down_months]},
            index=['Up Months', 'Down Months'])

        df_bench3_up_down_months = pd.DataFrame({
            'UP and DOWN Months': [df_bench3_up_months, df_bench3_down_months]},
            index=['Up Months', 'Down Months'])

        sort_fund_bench1 = cf.sort_returns_by_benchmark(df_fund, df_benchmark1)
        sort_fund_bench2 = cf.sort_returns_by_benchmark(df_fund, df_benchmark2)
        sort_fund_bench3 = cf.sort_returns_by_benchmark(df_fund, df_benchmark3)

        fund_bench1_rolling_corr = cf.calculate_rolling_correlation(df_fund, df_benchmark1)
        fund_bench2_rolling_corr = cf.calculate_rolling_correlation(df_fund, df_benchmark2)
        fund_bench3_rolling_corr = cf.calculate_rolling_correlation(df_fund, df_benchmark3)

        fund_rolling_ann_return_positive_count, fund_rolling_ann_return_negative_count, fund_rolling_ann_return = cf.rolling_annualized_returns(
            df_fund.iloc[:, 0], periods=[12, 36, 60])
        bench1_rolling_ann_return_positive_count, bench1_rolling_ann_return_negative_count, bench1_rolling_ann_return = cf.rolling_annualized_returns(
            df_benchmark1.iloc[:, 0], periods=[12, 36, 60])
        bench2_rolling_ann_return_positive_count, bench2_rolling_ann_return_negative_count, bench2_rolling_ann_return = cf.rolling_annualized_returns(
            df_benchmark2.iloc[:, 0], periods=[12, 36, 60])
        bench3_rolling_ann_return_positive_count, bench3_rolling_ann_return_negative_count, bench3_rolling_ann_return = cf.rolling_annualized_returns(
            df_benchmark3.iloc[:, 0], periods=[12, 36, 60])

        fund_rolling_ann_volatility = cf.rolling_annualized_volatility(df_fund.iloc[:, 0], periods=[12, 36, 60])
        bench1_rolling_ann_volatility = cf.rolling_annualized_volatility(df_benchmark1.iloc[:, 0], periods=[12, 36, 60])
        bench2_rolling_ann_volatility = cf.rolling_annualized_volatility(df_benchmark2.iloc[:, 0], periods=[12, 36, 60])
        bench3_rolling_ann_volatility = cf.rolling_annualized_volatility(df_benchmark3.iloc[:, 0], periods=[12, 36, 60])

        fund_bench1_rolling_beta_df = cf.calculate_rolling_beta(df_fund.iloc[:, 0], df_benchmark1.iloc[:, 0])
        if not df_benchmark2.isna().any().any():
            fund_bench2_rolling_beta_df = cf.calculate_rolling_beta(df_fund.iloc[:, 0], df_benchmark2.iloc[:, 0])
        if not df_benchmark3.isna().any().any():
            fund_bench3_rolling_beta_df = cf.calculate_rolling_beta(df_fund.iloc[:, 0], df_benchmark3.iloc[:, 0])

        fund_bench1_rolling_alpha_df = cf.calculate_rolling_alpha(df_fund.iloc[:, 0], df_benchmark1.iloc[:, 0])
        if not df_benchmark2.isna().any().any():
            fund_bench2_rolling_alpha_df = cf.calculate_rolling_alpha(df_fund.iloc[:, 0], df_benchmark2.iloc[:, 0])
        if not df_benchmark3.isna().any().any():
            fund_bench3_rolling_alpha_df = cf.calculate_rolling_alpha(df_fund.iloc[:, 0], df_benchmark3.iloc[:, 0])

        scenario_analysis_df = cf.scenario_analysis_cumulative_return(df_fund.iloc[:, 0], df_benchmark1.iloc[:, 0], df_benchmark2.iloc[:, 0],
                                            df_benchmark3.iloc[:, 0])



        # Create plots
        fig_ann_return = ff.create_table(df_ann_return, index=True).update_layout(title='Fund and Benchmark Annualized Returns')
        fig_ann_summary = ff.create_table(df_ann_summary, index=True).update_layout(title='Fund Annualized Returns')

        fig_df_vol_summary = ff.create_table(df_vol_summary, index=True).update_layout(title='Fund Annualized Volatility')

        fig_rolling_ir_df_fund_bench1 = px.line(rolling_ir_df_fund_bench1,
                                                x=rolling_ir_df_fund_bench1.index,
                                                y=rolling_ir_df_fund_bench1.columns,
                                                title='Rolling 12M, 36M and 60M Ann. Information Ratio with ' +
                                                      df_benchmark1.columns[0])
        fig_rolling_ir_df_fund_bench2 = px.line(rolling_ir_df_fund_bench2,
                                                x=rolling_ir_df_fund_bench2.index,
                                                y=rolling_ir_df_fund_bench2.columns,
                                                title='Rolling 12M, 36M and 60M Ann. Information Ratio with ' +
                                                      df_benchmark2.columns[0])
        fig_rolling_ir_df_fund_bench3 = px.line(rolling_ir_df_fund_bench3,
                                                x=rolling_ir_df_fund_bench3.index,
                                                y=rolling_ir_df_fund_bench3.columns,
                                                title='Rolling 12M, 36M and 60M Ann. Information Ratio with ' +
                                                      df_benchmark3.columns[0])
        fig_rolling_ir_df_fund_bench1.update_layout(paper_bgcolor='#f0f0f0')
        fig_rolling_ir_df_fund_bench2.update_layout(paper_bgcolor='#f0f0f0')
        fig_rolling_ir_df_fund_bench3.update_layout(paper_bgcolor='#f0f0f0')

        fig_fund_rolling_ann_return = px.line(fund_rolling_ann_return, x=fund_rolling_ann_return.index,
                                              y=fund_rolling_ann_return.columns,
                                              title='Rolling 12M, 36M and 60M Annualized Return of ' +
                                                    df_fund.columns[0])
        fig_bench1_rolling_ann_return = px.line(bench1_rolling_ann_return, x=bench1_rolling_ann_return.index,
                                                y=bench1_rolling_ann_return.columns,
                                                title='Rolling 12M, 36M and 60M Annualized Return of ' +
                                                      df_benchmark1.columns[0])
        fig_bench2_rolling_ann_return = px.line(bench2_rolling_ann_return, x=bench2_rolling_ann_return.index,
                                                y=bench2_rolling_ann_return.columns,
                                                title='Rolling 12M, 36M and 60M Annualized Return of ' +
                                                      df_benchmark2.columns[0])
        fig_bench3_rolling_ann_return = px.line(bench3_rolling_ann_return, x=bench3_rolling_ann_return.index,
                                                y=bench3_rolling_ann_return.columns,
                                                title='Rolling 12M, 36M and 60M Annualized Return of ' +
                                                      df_benchmark3.columns[0])
        fig_fund_rolling_ann_return.update_layout(yaxis_tickformat='.2%', paper_bgcolor='#f0f0f0')
        fig_bench1_rolling_ann_return.update_layout(yaxis_tickformat='.2%', paper_bgcolor='#f0f0f0')
        fig_bench2_rolling_ann_return.update_layout(yaxis_tickformat='.2%', paper_bgcolor='#f0f0f0')
        fig_bench3_rolling_ann_return.update_layout(yaxis_tickformat='.2%', paper_bgcolor='#f0f0f0')

        fig_fund_rolling_ann_volatility = px.line(fund_rolling_ann_volatility, x=fund_rolling_ann_volatility.index,
                                              y=fund_rolling_ann_volatility.columns,
                                              title='Rolling 12M, 36M and 60M Annualized Volatility of ' +
                                                    df_fund.columns[0])
        fig_bench1_rolling_ann_volatility = px.line(bench1_rolling_ann_volatility, x=bench1_rolling_ann_volatility.index,
                                                  y=bench1_rolling_ann_volatility.columns,
                                                  title='Rolling 12M, 36M and 60M Annualized Volatility of ' +
                                                        df_benchmark1.columns[0])
        fig_bench2_rolling_ann_volatility = px.line(bench2_rolling_ann_volatility, x=bench2_rolling_ann_volatility.index,
                                                  y=bench2_rolling_ann_volatility.columns,
                                                  title='Rolling 12M, 36M and 60M Annualized Volatility of ' +
                                                        df_benchmark2.columns[0])
        fig_bench3_rolling_ann_volatility = px.line(bench3_rolling_ann_volatility, x=bench3_rolling_ann_volatility.index,
                                                  y=bench3_rolling_ann_volatility.columns,
                                                  title='Rolling 12M, 36M and 60M Annualized Volatility of ' +
                                                        df_benchmark3.columns[0])
        fig_fund_rolling_ann_volatility.update_layout(yaxis_tickformat='.2%', paper_bgcolor='#f0f0f0')
        fig_bench1_rolling_ann_volatility.update_layout(yaxis_tickformat='.2%', paper_bgcolor='#f0f0f0')
        fig_bench2_rolling_ann_volatility.update_layout(yaxis_tickformat='.2%', paper_bgcolor='#f0f0f0')
        fig_bench3_rolling_ann_volatility.update_layout(yaxis_tickformat='.2%', paper_bgcolor='#f0f0f0')

        fig_capture_ratio_df_benchmark1 = go.Figure()
        for col in capture_ratio_df_benchmark1.columns:
            fig_capture_ratio_df_benchmark1.add_trace(
                go.Bar(x=capture_ratio_df_benchmark1.index, y=capture_ratio_df_benchmark1[col],
                       name=col,
                       text=capture_ratio_df_benchmark1[col].apply(lambda x: f'{x:.2f}'),
                       textposition='auto'))
        fig_capture_ratio_df_benchmark1.update_layout(
            title='Upside and Downside Capture Ratio against ' + df_benchmark1.columns[0], xaxis_title='Benchmark',
            yaxis_title='Upside/Downside Capture Ratio', yaxis_tickformat='.2',
            barmode='group',
            paper_bgcolor='#f0f0f0'
        )

        if df.shape[1] == 3:
            fig_capture_ratio_df_benchmark2 = go.Figure()
            for col in capture_ratio_df_benchmark2.columns:
                fig_capture_ratio_df_benchmark2.add_trace(
                    go.Bar(x=capture_ratio_df_benchmark2.index, y=capture_ratio_df_benchmark2[col],
                           name=col,
                           text=capture_ratio_df_benchmark2[col].apply(lambda x: f'{x:.2f}'),
                           textposition='auto'))
            fig_capture_ratio_df_benchmark2.update_layout(
                title='Upside and Downside Capture Ratio against ' + df_benchmark2.columns[0], xaxis_title='Benchmark',
                yaxis_title='Upside/Downside Capture Ratio', yaxis_tickformat='.2',
                barmode='group',
                paper_bgcolor='#f0f0f0'
            )

        if df.shape[1] == 4:
            fig_capture_ratio_df_benchmark2 = go.Figure()
            for col in capture_ratio_df_benchmark2.columns:
                fig_capture_ratio_df_benchmark2.add_trace(
                    go.Bar(x=capture_ratio_df_benchmark2.index, y=capture_ratio_df_benchmark2[col],
                           name=col,
                           text=capture_ratio_df_benchmark2[col].apply(lambda x: f'{x:.2f}'),
                           textposition='auto'))
            fig_capture_ratio_df_benchmark2.update_layout(
                title='Upside and Downside Capture Ratio against ' + df_benchmark2.columns[0], xaxis_title='Benchmark',
                yaxis_title='Upside/Downside Capture Ratio', yaxis_tickformat='.2',
                barmode='group',
                paper_bgcolor='#f0f0f0'
            )
            fig_capture_ratio_df_benchmark3 = go.Figure()
            for col in capture_ratio_df_benchmark3.columns:
                fig_capture_ratio_df_benchmark3.add_trace(
                    go.Bar(x=capture_ratio_df_benchmark3.index, y=capture_ratio_df_benchmark3[col],
                           name=col,
                           text=capture_ratio_df_benchmark3[col].apply(lambda x: f'{x:.2f}'),
                           textposition='auto'))
            fig_capture_ratio_df_benchmark3.update_layout(
                title='Upside and Downside Capture Ratio against ' + df_benchmark3.columns[0], xaxis_title='Benchmark',
                yaxis_title='Upside/Downside Capture Ratio', yaxis_tickformat='.2',
                barmode='group',
                paper_bgcolor='#f0f0f0'
            )


        fig_avg_up_down_magnitude_fund = go.Figure(data=[
                                            go.Bar(
                                                name=type,
                                                x=[type],
                                                y=[value],
                                                text=[f'{value:.2%}'],
                                                textposition='auto'
                                            )
                                            for type, value in
                                            zip(avg_up_down_magnitude_fund.columns, avg_up_down_magnitude_fund.iloc[0])
                                            ])
        fig_avg_up_down_magnitude_fund.update_layout(
            title=df_fund.columns[0] + ' Average Up and Down Magnitude (by Month)',
            yaxis_title='Average Up and Down Magnitude (Monthly)',
            yaxis_tickformat='.2%',
            paper_bgcolor='#f0f0f0')

        fig_avg_up_down_magnitude_bench1 = go.Figure(data=[
            go.Bar(
                name=type,
                x=[type],
                y=[value],
                text=[f'{value:.2%}'],
                textposition='auto'
            )
            for type, value in
            zip(avg_up_down_magnitude_bench1.columns, avg_up_down_magnitude_bench1.iloc[0])
        ])
        fig_avg_up_down_magnitude_bench1.update_layout(
            title=df_benchmark1.columns[0] + ' Average Up and Down Magnitude (by Month)',
            yaxis_title='Average Up and Down Magnitude (Monthly)',
            yaxis_tickformat='.2%',
            paper_bgcolor='#f0f0f0')

        fig_avg_up_down_magnitude_bench2 = go.Figure(data=[
            go.Bar(
                name=type,
                x=[type],
                y=[value],
                text=[f'{value:.2%}'],
                textposition='auto'
            )
            for type, value in
            zip(avg_up_down_magnitude_bench2.columns, avg_up_down_magnitude_bench2.iloc[0])
        ])
        fig_avg_up_down_magnitude_bench2.update_layout(
            title=df_benchmark2.columns[0] + ' Average Up and Down Magnitude (by Month)',
            yaxis_title='Average Up and Down Magnitude (Monthly)',
            yaxis_tickformat='.2%',
            paper_bgcolor='#f0f0f0')

        fig_avg_up_down_magnitude_bench3 = go.Figure(data=[
            go.Bar(
                name=type,
                x=[type],
                y=[value],
                text=[f'{value:.2%}'],
                textposition='auto'
            )
            for type, value in
            zip(avg_up_down_magnitude_bench3.columns, avg_up_down_magnitude_bench3.iloc[0])
        ])
        fig_avg_up_down_magnitude_bench3.update_layout(
            title=df_benchmark3.columns[0] + ' Average Up and Down Magnitude (by Month)',
            yaxis_title='Average Up and Down Magnitude (Monthly)',
            yaxis_tickformat='.2%',
            paper_bgcolor='#f0f0f0')

        '''
        fig_avg_up_magnitude = go.Figure(
            data=[go.Bar(x=avg_up_magnitude.index,
                         y=avg_up_magnitude['Average Up Magnitude'],
                         text=[f"{x:.2f}%" for x in avg_up_magnitude_in_percentage],
                         textposition='outside')
                ]
        )
        fig_avg_up_magnitude.update_layout(
            title='Fund and Benchmark Average Up Magnitude (by Month)',
            xaxis_title='Fund and Benchmarks',
            yaxis_title='Average Up Magnitude (Monthly)',
            yaxis_tickformat='.2%'
        )

        fig_avg_down_magnitude = go.Figure(
            data=[go.Bar(x=avg_down_magnitude.index,
                         y=avg_down_magnitude['Average Down Magnitude'],
                         text=[f"{x:.2f}%" for x in avg_down_magnitude_in_percentage],
                         textposition='outside')
                ]
        )
        fig_avg_down_magnitude.update_layout(
            title='Fund and Benchmark Average Down Magnitude (by Month)',
            xaxis_title='Fund and Benchmarks',
            yaxis_title='Average Down Magnitude (Monthly)',
            yaxis_tickformat='.2%')
        '''

        fig_up_and_down_months_fund = go.Figure(data=[go.Pie(labels=['Up Months %', 'Down Months %'],
                                                             values=df_fund_up_down_months['UP and DOWN Months']
                                                             )
                                                      ]
                                                )
        fig_up_and_down_months_fund.update_layout(title='Up and Down Months Percentage - ' + df_fund.columns[0],
                                                  paper_bgcolor='#f0f0f0')

        fig_up_and_down_months_bench1 = go.Figure(data=[go.Pie(labels=['Up Months %', 'Down Months %'],
                                                             values=df_bench1_up_down_months['UP and DOWN Months']
                                                             )
                                                      ]
                                                )
        fig_up_and_down_months_bench1.update_layout(title='Up and Down Months Percentage - ' + df_benchmark1.columns[0],
                                                    paper_bgcolor='#f0f0f0')

        fig_up_and_down_months_bench2 = go.Figure(data=[go.Pie(labels=['Up Months %', 'Down Months %'],
                                                             values=df_bench2_up_down_months['UP and DOWN Months']
                                                             )
                                                      ]
                                                )
        fig_up_and_down_months_bench2.update_layout(title='Up and Down Months Percentage - ' + df_benchmark2.columns[0],
                                                    paper_bgcolor='#f0f0f0')

        fig_up_and_down_months_bench3 = go.Figure(data=[go.Pie(labels=['Up Months %', 'Down Months %'],
                                                             values=df_bench3_up_down_months['UP and DOWN Months']
                                                             )
                                                      ]
                                                )
        fig_up_and_down_months_bench3.update_layout(title='Up and Down Months Percentage - ' + df_benchmark3.columns[0],
                                                    paper_bgcolor='#f0f0f0')



        fund_min_return = df_fund.min()
        fund_min_return = fund_min_return.item()
        bench1_min_return = df_benchmark1.min()
        bench1_min_return = bench1_min_return.item()
        bench2_min_return = df_benchmark2.min()
        bench2_min_return = bench2_min_return.item()
        bench3_min_return = df_benchmark3.min()
        bench3_min_return = bench3_min_return.item()

        fund_max_return = df_fund.max()
        fund_max_return = fund_max_return.item()
        bench1_max_return = df_benchmark1.max()
        bench1_max_return = bench1_max_return.item()
        bench2_max_return = df_benchmark2.max()
        bench2_max_return = bench2_max_return.item()
        bench3_max_return = df_benchmark3.max()
        bench3_max_return = bench3_max_return.item()
        bin_width = 0.005

        fig_return_distribution_fund = go.Figure(data=[go.Histogram(x=df_fund.iloc[:, 0].tolist(),
                                                                    xbins=dict(start=fund_min_return,
                                                                               end=fund_max_return, size=bin_width))])
        fig_return_distribution_fund.update_layout(title='Monthly Return Distribution - ' + df_fund.columns[0],
                                                   xaxis_title='Monthly Returns', yaxis_title='Frequency',
                                                   xaxis_tickformat='%{x:.2f}%',
                                                   yaxis_tickformat=',d',
                                                   paper_bgcolor='#f0f0f0'
                                                   )
        tickvals = fig_return_distribution_fund.data[0].xbins['start'] + bin_width * (
                    0.5 + np.arange(len(df_fund.iloc[:, 0].tolist())))
        ticktext = ['{:.2f}%'.format(val * 100) for val in tickvals]
        fig_return_distribution_fund.update_xaxes(
            tickmode='array',
            tickvals=tickvals,
            ticktext=ticktext
        )


        fig_return_distribution_bench1 = go.Figure(data=[go.Histogram(x=df_benchmark1.iloc[:, 0].tolist(),
                                                                    xbins=dict(start=bench1_min_return,
                                                                               end=bench1_max_return, size=bin_width))])
        fig_return_distribution_bench1.update_layout(title='Monthly Return Distribution - ' + df_benchmark1.columns[0],
                                                   xaxis_title='Monthly Returns', yaxis_title='Frequency',
                                                   xaxis_tickformat='%{x:.2f}%',
                                                   yaxis_tickformat=',d',
                                                    paper_bgcolor='#f0f0f0'
                                                   )
        tickvals = fig_return_distribution_bench1.data[0].xbins['start'] + bin_width * (
                    0.5 + np.arange(len(df_benchmark1.iloc[:, 0].tolist())))
        ticktext = ['{:.2f}%'.format(val * 100) for val in tickvals]
        fig_return_distribution_bench1.update_xaxes(
            tickmode='array',
            tickvals=tickvals,
            ticktext=ticktext
        )

        if df_benchmark2.notna().all().all():
            fig_return_distribution_bench2 = go.Figure(data=[go.Histogram(x=df_benchmark2.iloc[:, 0].tolist(),
                                                                          xbins=dict(start=bench2_min_return,
                                                                                     end=bench2_max_return,
                                                                                     size=bin_width))])
            fig_return_distribution_bench2.update_layout(title='Monthly Return Distribution - ' + df_benchmark2.columns[0],
                                                         xaxis_title='Monthly Returns', yaxis_title='Frequency',
                                                         xaxis_tickformat='%{x:.2f}%',
                                                         yaxis_tickformat=',d',
                                                         paper_bgcolor='#f0f0f0'

                                                         )
            tickvals = fig_return_distribution_bench2.data[0].xbins['start'] + bin_width * (
                        0.5 + np.arange(len(df_benchmark2.iloc[:, 0].tolist())))
            ticktext = ['{:.2f}%'.format(val * 100) for val in tickvals]
            fig_return_distribution_bench2.update_xaxes(
                tickmode='array',
                tickvals=tickvals,
                ticktext=ticktext
            )

        if df_benchmark3.notna().all().all():
            fig_return_distribution_bench3 = go.Figure(data=[go.Histogram(x=df_benchmark3.iloc[:, 0].tolist(),
                                                                          xbins=dict(start=bench3_min_return,
                                                                                     end=bench3_max_return,
                                                                                     size=bin_width))])
            fig_return_distribution_bench3.update_layout(title='Monthly Return Distribution - ' + df_benchmark3.columns[0],
                                                         xaxis_title='Monthly Returns', yaxis_title='Frequency',
                                                         xaxis_tickformat='%{x:.2f}%',
                                                         yaxis_tickformat=',d',
                                                         paper_bgcolor='#f0f0f0'
                                                         )
            tickvals = fig_return_distribution_bench3.data[0].xbins['start'] + bin_width * (
                        0.5 + np.arange(len(df_benchmark3.iloc[:, 0].tolist())))
            ticktext = ['{:.2f}%'.format(val * 100) for val in tickvals]
            fig_return_distribution_bench3.update_xaxes(
                tickmode='array',
                tickvals=tickvals,
                ticktext=ticktext
            )

        fig_sort_fund_bench1 = go.Figure()
        fig_sort_fund_bench2 = go.Figure()
        fig_sort_fund_bench3 = go.Figure()
        fund_data_sort_by_bench1 = sort_fund_bench1[sort_fund_bench1['Fund or Benchmark'] == 'Fund']
        fund_data_sort_by_bench2 = sort_fund_bench2[sort_fund_bench2['Fund or Benchmark'] == 'Fund']
        fund_data_sort_by_bench3 = sort_fund_bench3[sort_fund_bench3['Fund or Benchmark'] == 'Fund']
        benchmark1_data = sort_fund_bench1[sort_fund_bench1['Fund or Benchmark'] == 'Benchmark']
        benchmark2_data = sort_fund_bench2[sort_fund_bench2['Fund or Benchmark'] == 'Benchmark']
        benchmark3_data = sort_fund_bench3[sort_fund_bench3['Fund or Benchmark'] == 'Benchmark']
        fig_sort_fund_bench1.add_trace(go.Bar(y=fund_data_sort_by_bench1['Return'], name=df_fund.columns[0],
                                              marker_color='blue'))
        fig_sort_fund_bench1.add_trace(go.Bar(y=benchmark1_data['Return'], name=df_benchmark1.columns[0],
                                              marker_color='orange'))
        fig_sort_fund_bench2.add_trace(go.Bar(y=fund_data_sort_by_bench2['Return'], name=df_fund.columns[0],
                                              marker_color='blue'))
        fig_sort_fund_bench2.add_trace(go.Bar(y=benchmark2_data['Return'], name=df_benchmark2.columns[0],
                                              marker_color='orange'))
        fig_sort_fund_bench3.add_trace(go.Bar(y=fund_data_sort_by_bench3['Return'], name=df_fund.columns[0],
                                              marker_color='blue'))
        fig_sort_fund_bench3.add_trace(go.Bar(y=benchmark3_data['Return'], name=df_benchmark3.columns[0],
                                              marker_color='orange'))
        fig_sort_fund_bench1.update_layout(
            title='Monthly Returns of ' + df_fund.columns[0] + ' and ' + df_benchmark1.columns[0],
            yaxis_title='Returns', yaxis_tickformat='.2%',paper_bgcolor='#f0f0f0')
        fig_sort_fund_bench2.update_layout(
            title='Monthly Returns of ' + df_fund.columns[0] + ' and ' + df_benchmark2.columns[0],
            yaxis_title='Returns', yaxis_tickformat='.2%',paper_bgcolor='#f0f0f0')
        fig_sort_fund_bench3.update_layout(
            title='Monthly Returns of ' + df_fund.columns[0] + ' and ' + df_benchmark3.columns[0],
            yaxis_title='Returns', yaxis_tickformat='.2%',paper_bgcolor='#f0f0f0')

        fig_fund_bench1_rolling_corr = px.line(fund_bench1_rolling_corr, x=fund_bench1_rolling_corr.index,
                                               y=fund_bench1_rolling_corr.columns,
                                               title='Rolling 12M, 36M and 60M Correlation between '
                                                     + df_fund.columns[0] + ' & ' + df_benchmark1.columns[0])
        fig_fund_bench1_rolling_corr.update_layout(paper_bgcolor='#f0f0f0')
        fig_fund_bench2_rolling_corr = px.line(fund_bench2_rolling_corr, x=fund_bench2_rolling_corr.index,
                                               y=fund_bench2_rolling_corr.columns,
                                               title='Rolling 12M, 36M and 60M Correlation between '
                                                     + df_fund.columns[0] + ' & ' + df_benchmark2.columns[0])
        fig_fund_bench2_rolling_corr.update_layout(paper_bgcolor='#f0f0f0')
        fig_fund_bench3_rolling_corr = px.line(fund_bench3_rolling_corr, x=fund_bench3_rolling_corr.index,
                                               y=fund_bench3_rolling_corr.columns,
                                               title='Rolling 12M, 36M and 60M Correlation between '
                                                     + df_fund.columns[0] + ' & ' + df_benchmark3.columns[0])
        fig_fund_bench3_rolling_corr.update_layout(paper_bgcolor='#f0f0f0')

        fig_fund_bench1_rolling_beta_df = px.line(fund_bench1_rolling_beta_df, x=fund_bench1_rolling_beta_df.index,
                                              y=fund_bench1_rolling_beta_df.columns,
                                              title='Rolling 12M, 36M and 60M Beta of ' +
                                                        df_fund.columns[0] + ' & ' +
                                                        df_benchmark1.columns[0])
        fig_fund_bench1_rolling_beta_df.update_layout(paper_bgcolor='#f0f0f0')
        if not df_benchmark2.isna().any().any():
            fig_fund_bench2_rolling_beta_df = px.line(fund_bench2_rolling_beta_df, x=fund_bench2_rolling_beta_df.index,
                                                      y=fund_bench2_rolling_beta_df.columns,
                                                      title='Rolling 12M, 36M and 60M Beta of ' +
                                                            df_fund.columns[0] + ' & ' +
                                                            df_benchmark2.columns[0])
            fig_fund_bench2_rolling_beta_df.update_layout(paper_bgcolor='#f0f0f0')
        if not df_benchmark3.isna().any().any():
            fig_fund_bench3_rolling_beta_df = px.line(fund_bench3_rolling_beta_df, x=fund_bench3_rolling_beta_df.index,
                                                      y=fund_bench3_rolling_beta_df.columns,
                                                      title='Rolling 12M, 36M and 60M Beta of ' +
                                                            df_fund.columns[0] + ' & ' +
                                                            df_benchmark3.columns[0])
            fig_fund_bench3_rolling_beta_df.update_layout(paper_bgcolor='#f0f0f0')

        fig_fund_bench1_rolling_alpha_df = px.line(fund_bench1_rolling_alpha_df, x=fund_bench1_rolling_alpha_df.index,
                                                  y=fund_bench1_rolling_alpha_df.columns,
                                                  title='Rolling 12M, 36M and 60M Alpha of ' +
                                                        df_fund.columns[0] + ' & ' +
                                                        df_benchmark1.columns[0])
        fig_fund_bench1_rolling_alpha_df.update_layout(yaxis_tickformat='.2%',paper_bgcolor='#f0f0f0')
        if not df_benchmark2.isna().any().any():
            fig_fund_bench2_rolling_alpha_df = px.line(fund_bench2_rolling_alpha_df, x=fund_bench2_rolling_alpha_df.index,
                                                       y=fund_bench2_rolling_alpha_df.columns,
                                                       title='Rolling 12M, 36M and 60M Alpha of ' +
                                                             df_fund.columns[0] + ' & ' +
                                                             df_benchmark2.columns[0])
            fig_fund_bench2_rolling_alpha_df.update_layout(yaxis_tickformat='.2%',paper_bgcolor='#f0f0f0')
        if not df_benchmark3.isna().any().any():
            fig_fund_bench3_rolling_alpha_df = px.line(fund_bench3_rolling_alpha_df, x=fund_bench3_rolling_alpha_df.index,
                                                       y=fund_bench3_rolling_alpha_df.columns,
                                                       title='Rolling 12M, 36M and 60M Alpha of ' +
                                                             df_fund.columns[0] + ' & ' +
                                                             df_benchmark3.columns[0])
            fig_fund_bench3_rolling_alpha_df.update_layout(yaxis_tickformat='.2%',paper_bgcolor='#f0f0f0')

        fig_scenario_analysis_df = go.Figure()
        for col in scenario_analysis_df.columns:
            fig_scenario_analysis_df.add_trace(go.Bar(x=scenario_analysis_df.index, y=scenario_analysis_df[col],
                                                      name=col,
                                                      text=scenario_analysis_df[col].apply(lambda x: f'{x*100:.2f}%'),
                                                      textposition='auto'))
        fig_scenario_analysis_df.update_layout(
            title='Scenario Analysis',xaxis_title='Events',yaxis_title='Returns',yaxis_tickformat='.2%',barmode='group',
            paper_bgcolor='#f0f0f0'
        )

        fig_drawdown = px.line(df_drawdown, x=df_drawdown.index, y=df_drawdown.columns, title='Drawdown Analysis')
        fig_drawdown.update_layout(yaxis_tickformat='.2%',paper_bgcolor='#f0f0f0')

        if not df_benchmark2.isna().any().any() and not df_benchmark3.isna().any().any():
            return [
                    html.Div([
                        html.Div(dcc.Graph(figure=fig_ann_return), style={'width': '50%', 'margin': '1px'})
                    ], style={'display': 'flex', 'justify-content': 'space-between'}),
                    html.Div([
                        html.Div(dcc.Graph(figure=fig_ann_summary), style={'width': '50%', 'margin-top': '10px'}),
                        html.Div(dcc.Graph(figure=fig_df_vol_summary), style={'width': '50%', 'margin-top': '10px'})
                    ], style={'display': 'flex', 'justify-content': 'space-between'}),
                    html.Div([
                        html.Div(dcc.Graph(figure=fig_fund_rolling_ann_return), style={'margin-top': '20px', 'width': '50%'}),
                        html.Div(dcc.Graph(figure=fig_fund_rolling_ann_volatility), style={'margin-top': '20px', 'width': '50%'})
                    ], style={'display': 'flex', 'justify-content': 'space-between'}),
                    html.Div([
                        html.Div(dcc.Graph(figure=fig_bench1_rolling_ann_return), style={'width': '50%'}),
                        html.Div(dcc.Graph(figure=fig_bench1_rolling_ann_volatility), style={'width': '50%'})
                    ], style={'display': 'flex', 'justify-content': 'space-between'}),
                    html.Div([
                        html.Div(dcc.Graph(figure=fig_bench2_rolling_ann_return), style={'width': '50%'}),
                        html.Div(dcc.Graph(figure=fig_bench2_rolling_ann_volatility), style={'width': '50%'})
                    ], style={'display': 'flex', 'justify-content': 'space-between'}),
                    html.Div([
                        html.Div(dcc.Graph(figure=fig_bench3_rolling_ann_return), style={'width': '50%'}),
                        html.Div(dcc.Graph(figure=fig_bench3_rolling_ann_volatility), style={'width': '50%'})
                    ], style={'display': 'flex', 'justify-content': 'space-between'}),
                    html.Div([
                        html.Div(dcc.Graph(figure=fig_avg_up_down_magnitude_fund), style={'width': '50%'}),
                        html.Div(dcc.Graph(figure=fig_avg_up_down_magnitude_bench1), style={'width': '50%'}),
                        html.Div(dcc.Graph(figure=fig_avg_up_down_magnitude_bench2), style={'width': '50%'}),
                        html.Div(dcc.Graph(figure=fig_avg_up_down_magnitude_bench3), style={'width': '50%'}),
                    ], style={'display': 'flex', 'justify-content': 'space-between'}),
                    html.Div([
                        html.Div(dcc.Graph(figure=fig_up_and_down_months_fund), style={'width': '50%'}),
                        html.Div(dcc.Graph(figure=fig_up_and_down_months_bench1), style={'width': '50%'}),
                        html.Div(dcc.Graph(figure=fig_up_and_down_months_bench2), style={'width': '50%'}),
                        html.Div(dcc.Graph(figure=fig_up_and_down_months_bench3), style={'width': '50%'}),
                    ], style={'display': 'flex', 'justify-content': 'space-between'}),
                    html.Div([
                        html.Div(dcc.Graph(figure=fig_return_distribution_fund), style={'width': '50%'}),
                        html.Div(dcc.Graph(figure=fig_return_distribution_bench1), style={'width': '50%'}),
                        html.Div(dcc.Graph(figure=fig_return_distribution_bench2), style={'width': '50%'}),
                        html.Div(dcc.Graph(figure=fig_return_distribution_bench3), style={'width': '50%'})
                    ], style={'display': 'flex', 'justify-content': 'space-between'}),
                    html.Div([
                        html.Div(dcc.Graph(figure=fig_sort_fund_bench1), style={'width': '50%'}),
                        html.Div(dcc.Graph(figure=fig_sort_fund_bench2), style={'width': '50%'}),
                        html.Div(dcc.Graph(figure=fig_sort_fund_bench3), style={'width': '50%'})
                    ], style={'display': 'flex', 'justify-content': 'space-between'}),
                    html.Div([
                        html.Div(dcc.Graph(figure=fig_capture_ratio_df_benchmark1), style={'width': '50%'}),
                        html.Div(dcc.Graph(figure=fig_capture_ratio_df_benchmark2), style={'width': '50%'}),
                        html.Div(dcc.Graph(figure=fig_capture_ratio_df_benchmark3), style={'width': '50%'})
                    ], style={'display': 'flex', 'justify-content': 'space-between'}),
                    html.Div([
                        html.Div(dcc.Graph(figure=fig_rolling_ir_df_fund_bench1), style={'width': '50%'}),
                        html.Div(dcc.Graph(figure=fig_rolling_ir_df_fund_bench2), style={'width': '50%'}),
                        html.Div(dcc.Graph(figure=fig_rolling_ir_df_fund_bench3), style={'width': '50%'})
                    ], style={'display': 'flex', 'justify-content': 'space-between'}),
                    html.Div([
                        html.Div(dcc.Graph(figure=fig_fund_bench1_rolling_corr), style={'width': '50%'}),
                        html.Div(dcc.Graph(figure=fig_fund_bench2_rolling_corr), style={'width': '50%'}),
                        html.Div(dcc.Graph(figure=fig_fund_bench3_rolling_corr), style={'width': '50%'})
                    ], style={'display': 'flex', 'justify-content': 'space-between'}),
                    html.Div([
                        html.Div(dcc.Graph(figure=fig_fund_bench1_rolling_beta_df), style={'width': '50%'}),
                        html.Div(dcc.Graph(figure=fig_fund_bench2_rolling_beta_df), style={'width': '50%'}),
                        html.Div(dcc.Graph(figure=fig_fund_bench3_rolling_beta_df), style={'width': '50%'})
                    ], style={'display': 'flex', 'justify-content': 'space-between'}),
                    html.Div([
                        html.Div(dcc.Graph(figure=fig_fund_bench1_rolling_alpha_df), style={'width': '50%'}),
                        html.Div(dcc.Graph(figure=fig_fund_bench2_rolling_alpha_df), style={'width': '50%'}),
                        html.Div(dcc.Graph(figure=fig_fund_bench3_rolling_alpha_df), style={'width': '50%'})
                    ], style={'display': 'flex', 'justify-content': 'space-between'}),
                    html.Div(dcc.Graph(figure=fig_scenario_analysis_df), style={'width': '100%', 'margin': '10px'}),
                    html.Div(dcc.Graph(figure=fig_drawdown), style={'width': '100%', 'margin': '10px'})
                ], f"Fund Analyzed: {fund_name}", f"Analysis Period: {fund_analysis_start_date_month}{fund_analysis_start_date_year} - {fund_analysis_end_date_month}{fund_analysis_end_date_year}"
        elif not df_benchmark2.isna().any().any() and df_benchmark3.isna().any().any():
            return [
                       html.Div([
                           html.Div(dcc.Graph(figure=fig_ann_return), style={'width': '50%', 'margin': '1px'})
                       ], style={'display': 'flex', 'justify-content': 'space-between'}),
                       html.Div([
                           html.Div(dcc.Graph(figure=fig_ann_summary), style={'width': '50%', 'margin-top': '10px'}),
                           html.Div(dcc.Graph(figure=fig_df_vol_summary), style={'width': '50%', 'margin-top': '10px'})
                       ], style={'display': 'flex', 'justify-content': 'space-between'}),
                       html.Div([
                           html.Div(dcc.Graph(figure=fig_fund_rolling_ann_return), style={'margin-top': '20px', 'width': '50%'}),
                           html.Div(dcc.Graph(figure=fig_fund_rolling_ann_volatility), style={'margin-top': '20px', 'width': '50%'})
                       ], style={'display': 'flex', 'justify-content': 'space-between'}),
                       html.Div([
                           html.Div(dcc.Graph(figure=fig_bench1_rolling_ann_return), style={'width': '50%'}),
                           html.Div(dcc.Graph(figure=fig_bench1_rolling_ann_volatility), style={'width': '50%'})
                       ], style={'display': 'flex', 'justify-content': 'space-between'}),
                       html.Div([
                           html.Div(dcc.Graph(figure=fig_bench2_rolling_ann_return), style={'width': '50%'}),
                           html.Div(dcc.Graph(figure=fig_bench2_rolling_ann_volatility), style={'width': '50%'})
                       ], style={'display': 'flex', 'justify-content': 'space-between'}),
                       html.Div([
                           html.Div(dcc.Graph(figure=fig_avg_up_down_magnitude_fund), style={'width': '50%'}),
                           html.Div(dcc.Graph(figure=fig_avg_up_down_magnitude_bench1), style={'width': '50%'}),
                           html.Div(dcc.Graph(figure=fig_avg_up_down_magnitude_bench2), style={'width': '50%'})
                       ], style={'display': 'flex', 'justify-content': 'space-between'}),
                       html.Div([
                           html.Div(dcc.Graph(figure=fig_up_and_down_months_fund), style={'width': '50%'}),
                           html.Div(dcc.Graph(figure=fig_up_and_down_months_bench1), style={'width': '50%'}),
                           html.Div(dcc.Graph(figure=fig_up_and_down_months_bench2), style={'width': '50%'})
                       ], style={'display': 'flex', 'justify-content': 'space-between'}),
                       html.Div([
                           html.Div(dcc.Graph(figure=fig_return_distribution_fund), style={'width': '50%'}),
                           html.Div(dcc.Graph(figure=fig_return_distribution_bench1), style={'width': '50%'}),
                           html.Div(dcc.Graph(figure=fig_return_distribution_bench2), style={'width': '50%'})
                       ], style={'display': 'flex', 'justify-content': 'space-between'}),
                       html.Div([
                           html.Div(dcc.Graph(figure=fig_sort_fund_bench1), style={'width': '50%'}),
                           html.Div(dcc.Graph(figure=fig_sort_fund_bench2), style={'width': '50%'})
                       ], style={'display': 'flex', 'justify-content': 'space-between'}),
                       html.Div([
                           html.Div(dcc.Graph(figure=fig_capture_ratio_df_benchmark1), style={'width': '50%'}),
                           html.Div(dcc.Graph(figure=fig_capture_ratio_df_benchmark2), style={'width': '50%'})
                       ], style={'display': 'flex', 'justify-content': 'space-between'}),
                       html.Div([
                           html.Div(dcc.Graph(figure=fig_rolling_ir_df_fund_bench1), style={'width': '50%'}),
                           html.Div(dcc.Graph(figure=fig_rolling_ir_df_fund_bench2), style={'width': '50%'})
                       ], style={'display': 'flex', 'justify-content': 'space-between'}),
                       html.Div([
                           html.Div(dcc.Graph(figure=fig_fund_bench1_rolling_corr), style={'width': '50%'}),
                           html.Div(dcc.Graph(figure=fig_fund_bench2_rolling_corr), style={'width': '50%'})
                       ], style={'display': 'flex', 'justify-content': 'space-between'}),
                       html.Div([
                           html.Div(dcc.Graph(figure=fig_fund_bench1_rolling_beta_df), style={'width': '50%'}),
                           html.Div(dcc.Graph(figure=fig_fund_bench2_rolling_beta_df), style={'width': '50%'})
                       ], style={'display': 'flex', 'justify-content': 'space-between'}),
                       html.Div([
                           html.Div(dcc.Graph(figure=fig_fund_bench1_rolling_alpha_df), style={'width': '50%'}),
                           html.Div(dcc.Graph(figure=fig_fund_bench2_rolling_alpha_df), style={'width': '50%'})
                       ], style={'display': 'flex', 'justify-content': 'space-between'}),
                       html.Div(dcc.Graph(figure=fig_scenario_analysis_df), style={'width': '100%', 'margin': '10px'}),
                       html.Div(dcc.Graph(figure=fig_drawdown), style={'width': '100%', 'margin': '10px'})
                   ], f"Fund Analyzed: {fund_name}", f"Analysis Period: {fund_analysis_start_date_month}{fund_analysis_start_date_year} - {fund_analysis_end_date_month}{fund_analysis_end_date_year}"
        elif df_benchmark2.isna().any().any() and df_benchmark3.isna().any().any():
            return [
                       html.Div([
                           html.Div(dcc.Graph(figure=fig_ann_return), style={'width': '50%', 'margin': '1px'})
                       ], style={'display': 'flex', 'justify-content': 'space-between'}),
                       html.Div([
                           html.Div(dcc.Graph(figure=fig_ann_summary), style={'width': '50%', 'margin-top': '10px'}),
                           html.Div(dcc.Graph(figure=fig_df_vol_summary), style={'width': '50%', 'margin-top': '10px'})
                       ], style={'display': 'flex', 'justify-content': 'space-between'}),
                       html.Div([
                           html.Div(dcc.Graph(figure=fig_fund_rolling_ann_return),
                                    style={'margin-top': '20px', 'width': '50%'}),
                           html.Div(dcc.Graph(figure=fig_fund_rolling_ann_volatility),
                                    style={'margin-top': '20px', 'width': '50%'})
                       ], style={'display': 'flex', 'justify-content': 'space-between'}),
                       html.Div([
                           html.Div(dcc.Graph(figure=fig_bench1_rolling_ann_return), style={'width': '50%'}),
                           html.Div(dcc.Graph(figure=fig_bench1_rolling_ann_volatility), style={'width': '50%'})
                       ], style={'display': 'flex', 'justify-content': 'space-between'}),
                       html.Div([
                           html.Div(dcc.Graph(figure=fig_avg_up_down_magnitude_fund), style={'width': '50%'}),
                           html.Div(dcc.Graph(figure=fig_avg_up_down_magnitude_bench1), style={'width': '50%'})
                       ], style={'display': 'flex', 'justify-content': 'space-between'}),
                       html.Div([
                           html.Div(dcc.Graph(figure=fig_up_and_down_months_fund), style={'width': '50%'}),
                           html.Div(dcc.Graph(figure=fig_up_and_down_months_bench1), style={'width': '50%'})
                       ], style={'display': 'flex', 'justify-content': 'space-between'}),
                       html.Div([
                           html.Div(dcc.Graph(figure=fig_return_distribution_fund), style={'width': '50%'}),
                           html.Div(dcc.Graph(figure=fig_return_distribution_bench1), style={'width': '50%'})
                       ], style={'display': 'flex', 'justify-content': 'space-between'}),
                       html.Div([
                           html.Div(dcc.Graph(figure=fig_sort_fund_bench1), style={'width': '100%'})
                       ], style={'display': 'flex', 'justify-content': 'space-between'}),
                       html.Div([
                           html.Div(dcc.Graph(figure=fig_capture_ratio_df_benchmark1), style={'width': '50%'})
                       ], style={'display': 'flex', 'justify-content': 'space-between'}),
                       html.Div([
                           html.Div(dcc.Graph(figure=fig_rolling_ir_df_fund_bench1), style={'width': '50%'})
                       ], style={'display': 'flex', 'justify-content': 'space-between'}),
                       html.Div([
                           html.Div(dcc.Graph(figure=fig_fund_bench1_rolling_corr), style={'width': '100%'})
                       ], style={'display': 'flex', 'justify-content': 'space-between'}),
                       html.Div([
                           html.Div(dcc.Graph(figure=fig_fund_bench1_rolling_beta_df), style={'width': '100%'})
                       ], style={'display': 'flex', 'justify-content': 'space-between'}),
                       html.Div([
                           html.Div(dcc.Graph(figure=fig_fund_bench1_rolling_alpha_df), style={'width': '100%'})
                       ], style={'display': 'flex', 'justify-content': 'space-between'}),
                       html.Div(dcc.Graph(figure=fig_scenario_analysis_df), style={'width': '100%', 'margin': '10px'}),
                       html.Div(dcc.Graph(figure=fig_drawdown), style={'width': '100%', 'margin': '10px'})
                   ], f"Fund Analyzed: {fund_name}", f"Analysis Period: {fund_analysis_start_date_month}{fund_analysis_start_date_year} - {fund_analysis_end_date_month}{fund_analysis_end_date_year}"


        '''
        if export_clicks:
            html_content = f"<h1>{'Fund Analysis Dashboard (by Sam CS Wong)'}</h1>"
            pdfkit.from_string(html_content, 'dashboard.pdf', configuration=pdfkit.configuration(wkhtmltopdf='C:/Users/Sam/OneDrive/Documents/Python_FundDashboard/wkhtmltopdf/bin/wkhtmltopdf.exe'))
                
            return (
                [
                html.Div(dcc.Graph(figure=fig_ann_return), style={'width': '50%', 'margin': '1px'}),
                html.Div([
                    html.Div(dcc.Graph(figure=fig_ann_summary), style={'width': '50%', 'margin-top': '10px'}),
                    html.Div(dcc.Graph(figure=fig_df_vol_summary), style={'width': '50%', 'margin-top': '10px'})
                ], style={'display': 'flex', 'justify-content': 'space-between'}),
                html.Div([
                    html.Div(dcc.Graph(figure=fig_fund_rolling_ann_return), style={'margin-top': '20px', 'width': '50%'}),
                    html.Div(dcc.Graph(figure=fig_fund_rolling_ann_volatility), style={'margin-top': '20px', 'width': '50%'})
                ], style={'display': 'flex', 'justify-content': 'space-between'}),
                html.Div([
                    html.Div(dcc.Graph(figure=fig_bench1_rolling_ann_return), style={'width': '50%'}),
                    html.Div(dcc.Graph(figure=fig_bench1_rolling_ann_volatility), style={'width': '50%'})
                ], style={'display': 'flex', 'justify-content': 'space-between'}),
                html.Div([
                    html.Div(dcc.Graph(figure=fig_bench2_rolling_ann_return), style={'width': '50%'}),
                    html.Div(dcc.Graph(figure=fig_bench2_rolling_ann_volatility), style={'width': '50%'})
                ], style={'display': 'flex', 'justify-content': 'space-between'}),
                html.Div([
                    html.Div(dcc.Graph(figure=fig_bench3_rolling_ann_return), style={'width': '50%'}),
                    html.Div(dcc.Graph(figure=fig_bench3_rolling_ann_volatility), style={'width': '50%'})
                ], style={'display': 'flex', 'justify-content': 'space-between'}),
                html.Div([
                    html.Div(dcc.Graph(figure=fig_avg_up_down_magnitude_fund), style={'width': '50%'}),
                    html.Div(dcc.Graph(figure=fig_avg_up_down_magnitude_bench1), style={'width': '50%'}),
                    html.Div(dcc.Graph(figure=fig_avg_up_down_magnitude_bench2), style={'width': '50%'}),
                    html.Div(dcc.Graph(figure=fig_avg_up_down_magnitude_bench3), style={'width': '50%'}),
                ], style={'display': 'flex', 'justify-content': 'space-between'}),
                html.Div([
                    html.Div(dcc.Graph(figure=fig_up_and_down_months_fund), style={'width': '50%'}),
                    html.Div(dcc.Graph(figure=fig_up_and_down_months_bench1), style={'width': '50%'}),
                    html.Div(dcc.Graph(figure=fig_up_and_down_months_bench2), style={'width': '50%'}),
                    html.Div(dcc.Graph(figure=fig_up_and_down_months_bench3), style={'width': '50%'}),
                ], style={'display': 'flex', 'justify-content': 'space-between'}),
                html.Div([
                    html.Div(dcc.Graph(figure=fig_return_distribution_fund), style={'width': '50%'}),
                    html.Div(dcc.Graph(figure=fig_return_distribution_bench1), style={'width': '50%'}),
                    html.Div(dcc.Graph(figure=fig_return_distribution_bench2), style={'width': '50%'}),
                    html.Div(dcc.Graph(figure=fig_return_distribution_bench3), style={'width': '50%'})
                ], style={'display': 'flex', 'justify-content': 'space-between'}),
                html.Div([
                    html.Div(dcc.Graph(figure=fig_sort_fund_bench1), style={'width': '50%'}),
                    html.Div(dcc.Graph(figure=fig_sort_fund_bench2), style={'width': '50%'}),
                    html.Div(dcc.Graph(figure=fig_sort_fund_bench3), style={'width': '50%'})
                ], style={'display': 'flex', 'justify-content': 'space-between'}),
                html.Div([
                    html.Div(dcc.Graph(figure=fig_fund_bench1_rolling_corr), style={'width': '50%'}),
                    html.Div(dcc.Graph(figure=fig_fund_bench2_rolling_corr), style={'width': '50%'}),
                    html.Div(dcc.Graph(figure=fig_fund_bench3_rolling_corr), style={'width': '50%'})
                ], style={'display': 'flex', 'justify-content': 'space-between'}),
                html.Div([
                    html.Div(dcc.Graph(figure=fig_fund_bench1_rolling_beta_df), style={'width': '50%'}),
                    html.Div(dcc.Graph(figure=fig_fund_bench2_rolling_beta_df), style={'width': '50%'}),
                    html.Div(dcc.Graph(figure=fig_fund_bench3_rolling_beta_df), style={'width': '50%'})
                ], style={'display': 'flex', 'justify-content': 'space-between'}),
                html.Div([
                    html.Div(dcc.Graph(figure=fig_fund_bench1_rolling_alpha_df), style={'width': '50%'}),
                    html.Div(dcc.Graph(figure=fig_fund_bench2_rolling_alpha_df), style={'width': '50%'}),
                    html.Div(dcc.Graph(figure=fig_fund_bench3_rolling_alpha_df), style={'width': '50%'})
                ], style={'display': 'flex', 'justify-content': 'space-between'}),
                html.Div(dcc.Graph(figure=fig_scenario_analysis_df), style={'width': '100%', 'margin': '10px'}),
                html.Div(dcc.Graph(figure=fig_drawdown), style={'width': '100%', 'margin': '10px'})
                ],
                f"Fund Analyzed: {fund_name}",\
                f"Analysis Period: {fund_analysis_start_date_month}{fund_analysis_start_date_year} - {fund_analysis_end_date_month}{fund_analysis_end_date_year}", \
                "Dashboard exported to PDF successfully!"
            )
        else:
            return (
                [
                    html.Div(dcc.Graph(figure=fig_ann_return), style={'width': '50%', 'margin': '1px'}),
                    html.Div([
                        html.Div(dcc.Graph(figure=fig_ann_summary), style={'width': '50%', 'margin-top': '10px'}),
                        html.Div(dcc.Graph(figure=fig_df_vol_summary), style={'width': '50%', 'margin-top': '10px'})
                    ], style={'display': 'flex', 'justify-content': 'space-between'}),
                    html.Div([
                        html.Div(dcc.Graph(figure=fig_fund_rolling_ann_return),
                                 style={'margin-top': '20px', 'width': '50%'}),
                        html.Div(dcc.Graph(figure=fig_fund_rolling_ann_volatility),
                                 style={'margin-top': '20px', 'width': '50%'})
                    ], style={'display': 'flex', 'justify-content': 'space-between'}),
                    html.Div([
                        html.Div(dcc.Graph(figure=fig_bench1_rolling_ann_return), style={'width': '50%'}),
                        html.Div(dcc.Graph(figure=fig_bench1_rolling_ann_volatility), style={'width': '50%'})
                    ], style={'display': 'flex', 'justify-content': 'space-between'}),
                    html.Div([
                        html.Div(dcc.Graph(figure=fig_bench2_rolling_ann_return), style={'width': '50%'}),
                        html.Div(dcc.Graph(figure=fig_bench2_rolling_ann_volatility), style={'width': '50%'})
                    ], style={'display': 'flex', 'justify-content': 'space-between'}),
                    html.Div([
                        html.Div(dcc.Graph(figure=fig_bench3_rolling_ann_return), style={'width': '50%'}),
                        html.Div(dcc.Graph(figure=fig_bench3_rolling_ann_volatility), style={'width': '50%'})
                    ], style={'display': 'flex', 'justify-content': 'space-between'}),
                    html.Div([
                        html.Div(dcc.Graph(figure=fig_avg_up_down_magnitude_fund), style={'width': '50%'}),
                        html.Div(dcc.Graph(figure=fig_avg_up_down_magnitude_bench1), style={'width': '50%'}),
                        html.Div(dcc.Graph(figure=fig_avg_up_down_magnitude_bench2), style={'width': '50%'}),
                        html.Div(dcc.Graph(figure=fig_avg_up_down_magnitude_bench3), style={'width': '50%'}),
                    ], style={'display': 'flex', 'justify-content': 'space-between'}),
                    html.Div([
                        html.Div(dcc.Graph(figure=fig_up_and_down_months_fund), style={'width': '50%'}),
                        html.Div(dcc.Graph(figure=fig_up_and_down_months_bench1), style={'width': '50%'}),
                        html.Div(dcc.Graph(figure=fig_up_and_down_months_bench2), style={'width': '50%'}),
                        html.Div(dcc.Graph(figure=fig_up_and_down_months_bench3), style={'width': '50%'}),
                    ], style={'display': 'flex', 'justify-content': 'space-between'}),
                    html.Div([
                        html.Div(dcc.Graph(figure=fig_return_distribution_fund), style={'width': '50%'}),
                        html.Div(dcc.Graph(figure=fig_return_distribution_bench1), style={'width': '50%'}),
                        html.Div(dcc.Graph(figure=fig_return_distribution_bench2), style={'width': '50%'}),
                        html.Div(dcc.Graph(figure=fig_return_distribution_bench3), style={'width': '50%'})
                    ], style={'display': 'flex', 'justify-content': 'space-between'}),
                    html.Div([
                        html.Div(dcc.Graph(figure=fig_sort_fund_bench1), style={'width': '50%'}),
                        html.Div(dcc.Graph(figure=fig_sort_fund_bench2), style={'width': '50%'}),
                        html.Div(dcc.Graph(figure=fig_sort_fund_bench3), style={'width': '50%'})
                    ], style={'display': 'flex', 'justify-content': 'space-between'}),
                    html.Div([
                        html.Div(dcc.Graph(figure=fig_fund_bench1_rolling_corr), style={'width': '50%'}),
                        html.Div(dcc.Graph(figure=fig_fund_bench2_rolling_corr), style={'width': '50%'}),
                        html.Div(dcc.Graph(figure=fig_fund_bench3_rolling_corr), style={'width': '50%'})
                    ], style={'display': 'flex', 'justify-content': 'space-between'}),
                    html.Div([
                        html.Div(dcc.Graph(figure=fig_fund_bench1_rolling_beta_df), style={'width': '50%'}),
                        html.Div(dcc.Graph(figure=fig_fund_bench2_rolling_beta_df), style={'width': '50%'}),
                        html.Div(dcc.Graph(figure=fig_fund_bench3_rolling_beta_df), style={'width': '50%'})
                    ], style={'display': 'flex', 'justify-content': 'space-between'}),
                    html.Div([
                        html.Div(dcc.Graph(figure=fig_fund_bench1_rolling_alpha_df), style={'width': '50%'}),
                        html.Div(dcc.Graph(figure=fig_fund_bench2_rolling_alpha_df), style={'width': '50%'}),
                        html.Div(dcc.Graph(figure=fig_fund_bench3_rolling_alpha_df), style={'width': '50%'})
                    ], style={'display': 'flex', 'justify-content': 'space-between'}),
                    html.Div(dcc.Graph(figure=fig_scenario_analysis_df), style={'width': '100%', 'margin': '10px'}),
                    html.Div(dcc.Graph(figure=fig_drawdown), style={'width': '100%', 'margin': '10px'})
                ],
                f"Fund Analyzed: {fund_name}",
                f"Analysis Period: {fund_analysis_start_date_month}{fund_analysis_start_date_year} - {fund_analysis_end_date_month}{fund_analysis_end_date_year}",
                ""
            )
            '''
    else:
        # return [], "", "", ""
        return [], "", ""

if __name__ == '__main__':
    app.run_server(debug=True)
