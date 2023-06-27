
import gradio as gr

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import plotly.graph_objs as go


# load datasets
predictions = pd.read_csv("../data/processed/predictions.csv")
optimal_prices = pd.read_csv("../data/processed/optimal_prices.csv")
    
unique_skus = sorted(predictions['SKU'].unique())

def plot_demand_curve_gradio(sku):
    # Load the predictions dataframe
    #predictions = pd.read_csv('../data/processed/predictions.csv')

    # Select the predictions for the given SKU
    sku_predictions = predictions[predictions['SKU'] == sku]

    # Create a scatter plot of the predicted sales against the price points
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sku_predictions['price'], y=sku_predictions['predicted_sales_xgboost'], mode='lines+markers'))


    # Set the plot title and axis labels
    fig.update_layout(title=f'Demand Curve for SKU {sku}', xaxis_title='Price', yaxis_title='Predicted Sales')
    fig.update_xaxes(title_font=dict(size=18), tickfont=dict(size=14))
    fig.update_yaxes(title_font=dict(size=18), tickfont=dict(size=14))

    # Set the legend font size and position
    fig.update_layout(legend=dict(font=dict(size=14), yanchor='top', y=0.99, xanchor='left', x=0.01), legend_title_text='SKU')

    # Set the plot background color and grid lines
    fig.update_layout(plot_bgcolor='white', xaxis=dict(showgrid=True, gridcolor='lightgray'), yaxis=dict(showgrid=True, gridcolor='lightgray'))


    # Return the plot as an HTML div
    return gr.update(value=fig, visible=True)

def plot_optimal_prices_gradio(sku):
    # Load the optimal prices dataframe
    #optimal_prices = pd.read_csv('../data/processed/optimal_prices.csv')

    # Select the optimal prices for the given SKU
    sku_optimal_prices = optimal_prices[optimal_prices['SKU'] == sku]

    # Create a bar chart of the optimal prices at different price points
    fig = go.Figure()
    fig.add_trace(go.Bar(x=['Base Price', '-10%', '-15%', '+10%', '+15%'], y=sku_optimal_prices['optimal_prices'], name=sku))

    # Set the plot title and axis labels
    fig.update_layout(title=f'Optimal Prices for SKU {sku}', xaxis_title='Price Point', yaxis_title='Optimal Price', font=dict(size=14))
    fig.update_xaxes(title_font=dict(size=18), tickfont=dict(size=14))
    fig.update_yaxes(title_font=dict(size=18), tickfont=dict(size=14))

    # Set the plot background color and grid lines
    fig.update_layout(plot_bgcolor='white', xaxis=dict(showgrid=True, gridcolor='lightgray'), yaxis=dict(showgrid=True, gridcolor='lightgray'))

    # Return the plot as an HTML div
    return gr.update(value=fig, visible=True)




with gr.Blocks() as demo:
    gr.Markdown("""
    # Relu Demand Curve & Optimal Pricing Dashboard 💰 #
    
   Version: 1.0.0
    
    Enter Values below then click Predict and Pricing Buttons to see results.  
    
    """)
    with gr.Row():
        with gr.Column():           
            sku = gr.Dropdown(
                label="Select SKU",
                choices=unique_skus,
                value=lambda: random.choice(unique_skus),
            )
            

        with gr.Column():

            plot = gr.Plot()
            with gr.Row():
                predict_btn = gr.Button(value="Predict")
            predict_btn.click(
                plot_demand_curve_gradio,
                inputs=[sku],

                outputs= plot

            )
        with gr.Column():

            plot = gr.Plot()
            with gr.Row():

                optimal_price_btn = gr.Button(value="Optimal Price")
                optimal_price_btn.click(
                    plot_optimal_prices_gradio,
                    inputs=[sku],
                    outputs=plot

                )


demo.launch(share=False)

