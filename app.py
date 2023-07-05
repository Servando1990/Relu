# load datasets
import pandas as pd
import gradio as gr
import pandas as pd

import random
import plotly.graph_objs as go


predictions = pd.read_csv("../data/processed/predictions.csv")
optimal_prices = pd.read_csv("../data/processed/optimal_prices.csv")
amazon = pd.read_csv("../data/raw/Amazon_Sale_Report.csv")
pl = pd.read_csv("../data/raw/PLMarch2021.csv")

predictions = predictions.merge(amazon[['SKU', 'Category']], on='SKU', how='left')

# rename sku column to match optimal prices
pl.rename(columns={'Sku': 'SKU'}, inplace=True)

# write and an underscore on whitespace for every column in pl dataframe
pl.columns = pl.columns.str.replace(' ', '_')

pl.Category.rename({'Kurta': 'kurta', 'Kurta Set':'kurta', 'Gown':'Ethnic Dress', 'Tops':'Set','Nill':'Top' }, inplace=True)


competitor_columns = ['Amazon_MRP', 'Flipkart_MRP', 'Ajio_MRP', 'Limeroad_MRP', 'Myntra_MRP', 'Paytm_MRP', 'Snapdeal_MRP']
from src.data.utils import Utils
# convert to int the competitor_columns in the pl dataframe
for col in competitor_columns:
    pl[col] = pl[col].apply(Utils.convert_to_int)


    
unique_skus = sorted(predictions['SKU'].unique())
unique_category = sorted(pl['Category'].unique())

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

def plot_competitors_prices_gradio(category):

    sku_competitors_results = pl[pl['Category'] == category]


    # Calculate average prices for each competitor within the selected category
    competitor_columns = ['Amazon_MRP', 'Flipkart_MRP', 'Ajio_MRP', 'Limeroad_MRP', 'Myntra_MRP', 'Paytm_MRP', 'Snapdeal_MRP']
    avg_prices = [sku_competitors_results[col].median() for col in competitor_columns]

    # Create a bar chart of the average competitors prices
    fig = go.Figure()
    fig.add_trace(go.Bar(x=competitor_columns, y=avg_prices, name=category))

    # set the plot title and axis labels
    fig.update_layout(title=f'Competitors Prices for Category {category}', xaxis_title='Competitor', yaxis_title='Median Price', font=dict(size=14))
    fig.update_xaxes(title_font=dict(size=18), tickfont=dict(size=14))
    fig.update_yaxes(title_font=dict(size=18), tickfont=dict(size=14))

    # Set the plot background color and grid lines
    fig.update_layout(plot_bgcolor='white', xaxis=dict(showgrid=True, gridcolor='lightgray'), yaxis=dict(showgrid=True, gridcolor='lightgray'))

    # return the plot as an HTML div
    return gr.update(value=fig, visible=True)




with gr.Blocks() as demo:
    gr.Markdown("""
    # Relu Demand Curve & Optimal Pricing Dashboard 💰 #
    
   Version: 1.0.0
    
    Enter Values below then click Predict and Pricing Buttons to see results.  
    
    """)
    with gr.Row(equal_height=True):
      
        sku = gr.Dropdown(
            label="Select SKU",
            choices=unique_skus,
            value=lambda: random.choice(unique_skus),
        )

    with gr.Row(equal_height=True):

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
        # return plot_comeptitors_prices_gradio(sku) as a plot for every sku
    with gr.Row(equal_height=True):
            category = gr.Dropdown(
            label="Select Category",
            choices=unique_category,
            value=lambda: random.choice(unique_category))


            with gr.Column():
                plot = gr.Plot()
                with gr.Row():
                    competitor_btn = gr.Button(value="Competitors Prices")
                    competitor_btn.click(
                        plot_competitors_prices_gradio,
                        inputs=[category],
                        outputs=plot)



demo.launch(share=False)

