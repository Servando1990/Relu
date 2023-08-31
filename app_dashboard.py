# create a function that uutputs todays date
from turtle import up
import pandas as pd
# import datetime module
import datetime
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import gradio as gr
import random


# create a function that outputs todays date in the format of "Monday, January 10th 2020 at 1:15pm"
def get_date():
    # get todays date
    date = datetime.datetime.now()
    # format the date
    date = date.strftime("%A, %B %dth %Y at %I:%M%p")
    # return the date
    return date

current_date = get_date()



def this_month_fn(df, metric):
    start_date = datetime.datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    end_date = datetime.datetime.now()
    return df[(df['date'] >= start_date) & (df['date'] <= end_date)][['date', 'channel', metric]]


def this_week_fn(df, metric):
    start_date = datetime.datetime.now() - datetime.timedelta(days=datetime.datetime.now().weekday())
    start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = datetime.datetime.now()
    return df[(df['date'] >= start_date) & (df['date'] <= end_date)][['date', 'channel', metric]]



def last_month_fn(df, metric):
    start_date = datetime.datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0) - datetime.timedelta(days=1)
    start_date = start_date.replace(day=1)
    end_date = datetime.datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0) - datetime.timedelta(seconds=1)
    return df[(df['date'] >= start_date) & (df['date'] <= end_date)][['date', 'channel', metric]]

def prev_month_fn(df, metric):
    start_date = datetime.datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0) - datetime.timedelta(days=1)
    start_date = start_date.replace(day=1)
    end_date = datetime.datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0) - datetime.timedelta(seconds=1)
    return df[(df['date'] >= start_date) & (df['date'] <= end_date)][['date', 'channel', metric]]

def last_week_fn(df, metric):
    start_date = datetime.datetime.now() - datetime.timedelta(days=datetime.datetime.now().weekday() + 7)
    start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = start_date + datetime.timedelta(days=6, hours=23, minutes=59, seconds=59)
    return df[(df['date'] >= start_date) & (df['date'] <= end_date)][['date', 'channel', metric]]

def prev_week_fn(df, metric):
    start_date = datetime.datetime.now() - datetime.timedelta(days=datetime.datetime.now().weekday() + 7)
    start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = start_date + datetime.timedelta(days=6, hours=23, minutes=59, seconds=59)
    return df[(df['date'] >= start_date) & (df['date'] <= end_date)][['date', 'channel', metric]]



date_rng = pd.date_range(start='2023-01-01', end='2023-08-18', freq='D')
channels = ["deplay.nl", "amazon.com", "bol.nl"]


def generate_dataset(channels, date_rng):
    np.random.seed(42)
    revenue = []
    channel_data = []
    traffic_data = []
    conversion_rate = []
    average_check = []

    traffic_multipliers = {
        "deplay.nl": 1.5,
        "amazon.com": 1.2,
        "bol.nl": 1.3
    }

    for i in range(len(date_rng)):
        base_revenue = 100 + i * 2
        daily_fluctuation = np.sin(i / 7) * 20
        random_noise = np.random.normal(0, 10)
        base_traffic = 1000 + i * 5

        for ch in channels:
            base_revenue = 100 + i * 2 + np.random.randint(-10, 10) # Base revenue with random variation
            daily_fluctuation = np.sin(i / 7 + np.random.random()) * 20 # Fluctuation with random phase
            daily_fluctuation_conversion_rate = np.sin(i / 7 + np.random.random()) * 0.05
            daily_fluctuation_average_check = np.sin(i / 7 + np.random.random()) * 5
            random_noise = np.random.normal(0, 10) # Random noise
            daily_revenue = base_revenue + daily_fluctuation + random_noise
            daily_traffic = base_traffic * traffic_multipliers[ch] + np.sin(i / 5 + np.random.random()) * 50 + np.random.normal(0, 20) # Traffic with fluctuations
            daily_conversion_rate = 0.01 + daily_fluctuation_conversion_rate + np.random.normal(0, 0.005)
            daily_average_check = 50 + daily_fluctuation_average_check + np.random.normal(0, 10)
            revenue.append(daily_revenue)
            channel_data.append(ch)
            traffic_data.append(daily_traffic)
            conversion_rate.append(daily_conversion_rate)
            average_check.append(daily_average_check)

    data = {
        'date': [date for date in date_rng for _ in channels],
        'channel': channel_data,
        'revenue': revenue,
        'traffic': traffic_data,
        'conversion_rate': conversion_rate,
        'average check': average_check
    }

    df = pd.DataFrame(data)

    # Derive profit from revenue assuming 20% profit margin
    df['profit'] = df['revenue'] * 0.2
    # Derive sales from revenue using a variable conversion factor for each channel
    conversion_factors = {
        "deplay.nl": 1.2,
        "amazon.com": 1.15,
        "bol.nl": 1.3
    }
    sales = [revenue[i] * conversion_factors[channel_data[i]] + np.random.normal(0, 5) for i in range(len(revenue))]

    # Add the sales column to the data dictionary
    df['sales'] = sales

    return df

df = generate_dataset(channels, date_rng)


current_time_fn = this_month_fn  # Default to this month
current_time_frame = "This Month"  # Default time frame label
current_channel = None  # Default to all channels

channel_colors = {
    "deplay.nl": "red",
    "amazon.com": "yellow",
    "bol.nl": "green"
}

# Create a fucntion called calculate_metrics

def calculate_metrics(df, metric, current_time_fn, prev_time_fn, current_channel=None):
    current_values = current_time_fn(df, metric)
    #print(f"Current Values ({metric}):")
    #print(current_values)
    previous_values = prev_time_fn(df, metric)
    #print(f"Previous Values ({metric}):")
    #print(previous_values)

    if current_channel:
        current_values = current_values[current_values['channel'] == current_channel]
        previous_values = previous_values[previous_values['channel'] == current_channel]

    current_sum = round(current_values[metric].sum(), 2)
    previous_sum = round(previous_values[metric].sum(), 2)
    #print(f"Current Sum ({metric}): {current_sum}")
    #print(f"Previous Sum ({metric}): {previous_sum}")

    if previous_sum != 0:
        percentage_change = round((current_sum - previous_sum) / previous_sum * 100, 2)
    else:
        percentage_change = 0

    return current_sum, percentage_change


# Mapping time functions to their labels
time_frame_to_function_mapping = {
    "This Month": this_month_fn,
    "This Week": this_week_fn,
    "Last Week": last_week_fn,
    "Last Month": last_month_fn
}

# Mapping of current time frame functions to previous time frame functions
current_to_prev_time_fn_mapping = {
    this_month_fn: last_month_fn,
    this_week_fn: last_week_fn,
    last_week_fn: prev_week_fn,
    last_month_fn: prev_month_fn
}


def plot_metric(df, metric, time_frame=None, channel=None):

    current_time_fn = time_frame_to_function_mapping[time_frame]
    prev_time_fn = current_to_prev_time_fn_mapping[current_time_fn]
    current_sum, percentage_change = calculate_metrics(df, metric, current_time_fn, prev_time_fn, channel)

    title = ""

    if metric == "traffic":
        value_str = f"{current_sum:,.0f}"
        percentage_color = "green" if percentage_change >= 0 else "red"
        percentage_str = " sessions {:+.2f}%".format(percentage_change)
        title = f"<b style='font-size: 20px;'>{metric.title()}: {value_str}  {percentage_str} </b> "
    else:

        value_str = f"€{current_sum:,.2f}"
        percentage_color = "green" if percentage_change >= 0 else "red"
        percentage_str = f"<span style='color: {percentage_color};'>{percentage_change:+.2f}%</span>"
        title = f"<b style='font-size: 20px;'>{metric.title()}: {value_str}  {percentage_str} </b> "


    fig = go.Figure()
    
    if channel:
        df_channel = df[df['channel'] == channel]
        line_color = channel_colors[channel]
        fig.add_trace(go.Scatter(x=df['date'], y=df_channel[metric], mode='lines', name=channel, line_color=line_color ))
    else:
        for ch in channels:
            df_channel = df[df['channel'] == ch]
            line_color = channel_colors[ch]
            fig.add_trace(go.Scatter(x=df_channel['date'], y=df_channel[metric], mode='lines', name=ch, line_color=line_color))
    
    if time_frame:
        title += f"for {time_frame} "
    if channel:
        title += f"in {channel}"
       
    fig.update_layout(title={
        'text': title,
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'}
     ),
    

    
    return gr.update(value=fig, visible=True)



def update_plot(metric):
    time_frame = current_time_frame
    channel = current_channel
    #revenue_plot = plot_metric(df, "revenue", time_frame, channel)
    #profit_plot = plot_metric(df, "profit", time_frame, channel) # Assuming the profit column exists

    return plot_metric(df, metric, time_frame, channel)


def plot_quadrant():
        # Labels and their coordinates
    labels = ["Amazon", "Samsung", "Apple", "Walmart", "You"]
    x_values = [random.uniform(-10, 10) for _ in range(4)] + [0]  # 'You' is at the origin (0, 0)
    y_values = [random.uniform(-10, 10) for _ in range(4)] + [0]  # 'You' is at the origin (0, 0)

    # Different sizes for the markers
    marker_sizes = [10, 20, 30, 40, 50]

    # Different colors for the markers
    marker_colors = ['red', 'green', 'blue', 'purple', 'orange']

    # Create the figure
    fig = go.Figure()

    # Add scatter plot for the data points
    fig.add_trace(go.Scatter(
        x=x_values,
        y=y_values,
        mode='markers+text',
        marker=dict(size=marker_sizes, color=marker_colors),
        text=labels,
        textposition="top center"
    ))

    # Draw lines for the X and Y axes to divide the plot into 4 quadrants
    fig.add_shape(
        go.layout.Shape(type='line', x0=0, x1=0, y0=-10, y1=10,
                        line=dict(color='Grey', width=1))
    )

    fig.add_shape(
        go.layout.Shape(type='line', x0=-10, x1=10, y0=0, y1=0,
                        line=dict(color='Grey', width=1))
    )

    # Set axis labels
    fig.update_layout(
        xaxis_title='Better Stock',
        yaxis_title='Better Pricing',
        xaxis=dict(range=[-10, 10]),
        yaxis=dict(range=[-10, 10])
    )

    # Show the plot
    #fig.show()
    return gr.update(value=fig, visible=True)

def select_time_frame(time_fn, time_frame):
    global current_time_fn, current_time_frame
    current_time_fn = time_fn
    current_time_frame = time_frame
    return update_plot('revenue'), update_plot('profit'), update_plot('sales'), update_plot('traffic'), update_plot('conversion_rate'), update_plot('average check')

def select_channel(channel):
    global current_channel
    current_channel = channel
    return update_plot('revenue'), update_plot('profit'), update_plot('sales'), update_plot('traffic'), update_plot('conversion_rate'), update_plot('average check')

def reset_plot():
    global current_time_fn, current_time_frame, current_channel
    current_time_fn = this_month_fn
    current_time_frame = "This Month"
    current_channel = None
    return update_plot('revenue'), update_plot('profit'), update_plot('sales'), update_plot('traffic'), update_plot('conversion_rate'), update_plot('average check')


def same_auth(username, password):
    return username == password

# Create DataFrame
data = {
    "Category": ["phones", "earphones", "laptops", "monitors", "speakers"],
    "Index": [0.98]*5,
    "Amazon": [0.98]*5,
    "Walmart": [0.96]*5,
    "Samsung": [1.08]*5,
    "Apple": [1.01]*5
}

#df_index = pd.DataFrame(data)

# Style function to color cells
def color_cells(val):
    color = 'green' if val < 1 else 'red'
    return 'background-color: %s' % color

# Apply styling function
#styled_df = df_index.style.applymap(color_cells, subset=df.columns.difference(['Category', 'Index']))

# Display or save styled DataFrame
# Convert styled DataFrame to HTML
#html = styled_df.render()


css = """
#deplay {background-color: #FFAAAA; color: #333; font-weight: bold;}
#amazon {background-color: #FFFFAA; color: #333; font-weight: bold;}
#bol {background-color: #AAFFAA; color: #333; font-weight: bold;}
"""


with gr.Blocks(theme= gr.themes.Soft(), css=css,) as demo:
    gr.HTML(f"""
    <div style="font-size: 24px; font-weight: bold;">
        <span>Deplay Dashboard</span>
        <span style="margin-left: 10px; color: #007bff;">{current_date}</span>
    </div>

    """)
    with gr.Row(equal_height=True):
        this_month_button = gr.Button(size = "lg", value= "This Month")
        this_week_button = gr.Button(size = "lg", value= "This  Week")
        last_week_button = gr.Button(size = "lg", value= "Last week")
        last_month_button = gr.Button(size = "lg", value= "Last Month")

    gr.HTML(f"""
    <div style="font-size: 24px; font-weight: bold;">
            <span>Marketplaces</span>""")
    with gr.Row(equal_height=True, css=css):
        deplay_button = gr.Button(size="sm", value="deplay.nl", elem_id='deplay', elem_classes='feedback')
        bol_button = gr.Button(size="sm", value="bol.nl", elem_id='bol', elem_classes='feedback')    
        amazon_button = gr.Button(size="sm", value="amazon.com", elem_id='amazon', elem_classes='feedback')
    

    #plot = gr.Plot()
    #plot.update(value=update_plot())
    with gr.Row():
        revenue_plot = gr.Plot()
        revenue_plot.update(value = update_plot("revenue"))
        profit_plot = gr.Plot()
        profit_plot.update(value = update_plot("profit"))
        sales_plot = gr.Plot()
        sales_plot.update(value = update_plot("sales"))
    #with gr.Column():
        #with gr.Row():

    with gr.Column():
        with gr.Row():
            conversion_rate_plot = gr.Plot()
            conversion_rate_plot.update(value = update_plot("conversion_rate"))
            average_check_plot = gr.Plot()
            average_check_plot.update(value = update_plot("average check"))
            traffic_plot = gr.Plot()
            traffic_plot.update(value = update_plot("traffic"))

    with gr.Column():
        with gr.Row():
            reset_button = gr.Button(size="sm", value="Reset")
            reset_button.click(reset_plot, 
                               outputs=[revenue_plot,
                                         profit_plot,
                                         sales_plot,
                                         traffic_plot,
                                         conversion_rate_plot,
                                         average_check_plot])
    with gr.Column(equal_height=True):
        with gr.Row(equal_height=True):

            gr.HTML("""
            <div style="font-size: 24px; font-weight: bold; text-align: center; margin: auto;">
                <h1>Import Results</h1>
                <div>Scanned SKU: <span style="font-size: larger; font-weight: bold;">23</span></div>
                <div>Error SKU: <span style="font-size: larger; font-weight: bold;">32</span></div>
                <div>Duplicated SKU: <span style="font-size: larger; font-weight: bold;">45</span></div>
            </div>
            """)

    #with gr.Column(equal_height=True):
        with gr.Row(equal_height=True):
            gr.Markdown("""# Need more Metrics? """)
        with gr.Row():
            gr.Textbox(label="Metrics", placeholder= 'Submit a metrics here...')
            gr.Button(size="sm", value="Submit")
    with gr.Column(equal_height=True):
        quadrant_plot = gr.Plot()
        quadrant_button = gr.Button(size="sm", value="Quadrant")
        quadrant_button.click(plot_quadrant, outputs=[quadrant_plot])
        #plot_quadrant.update(value=quadrant_plot)
    #with gr.Column(equal_height=True):
       # with gr.Row(equal_height=True):
            #gr.HTML("""
            #<div style="font-size: 24px; font-weight: bold; text-align: center; margin: auto;">
             #   <h1>Styled DataFrame</h1>
            #</div>
            #""" + html)  # Add DataFrame HTML here
        

        
        

    # Timeframe buttons
    this_month_button.click(lambda: select_time_frame(this_month_fn, "This Month"),
                             outputs=[revenue_plot, 
                                      profit_plot,
                                      sales_plot,
                                      traffic_plot,
                                      conversion_rate_plot,
                                      average_check_plot])
    this_week_button.click(lambda: select_time_frame(this_week_fn, "This Week"), 
                           outputs=[revenue_plot,
                                     profit_plot,
                                     sales_plot,
                                     traffic_plot,
                                     conversion_rate_plot,
                                     average_check_plot])
    last_week_button.click(lambda: select_time_frame(last_week_fn, "Last Week"),
                            outputs=[revenue_plot,
                                      profit_plot,
                                      sales_plot,
                                      traffic_plot,
                                      conversion_rate_plot,
                                      average_check_plot])
    last_month_button.click(lambda: select_time_frame(last_month_fn, "Last Month"), 
                            outputs=[revenue_plot,
                                      profit_plot,
                                      sales_plot,
                                      traffic_plot,
                                      conversion_rate_plot,
                                      average_check_plot])

    deplay_button.click(lambda: select_channel('deplay.nl'), 
                        outputs=[revenue_plot,
                                  profit_plot,
                                  sales_plot,
                                  traffic_plot,
                                  conversion_rate_plot,
                                  average_check_plot])
    bol_button.click(lambda: select_channel('bol.nl'), 
                     outputs=[revenue_plot,
                               profit_plot,
                               sales_plot,
                               traffic_plot,
                               conversion_rate_plot,
                               average_check_plot])
    amazon_button.click(lambda: select_channel('amazon.com'), 
                        outputs=[revenue_plot,
                                  profit_plot,
                                  sales_plot,
                                  traffic_plot,
                                  conversion_rate_plot,
                                  average_check_plot])
    
   

demo.launch(inbrowser=True, share=False)



