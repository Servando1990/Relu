# create a function that uutputs todays date
import pandas as pd
# import datetime module
import datetime
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import gradio as gr

# create a function that outputs todays date in the format of "Monday, January 10th 2020 at 1:15pm"
def get_date():
    # get todays date
    date = datetime.datetime.now()
    # format the date
    date = date.strftime("%A, %B %dth %Y at %I:%M%p")
    # return the date
    return date

current_date = get_date()



def this_month_fn(df):
    start_date = datetime.datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    end_date = datetime.datetime.now()
    return df[(df['date'] >= start_date) & (df['date'] <= end_date)]

def this_week_fn(df):
    start_date = datetime.datetime.now() - datetime.timedelta(days=datetime.datetime.now().weekday())
    start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = datetime.datetime.now()
    return df[(df['date'] >= start_date) & (df['date'] <= end_date)]



def last_month_fn(df):
    start_date = datetime.datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0) - datetime.timedelta(days=1)
    start_date = start_date.replace(day=1)
    end_date = datetime.datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0) - datetime.timedelta(seconds=1)
    return df[(df['date'] >= start_date) & (df['date'] <= end_date)]

def last_week_fn(df):
    start_date = datetime.datetime.now() - datetime.timedelta(days=datetime.datetime.now().weekday() + 7)
    start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = start_date + datetime.timedelta(days=6, hours=23, minutes=59, seconds=59)
    return df[(df['date'] >= start_date) & (df['date'] <= end_date)]


date_rng = pd.date_range(start='2023-01-01', end='2023-08-18', freq='D')
channels = ["deplay.nl", "amazon.com", "bol.nl"]

# Create a revenue series with more realistic fluctuations
np.random.seed(42)
revenue = []
channel_data = []
for i in range(len(date_rng)):
    base_revenue = 100 + i * 2
    daily_fluctuation = np.sin(i / 7) * 20
    random_noise = np.random.normal(0, 10)
    for ch in channels:
        base_revenue = 100 + i * 2 + np.random.randint(-10, 10) # Base revenue with random variation
        daily_fluctuation = np.sin(i / 7 + np.random.random()) * 20 # Fluctuation with random phase
        random_noise = np.random.normal(0, 10) # Random noise
        daily_revenue = base_revenue + daily_fluctuation + random_noise
        revenue.append(daily_revenue)
        channel_data.append(ch)

data = {
    'date': [date for date in date_rng for _ in channels],
    'channel': channel_data,
    'revenue': revenue
}

df = pd.DataFrame(data)


current_time_fn = this_month_fn  # Default to this month
current_time_frame = "This Month"  # Default time frame label
current_channel = None  # Default to all channels

channel_colors = {
    "deplay.nl": "red",
    "amazon.com": "yellow",
    "bol.nl": "green"
}


def plot_data(df, time_frame=None, channel=None):


    fig = go.Figure()
    
    if channel:
        df_channel = df[df['channel'] == channel]
        line_color = channel_colors[channel]
        fig.add_trace(go.Scatter(x=df['date'], y=df_channel['revenue'], mode='lines', name=channel, line_color=line_color ))
    else:
        for ch in channels:
            df_channel = df[df['channel'] == ch]
            line_color = channel_colors[ch]
            fig.add_trace(go.Scatter(x=df_channel['date'], y=df_channel['revenue'], mode='lines', name=ch, line_color=line_color))
    
    # Calculate the sum of revenue for the selected timeframe and channel(s)
    revenue_sum = round(df['revenue'].sum(), 2)
    # Format the revenue sum with commas and add the Euro sign
    revenue_str = f"€{revenue_sum:,.2f}"
    
    title = f"<b style='font-size: 20px;'>Revenue: {revenue_str}</b> "
    #title = "Revenue: ${} ".format(revenue_sum)  # Include the revenue sum in the title
    if time_frame:
        title += f"for {time_frame} "
    if channel:
        title += f"in {channel}"
       
    fig.update_layout(title={
        'text': title,
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'}, 
        xaxis_title='Date', yaxis_title='Revenue')
    
    return gr.update(value=fig, visible=True)


def update_plot():
    # Apply the time filter function
    df_filtered_time = current_time_fn(df)
    if current_channel:
        # Apply the channel filter
        df_filtered_time = df_filtered_time[df_filtered_time['channel'] == current_channel]
    # Update the plot with the filtered data and selected channel
    return plot_data(df_filtered_time, current_time_frame, current_channel)

def select_time_frame(time_fn, time_frame):
    global current_time_fn, current_time_frame
    current_time_fn = time_fn
    current_time_frame = time_frame
    return update_plot()

def select_channel(channel):
    global current_channel
    current_channel = channel
    return update_plot()

def reset_plot():
    global current_time_fn, current_time_frame, current_channel
    current_time_fn = this_month_fn
    current_time_frame = "This Month"
    current_channel = None
    return update_plot()


def same_auth(username, password):
    return username == password

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
    




    plot = gr.Plot()
    plot.update(value=update_plot())


    this_month_button.click(lambda: select_time_frame(this_month_fn, "This Month"), outputs=plot)
    this_week_button.click(lambda: select_time_frame(this_week_fn, "This Week"), outputs=plot)
    last_week_button.click(lambda: select_time_frame(last_week_fn, "Last Week"), outputs=plot)
    last_month_button.click(lambda: select_time_frame(last_month_fn, "Last Month"), outputs=plot)

    deplay_button.click(lambda: select_channel('deplay.nl'), outputs=plot)
    bol_button.click(lambda: select_channel('bol.nl'), outputs=plot)
    amazon_button.click(lambda: select_channel('amazon.com'), outputs=plot)
    
    with gr.Column():
        with gr.Row():
            reset_button = gr.Button(size="sm", value="Reset")
            reset_button.click(reset_plot, outputs=plot)


    

demo.launch(inbrowser=True, share=False )



