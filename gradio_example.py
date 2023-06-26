
import gradio as gr
import random
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shap
import pickle

from torch import frac
from train import load_model, load_preprocessor, load_dataset
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, MinMaxScaler

matplotlib.use("Agg")

# load datasets
X_train, _, X_test, _ = load_dataset("./data/")
# load raw data
with open("./data/merged2.pkl", 'rb') as inputfile:
    merged2 = pickle.load(inputfile)
    
# read sklearn preprocessors
with open("./data/ordinal_encoder.pkl", 'rb') as inputfile:
    ordinal_encoder = pickle.load(inputfile)
    
with open("./data/label_encoder.pkl", 'rb') as inputfile2:
    label_encoder = pickle.load(inputfile2)
    
with open("./data/scaler.pkl", 'rb') as inputfile3:
    scaler = pickle.load(inputfile3)
    
# load widedeep preprocessors and models
tab_preprocessor = load_preprocessor("./output/")
model = load_model(tab_preprocessor,"./output/")
#explainer = shap.DeepExplainer(model, X_train.sample(100), session="pytorch")

# column info
categorical_columns = ['page',
            'rank',
            'is_fba',
            'is_SFP',
            'has_prime_badge',
            'is_algorithmic',
            'date_month',
            'date_day_of_week',
            'date_week_of_year',
            'date_hour',
            "pid",
            "sid"]

continous_columns = ['sid_rating',
            'price',
            'sid_pos_fb',
            'pid_rating',
            'sid_rating_cnt',
            'shipping_price',
            'pid_rating_cnt',
            'date_day',
            'date_day_of_year']

nominal_cols = list(ordinal_encoder.feature_names_in_)
col_names = list(scaler.feature_names_in_)

# calculate column stats
unique_pid = sorted(merged2["pid"].unique())
unique_sid = sorted(merged2["sid"].unique())
unique_day = [int(i) for i in sorted(merged2["date_day"].unique())]
unique_month = [int(i) for i in sorted(merged2["date_month"].unique())]
unique_hour = [int(i) for i in sorted(merged2["date_hour"].unique())]
""" unique_rank = sorted(merged2["rank"].unique()) """
""" unique_is_fba = sorted(merged2["is_fba"].unique())
unique_is_SFP = sorted(merged2["is_SFP"].unique())
unique_has_prime_badge = sorted(merged2["has_prime_badge"].unique())
unique_is_algorithmic = sorted(merged2["is_algorithmic"].unique()) """

""" min_sid_rating = sorted(merged2["sid_rating"].unique()).min()
max_sid_rating = sorted(merged2["sid_rating"].unique()).max() """
""" min_sid_pos_fb = sorted(merged2["sid_pos_fb"].unique()).min()
max_sid_pos_fb = sorted(merged2["sid_pos_fb"].unique()).max() """
""" min_sid_rating_cnt = sorted(merged2["sid_rating_cnt"].unique()).min() """
max_sid_rating_cnt = int(max(sorted(merged2["sid_rating_cnt"].unique())))
""" min_pid_rating = sorted(merged2["pid_rating"].unique()).min()
max_pid_rating = sorted(merged2["pid_rating"].unique()).max() """
""" min_pid_rating_cnt = sorted(merged2["pid_rating_cnt"].unique()).min()"""
max_pid_rating_cnt = int(max(sorted(merged2["pid_rating_cnt"].unique())))
min_price = float(min(sorted(merged2["price"].unique())))
max_price = float(max(sorted(merged2["price"].unique())))
min_shipping_price = float(min(sorted(merged2["shipping_price"].unique())))
max_shipping_price = float(max(sorted(merged2["shipping_price"].unique())))


def predict(price, sid_rating, sid_pos_fb, sid_rating_cnt,
                        shipping_price, rank, pid_rating, pid_rating_cnt,
                        is_fba, is_SFP, has_prime_badge, is_algorithmic, date_month,
                        date_day, date_hour):
    pid = random.choice(unique_pid)
    sid = random.choice(unique_sid)    
    rank = rank-1
    page = 1 if rank <= 10 else 2
    date_locater = merged2.loc[(merged2['date_day'] == date_day) & (merged2['date_month'] == date_month)]
    try:
        date_day_of_week = int(date_locater["date_day_of_week"].values[0])
        date_week_of_year = int(date_locater["date_week_of_year"].values[0])
        date_day_of_year = int(date_locater["date_week_of_year"].values[0])
    except:
        date_locater = merged2.loc[merged2['date_month'] == date_month]
        unique_day_in_month = date_locater["date_day"].unique()
        date_day_idx = np.argmin([abs(i-date_day) for i in unique_day_in_month])
        date_day = int(unique_day_in_month[date_day_idx])
        
        date_locater = merged2.loc[(merged2['date_day'] == date_day) & (merged2['date_month'] == date_month)]
        date_day_of_week = int(date_locater["date_day_of_week"].values[0])
        date_week_of_year = int(date_locater["date_week_of_year"].values[0])
        date_day_of_year = int(date_locater["date_week_of_year"].values[0])
        
    cols=np.array([pid, sid, price, sid_rating, sid_pos_fb, sid_rating_cnt,
        shipping_price, page, rank, pid_rating, pid_rating_cnt,
        is_fba, is_SFP, has_prime_badge, is_algorithmic, date_month,
        date_day, date_hour, date_day_of_week, date_day_of_year,
        date_week_of_year]).reshape(1,21)
 
    df = pd.DataFrame(cols, columns=col_names)
    df[nominal_cols] = ordinal_encoder.transform(df[nominal_cols])    
    scaled_array = scaler.transform(df)   
    df = pd.DataFrame(scaled_array, columns=col_names)
    df = tab_preprocessor.transform(df)  
    pred_prob = model.predict_proba(X_tab = df)
    print(pred_prob)
    return {"YES": float(pred_prob[0][0]), "NO": 1 - float(pred_prob[0][0])}


""" def interpret(*args):   
    df = pd.DataFrame([args], columns=col_names)
    shap_values = explainer.shap_values(df)
    scores_desc = list(zip(shap_values[0], col_names))
    scores_desc = sorted(scores_desc)
    fig_m = plt.figure(tight_layout=True)
    plt.barh([s[1] for s in scores_desc], [s[0] for s in scores_desc])
    plt.title("Feature Shap Values")
    plt.ylabel("Shap Value")
    plt.xlabel("Feature")
    plt.tight_layout()
    return fig_m
 """



with gr.Blocks() as demo:
    gr.Markdown("""
    # Amazon BuyBox Classification with Deep Learning 💰 #
    
    This demo uses an TabTransformer classifier predicting BuyBox win/lose based on product and seller features, along with Shapley value-based *explanations*. The [Project Report for this Gradio demo is here](https://docs.google.com/document/d/1tZROZiJS-OqR6erWn3v7HDMzlygz7WQi2jDRYYs_s7Y).
    
    Enter Values below then click Predict Button to see the result. You may also click Interpret button to analyze it.  
    HINT: Try to increase Seller Rating features and the Price to see if you can still win the BuyBox with an **high profit margin**.
    """)
    with gr.Row():
        with gr.Column():
            
            date_day = gr.Dropdown(
                label="Select Day",
                choices=unique_day,
                value=lambda: random.choice(unique_day),
            )
            date_month = gr.Dropdown(
                label="Select Month",
                choices=unique_month,
                value=lambda: random.choice(unique_month),
            )
            date_hour = gr.Dropdown(
                label="Select Hour",
                choices=unique_hour,
                value=lambda: random.choice(unique_hour),
            )

            pid_rating = gr.Slider(label="Product Rating", minimum=1.0, maximum = 5.0, step=0.5, randomize=True)
            pid_rating_cnt = gr.Slider(label="Product Rating Count", minimum=0, maximum=max_pid_rating_cnt, step=100, randomize=True)

            sid_rating = gr.Slider(label="Seller Rating", minimum=1.0, maximum=5.0, step=0.5, randomize=True)
            sid_rating_cnt = gr.Slider(label="Seller Rating Count", minimum=0, maximum=max_sid_rating_cnt, step=100, randomize=True)
            sid_pos_fb = gr.Slider(label="Percentage of Positive Feedbacks", minimum=0, maximum=100, step=1, randomize=True)

            rank = gr.Dropdown(
                label="Seller Rank",
                choices=list(range(1,15)),
                value=lambda: int(random.choice(list(range(1,15)))),
            )
            
            is_fba = gr.Dropdown(
                label="Is Seller FBA?",
                choices=["yes","no"],
                value=lambda: random.choice(["yes","no"]),
            )
                        
            is_SFP = gr.Dropdown(
                label="Is Seller SFP?",
                choices=["yes","no"],
                value=lambda: random.choice(["yes","no"]),
            )
            
            has_prime_badge = gr.Dropdown(
                label="Has Seller Prime Badge?",
                choices=["yes","no"],
                value=lambda: random.choice(["yes","no"]),
            )
                        
            is_algorithmic = gr.Dropdown(
                label="Is Seller Algorithmic?",
                choices=["yes","no"],
                value=lambda: random.choice(["yes","no"]),
            )
            
            price = gr.Slider(label="Price Offered", minimum=0.0, maximum=float(int(max_price)), step=100, randomize=True)
            shipping_price = gr.Slider(label="Shipping Price Offered", minimum=0.0, maximum=max_shipping_price, step=10, randomize=True)

        with gr.Column():
            label = gr.Label(num_top_classes=2, label="Click Predict to see the result")
            plot = gr.Plot()
            with gr.Row():
                predict_btn = gr.Button(value="Predict")
                #interpret_btn = gr.Button(value="Explain")
            predict_btn.click(
                predict,
                inputs=[price, sid_rating, sid_pos_fb, sid_rating_cnt,
                        shipping_price, rank, pid_rating, pid_rating_cnt,
                        is_fba, is_SFP, has_prime_badge, is_algorithmic, date_month,
                        date_day, date_hour],
                outputs=[label],
            )
"""             interpret_btn.click(
                interpret,
                inputs=[pid, sid, price, sid_rating, sid_pos_fb, sid_rating_cnt,
                        shipping_price, page, rank, pid_rating, pid_rating_cnt,
                        is_fba, is_SFP, has_prime_badge, is_algorithmic, date_month,
                        date_day, date_hour, date_day_of_week, date_day_of_year,
                        date_week_of_year],
                outputs=[plot],
            ) """

demo.launch(share=False)
