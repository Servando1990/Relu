

# Relu Demand Curve & Pricing Model

## Description

Relu is a SaaS company that uses AI to help companies price their products. The company has a service that provides pricing recommendations for new products based on a variety of factors. The company is looking to improve the pricing model by incorporating additional data and using a more sophisticated model.

Curreently the model is an Xgboost model that takes in a variety of features and outputs a price. The model is trained on a open source dataset of historical pricing data. The model is is trained with Modal instances and the dashboard is deployed using Gradio on Hugging Face spaces


## Project Structure

```
├── README.md
├── data
│   ├── raw
│   ├── processed
│   └── results
├── notebooks
├── reports
├── requirements.txt
├── setup.py
├── src
│   ├── __init__.py
│   ├── data
│   ├── models
│   ├── pricing
│   ├── train
├── environment.yml
└── Makefile
└── requements.in
└── requements.txt
```

## Setup Instructions

Follow the steps below to set up the project environment.

### Prerequisites

- Conda (Miniconda or Anaconda) installed
- Python version 3.8

### Step 1: Clone the Repository

```bash
git clone <https://github.com/Servando1990/Relu>
```
This is a private repository. Please request access from the owner.


### Step 2: Create and Activate the Conda Environment

```bash
conda env create -f environment.yml
```
This command will create a conda environment named relu based on the environment.yml file and activate it.

### Step 3: Verify the Environment

```bash
conda activate relu
conda env list
```
### Step 4: Clean, Transform and Split the Data

The modules found in src.data are used to clean, transform and split the data. The data is stored in the data/processed folder. The data is cleaned and transformed and stored in the data/processed folder. The data is split into train, validation and test sets and stored in the data/results folder.

### Step 5: Feature Engineering

Module:
```bash
feature_engineering.py
```
is used to create new features

### Step 6: Train & Evalaute model.

class ```DemandCurveTrainer``` found in ```src.train.quantile_trainnin.py``` is used to train and evaluate a quantile regression approach for demand curve predictions. The model is evaluated using the validation set.


### Step 7: Run Pricing algorithm

class ```PricingOptimizer``` found in ```src.pricing.pricing_algorithm.py``` is used to run the pricing algorithm. The pricing algorithm takes in a trained model and a set of features and outputs a a optimal price with its corresponding sku.
