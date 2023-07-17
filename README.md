
# Relu Pricing Model

## Description

Relu is a SaaS company that uses AI to help companies price their products. The company has a service that provides pricing recommendations for new products based on a variety of factors. The company is looking to improve the pricing model by incorporating additional data and using a more sophisticated model.

Curreently the model is an Xgboost model that takes in a variety of features and outputs a price. The model is trained on a open source dataset of historical pricing data. The model is is trained with Modal instances and the dashboard is deployed using Gradio on Hugging Face spaces

This is version 0.1 for demostration purposes only. The model is not production ready.

## Project Structure

```
├── README.md
├── data
│   ├── raw
│   ├── processed
│   └── interim
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
make setup
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
is used to create new features. The data is stored in the data/processed folder.

### Step 6: Perfom inference and run pricing optimization.

Run the following command to perform inference and run pricing optimization.

```bash
python src/train/train_modal.py
```
You will need to create and account with Modal and request access to a token to the owner or create a new token in your Modal account.

### Step 7: Deploy the dashboard

Run the following command to deploy the dashboard.

```bash
python src/app.py
```
You will need to create and account with Hugging Face spaces and request access to Relu Organization to the owner.

