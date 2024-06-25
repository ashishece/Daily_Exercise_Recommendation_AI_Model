#!/usr/bin/env python
# coding: utf-8

# In[ ]:


cd Desktop


# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb

# Load data
df = pd.read_csv('enhanced_fitness_data.csv')

# Encode categorical features
df['fitness_level'] = df['fitness_level'].map({'beginner': 0, 'intermediate': 1, 'advanced': 2})
df['goal'] = df['goal'].map({'weight loss': 0, 'muscle gain': 1, 'general fitness': 2})
df['gender'] = df['gender'].map({'male': 0, 'female': 1})

# Features and targets
X = df[['age', 'fitness_level', 'goal', 'gender', 'chest', 'biceps', 'legs', 'thighs', 'shoulders']]
y = df[['target_weight_loss', 'target_biceps_growth', 'target_belly_measurement', 'target_chest', 'target_biceps', 'target_legs', 'target_thighs', 'target_shoulders']]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the LightGBM model
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'l1'},
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

models = {}
for target in y.columns:
    train_data = lgb.Dataset(X_train, label=y_train[target])
    test_data = lgb.Dataset(X_test, label=y_test[target])
    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[test_data],
        callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(10)]
    )
    models[target] = model

# Save the models
for target in y.columns:
    models[target].save_model(f'fitness_model_{target}.txt')

# Example prediction
example_input = pd.DataFrame({
    'age': [30],
    'fitness_level': [1],
    'goal': [1],
    'gender': [0],
    'chest': [90],
    'biceps': [35],
    'legs': [60],
    'thighs': [55],
    'shoulders': [105]
})

predictions = {}
for target in y.columns:
    model = models[target]
    prediction = model.predict(example_input)
    predictions[target] = prediction[0]

print("Example Predictions:")
print(predictions)


# In[ ]:


# Define the feature and target columns
feature_columns = ['age', 'fitness_level', 'goal', 'gender', 'chest', 'biceps', 'legs', 'thighs', 'shoulders']
target_columns = ['target_weight_loss', 'target_biceps_growth', 'target_belly_measurement', 'target_chest', 'target_biceps', 'target_legs', 'target_thighs', 'target_shoulders']

# Function to generate recommendations
def generate_recommendation(age, fitness_level, goal, gender, chest=None, biceps=None, legs=None, thighs=None, shoulders=None):
    # Create a dictionary for the input features with default values
    input_features = {
        'age': age,
        'fitness_level': fitness_level,
        'goal': goal,
        'gender': gender,
        'chest': chest if chest is not None else 0,
        'biceps': biceps if biceps is not None else 0,
        'legs': legs if legs is not None else 0,
        'thighs': thighs if thighs is not None else 0,
        'shoulders': shoulders if shoulders is not None else 0
    }
    
    # Convert to DataFrame
    example_input = pd.DataFrame([input_features])
    
    # Initialize predictions dictionary
    predictions = {}
    
    # Make predictions based on the goal
    if goal == 1:  # Muscle Gain
        for target in target_columns:
            model = lgb.Booster(model_file=f'fitness_model_{target}.txt')
            prediction = model.predict(example_input)
            predictions[target] = prediction[0]
        
        recommendation = f"Based on your input:\n"
        recommendation += f"- Age: {age}\n"
        recommendation += f"- Fitness Level: {'beginner' if fitness_level == 0 else 'intermediate' if fitness_level == 1 else 'advanced'}\n"
        recommendation += f"- Goal: Muscle Gain\n"
        recommendation += f"- Gender: {'Male' if gender == 0 else 'Female'}\n"
        recommendation += f"- Current Measurements (cm):\n"
        recommendation += f"  - Chest: {chest}\n"
        recommendation += f"  - Biceps: {biceps}\n"
        recommendation += f"  - Legs: {legs}\n"
        recommendation += f"  - Thighs: {thighs}\n"
        recommendation += f"  - Shoulders: {shoulders}\n"
        recommendation += "\nRecommendations:\n"
        recommendation += f"- To achieve your muscle gain goal, focus on strength training exercises such as bench press, squats, deadlifts, and shoulder press.\n"
        recommendation += f"- Ensure you progressively increase the weights you lift to continuously challenge your muscles and promote growth.\n"
        recommendation += f"- Maintain a balanced diet with a caloric surplus, ensuring you consume enough protein to support muscle repair and growth.\n"
        recommendation += f"- Suggested Target Biceps Growth: {predictions['target_biceps_growth']:.2f} cm\n"
        recommendation += f"- Suggested Target Chest Measurement: {predictions['target_chest']:.2f} cm\n"
        recommendation += f"- Suggested Target Biceps Measurement: {predictions['target_biceps']:.2f} cm\n"
        recommendation += f"- Suggested Target Legs Measurement: {predictions['target_legs']:.2f} cm\n"
        recommendation += f"- Suggested Target Thighs Measurement: {predictions['target_thighs']:.2f} cm\n"
        recommendation += f"- Suggested Target Shoulders Measurement: {predictions['target_shoulders']:.2f} cm\n"
    else:  # Weight Loss or General Fitness
        for target in ['target_weight_loss', 'target_belly_measurement']:
            model = lgb.Booster(model_file=f'fitness_model_{target}.txt')
            prediction = model.predict(example_input)
            predictions[target] = prediction[0]
        
        recommendation = f"Based on your input:\n"
        recommendation += f"- Age: {age}\n"
        recommendation += f"- Fitness Level: {'beginner' if fitness_level == 0 else 'intermediate' if fitness_level == 1 else 'advanced'}\n"
        recommendation += f"- Goal: {'Weight Loss' if goal == 0 else 'General Fitness'}\n"
        recommendation += f"- Gender: {'Male' if gender == 0 else 'Female'}\n"
        recommendation += "\nRecommendations:\n"

        if goal == 0:  # Weight Loss
            recommendation += f"- To achieve your weight loss goal, incorporate both cardiovascular exercises (such as running, cycling, or swimming) and strength training into your routine.\n"
            recommendation += f"- Aim for at least 150 minutes of moderate aerobic activity or 75 minutes of vigorous activity each week, combined with muscle-strengthening activities on two or more days a week.\n"
            recommendation += f"- Pay attention to your diet, focusing on reducing caloric intake while ensuring you consume a balanced diet rich in nutrients.\n"
            recommendation += f"- Suggested Target Weight Loss: {predictions['target_weight_loss']:.2f} kg\n"
            recommendation += f"- Suggested Target Belly Measurement: {predictions['target_belly_measurement']:.2f} cm\n"
        else:  # General Fitness
            recommendation += f"- Maintain a balanced routine that includes a mix of cardiovascular exercises, strength training, and flexibility workouts.\n"
            recommendation += f"- Aim for at least 150 minutes of moderate aerobic activity or 75 minutes of vigorous activity each week, combined with muscle-strengthening activities on two or more days a week.\n"
            recommendation += f"- Ensure your diet is balanced and nutritious to support your overall fitness and health.\n"
            recommendation += f"- Regularly assess and adjust your routine to keep challenging your body and avoiding plateaus.\n"
    
    return recommendation

# Function to get user input
def get_user_input():
    age = int(input("Enter your age: "))
    fitness_level = input("Enter your fitness level (beginner, intermediate, advanced): ").strip().lower()
    fitness_level = {'beginner': 0, 'intermediate': 1, 'advanced': 2}[fitness_level]
    goal = input("Enter your goal (weight loss, muscle gain, general fitness): ").strip().lower()
    goal = {'weight loss': 0, 'muscle gain': 1, 'general fitness': 2}[goal]
    gender = input("Enter your gender (male, female): ").strip().lower()
    gender = {'male': 0, 'female': 1}[gender]
    
    if goal == 1:  # Muscle Gain
        chest = float(input("Enter your chest measurement (cm): "))
        biceps = float(input("Enter your biceps measurement (cm): "))
        legs = float(input("Enter your legs measurement (cm): "))
        thighs = float(input("Enter your thighs measurement (cm): "))
        shoulders = float(input("Enter your shoulders measurement (cm): "))
        return age, fitness_level, goal, gender, chest, biceps, legs, thighs, shoulders
    else:
        return age, fitness_level, goal, gender, None, None, None, None, None

# Get user input
user_input = get_user_input()

# Generate and print the recommendation
recommendation = generate_recommendation(*user_input)
print(recommendation)

