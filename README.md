Step 1: Import Libraries and Load Dataset
We import necessary libraries such as pandas for data manipulation, numpy for numerical operations, and scikit-learn for machine learning tasks. We then load the dataset containing features like carat, cut, color, clarity, and price.
Step 2: Preprocess the Data
⦁	Handle Categorical Features: We convert categorical features (cut, color, clarity) into numerical values using one-hot encoding, which creates binary columns for each category.
⦁	Separate Features and Target Variable: We separate the dataset into features (X) and the target variable (y), which is the price of the diamonds.
⦁	Split the Data: We split the data into training and testing sets to evaluate the model's performance on unseen data.
⦁	Standardize the Features: We scale the feature values to have a mean of 0 and a standard deviation of 1, which helps improve the performance of the linear regression model.
Step 3: Train the Model
We initialize a linear regression model and train it using the training data. The model learns the relationship between the features and the target variable (price).
Step 4: Evaluate the Model
⦁	Make Predictions: We use the trained model to predict diamond prices on the test set.
⦁	Calculate Mean Squared Error (MSE): We evaluate the model's performance using MSE, which measures the average squared difference between predicted and actual prices.
⦁	Calculate Root Mean Squared Error (RMSE): We also compute RMSE, the square root of MSE, which is in the same units as the target variable and provides a more interpretable error metric.
