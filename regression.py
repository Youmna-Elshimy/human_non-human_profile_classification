#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


# In[14]:


import pandas as pd
df = pd.read_csv("twitter_user_data.csv", encoding="ANSI")


# In[15]:


#dropping unnecessary columns
df.drop(['_unit_id', 'name', 'profileimage','sidebar_color','link_color','name','gender_gold','profile_yn_gold'],axis=1,inplace=True)


# In[16]:


df['gender'].unique()


# In[17]:


df['gender'].fillna(df['gender'].mode()[0], inplace=True)  # Fill 'gender' with the mode (most frequent value)
df['description'].fillna('NA', inplace=True)  # Fill 'description' with 'NA'
df['gender:confidence'].fillna(df['gender:confidence'].mean(), inplace=True)  # Fill 'gender:confidence' with mean


# In[18]:


# Convert date columns to datetime format
df['_last_judgment_at'] = pd.to_datetime(df['_last_judgment_at'], errors='coerce')
df['tweet_created'] = pd.to_datetime(df['tweet_created'], errors='coerce')


# In[19]:


# Feature Engineering: Create new features
df['description_length'] = df['description'].apply(lambda x: len(str(x)))  # Length of description
df['tweet_to_retweet_ratio'] = df['tweet_count'] / (df['retweet_count'] + 1)  # Avoid division by zero



# In[20]:


df['gender'].unique()


# In[21]:


from sklearn.preprocessing import LabelEncoder

# Initialize label encoder
le = LabelEncoder()

# Encode categorical features
df['gender'] = le.fit_transform(df['gender'])
df['profile_yn'] = le.fit_transform(df['profile_yn'])




# In[22]:


# Select only the numeric columns for correlation
numeric_data = df.select_dtypes(include=['float64', 'int64'])

# Generate the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Numerical Features')
plt.show()


# In[23]:


# Scatter plot for tweet_count vs gender:confidence
plt.figure(figsize=(8, 6))
sns.scatterplot(x='tweet_count', y='gender:confidence', data=df)
plt.title('Tweet Count vs Gender Confidence')
plt.show()

# Scatter plot for tweet_to_retweet_ratio vs gender:confidence
plt.figure(figsize=(8, 6))
sns.scatterplot(x='tweet_to_retweet_ratio', y='gender:confidence', data=df)
plt.title('Tweet to Retweet Ratio vs Gender Confidence')
plt.show()


# In[24]:


sns.countplot(x='gender', data=df)
plt.title('Distribution of Gender')
plt.show()


# In[25]:


# Define the features (X) and target variable (y)
X = df[['tweet_count', 'retweet_count', 'fav_number', 'description_length', 'tweet_to_retweet_ratio', 'gender', 'profile_yn']]
y = df['gender:confidence']


# In[26]:


from sklearn.model_selection import train_test_split

# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[27]:


from sklearn.linear_model import LinearRegression

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)


# In[28]:


y_pred = model.predict(X_test)


# In[29]:


from sklearn.metrics import mean_squared_error, r2_score

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
# Calculate R-squared
r2 = r2_score(y_test, y_pred)



print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")



# In[30]:


import numpy as np

# Scatter plot for actual vs predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, label='Predicted vs Actual', color='blue')

# Plot the diagonal line (where predicted = actual)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2, label='Perfect Fit (y=x)')

plt.xlabel('Actual Gender Confidence')
plt.ylabel('Predicted Gender Confidence')
plt.title('Actual vs Predicted Gender Confidence')
plt.legend()
plt.show()


# In[31]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# In[32]:


# Initialize the Random Forest model with 100 trees (estimators)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model on the training data
rf_model.fit(X_train, y_train)


# In[33]:


# Make predictions on the test set
y_pred_rf = rf_model.predict(X_test)


# In[34]:


# Calculate Mean Squared Error (MSE)
mse_rf = mean_squared_error(y_test, y_pred_rf)

# Calculate R-squared
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Random Forest - Mean Squared Error: {mse_rf}")
print(f"Random Forest - R-squared: {r2_rf}")


# In[35]:


# Scatter plot for actual vs predicted values for Random Forest
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_rf, label='Predicted vs Actual', color='green')

# Plot the diagonal line (perfect fit)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2, label='Perfect Fit (y=x)')

plt.xlabel('Actual Gender Confidence')
plt.ylabel('Predicted Gender Confidence')
plt.title('Random Forest: Actual vs Predicted Gender Confidence')
plt.legend()
plt.show()


# In[36]:


# Linear Regression metrics
mse_linear = mean_squared_error(y_test, y_pred)
r2_linear = r2_score(y_test, y_pred)

# Random Forest metrics
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# Print the comparison
print(f"Linear Regression - Mean Squared Error: {mse_linear: .2f}")
print(f"Linear Regression - R-squared: {r2_linear: .2f}")

print(f"\nRandom Forest - Mean Squared Error: {mse_rf: .2f}")
print(f"Random Forest - R-squared: {r2_rf: .2f}")


# In[37]:


import matplotlib.pyplot as plt

# Plot Linear Regression results
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.title('Linear Regression: Actual vs Predicted')
plt.xlabel('Actual Gender Confidence')
plt.ylabel('Predicted Gender Confidence')

# Plot Random Forest results
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_rf, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.title('Random Forest: Actual vs Predicted')
plt.xlabel('Actual Gender Confidence')
plt.ylabel('Predicted Gender Confidence')

plt.tight_layout()
plt.show()


# In[38]:


# Get feature importances from the Random Forest model
importances = rf_model.feature_importances_
feature_names = X.columns

# Create a DataFrame to display feature importances
feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

# Display the feature importances
print(feature_importances)

# Plot the feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title('Feature Importances from Random Forest')
plt.show()


# In[39]:


# Calculate residuals (actual - predicted)
residuals = y_test - y_pred_rf

# Plot residuals
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True, color='blue')
plt.title('Distribution of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()


# In[40]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# In[41]:


model = GradientBoostingRegressor()
model.fit(X_train, y_train)

# Step 4: Make Predictions
y_pred = model.predict(X_test)

# Step 5: Evaluate the Model
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)


# In[42]:


plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True, color='blue')
plt.title('Distribution of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()


# In[43]:


importances = model.feature_importances_
feature_names = X.columns

# Create a DataFrame to display feature importances
feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

# Display the feature importances
print(feature_importances)

# Plot the feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title('Feature Importances from Gradient Boosting')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




