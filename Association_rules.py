#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

# Load the dataset
df = pd.read_csv('twitter_user_data.csv', encoding='ISO-8859-1')

## Check null values and basic info
# print(df.isna().sum())
# print(df.info())


# In[3]:


# Drop irrelevant or nearly empty columns
df = df.drop(columns=['gender_gold', 'profile_yn_gold', 'tweet_coord'])

# Fill missing values
df['gender'].fillna('unknown', inplace=True)
df['description'].fillna('No Description', inplace=True)
df['tweet_location'].fillna('Unknown', inplace=True)
df['user_timezone'].fillna('Unknown', inplace=True)

# Create human vs non-human classification
df['col'] = df['gender'].map({'male': 'human', 'female': 'human', 'brand': 'non-human', 'unknown': 'unknown'})

# Transform binary categorical columns
df['description_bin'] = df['description'].apply(lambda x: 0 if x == 'No Description' else 1)

# Create bins for numerical columns
bins_retweet_count = [0, 1, 2, df['retweet_count'].max()]
bins_fav_number = [0, 1, 60, df['fav_number'].max()]
bins_tweet_count = [1, 20, 450, df['tweet_count'].max()]

df['retweet_count_bin'] = pd.cut(df['retweet_count'], bins=bins_retweet_count, labels=['low', 'medium', 'high'])
df['fav_number_bin'] = pd.cut(df['fav_number'], bins=bins_fav_number, labels=['low', 'medium', 'high'])
df['tweet_count_bin'] = pd.cut(df['tweet_count'], bins=bins_tweet_count, labels=['low', 'medium', 'high'])

# Create derived feature (account age)
df['created'] = pd.to_datetime(df['created'])
df['account_age'] = (pd.to_datetime('today') - df['created']).dt.days


# In[4]:


# Visualizations
plt.figure(figsize=(6, 4))
sns.countplot(x='col', data=df)
plt.title('Human vs Non-Human Distribution')
plt.show()

plt.figure(figsize=(6, 4))
sns.scatterplot(x='retweet_count', y='tweet_count', hue='col', data=df)
plt.title('Retweet Count vs Tweet Count by Human/Non-Human')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='fav_number', y='tweet_count', hue='col', data=df)
plt.title('Favorite Count vs Tweet Count by Human/Non-Human')
plt.show()


# In[5]:


# Select relevant columns for the Apriori algorithm
df_apriori = df[['col', 'description_bin', 'fav_number_bin', 'retweet_count_bin', 'tweet_count_bin', 'user_timezone']]#, 'profile_yn', 'gender:confidence' 'link_color_bin', 'sidebar_color_bin']]

# Convert categorical and binary features into one-hot encoding
df_apriori_onehot = pd.get_dummies(df_apriori, columns=['col', 'fav_number_bin', 'retweet_count_bin', 'tweet_count_bin', 'user_timezone'])#, 'link_color_bin', 'sidebar_color_bin'])

# Convert non-binary values to binary (0 and 1)
df_apriori_onehot[df_apriori_onehot.columns] = df_apriori_onehot[df_apriori_onehot.columns].apply(lambda x: x.apply(lambda y: 1 if y > 0 else 0))

# View the transformed data
print(df_apriori_onehot.head())


# In[6]:


# Apply the Apriori algorithm
frequent_itemsets = apriori(df_apriori_onehot, min_support=0.05, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.6)

rules = rules.sort_values(by='lift', ascending=False)

# View the top association rules
print(rules.head())


# In[7]:


# Visualize frequent itemsets by support
plt.figure(figsize=(10, 6))
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
sns.barplot(x='support', y='itemsets', data=frequent_itemsets.sort_values(by='support', ascending=False).head(10))
plt.title('Top 10 Frequent Itemsets by Support')
plt.show()

# Visualize the rules by confidence and lift
plt.figure(figsize=(10, 6))
sns.scatterplot(x='confidence', y='lift', hue='antecedents', size='support', data=rules)
plt.title('Association Rules by Confidence and Lift')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Adjust this to move legend outside
plt.show()

# Visualize rules with the highest lift
plt.figure(figsize=(10, 6))
sns.barplot(x='lift', y='antecedents', data=rules.sort_values(by='lift', ascending=False).head(10))
plt.title('Top 10 Rules by Lift')
plt.show()

# Summary of results and findings
print("Summary of Findings:")
print(f"The Apriori algorithm identified {len(frequent_itemsets)} frequent itemsets.")
print(f"{len(rules)} association rules were generated with confidence above 0.6.")

# Interpret top rules
print("\nTop 5 rules by lift:")
for idx, rule in rules.head().iterrows():
    print(f"Rule {idx+1}: {list(rule['antecedents'])} -> {list(rule['consequents'])}")
    print(f"   Support: {rule['support']:.4f}, Confidence: {rule['confidence']:.4f}, Lift: {rule['lift']:.4f}\n")

## Identify potential misinformation
#high_lift_rules = rules[rules['lift'] > 2]
#print("Potential misinformation indicators (rules with high lift):")
#for idx, rule in high_lift_rules.iterrows():
#    print(f"- {list(rule['antecedents'])} strongly associated with {list(rule['consequents'])}")


# In[ ]:




