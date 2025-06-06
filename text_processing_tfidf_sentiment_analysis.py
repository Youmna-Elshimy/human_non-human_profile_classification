# -*- coding: utf-8 -*-
"""text_processing.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1eyUQqoCYhToLLdb8CEMVZnKtqINcz8K4
"""

#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
nltk.download('punkt')
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob

#load the dataset
data = pd.read_csv("twitter_user_data.csv", encoding="ISO-8859-1")

#select the "text", "description", and "gender" columns
selected_columns = data.filter(items=['text', 'description', 'gender'])
print(selected_columns)

#check the number of unique values in each column
unique_values = selected_columns.nunique()

#plot the number of unique values in each column
plt.figure(figsize=(10, 6))
unique_values.plot(kind='bar', color='lightcoral')

plt.title('Number of Unique Values in Selected Columns')
plt.xlabel('Columns')
plt.ylabel('Number of Unique Values')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

#remove duplicate rows in "description" and "text"
data_cleaned = selected_columns.drop_duplicates(subset=['text'])
data_cleaned = selected_columns.drop_duplicates(subset=['description'])

#check for missing values
missing_values = data_cleaned.isnull().sum()
print(missing_values)

#plotting missing values per column
plt.figure(figsize=(10, 6))
missing_values.plot(kind='bar', color='lightcoral')
plt.title('Missing Values per Column')
plt.xlabel('Columns')
plt.ylabel('Number of Missing Values')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

#remove missing data
data_cleaned = data_cleaned.dropna(subset=['text', 'description', 'gender'])
print(data_cleaned)

#check different gender values
data_cleaned['gender'].unique()

# Check the frequency of each unique gender value
gender_counts = data_cleaned['gender'].value_counts()

# Plotting
plt.figure(figsize=(10, 6))
gender_counts.plot(kind='bar', color='lightcoral')

plt.title('Frequency of Gender Categories')
plt.xlabel('Gender')
plt.ylabel('Frequency')
plt.xticks(rotation=0)
plt.tight_layout()

# Show the plot
plt.show()

#remove rows where "gender" is "unknown"
data_cleaned = data_cleaned[data_cleaned['gender'] != 'unknown']

#print number of rows after cleaning data
print(data_cleaned.shape[0])

#adding columns for description length and text length
data_cleaned['description_length'] = data_cleaned['description'].apply(len)
data_cleaned['text_length'] = data_cleaned['text'].apply(len)

#calculating the mean description length and text length by gender
avg_lengths = data_cleaned.groupby('gender')[['description_length', 'text_length']].mean()

#plotting the average description length and text length by gender
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

#plot for description length
avg_lengths['description_length'].plot(kind='bar', ax=ax[0], color='skyblue')
ax[0].set_title('Average Description Length by Gender')
ax[0].set_ylabel('Average Description Length')
ax[0].set_xlabel('Gender')

#plot for text length
avg_lengths['text_length'].plot(kind='bar', ax=ax[1], color='lightgreen')
ax[1].set_title('Average Text Length by Gender')
ax[1].set_ylabel('Average Text Length')
ax[1].set_xlabel('Gender')

#show plots
plt.tight_layout()
plt.show()

#load stopwords
stop_words = set(stopwords.words('english'))

#function to clean text
def clean_text(text):
  #lowercase and tokenize
  tokens = word_tokenize(text.lower())
  #remove numbers/punctuation
  cleaned_text = [word for word in tokens if word.isalpha()]
  #remove stopwords
  filtered_text = [word for word in cleaned_text if word not in stop_words]
  #join the cleaned tokens back into a single string
  return ' '.join(filtered_text)
  #return filtered_text

#apply the function to 'text' and 'description'
data_cleaned['cleaned_text'] = data_cleaned['text'].apply(lambda x: clean_text(str(x)))
data_cleaned['cleaned_description'] = data_cleaned['description'].apply(lambda x: clean_text(str(x)))

#display the first few rows to verify
print(data_cleaned[['cleaned_text', 'cleaned_description']].head())

#concatenate text and description into a single column
data_cleaned['combined_text'] = data_cleaned['cleaned_text'] + ' ' + data_cleaned['cleaned_description']

#create the TF-IDF Vectorizer
tfidf = TfidfVectorizer()

#apply TF-IDF to "combined_text"
tfidf_matrix = tfidf.fit_transform(data_cleaned['combined_text'])

#convert the TF-IDF matrix to a data frame for analysis
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())

#add the gender column for analysis
tfidf_df['gender'] = data_cleaned['gender'].values

#group by gender and calculate the mean TF-IDF score for each word
tfidf_by_gender = tfidf_df.groupby('gender').mean()

#get the top 20 words by TF-IDF score for male
top_words_male = tfidf_by_gender.T['male'].sort_values(ascending=False).head(20)

#plot the top words for male
plt.figure(figsize=(10, 6))
plt.barh(top_words_male.index, top_words_male.values, color='blue')
plt.title('Top 20 Words for Male Profiles by TF-IDF Score')
plt.gca().invert_yaxis()  # Highest values at the top
plt.tight_layout()
plt.show()

#get the top 20 words by TF-IDF score for female
top_words_female = tfidf_by_gender.T['female'].sort_values(ascending=False).head(20)

#plot the top words for female
plt.figure(figsize=(10, 6))
plt.barh(top_words_female.index, top_words_female.values, color='green')
plt.title('Top 20 Words for Female Profiles by TF-IDF Score')
plt.gca().invert_yaxis()  # Highest values at the top
plt.tight_layout()
plt.show()

#get the top 20 words by TF-IDF score for brand
top_words_brand = tfidf_by_gender.T['brand'].sort_values(ascending=False).head(20)

#plot the top words for brand
plt.figure(figsize=(10, 6))
plt.barh(top_words_brand.index, top_words_brand.values, color='orange')
plt.title('Top 20 Words for Brand Profiles by TF-IDF Score')
plt.gca().invert_yaxis()  # Highest values at the top
plt.tight_layout()
plt.show()

#function to get sentiment polarity
def get_polarity(text):
    return TextBlob(text).sentiment.polarity

#apply sentiment function to "combined_text"
data_cleaned['polarity'] = data_cleaned['combined_text'].apply(lambda x: get_polarity(str(x)))

#plot the distribution of sentiment polarity
plt.figure(figsize=(10, 6))
data_cleaned['polarity'].hist(bins=50, color='purple')
plt.title('Sentiment Polarity Distribution')
plt.xlabel('Sentiment Polarity')
plt.ylabel('Frequency')
plt.show()

#group polarity by "gender" and calculate the mean polarity for each group
sentiment_by_gender = data_cleaned.groupby('gender')['polarity'].mean()

#plot average polarity for each gender category
plt.figure(figsize=(10, 6))
sentiment_by_gender.plot(kind='bar', color=['blue', 'green', 'orange'])
plt.title('Average Sentiment Polarity by Gender Category')
plt.ylabel('Average Sentiment Polarity')
plt.xlabel('Gender Category')
plt.xticks(rotation=0)
plt.show()

#function to get sentiment subjectivity
def get_subjectivity(text):
   return TextBlob(text).sentiment.subjectivity

#apply subjectivity function to "combined_text"
data_cleaned['subjectivity'] = data_cleaned['combined_text'].apply(lambda x: get_subjectivity(str(x)))

#plot the distribution of sentiment subjectivity
plt.figure(figsize=(10, 6))
data_cleaned['subjectivity'].hist(bins=50, color='skyblue')
plt.title('Sentiment Subjectivity Distribution')
plt.xlabel('Sentiment Subjectivity')
plt.ylabel('Frequency')
plt.show()

#group subjectivity by "gender" and calculate the mean polarity for each group
subjectivity_by_gender = data_cleaned.groupby('gender')['subjectivity'].mean()

#plot average subjectivity for each gender category
plt.figure(figsize=(10, 6))
subjectivity_by_gender.plot(kind='bar', color=['blue', 'green', 'orange'])
plt.title('Average Sentiment Subjectivity by Gender Category')
plt.ylabel('Average Sentiment Subjectivity')
plt.xlabel('Gender Category')
plt.xticks(rotation=0)
plt.show()