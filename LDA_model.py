# -*- coding: utf-8 -*-
"""Untitled9.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1HS6TI093dTyLknP0l242f910ffHmCuYP
"""

import pandas as pd
import re
import nltk
import gensim
import gensim.corpora as corpora
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models import CoherenceModel
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load dataset and retain relevant columns (text, gender)
data = pd.read_csv('twitter_user_data.csv',encoding='ISO-8859-1')
data = data[['text', 'gender']]  # Keep only relevant columns
data.head()

# Filter rows to retain only male, female, and brand in the gender column
data = data[data['gender'].isin(['male', 'female', 'brand'])]

# Preprocess the text data for LDA
def preprocess(text):
    # Check if text is a string before processing
    if isinstance(text, str):
        # Remove URLs, mentions, hashtags, and non-alphabetical characters
        text = re.sub(r"http\S+|www\S+|@\w+|#\w+|[^A-Za-z\s]", '', text)
        # Tokenize and remove stopwords
        tokens = word_tokenize(text.lower())
        tokens = [word for word in tokens if word not in stopwords.words('english')]
        # Lemmatize
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        return tokens
    else:
        # Return an empty list if text is not a string
        return []

# Apply preprocessing to the dataset
data['processed_text'] = data['text'].apply(preprocess)
data.head()

#plot to visualize the distribution of text length by gender
#Filter the rows based on gender (keep only 'male', 'female', and 'brand')
filtered_data = data[data['gender'].isin(['male', 'female', 'brand'])]

#Add a column to calculate the length of each tweet
filtered_data['text_length'] = filtered_data['text'].apply(lambda x: len(str(x)))

#Plot the distribution of text length by gender
plt.figure(figsize=(10, 6))
sns.histplot(data=filtered_data, x='text_length', hue='gender', multiple='stack', kde=True)
plt.title('Distribution of Text Length by Gender')
plt.xlabel('Text Length')
plt.ylabel('Frequency')
plt.show()

#Plot for row count before and after filtering the gender column
# Count the number of rows before filtering
initial_count = data.shape[0]

# Filter the rows based on gender column (only male, female, and brand)
filtered_data = data[data['gender'].isin(['male', 'female', 'brand'])]

# Count the number of rows after filtering
filtered_count = filtered_data.shape[0]

# Create a dataframe to store the counts
counts_df = pd.DataFrame({
    'Condition': ['Before Filtering', 'After Filtering'],
    'Row Count': [initial_count, filtered_count]
})

# Plot the counts
plt.figure(figsize=(8, 6))
sns.barplot(x='Condition', y='Row Count', data=counts_df)
plt.title('Row Count Before and After Filtering by Gender')
plt.ylabel('Number of Rows')
plt.show()

# Text Processing: Create dictionary and corpus needed for LDA
id2word = corpora.Dictionary(data['processed_text'])
corpus = [id2word.doc2bow(text) for text in data['processed_text']]

# Build the LDA model with 10 topics
lda_model = gensim.models.LdaMulticore(corpus=corpus, id2word=id2word, num_topics=10, random_state=100, chunksize=100, passes=10, per_word_topics=True)

# Print the topics
for idx, topic in lda_model.print_topics(-1):
    print(f'Topic: {idx} \nWords: {topic}')

# Compute coherence score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data['processed_text'], dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print(f'Coherence Score: {coherence_lda}')

# Visualize Word Clouds for each Topic
def plot_word_clouds(lda_model, num_topics, terms):
    for i in range(num_topics):
        plt.figure(figsize=(8, 8))
        wordcloud = WordCloud(background_color='white').generate_from_frequencies(dict(lda_model.show_topic(i, 50)))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title(f'Topic {i+1}')
        plt.show()

plot_word_clouds(lda_model, num_topics=10, terms=id2word)

# Visualize Topic Distribution Across Documents
def plot_topic_distribution(lda_model, corpus):
    topic_dist = [max(prob for _, prob in doc) for doc in lda_model.get_document_topics(corpus)]
    sns.histplot(topic_dist, kde=True)
    plt.title('Topic Distribution Across Documents')
    plt.xlabel('Dominant Topic Probability')
    plt.ylabel('Document Count')
    plt.show()

plot_topic_distribution(lda_model, corpus)

