# Import necessary libraries
import glob
import os
import re
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from textstat import textstat

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize an empty list to hold the data
data = []

# Specify the path where the letters are stored
# Replace 'C:\Users\admin\OneDrive\Desktop\Warren Buffet Project\extracted_letters' with your actual path
path = r'C:\Users\admin\OneDrive\Desktop\Warren Buffet Project\extracted_letters'

# Use glob to find all text files in the directory
file_list = glob.glob(os.path.join(path, '*.txt'))

# Iterate over each file
for filepath in sorted(file_list):
    # Extract the filename from the path
    filename = os.path.basename(filepath)
    
    # Use regular expression to extract the year from the filename
    # This regex matches four consecutive digits
    match = re.search(r'(\d{4})', filename)
    if match:
        # Convert the extracted year to an integer
        year = int(match.group(1))
        
        # Read the contents of the file
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Append a dictionary with the year and text to the data list
        data.append({'year': year, 'text': text})
    else:
        print(f'No year found in filename {filename}')

# Create a DataFrame from the data list
df = pd.DataFrame(data)

# Sort the DataFrame by year
df = df.sort_values('year').reset_index(drop=True)

# Display the first few rows of the DataFrame
print(df.head())

# EDA Steps

# 1. Word Count
df['word_count'] = df['text'].apply(lambda x: len(nltk.word_tokenize(x)))

# 2. Character Count
df['char_count'] = df['text'].apply(len)

# 3. Average Word Length
def avg_word_length(text):
    words = nltk.word_tokenize(text)
    return sum(len(word) for word in words) / len(words)

df['avg_word_length'] = df['text'].apply(avg_word_length)

# 4. Sentence Count
df['sentence_count'] = df['text'].apply(lambda x: len(nltk.sent_tokenize(x)))

# 5. Average Sentence Length (in words)
df['avg_sentence_length'] = df['word_count'] / df['sentence_count']

# 6. Readability Scores
df['flesch_reading_ease'] = df['text'].apply(textstat.flesch_reading_ease)
df['flesch_kincaid_grade'] = df['text'].apply(textstat.flesch_kincaid_grade)

# Display the updated DataFrame
print(df.head())

# Set the style for seaborn
sns.set(style='whitegrid', font_scale=1.2)

# Visualizations

# 1. Word Count Over Time
plt.figure(figsize=(12, 6))
sns.lineplot(x='year', y='word_count', data=df, marker='o')
plt.title('Word Count Over Time')
plt.xlabel('Year')
plt.ylabel('Word Count')
plt.show()

# 2. Character Count Over Time
plt.figure(figsize=(12, 6))
sns.lineplot(x='year', y='char_count', data=df, marker='o', color='orange')
plt.title('Character Count Over Time')
plt.xlabel('Year')
plt.ylabel('Character Count')
plt.show()

# 3. Average Word Length Over Time
plt.figure(figsize=(12, 6))
sns.lineplot(x='year', y='avg_word_length', data=df, marker='o', color='green')
plt.title('Average Word Length Over Time')
plt.xlabel('Year')
plt.ylabel('Average Word Length')
plt.show()

# 4. Average Sentence Length Over Time
plt.figure(figsize=(12, 6))
sns.lineplot(x='year', y='avg_sentence_length', data=df, marker='o', color='red')
plt.title('Average Sentence Length (in words) Over Time')
plt.xlabel('Year')
plt.ylabel('Average Sentence Length')
plt.show()

# 5. Flesch Reading Ease Score Over Time
plt.figure(figsize=(12, 6))
sns.lineplot(x='year', y='flesch_reading_ease', data=df, marker='o', color='purple')
plt.title('Flesch Reading Ease Score Over Time')
plt.xlabel('Year')
plt.ylabel('Reading Ease Score')
plt.show()

# 6. Flesch-Kincaid Grade Level Over Time
plt.figure(figsize=(12, 6))
sns.lineplot(x='year', y='flesch_kincaid_grade', data=df, marker='o', color='brown')
plt.title('Flesch-Kincaid Grade Level Over Time')
plt.xlabel('Year')
plt.ylabel('Grade Level')
plt.show()

# Sentiment Analysis

from textblob import TextBlob

# Calculate Polarity and Subjectivity
df['polarity'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
df['subjectivity'] = df['text'].apply(lambda x: TextBlob(x).sentiment.subjectivity)

# Plot Polarity over Time
plt.figure(figsize=(12,6))
sns.lineplot(x='year', y='polarity', data=df, marker='o', color='blue')
plt.title('Sentiment Polarity Over Time')
plt.xlabel('Year')
plt.ylabel('Polarity')
plt.show()

# Plot Subjectivity over Time
plt.figure(figsize=(12,6))
sns.lineplot(x='year', y='subjectivity', data=df, marker='o', color='green')
plt.title('Sentiment Subjectivity Over Time')
plt.xlabel('Year')
plt.ylabel('Subjectivity')
plt.show()

# Topic Modeling

from gensim import corpora
from gensim.utils import simple_preprocess
from gensim.models.ldamodel import LdaModel
from nltk.stem import WordNetLemmatizer

# Preprocessing function
def preprocess(text):
    lemmatizer = WordNetLemmatizer()
    tokens = simple_preprocess(text, deacc=True)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in nltk.corpus.stopwords.words('english')]
    return tokens

df['tokens'] = df['text'].apply(preprocess)

# Create Dictionary and Corpus
dictionary = corpora.Dictionary(df['tokens'])
corpus = [dictionary.doc2bow(text) for text in df['tokens']]

# Build LDA Model
lda_model = LdaModel(corpus=corpus, num_topics=5, id2word=dictionary, passes=15)

# Print the Keyword in the topics
for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))

# Assign Dominant Topic to Each Document
def get_dominant_topic(ldamodel, bow):
    topics = ldamodel.get_document_topics(bow)
    if topics:
        return max(topics, key=lambda x: x[1])[0]
    else:
        return -1

df['dominant_topic'] = [get_dominant_topic(lda_model, doc) for doc in corpus]

# Plot Dominant Topic over Time
plt.figure(figsize=(12,6))
sns.countplot(x='dominant_topic', data=df, palette='viridis')
plt.title('Document Count per Dominant Topic')
plt.xlabel('Topic')
plt.ylabel('Number of Documents')
plt.show()

# Keyword and N-gram Analysis

from collections import Counter
from nltk.util import ngrams

# Function to get N-grams
def get_ngrams(text, n):
    tokens = [token for token in simple_preprocess(text) if token not in nltk.corpus.stopwords.words('english')]
    n_grams = ngrams(tokens, n)
    return [' '.join(grams) for grams in n_grams]

# Get Bigrams and Trigrams
df['bigrams'] = df['text'].apply(lambda x: get_ngrams(x, 2))
df['trigrams'] = df['text'].apply(lambda x: get_ngrams(x, 3))

# Most Common Bigrams
all_bigrams = [bigram for bigrams in df['bigrams'] for bigram in bigrams]
bigram_counts = Counter(all_bigrams)
print("Most Common Bigrams:")
print(bigram_counts.most_common(20))

# Most Common Trigrams
all_trigrams = [trigram for trigrams in df['trigrams'] for trigram in trigrams]
trigram_counts = Counter(all_trigrams)
print("Most Common Trigrams:")
print(trigram_counts.most_common(20))

# Plot Most Common Bigrams
most_common_bigrams = bigram_counts.most_common(10)
bigram_df = pd.DataFrame(most_common_bigrams, columns=['bigram', 'count'])
plt.figure(figsize=(12,6))
sns.barplot(x='count', y='bigram', data=bigram_df, palette='Blues_d')
plt.title('Top 10 Bigrams')
plt.xlabel('Count')
plt.ylabel('Bigram')
plt.show()

# Evolution of Investment Philosophy

investment_keywords = ['value investing', 'intrinsic value', 'margin of safety', 'long-term', 'compounding', 'shareholder value']

def keyword_count(text):
    counts = {}
    text_lower = text.lower()
    for keyword in investment_keywords:
        counts[keyword] = text_lower.count(keyword)
    return counts

df['keyword_counts'] = df['text'].apply(keyword_count)
keyword_df = pd.DataFrame(df['keyword_counts'].tolist())
keyword_df['year'] = df['year']
keyword_df = keyword_df.set_index('year')

# Plotting Keyword Frequencies over Time
keyword_df.plot(figsize=(12,6), marker='o')
plt.title('Investment Keywords Frequency Over Time')
plt.xlabel('Year')
plt.ylabel('Count')
plt.show()

# Communication Style Analysis

# Lexical Diversity
df['lexical_diversity'] = df['tokens'].apply(lambda x: len(set(x)) / len(x))

# Plot Lexical Diversity over Time
plt.figure(figsize=(12,6))
sns.lineplot(x='year', y='lexical_diversity', data=df, marker='o', color='magenta')
plt.title('Lexical Diversity Over Time')
plt.xlabel('Year')
plt.ylabel('Lexical Diversity')
plt.show()

# Use of Second Person Pronouns
second_person_pronouns = ['you', 'your', 'yours']

def count_second_person(text):
    words = nltk.word_tokenize(text.lower())
    count = sum(1 for word in words if word in second_person_pronouns)
    return count

df['second_person_count'] = df['text'].apply(count_second_person)

# Plot Second Person Pronoun Usage over Time
plt.figure(figsize=(12,6))
sns.lineplot(x='year', y='second_person_count', data=df, marker='o', color='teal')
plt.title('Second Person Pronoun Usage Over Time')
plt.xlabel('Year')
plt.ylabel('Count')
plt.show()

# Rhetorical Questions Count
df['rhetorical_questions'] = df['text'].apply(lambda x: x.count('?'))

# Plot Rhetorical Questions over Time
plt.figure(figsize=(12,6))
sns.lineplot(x='year', y='rhetorical_questions', data=df, marker='o', color='orange')
plt.title('Rhetorical Questions Over Time')
plt.xlabel('Year')
plt.ylabel('Count')
plt.show()

# Advanced NLP Techniques

import spacy

# Load spacy model
nlp = spacy.load('en_core_web_sm')

# Named Entity Recognition (NER)
def extract_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

df['entities'] = df['text'].apply(extract_entities)

# Analyze most common entities
all_entities = [ent for entities in df['entities'] for ent in entities]
entity_counts = Counter(all_entities)
print("Most Common Entities:")
print(entity_counts.most_common(20))

# Save the updated DataFrame to a CSV file for future use
df.to_csv('berkshire_letters_with_analysis.csv', index=False)
