#!/usr/bin/env python
# coding: utf-8

#                     SPECIFIC OBJECTIVES
# 
# i.To identify high-risk populations for focused mental health treatments.
# 
# ii.To identify relevant features in social media data that are indicative of mental health
# disorders.
# 
# iii. To develop and train a machine learning model to predict early signs of mental health
# issues based on social media posts.
# 
# iv. To evaluate the performance of the prediction model through metrics such as accuracy,
# precision, recall, and F1 score, ensuring the model’s ability to correctly identify
# individuals at risk

# Importing Necessary Libraries

# This project utilizes key Python libraries for data analysis, visualization, and NLP. pandas handles structured data, while numpy supports numerical operations. matplotlib and seaborn create insightful visualizations. wordcloud helps generate word frequency clouds, and re (regular expressions) aids in text processing. For NLP, nltk provides tools like stopwords (removes common words), word_tokenize (splits text), and WordNetLemmatizer (converts words to their root forms). These libraries streamline data analysis and natural language processing tasks.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report


# This code reads the Mental-Health-Twitter.csv file into a pandas DataFrame and displays the first five rows.
# 
# 

# In[2]:


df = pd.read_csv("E:/Downloads/Mental-Health-Twitter.csv")
df.head()


# Dataset Information

# This command provides a summary of the dataset, including the number of entries, column names, data types, and missing values.

# In[3]:


df.info()


# Statistical Summary of the Dataset

# This command returns descriptive statistics for numerical columns, including count, mean, standard deviation, min, max, and quartiles.

# In[4]:


df.describe()


# Dropping Irrelevant Columns

# This command removes the columns 'Unnamed:0','post_id', and 'user_id' from the dataset, modifying it in place to keep only relevant data.

# In[5]:


df.drop(['Unnamed: 0', 'post_id', 'user_id'], axis=1, inplace=True)


# Previewing the Updated Dataset

# This command displays the first five rows after dropping unnecessary columns, allowing us to very changes.

# In[6]:


df.head()


# Visualizing the Distribution of Mental Health Labels

# This command creates a count plot to show the frequency of each mental health label in the dataset, helping us understand the class distribution.

# In[7]:


sns.countplot(x=df['label'])
plt.title('Distribution of Mental Health Labels')
plt.show()


# Word Cloud of Mental Health Posts Before Cleaning

# This code generates a word cloud from the combined text in the 'post_text' column, visually representing the most frequent words in the dataset

# In[8]:


text = ' '.join(df['post_text'])
wordcloud = WordCloud(width=800, height=400, background_color = 'black').generate(text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Word Cloud of Mental Health Posts')
plt.show()


# Checking for Missing Values

# This command counts the number of missing values in each column, helping identify data quality issues in the dataset

# In[9]:


df.isnull().sum()


# Checking for Duplicate Entries

# This command counts the number of duplicate rows in the dataset, helping identify and remove redundant data.

# In[10]:


df.duplicated().sum()


# Removing Duplicate Entries

# This command removes duplicate rows from the dataset, ensuring data quality and preventing redundancy.

# In[11]:


df.drop_duplicates(inplace=True)


# Dataset Dimensions

# This command returns the number of rows and columns in the dataset, helping us verify its size after cleaning.

# In[12]:


df.shape


# Initialize Lemmatizer

# This code initializes a lemmatizer to reduce words to their base forms and defines a set of stopwords to remove common words, improving text analysis.

# In[13]:


lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


# Text Preprocessing Function

# This function cleans and processes text data to enhance analysis. It converts text to lowercase and removes URLs, hashtags, mentions, and retweet indicators to eliminate unnecessary elements. Special characters and punctuation are also removed, ensuring a cleaner dataset. The text is then tokenized into individual words, after which stopwords are filtered out.Lemmatization is applied to reduce words to their base forms making the text more uniform and meaningful for further processing.

# In[14]:


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'\brt\b', '', text)
    text = re.sub(r'\W+', ' ', text)
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Remove stopwords & lemmatize
    return ' '.join(words)


# Applying Text Preprocessing to the Dataset

# This command applies the preprocess_text function to the 'post_text' column, ensuring all text data is cleaned, tokenized, lemmatized, and free from unnecessary elements for further analysis.

# In[15]:


df['post_text'] = df['post_text'].apply(preprocess_text)


# In[16]:


df.head()


# Generating a Word Cloud from Cleaned Text

# This code combines all preprocessed text from the post_text column into a single string and generates a word cloud, visually highlighting the most frequently used words. The resulting word cloud helps identify key themes and patterns in the dataset.

# In[17]:


text = ' '.join(df['post_text'])

wordcloud = WordCloud(width=800, height=400, background_color='black').generate(text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# TF-IDF Feature Extraction

# This code uses TF-IDF (Term Frequency-Inverse Document Frequency) Vectorization to convert the cleaned text into numerical features. The TfidfVectorizer extracts the top 5,000 most important words, assigning weights based on their importance in the dataset. The transformed data is then converted into a DataFrame for better visualization, where each column represents a word, and each row corresponds to a text entry with its respective TF-IDF scores.

# In[18]:


tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['post_text'])

tfidf_df = pd.DataFrame(X.toarray(), columns=tfidf.get_feature_names_out())

tfidf_df.head()


# Counting Label Occurrences

# 
# This command calculates and displays the frequency of each label in the dataset, helping identify class distribution and potential imbalances.

# In[19]:


label_counts = df['label'].value_counts()
print(label_counts)


# In[20]:


# Plot the counts
label_counts.plot(kind='bar', color=['blue', 'orange'])
plt.xlabel('Label')
plt.ylabel('Count')
plt.title('Distribution of Labels')
plt.xticks(rotation=0)
plt.show()


# Defining Features and Target Variable

# This code sets X as the TF-IDF-transformed text data and y as the target labels, preparing the dataset for model training and evaluation.

# In[21]:


X = tfidf_df
y = df['label']


# Splitting the Dataset

# This command splits the dataset into training (80%) and testing (20%) sets using train_test_split. The random_state=42 ensures reproducibility, meaning the split will be the same every time the code runs.

# In[22]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Verifying Dataset Split

# This command prints the shapes of the training and testing sets, confirming the number of samples and features in each subset after splitting.

# In[23]:


print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)


# Training a Logistic Regression Model

# 
# This code initializes a Logistic Regression model with a maximum of 1000 iterations to ensure convergence. The model is then trained using the TF-IDF-transformed training data (X_train) and corresponding labels (y_train).

# In[24]:


log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)


# Making Predictions with Logistic Regression

# This code uses the trained Logistic Regression model to predict labels for the test dataset (X_test) and stores the results in y_pred_log_reg. The predicted values represent the model’s classification of mental health labels.

# In[25]:


y_pred_log_reg = log_reg.predict(X_test)
y_pred_log_reg


# In[26]:


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred_log_reg)
print("Logistic Regression Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred_log_reg))


# In[27]:


# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred_log_reg)
cm


# In[28]:


# Initialize and train the model
nn = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300)
nn.fit(X_train, y_train)


# In[29]:


# Make predictions
y_pred_nn = nn.predict(X_test)
y_pred_nn


# In[30]:


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred_nn)
print("Neural Network Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred_nn))


# In[31]:


# Initialize and train the model
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)


# In[32]:


# Make predictions
y_pred_rf = rf.predict(X_test)
y_pred_rf


# In[33]:


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred_rf))


# In[34]:


svm_model = SVC(kernel='linear',random_state=42)
svm_model.fit(X_train, y_train)


# In[35]:


y_pred = svm_model.predict(X_test)


# In[36]:


y_pred


# In[37]:


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred))


# In[ ]:





# In[39]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


# In[40]:


# Define the model
rf = RandomForestClassifier(random_state=42)


# In[41]:


# Define the hyperparameters to tune
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}


# In[42]:


# GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                           cv=3, scoring='accuracy', n_jobs=-1, verbose=2)


# In[ ]:


# Fit the model
grid_search.fit(X_train, y_train)


# In[ ]:


# Best parameters and accuracy
print("Best Parameters:", grid_search.best_params_)
print("Best Accuracy:", grid_search.best_score_)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




