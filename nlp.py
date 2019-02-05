
# coding: utf-8

# In[1]:


# importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
import re


# In[2]:


# importing the datasets
train =pd.read_csv("train.csv")
test =pd.read_csv("test.csv")


# In[3]:


train.head()


# In[4]:


# The data has 3 columns id,label and tweet.label is the binary target variable and tweet contains the tweets that we will clean and process.
# combining the train and test datasets so that no need of doing the preprocessing steps twice.
combine = train.append(test,ignore_index = True )


# In[5]:


# from the train dataset we observe that the tweets are with #'s.
combine.head()


# In[6]:


# preprocessing the tweets
# defining a function for preprocessing
import string
def processTweet2(tweet):
   # process the tweets
   
   # Convert to lower case
    tweet = tweet.lower()
   # Convert www.* https?://* to URL

    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #trim
    tweet = tweet.strip('\'"')
    return tweet    


# In[7]:


combine['tweet'] = np.vectorize(processTweet2)(combine['tweet'])
combine['tweet'].head()


# In[8]:


# Removing Punctuations,numbers and special characters
combine['tweet'] = combine['tweet'].str.replace("[^a-zA-Z#]", " ")
 


# In[9]:


combine['tweet'].head()


# In[10]:


# Removing the short words
combine['tweet'] = combine['tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))


# In[11]:


combine['tweet'].head()


# In[12]:


combine.head()


# In[13]:


# Tokenization
tokenize_tweet = combine['tweet'].apply(lambda x: x.split())
tokenize_tweet.head()


# In[14]:


# stemming
from nltk.stem.porter import *
stemmer = PorterStemmer()

tokenize_tweet = tokenize_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming
tokenize_tweet.head()


# In[15]:


for i in range(len(tokenize_tweet)):
    tokenize_tweet[i] = ' '.join(tokenize_tweet[i])

combine['tweet'] = tokenize_tweet


# In[16]:


combine.head()


# In[17]:


#EDA
# understanding the common words in the tweets by using word cloud 
all_words = ' '.join([text for text in combine['tweet']])
from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


# In[18]:


# from the above plot we obseve that twetter and appl are most frequent words which are positive tweets
# now we will plot word cloud separately for both positive and negative classes
# plotting word cloud for positive class
positive_words =' '.join([text for text in combine['tweet'][combine['label'] == 0]])

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(positive_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()




# In[19]:


# from the above plot we observe that twitter and iphon words are most frequent ones
# plotting word cloud for negative class
negative_words = ' '.join([text for text in combine['tweet'][combine['label'] == 1]])
wordcloud = WordCloud(width=800, height=500,
random_state=21, max_font_size=110).generate(negative_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


# In[20]:


#from the above plot we have a good pretty text data to work on
# Extracting the features from the cleaned tweets
# 1. Bag of words
from sklearn.feature_extraction.text import CountVectorizer
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
# bag-of-words feature matrix
bow = bow_vectorizer.fit_transform(combine['tweet'])


# In[21]:


# Train and test split
from sklearn.model_selection import train_test_split
train_bow = bow[:7920,:]
test_bow = bow[7920:,:]


# In[22]:


# splitting data into training and validation set
xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow, train['label'], random_state=42, test_size=0.3)


# In[23]:


# Building Model i.e., Logistic Regression
from sklearn.linear_model import LogisticRegression


# In[24]:


lreg = LogisticRegression()


# In[25]:


# training the model
lreg.fit(xtrain_bow, ytrain)


# In[26]:


# predicting validation dataset
prediction = lreg.predict_proba(xvalid_bow)


# In[27]:


# if prediction is greater than or equal to 0.3 than 1 else 0
prediction_int = prediction[:,1] >= 0.3
prediction_int = prediction_int.astype(np.int)


# In[28]:


#calculating f1 score
from sklearn.metrics import f1_score

f1_score(yvalid, prediction_int) 


# In[29]:


# predicting on test dataset
test_pred = lreg.predict_proba(test_bow)
test_pred_int = test_pred[:,1] >= 0.3
test_pred_int = test_pred_int.astype(np.int)
test['label'] = test_pred_int
submission = test[['id','label']]


# In[30]:


# writing a data to csv file
submission.to_csv('lreg_sub_bow.csv', index=False)


# In[31]:


# Extracting the features from the clean tweets
#2 TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
# TF-IDF feature matrix
tfidf = tfidf_vectorizer.fit_transform(combine['tweet'])


# In[32]:


# splitting train and test data
train_tfidf = tfidf[:7920,:]
test_tfidf = tfidf[7920:,:]


# In[33]:


xtrain_tfidf = train_tfidf[ytrain.index]
xvalid_tfidf = train_tfidf[yvalid.index]


# In[34]:


# applying logistic regression
lreg.fit(xtrain_tfidf, ytrain)


# In[35]:


# predicting validation dataset
predict_val= lreg.predict_proba(xvalid_tfidf)


# In[36]:


# if predict_val is greater than or equal to 0.3 than 1 else 0
prediction_int = predict_val[:,1] >= 0.3
prediction_int = prediction_int.astype(np.int)


# In[37]:


# calculating f1 score
f1_score(yvalid, prediction_int)


# In[38]:


# predicting on test dataset
test_tfidf_pred = lreg.predict_proba(test_tfidf)
test_tfidf_pred_int = test_tfidf_pred[:,1] >= 0.3
test_tfidf_pred_int = test_tfidf_pred_int.astype(np.int)
test['label'] = test_tfidf_pred_int
submission = test[['id','label']]


# In[42]:


########### Naive Bayes Model ####################
# using bag of words
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(xtrain_bow,ytrain)


# In[43]:


# Validating the model
preds = nb.predict(xvalid_bow)


# In[44]:


# evaluating the model 
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(yvalid,preds))
print('\n')
print(classification_report(yvalid,preds))



# In[50]:


# predicting on test dataset
test_pred = nb.predict(test_bow)
test_pred_int = test_pred_int.astype(np.int)
test['label'] = test_tfidf_pred_int
submission = test[['id','label']]


# In[53]:


# using tf-idf
nb.fit(xtrain_tfidf,ytrain)


# In[54]:


# validating the model
pred = nb.predict(xvalid_tfidf)


# In[56]:


# evaluating the model
print(confusion_matrix(yvalid,pred))
print('\n')
print(classification_report(yvalid,pred))




