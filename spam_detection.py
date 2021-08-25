#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd


# In[35]:


df = pd.read_csv("spam.csv")
df.head()


# In[36]:


df.groupby("category").describe()


# In[37]:



df['spam']=df['category'].apply(lambda x: 1 if x=='spam' else 0)
df.head()


# In[38]:


df.drop(["Unnamed: 2","Unnamed: 3","Unnamed: 4"],axis='columns',inplace=True)
df.head(10)


# In[39]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.message,df.spam,test_size=0.25)


# In[40]:



from sklearn.feature_extraction.text import CountVectorizer
v = CountVectorizer()
X_train_count = v.fit_transform(X_train.values)
X_train_count.toarray()[:2]


# In[41]:


from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train_count,y_train)


# In[42]:


emails = [
    'Hey mohan, can we get together to watch footbal game tomorrow?',
    'Upto 20% discount on parking, exclusive offer just for you. Dont miss this reward!'
]
emails_count = v.transform(emails)
model.predict(emails_count)


# In[43]:


X_test_count = v.transform(X_test)
model.score(X_test_count, y_test)


# In[ ]:




