#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


movies  = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')


# In[3]:


movies.head()


# In[4]:


credits.head()


# In[5]:


credits.head(1)['cast'].values


# In[6]:


credits.head(1)['crew'].values


# In[7]:


movies.merge(credits,on='title')


# In[8]:


movies.shape


# In[9]:


credits.shape


# In[10]:


movies = movies.merge(credits,on='title')


# In[11]:


movies.head()


# In[12]:


#genres
#id
#keywords
#title 
#overview
#cast
#crew
movies = movies[['id','genres','keywords','title','overview','cast','crew']]


# In[13]:


movies.info()


# In[14]:


movies.head()


# In[15]:


movies.isnull().sum()


# In[16]:


movies.dropna(inplace = True)


# In[17]:


movies.duplicated().sum()


# In[18]:


movies.iloc[0].genres


# In[ ]:


#'[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]'
#['Action','Adventure','FFantasy','SciFi']


# In[19]:


import ast


# In[20]:


def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])   
    return L


# In[96]:


[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]


# In[21]:


movies['genres'] = movies['genres'].apply(convert)


# In[22]:


movies.head()


# In[23]:


movies['keywords'] = movies['keywords'].apply(convert)


# In[24]:


movies.head()


# In[25]:


def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter !=3 :   
            L.append(i['name'])
            counter+=1
        else :
            break
    return L


# In[26]:


movies['cast'] = movies['cast'].apply(convert3)


# In[27]:


movies.head()


# In[28]:


def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] =='Director':
            L.append(i['name'])
            break
    return L


# In[29]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[30]:


movies.head()


# In[31]:


movies['overview'][0]


# In[32]:


movies['overview'] = movies['overview'].apply(lambda x :x.split())


# In[33]:


movies.head()


# In[34]:


movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ", "")for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ", "")for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ", "")for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ", "")for i in x])


# In[35]:


movies.head()


# In[36]:


movies['tags'] = movies['overview']  + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# In[37]:


movies.head()


# In[38]:


new_df = movies[['id','title','tags']]


# In[39]:


new_df


# In[40]:


new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))


# In[41]:


new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())


# In[42]:


new_df.head()


# In[43]:


new_df.head()


# In[50]:


import nltk


# In[51]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[52]:


def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


# In[53]:


new_df['tags'] = new_df['tags'] = new_df['tags'].apply(stem)


# In[46]:


vectors = cv.fit_transform(new_df['tags']).toarray()


# In[45]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 2000, stop_words ='english')


# In[47]:


cv.get_feature_names()


# In[ ]:





# In[54]:


cv.fit_transform(new_df['tags']).toarray()


# In[55]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 2000, stop_words ='english')


# In[56]:


vectors = cv.fit_transform(new_df['tags']).toarray()


# In[57]:


vectors


# In[58]:


cv.get_feature_names()


# In[73]:


from sklearn.metrics.pairwise import cosine_similarity


# In[60]:


cosine_similarity(vectors).shape


# In[74]:


similarity = cosine_similarity(vectors)


# In[75]:


similarity


# In[63]:


similarity[0]


# In[64]:


sorted(list(enumerate(similarity[0])), reverse =True, key = lambda x:x[1])[1:6]


# In[ ]:





# In[65]:


def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse =True, key = lambda x:x[1])[1:6]
    for i in movies_list:
        print(new_df.iloc[i[0]].title)


# In[78]:


recommend('Avatar')


# In[81]:


new_df


# In[67]:


new_df.iloc[12].title


# In[68]:


import pickle


# In[71]:


pickle.dump(new_df.to_dict(), open('movies_dict.pkl','wb'))


# In[77]:


pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:





# In[ ]:





# In[ ]:




