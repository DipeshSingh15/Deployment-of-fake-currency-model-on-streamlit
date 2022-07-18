#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st


# In[2]:


st.title("Fake Currencies Detection")


# In[3]:


st.write("This model will decide the status of currencies ")


# In[4]:


import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn import tree
import seaborn as sns


# In[5]:


df=pd.read_csv("C:\\Users\\Dipesh Singh\\Downloads\\fake currency prediction.csv")
df


# In[6]:


st.header("Dataset used to train")
st.dataframe(df)


# In[7]:


st.header("Plots to explore dataset")
st.subheader("Line chart")
st.line_chart(df)
st.subheader("Area chart")
st.area_chart(df)
st.subheader("Bar chart")
st.bar_chart(df)


# In[8]:


st.set_option('deprecation.showPyplotGlobalUse', False)


# In[9]:


st.header("Heatmap")
df.corr()
sns.heatmap(df.corr(),annot=True)
st.pyplot()


# In[10]:


st.header("Pairplot")
sns.pairplot(df, hue='Target')
st.pyplot()


# In[11]:


plt.figure(figsize=(8,6))
plt.title('Distribution of Target', size=18)
sns.countplot(x=df['Target'])
target_count = df.Target.value_counts()
plt.annotate(s=target_count[0], xy=(-0.04,10+target_count[0]), size=14)
plt.annotate(s=target_count[1], xy=(0.96,10+target_count[1]), size=14)
plt.ylim(0,900)
st.pyplot()


# In[12]:


nb_to_delete = target_count[0] - target_count[1]
df = df.sample(frac=1, random_state=42).sort_values(by='Target')
df = df[nb_to_delete:]
print(df['Target'].value_counts())


# In[13]:


x = df.loc[:, df.columns != 'Target']
y = df.loc[:, df.columns == 'Target']


# In[14]:


from sklearn.preprocessing import StandardScaler


# In[15]:


sc=StandardScaler()
sc.fit(x)
x_train=sc.transform(x)


# In[ ]:





# In[16]:


var=st.slider("Variance of currency",min_value=-7.042100,max_value=6.824800)
skew=st.slider("skewness of currency",min_value=-13.773100,max_value=12.951600)
kurt=st.slider("kurtosis of currency",min_value=-5.286100,max_value=17.927400)
entr=st.slider("Entropy of currency",min_value=-8.548200,max_value=2.449500)


# In[17]:


x_test1={"variance":var,"skewness":skew,"kurtosis":kurt,"entropy":entr}
x_test=pd.DataFrame([x_test1])
x_test


# In[18]:


x_test=sc.transform(x_test)


# In[19]:


from sklearn import svm


# In[20]:


clf = svm.SVC()
clf.fit(x_train,y)


# In[21]:


y_pred=clf.predict(x_test)
y_pred


# In[22]:


if y_pred==1:
    st.write("Currency is fake")
else:
    st.write("Currency is original")


# In[ ]:




