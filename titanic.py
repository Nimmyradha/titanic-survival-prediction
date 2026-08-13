#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_csv("train.csv")
df.head()


# In[2]:


df.info()


# In[3]:


df['Age'] = df['Age'].fillna(df['Age'].median())


# In[4]:


# Fill missing Age with the median age
df['Age'] = df['Age'].fillna(df['Age'].median())

# Fill missing Embarked with the most common value
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Drop Cabin (too many missing values to use reliably), plus columns not useful for a first model
df = df.drop(columns=['Cabin', 'Ticket', 'Name'])

df.info()


# In[5]:


# Convert Sex to numbers: male=0, female=1
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Convert Embarked into numeric columns (one-hot encoding)
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

df.head()


# In[6]:


# Separate features (inputs) from target (what we're predicting)
X = df.drop(columns=['Survived', 'PassengerId'])
y = df['Survived']

X.head()


# In[7]:


df[['Survived', 'Pclass', 'Sex', 'Age']].head()


# In[8]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape)
print(X_test.shape)


# In[9]:


from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)


# In[10]:


from sklearn.metrics import accuracy_score

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(accuracy)


# In[ ]:


# Titanic Survival Prediction

A simple end-to-end ML project predicting Titanic passenger survival using a Decision Tree Classifier.

**Steps:** data cleaning, feature encoding, train/test split, model training, evaluation.
**Result:** ~80% accuracy on the test set.

