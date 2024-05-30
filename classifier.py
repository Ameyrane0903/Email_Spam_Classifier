import numpy as np
import pandas as pd
# import chardet
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import nltk
import seaborn as sns

df = pd.read_csv('spam.csv', encoding='latin1')
df.drop(columns =['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace = True)
df.rename(columns={'v1':'target' , 'v2' : 'text'}, inplace = True)

encoder = LabelEncoder()

df['target'] = encoder.fit_transform(df['target'])
df.isnull().sum()
df.duplicated().sum()
df = df.drop_duplicates(keep ='first')

df['target'].value_counts()

plt.pie(df['target'].value_counts(), labels = ['ham','spam'], autopct = '%0.2f')
plt.show()

nltk.download('punkt')

df['no.characters'] = df['text'].apply(len)
df['no.words'] = df['text'].apply(lambda x :len(nltk.word_tokenize(x)))
df['no.sentences'] = df['text'].apply(lambda x :len(nltk.sent_tokenize(x)))
df[df['target'] == 0][['no.characters','no.words','no.sentences']].describe()
df[df['target'] == 1][['no.characters','no.words','no.sentences']].describe()

# Text Preprocessing
def transform_text(text):
  text = text.lower()                          #Lowercase
  text = nltk.word_tokenize(text)              #Tokenization

  y = []
  for i in text:                               #Considering only alphanumeric
    if i.isalnum():
      y.append(i)
  
  text = y[:]
  y.clear()
  
  for i in text :                             #Removing Stopwords and Punctuation
    if i not in stopwords.words('english') and i not in string.punctuation:
      y.append(i)
  text = y[:]
  y.clear()

  for i in text :                             #Stemming
    y.append(ps.stem(i))
    
  return " ".join(y)


nltk.download('stopwords')

from nltk.corpus import stopwords
stopwords.words('english')

import string
string.punctuation

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

df['transformed_text'] = df['text'].apply(transform_text)

from wordcloud import WordCloud
wc = WordCloud(width = 500, height = 500, min_font_size = 10, background_color ='white')

spam_wc = wc.generate(df[df['target'] == 1]['transformed_text'].str.cat(sep=" "))

plt.imshow(spam_wc)

ham_wc = wc.generate(df[df['target'] == 0]['transformed_text'].str.cat(sep=" "))

plt.imshow(ham_wc)

spam_corpus =[]
for msg in df[df['target'] == 1]['transformed_text'].tolist():
  for word in msg.split():
    spam_corpus.append(word)

len(spam_corpus)

from collections import Counter
sns.barplot(pd.DataFrame(Counter(spam_corpus).most_common(30))[1])


ham_corpus =[]
for msg in df[df['target'] == 0]['transformed_text'].tolist():
  for word in msg.split():
    ham_corpus.append(word)

len(ham_corpus)

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
cv = CountVectorizer()
tf = TfidfVectorizer(max_features = 3000) #max features is used to improve the model performance

# Tfidf Technique is used

X = tf.fit_transform(df['transformed_text']).toarray()
y = df['target'].values


# Naive Bayes Classifier is used for both CV and Tfidf
#**Precison for Tfidf is better than CV therefore we use Tfidf**


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2, random_state = 2)

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB 
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()

gnb.fit(X_train, y_train)
y_pred1 = gnb.predict(X_test)
print(accuracy_score(y_test, y_pred1))
print(confusion_matrix(y_test, y_pred1))
print(precision_score(y_test, y_pred1))

#**MNB gives the best precison**

mnb.fit(X_train, y_train)
y_pred2 = mnb.predict(X_test)
print(accuracy_score(y_test, y_pred2))
print(confusion_matrix(y_test, y_pred2))
print(precision_score(y_test, y_pred2))

bnb.fit(X_train, y_train)
y_pred3 = bnb.predict(X_test)
print(accuracy_score(y_test, y_pred3))
print(confusion_matrix(y_test, y_pred3))
print(precision_score(y_test, y_pred3))

import pickle
pickle.dump(tf,open('vectorizer.pkl','wb'))
pickle.dump(mnb,open('model.pkl','wb'))