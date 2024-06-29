
import numpy as np
import seaborn as sns
import pandas as pd
import string
from nltk.corpus import stopwords 
from nltk.stem import SnowballStemmer
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier, StackingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score
import nltk
from wordcloud import WordCloud

nltk.download('stopwords')
stemmer = SnowballStemmer(language='english')

file_path = 'C:\\Users\\divya dharshini\\Downloads\\spam100.csv'
df = pd.read_csv(file_path, encoding='latin1')

print(df.head())
print(df.sample(5))
print(df.shape)

'''=======DATA CLEANING======'''

print(df.info())

df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)
print(df.sample(5))

df.rename(columns={'v1':'target','v2':'text'},inplace=True)
print(df.sample(5))


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

df['target'] = encoder.fit_transform(df['target'])
print(df.head())

print(df.isnull().sum())
df.duplicated().sum()
df = df.drop_duplicates(keep='first')
df.duplicated().sum()
df.shape

'''==============EDA================'''

df.head()
df['target'].value_counts()

import matplotlib.pyplot as plt
plt.pie(df['target'].value_counts(), labels=['ham','spam'],autopct="%0.2f")
plt.show()


import nltk
nltk.download('punkt')

df['num_characters'] = df['text'].apply(len)
df.head()

df['num_words'] = df['text'].apply(lambda x:len(nltk.word_tokenize(x)))
df.head()

df['num_sentences'] = df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))
df.head()

df[['num_characters','num_words','num_sentences']].describe()

df[df['target'] == 0][['num_characters','num_words','num_sentences']].describe()

df[df['target'] == 1][['num_characters','num_words','num_sentences']].describe()

numeric = df[['target', 'num_characters', 'num_words', 'num_sentences']]



import seaborn as sns
plt.figure(figsize=(12,6))
sns.histplot(df[df['target'] == 0]['num_characters'])
sns.histplot(df[df['target'] == 1]['num_characters'],color='red')
plt.legend()
plt.title('Distribution of Number of Characters')
plt.xlabel('Number of Characters')
plt.show()

plt.figure(figsize=(12,6))
sns.histplot(df[df['target'] == 0]['num_words'])
sns.histplot(df[df['target'] == 1]['num_words'],color='red')
plt.legend()
plt.title('Distribution of Number of Words')
plt.xlabel('Number of Words')
plt.show()

sns.pairplot(df,hue='target')
plt.title('Pairplot')
plt.show()


'''
sns.heatmap(df.corr(),annot=True)
plt.show()
'''

sns.heatmap(numeric.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

'''==================DATA PRE-PROCESSING==========================='''

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    y=[]
    for i in text:
        word = stemmer.stem(i)
        y.append(word)
    
            
    return " ".join(y)

transform_text("I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today.")

df['text'][10]

from nltk.stem.porter import PorterStemmer
from collections import Counter
ps = PorterStemmer()
ps.stem('loving')

df['transformed_text'] = df['text'].apply(transform_text)
df.head()



wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
spam_wc = wc.generate(df[df['target'] == 1]['transformed_text'].str.cat(sep=" "))
plt.figure(figsize=(15, 6))
plt.imshow(spam_wc)
plt.title('Spam Word Cloud')
plt.axis('off')
plt.show()

ham_wc = wc.generate(df[df['target'] == 0]['transformed_text'].str.cat(sep=" "))
plt.figure(figsize=(15, 6))
plt.imshow(ham_wc)
plt.title('Ham Word Cloud')
plt.axis('off')
plt.show()
#-----------------------------------------------------------------------------------------

spam_corpus = [word for msg in df[df['target'] == 1]['transformed_text'].tolist() for word in msg.split()]
ham_corpus = [word for msg in df[df['target'] == 0]['transformed_text'].tolist() for word in msg.split()]

sns.barplot(x=pd.DataFrame(Counter(spam_corpus).most_common(30))[0], y=pd.DataFrame(Counter(spam_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.title('Top 30 Words in Spam Messages')
plt.show()

sns.barplot(x=pd.DataFrame(Counter(ham_corpus).most_common(30))[0], y=pd.DataFrame(Counter(ham_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.title('Top 30 Words in Ham Messages')
plt.show()

df.head()

'''==========================MODEL BUILDING======================'''

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features=3000)

X = tfidf.fit_transform(df['transformed_text']).toarray()
X.shape

y = df['target'].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score
gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()
gnb.fit(X_train,y_train)
y_pred1 = gnb.predict(X_test)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1))

mnb.fit(X_train,y_train)
y_pred2 = mnb.predict(X_test)
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(precision_score(y_test,y_pred2))

bnb.fit(X_train,y_train)
y_pred3 = bnb.predict(X_test)
print(accuracy_score(y_test,y_pred3))
print(confusion_matrix(y_test,y_pred3))
print(precision_score(y_test,y_pred3))

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

svc = SVC(kernel='sigmoid', gamma=1.0)
knc = KNeighborsClassifier()
mnb = MultinomialNB()
dtc = DecisionTreeClassifier(max_depth=5)
lrc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=50, random_state=2)
abc = AdaBoostClassifier(n_estimators=50, random_state=2)
bc = BaggingClassifier(n_estimators=50, random_state=2)
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=50,random_state=2)
xgb = XGBClassifier(n_estimators=50,random_state=2)

clfs = {
    'SVC': svc,
    'KN': knc, 
    'NB': mnb, 
    'DT': dtc, 
    'LR': lrc, 
    'RF': rfc, 
    'AdaBoost': abc, 
    'BgC': bc, 
    'ETC': etc,
    'GBDT': gbdt,
    'xgb': xgb
}

accuracy_scores = []
precision_scores = []

def train_classifier(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    return accuracy, precision

for name, clf in clfs.items():
    current_accuracy, current_precision = train_classifier(clf, X_train, y_train, X_test, y_test)
    print("For ", name)
    print("Accuracy - ", current_accuracy)
    print("Precision - ", current_precision)
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)


print("Length of clfs.keys():", len(clfs.keys()))
print("Length of accuracy_scores:", len(accuracy_scores))
print("Length of precision_scores:", len(precision_scores))


min_length = min(len(clfs), len(accuracy_scores), len(precision_scores))
clfs_keys = list(clfs.keys())[:min_length]
accuracy_scores = accuracy_scores[:min_length]
precision_scores = precision_scores[:min_length]


performance_df = pd.DataFrame({
    'Algorithm': clfs_keys,
    'Accuracy': accuracy_scores,
    'Precision': precision_scores
}).sort_values('Precision', ascending=False)
print(performance_df)

performance_df1 = pd.melt(performance_df, id_vars="Algorithm")
print(performance_df1)

sns.catplot(x='Algorithm', y='value', hue='variable', data=performance_df1, kind='bar', height=5)
plt.ylim(0.5, 1.0)
plt.show()


temp_df_3000 = pd.DataFrame({
    'Algorithm': clfs_keys,
    'Accuracy_max_ft_3000': accuracy_scores,
    'Precision_max_ft_3000': precision_scores
}).sort_values('Precision_max_ft_3000', ascending=False)

temp_df_scaling = pd.DataFrame({
    'Algorithm': clfs_keys,
    'Accuracy_scaling': accuracy_scores,
    'Precision_scaling': precision_scores
}).sort_values('Precision_scaling', ascending=False)

temp_df_chars = pd.DataFrame({
    'Algorithm': clfs_keys,
    'Accuracy_num_chars': accuracy_scores,
    'Precision_num_chars': precision_scores
}).sort_values('Precision_num_chars', ascending=False)


new_df = performance_df.merge(temp_df_3000, on='Algorithm').merge(temp_df_scaling, on='Algorithm').merge(temp_df_chars, on='Algorithm')
print(new_df)


svc = SVC(kernel='sigmoid', gamma=1.0, probability=True)
mnb = MultinomialNB()
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)

voting = VotingClassifier(estimators=[('svm', svc), ('nb', mnb), ('et', etc)], voting='soft')
voting.fit(X_train, y_train)

y_pred = voting.predict(X_test)
print("Voting Classifier Accuracy:", accuracy_score(y_test, y_pred))
print("Voting Classifier Precision:", precision_score(y_test, y_pred))





