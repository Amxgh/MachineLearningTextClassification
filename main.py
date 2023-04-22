import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

import re, string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from pattern.text.en import lemma


def lemmat(text):
    return " ".join([lemma(wd) for wd in text.split()])


def preprocess(text):
    text = text.lower()
    text = text.strip()
    text = re.compile('[^a-zA-Z0-9%$]').sub(' ', text)
    text = re.sub(r'[^\w\s%$]', '', str(text).lower().strip())
    text = re.sub(r'\s+', ' ', text)
    return text


# STOPWORD REMOVAL
def stopword(string):
    a = [i for i in string.split() if i not in stopwords.words('english')]
    return ' '.join(a)

# Creating a dataframe from the tsv file
cw = pd.read_csv('checkworthy_labeled.tsv',sep = '\t')


cw["Text"] = ((cw["Text"].apply(preprocess)).apply(stopword))
print(cw)

cw.drop(['Sentence_id'], axis=1, inplace=True)
# Train Test
train, test = train_test_split(cw, train_size=0.8, shuffle=True)

# Tokenizing text
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(train.Text)

# Term Frequencies
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)

# Term Frequency times Inverse Document Frequency
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = MultinomialNB(alpha=0.03, force_alpha=True).fit(X_train_tfidf, train.class_label)

checking = pd.read_csv('checkworthy_eval.tsv', sep="\t")
checking["Text"] = ((checking["Text"].apply(preprocess)).apply(stopword))


X_new_counts = count_vect.transform(checking.Text)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)

print(predicted)
checking["Category"] = predicted

checking.drop(['Text'], axis=1, inplace=True)
checking.rename(columns = {'Sentence_id':'Id'}, inplace = True)
print(checking)
gfg_csv_data = checking.to_csv('checkworthy_eval_prediction.csv', index = False)