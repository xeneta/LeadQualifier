import re
import cPickle
import xlrd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_recall_fscore_support
# from sklearn.naive_bayes import GaussianNB

from sklearn import neighbors
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

stemmer = SnowballStemmer('english')
stopWords = set(stopwords.words('english'))
vectorizer = CountVectorizer(
    analyzer='word', max_features=10000, ngram_range=(1, 3)
)
tfidf_transformer = TfidfTransformer(norm='l1', use_idf=True)


def getSheet(name):
    sheets = xlrd.open_workbook(name)
    sheet = sheets.sheet_by_index(0)
    return sheet


def cleanUp(descriptions):
    clean_descriptions = []
    index = 0
    for desc in descriptions:
        desc = re.sub("[^a-zA-Z]", " ", desc)
        stemmed_desc = []
        for word in desc.split():
            stemmed_word = stemmer.stem(word)
            stemmed_desc.append(stemmed_word)
        meaningful_desc = [
            word for word in stemmed_desc if word not in stopWords
        ]
        clean_descriptions.append(" ".join(meaningful_desc))
        index += 1
    return clean_descriptions


def getDescriptionsFromSheet(sheet):
    descriptions = []
    for index in xrange(1, sheet.nrows):
        description = sheet.cell_value(index, 1)
        descriptions.append(description)
    return descriptions

def getScores(clf, X, y):
    predictions = clf.predict(X)
    scores = precision_recall_fscore_support(y, predictions, average='binary')
    return scores

def transform(data, transformer):
    return transformer.transform(data).toarray()

def runForest(X_train, y_train):
    forest = RandomForestClassifier(n_estimators=90, random_state=42)
    forest.fit(X_train, y_train)
    return forest

# def runGaussianNB(X_train, X_test, y_train, y_test):
#    clf=GaussianNB()
#    clf.fit(X_train,y_train)
#    return clf

# def runKNN(X_train, X_test, y_train, y_test):
#     knn = neighbors.KNeighborsClassifier()
#     knn.fit(X_train, y_train)
#     score = knn.score(X_test, y_test)
#     return score

def createLabelArray(qualified, disqualified):
    labels = []
    for q in qualified:
        labels.append(1)
    for d in disqualified:
        labels.append(0)
    return np.array(labels)


qualified_sheet = getSheet('input/qualified.xlsx')
qualified_descriptions = getDescriptionsFromSheet(qualified_sheet)
qualified_clean_descriptions = cleanUp(qualified_descriptions)
disqualified_sheet = getSheet('input/disqualified.xlsx')
disqualified_descriptions = getDescriptionsFromSheet(disqualified_sheet)
disqualified_clean_descriptions = cleanUp(disqualified_descriptions)

X = qualified_clean_descriptions + disqualified_clean_descriptions
y = createLabelArray(
    qualified_clean_descriptions, disqualified_clean_descriptions
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Fit the vectorizer on the training data
vectorizer.fit(X_train)

# Vectorize all the data
X_train = transform(X_train, vectorizer)
X_test = transform(X_test, vectorizer)

# Fit the tfidf transformer on the vectorized training data
tfidf_transformer.fit(X_train)

# Tfidf transform all the data
X_train = transform(X_train, tfidf_transformer)
X_test = transform(X_test, tfidf_transformer)

# gnb, gnb_score=runGaussianNB(X_train, X_test, y_train, y_test)

forest = runForest(X_train, y_train)
forest_scores = getScores(forest, X_test, y_test)
print 'Random Forest scores: ', forest_scores

# print 'Random Gaussian Naive Bayes score: ', gnb_score

# Dump the algoruthms
with open('../qualify_leads/algorithms/forest', 'wb') as f:
    cPickle.dump(forest, f)

# with open('gnb', 'wb') as fi:
#    cPickle.dump(gnb, fi)

with open('../qualify_leads/algorithms/vectorizer', 'wb') as file:
    cPickle.dump(vectorizer, file)

with open('../qualify_leads/algorithms/tfidf_transformer', 'wb') as file:
    cPickle.dump(tfidf_transformer, file)
