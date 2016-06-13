import re
import cPickle
import xlrd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.cross_validation import train_test_split
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


def cleanUp(descriptions, disc=False):
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


def transformByTfIdf(X_train, X_test, transformer):
    X_train = transformer.transform(X_train).toarray()
    X_test = transformer.transform(X_test).toarray()
    return X_train, X_test


def transformByVectorizer(X_train, X_test, vectorizer):
    X_train = vectorizer.transform(X_train).toarray()
    X_test = vectorizer.transform(X_test).toarray()
    return X_train, X_test


def vectorize(descriptions):
    vectorized_descriptions = vectorizer.fit_transform(descriptions)
    return vectorized_descriptions.toarray(), vectorizer


def runForest(X_train, X_test, Y_train, Y_test):
    forest = RandomForestClassifier(n_estimators=50, random_state=1)
    forest = forest.fit(X_train, Y_train)
    score = forest.score(X_test, Y_test)
    return score, forest

# def runGaussianNB(X_train, X_test, Y_train, Y_test):
#    clf=GaussianNB()
#    clf.fit(X_train,Y_train)
#    score=clf.score(X_test, Y_test)
#    return clf, score

def runKNN(X_train, X_test, Y_train, Y_test):
    knn = neighbors.KNeighborsClassifier()
    knn.fit(X_train, Y_train)
    score = knn.score(X_test, Y_test)
    return score


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
disqualified_clean_descriptions = cleanUp(disqualified_descriptions, True)

X = qualified_clean_descriptions + disqualified_clean_descriptions
Y = createLabelArray(
    qualified_clean_descriptions, disqualified_clean_descriptions
)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42
)
vectorizer.fit(X_train)
X_train, X_test = transformByVectorizer(X_train, X_test, vectorizer)
tfidf_transformer.fit(X_train)
X_train, X_test = transformByTfIdf(X_train, X_test, tfidf_transformer)

# gnb, gnb_score=runGaussianNB(X_train, X_test, Y_train, Y_test)
forest_score, forest = runForest(X_train, X_test, Y_train, Y_test)

print 'Random Forest score: ', forest_score
# print 'Random Gaussian Naive Bayes score: ', gnb_score

with open('../qualify_leads/algorithms/forest', 'wb') as f:
    cPickle.dump(forest, f)

# with open('gnb', 'wb') as fi:
#    cPickle.dump(gnb, fi)

with open('../qualify_leads/algorithms/vectorizer', 'wb') as file:
    cPickle.dump(vectorizer, file)

with open('../qualify_leads/algorithms/tfidf_transformer', 'wb') as file:
    cPickle.dump(tfidf_transformer, file)
