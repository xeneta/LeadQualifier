import cPickle
import re
import numpy as np
import xlrd, xlwt
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.externals import joblib

stemmer = SnowballStemmer('english')
stopWords = set(stopwords.words('english'))

def getSheet(name):
    sheets = xlrd.open_workbook(name)
    sheet = sheets.sheet_by_index(0)
    return sheet

vectorizer = joblib.load("algorithms/vectorizer")
forest = joblib.load("algorithms/forest")
###gnb=joblib.load("algorithms/gnb")
tfidf_transformer = joblib.load("algorithms/tfidf_transformer")

def cleanUp(descriptions):
    clean_descriptions = []
    for desc in descriptions:
        desc = re.sub("[^a-zA-Z]", " ", desc)
        stemmed_desc = []
        for word in desc.split():
            stemmed_word = stemmer.stem(word)
            stemmed_desc.append(stemmed_word)
        meaningful_desc = [word.lower() for word in stemmed_desc if not word in stopWords]
        clean_descriptions.append(" ".join(meaningful_desc))
    return clean_descriptions

def getUrlsFromSheet(sheet):
    urls = []
    for index in xrange(1, sheet.nrows):
        url = sheet.cell_value(index, 0)
        urls.append(url)
    return urls

def getDescriptionsFromSheet(sheet):
    descriptions = []
    for index in xrange(1, sheet.nrows):
        description = sheet.cell_value(index, 1)
        descriptions.append(description)
    return descriptions

def transform(data, transformer):
    return transformer.transform(data).toarray()

def saveData(descriptions, urls, predictions):
    wb = xlwt.Workbook()
    ws = wb.add_sheet('Companies')
    ws.write(0, 0, 'Name')
    ws.write(0, 1, 'Description')
    ws.write(0, 2, 'Prediction')
    length = len(descriptions)
    for index in xrange(length):
        ws.write(index + 1, 0, urls[index])
        ws.write(index + 1, 1, descriptions[index])
        ws.write(index + 1, 2, predictions[index])
    wb.save('output/predictions.xls')

def qualifyLeads():
    sheet = getSheet('input/data.xlsx')
    descriptions = getDescriptionsFromSheet(sheet)
    urls = getUrlsFromSheet(sheet)
    clean_descriptions = cleanUp(descriptions)
    vectorized_descriptions = transform(clean_descriptions, vectorizer)
    transformed_descriptions = transform(vectorized_descriptions, tfidf_transformer)
    predictions = forest.predict(transformed_descriptions)
    ###predictions = gnb.predict(transformed_descriptions)
    saveData(descriptions, urls, predictions)

qualifyLeads()
