# LeadQualifier

This repo is a collection of scripts we use at Xeneta to **qualify sales leads** with machine learning. Read more about this project in the Medium article [Boosting Sales With Machine Learning](https://medium.com/xeneta/boosting-sales-with-machine-learning-fbcf2e618be3).

You can use this repo for **two things:**

1. Use **our data** to experiment with your own algorithms
2. Create a lead qualifier for your company, using **your own data**

## Setup

Start off by running the following command:

    pip install -r requirements.txt

You'll also need to download the stopword from the [nltk](http://www.nltk.org/index.html) package. Run the Python interpreter and type the following:

    import nltk
    nltk.download('stopwords')

# 1. Experiment with your own algorithms

We'd love to see more algorithms on the leaderboard, so send us a pull request once you've implemented one.

## [Xeneta Qualifier](https://github.com/xeneta/LeadQualifier/tree/master/xeneta_qualifier)

We've provided you with our vectorized and transformed data [here](https://github.com/xeneta/LeadQualifier/tree/master/xeneta_qualifier/data). We can unfortunately not share the raw text data, as it contains sensitive company information (who our customers are).

To test our your own algorithm, simply add it the [run.py](https://github.com/xeneta/LeadQualifier/blob/master/xeneta_qualifier/run.py) file and run the script:

    python run.py

Thanks to [lampts](https://github.com/lampts) for implementing the best performing algorithm so far, the [SGDClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html).

### Leaderboard:

| Algorithm     | Precision     | Recall | F1 Score|
| ------------- |:--------------| :------|:--------|
| SGD Classifier| 0.872         | 0.940  | 0.905   |
| Random Forest | 0.845         | 0.915  | 0.878   |



# 2. Create your own lead qualifier

To create your own lead qualifier, you'll need to get hold of company descriptions (to create your dataset). We currently use [FullContact](https://www.fullcontact.com/developer) for this. 

## [Train Algorithm](https://github.com/xeneta/LeadQualifier/tree/master/train_algorithm)

This script trains an algorithm on **your own** input data. It expects two excel sheets named **qualified** and **disqualified** in the [input](https://github.com/xeneta/LeadQualifier/tree/master/train_algorithm/input) folder. These sheets need to contain two columns:

- URL
- Description


![](https://raw.githubusercontent.com/xeneta/LeadQualifier/master/img/sheet.png)

Run the script:

    python run.py

It'll dump three files into the [qualify_leads](https://github.com/xeneta/LeadQualifier/tree/master/qualify_leads) project:

- algorithm
- vectorizer
- tfidf_vectorizer

You're now ready to start classifying your sales leads!

## [Qualify Leads](https://github.com/xeneta/LeadQualifier/tree/master/qualify_leads)

This is the script that actually predicts the quality of your leads. Add an excel sheet named **data** in the [input](https://github.com/xeneta/LeadQualifier/tree/master/qualify_leads/input) folder. Use the same format as the example file that's already there.

Run the script:

    python run.py

It'll output an excel sheet with a column named **Prediction**, where 1 equals *qualified* and 0 equals *disqualified*:

![](https://raw.githubusercontent.com/xeneta/LeadQualifier/master/img/predictions_sheet.png)

Got questions? Email me at per@xeneta.com.
