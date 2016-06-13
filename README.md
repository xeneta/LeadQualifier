# LeadQualifier

This repo is a collection of scripts we use at Xeneta to qualify sales leads with machine learning. Read more about this project in the Medium article [Boosting Sales With Machine Learning](https://medium.com/xeneta/boosting-sales-with-machine-learning-fbcf2e618be3).

Our simple [scikit-learn Random Forest](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) algorithm obtains an accuracy of 86,4% on the testing data.

Unfortunately, we can not provide you with the raw text dataset we've used to train our algorithm, as it contains sensitive company information (who our customers are). However, we have added our own vectorized & transformed data in the [xeneta_dataset](https://github.com/xeneta/LeadQualifier/tree/master/xeneta_dataset) folder (which is the final input to the RF algorithm). Feel free to play around with this data and see if you can beat our accuracy.


To run create your own lead qualifier, you'll need to run these scripts on your own customer data, which you can do using the FullContact API.

## Setup

Start off by running the following command:

    pip install -r requirements.txt

You'll also need to download the stopword from the [nltk](http://www.nltk.org/index.html) package. Run the Python interpreter and type the following:

    import nltk
    nltk.download('stopwords')

## [Train Algorithm](https://github.com/xeneta/LeadQualifier/tree/master/train_algorithm)

This script trains an algorithm on **your own** input data (not ours). It expects two excel sheets named **qualified** and **disqualified** in the [input](https://github.com/xeneta/LeadQualifier/tree/master/train_algorithm/input) folder. These sheets need to contain two columns:

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

## [Beat Our Results](https://github.com/xeneta/LeadQualifier/tree/master/xeneta_qualifier)

If you want to play around with the Xeneta dataset, we've provided you with our vectorized and transformed data together with a script where you can easily test your own algorithms.

Run the script:

    python run.py

Got questions? Email me at per@xeneta.com.
