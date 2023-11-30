"""
    Wesley Yang, 11/28/2023
    Sci-kit learn tutorial used for reference: https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html#evaluation-of-the-performance-on-the-test-set
"""

import csv
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn import metrics

# prevent field size too large error when reading csv file
# may not work on all operating systems
csv.field_size_limit(2147483647)

# import dataset and split into feature and class arrays
with open('steam_reviews_smol.csv', newline='', encoding='utf8') as f:
    csvreader = csv.reader(f)
    features, classes = [], []
    next(csvreader)     # remove header
    print('Loading "steam_reviews_smol.csv...\n')
    progress = 0
    for review in csvreader:
        features.append(review[0])
        classes.append(review[1])
        progress += 1
        if progress % 1000000 == 0:
            print(f'{progress:,} lines read')
    print('\nLoading complete\n')

# split into training and testing sets
features_train, features_test, classes_train, classes_test = train_test_split(features, classes)

# create a pipeline to vectorize the reviews, transform them using tf-idf, and create the classifier
review_classifier = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('transformer', TfidfTransformer()),
    ('classifier', MultinomialNB())
])

# train the classifier
print('Training Multinomial Naive Bayes Classifier...')
review_classifier.fit(features_train, classes_train)
print('Tranining complete')

# test the classifier
predictions = review_classifier.predict(features_test)
print(metrics.classification_report(classes_test, predictions))
