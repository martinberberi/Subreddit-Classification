# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Project 3 Subreddit Classsification Using Natural Language Processing (NLP)
## Distinguishing Between Subreddit Posts From the r/Patriots & r/NFL

* Problem Statement
* Methodology
* Data Dictionary
* Model Selection & Evaluation
* Conclusions

### Problem Statement
Patriots & NFL subreddits posts were gathered from Reddit. This project attempts to detect Patriots posts from NFL posts using Natural Language Processing (NLP) and building classification models with data from both subreddits. As a fan of New England Patriots team I would like to see how Patriots posts are filtered from NFL posts.

### Methodology

#### Data Gathering
I used Pushshift Api to scrape data from Patriots and NFL subreddits in order to create a classification model to detect posts from each subreddit. In total i scrapped 39,989 posts from both subreddits: 19,993 from Patriots subreddit and 19,996 from NFL subreddit.

#### Exploratory Data Analysis
Investigate data and create data visualization to observe patterns and distinguish each category characteristics

#### Natural Language Processing
Prepare the data for modeling. After cleaned by duplicates, punctuation & Vectorize the data

#### Modeling
Find the combination of model and vectorizer for the best accuracy score.

#### Data Dictionary
|Feature|Type|Dataset|Description|
|---|---|---|---|
|**subreddit**|object|final_df|which subreddit the post belong to|
|**title**|object|final_df|title of particular subreddit post|
|**author**|object|final_df|author of the post|
|**domain**|object|final_df|domain referenced to the post|
|**created_utc**|datetime|final_df|date and time post created|

### Model Selection

#### Model 1
CountVectorizer(stop_words=None, ngram_range=(1,3), max_df = .75, min_df = 2, lowercase = True, binary = True)
LogisticRegression(C=0.35, solver='liblinear')

Train score 0.95
Test score 0.86

#### Model 2
TfidVectorizer(stop_words=None, ngram_range=(1,2), max_df = .75, min_df = 2, lowercase = True, binary = True)
LogisticRegression(C=1, solver='liblinear')

Train score 0.92
Test score 0.86

#### Model 3
CountVectorizer(stop_words=None, ngram_range=(1,2), max_df = .75, min_df = 2, lowercase = True, binary = True)
MultinomialNB()

Train score 0.91
Test score 0.85

#### Model 4
TfidVectorizer(stop_words=None, ngram_range=(1,2), max_df = .75, min_df = 2, lowercase = True, binary = True)
MultinomialNB()

Train score 0.91
Test score 0.85

#### Model Evaluation
Throug Pipeline and Gridsearch, I tested four different combination of 2 models and 2 Vectorizers with hyper-parameters. My best model in terms of highest accuracy score was Model 1. Also for interpreting the coefficients the Logistic Regression and CounterVectorizer i used.
* Accuracy score: 86.17%
* Precision score: 84.87%
* Sensitivity score: 88.47%
* Specificity score: 83.8%
* Missclafication score: 13.83%

### Conclusions
The best model to distinguish the patriots post from nfl post is Logistic Regression with CounterVectorizer with accuracy score of 86%.(86% of the posts are predicted correctly). Even though the good accuracy score is not the perfect model because of overfitting issue. There is a disadvantage as well, most of the post description are images and videos and for that reason i trained the model only in title.
