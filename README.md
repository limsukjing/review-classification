# Review Classification

This is a simple Python program designed to classify user-submitted reviews into their respective categories and obtain a more concise representation 
of consumer opinions by utilizing text analysis techniques.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development purposes.

### Prerequisites

If you already have Python 2 or Python 3 installed, you're good to go! If not, install and set up Python 3 on your machine using the packages
available [here](https://www.python.org/getit/). Have a look at the following links:

* [Linux/UNIX](https://docs.python.org/3/using/unix.html)
* [macOS](https://docs.python.org/3/using/mac.html)
* [Windows](https://docs.python.org/3/using/windows.html)

### Installing

Use Git from the command line to clone this repo to your local machine:  

```
$ git clone https://github.com/limsukjing/review-classification.git
```

To run the main script: 

```
python classification.py 
```

## Algorithm

1. **Data preparation:** a total of 200 documents representing the positive and negative class are scraped from two main review websites, i.e. 
[Rotten Tomatoes](https://www.rottentomatoes.com/) and [50 Best](https://www.theworlds50best.com/) by running the `scraper.py` file. 
2. **Data pre-processing:** as text data scraped from websites are highly unstructured and polarized, basic cleaning such as discarding HTML tags
that contain irrelevant texts and removing control characters should be performed before writing the data to a CSV file. 
3. **Data understanding:** generate descriptive statistics using `data.describe()` and a word cloud to identify potential stop words, as well as
specific terms that are the most promising predictor of the positive/negative class. 
4. **Data cleaning/normalization:** features that are not optimal for analysis should be replaced or removed as the corpus contains a massive number 
of columns. A few of the NLP techniques are used repeatedly throughout this process and they are as follows:
    - **Tokenization:** split a document into a list of tokens. 
    - **POS tagging:** assign specific lexical categories to words based on the contexts in which they occur.
    - **Named-entity recognition (NER):** identify proper nouns (NNP) in certain categories such as names of actors/actresses and locations. 
5. **Text Vectorization:** the process of representing a corpus as vectors, i.e. feature extraction. It is an essential step in text analysis as it
reduces the dimensionality of the data without losing important information.
6. **Feature Reduction/selection:** irrelevant and redundant features should be removed to maximize the performance of the models, hence the vectors 
generated in **Step 5** are further reduced to a subset of 12 using feature selection. 
7. **Text Mining/modelling:** the SVM and Naïve Bayes are the most suitable algorithms for this particular classification task as the output data
type is binomial, i.e. **movie_review (positive)** and **not_movie_review (negative)**, and they generally work well with text classification
problems.
8. **Model Evaluation:** the dataset has to be split into two subsets — one subset is used for training the model, while the other subset is used to 
evaluate the performance of the classification model in terms of its accuracy, precision and recall. 
    - **Validation:** When it comes to generating a training/test dataset, the Cross Validation method appears to be a better approach as it allows 
    the model to train recursively on multiple train/test splits, as opposed to the Split Validation method which relies on a single split.

## Built With

* [beautifulsoup4](https://pypi.org/project/beautifulsoup4/) - A library that makes it easy to scrape information from web pages.
* [Matplotlib](https://matplotlib.org/) - A comprehensive library for creating static, animated and interactive visualizations in Python.
* [NLTK](https://www.nltk.org/) - A leading platform for building Python programs to work with human language data.
* [NumPy](https://numpy.org/) - The fundamental package for scientific computing with Python, which is used to generate multi-dimensional arrays that
can be manipulated.
* [pandas](https://pandas.pydata.org/) - An open-source library for data manipulation and analysis, particularly useful when dealing with large data 
sets. 
* [Requests](https://requests.readthedocs.io/en/master/) - A simple library for making HTTP requests in Python. 
* [scikit-learn](https://scikit-learn.org/stable/) - A machine learning library that features various classification algorithms including SVM
`sklearn.svm.SVC` and Naïve Bayes `sklearn.naive_bayes.ComplementNB`. It also provides utilities for further feature processing 
and detailed performance analysis of the classification results. 
* [seaborn](https://seaborn.pydata.org/) - A data visualization library built on top of Matplotlib that provides a high-level interface for generating
statistical graphics.
* [wordcloud](https://pypi.org/project/wordcloud/) - A library to generate word cloud in Python. 

## Author

**Suk Jing Lim** - Please [email](mailto:limsukjing@gmail.com) me if you have any questions.

## Acknowledgement

4th year Text Analysis project supervised by **Dr. Aurelia Power**.