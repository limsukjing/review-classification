import nltk
import pandas as pd
import re
from sklearn.svm import SVC
from sklearn.naive_bayes import ComplementNB

import scraper
import utility

import warnings

warnings.filterwarnings('ignore')

# TODO: uncomment this line of code to run the scraper
# scraper.run()

# read and generate summary statistics of the data
data = pd.read_csv('data.csv')
data_stats = data.describe()

# retrieve the documents for each class using Boolean masks
# store the documents in a list
pos_docs = list(data.document[data.label == 'movie_review'])  # pos
neg_docs = list(data.document[data.label == 'not_movie_review'])  # neg

# merge each list into a single text
pos_texts = '. '.join(pos_docs)
neg_texts = '. '.join(neg_docs)

# (2c) generate a WordCloud by calling the generate_clouds(docs, labels) method from utility
# determine which terms/words might be predictive of each topic (font size)
# TODO: uncomment this line of code to generate the WordCloud
# utility.generate_clouds([pos_texts, neg_texts], ['movie_review', 'not_movie_review'])

# generate three types of baseline matrices by calling the generate_matrices(matrices, docs, labels) method from utility
# establish baseline matrices before any cleaning/normalising techniques are applied
docs = list(data.document)
labels = list(data.label)


def baseline():
    # (3a) generate three types of vectors
    baseline_matrix_type = [utility.generate_count_matrix, utility.generate_tf_matrix, utility.generate_tfidf_matrix]
    baseline_matrix_list = utility.generate_matrices(baseline_matrix_type, docs, labels)  # [200 rows x 12058 columns]
    # for matrix in baseline_matrix_list:
    #     print(matrix)

    # split each baseline matrix into features (X) and label (y) column
    baseline_X_list, baseline_y_list = utility.split_feature_label(baseline_matrix_list)
    for baseline_X, baseline_y in zip(baseline_X_list, baseline_y_list):
        print(baseline_X.shape, baseline_y.shape)  # [(200, 11942) (200,)]

    # split each X/y baseline matrix into training and test sets
    baseline_sets = utility.split_train_test(baseline_X_list, baseline_y_list)

    # model selection, fitting, and evaluation
    matrices = ['Count', 'TF', 'TF-IDF']

    # split Validation
    print('\nBaseline Support-Vector Machine (SVM) Model - Split:')
    svm_model = SVC(kernel='linear', C=1.0, random_state=1, gamma='scale')
    baseline_svm_scores = utility.split_validate(svm_model, matrices, baseline_sets, pos_label='movie_review', print_=True)

    print('\nBaseline Naive Bayes (NB) Model - Split:')
    nb_model = ComplementNB(alpha=0.0, fit_prior=True, class_prior=None, norm=False)
    baseline_nb_scores = utility.split_validate(nb_model, matrices, baseline_sets, pos_label='movie_review', print_=True)

    # TODO: uncomment this line of code to generate the bar charts
    # utility.generate_bar_matrix('baseline SVM', baseline_svm_scores, matrices)
    # utility.generate_bar_matrix('baseline NB', baseline_nb_scores, matrices)

    # generate sorted attributes to identify suitable pre-processing techniques/tasks before vectorisation
    sorted_attributes = sorted(baseline_matrix_list[0].columns)

    count = 0
    for attribute in sorted_attributes:
        if re.match(r'\d+.*', attribute):
            count += 1
        else:
            pass
    print('\nDigit count: %d' % count)


# (2d) perform 3 cleaning/normalising tasks prior to vectorisation
def before_vectorisation():
    all_docs = docs.copy()
    all_labels = labels.copy()

    # (1) remove digits and punctuations by calling the remove_digit_punc(docs) method from utility
    all_docs = utility.remove_digit_punc(all_docs)

    # (2) tokenize the corpus
    tokenized_docs = [nltk.word_tokenize(doc) for doc in all_docs]

    # (3) remove stop words by calling the remove_stop_words(tokenized_docs) method from utility
    tokenized_docs = utility.remove_stop_words(tokenized_docs)

    # (4) assign POS tags to each token in the corpus
    tagged_docs = [nltk.pos_tag(doc) for doc in tokenized_docs]

    # (5) remove/replace tokens tagged with certain POS by calling the remove_pos_tags(tagged_docs) method from utility
    tokenized_docs = utility.remove_pos_tags(tagged_docs)

    # (6) filter remaining corpus based on the token lengths
    tokenized_docs = [[token for token in doc if 4 <= len(token) <= 15 or (len(token) > 15 and '_' in token)] for doc in tokenized_docs]
    all_docs = [' '.join(doc) for doc in tokenized_docs]
    vectorisation(all_docs, all_labels)


def vectorisation(all_docs, all_labels):
    matrix_type = [utility.generate_count_matrix, utility.generate_tf_matrix, utility.generate_tfidf_matrix]
    matrix_list = utility.generate_matrices(matrix_type, all_docs, all_labels)  # [200 rows x 7604 columns]
    after_vectorisation(matrix_list)


# (3b) perform 3 cleaning/normalising tasks after vectorisation
def after_vectorisation(matrix_list):
    # (1) generate sorted attributes to identify suitable pre-processing techniques/tasks after vectorisation
    sorted_attributes = sorted(matrix_list[0].columns)

    # (1) discard invalid / misspelled features by calling the remove_invalid_columns(matrices) method from utility
    matrix_list = utility.remove_invalid_columns(matrix_list)

    # (2) remove highly correlated features by calling the remove_correlated_columns(matrices) method from utility
    matrix_list = utility.remove_correlated_columns(matrix_list)

    # split each matrix into features (X) and label (y) column
    Xs, ys = utility.split_feature_label(matrix_list)
    for X, y in zip(Xs, ys):
        print(X.shape, y.shape)

    # (3) select features by calling the select_feature_tree(X, y, print_, n_features) method from utility
    reduced_Xs = []
    for X, y in zip(Xs, ys):
        reduced_Xs.append(utility.select_feature_tree(X, y, print_=True, n_features=12))

    # split each X/y matrix into training and test sets
    sets = utility.split_train_test(reduced_Xs, ys)

    # (4a-e) model selection, fitting, and evaluation
    matrices = ['Count', 'TF', 'TF-IDF']
    res_labels = ['Accuracy', 'Precision', 'Recall']

    # cross validation
    print('\nReduced Support-Vector Machine (SVM) Model - Cross:')
    svm_model = SVC(kernel='linear', C=1.0, random_state=1, gamma='scale')
    cross_svm_scores = utility.cross_validate(svm_model, matrices, sets, pos_label='movie_review', folds=5, print_=True)

    print('\nReduced Naive Bayes (NB) Model - Cross:')
    nb_model = ComplementNB(alpha=0.0, fit_prior=True, class_prior=None, norm=False)
    cross_nb_scores = utility.cross_validate(nb_model, matrices, sets, pos_label='movie_review', folds=5, print_=True)

    # TODO: uncomment this line of code to generate the bar charts
    # utility.generate_bar_algorithm(['SVM', 'NB'], [cross_svm_scores, cross_nb_scores], matrices, res_labels)

    # perform hyperparameter tuning and run cross validation again to see whether model has been improved
    print('\nTuned Support-Vector Machine (SVM) Model - Cross:')
    svm_model_tuned = SVC(kernel='rbf', C=1.0, random_state=1, gamma='scale')  # kernel: non-linear split
    cross_svm_scores_tuned = utility.cross_validate(svm_model_tuned, matrices, sets, pos_label='movie_review', folds=5, print_=True)

    print('\nTuned Naive Bayes (NB) Model - Cross:')
    nb_model_tuned = ComplementNB(alpha=1.0, fit_prior=True)  # alpha: smoothing
    cross_nb_scores_tuned = utility.cross_validate(nb_model_tuned, matrices, sets, pos_label='movie_review', folds=5, print_=True)

    # tuned model performs better
    print('\nSupport-Vector Machine (SVM) Model - Split:')
    split_svm_scores = utility.split_validate(svm_model_tuned, matrices, sets, pos_label='movie_review', print_=True)

    # reduced model performs better
    print('\nNaive Bayes (NB) Model - Split:')
    split_nb_scores = utility.split_validate(nb_model, matrices, sets, pos_label='movie_review', print_=True)

    # TODO: uncomment this line of code to generate the bar charts
    # utility.generate_bar_matrix('Cross Validation SVM', cross_svm_scores, matrices)
    # utility.generate_bar_matrix('Split Validation SVM', split_svm_scores, matrices)
    # utility.generate_bar_matrix('Cross Validation NB', cross_nb_scores, matrices)
    # utility.generate_bar_matrix('Split Validation NB', split_nb_scores, matrices)
    utility.generate_bar_algorithm(['SVM', 'NB'], [split_svm_scores, split_nb_scores], matrices, res_labels)


if __name__ == "__main__":
    # baseline()
    before_vectorisation()
