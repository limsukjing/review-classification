import bs4
from itertools import groupby
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import re
import requests
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score as acc, precision_score as pr, recall_score as rec
from sklearn.model_selection import train_test_split, cross_val_score as xval
from sklearn.tree import DecisionTreeClassifier
import wordcloud


# function that retrieves data from the specified URL and use BeautifulSoup to parse the HTML content
# returns a soup object
def scrape_data(url):
    page = requests.get(url)  # GET request
    data = page.text
    return bs4.BeautifulSoup(data, 'html.parser')


# function that takes a list of irrelevant tags and removes them from the html tree of a soup
# returns the text of the remaining tags
def remove_tags(tags, soup):
    [s.extract() for s in soup(tags)]
    return soup.text


# function that creates and returns a labelled DataFrame
def build_corpus(docs, labels):
    df = np.array(docs)
    df = pd.DataFrame({
        'document': df,
        'label': labels
    })
    df = df.sample(len(df))  # randomise the order of rows
    df = df.reset_index(drop=True)  # drop parameter to avoid the old index being added as a column
    return df


# function that removes tabs, carriage returns, new lines and specific Unicode special characters (typically control
# characters) and returns the DataFrame
def clean_doc(df):
    df = re.sub(r'[\r\n\t\xa0\x88\x89]', ' ', df)
    df = re.sub(r'\s{2,}', ' ', df)  # replace 2 or more whitespaces with a single space
    return df


# function that takes a list of documents, labels and plots a WordCloud
def generate_clouds(docs, labels):
    plt.figure(figsize=(10, 5)).canvas.set_window_title('WordCloud')
    for i in range(len(docs)):
        cloud = wordcloud.WordCloud(width=800, height=800, background_color='black', min_font_size=10).generate(docs[i])
        plt.subplot(1, 2, i + 1)
        plt.title(labels[i])
        plt.imshow(cloud)
        plt.axis('off')
    plt.show()


# function that generates a matrix of token counts using CountVectorizer
# returns a DataFrame where all NaN values are replaced with 0
def generate_count_matrix(docs):
    vectorizer = CountVectorizer(decode_error='replace', strip_accents='unicode', lowercase=True)
    X = vectorizer.fit_transform(docs)  # transform the data and return the term-document matrix
    features = list(vectorizer.get_feature_names())
    count_matrix = pd.DataFrame(X.toarray(), columns=features)
    return count_matrix.fillna(0)


# function that takes in a term-document matrix and computes the total number of tokens/terms in each document
def compute_doc_length(matrix):
    return np.sum(matrix, axis=1)  # horizontally across columns


# function that generates a matrix of normalised frequencies by applying tf weighting to each cell
# formula: (number of time the term occurs in doc A) / (total number of terms in doc A)
def generate_tf_matrix(docs):
    count_matrix = generate_count_matrix(docs)
    doc_length = compute_doc_length(count_matrix)
    return count_matrix.divide(doc_length, axis=0)  # vertically across rows


# function that computes the idf of each attribute
# formula: log (total number of documents in the corpus / number of docs where the term appears)
def compute_idf(matrix):
    doc_frequency = matrix[matrix > 0].count(axis=0)  # vertically across rows
    return np.log2(len(matrix) / doc_frequency)


# function that generates a term-document matrix by applying tf-idf weighting to each cell
def generate_tfidf_matrix(docs):
    vectorizer = TfidfVectorizer(decode_error='replace', strip_accents='unicode', lowercase=True)
    X = vectorizer.fit_transform(docs)  # transform the data and return the term-document matrix
    features = list(vectorizer.get_feature_names())
    tfidf_matrix = pd.DataFrame(X.toarray(), columns=features)
    return tfidf_matrix.fillna(0)


# function that generates a list of matrices (types: frequency counts, TF, TF-IDF)
def generate_matrices(matrices, docs, labels):
    matrix_list = []
    for matrix in matrices:
        matrix = matrix(docs)
        matrix['label'] = labels
        matrix_list.append(matrix)
    return matrix_list


# function that splits each matrix into two columns: features (input - X) and label (output - y)
# returns a list of three DataFrames
def split_feature_label(matrices):
    # feature_columns = list(matrices[0].columns)[0: len(matrices[0].columns) - 1]  # exclude the label column
    X_list, y_list = [], []
    for matrix in matrices:
        feature_columns = list(matrix.columns)[0: len(matrix.columns) - 1]  # exclude the label column
        X_list.append(matrix[feature_columns])
        y_list.append(matrix['label'])
    return X_list, y_list


# function that splits each X/y matrix into training and test sets
def split_train_test(X_list, y_list):
    sets = []
    for X, y in zip(X_list, y_list):
        # test_size of 0.2 randomly selects 80% of the instances for training (160) and the remaining 20% for testing (40)
        # splits data into 4 sets for each matrix: X_train [0], X_test [1], y_train [2], y_test [3]
        sets.append(train_test_split(X, y, test_size=0.2, random_state=1))
    return sets


# function that uses split validation to fit a model for a given classification algorithm
# evaluates the prediction performance (actual vs. predicted y_test label) of 3 types of matrices for each classification model
def split_validate(model, matrices, sets, pos_label, print_=False):
    scores = []
    for i in range(len(matrices)):
        model.fit(sets[i][0], sets[i][2])  # fit the model according to the given training sets (X_train, y_train)
        y_pred = model.predict(sets[i][1])  # perform classification on the given test set (X_test)
        accuracy = acc(y_true=sets[i][3], y_pred=y_pred)
        precision = pr(y_true=sets[i][3], y_pred=y_pred, pos_label=pos_label)
        recall = rec(y_true=sets[i][3], y_pred=y_pred, pos_label=pos_label)
        scores.append((accuracy, precision, recall))
        if print_:
            print('-----------', matrices[i], '-------------')
            print('Split Validation Accuracy: %0.2f' % accuracy)
            print('Split Validation Precision: %0.2f' % precision)
            print('Split Validation Recall: %0.2f' % recall)
    return scores


# function that uses cross validation to fit a model for a given classification algorithm
# evaluates the prediction performance (actual vs. predicted y_test label) of 3 types of matrices for each classification model
def cross_validate(model, matrices, sets, pos_label, folds=5, print_=False):
    scores = []
    for i in range(len(matrices)):
        model.fit(sets[i][0], sets[i][2])  # fit the model according to the given training sets (X_train, y_train)
        # perform classification on the given test set (5 iterations)
        accuracy = xval(model, sets[i][0], sets[i][2], cv=folds).mean()
        precision = xval(model, sets[i][0], sets[i][2], cv=folds, scoring='precision_macro').mean()
        recall = xval(model, sets[i][0], sets[i][2], cv=folds, scoring='recall_macro').mean()
        scores.append((accuracy, precision, recall))
        if print_:
            print('-----------', matrices[i], '-------------')
            print('Cross Validation Accuracy: %0.2f' % accuracy)
            print('Cross Validation Precision: %0.2f' % precision)
            print('Cross Validation Recall: %0.2f' % recall)
    return scores


# function that creates bar charts for each matrix to visualize the classification results in terms of accuracy, precision and recall
def generate_bar_matrix(title, scores, matrices):
    fig, (accuracy, precision, recall) = plt.subplots(1, 3, sharey='row', figsize=(12, 5))
    fig.canvas.set_window_title('Bar Chart')
    accuracy.set_title('Accuracy')
    accuracy.bar(matrices, [score[0] for score in scores], color='green')
    precision.set_title('Precision')
    precision.bar(matrices, [score[1] for score in scores], color='blue')
    recall.set_title('Recall')
    recall.bar(matrices, [score[2] for score in scores], color='orange')
    fig.suptitle('Performance of %s classifier' % title)
    plt.show()


# function that creates bar charts for each algorithm to visualize the classification results in terms of accuracy, precision and recall
def generate_bar_algorithm(titles, scores, matrices, labels):
    width = 0.35  # width of each bar
    plt.figure(figsize=(10, 9)).canvas.set_window_title('Bar Charts by Algorithms')
    for i in range(len(matrices)):
        plt.subplot(str('31' + str(i + 1)))
        x = np.arange(len(labels))
        ax = plt.gca()
        ax.bar(x - width / 2, scores[0][i], width, label=titles[0])
        ax.bar(x + width / 2, scores[1][i], width, label=titles[1])
        ax.set_title(matrices[i])
        ax.set_ylabel('Scores')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
    plt.show()


# function that replaces numbers representing valid years with 'YEAR' and discards all remaining numbers
# also removes punctuations and replaces multiple spaces with a single space
def remove_digit_punc(docs):
    docs = [re.sub(r'\(*(19|20)\d{2}\)*', 'year', doc) for doc in docs]
    docs = [re.sub(r'\d+(.|\w*)', '', doc) for doc in docs]
    docs = [re.sub(r'[!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~€]+', '', doc) for doc in docs]
    docs = [re.sub(r'\s{2,}', ' ', doc) for doc in docs]
    return docs


# function that removes stop words (nltk's and custom) from the corpus
def remove_stop_words(tokenized_docs):
    stop_words = stopwords.words('english')  # NLTK
    custom_stop_words = ['rotten', 'tomatoes', 'harry', 'potter', 'ranked', 'worst', 'best', 'tomatometer', 'view',
                         'guides', 'binge', 'central', 'springsummer', 'tv', 'guide', 'rt', 'news', 'indie', 'fresh',
                         'list', 'women', 'fateful', 'journeys', 'aspiring', 'sommelier', 'holy', 'schnikes',
                         'tommy', 'boy', 'still', 'hilarious', 'years', 'later', 'tickets', 'showtimes', 'trending',
                         'netflix', 'series', 'free', 'add', 'article', 'super', 'reviewer', 'ratings', 'reviews',
                         'explanation', 'watch', 'rent', 'buy', 'online', 'disney', 'see', 'discstreaming']  # custom
    docs = [[token for token in doc if token.lower() not in (stop_words + custom_stop_words)] for doc in tokenized_docs]
    return docs


# function that filters tokens tagged as proper noun (NNP) and concatenates two consecutive NNP tokens using an
# underscore, also removes a list of other POS tags from the corpus
# POS: Coordinating Conjunction, Cardinal Number, Determiner, Preposition, Modal / Auxiliary Verb, Possessive Ending,
# Personal and Possessive, Pronoun, Particle, To, Wh-determiner, Wh-pronoun, Possessive Wh-pronoun, Wh-adverb
def remove_pos_tags(tagged_docs):
    new_tagged_docs = []
    for doc in tagged_docs:
        groups = groupby(doc, key=lambda x: x[1])  # group by tags
        entities = [[t for t, _ in tokens if t != '’'] for tag, tokens in groups if tag == 'NNP']  # named entities
        entities = [('_'.join(entity), 'NNP') for entity in entities if len(entity) == 2]
        lists = [(token, tag) for (token, tag) in doc if tag != 'NNP'] + entities
        new_tagged_docs.append(lists)
    remove_list = ['CC', 'CD', 'DT', 'IN', 'MD', 'POS', 'PRP', 'PRP$', 'RP', 'TO', 'VB', 'VBD', 'VBG', 'VBN', 'VBP',
                   'VBZ', 'WDT', 'WP', 'WP$', 'WRB']
    new_tagged_docs = [[token for (token, tag) in doc if tag not in remove_list] for doc in new_tagged_docs]
    return new_tagged_docs


# function that discards columns that contain newly concatenated tokens that are invalid, or misspelled words from the corpus
# noinspection SpellCheckingInspection
def remove_invalid_columns(matrices):
    invalid_features = ['_focusing', 'a_la', 'academyaward_r', 'africanamerican', 'americanitalian', 'arpege_', 'cove_oscar', 'c_zeitgeist',
                        'emotiona', 'heaven_im', 'hypersustainable_septime', 'japaneseindian', 'lebaneseportuguese_za', 'menton_italianfrench',
                        'mexican_japanesemediterranean', 'morocco_im', 'movies_movies', 'otooles', 'oh_diana', 'oh_im', 'olivia_wide',
                        'p_spiderman', 'part_journal', 'rating_pgfor', 'regardless_id', 'samantha_boyhood', 'studio_aabout', 'vietnam_throughout',
                        'wwi_diana', 'wwi_hey', 'wwi_im', 'yeah_im', 'yes_holland', 'yet_pinocchio', 'aboveearth', 'acrosstheboard', 'againwhich',
                        'allnew', 'answersand', 'armsbonus', 'asthetic', 'bodyshazam', 'brotherstoby', 'btoob', 'caviarwatch', 'centerfont',
                        'centurys', 'challengemade', 'charachters', 'childrens', 'citys', 'civilizationat', 'coalitionsact', 'communitys',
                        'companys', 'crisisthe', 'dissapointong', 'emotionnal', 'endyet', 'escalatedand', 'everythingi', 'facessome', 'familys',
                        'filmatms', 'forebearsa', 'fourtime', 'franchiseand', 'friendsmedgar', 'fthe', 'fullon', 'genrey', 'goingson',
                        'halfcomanche', 'handson', 'hattrickalbeit', 'hattyin', 'hcentury', 'hehi', 'hfloor', 'historyc', 'husbandsatm',
                        'ighostbustersi', 'ilaurai', 'imuhammad', 'istar', 'ithe', 'kamebish', 'mens', 'migrationan', 'neighborsharon',
                        'noreservations', 'nostalgiatinged', 'oftforgotten', 'particualrly', 'peformance', 'personailties', 'philosphy',
                        'plushjoin', 'procedral', 'pwace', 'pyschiatrists', 'schoolthe', 'sciencey', 'selo', 'shhhhh', 'shitzillion',
                        'showstealingly', 'silouhette', 'sizefacecentury', 'slavesugh', 'sohurray', 'standardstobe', 'standardstobe', 'suitand',
                        'suitwhy', 'teame', 'teenagerboth', 'threemichelin', 'tidbits', 'tmeteri', 'togethera', 'tosend', 'transistion',
                        'turbulant', 'unfortuante', 'villians', 'warfareno', 'warhow', 'wayas', 'wery', 'weve', 'wifes', 'wordshazamthis',
                        'worldwith', 'yearor', 'yearyear', 'yearyearish', 'yuba', 'consensus_capturing', 'consensus_favourite',
                        'consensus_moonlight', 'consensus_spotlight', 'contact_building', 'contact_franko', 'contact_heumarkt', 'contact_largo',
                        'contact_plaza', 'diy_blt', 'doc_ock', 'el_bulliinfluenced']
    matrices = [matrix.drop(invalid_features, axis=1) for matrix in matrices]
    return matrices


# function that discards columns that are highly correlated to movie, documentary and restaurant
# noinspection SpellCheckingInspection
def remove_correlated_columns(matrices):
    # TODO: uncomment these lines of code to generate the heatmap
    # demo_matrix = matrices[0].iloc[:, :40]
    # plt.figure(figsize=(10, 8)).canvas.set_window_title('Correlation Matrix')
    # plt.title('Correlation Matrix of First 40 Features')
    # demo_corr = demo_matrix.corr()
    # sns.heatmap(demo_corr, cmap="YlGnBu", linewidths=0.1)
    # plt.show()

    corr = matrices[0].corr().abs()  # correlation matrix
    corr_m = corr['movie']
    corr_d = corr['documentary']
    corr_r = corr['restaurant']
    correlated_features = corr_m[corr_m > 0.7] + corr_d[corr_d > 0.7] + corr_r[corr_r > 0.7]
    correlated_features = ['actually', 'arent', 'aunt', 'blame', 'bunch', 'clearly', 'comicbook', 'course', 'destroy', 'didnt', 'dont',
                           'essentially', 'fact', 'firstly', 'havent', 'honestly', 'idea', 'incredibly', 'movies', 'naturally', 'needs', 'opinion',
                           'people', 'positive', 'ready', 'really', 'reason', 'secondly', 'shit', 'solid', 'something', 'sport', 'spite', 'thats',
                           'theres', 'thing', 'things', 'thumbs']
    matrices = [matrix.drop(correlated_features, axis=1) for matrix in matrices]
    return matrices


# function that uses DecisionTreeClassifier to reduce the number of features in each matrix
# also prints the best n features (specified by users)
def select_feature_tree(X, y, print_=False, n_features=12):
    dtc = DecisionTreeClassifier(random_state=1)
    dtc = dtc.fit(X, y)
    model = SelectFromModel(dtc, prefit=True)
    X_reduced = model.transform(X)
    if print_:
        print("Rows X Columns:", X_reduced.shape)
        scores = pd.DataFrame(dtc.feature_importances_)
        columns = pd.DataFrame(X.columns)
        feature_scores = pd.concat([columns, scores], axis=1)
        feature_scores.columns = ['Term', 'Score']
        print(feature_scores.nlargest(n_features, 'Score'))  # n best features
    return X_reduced
