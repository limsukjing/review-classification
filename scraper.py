import re
import utility


def run():
    # (1) scrape websites by calling the scrape_data(url) method from utility
    # retrieve a total of 200 soups, i.e. one positive (100) and two negative classes (50 each)
    movie_url = 'https://www.rottentomatoes.com/top/bestofrt/'  # Rotten Tomatoes (movie review - positive class)
    documentary_url = 'https://www.rottentomatoes.com/top/bestofrt/top_100_documentary_movies/'  # Rotten Tomatoes (documentary review -  related to positive class)
    restaurant_url = 'https://www.theworlds50best.com/list/1-50'  # The World's 50 Best Restaurants (restaurant review - unrelated to positive class)

    movie_soup = utility.scrape_data(movie_url)
    documentary_soup = utility.scrape_data(documentary_url)
    restaurant_soup = utility.scrape_data(restaurant_url)

    # 100 positive soups
    movie_links = []
    for link in movie_soup.body.find_all('a', href=re.compile(r'^/m/.+'), class_='unstyled articleLink'):
        movie_links.append('https://www.rottentomatoes.com' + link.get('href'))  # '/m/(title)'
    movie_size = len(movie_links)

    # 50 negative soups
    documentary_links = []
    for link in documentary_soup.body.find_all('a', href=re.compile(r'^/m/.+'), class_='unstyled articleLink')[50:101]:
        documentary_links.append('https://www.rottentomatoes.com' + link.get('href'))  # '/m/(title)'
    documentary_size = len(documentary_links)

    # 50 negative soups
    restaurant_links = []
    for link in restaurant_soup.find_all('a', class_='item')[:50]:
        restaurant_links.append('https://www.theworlds50best.com' + link.get('href'))  # '/the-list/(id)/(name)'
    restaurant_size = len(restaurant_links)

    pos_soup = [utility.scrape_data(url) for url in movie_links]
    pos_size = len(pos_soup)

    neg_soup = [utility.scrape_data(url) for url in (documentary_links + restaurant_links)]
    neg_size = len(neg_soup)

    # (2a) remove html and character encoding before writing the data to a csv file
    # retrieve all descendants of the body tag from each soup and put irrelevant tags into a list
    pos_tags = []
    for soup in pos_soup:
        pos_tags.append(set([child.name for child in soup.body.descendants if child.name is not None]))

    neg_tags = []
    for soup in neg_soup:
        neg_tags.append(set([child.name for child in soup.body.descendants if child.name is not None]))

    # inspect pos_tags/neg_tags for each soup and decide which tags to remove
    for soup in (pos_soup + neg_soup):
        for div in soup.find_all('div', id=['movies_sidebar']):
            div.decompose()
        for div in soup.find_all('div', class_=['modal-dialog', 'rate-and-review-widget__module']):
            div.decompose()
        for section in soup.find_all('section', class_=['mop-ratings-wrap__row js-scoreboard-container', 'panel panel-rt panel-box']):
            section.decompose()

    pos_tags_to_remove = ['span', 'button', 'script', 'noscript', 'style', 'input', 'table', 'img', 'form', 'nav', 'header', 'footer']
    neg_tags_to_remove = ['span', 'button', 'script', 'noscript', 'style', 'input', 'table', 'img', 'form', 'nav', 'header', 'footer', 'iframe', 'hr', 'br']

    # remove a list of irrelevant tags from each soup by calling the remove_tags(tags, soup) method from utility
    pos_docs = [utility.remove_tags(pos_tags_to_remove, soup) for soup in pos_soup]
    neg_docs = [utility.remove_tags(neg_tags_to_remove, soup) for soup in neg_soup]

    # join the lists (a total of 200 soups)
    docs = pos_docs + neg_docs
    doc_size = len(docs)

    # (2b) create a DataFrame with pos/neg labels by calling the build_corpus(docs, labels) method from utility
    labels = ['movie_review'] * len(pos_docs) + ['not_movie_review'] * len(neg_docs)
    df = utility.build_corpus(docs, labels)

    # remove tabs, carriage returns, new lines and specific Unicode special characters from the DataFrame
    # by calling the clean_doc(df) method from utility
    df['document'] = df['document'].apply(utility.clean_doc)
    # print(df)

    # export the DataFrame to a CSV file
    df.to_csv('data.csv', encoding='utf-8', index=False)
