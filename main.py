import re
import string

import matplotlib.pyplot as plt
import pandas as pd
import pymorphy2
import requests
import tqdm
from bs4 import BeautifulSoup
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud

STOPWORDS = stopwords.words('russian')
STOPWORDS.extend(['pic', 'jpg', 'зачет', 'незачет', 'это', 'чтецу'])
authors_file = './files/authors.csv'
questions_file = './files/questions.csv'

morph = pymorphy2.MorphAnalyzer()


def parse_authors(file):
    URLs = [
        'https://db.chgk.info/people?sort=desc&order=%D0%92%D0%BE%D0%BF%D1%80%D0%BE%D1%81%D0%BE%D0%B2',
        'https://db.chgk.info/people?page=1&sort=desc&order=%D0%92%D0%BE%D0%BF%D1%80%D0%BE%D1%81%D0%BE%D0%B2'
    ]
    authors = []
    for URL in URLs:
        page = requests.get(URL)
        soup = BeautifulSoup(page.content, 'html.parser')
        tbody = soup.find('tbody')
        trs = tbody.find_all('tr')
        for tr in trs:
            place = tr.find('td').text
            a = tr.find('a')
            href = a['href']
            name = a.text
            authors.append((place, href, name))
    authors = pd.DataFrame(authors, columns=['place', 'href', 'name'])
    authors.set_index('place', inplace=True)
    authors.to_csv(file)


def read_authors(file):
    return pd.read_csv(file)


def get_questions_by_author(author_name):
    url = f'https://db.chgk.info/xml/search/questions/author_{author_name}/types1/QAZCU/from_2000-01-01/limit1000'
    xml_file = requests.get(url).content
    f = open(f"./files/{author_name}.xml", "wb")
    f.write(xml_file)
    f.close()


def get_questions(authors: pd.DataFrame):
    for _, author in tqdm.tqdm(authors.iterrows()):
        nickname = author['href'].split('/')[2]
        get_questions_by_author(nickname)


def read_questions_by_author(xml_file) -> pd.DataFrame:
    questions = pd.read_xml(xml_file)
    questions.drop([0], inplace=True)
    questions.drop(columns=['total'], inplace=True)
    questions.insert(0, 'Author', re.split(r'\\|\.|/', xml_file)[-2])
    return questions


def combine_questions(xml_files, result_file):
    frames = []
    for xml_file in xml_files:
        questions = read_questions_by_author(xml_file)
        frames.append(questions)
    df = pd.concat(frames)
    df.to_csv(result_file)


def read_df(file) -> pd.DataFrame:
    df = pd.read_csv(file, index_col=[0])
    df['tourPlayedAt'] = pd.to_datetime(df['tourPlayedAt'])
    df = df[df['tourPlayedAt'] < '01-01-2019']
    df['WordCount'] = df['Question'].apply(lambda x: len(str(x).split()))
    df['PreprocessedQuestion'] = df['Question'].apply(
        lambda x: preprocess_text(x))
    df['PreprocessedAnswer'] = df['Answer'].apply(
        lambda x: preprocess_text(x))
    return df


def get_top_authors(df: pd.DataFrame, top=50):
    grouped = df.groupby('Author')
    qcount = grouped.size().sort_values(ascending=False)[:top]
    return qcount.index.to_series(index=list(range(len(qcount))))


def show_plots(df: pd.DataFrame):
    q_years = df.groupby(pd.Grouper(key='tourPlayedAt', freq='Y')).size()
    plt.figure()
    q_years.plot()
    plt.figure()
    q_auth = df.groupby('Author').size().sort_values(ascending=False)
    q_auth.plot(kind='bar')
    top50 = get_top_authors(df, 50)
    q_y_auth = df.groupby(
        ['Author',
         pd.Grouper(key='tourPlayedAt', freq='Y')]).size().unstack().fillna(
        0).cumsum(axis=1).T[top50[:10]]
    q_y_auth.plot.area()
    plt.figure()
    q_len = df.groupby('Author')['WordCount'].mean()[top50]
    q_len.plot(kind='bar')
    plt.show()


def get_corpus(data):
    corpus = []
    for phrase in data:
        corpus.extend(word_tokenize(phrase, language='russian'))
    return corpus


def str_corpus(corpus):
    str_corpus = ''
    for i in corpus:
        str_corpus += ' ' + i
    str_corpus = str_corpus.strip()
    return str_corpus


def get_wordCloud(corpus):
    wordCloud = WordCloud(background_color='white',
                          stopwords=STOPWORDS,
                          width=3000,
                          height=2500,
                          max_words=200,
                          random_state=42
                          ).generate(str_corpus(corpus))
    return wordCloud


def show_wordcloud(df: pd.DataFrame):
    corpus = get_corpus(df['PreprocessedQuestion'].values)
    wc_q = get_wordCloud(corpus)
    answers = get_corpus(df['PreprocessedAnswer'].values)
    wc_a = get_wordCloud(answers)
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(wc_q)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(wc_a)
    plt.axis('off')
    plt.show()


def preprocess_text(text):
    text = text.lower()
    text = text.strip()
    text = re.compile('<.*?>').sub('', text)
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)
    text = re.sub('\s+', ' ', text)
    text = re.sub(r'\[[0-9]*\]', ' ', text)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    text = re.sub(r'\d', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def normal_form(word):
    return morph.parse(word)[0].normal_form


def get_dataset(df: pd.DataFrame):
    for ind, question in df.iterrows():
        tokens = word_tokenize(question['PreprocessedQuestion'],
                               language='russian')
        stemmed = list(map(normal_form, tokens))
        print(stemmed)
        return


# parse_authors(authors_file)
# authors_df = read_authors(authors_file)
# get_questions(authors_df)
# files = glob.glob('./files/*.xml')
# combine_questions(files, questions_file)
df = read_df(questions_file)
# show_plots(df)
# show_wordcloud(df)
get_dataset(df)
