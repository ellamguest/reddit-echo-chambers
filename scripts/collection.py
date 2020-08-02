from google.cloud import storage
from google.cloud import bigquery
from .tools import (run_job, extract_table, get_blob, 
                    delete_table, bigquery_client, storage_client, 
                    insert_to_name, store_blob, cache_path,
                    dataset_id, parse_csv_blob, tokens_to_csc, tokens_to_csr)
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from google.cloud.exceptions import NotFound 
import pandas as pd

""" AUTHOR DATA """
def get_all_subreddit_author_counts(date):
    """
    Aggregates subreddit, author, num_comments for date
    Stores in table date:all_subreddit_author_counts
    """

    print("Getting all subreddit author counts")

    query = f"""SELECT subreddit, author, count(*) as num_comments
                FROM
                `fh-bigquery.reddit_comments.{date}`
                GROUP BY subreddit, author
                """

    run_job(query, dataset_id=date, table_id='all_subreddit_author_counts')

def get_subreddit_total_counts(date):
    print("Getting subreddit total (botless) counts")
    query = f"""SELECT subreddit, count(*) as num_authors, sum(num_comments) as num_comments
                FROM `{date}.all_subreddit_author_counts`
                WHERE (ends_with(lower(author), 'bot') is False)
                AND (author not in ('AutoModerator', '[deleted]', 'autotldr', 'Mentioned_Videos',
                                    'TweetPoster', 'xkcd_transcriber', 'imgurtranscriber',
                                    'The-Paranoid-Android')
                    )
                GROUP BY subreddit
                """
    run_job(query, dataset_id=date, table_id='subreddit_total_counts')


def get_top_subreddits(date, num_subreddits=1000):
    print(f"Getting top {num_subreddits} subreddits in {date} by author count")
    query = f"""SELECT subreddit, num_authors, num_comments FROM `{date}.subreddit_total_counts`
    ORDER BY num_authors DESC
    LIMIT {num_subreddits}"""
    run_job(query, dataset_id=date, table_id=f"top_{num_subreddits}_subreddits")


def get_author_subset(date, quantile=0.75, num_subreddits=1000):
    print(f"(Re-)Removing bots and subseting top {quantile} of authors per subreddit by num_comments")
    query = f"""with BOTLESS as
                (select author, subreddit, num_comments
                from `{date}.all_subreddit_author_counts`
                WHERE (ends_with(lower(author), 'bot') is False)
                    AND (author not in ('AutoModerator', '[deleted]', 'autotldr', 'Mentioned_Videos',
                                        'TweetPoster', 'xkcd_transcriber', 'imgurtranscriber',
                                        'The-Paranoid-Android')
                        )
                    AND (subreddit in (SELECT subreddit
                        FROM `{date}.top_{num_subreddits}_subreddits`))
                    ),

                UQR as
                (SELECT author, subreddit, num_comments, percentile_cont(num_comments, {quantile}) OVER (PARTITION BY subreddit) AS subreddit_uqr
                FROM BOTLESS)
                select author, subreddit, num_comments, subreddit_uqr
                FROM UQR
                where num_comments > subreddit_uqr"""
    run_job(query, dataset_id=date, table_id=f"top_{num_subreddits}_subreddits_top_authors")


def store_top_subreddit_author_counts(date, num_subreddits=1000):
    print(f"Getting subreddit author counts from top {num_subreddits}")
    query = f"""SELECT subreddit, author, num_comments
    FROM `{date}.top_{num_subreddits}_subreddits_top_authors`"""

    table_id = f"top_{num_subreddits}_subreddit_author_subset"
    run_job(query, dataset_id=date, table_id=table_id)

    blob_name = table_id + '.csv'

    extract_table(dataset_id=date, table_id=table_id,
                    bucket_name=date, blob_name=blob_name)

def store_author_adj(date, num_subreddits=1000):
    print(f"Getting author adjacency matrix for {date}")
    blob = get_blob(bucket_name=date,
                    blob_name=f"top_{num_subreddits}_subreddit_author_subset.csv")

    tokens = parse_csv_blob(blob)
    csr, row_lookup, col_lookup = tokens_to_csr(tokens, indicator=True, return_lookup=True)

    adj_sparse = csr.dot(csr.T)
    adj = pd.DataFrame(adj_sparse.toarray())
    adj.index = adj.index.map(lambda x: row_lookup[x])
    adj.columns = adj.columns.map(lambda x: row_lookup[x])

    adj_blob_name = insert_to_name(blob.name, '_adj')
    store_blob(adj, bucket_name=date, blob_name=adj_blob_name)

def run_author_data(date, num_subreddits=1000):
    get_all_subreddit_author_counts(date=date)

    get_subreddit_total_counts(date=date)

    get_top_subreddits(date, num_subreddits)

    get_author_subset(date)

    store_top_subreddit_author_counts(date=date, num_subreddits=num_subreddits)

    store_author_adj(date, num_subreddits)


""" TEXT DATA """
def get_subreddit_word_counts(date, num_subreddits, update=False):
    dataset_id = date
    table_id = f"top_{num_subreddits}_subreddit_word_counts"

    if update:
        try:
            delete_table(dataset_id, table_id)
        except NotFound:
            pass

    print(f"Getting subreddit-level word counts table for {date} for top {num_subreddits} subreddits")
    query = f"""WITH
    nested_words AS (
    SELECT
        subreddit, author,
        REGEXP_EXTRACT_ALL( REGEXP_REPLACE(REGEXP_REPLACE(LOWER(body), '&amp;', '&'), r'&[a-z]{{2,4}};', '*'), r'[a-z]{{2,20}}\\'?[a-z]+') AS arr
    FROM
        `fh-bigquery.reddit_comments.{date}`
    WHERE
        body NOT IN ('[deleted]', '[removed]')
        AND subreddit IN (SELECT subreddit FROM `{date}.top_{num_subreddits}_subreddits`)
        AND author IN (SELECT author FROM `{date}.top_{num_subreddits}_subreddits_top_authors`)
        )
    SELECT
    subreddit,
    REGEXP_REPLACE(word, r"([\'])", '') AS word,
    COUNT(*) AS count
    FROM
    nested_words,
    UNNEST(arr) AS word
    GROUP BY
    subreddit,
    word"""

    run_job(query, dataset_id=dataset_id, table_id=table_id)

def get_word_counts(date, num_subreddits, update=False):
    dataset_id = date
    table_id = f"top_{num_subreddits}_word_counts"

    if update:
        try:
            delete_table(dataset_id, table_id)
        except NotFound:
            pass

    print(f"Getting word counts table for {date} for top {num_subreddits} subreddits and top quartile of words")
    query = f"""WITH
    word_counts AS (
    SELECT
        word,
        COUNT(*) AS subreddit_count,
        SUM(count) AS total_count
    FROM
        `{date}.top_{num_subreddits}_subreddit_word_counts`
    GROUP BY
        word),

    repeats AS (SELECT * FROM word_counts WHERE (word in (SELECT word FROM word_counts WHERE (subreddit_count > 1)))),

    uqr as
        (SELECT q
        FROM (SELECT approx_quantiles(total_count, 4) as quantiles FROM repeats), UNNEST(quantiles) as q
        WITH OFFSET index
        WHERE index = 3),

    top_words as
    (SELECT * FROM repeats
    WHERE total_count >= (SELECT q from uqr)),

    word_ratios AS (
    SELECT
        word,
        subreddit_count,
        total_count,
        {num_subreddits}/subreddit_count AS subreddit_ratio
    FROM
        top_words
        )
    SELECT
    *,
    LOG(subreddit_ratio) AS idf
    FROM
    word_ratios
    ORDER BY total_count"""

    run_job(query, dataset_id=dataset_id, table_id=table_id)

def get_subreddit_total_word_counts(date, num_subreddits, update=False):
    dataset_id = date
    table_id = f"top_{num_subreddits}_subreddit_total_word_counts"

    if update:
        try:
            delete_table(dataset_id, table_id)
        except NotFound:
            pass

    print(f"Getting total word counts in {date} for top {num_subreddits} subreddits")

    query = f"""
            SELECT subreddit, count(word) as unique_words, sum(count) as total_words
            FROM `{date}.top_{num_subreddits}_subreddit_word_counts`
            GROUP BY subreddit
            """

    run_job(query, dataset_id=dataset_id, table_id=table_id)

    bucket_name = date
    if storage_client().get_bucket(bucket_name).exists() is False:
        storage_client().create_bucket(bucket_name)

    extract_table(dataset_id=date,
                  table_id=table_id,
                  bucket_name=bucket_name,
                  blob_name=table_id + '.csv')


def get_subreddit_word_tfidfs(date, num_subreddits, update=False):
    dataset_id = date
    table_id = f"top_{num_subreddits}_subreddit_word_tfidfs"

    if update:
        try:
            delete_table(dataset_id, table_id)
        except NotFound:
            pass

    print(f"Getting subreddit word tfidf data table for {date} for top {num_subreddits} subreddits")

    query = f"""with all_counts as (SELECT * FROM `{date}.top_{num_subreddits}_word_counts` AS word_counts LEFT JOIN `{date}.top_{num_subreddits}_subreddit_word_counts` AS subreddit_counts
    USING(word)),

    calc_tf as (SELECT *, count/total_count as tf 
    FROM all_counts)

    SELECT subreddit, word, count, subreddit_count, total_count, tf, idf, tf*idf as tfidf
    FROM calc_tf"""

    run_job(query, dataset_id=dataset_id, table_id=table_id)


def store_tfidf(date, num_subreddits, lower=0.2, upper=0.8, update=False):
    """
    creating table with only columns subreddit, word, tfidf to extract to
    storage buckets only select words that occur in between lower and upper
    bounds of subreddit count
    """

    print(f"""Getting subreddit word tfidf table for {date}
                for top {num_subreddits} subreddits,
                subsetting tfidf values between {lower} and {upper}
                and exporting to storage bucket""")

    dataset_id = date
    table_id = f"top_{num_subreddits}_subreddit_word_tfidf_subset"

    if update:
        try:
            delete_table(dataset_id, table_id)
        except NotFound:
            pass

    query = f"""
            SELECT subreddit, word, tfidf
            FROM`{date}.top_{num_subreddits}_subreddit_word_tfidfs`
            WHERE subreddit_count between {int(num_subreddits*lower)}
                                        and {int(num_subreddits*upper)}
            """

    run_job(query, dataset_id=dataset_id, table_id=table_id)

    bucket_name = date
    if storage_client().get_bucket(bucket_name).exists() is False:
        storage_client().create_bucket(bucket_name)

    extract_table(dataset_id=date,
                  table_id=table_id,
                  bucket_name=bucket_name,
                  blob_name=table_id + '.csv')


def get_word_stems(date, num_subreddits):
    import nltk
    from nltk.stem import PorterStemmer
    from nltk.stem import LancasterStemmer
    from nltk.stem import WordNetLemmatizer

    table_id = f"top_{num_subreddits}_word_counts"

    query = f"""SELECT *
                FROM
                `{date}.{table_id}`
                """
    client = bigquery_client()

    job_config = bigquery.QueryJobConfig()
    job_config.use_legacy_sql = False

    query_job = client.query(query, job_config=job_config)

    df = query_job.to_dataframe()

    porter = PorterStemmer()
    lancaster=LancasterStemmer()
    wordnet_lemmatizer = WordNetLemmatizer()

    copy = df.copy()

    copy['porter']=copy['word'].apply(lambda x: porter.stem(x))
    copy['lancaster']=copy['word'].apply(lambda x: lancaster.stem(x))
    copy['lemmatised']=copy['word'].apply(lambda x: wordnet_lemmatizer.lemmatize(x))

    store_blob(copy, bucket_name=date, blob_name=f"{table_id}_stems.csv")

def store_stemmed_tfidf(date, num_subreddits, update=False):
    """
    left joining tfidf subset with token lemmatised stems
    """

    print(f"""Getting subreddit word tfidf table for {date}
                for top {num_subreddits} subreddits,
                subsetting tfidf values between {lower} and {upper}
                and exporting to storage bucket""")

    dataset_id = date
    table_id = f"top_{num_subreddits}_subreddit_word_tfidf_subset_stemmed"

    if update:
        try:
            delete_table(dataset_id, table_id)
        except NotFound:
            pass

    query = """ SELECT subreddit, lemmatised, count FROM
            (SELECT * FROM `ella-thesis.2019_01.top_1000_subreddit_word_tfidf_subset` T
            LEFT JOIN (SELECT word as word_2, lemmatised FROM `ella-thesis.2019_01.top_1000_word_counts_stems`) S
            ON T.word = S.word_2)
            """

    run_job(query, dataset_id=dataset_id, table_id=table_id)

    bucket_name = date
    if storage_client().get_bucket(bucket_name).exists() is False:
        storage_client().create_bucket(bucket_name)

    extract_table(dataset_id=date,
                  table_id=table_id,
                  bucket_name=bucket_name,
                  blob_name=table_id + '.csv')

def store_text_sim_stemmed(date, num_subreddits):
    from sklearn.metrics.pairwise import cosine_similarity

    print(f"Getting pairwise tfidf-based cosine simiarity for {date}")
    bucket_name = date

    filename = f"top_{num_subreddits}_subreddit_word_tfidf_subset_stemmed.csv"
    blob = get_blob(bucket_name, filename)

    tokens = parse_csv_blob(blob)
    csr, row_lookup, col_lookup = tokens_to_csr(tokens,
                                                indicator=False,
                                                row='subreddit',
                                                col='lemmatised',
                                                data='tfidf',
                                                return_lookup=True)

    sim_sparse = cosine_similarity(csr, dense_output=False)
    sim_adj = pd.DataFrame(sim_sparse.toarray())
    sim_adj.index = sim_adj.index.map(lambda x: row_lookup[x])
    sim_adj.columns = sim_adj.columns.map(lambda x: row_lookup[x])

    sim_blob_name = insert_to_name(blob.name, '_sim')
    store_blob(sim_adj, bucket_name, sim_blob_name)


def store_text_sim(date, num_subreddits):
    from sklearn.metrics.pairwise import cosine_similarity

    print(f"Getting pairwise tfidf-based cosine simiarity for {date}")
    bucket_name = date

    filename = f"top_{num_subreddits}_subreddit_word_tfidf_subset.csv"
    blob = get_blob(bucket_name, filename)

    tokens = parse_csv_blob(blob)
    csr, row_lookup, col_lookup = tokens_to_csr(tokens,
                                                indicator=False,
                                                row='subreddit',
                                                col='word',
                                                data='tfidf',
                                                return_lookup=True)

    sim_sparse = cosine_similarity(csr, dense_output=False)
    sim_adj = pd.DataFrame(sim_sparse.toarray())
    sim_adj.index = sim_adj.index.map(lambda x: row_lookup[x])
    sim_adj.columns = sim_adj.columns.map(lambda x: row_lookup[x])

    sim_blob_name = insert_to_name(blob.name, '_sim')
    store_blob(sim_adj, bucket_name, sim_blob_name)


def run_text_data(date, num_subreddits=1000):
    get_subreddit_word_counts(date, num_subreddits)
    get_word_counts(date, num_subreddits)
    get_subreddit_word_tfidfs(date, num_subreddits)

    store_tfidf(date, num_subreddits)
    store_text_sim(date, num_subreddits)



""" TOPIC DATA """
def get_subreddit_labels(date, subreddits):
    "Manually add labels for untagged subreddits"

    topics = (pd.read_csv(cache_path(f"{date}/sub_topics.csv"),
                          index_col=0, header=None)[1]
                .str.split(', ', expand=True)[0].to_dict())

    missing = [sub for sub in subreddits if sub not in topics.keys()]

    import time
    import webbrowser

    n = 1
    for sub in missing:
        time.sleep(1)
        webbrowser.open(f'https://www.reddit.com/r/{sub}/top/?t=all')
        cat = input(f'{sub} topic: ')
        topics[sub] = cat
        n += 1

        if n % 10 == 0:
            pd.Series(topics).to_csv(
                    cache_path(f"{date}/sub_topics_updated.csv"))


""" SUBREDDIT DATA """
import json
import praw

def Reddit():
    with open('credentials/praw.json', 'r') as f:
        creds = json.load(f)

    reddit = praw.Reddit(client_id=creds['client_id'],
                         client_secret=creds['client_secret'],
                         user_agent=creds['user_agent'])

    f.close()

    return reddit

def subreddit_data(subreddit):
    from prawcore.exceptions import NotFound
    from prawcore.exceptions import RequestException
    from prawcore.exceptions import Forbidden  

    """Returns a dict of subreddit details via the Reddit API"""
    reddit = Reddit()
    data = reddit.subreddit(subreddit)

    try:
        output = {'description': data.description,
                'public_description': data.public_description,
                'over18': data.over18,
                'advertiser_category': data.advertiser_category,
                'hide_ads': data.hide_ads,
                'header_title': data.header_title,
                'icon_img': data.icon_img,
                'key_color': data.key_color,
                'lang': data.lang,
                'name': data.name,
                'public_traffic': data.public_traffic,
                'quarantine': data.quarantine,
                'rules': data.rules(),
                'title': data.title,
                'created_utc': data.created_utc
                }

    except NotFound:
        print("    praw error: not found")
        output = {'error': 'NotFound'}

    except RequestException:
        print("    praw error: request exception")
        output = {'error': 'RequestException'}

    except Forbidden:
        print("    praw error: forbidden")
        output = {'error': 'Forbidden'}

    return output

def political_subreddit_data():
    import praw
    import json
    import time
    import pandas as pd
    from .metaecho import sub_topics 
    from .tools import cache_path, tables_path

    topics = sub_topics()
    pol_subs = topics[topics=='political'].index

    data = {}
    for sub in pol_subs:
        print(sub)
        data[sub] = subreddit_data(sub)
        time.sleep(2)

    output = pd.DataFrame(data).T

    polarity = {
        'politics':'N',
        'MGTOW':'n/a',
        'The_Donald':'R',
        'ABoringDystopia':'L',
        'COMPLETEANARCHY':'L',
        'ChapoTrapHouse':'L',
        'ENLIGHTENEDCENTRISM':'L',
        'LateStageCapitalism':'L',
        'Libertarian':'R',
        'esist':'L',
        'Trumpgret':'L',
        'SandersForPresident':'L',
        'PoliticalHumor':'N',
        'PoliticalDiscussion':'N',
        'Fuckthealtright':'L',
        'neoliberal':'R',
        'ukpolitics':'N',
        'socialism':'L',
        'MensRights':'n/a',
        'worldpolitics':'N',
        'Conservative':'R',
        'The_Mueller':'R',
        'beholdthemasterrace':'L'
    }

    output['polarity'] = output.index.map(lambda x: polarity[x])
    output['created'] = pd.to_datetime(output['created_utc'], unit='s').dt.date

    output.index.name = 'subreddit'

    output.to_csv(cache_path("political_subreddit_data.csv"))

    table = output[['polarity','created','public_description']].sort_values('created')
    table.columns = ['polarity','created','description']
    table.index = [x.replace('_','\_') for x in table.index]
    table.index.name = 'subreddit'
    table.description = "$" + table.description + "$"
    table.to_csv(tables_path(f"political_subreddit_descriptions.csv"))