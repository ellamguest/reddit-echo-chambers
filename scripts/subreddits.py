"""
- Runs subreddit-level descriptive statistics
- corresponds with Chapter 4: Analysis 1 - Subreddit Level Echo Chambers
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix

from google.cloud import storage
from google.cloud import bigquery

from .tools import (run_job, extract_table, get_blob, delete_table,
                    bigquery_client, storage_client, insert_to_name,
                    cache_path, output_path, figures_path, tables_path,
                    get_log, sub_topics,
                    blob_to_df, store_blob, dataset_id, create_bucket,
                    write_sub_topics)
from .collection import parse_csv_blob, tokens_to_csr, tokens_to_csc
from .plotting import *

date='2019_01'
num_subreddits=1000

""" CALCULATE STATS """
def calc_sub_insub(date, num_subreddits, update=False):
    dataset_id=date
    table_id="subreddit_insub_ranges"

    if update:
        try:
            delete_table(dataset_id, table_id)
        except NotFound:
            pass

    query = f"""with author_total_comments as (
            with authors as (SELECT DISTINCT(author) as author FROM `{date}.top_{num_subreddits}_subreddit_author_subset`)
            SELECT author, sum(num_comments) as total_comments from `{date}.all_subreddit_author_counts` 
            WHERE author in (SELECT author from authors)
            GROUP BY author),

            insubs as (with joined as (SELECT * 
            FROM `{date}.top_{num_subreddits}_subreddit_author_subset` LEFT JOIN author_total_comments
            USING (author))
            SELECT author, subreddit, num_comments, total_comments as author_total_comments, num_comments/total_comments as author_insub
            FROM joined)

            SELECT subreddit,
            ANY_VALUE(min) AS min,
            ANY_VALUE(lower) AS lower,
            ANY_VALUE(median) AS median,
            ANY_VALUE(upper) AS upper,
            ANY_VALUE(max) AS max,
            count(*) AS author_count
            FROM (SELECT subreddit,
                    PERCENTILE_CONT(author_insub, 0) OVER(PARTITION BY subreddit) AS min,
                    PERCENTILE_CONT(author_insub, 0.25) OVER(PARTITION BY subreddit) AS lower,
                    PERCENTILE_CONT(author_insub, 0.5) OVER(PARTITION BY subreddit) AS median,
                    PERCENTILE_CONT(author_insub, 0.75) OVER(PARTITION BY subreddit) AS upper,
                    PERCENTILE_CONT(author_insub, 1) OVER(PARTITION BY subreddit) AS max
                    FROM insubs)
            GROUP BY subreddit"""

    run_job(query, dataset_id=dataset_id, table_id=table_id)

def sub_insub_ranges(date):
    query = f"""SELECT * FROM `{date}.subreddit_insub_ranges`"""

    result = run_job(query)

    df = result.to_dataframe()
    df['iqr'] = df['upper']=df['lower']
    df = df.set_index('subreddit', drop=True)

    return df.sort_index()

""" DIVERSITY MEASURES """
def calc_gini(values, adjusted=True):
    """
    calculate the gini coefficient for a list of values
    a measure of inequality, 0 = total equality, 1 = total inequality
    if adjusted is True, attempts to control for small sample bias
    by (n/(n-1))
    """
    import numpy as np
    np.seterr(divide='ignore', invalid='ignore')
    
    v=np.sort(values)

    sum_iy = 0
    for i, y in enumerate(v):
        i += 1
        sum_iy += i*y

    sum_y = sum(v)
    n = len(v)

    G = (((2*sum_iy)/(n*sum_y)) - ((n+1)/n))

    if adjusted:
        if n>1:
            weight = n/(n-1)
            return G * weight
        else:
            return G

    else:
        return G



def calc_blau(values):
    """
    Calculates Blau's index, a common diversity measure representing variety of the given values
    Aka Gini-Simpson or Gibbs-Martin index
    """
    import numpy as np
    np.seterr(divide='ignore', invalid='ignore')

    pi = values/np.sum(values)
    pi2 = [p**2 for p in pi]
    sum_pi2 = np.sum(pi2)
    
    return 1-sum_pi2

def desc_stats(values):
    from scipy.stats import entropy
    import numpy as np
    np.seterr(divide='ignore', invalid='ignore')
    
    """
    Input:
    - array of comment count values
    Returns:
    - sub_count: number of unique subreddits
    - com_count: total number of comments
    - entropy: entropy of values
    - blau: blau of values
    - gini: gini of values
    """ 

    v = np.nan_to_num(values)
    desc =  {'sub_count':np.count_nonzero(v),
            'com_count':np.sum(v),
            'entropy': entropy(v),    
            'gini': calc_gini(v),
            'blau': calc_blau(v)}

    desc['avg_com'] = (desc['com_count']/
                                desc['sub_count'])

    desc['median_com'] = np.median(v)

    return desc

def fast_desc(values):
    lower, median, upper = np.nanquantile(values, [.25,.5,.75])
 
    return pd.Series({
        'mean':np.nanmean(values),
        'std':np.nanstd(values),
        'min':np.nanmin(values),
        '25%':lower,
        '50%':median,
        '75%':upper,
        'iqr':upper-lower,
        'max':np.nanmax(values)
    })


def calc_subreddit_diversity_stats(csr, sub_lookup, date):
    from tqdm import tqdm

    output = {}

    for i in tqdm(range(csr.shape[0])):
        values = csr[i].data
        output[i] = desc_stats(values)

    results = pd.DataFrame(output).T
    results.index = results.index.map(lambda x: sub_lookup[x])
    results = (results.rename(
                columns={'sub_count':'aut_count'}))

    bucket_name = dataset_id(date)
    store_blob(df=results.round(6),
                bucket_name=bucket_name,
                blob_name='sub_diversity_stats.csv')
    
def calc_subreddit_insubreddit_stats(csr, sub_lookup, date):
    print('calculating author-subreddit insubreddit ratios')
    counts = csr.sum(axis=0)
    insub = csr/counts

    from tqdm import tqdm

    output = {}

    for i in tqdm(range(insub.shape[0])):
        values = insub[i].data
        output[i] = fast_desc(values)

    results = pd.DataFrame(output).T
    results.index = results.index.map(lambda x: sub_lookup[x])

    bucket_name = dataset_id(date)
    store_blob(df=results.round(6),
                bucket_name=bucket_name,
                blob_name='sub_insub_desc_stats.csv')

def calc_aggregate_author_stats(tokens, csc, sub_lookup, date):
    from scipy.sparse import dok_matrix
    from tqdm import tqdm

    num_desc_stats = len(desc_stats([0,0,0,0]).values())
    author_stats = dok_matrix((csc.shape[1], num_desc_stats))

    for i in tqdm(range(csc.shape[1])):
        values = csc[:,i].data
        author_stats[i] = np.fromiter(desc_stats(values).values(), dtype=float)

    dense = author_stats.todense()
    
    inc = tokens_to_csr(tokens, indicator=True, row='author',col='subreddit', return_lookup=False)

    incs = {}
    d = {}

    bucket_name = dataset_id(date)
    stats = desc_stats([0,0,0]).keys()
    
    for i, stat in tqdm(enumerate(stats)):
        print(stat)
        v = dense[:,i]
        m = inc.multiply(v).tocsc()

        for sub in range(m.shape[1]):
            v = m[:,sub].data
            d[sub] = fast_desc(v)

        df = pd.DataFrame(d).T
        df.index = df.index.map(lambda x: sub_lookup[x])
        store_blob(df, bucket_name, f"author_{stat}_aggregate_stats.csv")

def calc_bmax(n, k):
    """
    n: group size, eg number of comments
    k: number of categories, eg number of authors

    returns the maximum possible blau index

    from Solanas at al 2012
    """

    a = np.remainder(n, k)
    """
    a = n - k * int[n/k]
    where int[] denotes the integer function, floor

    which if just the remainder of n / k
    """

    numer = (n*2)*(k-1) + a*(a-k)
    denom = k*(n*2)

    return numer/denom

@style
def sub_diversity_figures(date='2019_01'):
    filename = "sub_diversity_stats.csv"
    bucket_name = dataset_id(date)
    df = get_blob(bucket_name, blob_name=filename, df=True)
    
    log = get_log(df)
    log.columns = [x + '_log' for x in log.columns]

    keep = pd.concat([log[['aut_count_log','com_count_log','avg_com_log']],df[['gini']]], axis=1)

    for col in keep.columns:
        print(col)
        plt.hist(keep[col])
        plt.xlabel(col)
        plt.ylabel('frequency')
        plt.savefig(figures_path(f"{date}/{col}_hist.png"))
        plt.close()

    corr = keep.corr()
    heatmap(corr, annot=True, figname='sub_diversity_corr.png')

    sns.pairplot(keep)
    plt.savefig(figures_path(f"{date}/sub_diversity_pairplot.png"))
    plt.close()

    sub_div_pol_plots(keep, date=date)

@style
def heatmap(df, annot=False, figname=False, date='2019_01', mask=True, center=0):
    copy = df.copy().select_dtypes('number')
    if mask:
    # Generate a mask for the upper triangle
        mask = np.zeros_like(copy, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        mask=mask
    else:
        mask=None
        # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(copy, mask=mask, cmap='RdBu', center=center, annot=annot)

    if figname:
        plt.savefig(figures_path(f"{date}/{figname}"))

    plt.close()


def load_pol_subs():
    df =  (pd.read_csv(tables_path("political_subreddit_descriptions.csv"))
                .sort_values(['polarity','subreddit']))

    col_keys = {'A':'purple','L':'blue','N':'green','R':'red'}
    df['col'] = df.polarity.map(lambda x: col_keys[x])

    return df

def pol_comparison(stats, date='2019_01'):
    import re

    df = stats.copy()
    pol_subs = load_pol_subs()
    pol_subs = pol_subs.set_index('subreddit')
    pol_subs.index = pol_subs.index.str.replace(r"\\_",r"_", regex=True)

 #   subs=['changemyview']
 #   subs.extend(pol_subs.index)

    pol_stats = df.loc[pol_subs.index]
    pol_stats['polarity'] = pol_subs['polarity']
    pol_stats['label'] = pol_stats["polarity"].map(str) + ' - ' + pol_stats.index

    rank = df.rank(pct=True)

    pol_ranks = rank.loc[pol_subs.index]
    pol_ranks['polarity'] = pol_subs['polarity']
    pol_ranks['label'] = pol_ranks["polarity"].map(str) + ' - ' + pol_ranks.index

    # adding polarity reference colors
    col_keys = {'A':'purple','L':'blue','N':'green','R':'red'}
    pol_stats['col'] = pol_stats.polarity.map(lambda x: col_keys[x])
    pol_ranks['col'] = pol_ranks.polarity.map(lambda x: col_keys[x])

    return pol_stats, pol_ranks
    

#@style #breaking pairplot
def sub_div_pol_plots(stats, date='2019_01'):
    pol_stats, pol_ranks = pol_comparison(stats)

    col_keys = {'A':'purple','L':'blue','N':'green','R':'red'}
    pol_stats['col'] = pol_stats.polarity.map(lambda x: col_keys[x])
    pol_ranks['col'] = pol_ranks.polarity.map(lambda x: col_keys[x])

    pol_stats.to_csv(tables_path(f"{date}/pol_sub_stats.csv"))
    pol_ranks.to_csv(tables_path(f"{date}/pol_sub_ranks.csv"))
    
    sns.pairplot(pol_stats, hue='polarity')
    plt.savefig(figures_path(f"{date}/pol_stats_pairplot.png"))
    plt.close()

    copy = pol_ranks.copy()
    copy.index = pol_ranks["polarity"].map(str) + ' - ' + pol_ranks.index
    heatmap(copy[['aut_count_log','avg_com_log','com_count_log','gini']],
                annot=True, figname='pol_rank_heat.png', mask=False)

    sns.pairplot(copy[['aut_count_log','avg_com_log','com_count_log','gini','polarity']], hue='polarity')
    plt.savefig(figures_path(f"{date}/pol_ranks_pairplot.png"))
    plt.close()

@style
def political_agg_aut_plots(pol_stats):
    pol_stats.plot('com_count','gini', kind='scatter', color=pol_stats.col)
    plt.scatter(pol_stats['com_count'],pol_stats['gini'], col=pol_stats['col'])
    plt.savefig(figures_path(f"{date}/pol_agg_com_countxgini.png"))
    plt.close()

def density_histogram(values, title=None, n_bins=20,
            xlabel='portion of authors',
            ylabel='author in-subreddit ratio',
            figsize=(width,height)):

    n, bin_edges = np.histogram(values, n_bins)
    # Normalize it, so that every bins value gives the probability of that bin
    bin_probability = n/float(n.sum())
    # Get the mid points of every bin
    bin_middles = (bin_edges[1:]+bin_edges[:-1])/2.
    # Compute the bin-width
    bin_width = bin_edges[1]-bin_edges[0]
    # Plot the histogram as a bar plot
    plt.figure(figsize=figsize)

    plt.bar(bin_middles, bin_probability, width=bin_width)
    plt.locator_params(tight=True, axis ='x', nbins=n_bins)
    plt.xlim((-0.001,1.001))

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if title:
        plt.title(title)

    plt.show()
    plt.close()

def insub_analysis(date='2019_01'):
    insub = sub_insub_ranges(date=date)
    insub = insub.drop('author_count', axis=1)

    sns.pairplot(insub)
    plt.savefig(figures_path(f"{date}/insub_pairplot.png"))
    plt.close()

    pol_stats, pol_ranks = pol_comparison(insub)
    pol_stats.to_csv(tables_path(f"{date}/pol_insub_stats.csv"))
    pol_ranks.to_csv(tables_path(f"{date}/pol_insub_ranks.csv"))

    sns.pairplot(pol_stats, hue='polarity')
    plt.savefig(figures_path(f"{date}/pol_insub_stats_pairplot.png"))
    plt.close()

    sns.pairplot(pol_ranks, hue='polarity')
    plt.savefig(figures_path(f"{date}/pol_insub_ranks_pairplot.png"))
    plt.close()

    heatmap(pol_ranks, annot=True, figname='pol_insub_rank_heat.png', mask=False)

    sns.violinplot(x='median', y='polarity',data=pol_ranks)
    plt.xlabel('median insub value')
    plt.savefig(figures_path(f"{date}/pol_insub_median_violin.png"))
    plt.close()

    sns.violinplot(x='iqr', y='polarity',data=pol_ranks)
    plt.xlabel('median insub value')
    plt.savefig(figures_path(f"{date}/pol_insub_iqr_violin.png"))
    plt.close()

def author_diversity(date='2019_01', inverse_gini=True):
    author_stats = ['avg_com', 'com_count', 'sub_count', 'gini']
    d = {}
    for s in author_stats:
        df = get_blob(dataset_id(date), f"author_{s}_aggregate_stats.csv", df=True)
        d[s] = df['50%']
    medians = pd.DataFrame(d)
    if inverse_gini:
        print('Inverting gini until re-run statistics')
        medians['gini'] = 1-medians['gini']
    
    s, r = pol_comparison(medians)
    heatmap(r, annot=True, figname='aut_median_pol.png', mask=False)

    heatmap(medians.corr(), annot=True, figname='aut_median_corr.png')
    sns.pairplot(medians)
    plt.savefig(figures_path(f"{date}/aut_median_pairplot.png"))
    plt.close()


""" plotting """
@style
def pol_barplot(df, v, savefig=False):
    data = df.sort_values(v, ascending=True).reset_index(drop=False)

    fig, ax = plt.subplots()
    ax.barh(data.index, data[v], color = data['col'])
    ax.set_yticks(data.index)
    ax.set_yticklabels(data['subreddit'])
    
    plt.tight_layout()
    
    if savefig:
        plt.savefig(figures_path(f"{date}/pol_within_{v}.png"))

    plot.close()
    
def plot_pol_within_barplots():
    pol_within = (pd.read_csv(
                    tables_path(f"{date}/pol_sub_stats.csv"),
                    index_col='label'))

    pol_barplot(pol_within, 'median_com', savefig=True)

    pol_barplot(pol_within, 'com_count', savefig=True)
    pol_barplot(pol_within, 'aut_count', savefig=True)

    # median_com
    filename = "sub_diversity_stats.csv"
    bucket_name = dataset_id(date)
    df = get_blob(date, blob_name=filename, df=True)
    pol_stats, pol_ranks = pol_comparison(df)

    pol_barplot(pol_stats, 'median_com', savefig=True)

    between = load_median_author_level_stats(date)
    pol_between, pol_ranks = pol_comparison(between)
    pol_barplot(pol_between, 'insub', savefig=True)

def within_scatterplots(date):
    within = load_within_stats(date)
    pol_cols = pol_subs.set_index('subreddit')['col'].to_dict()
    within['col'] = (within.index.map(
                    lambda x: pol_cols[x] if x
                    in pol_cols.keys() else 'grey'))
    scatterplot(within,
                'median_com', 'insub',
                title=None,
                figname="within_scatterplot",
                size=100, date='2019_01')

    pol_stats, pol_ranks = pol_comparison(within)

    scatterplot(pol_stats,
                'median_com', 'insub',
                title=None,
                c=pol_stats['col'],
                figname="within_pol_scatterplot",
                size=300, date='2019_01')

def termeh_insub_plots(within):
    within['median_total_com'] = within['median_com']/within['insub']

    termeh = pd.DataFrame({'incount':within['median_com'],
                          'total':within['median_total_com']})
    termeh['outcount'] = termeh['total']-termeh['incount']
    termeh = termeh.sort_values('outcount')

    plt.scatter(termeh['outcount'],
                termeh['incount'])
    plt.plot([0,100],[0,100], color='grey')

    plt.xlabel('median author out comments')
    plt.ylabel('median author in comments')
    
    plt.tight_layout()
    plt.savefig(figures_path(f"{date}/in_out_scatterplot.png"))
    plt.close()

    pol_termeh, pol_ranks = pol_comparison(termeh)

    (plt.scatter(x=pol_termeh['outcount'],
                 y=pol_termeh['incount'],
                 color=pol_termeh['col'],
                s=100))

    plt.plot([0,150],[0,150], color='grey')

    plt.xlabel('median author out comments')
    plt.ylabel('median author in comments')

    plt.xlim(xmin=-0.5)
    plt.ylim(ymin=-0.5)
    
    plt.tight_layout()
    plt.savefig(figures_path(f"{date}/in_out_pol_scatterplot.png"))
    plt.close()


""" RUN """

def run_diversity_stats(date, num_subreddits=1000):
    calc_sub_insub(date, num_subreddits)

    blob = get_blob(bucket_name=date,
                    blob_name= f"top_{num_subreddits}_subreddit_author_subset.csv")

    tokens = parse_csv_blob(blob)

    csr, sub_lookup, aut_lookup = tokens_to_csr(tokens,
                                                indicator=False,
                                                return_lookup=True)

    calc_subreddit_diversity_stats(csr, sub_lookup, date)

    calc_subreddit_insubreddit_stats(csr, sub_lookup, date)

    csc = tokens_to_csc(tokens, indicator=False, return_lookup=False)

    calc_aggregate_author_stats(tokens, csc, sub_lookup, date)

def load_diversity_stats(date):
    filenames = {'sub_diversity':'sub_diversity_stats.csv',
                "insubreddit":"sub_insub_desc_stats.csv"}

    stats = [
        'avg_com','com_count',
        'gini', 'sub_count'
        ]
    for stat in stats:
      filenames[f"aut_{stat}"] = f"author_{stat}_aggregate_stats.csv"
  
    bucket_name = dataset_id(date)
    dfs = {}
    for k,v in filenames.items():
      dfs[k] = get_blob(bucket_name, blob_name=v, df=True)
  
    return pd.concat(dfs, axis=1)

def load_within_stats(date):
    filename = "sub_diversity_stats.csv"
    bucket_name = dataset_id(date)
    sub_diversity = get_blob(date, blob_name=filename, df=True)

    insubs = sub_insub_ranges(date)
    within = pd.DataFrame({'median_com':sub_diversity['median_com'],
                       'mean_com':sub_diversity['avg_com'],
                       'insub':insubs['median']})

    return within


def load_median_author_level_stats(date):
    bucket_name = dataset_id(date)

    dfs = {}
    stats = [
        'avg_com','com_count',
        'gini', 'sub_count'
        ]
    for stat in stats:
        blob_name = f"author_{stat}_aggregate_stats.csv"
        dfs[stat] = get_blob(bucket_name, blob_name=blob_name, df=True)['50%']

    dfs["insub"] = sub_insub_ranges(date)['median']
    
    return pd.concat(dfs, axis=1)[['sub_count','com_count','avg_com','gini','insub']]
    
    
def writeup():
    date='2019_01'
    sub_diversity_figures(date='2019_01')
    results = load_diversity_stats(date)
    pol_comparison(results)

    

    

