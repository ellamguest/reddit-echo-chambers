import os
from scripts.tools import *
from scripts.collection import *
from scripts.diversity import *
import matplotlib.pyplot as plt
import seaborn as sns
from functools import wraps
from .tools import figures_path, get_log, sub_topics

width = 10
height = width / 1.618

plt.style.use('seaborn')
rc = {
    #"text.usetex": True, #causing problems?
    "text.color":"black",
    "font.family": "serif",
    "font.size":12,
    "xtick.labelsize":12,
    "ytick.labelsize":12,
    "axes.titlesize":14,
    "axes.labelsize":14,
    "axes.titlepad":10,
    "figure.figsize": (width,height),
    "figure.dpi":120,
    "savefig.bbox":"tight",
    "savefig.format":"png",
    "savefig.dpi":100,
    "scatter.marker":".",
    "lines.color":"C1",
    "lines.markersize":1
}

def style(f):

    @wraps(f)
    def g(*args, **kwargs):
        with plt.style.context(rc):
            return f(*args, **kwargs)

    return g

@style
def scatterplot(df, x, y, c='blue', title=None, figname=None, size=20, date='2019_01'):
    plt.scatter(df[x],df[y], c=c, s=size, cmap='Blues')
    plt.xlabel(x)
    plt.ylabel(y)

    if title:
        plt.title(title)

    plt.tight_layout()

    if figname:
        plt.savefig(figures_path(f"{date}/{figname}.png"))
        plt.close()

    else:
        plt.show()

 

@style
def hist(df, v, log = False, title=None, figname=None, date='2019_01', cumulative=False):
    data = df[v]
    if log:
        data = get_log(data)
        v = v + ' (log)'
    plt.hist(data, cumulative=cumulative)
    plt.xlabel(v)
    plt.ylabel('freq')

    if title:
        plt.title(title)

    if figname:
        plt.savefig(figures_path(f"{date}/{figname}.png"))

    plt.close()

@style
def demographics_plots():
    d = {
    "Total":11,
    "Men":15,
    "Women":8,
    "18-29":22,
    "30-49":14,
    "50-64":6,
    "65+":1,
    "White":12,
    "Black":4,
    "Hispanic":14,
    "High school or less":6,
    "Some college":14,
    "College graduate":15,
        }

    c = {
    "Total":"black",
    "Men":"blue",
    "Women":"blue",
    "18-29":"red",
    "30-49":"red",
    "50-64":"red",
    "65+":"red",
    "White":"green",
    "Black":"green",
    "Hispanic":"green",
    "High school or less":"purple",
    "Some college":"purple",
    "College graduate":"purple",
        }

    df = pd.DataFrame({"n":d,"col":c}).sort_values(['col','n'], ascending=[False,True])

    plt.barh(y=df.index, width=df.n, color=df.col)
    
    plt.ylabel('Demographic group')
    plt.xlabel('% of U.S. adults who use Reddit')
    plt.tight_layout()
    
    plt.savefig(figures_path("2019_01/reddit_demographics.png"))
    
    plt.close()
    
@style
def topic_barplot(date='2019_01'):
    topics = sub_topics()

    top_topics = topics.value_counts().reset_index()
    top_topics.columns = ['topic', 'subreddit count']
    top_topics = top_topics[top_topics['subreddit count']>=20]

    fig, ax = plt.subplots()
    sns.set(style="whitegrid")
    sns.barplot(x='subreddit count', y='topic', data=top_topics, palette='Set1', ax=ax)

    # add total counts to end of bars
    for i in ax.patches:
        ax.text(i.get_width()+.3, i.get_y()+.6, \
                int(i.get_width()))

    plt.savefig(figures_path(f"{date}/topic_barplot.png"))
    plt.close()

@style
def plot_pol_sub_timeline():
    from scripts.diversity import load_pol_subs
    import pandas as pd
    from matplotlib.lines import Line2D
    
    plt.style.use('seaborn')
    
    df = load_pol_subs()
    df = (df.sort_values('created', ascending=False)
          .reset_index(drop=True)
          .reset_index())

    df['timestamp'] = pd.to_datetime(df['created']).astype(int)
    x,y = 'timestamp','index'
    c=df['col']

    plt.scatter(df[x],df[y], c=c, s=200)

    year_labels = ['2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018']
    year_timestamps = pd.to_datetime(year_labels).astype(int)
    plt.xticks(ticks=year_timestamps, labels=year_labels)
    plt.xlabel('date created')

    plt.yticks(ticks=df[y], labels=df['subreddit'])
    plt.ylabel('subreddit')
    
    legend_elements = [Line2D([0], [0], marker='o', color='w',
                          label='neutral', markerfacecolor='green', markersize=10),
                   Line2D([0], [0], marker='o', color='w',
                          label='right', markerfacecolor='red', markersize=10),
                   Line2D([0], [0], marker='o', color='w',
                          label='left', markerfacecolor='blue', markersize=10),
                   Line2D([0], [0], marker='o', color='w',
                          label='changemyview', markerfacecolor='purple', markersize=10)]
    
    plt.legend(handles=legend_elements)

    plt.tight_layout()

    plt.savefig(figures_path(f"2019_01/political_sub_timeline.png"))
    plt.close()



## CHAPTER 4 PLOTS
def subreddit_level_data(date):
    filename = "sub_diversity_stats.csv"
    bucket_name = dataset_id(date)
    df = get_blob(date, blob_name=filename, df=True)

    return df

def get_logged_df(df):
    log = get_log(df)
    log.columns = [x + '_log' for x in log.columns]

    return log

def make_distplot(values, label):
    sns.distplot(values, kde=False)
    #values.hist()
    plt.xlabel(label)
    plt.ylabel('frequency')
    plt.savefig(figures_path(f"{date}/{label}_hist.png"))
    plt.close() 

def get_mean_com_results(df):
    label = 'avg_com'
    values = df[label]
    logged = get_logged_df(df)[label+'_log']

    return values, logged

def plot_mean_com(values, logged):
    fig_label = 'mean_comments'

    make_distplot(values, fig_label)
    make_distplot(logged, fig_label+'_log')

def get_pol_mean_com_results(values, logged):
    pol_subs = load_pol_subs()
    pol_subs = pol_subs.set_index('subreddit')
    pol_subs.index = pol_subs.index.str.replace(r"\\_",r"_", regex=True)

    pol_subs['mean_comments'] = values.loc[pol_subs.index]
    pol_subs['mean_comments_log'] = logged.loc[pol_subs.index]

    return pol_subs

@style
def get_pol_data(df, stage='within'):
    pol_subs = load_pol_subs()
    pol_subs = pol_subs.set_index('subreddit')
    pol_subs.index = pol_subs.index.str.replace(r"\\_",r"_", regex=True)

    pol_stats = df.loc[pol_subs.index].round(3)
    pol_stats['polarity'] = pol_subs['polarity']
    pol_stats['col'] = pol_subs['col']
    pol_stats.index = pol_stats["polarity"].map(str) + ' - ' + pol_stats.index

    for label in df.columns:
        x = pol_stats.sort_values(label)
        x[label].plot(kind='barh',color=x['col'])

    rank = df.rank(pct=True)

    pol_ranks = rank.loc[pol_subs.index].round(3)
    pol_ranks['polarity'] = pol_subs['polarity']
    pol_ranks['col'] = pol_stats['col']
    pol_ranks.index = pol_ranks["polarity"].map(str) + ' - ' + pol_ranks.index

    heatmap(pol_ranks, annot=True, figname=f'pol_{stage}_rank_heat.png', mask=False)


def get_insub_results(author_data):
    label = 'insub'

    values = author_data[label]
    logged = get_logged_df(author_data)[label+'_log']

    fig_label = 'median_insubreddit'

    make_distplot(values, fig_label)
    make_distplot(logged, fig_label+'_log')

    return

def correlations(data, stage='within'):
    non_log = [x for x in data.columns if 'log' not in x]
    corr = data[non_log].corr()
    heatmap(corr, annot=True, figname=f'{stage}_sub_corr.png')

    return

def between_pol_bar_plots(between_data):
    pol_subs = load_pol_subs()
    pol_subs = pol_subs.set_index('subreddit')
    pol_subs.index = pol_subs.index.str.replace(r"\\_",r"_", regex=True)

    pol_stats = between_data.loc[pol_subs.index].round(3)
    pol_stats['polarity'] = pol_subs['polarity']
    pol_stats['col'] = pol_subs['col']
    pol_stats.index = pol_stats["polarity"].map(str) + ' - ' + pol_stats.index

    for label in between_data.columns:
        print(label)
        
        x = pol_stats.sort_values(label)
        x[label].plot(kind='barh',color=x['col'])
        plt.xlabel(label)
        plt.ylabel('frequency')
        plt.tight_layout()
        plt.savefig(figures_path(f"{date}/pol_between_{label}_hist.png"))
        plt.close()



@style
def between_plots(author_data):
    logged = get_logged_df(author_data)

    for label in author_data.columns:
        print(label)

        values = author_data[label]
        logged_values = logged[label+'_log']

        fig_label = 'between_'+label
        make_distplot(values, fig_label)
        make_distplot(logged_values, fig_label+'_log')


def run():
    # load data
    df = subreddit_level_data(date)
    values, logged = get_mean_com_results(df)
    plot_mean_com(values, logged)

    author_data = load_median_author_level_stats(date)
    between_data = author_data[['sub_count','com_count','avg_com','gini']]

    between_plots(author_data)

    get_insub_results(author_data)

    data = df[['com_count','avg_com']].merge(author_data['insub'], left_index=True, right_index=True)

    logged = get_logged_df(data)

    data = data.merge(logged, left_index=True, right_index=True)

    # make pol plots
    get_pol_data(data, stage='within')
    get_pol_data(between_data, stage='between')

    ranks = data.rank(pct=True)

    return

