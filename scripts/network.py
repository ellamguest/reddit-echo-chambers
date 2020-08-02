import numpy as np
import pandas as pd
from .tools import (get_upper_edges,  get_log, create_directories,
                    cache_path, output_path, figures_path, tables_path,
                    get_blob, load_defaults, sub_topics, store_blob, rescale)
from .diversity import load_pol_subs
from .setup import *
from .plotting import *
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import community
import re

date='2019_01'
num_subreddits=1000

# DATA PROCESSING
@style
def raw_counts(date='2019_01'):
    filename = cache_path(f"{date}/raw_subreddit_counts.csv")
    if filename.exists:
        df = pd.read_csv(filename, index_col=0)
    else:
        from .tools import bigquery, bigquery_client
        client = bigquery_client()
        job_config = bigquery.QueryJobConfig()
        job_config.use_legacy_sql = False

        query = f"""
        SELECT *
        FROM `author-subreddit-counts.{date}.subreddit_total_counts`
        """

        query_job = client.query(query, job_config=job_config)

        df = query_job.to_dataframe()
        df.to_csv(filename)

    df = np.log(df.select_dtypes(int))

    fig, axs = plt.subplots(2, 1, figsize = (height, width), sharex=False, sharey=False)

    # scatterplot
    axs[0].scatter(x=df['num_comments'],y=df['num_authors'], s=5)
    axs[0].set_xlabel('comment count (log)')
    axs[0].set_ylabel('author count (log)')

    # author hist
    axs[1].hist(df['num_authors'], density=True)
    axs[1].set_xlabel('author count (log)')
    axs[1].set_ylabel('subreddit count (normed)')
    
    plt.tight_layout()
    plt.savefig(figures_path(f"{date}/raw_subreddit_counts.png"), format="png")
    plt.close()

def fetch_data(date, num_subreddits=1000):
    author = get_blob(
        bucket_name=date,
        blob_name=f"top_{num_subreddits}_subreddit_author_subset_adj.csv",
        df=True)

    text = get_blob(
        bucket_name=date,
        blob_name=f"top_{num_subreddits}_subreddit_word_tfidf_subset_sim.csv",
        df=True)

    return author, text

def expected_adj(input_adj):
    """
    Takes observed adjacency matrix
    Return expected adjaceny matrix based on:
        - node degree distribution
        - # of edges
    """

    adj = input_adj.copy()
    d = np.diagonal(adj)
    copy = np.matrix(adj.copy())
    np.fill_diagonal(copy, 0)

    m = np.nansum(copy)
    kikj = d[:, None]*d[None, :]
    ex = kikj/(2*m)

    return pd.DataFrame(ex, index=input_adj.index, columns=input_adj.columns)

def overlap_coefficient(input_adj):
    """
    Take observed adjacency matrix
    Gets the sum of all rows
    For edge i,j returns edge/size of smallest node
    """
    adj = input_adj.copy()
    d = np.diagonal(adj)
    deg_dict = dict(zip(adj.columns, d))
    
    edges = get_upper_edges(adj, sorted=False)
    df = pd.DataFrame(edges, columns=['weight'])
    df['deg_A'] = df.index.map(lambda x: deg_dict[x[0]])
    df['deg_B'] = df.index.map(lambda x: deg_dict[x[1]])
    df['min_deg'] = df[['deg_A','deg_B']].min(axis=1)
    df['oc'] = df['weight']/df['min_deg']

    return df

@style
def author_counts(author, date, num_subreddits=1000):
    plt.hist(np.log(np.diag(author)))
    plt.xlabel("author count (log)")
    plt.savefig(figures_path(f"{date}/author_count_hist.png"))
    plt.close()

def load_sim_data(date='2019_01'):
    author, text = fetch_data(date)

    print('calculating oe')
    author_log = get_log(author)
    expected = expected_adj(author)
    oe = author/expected
    oe_log = get_log(oe)
    text_log = get_log(text)

    print("compiling datasets")
    edges = {
            'author':author,
            'author_log':author_log,
            'expected':expected,
            'oe':oe,
            'oe_log':oe_log,
            'text':text,
            'text_log':text_log}

    for k,v in edges.items():
        edges[k] = get_upper_edges(v, sorted=False)

    data = pd.concat(edges, axis=1)

    return data

@style
def scatterplot(edges, date='2019_01'):
    from numpy.polynomial.polynomial import polyfit

    data = edges.copy()

    y = data['oe_log']
    x = data['text_log']

    # Fit 
    b, m = polyfit(x, y, 1)

    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    plt.plot(x, y, '.', markersize=2)
    plt.plot(x, b + m * x, color="C1")
    circle = plt.Circle((-1.5,10),1.8, fill=False, lw=3)
    ax.add_patch(circle)

    if 'source_topic' not in data.columns:
        data = label_subs(data)
    # annotate outlier edges
    sfw = data[(data['source_topic']!='porn')|(data['target_topic']!='porn')]

    text_min = sfw.text_log.idxmin()
    text_max = sfw.text_log.idxmax()
    oe_min = sfw.oe_log.idxmin()
    oe_max = sfw.oe_log.idxmax()

    ids = list(set([text_min, text_max, oe_min, oe_max]))

    from matplotlib.lines import Line2D

    legend_elements = []

    for i, col in enumerate(['C2', 'C3', 'C4']):
        plt.plot(x.loc[ids[i]], y.loc[ids[i]], 'o', markersize=10, alpha=1, color=col)
        label = data.loc[ids[i]]['source'] + " -- " + data.loc[ids[i]]['target']
        legend_elements.append(Line2D([0],[0], marker='o', markersize=10, alpha=1, color='w', markerfacecolor=col, label=label))
    plt.legend(handles=legend_elements, loc='best')

    plt.savefig(figures_path(f"{date}/scatterplot_reg.png"), format='png')
    plt.close()

def regression(data, date, x='text_log', y='oe_log'):
    import statsmodels.api as sm

    copy = data.copy()
    df = copy[[x,y]].dropna()
    y = df[y]
    X = df[x]

    model = sm.OLS(y, X).fit()

    reg_summary = model.summary().as_csv()
    store_blob(df=reg_summary, bucket_name=date, blob_name='reg_summary.csv', csv=False)

    ms = model.summary().as_csv().replace(' ', '')
    model_summary = ms.lstrip("OLSRegressionResults\n")

    with open(tables_path(f"{date}/regression_summary_new.csv"), "w") as text_file:
        text_file.write(model_summary)

    copy['pred'] = model.predict(X)
    copy['resid'] = model.resid

    return copy

def run_residuals(date, num_subreddits=1000, data=None):
    if data is None:
        data = load_sim_data(date=date)

    print("calculating residuals")
    output = regression(data, date, x='text_log', y='oe_log')

    print('storing residuals')

    store_blob(output, bucket_name=date, blob_name=f"{num_subreddits}_residuals_output.csv")

    return output

def load_residuals(date, num_subreddits=1000):
    return get_blob(bucket_name=date,
                blob_name=f"{num_subreddits}_residuals_output.csv",
                df=True)

""" PLOTTING """
@style
def diagnostics(output, date='2019_01'):
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 2, figsize = (width, width), sharex=False, sharey=False)

    # fitted vs observed values scatter plot
    axs[0,0].scatter(output.pred, output.oe_log,
                    alpha=0.1, s=1)
    axs[0,0].set_xlabel('Fitted')
    axs[0,0].set_ylabel('Observed')
    axs[0,0].set_title('Fitted vs Observed Values')

    # fitted values vs residuals scatter plot
    axs[0,1].scatter(output.pred, output.resid,
                    alpha=0.1, s=1) # TODO: hexbin?
    axs[0,1].set_xlabel('Fitted')
    axs[0,1].set_ylabel('Residuals')
    axs[0,1].set_title('Fitted values vs Residuals')

    # scale-location plot
    from .tools import standardise

    if 'resid_standard' not in output.columns:
        print('Adding standardised residuals')
        output['resid_standard'] = standardise(output['resid'])
        output['resid_standard_sqrt'] = np.sqrt(output['resid_standard'])

    axs[1,0].scatter(output.pred, output.resid_standard_sqrt,
                     alpha=0.1, s=1) # TODO: hexbins?
    axs[1,0].set_xlabel('Fitted values')
    axs[1,0].set_ylabel(r'$\sqrt{Standardised\:Residuals}$')
    axs[1,0].set_title('Scale-Location')

    # qqplot
    from scipy.stats import probplot
    probplot(output.resid, plot=axs[1,1])
    axs[1,1].lines[0].set_color('C0')
    axs[1,1].set_title('Normal Q-Q')

    plt.subplots_adjust(top=1)

    plt.tight_layout()
    plt.savefig(figures_path(f"{date}/_diagnostics.png"), format="png")
    plt.close()


""" TOPIC LABELLING """

def label_subs(edges):
    """ Add topic labels to dataframe edges
        - assumes edges has multiindex [sub1, sub2]
        - resets index
        - maps columns 'sub1_topic' and 'sub2_topic'"""

    topics = sub_topics()

    copy = edges.copy()
    if 'source' not in copy.columns:
        copy = copy.reset_index()
        copy.columns = ['source', 'target'] + list(copy.columns[2:])

    copy['source_topic'] = copy['source'].map(lambda x: topics[x])
    copy['target_topic'] = copy['target'].map(lambda x: topics[x])

    return copy

@style
def plot_degree_hist(top):
    """input: top is the list of edges in the network"""
    
    edges = top.copy()[['source','target','resid']]
    edges_rev = edges.copy()
    edges_rev.columns = ['target','source','resid']
    directed_edges = pd.concat([edges,edges_rev], sort=True)

    degrees = directed_edges.source.value_counts().sort_values()

    degrees.hist()
    plt.xlabel('degree')
    plt.ylabel('count')
    plt.tight_layout()

    plt.savefig(figures_path(f"{date}/degree_hist.png"))
    plt.close()

    np.log(degrees).hist()
    plt.xlabel('degree (log)')
    plt.ylabel('count')
    plt.tight_layout()

    plt.savefig(figures_path(f"{date}/degree_hist_log.png"))
    plt.close()

@style
def network_topic_barplot(date='2019_01'):
    topics = load_membership()

    top_topics = topics['topic'].value_counts().reset_index()
    top_topics.columns = ['topic', 'subreddit count']
    top_topics = top_topics[top_topics['subreddit count']>=20]

    fig, ax = plt.subplots()
    sns.set(style="whitegrid")
    sns.barplot(x='subreddit count', y='topic', data=top_topics, palette='Set1', ax=ax)

    # add total counts to end of bars
    for i in ax.patches:
        ax.text(i.get_width()+.3, i.get_y()+.6, \
                int(i.get_width()))

    plt.savefig(figures_path(f"{date}/topic_barplot_network.png"))
    plt.close()

@style
def community_topics(membership=None, date="2019_01", num_subreddits=1000):
    topic_barplot(date='2019_01')

    if membership is None:
        membership = load_membership(date, num_subreddits)

    top_topics = membership.topic.value_counts().sort_values(ascending=False)
    min_topic_size=20
    keep_topics = top_topics[top_topics>min_topic_size].index

    piv = membership.groupby('size_order')['topic'].value_counts().unstack()[keep_topics]
    piv.T.round().to_csv(tables_path(f"{date}/community_topic_counts.csv"))

    sns.set_style("white")

    fig, ax = plt.subplots()
    sns.heatmap(piv.pipe(lambda s: s/s.sum()).round(2).pipe(lambda s: s.where(s > 0.05)).T, annot=True, cmap='Reds', ax=ax)

    plt.xlabel('community')
    plt.ylabel('subreddit topic')
    plt.savefig(figures_path(f"{date}/topic_breakdown_heatmap.png"))
    plt.close()

    fig, ax = plt.subplots()
    sns.heatmap(piv.divide(piv.sum(axis=1), axis='rows').round(2).pipe(lambda s: s.where(s > 0.05)).T, annot=True, cmap='Blues', ax=ax)

    plt.xlabel('community')
    plt.ylabel('subreddit topic')
    plt.savefig(figures_path(f"{date}/community_breakdown_heatmap.png"))
    plt.close()

    comm_size = pd.read_csv(tables_path(f"{date}/comm_size.csv"))
    labels = comm_size['size_order'].map(str) + ': ' + comm_size['labels']
    fig, ax = plt.subplots()
    ax.barh(labels, comm_size['size'])
    ax.set_ylabel('community label')
    ax.set_xlabel('number of subreddits')
    plt.gca().invert_yaxis()
    for i in ax.patches:
        ax.text(i.get_width()+.6, i.get_y()+.6, \
                int(i.get_width()))
    plt.tight_layout()
    plt.savefig(figures_path(f"{date}/community_labels.png"))
    plt.close()

    
# COMMUNITY DETECTION
def subset_df(df, variable='resid', q=0.95):
    copy = df.copy()
    threshold = copy[variable].quantile(q)

    return copy[copy[variable] >= threshold]

def edges_to_graph(df, weight):
    edges = df[['source', 'target']]
    if weight:
        edges['weight'] = df[weight]
    
    return nx.from_pandas_edgelist(df, edge_attr='weight')

def add_partitions(G):
    print("Getting louvain partitions, adding community to node attributes")
    p = community.best_partition(G, randomize=False)
    nx.set_node_attributes(G, p, 'community')

def get_node_data(G):
    return pd.DataFrame.from_dict(dict(G.nodes(data=True))).T

def partition_network(G):
    if nx.get_node_attributes(G, 'community') == {}:
        add_partitions(G)
    node_data = get_node_data(G)
    mems = node_data['community'].to_dict()

    edgelist = nx.to_pandas_edgelist(G)
    edgelist['source']= edgelist['source'].map(lambda x: mems[x])
    edgelist['target']= edgelist['target'].map(lambda x: mems[x])

    new_edges = edgelist.groupby(['source','target'])['weight'].count().reset_index()

    return new_edges

def ei_index(edgelist):
    internal = edgelist[edgelist['source'] == edgelist['target']].groupby('source')['weight'].sum()
    external = edgelist[edgelist['source'] != edgelist['target']].groupby('source')['weight'].sum()
    diff = external - internal
    total = external + internal
    ei = (diff/total).fillna(-1)

    return pd.DataFrame({'internal': internal,
                        'external': external,
                        'ei_index' : ei}).fillna(0)

def calc_yules(top):
    """
    yule's q
    also known as the Yule coefficient of association
    cross tab
    alter-ego similarity

            same | diff
    tie    |  A  |  B  |
    no tie |  C  |  D  |

    Q = (AD-BC)(AD+BC)

    AD = # same ties x # diff no ties
    BC = # diff ties x # same no ties

    numerator ~= expected ties minus unexpected ties
    denominator ~= total ties

    strength of the relationship between community co-membership and sharing a tie, controlling for group size
    can fall between – 1 and +1

    -1 one would indicate a total negative assocation
    a perfect 0.0 indicates no relationship between tie status and similarity
    perfect 1.0 therefore indicates a total positive association
    (i.e. all subreddits in the community share edges, and none shares an edge with a sub from another community)

    """

    top_rev = top.copy()
    old_columns = list(top.columns[2:])
    new_columns = ['target','source']
    new_columns.extend(old_columns)

    top_rev.columns = new_columns
    directed_edges = pd.concat([top,top_rev], sort=True)
    directed_edges.head()

    membership = load_membership()

    membership_dict = membership['size_order'].to_dict()

    x = directed_edges.copy()
    x['source_comm'] = x['source'].map(lambda x: membership_dict[x])
    x['target_comm'] = x['target'].map(lambda x: membership_dict[x])

    m = x.groupby('source_comm')['target_comm'].value_counts().unstack()
    As = np.diagonal(m)/2

    copy = m.copy()
    np.fill_diagonal(copy.values, 0)
    Bs = copy.sum()

    n = membership.size_order.value_counts()
    same_possible = (n*(n-1))/2
    Cs = same_possible - As
    Cs

    N = n.sum()
    alter_n = N-n
    alter_n
    diff_possible = (n*alter_n)
    Ds = diff_possible - Bs

    AD = np.multiply(As,Ds)
    BC = np.multiply(Bs,Cs)

    num = AD-BC
    denom = AD+BC

    Q = num.divide(denom)

    Q_dict = Q.to_dict()

    comm_size = pd.read_csv(tables_path(f"{date}/community_data.csv"))
    comm_size['yulesQ'] = comm_size['sizeOrder'].map(lambda x: Q_dict[x])
    comm_size.round(2).to_csv(tables_path(f"{date}/community_data_yules.csv"), index=False)

@style
def resid_hist(output, date='2019_01'):
    plt.hist(output['resid'])
    plt.savefig(figures_path(f"{date}/residuals_hist.png"))
    plt.close()

def map_topic(topics, x):
    try:
        return topics.loc[x]
    except:
        return None

def community_detection(output, date, num_subreddits=1000):
    edges = output[['source','target', 'resid']]
    edges.columns = ['source','target', 'weight']

    print("running community detection")
    top_edges = subset_df(edges, 'weight', q=0.95)
    G = edges_to_graph(top_edges, 'weight')
    add_partitions(G)

    membership = get_node_data(G).sort_values('community')

    topics = sub_topics()
    membership['topic'] = membership.index.map(lambda x: map_topic(topics, x))

    store_blob(df=membership, bucket_name=date, blob_name=f"community_membership.csv")

    community_edges = partition_network(G)
    store_blob(df=membership, bucket_name=date, blob_name="community_edges.csv")

    ei = ei_index(community_edges)

    store_blob(df=ei, bucket_name=date, blob_name=f"ei_index.csv")

    comm_size = (membership['community'].value_counts()
                .sort_index())

    table_results = pd.DataFrame({'n': comm_size, 'ei': ei['ei_index']})
    table_results.index.name = 'community'
    store_blob(df=table_results.round(1), bucket_name=date, blob_name=f"community_modularity.csv")

def manual_community_topic_labels(date='2019_01'):
    """these are the manual community topic labels for 2019_01"""
    comm_size = get_blob(bucket_name=date, blob_name=f"community_modularity.csv", df=True)
    comm_size.name = 'size'
    comm_size.index.name = 'community'
    comm_size = comm_size.reset_index()
    labels = {
                0:'images/discussion',
                1:'discussion/tv',
                2:'runescape',
                3:'funny/images',
                4:'gaming/tech',
                5:'pol/geo',
                6:'generalist',
                7:'porn',
                8:'music',
                9:'sports',
                10:'SE/DK',
                11:'SP/IT/PT',
                12:'NL'
            }

    membership = load_membership()
    order_keys = membership.drop_duplicates('community').set_index('community')['size_order'].to_dict()

    comm_size['size_order'] = comm_size['community'].map(lambda x: order_keys[x])
    comm_size = comm_size.sort_values('size_order').reset_index(drop=True)

    comm_size['labels']=comm_size['community'].map(lambda x: labels[x])
    comm_size.to_csv(tables_path(f"{date}/comm_size.csv"))

def load_membership(date='2019_01', num_subreddits=1000):
    membership = (pd.read_csv(output_path(
                        f"{date}/{num_subreddits}_community_membership.csv"),
                        index_col=0))

    comm_size = pd.read_csv(tables_path(f"{date}/comm_size.csv"))
    order_key = comm_size.set_index('community')['size_order'].to_dict()
    membership['size_order'] = membership['community'].map(lambda x: order_key[x])

    return membership


def local_density(membership=None, date='2019_01', num_subreddits=1000):
    if membership is None:
        membership = load_membership()

    nodecounts = membership.community.value_counts()
    possible_edges = ((nodecounts*nodecounts)-(nodecounts))/2

    edges = pd.read_csv(output_path(f"{date}/community_edges.csv"), index_col=0)
    local_edges = edges[edges['source'] == edges['target']].groupby('source')['weight'].sum()

    result = pd.DataFrame({'density':local_edges/possible_edges}).round(2)
    result.index.name  = 'community'

    result.to_csv(tables_path(f"{date}/local_densities.csv"))

def small_communities(membership=None, date='2019_01', num_subreddits=1000):
    if membership is None:
        membership = load_membership(date, num_subreddits)
    comm_size = membership.community.value_counts()
    small = comm_size[comm_size<10].index
    output = membership[membership.community.isin(small)]
    output.index.name = 'subreddit'

    output.to_csv(tables_path(f"{date}/small_communities.csv"))

def default_communities(membership=False, date='2019_01'):
    if membership is False:
        membership = load_membership()
    defaults = load_defaults()
    membership['default'] = membership.index.map(lambda x:
                        True if x in defaults.values else False)

    def_labels = membership[membership['default']==True].drop('default', axis=1)
    def_labels.index.name = 'subreddit'
    def_labels.to_csv(tables_path(f"{date}/default_labels.csv"))

    def_com = def_labels.community.value_counts().sort_index()

    def_com.index.name = 'community'
    def_com.name = 'n'
    (def_com.reset_index().to_csv(
                    tables_path(f"{date}/default_comm_counts.csv"),
                    index=False))



def comparing_community_algortihms():
    residuals = pd.read_csv("/Users/emg/GitHub/thesis/output/2019_01/1000_residuals_output_utf8.csv")
    edges = residuals[['source','target', 'resid']]
    edges.columns = ['source','target', 'weight']

    top_edges = subset_df(edges, 'weight', q=0.95)
    G = edges_to_graph(top_edges, 'weight')
    add_partitions(G)

    membership = get_node_data(G).sort_values('community')

    from networkx.algorithms.community import greedy_modularity_communities
    c = list(greedy_modularity_communities(G, weight='weight'))

    clauset_search = {}

    for i,x in enumerate(c):
        for subreddit in x:
            clauset_search[subreddit] = i

    membership['clauset'] = membership.index.map(lambda x: clauset_search[x])

    from networkx.algorithms.community import girvan_newman
    communities_generator = community.girvan_newman(G)
    top_level_communities = next(communities_generator)
    next_level_communities = next(communities_generator)
    sorted(map(sorted, next_level_communities))

def offset_pos(pos, x_offset=0, y_offset=0.1):
    pos_offset = {}
    for k,v in pos.items():
        x,y = v
        pos_offset[k] = x+x_offset, y+y_offset
    return pos_offset

def community_graph():
    el = pd.read_csv("/Users/emg/GitHub/thesis/output/2019_01/community_edgelist.csv")
    G = nx.from_pandas_edgelist(el, edge_attr='weight')

    nl = pd.read_csv("/Users/emg/GitHub/thesis/output/2019_01/community_nodelist.csv")
    nl = nl.set_index('Id')

    nl['node_size'] = [x*1000 for x in rescale(nl['numSubreddits'])]

    for col in nl.columns:
        d = nl[col].to_dict()
        nx.set_node_attributes(G, d, col)
        
    G.remove_edges_from(G.selfloop_edges())
    G.remove_edges_from([x for x in G.edges(data='weight') if x[-1] < 5])
    G.remove_nodes_from(list(nx.isolates(G)))

    colors = sns.color_palette("Dark2", 9)
    colors.insert(4,'red')

    color_dict = {}
    for i, x in enumerate(G.nodes):
        color_dict[x] = colors[i]
        
    nx.set_node_attributes(G, color_dict, 'color')

    f = plt.figure(1)
    ax = f.add_subplot(1,1,1)

    weights = [x[-1] for x in G.edges(data='weight')]
    edge_weights = rescale(weights, new_min=2,new_max=10)

    size = list(dict(G.node(data='numSubreddits')).values())
    node_size = rescale(size, new_min=1000,new_max=4500)

    pos = nx.kamada_kawai_layout(G)
    nx.draw_networkx(G, pos=pos, ax=ax,
                    node_color=colors[:9], node_size=node_size,
                    with_labels=False,
                    width=edge_weights, edge_color='grey',
                    alpha=0.7)

    label_dict = dict(G.nodes(data='labels'))
    pos_offset = offset_pos(pos)
    nx.draw_networkx_labels(G, pos=pos_offset,
                            with_labels=True, labels=label_dict,font_size=14)
                
    plt.axis('off')
    f.set_facecolor('w')

    f.tight_layout()
    plt.savefig(figures_path(f"{date}/community_graph.png"))
    plt.close()

""" END OF COMMUNITY DETECTION"""

def aggregate_community_data(date):
    size = pd.read_csv(tables_path(f"{date}/comm_size.csv"), index_col=0)
    modularity = pd.read_csv(tables_path(f"{date}/community_modularity.csv"), index_col=0).drop('n', axis=1)
    density = pd.read_csv(tables_path(f"{date}/local_densities.csv"), index_col=0)

    defaults = pd.read_csv(tables_path(f"{date}/default_comm_counts.csv"), index_col=0)
    data = [size, modularity, density, defaults]

    comm_data = pd.concat(data, axis=1)
    comm_data.columns = ['numSubreddits', 'labels', 'sizeOrder','ei', 'density', 'numDefaults']

    comm_data.sort_values('sizeOrder').to_csv(tables_path(f"{date}/community_data.csv"))

""" SUBSETTING """

def get_sub(edges, subreddit):
    if 'source' in edges.columns:
        subset = edges[(edges['source']==subreddit)|(edges['target']==subreddit)].copy()
    else:
        indices = [v for v in edges.index.values if subreddit in v]
        subset = edges.loc[indices].copy()

    return subset

def get_edge(edges, subA, subB):
    A = get_sub(edges, subA)
    if A.shape[0] == 0:
        print(subA, "not found")
        return
    B = get_sub(A, subB)
    if B.shape[0] == 0:
        print(subB, "not found")
    else:
        return B

@style
def pol_resids(labelled, membership=None, date='2019_01', num_subreddits=1000):
    if membership is None:
        membership = load_membership(date, num_subreddits)
    
    if 'source_topic' not in labelled.columns:
        label_subs(labelled)

    pol_communities = (membership[membership['topic']=='political']
                                .community.value_counts())
    print(pol_communities.shape[0], "communit(ies) include at least one political subreddit")
    pol_comm = (pol_communities.index[0])

    membership_subset = membership[membership['community']==pol_comm]
    topic_counts = membership_subset.topic.value_counts()
    topic_counts.columns = 'n'
    topic_counts.index.name = 'topic'
    topic_counts.to_csv(tables_path(f"{date}/pol_comm_topic_counts.csv"), header=True)

    subreddits = membership_subset.index
    subset = labelled[(labelled['source'].isin(subreddits)) & (labelled['target'].isin(subreddits))][['source','target','source_topic','target_topic','resid']].sort_values('resid', ascending=False)

    subset.resid.hist()
    plt.savefig(figures_path(f"{date}/pol_residuals_hist.png"))
    plt.close()

    top = subset_df(subset, 'resid', q=0.95)
    (top.rename({'resid':'weight'}, axis=1)
        .to_csv(output_path(f"{date}/pol_edges.csv")))
    pol_nodes = membership[membership['community']==5]
    keep = pol_nodes.topic.value_counts().index[:6].values
    pol_nodes['topic'] = pol_nodes['topic'].map(lambda x: x if x in keep else 'other')
    pol_nodes.to_csv(output_path(f"{date}/pol_nodes.csv"))

    sub_edges = pd.concat([top.source.value_counts(),top.target.value_counts()], axis=1, sort=True).fillna(0)
    sub_edges['total'] = sub_edges['source'] +sub_edges['target']
    sub_edges = sub_edges.merge(membership['topic'], how='left',left_index=True, right_index=True).sort_values('total', ascending=False)

    top.columns = ['Sub A', 'Sub B', 'Topic A', 'Topic B', 'resid']
    top.head(20).round(1).to_csv(tables_path(f"{date}/top_political_residuals.csv"), index=False)

    total = top.shape[0]
    sub_edges['pct'] = (sub_edges['total']/total).round(3)
    sub_edges.index.name='subreddit'
    s = sub_edges[['total']].head(20)
    s.columns = ['n']
    s.to_csv(tables_path(f"{date}/pol_sub_top_edge_counts.csv"), index=False)

    print(f"The top 20 of the {sub_edges.shape[0]} subreddits in the political community account for {sub_edges.head(20)['pct'].sum().round(2)} of edges")

    edges = subset_df(labelled, 'resid', q=0.95)
    edges = edges[(edges['source_topic'] == 'political') & (edges['target_topic'] == 'political')][['source', 'target', 'resid']].sort_values('resid', ascending=False)
    edges.to_csv(tables_path(f"{date}/top_pol_edges.csv"))

def get_double_pol_data():
    """ return df of edges in network between two political subreddits """
    pol_subs = load_pol_subs()
    pol_subs.subreddit=pol_subs.subreddit.str.replace('\\','')
    subreddits = pol_subs.subreddit
    pol_subs = pol_subs.set_index('subreddit')

    data = pd.read_csv("/Users/emg/GitHub/thesis/output/2019_01/1000_residuals_output_utf8.csv")
    labelled = label_subs(data)
    labelled['resid_rank'] = labelled.resid.rank(pct=True)
    top = subset_df(labelled, 'resid', q=0.95)

    double_pol = (top[(top['source'].isin(subreddits)) & (top['target'].isin(subreddits))]
              [['source','target','source_topic','target_topic','resid','resid_rank']].sort_values('resid', ascending=False))
    
    return double_pol

@style
def plot_top_political_edges(pol_edges, pol_subs):
    x = pol_edges[pol_edges['resid']>6.3][['source','target','resid','source_pol','target_pol']]
    x.round(2).to_csv(tables_path(f"{date}/top_20_political_edges.csv"), index=False)

    f = plt.figure(1)
    ax = f.add_subplot(1,1,1)

    G = nx.from_pandas_edgelist(x, edge_attr='resid')
    s = list(G.nodes())
    subset = pol_subs.loc[s]
    for col in subset.columns:
        d = subset[col].to_dict()
        nx.set_node_attributes(G, d, col)

    colors = dict(G.nodes(data='col')).values()
    
    pos = nx.spring_layout(G, k=1.2, weight='resid')
    nx.draw_networkx(G, pos=pos, with_labels=False, node_color=colors, alpha=0.3)
    nx.draw_networkx_labels(G, pos=pos, with_labels=True)
                
    plt.axis('off')
    f.set_facecolor('w')

    f.tight_layout()
    plt.savefig(figures_path(f"{date}/top_pol_edges_graph.png"))
    plt.close()

def non_pol_neighbours_graph():
    """
    creates a network plot of pol subs and the edges they share
    with their 10 closest neighbours
    saves as non_pol_neighbours_graph.png
    """
    data = pd.read_csv("/Users/emg/GitHub/thesis/output/2019_01/1000_residuals_output_utf8.csv", index_col=0)

    labelled = label_subs(data)
    labelled['resid_rank'] = labelled.resid.rank(pct=True)
    top = subset_df(labelled, 'resid', q=0.95)

    edges = top.copy()[['source','target','resid']]
    edges_rev = edges.copy()
    edges_rev.columns = ['target','source','resid']
    directed_edges = pd.concat([edges,edges_rev], sort=True)
    directed_edges['resid_rank'] = directed_edges['resid'].rank(pct=True)

    df = label_subs(directed_edges)

    pol_subs = load_pol_subs()
    pol_names = pol_subs.subreddit.str.replace('\\','')
    pol_subs.subreddit=pol_subs.subreddit.str.replace('\\','')

    pol_neighbours = df[df['source'].isin(pol_names)].sort_values('resid', ascending=False)

    top_pol_neigh = pol_neighbours.groupby('source').head(10).sort_values(['source','resid'], ascending=[True,False])
  
    x = top_pol_neigh[~top_pol_neigh.target.isin(pol_names)][['source','target']]

    col_dict = pol_subs.set_index('subreddit').col.to_dict()
    for sub in x.target.unique():
        col_dict[sub] = 'gray'

    G = nx.from_pandas_edgelist(x)
    nx.set_node_attributes(G, col_dict, 'col')

    f = plt.figure(1)
    ax = f.add_subplot(1,1,1)

    colors = dict(G.nodes(data='col')).values()

    pos = nx.spring_layout(G, k=0.2)
    nx.draw_networkx(G, pos=pos, with_labels=False, node_color=colors, alpha=0.3)
    #nx.draw_networkx_labels(G, pos=pos, with_labels=True)

    plt.axis('off')
    f.set_facecolor('w')
    
    f.tight_layout()
    plt.savefig(figures_path(f"{date}/non_pol_neighbours_graph.png"))
    plt.close()


@style
def plot_pol_edge_counts(double_pol):
    """ save barplot of number of edges each political subreddits shares
    with other politcal subreddits in the network """
    pol_subs = load_pol_subs()
    pol_subs.subreddit=pol_subs.subreddit.str.replace('\\','')
    subreddits = pol_subs.subreddit
    pol_subs = pol_subs.set_index('subreddit')
    
    el = double_pol[['source','target','resid']]
    el_copy = el.copy()
    el_copy.columns = ['target','source','resid']

    pe = pd.concat([el,el_copy])
    pol_subs['num_pol_edges'] = pe.source.value_counts()
    pol_subs = pol_subs.sort_values('num_pol_edges',ascending=True)
    pol_subs['num_pol_edges'].plot(kind='barh',color=pol_subs['col'])

    plt.xticks(np.arange(0, 22, 2.0))
    plt.tight_layout()
    plt.savefig(figures_path(f"{date}/pol_edge_counts.png"))


@style
def pol_subreddits_resids_hist(pol_edges):
    pol_edges.resid.hist()
    plt.xlabel('residual')
    plt.ylabel('frequency')
    plt.tight_layout()
    plt.savefig(figures_path(f"{date}/pol_subreddits_resids_hist"))

    # also get table of highest resids between political subreddit pairs
    (pol_edges[pol_edges['resid']>6.3]
        [['source','target','resid','source_pol','target_pol']]
        .round(2)
        .to_csv(tables_path(f"{date}/top_20_political_edges.csv")
        , index=False))

@style
def plot_cmv_pol_edge_values(top):
    pol_subs = load_pol_subs()
    pol_subs.index=pol_subs.subreddit.str.replace('\\','')
    pol_subs_cols = pol_subs.col.to_dict()
    
    cmv = top[(top['source']=='changemyview')|(top['target']=='changemyview')][['source','target','resid']]
    cmv_rev = cmv.copy()
    cmv_rev.columns = ['target','source','resid']
    output = pd.concat([cmv, cmv_rev], sort=True).sort_values('resid', ascending=True)
    output['resid_rank'] = output.resid.rank(pct=True)
    output = output[output.source=='changemyview']

    cmv_pol = output[output.target.isin(pol_subs.index)].set_index('target')
    cmv_pol['target_col'] = cmv_pol.index.map(lambda x: pol_subs_cols[x])
    cmv_pol.resid_rank.plot(kind='barh', color=cmv_pol['target_col'])
    plt.xlabel('percentile')
    plt.ylabel('political subreddit')
    

    plt.tight_layout()
    plt.savefig(figures_path(f"{date}/cmv_pol_edge_barplot.png"))
    plt.close()

@style
def plot_pol_sub_degree_rank(pol_subs, degrees):
    pol_subs['degree_rank'] = degrees.rank(pct=True).loc[pol_subs.index]
    pol_subs = pol_subs.sort_values('degree_rank', ascending=True)

    pol_subs['degree_rank'].plot(kind='barh',color=pol_subs['col'])
    plt.xlabel('degree percentile')

    plt.tight_layout()
    plt.savefig(figures_path(f"{date}/pol_sub_degree_rank.png"))
    plt.close()
    

def sports_subreddits(date='2019_01', num_subreddits=1000):
    membership = pd.read_csv(output_path(f"{date}/{num_subreddits}_community_membership.csv"), index_col=0)

    sports_communities = (membership[membership['topic']=='sports']
                                .community.value_counts())
    print(sports_communities.shape[0], "communit(ies) include at least one sports subreddit")
    sports_comm = (sports_communities.index[0])


    subset = membership[(membership['topic']=='sports')|(membership['community']==sports_comm)].sort_values(['community', 'topic'], ascending=[True, False])
    subset.index.name = 'subreddit'
    subset.to_csv(tables_path(f'{date}/sports_subreddits.csv'))

""" END OF SUBSETTING """

""" GENERAL """
def run_community_detection(date, num_subreddits=1000):
    create_directories(date)
    data = load_sim_data(date)

    # dropping null edges and renaming subreddits for network analysis
    edges = data[data['author']>0].copy().reset_index()
    edges.columns = ['source', 'target'] + list(edges.columns[2:])

    output = run_residuals(date=date, num_subreddits=num_subreddits, data=edges)
    community_detection(output=output, date=date, num_subreddits=num_subreddits)

def community_plots(**kwargs):
    print("plotting:")
    network_topic_barplot()

    print("raw counts")
    raw_counts(**kwargs)

    output = load_residuals(**kwargs)
    print('author count hist')
    author_counts(**kwargs)
    print('scatterplot')
    scatterplot(output, **kwargs)
    print(' diagnostics')
    diagnostics(output, **kwargs)
    print('community topic heatmaps')
    community_topics(**kwargs)
    print('residiuals hist')
    resid_hist(output, **kwargs)

    labelled = label_subs(output, **kwargs)
    print('political resids hist')
    pol_resids(labelled, **kwargs)

def run(labelled, date, num_subreddits=1000):
    membership = pd.read_csv(output_path(f"{date}/{num_subreddits}_community_membership.csv"), index_col=0)

    local_density(membership=membership)
    small_communities(membership=membership)

    default_communities(membership=membership)

    print()
    print("getting topic breakdown by communities")
    community_topics(membership=membership)

    print("getting political residuals")
    pol_resids(labelled, membership)

    print()
    print("getting sports subreddits table")
    sports_subreddits()

    print("getting aggregate table of community data")
    aggregate_community_data(date=date)





