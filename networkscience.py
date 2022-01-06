import sys
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from scipy import sparse
from scipy import stats

def display_top_n(df, n, label):
    """Display Nodes sorted by $label, with the $n highest values
    
        Args:
            df (DataFrame): place where information is contained
            n (int): number of rows to display
            label (str): identifier of a column of the df, rows 
                will be sorted for this value

        Return:
            display the table
    """
    print('Top', n, 'for', label)
    pd.set_option('display.max_rows', n)
    display(df.sort_values(label, ascending=False)[['Nodes', label]].head(n))
    

def plot_probability_loglog(ax, x, y, title, cumulative=False):
    """Scatter plot in a loglog scale with predefined parameters

    Args:
        ax (Axes): axes where plot
        x (Array): x-value to plot
        v (Array): y-value to plot
        title (str): title of the plot
        cumulative (bool): if true y axis has lable P_k, otherwise p_k

    Return:
        draws a plot in the given axes
    """
    ax.loglog(x, y, 'o', markersize = 4)
    ax.grid(which='both', linestyle='--', linewidth=0.5)
    ax.set_title(title, size = 22)
    ax.set_xlabel("k", size = 20)
    if cumulative:
        ax.set_ylabel("P_k", size = 20)
    else:
        ax.set_ylabel("p_k", size = 20)
    ax.tick_params(labelsize=18)
    ax.tick_params(labelsize=18)


def degree_disribution(adj, nodes, print_graph=False):
    """Plot the degree distribution of nodes, for a given network.

    Args:
        adj (sparse matrix): the adjacency matrix of the network
        nodes (DataFrame): a DataFrame containing the nodes, the 
            order in the serie must be the same of the adjacency matrix one.
        print_graph (bool): if true plot the distribution of nodes vs 
            the degree of the nodes in a loglog scale

    Returns:
        a DataFrame containing nodes, their in and out degree
    """

    N = len(nodes.index)
    df = nodes.copy()
    d_in = adj.dot(np.ones(N))
    d_out = (adj.T).dot(np.ones(N))
    d_in = d_in.astype(int)
    d_out = d_out.astype(int)

    df['in degree'] = d_in
    df['out degree'] = d_out

    if print_graph:

        d_in = d_in[d_in > 0]
        d_out = d_out[d_out > 0]

        ############# in #############
        sorted_d = sorted(d_in)
        occurrence = Counter(sorted_d)
        x_in = list(occurrence.keys())
        y_in = list(occurrence.values())
        y_in = y_in/np.sum(y_in)

        Pk_in = 1 - np.cumsum(y_in) # complementary cumulative

        # set the last value of Pk (that is equal to 0 
        # and generates problems when plotting in the log-scale) 
        # equal to 1 and sort Pk in decreasing order to put 
        # the 1 at the beginning of the array
        Pk_in[-1] = 1 
        Pk_in = sorted(Pk_in, reverse = True)


        ############# out #############
        sorted_d = sorted(d_out)
        occurrence = Counter(sorted_d)
        x_out = list(occurrence.keys())
        y_out = list(occurrence.values())
        y_out = y_out/np.sum(y_out)

        Pk_out = 1 - np.cumsum(y_out) # complementary cumulative

        # set the last value of Pk (that is equal to 0 
        # and generates problems when plotting in the log-scale) 
        # equal to 1 and sort Pk in decreasing order to put 
        # the 1 at the beginning of the array
        Pk_out[-1] = 1 
        Pk_out = sorted(Pk_out, reverse = True)

        # Plotting    
        fig, ax = plt.subplots(2, 2, figsize = (30, 20))
        plot_probability_loglog(ax[0,0], x_in, y_in, "IN-Degree Distribution")
        plot_probability_loglog(ax[0,1], x_in, Pk_in, "IN-CCDF", True)
        plot_probability_loglog(ax[1,0], x_out, y_out, "OUT-Degree Distribution")
        plot_probability_loglog(ax[1,1], x_out, Pk_out, "OUT-CCDF", True)
        plt.show()
    
    return df
    #hh, aa = nx.algorithms.link_analysis.hits_alg.hits(nx.DiGraph(adj_matrix_crs.T), tol = 1e-4/len(nx.DiGraph(adj_matrix_crs.T)))

def find_components(adj, nodes):
    """Divide nodes depending on the component they belong

    Args:
        adj (sparse matrix): the adjacency matrix of the network
        nodes (DataFrame): a DataFrame containing the nodes, the 
            order in the serie must be the same of the adjacency matrix one.
    Returns:
        a DataFrame containing nodes and their component
    """
    n = nodes.copy()
    out = sparse.csgraph.connected_components(adj)
    n['component'] = out[1]
    return n

def fraction_in_giant(adj):
    """Tanto per curiosità, mi vedi?"""
    out = sparse.csgraph.connected_components(adj)
    unique, counts = np.unique(out[1], return_counts=True)
    unique_df = pd.DataFrame({'u': unique, 'counts':counts})
    giant_size = unique_df['counts'].max()
    return giant_size/np.sum(unique_df['counts'])

def keep_giant(node_component, whole_df):
    """Create a new adjacency matrix and a edge dataframe with only nodes of the giant component

    Args:
        node_component (DataFrame): dataframe containing node and component, as output of find_components
        whole_df (DataFrame): a DataFrame containing the edges of the whole graph. Columns must be: 'source', 'target' and 'weight'
    Returns:
        the adjacency matrix of the giant component and two DataFrame containing edges and nodes of the giant component
    """
    unique, counts = np.unique(node_component['component'], return_counts=True)
    unique_df = pd.DataFrame({'u': unique, 'counts':counts})
    giant = unique_df['counts'].argmax()
    
    mask = node_component['component'] != giant
    
    node_component.loc[mask, 'component'] = np.nan
    giant_edges = whole_df.copy()
    node_component.rename({'Nodes': 'source'},axis=1, inplace=True)
    giant_edges = pd.merge(giant_edges, node_component, on="source")
    node_component.rename({'source': 'target', 'component':'comp'},axis=1, inplace=True)
    giant_edges = pd.merge(giant_edges, node_component, on="target")
    giant_edges.dropna(inplace=True)
    
    giant_nodes = (node_component.dropna()).copy()
    giant_nodes.rename({'target': 'Nodes'},axis=1, inplace=True)
    giant_nodes['NodeID'] = np.arange(len(giant_nodes.index))
    giant_nodes.drop('comp', axis=1, inplace=True)
    giant_nodes.reset_index(drop=True, inplace=True)

    giant_edges =giant_edges[['source', 'target', 'weight']].copy()
    giant_nodes.rename({'Nodes': 'source', 'NodeID':'sourceID'},axis=1, inplace=True)
    giant_edges = pd.merge(giant_edges, giant_nodes, on="source")
    giant_nodes.rename({'source': 'target', 'sourceID':'targetID'},axis=1, inplace=True)
    giant_edges = pd.merge(giant_edges, giant_nodes, on="target")
    giant_nodes.rename({'target': 'Nodes', 'targetID':'NodeID'},axis=1, inplace=True)
    giant_serie = giant_edges.groupby(['sourceID', 'targetID']).sum()
    row = np.array(giant_serie.index.get_level_values(1).tolist())
    col = np.array(giant_serie.index.get_level_values(0).tolist())
    val = giant_serie.values
    new_adj = sparse.csr_matrix((val.flatten(), (row, col)), shape=(len(giant_nodes.index), len(giant_nodes.index)))
    return new_adj, giant_edges, giant_nodes

def hits_alg(adj, nodes, score_df=None, print_graph=False):
    """Calculate hits scores for hub and authorities

    Args:
        adj (sparse matrix): the adjacency matrix of the network
        nodes (DataFrame): a DataFrame containing the nodes, the 
            order in the serie must be the same of the adjacency matrix one.
        score_df (DataFrame): a DataFrame containing the nodes, their 
            in and out degree, if print_graph is true it can't be None
        print_graph (bool): if true plot hub score vs out degree and 
            autority score vs in degree in a scatter plot

    Returns:
        a DataFrame containing nodes, their hub and autority score
    """
    hh, aa = nx.algorithms.link_analysis.hits_alg.hits(nx.DiGraph(adj.T), tol = 1e-4/len(nx.DiGraph(adj.T)))
    df_hh = pd.DataFrame.from_dict(hh, orient='index')
    df_hh.rename({0:'hits hub'},axis = 1, inplace=True)
    df_aa = pd.DataFrame.from_dict(aa, orient='index')
    df_aa.rename({0:'hits autority'},axis = 1, inplace=True)
    df = pd.concat([nodes, df_hh, df_aa], axis=1)
    if (print_graph):
        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,5))
        temp = pd.merge(score_df, df, on='Nodes')
        temp.plot('in degree', 'hits autority', kind='scatter', ax=ax1)
        temp.plot('out degree', 'hits hub', kind='scatter', ax=ax2)
        plt.tight_layout()
        plt.show()
    return df

def pagerank_alg(adj, nodes, score_df=None, print_graph=False):
    """Calculate PageRank scores for hub and authorities

    Args:
        adj (sparse matrix): the adjacency matrix of the network
        nodes (DataFrame): a DataFrame containing the nodes, the order 
            in the serie must be the same of the adjacency matrix one.
        score_df (DataFrame): a DataFrame containing the nodes, their 
            in and out degree, if print_graph is true it can't be None
        print_graph (bool): if true plot hub score vs out degree and 
            autority score vs in degree in a scatter plot

    Returns:
        a DataFrame containing nodes, their hub and autority score
    """
    aa = nx.algorithms.link_analysis.pagerank_alg.pagerank(nx.DiGraph(adj.T), tol = (1e-4)/len(nx.DiGraph(adj.T)))
    hh = nx.algorithms.link_analysis.pagerank_alg.pagerank(nx.DiGraph(adj), tol = (1e-4)/len(nx.DiGraph(adj.T)))
    df_hh = pd.DataFrame.from_dict(hh, orient='index')
    df_hh.rename({0:'PageRank hub'},axis = 1, inplace=True)
    df_aa = pd.DataFrame.from_dict(aa, orient='index')
    df_aa.rename({0:'PageRank autority'},axis = 1, inplace=True)
    df = pd.concat([nodes, df_hh, df_aa], axis=1)
    if (print_graph):
        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,5))
        temp = pd.merge(score_df, df, on='Nodes')
        temp.plot('in degree', 'PageRank autority', kind='scatter', ax=ax1)
        temp.plot('out degree', 'PageRank hub', kind='scatter', ax=ax2)
        plt.tight_layout()
        plt.show()
    return df

def assortativity_calc(edges, adj, nodes, print_graph=False):
    """Calculate the assortativity of a graph
    
    Args:
        edges (DataFrame): dataframe with columns called  'source' and 
            'target' representing the edges in the network
        adj (sparse matrix): the adjacency matrix of the network
        nodes (DataFrame): a DataFrame containing the nodes, the order 
            in the serie must be the same of the adjacency matrix one.
        print_graph (bool): if true plot assortativity plots.

    Returns:
        tuple containing the slopes of assortativity plots.
    """
    edges.drop_duplicates(inplace=True)
    n = degree_disribution(adj, nodes)
    n.rename({'Nodes': 'source','in degree': 'source in degree', 'out degree': 'source out degree'},axis=1, inplace=True)
    cross_df = pd.merge(edges, n, on="source")
    n.rename({'source': 'target','source in degree': 'target in degree', 'source out degree': 'target out degree'},axis=1, inplace=True)
    cross_df = pd.merge(cross_df, n, on="target")
    in_neigh = cross_df[['source', 'target in degree', 'target out degree']].groupby('source', as_index=False).mean()
    out_neigh = cross_df[['target', 'source in degree', 'source out degree']].groupby('target', as_index=False).mean()
    in_neigh.rename({'source':'Nodes', 'target in degree': 'Average target in degree', 'target out degree': 'Average target out degree'},axis=1, inplace=True)
    out_neigh.rename({'target':'Nodes', 'source in degree': 'Average source in degree', 'source out degree': 'Average source out degree'},axis=1, inplace=True)
    n.rename({'target': 'Nodes','target in degree': 'in degree', 'target out degree': 'out degree'},axis=1, inplace=True)
    cross_df = pd.merge(n, in_neigh, on="Nodes", how='left')
    cross_df = pd.merge(cross_df, out_neigh, on="Nodes", how='left')
    cross_df.fillna(0, inplace=True)
    x = ['out degree','out degree','in degree','in degree']
    y = ['Average target in degree', 'Average target out degree', 'Average source in degree', 'Average source out degree']
    if print_graph:
        fig, ax = plt.subplots(2,2,figsize=(18,15))

    def calc_mu(cross_df, x, y, print_graph=False, ax=None):
        means = cross_df[[x,y]].groupby(x, as_index=False).mean()
        means[means==0] = np.nan
        means.dropna(inplace=True)
        means['log '+x] = np.log10(means[x])
        means['log '+y] = np.log10(means[y])
        interpolation = stats.linregress(means['log '+x], means['log '+y])


        inter_x = np.logspace(0, max(means['log '+x]),1000)
        inter_y = inter_x**(interpolation.slope)*10**(interpolation.intercept)

        ax.loglog(cross_df[x], cross_df[y], 'o', markersize=3)
        ax.loglog(means[x], means[y], 'o', markersize=3)
        ax.plot(inter_x,inter_y,linewidth=3)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        return interpolation.slope
        
    ax = ax.reshape(4)
    mu = ()
    for e_x, e_y, e_ax in zip(x, y, ax):
        e_mu = calc_mu(cross_df, e_x, e_y, print_graph, e_ax)
        mu += (e_mu, )
    if print_graph:
        plt.show()
    return mu

# Stupid funny thing
def progressBar(current, total, barLength = 20):
    """I was waiting for too long"""

    percent = float(current) * 100 / total
    arrow   = '-' * int(percent/100 * barLength - 1) + '>'
    spaces  = ' ' * (barLength - len(arrow))
    sys.stdout.write('\rProgress: [%s%s] %d %%' % (arrow, spaces, percent))

def visualize_adj(adj):
    fig, ax = plt.subplots(2,2,figsize=(12,10))
    a_adj = adj.toarray()
    sns.heatmap(a_adj, cmap="Blues", ax=ax[0,0])
    a_adj[a_adj>1000] = 1000
    sns.heatmap(a_adj, cmap="Blues", ax=ax[0,1])
    a_adj[a_adj>100] = 100
    sns.heatmap(a_adj, cmap="Blues", ax=ax[1,0])
    a_adj[a_adj>10] = 10
    sns.heatmap(a_adj, cmap="Blues", ax=ax[1,1])
    plt.tight_layout()
    plt.show()