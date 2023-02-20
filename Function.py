import numpy as np
import networkx as nx
from grakel import Graph
from grakel.kernels import ShortestPath

def networkx(filename):
    """--------------------------------------------------------------------------------
                 function:       把一个含有边的txt数据集表示成networkx
                 Parameters:     filename：文件名称 .txt格式
                 Returns：       G：表示成networkx的图
                ---------------------------------------------------------------------------------"""
    fin = open(filename, 'r')
    G = nx.Graph()
    for line in fin:
        data = line.split()
        if data[0] != '#':
            G.add_edge(int(data[0]), int(data[1]))
    return G

def calculate_inedges_outedges(C, G):
    """--------------------------------------------------------------------------------
                 function:       计算社区C的内部边数和外部边数
                 Parameters:     C：社区 G：网络图
                 Returns：       inedges:社区的内部边数   outedges：社区的外部边数
                 ---------------------------------------------------------------------------------"""
    inedges = len(G.subgraph(C).edges())
    node_nei = []
    for i in C:
        node_nei.extend(list(G.neighbors(i)))
    N = list(set(node_nei) - set(C))
    outedges = sum([len(list(set(list(G.neighbors(node))).intersection(set(C)))) for node in N])
    return inedges, outedges

def cal_nei(C,G):
    """--------------------------------------------------------------------------------
                   function:       计算社区C的邻居
                   Parameters:     C：社区 G：网络图
                   Returns：       社区C的邻居N
                   ---------------------------------------------------------------------------------"""
    C_nei = []
    for i in C:
        C_nei.extend(list(G.neighbors(i)))
    N = list(set(C_nei) - set(C))
    return N

def big_dataset_seed():
    """--------------------------------------------------------------------------------
                      function:       获取5个数据集的种子节点
                      Returns：       数据集的种子节点
                      ---------------------------------------------------------------------------------"""
    seed=[]
    with open("dataset/seed",'r') as f:
        for line in f.readlines():
            line=line.strip().split()
            line=list(map(int,line))
            seed.append(line)
    return seed

def com_trans_graph(knowcom, file_edge):
    """--------------------------------------------------------------------------------
                 function:       将一组已知社区转换为最短路径形式表示的图
                 Parameters:     knowcom：给定的已知社区
                                 file_edge:网络图
                 Returns：       shortest_graph: 已知社区的最短路径图
                 ---------------------------------------------------------------------------------"""
    G= networkx(file_edge)
    shortest_graph = []
    for com in knowcom:
        edges = G.subgraph(com).edges()
        G1 = nx.Graph()
        G1.add_edges_from(edges)
        adj = np.array(nx.adjacency_matrix(G1).todense())
        shortest_graph.append(Graph(adj))
    return shortest_graph

def spgraph(C, G):
    """--------------------------------------------------------------------------------
                 function:       将一个社区转换成最短路径图
                 Parameters:    C：社区 G: 网络图
                 Returns：      com_shortest_graph:一个社区的最短路径图
                 ---------------------------------------------------------------------------------"""
    g1 = nx.Graph()
    edges = G.subgraph(C).edges()
    g1.add_edges_from(edges)
    com_shortest_graph = Graph(np.array(nx.adjacency_matrix(g1).todense()))
    return com_shortest_graph

def cal_common_nei(N, G):
    """--------------------------------------------------------------------------------
                     function:      计算N中节点的公共邻居个数
                     Parameters:     N：社区的邻居节点 G: 网络图
                     Returns：       nei_length: 公共邻居个数
                     ---------------------------------------------------------------------------------"""
    nei_length = []
    for node in N:
        second_nei = list(G.neighbors(node))
        nei_length.append(len(set(N) & set(second_nei)))
    return nei_length

def pick_max_degree(node_index, N, G):
    """--------------------------------------------------------------------------------
                     function:      公共邻居个数一样，挑选节点度大的节点
                     Parameters:     node_index：节点索引N： 邻居节点 G: 网络图
                     Returns：     max_degree_node: 度大的节点
                     ---------------------------------------------------------------------------------"""
    node= [N[m] for m in node_index]
    node_degree = [G.degree(n) for n in node]
    max_length_index = node_degree.index(max(node_degree))
    max_degree_node = node[max_length_index]
    return max_degree_node

def cal_sub_com(G, com):
    """--------------------------------------------------------------------------------
                    function:       提取已知社区的节点子图
                    Parameters:     com：给定的已知社区  G:网络图
                    Returns：       node_subgraph: 已知社区对应的节点子图
                    ---------------------------------------------------------------------------------"""
    com_subgraph=[]
    for com1 in com:
        for node in com1:
            dd = list(set(list(G.neighbors(node))) & set(com1))
            dd.append(node)
            com_subgraph.append(sorted(dd))
    node_subgraph = []
    for com_subgraph1 in com_subgraph:
        if len(com_subgraph1) >= 2 and com_subgraph1 not in node_subgraph:
            node_subgraph.append(com_subgraph1)
    return node_subgraph

def cal_weight(G, node_subgraph):
    """--------------------------------------------------------------------------------
                    function:       计算节点子图间的相似性以及节点子图对应的权重
                    Parameters:     node_subgraph：节点子图  G:网络图
                    Returns：       subgraph: 子图
                                    weight： 权重
                    ---------------------------------------------------------------------------------"""
    node_subgraph_simi1=[]
    for node_subgraph1 in node_subgraph:
        shortestpath_graph1=spgraph(node_subgraph1, G)
        sp = ShortestPath(normalize=True, with_labels=False)
        sp.fit_transform([shortestpath_graph1])
        node_subgraph_simi2 = []
        for node_subgraph2 in node_subgraph:
            shortestpath_graph2 = spgraph(node_subgraph2, G)
            node_subgraph_simi2.append(sp.transform([shortestpath_graph2])[0])
        node_subgraph_simi1.append(node_subgraph_simi2)
    for i in node_subgraph_simi1:
        for j in range(len(i)):
            if i[j] != 1.0:
                i[j] = 0
    simi_value = [node_subgraph_simi1.index(subgraph_simi) for subgraph_simi in node_subgraph_simi1]
    simi_value_dict = {}
    for value in [node_subgraph_simi1.index(subgraph_simi) for subgraph_simi in node_subgraph_simi1]:
        simi_value_dict[value] = simi_value.count(value)
    key = simi_value_dict.keys()
    weight = list(simi_value_dict.values())
    subgraph = []
    for key1 in key:
        subgraph.append(node_subgraph[key1])
    return subgraph,weight

def cal_inner_outer_C(C, G):
    """--------------------------------------------------------------------------------
                    function:       在无向权重图计算社区C的内部边和外部边
                    Parameters:     C：社区  G：网络无向权重图
                    Returns：       inedges:社区的内部边
                                    outedges：社区的外部边
                    ---------------------------------------------------------------------------------"""
    inedges = G.subgraph(C).edges()
    node_nei = []
    for i in C:
        node_nei.extend(list(G.neighbors(i)))
    N = list(set(node_nei) - set(C))
    a = []
    for ii in N:
        a.extend(G.edges(ii))
    outedges = []
    for iii in a:
        if (list(iii)[1] in C):
            outedges.append(iii)
    return inedges, outedges

def graph_subgraph_weight(com, file2):
    """--------------------------------------------------------------------------------
                    function:       计算已知社区的社区子图，节点子图，以及节点子图的权重
                    Parameters:     Com：社区   file2：网络图
                    Returns：       com:社区的内部边数   subgraph：社区的外部边数    weight；权重
                    ---------------------------------------------------------------------------------"""
    G = networkx(file2)
    subgraph, weight = cal_weight(G, cal_sub_com(G, com))
    return com, subgraph, weight

def get_knowcom(file):
    """--------------------------------------------------------------------------------
                       function:       获得数据集的五组已知社区
                       Parameters:     file：已知社区文件
                       Returns：       knowcom:五组已知社区
                       ---------------------------------------------------------------------------------"""
    com=[]
    with open(file,'r') as f:
        for line in f.readlines():
            line=line.strip().split()
            line=list(map(int,line))
            com.append(line)
    knowcom=[com[:10],com[10:20],com[20:30],com[30:40],com[40:50]]
    return knowcom

def node_sub(nodev,G,C):
    """--------------------------------------------------------------------------------
                          function:       节点相似性阶段，获取节点子图对应的节点
                          Parameters:      nodev:节点 G：网络图 C社区
                          Returns：       节点子图对应的节点
                          ---------------------------------------------------------------------------------"""
    first_node = [nodev]
    pick_second_node = list(set(list(G.neighbors(nodev))) & set(C))
    common_nei_length = cal_common_nei(pick_second_node, G)
    second_node = pick_second_node[common_nei_length.index(max(common_nei_length))]
    third_nodes = list(set(list(G.neighbors(second_node))) & set(C))
    local_com = list(set(first_node).union(third_nodes).union([second_node]))
    return local_com

