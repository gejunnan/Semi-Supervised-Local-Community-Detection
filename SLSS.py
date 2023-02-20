import Function
import datetime
from grakel.kernels import ShortestPath
import numpy as np
import multiprocessing
import networkx as nx
import math
import copy
import warnings
import os

warnings.simplefilter("ignore")
np.seterr(divide='ignore', invalid='ignore')


def Alg1(nodei, G):
    C = []
    N = []
    C.append(nodei)
    e_out = G.degree(nodei)
    e_c = 0
    M = e_c / (e_out + 1e-9)
    for node in G.neighbors(nodei):
        N.append(node)
    while True:
        Mmax = -1000
        vBest = 0
        e_cBest = 0
        e_outBest = 0
        for nodev in N:
            dv = G.degree(nodev)
            x = 0
            for node_n in G.neighbors(nodev):
                if node_n in C:
                    x += 1
            y = dv - x
            e_cCurrent = e_c + x
            e_outCurrent = e_out - x + y
            Mnew = round((e_cCurrent / (e_outCurrent + 1e-9)), 4)
            if Mnew > Mmax:
                Mmax = Mnew
                vBest = nodev
                e_cBest = e_cCurrent
                e_outBest = e_outCurrent
        if Mmax >= M:
            M = Mmax
            e_c = e_cBest
            e_out = e_outBest
            C.append(vBest)
            N.remove(vBest)
            for n in G.neighbors(vBest):
                if n not in C:
                    N.append(n)
            N = list(set(N))
        else:
            break
    return C


def Alg5(fileedge, Ck):
    maxsize = max([len(com) for com in Ck])
    sp_graph = Function.com_trans_graph(Ck, fileedge)
    sp = ShortestPath(normalize=True, with_labels=False)
    sp.fit_transform(sp_graph)
    sp2 = np.nan_to_num(sp.transform(sp_graph))
    mean = []
    for i1 in range(len(sp2)):
        for i2 in range(i1 + 1, len(sp2)):
            mean.append(sp2[i1][i2])
    Smed = sorted(mean)[math.ceil(len(mean) / 2)] - 1
    Gsim = nx.Graph()  # 构建一个无向带权图
    for i in range(len(sp2)):
        for j in range(i + 1, len(sp2)):
            if sp2[i][j] > Smed:
                Gsim.add_edge(i, j, weight=sp2[i][j])
    return Gsim, Smed, sp_graph, maxsize


def constructG(C, fileedge, Gsim, Ck, sp_graph, Smed):
    G = Function.networkx(fileedge)
    C_sp_graph = Function.spgraph(C, G)
    sp0 = ShortestPath(normalize=True, with_labels=False)
    sp0.fit_transform(sp_graph)
    sp00 = np.nan_to_num(sp0.transform([C_sp_graph])).tolist()
    Gsmi_c = copy.deepcopy(Gsim)
    for jjj in range(len(sp_graph)):
        if sp00[0][jjj] > Smed:
            Gsmi_c.add_edge(len(sp_graph), jjj, weight=sp00[0][jjj])
    return Gsmi_c, Ck, len(sp_graph)


def Alg6(nodei, G, Ck):
    C = []
    N = []
    node_knowncom = []
    if nodei not in G.nodes:
        for com1 in Ck:
            node_knowncom.append(com1)
    else:
        for node in list(G.neighbors(nodei)):
            N.append(node)
        C.append(nodei)
        edge = G.edges(nodei)
        e_c = 0
        e_out = []
        for i in edge:
            weight = list(G.get_edge_data(list(i)[0], list(i)[1]).values())[0]
            e_out.append(weight)
        M = e_c / (sum(e_out) + 1e-9)
        while True:
            Mmax = -1000
            vbest = 0
            candidate_node = []
            if len(N) == 0:
                break
            for nodev in N:
                candidate_node.append(nodev)
                candidate_com = list(set(C) | {nodev})
                inedge, outedge = Function.cal_inner_outer_C(candidate_com, G)
                com_inedge = []
                for ii in inedge:
                    com_inedge1 = list(G.get_edge_data(list(ii)[0], list(ii)[1]).values())[0]
                    com_inedge.append(com_inedge1)
                com_outedge = []
                for jj in outedge:
                    com_outedge1 = list(G.get_edge_data(list(jj)[0], list(jj)[1]).values())[0]
                    com_outedge.append(com_outedge1)
                Mnew = sum(com_inedge) / (sum(com_outedge) + 1e-9)
                if Mnew > Mmax:
                    Mmax = Mnew
                    vbest = nodev
            if Mmax >= M:
                M = Mmax
                C.append(vbest)
                N.remove(vbest)
                for n in G.neighbors(vbest):
                    if n not in C:
                        N.append(n)
                N = list(set(N))
            else:
                break
        del C[0]
        for iii in C:
            node_knowncom.append(Ck[iii])
    return node_knowncom


def select_first_node(nodei, G):
    C = [nodei]
    N = list(G.neighbors(nodei))
    comm_nei = Function.cal_common_nei(N, G)
    node_index = [i for i, x in enumerate(comm_nei) if x == max(comm_nei)]
    C.append(Function.pick_max_degree(node_index, N, G))
    N = Function.cal_nei(C, G)
    return C, N


def Alg7(C1, N1, local_com, G, global_com, maxsize, weight):
    C = C1
    N = N1
    global_used_similarity = []
    local_used_similarity = []
    local_sp = ShortestPath(normalize=True, with_labels=False)
    local_sp.fit_transform(local_com)
    global_sp = ShortestPath(normalize=True, with_labels=False)
    global_sp.fit_transform(global_com)
    local_subgraph_weight = weight
    while True:
        if len(N) == 0:
            break
        if len(C) >= maxsize:
            break
        Mmax = -100000
        candidate_local_similarity1 = []
        candidate_node = []
        candidate_global_similarity = []
        candidate_modularity = []
        record_global_node = []
        for nodev in N:
            candidate_node.append(nodev)
            com_inedge, com_outedge = Function.calculate_inedges_outedges(list(set(C) | {nodev}), G)
            Mnew = round((com_inedge / (com_outedge + 1e-9)), 4)
            candidate_modularity.append(round((com_inedge / (com_outedge + 1e-9)), 4))
            candidate_com_graph = Function.spgraph(list(set(C) | {nodev}), G)
            candidate_similarity = np.nan_to_num(global_sp.transform([candidate_com_graph]))
            candidate_similarity = candidate_similarity[0].tolist()
            Snew = round((sum(candidate_similarity) / len(candidate_similarity)), 4)
            candidate_global_similarity.append(Snew)
            if (len(global_used_similarity) == 0):
                if Snew >= np.nan_to_num(
                        np.mean(global_used_similarity)) and Mnew >= Mmax:
                    Mmax = Mnew
                    if len(record_global_node) != 0:
                        record_global_node[0] = nodev
                    else:
                        record_global_node.append(nodev)
            elif (len(global_used_similarity) <= 2 and len(global_used_similarity) >= 1):
                if Snew >= min(global_used_similarity) and Mnew >= Mmax:
                    Mmax = Mnew
                    if len(record_global_node) != 0:
                        record_global_node[0] = nodev
                    else:
                        record_global_node.append(nodev)
            else:
                if Snew >= min(global_used_similarity[-3:]) and Mnew >= Mmax:
                    Mmax = Mnew
                    if len(record_global_node) != 0:
                        record_global_node[0] = nodev
                    else:
                        record_global_node.append(nodev)
        if len(record_global_node) != 0:
            node = record_global_node[0]
        else:
            break
        node1 = node
        index_node1 = candidate_node.index(node1)
        if len(candidate_local_similarity1) == 0:
            local_subgraph1 = Function.spgraph(Function.node_sub(node1, G, C), G)
            weight_s1 = np.multiply(np.array(local_subgraph_weight),np.array(np.nan_to_num(local_sp.transform([local_subgraph1]))[0].tolist())).tolist()
            local_used_similarity.append(round((sum(weight_s1) / sum(local_subgraph_weight)), 4))
            global_used_similarity.append(candidate_global_similarity[index_node1])
        else:
            break
        C.append(node1)
        N.remove(node1)
        for n in G.neighbors(node1):
            if n not in C:
                N.append(n)
        N = list(set(N))
    return C, N, local_used_similarity


def Alg8(C1, N1, local_com, G, maxsize, weight, localsmi):
    C = C1
    N = N1
    local_used_similarity = localsmi
    local_sp = ShortestPath(normalize=True, with_labels=False)
    local_sp.fit_transform(local_com)
    local_subgraph_weight = weight
    while True:
        if len(N) == 0:
            break
        if len(C) >= maxsize:
            break
        candidate_local_similarity1 = []
        candidate_node = []
        Smax = -10000
        record_local_node = []
        for nodev in N:
            candidate_node.append(nodev)
            local_subgraph1 = Function.spgraph(Function.node_sub(nodev, G, C), G)
            weight_s1 = np.multiply(np.array(local_subgraph_weight),np.array(np.nan_to_num(local_sp.transform([local_subgraph1]))[0].tolist())).tolist()
            Similarity1 = round((sum(weight_s1) / sum(local_subgraph_weight)), 4)
            candidate_local_similarity1.append(Similarity1)
            if Similarity1 >= np.mean(local_used_similarity) and Similarity1 >= Smax:
                Smax = Similarity1
                if len(record_local_node) != 0:
                    record_local_node[0] = nodev
                else:
                    record_local_node.append(nodev)
        if len(record_local_node) != 0:
            node = record_local_node[0]
        else:
            break
        node1 = node
        local_used_similarity.append(candidate_local_similarity1[candidate_node.index(node1)])
        C.append(node1)
        N.remove(node1)
        for n in G.neighbors(node1):
            if n not in C:
                N.append(n)
        N = list(set(N))
    return C, N


def select_com(C, filename1111, comcom11, G):
    detected_C_graph = Function.spgraph(C, G)
    sp_graph = Function.knowknow(comcom11, filename1111)
    sp = ShortestPath(normalize=True, with_labels=False)
    sp.fit_transform(sp_graph)
    sp00 = sp.transform([detected_C_graph])
    sp00 = np.nan_to_num(sp00)
    sp00 = sp00.tolist()[0]
    comSim = {}
    for iq1 in range(len(sp00)):
        comSim[iq1] = sp00[iq1]
    comSort = sorted(comSim.items(), key=lambda x: x[1], reverse=True)
    coms_id = [list(yy)[0] for yy in comSort[:10]]
    return coms_id


def Go(para):
    i = para[0]
    fileedge, Ck, file, = para[1], para[2], para[3]
    f = open(file + str(i), "w", encoding="utf-8")
    G = Function.networkx(fileedge)
    C11 = Alg1(i, G)
    gw = nx.Graph()
    gw.add_nodes_from(list(G.neighbors(i)))
    gw.add_edges_from(G.subgraph(G.neighbors(i)).edges())
    componentsNodes = max(nx.connected_components(gw), key=len)
    extraNodes = list(componentsNodes.union(C11))
    extraNodes.append(i)
    extraNodes = list(set(extraNodes))
    extrasubG = G.subgraph(extraNodes)
    temp = copy.deepcopy(extraNodes)
    extraNodes.remove(i)
    for ii in extraNodes:
        if extrasubG.degree(ii) == 1 and ii not in list(extrasubG.neighbors(i)):
            temp.remove(ii)
    Gsim, Smed, sp_graph, maxsize = Alg5(fileedge, Ck)
    Gsim_c, Ck, Smed = constructG(temp, fileedge, Gsim, Ck, sp_graph, Smed)
    node_knowncom = Alg6(Smed, Gsim_c, Ck)
    c_graph, n_subgraph, weight = Function.graph_subgraph_weight(node_knowncom, fileedge)
    local_com = Function.com_trans_graph(n_subgraph, fileedge)
    global_com = Function.com_trans_graph(c_graph, fileedge)
    C1, N1 = select_first_node(i, G)
    C2, N2, nodesim = Alg7(C1, N1, local_com, G, global_com, maxsize, weight)
    C, N = Alg8(C2, N2, local_com, G, maxsize, weight, nodesim)
    f.writelines(str(C) + "\n")
    f.close()


if __name__ == '__main__':
    seed = Function.big_dataset_seed()
    known_com = ['dataset/amazon_knowcom', 'dataset/dblp_knowcom', 'dataset/facebook_knowcom','dataset/twitter_knowcom', 'dataset/lj_knowcom']
    file_edge = ['amazon-1.90.ungraph.txt', 'dblp-1.90.ungraph.txt', 'facebook-1.90.ungraph.txt', 'twitter-1.90.ungraph.txt', 'lj-1.90.ungraph.txt']
    path = ['amazon', 'dblp', 'facebook', 'twitter', 'lj']
    datasets = list(zip(seed, known_com, file_edge, path))
    for filename in datasets:
        print("数据集：", filename[3])
        dataset_knowcom = Function.get_knowcom(filename[1])
        for i in range(1, 6):
            print("第", i, "组已知社区")
            start = datetime.datetime.now()
            Ck = dataset_knowcom[i - 1]
            fileedge = "dataset/" + filename[2]
            nodelist = filename[0]
            savefile = "Results/" + filename[3] + "/" + str(i) + "/"
            if os.path.isdir(os.path.dirname(savefile)) == False:
                os.makedirs(os.path.dirname(savefile))
            plist = [[node, fileedge, Ck, savefile] for node in
                     nodelist]
            pool = multiprocessing.Pool(processes=1)
            pool.map(Go, plist)
            pool.close()
            pool.join()
            end = datetime.datetime.now()
            print("节点个数：", len(nodelist))
            print("耗时：{}".format(end - start))
