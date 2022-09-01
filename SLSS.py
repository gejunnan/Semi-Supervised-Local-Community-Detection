import Function
import datetime
from grakel.kernels import ShortestPath
import numpy as np
import multiprocessing
import networkx as nx
import copy
import warnings
import os
warnings.simplefilter("ignore")
np.seterr(divide='ignore', invalid='ignore')

def Alg1(nodei, G):
    "M方法获得种子节点nodei所在的社区"
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

def Alter_Alg1(nodei, G):
    "从给定的已知社区中挑选节点i所对应的已知社区"
    C = []
    N = []
    for node in list(G.neighbors(nodei)):
        N.append(node)
    C.append(nodei)
    edge = G.edges(nodei)
    e_c = 0
    e_out = []
    for i in edge:
        weight = list(G.get_edge_data(list(i)[0], list(i)[1]).values())[0]
        e_out.append(weight)
    M = e_c / sum(e_out)
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
    return C,M

def Ck_Simi(nv0,Ckv,fileedge):
    sp_graph = Function.com_trans_graph(Ckv, fileedge)
    sp = ShortestPath(normalize=True, with_labels=False)
    sp.fit_transform(sp_graph)
    similarity = sp.transform(sp_graph)
    simi= np.nan_to_num(similarity)
    edge_sim={}
    for i in range(len(simi)):
        for j in range(i+1,len(simi)):
            edge_sim[(i,j)]=simi[i][j]
    know_sim=sorted(edge_sim.items(),key=lambda x:x[1],reverse=True)
    modu = []
    comid=[]
    n=0
    g1=nx.Graph()
    while n<len(know_sim):
        g1.add_edge(list(know_sim[n][0])[0], list(know_sim[n][0])[1], weight=know_sim[n][1])
        if nv0 in g1.nodes and n>=(len(simi)-1):
            C,M =Alter_Alg1(nv0,g1)
            if M>1000:
                if (len(C))==(len(simi)-1):
                    modu.append(-1)
                    comid.append(C)
                else:
                    modu.append(1000)
                    comid.append(C)
                break
            else:
                modu.append(M)
                comid.append(C)
        n+=1
    index = modu.index(max(modu))
    return comid[index]


def Alg5(i,G,Ck,filename):
    "为种子节点i从已知社区中挑选已知社区"
    C = Alg1(i, G)
    gw = nx.Graph()
    gw.add_nodes_from(list(G.neighbors(i)))
    gw.add_edges_from(G.subgraph(G.neighbors(i)).edges())
    componentsNodes = max(nx.connected_components(gw), key=len)
    extraNodes = list(componentsNodes.union(C))
    extraNodes.append(i)
    extraNodes = list(set(extraNodes))
    extrasubG = G.subgraph(extraNodes)
    temp = copy.deepcopy(extraNodes)
    extraNodes.remove(i)
    for ii in extraNodes:
        if extrasubG.degree(ii) == 1 and ii not in list(extrasubG.neighbors(i)):
            temp.remove(ii)
    com10 = copy.deepcopy(Ck)
    nv0 = len(com10)
    com10.append(temp)
    comid = Ck_Simi(nv0, com10, filename)
    com1010 = []
    for uy in comid:
        com1010.append(com10[uy])
    return com1010


def select_first_node(nodei, G):
    "SLSS方法，挑选公共邻居数且度大的节点"
    C = [nodei]
    N = list(G.neighbors(nodei))
    comm_nei = Function.cal_common_nei(N, G)
    node_index = [i for i, x in enumerate(comm_nei) if x == max(comm_nei)]
    C.append(Function.pick_max_degree(node_index, N, G))
    N = Function.cal_nei(C, G)
    return C, N

def Alg6(C1,N1,local_com,G,global_com,maxsize,quanzhong):
    "社区相似性阶段，当候选社区相似性都大于社区历史相似性的均值，选择模块度大的"
    C=C1
    N=N1
    global_used_similarity=[]#存储全局历史相似性值
    local_used_similarity=[]#存储局部历史相似性值
    local_sp = ShortestPath(normalize=True, with_labels=False)  # 计算节点相似性的最短路径核表示
    local_sp.fit_transform(local_com)
    global_sp = ShortestPath(normalize=True, with_labels=False)  # 计算社区相似性的最短路径核表示
    global_sp.fit_transform(global_com)
    local_subgraph_weight = quanzhong
    while True:
        if len(N)==0:
            break
        if len(C)>=maxsize:#判断寻找的社区是否超过最大社区尺寸
            break
        Mmax=-1000000
        candidate_local_similarity1=[]#存储候选社区的节点相似性
        candidate_node=[]#存储候选节点
        candidate_global_similarity=[]#存储候选社区的社区相似性
        candidate_modularity=[]#存储候选社区的模块度
        record_global_node=[]#社区相似性阶段，记录从N中选中的节点
        for nodev in N:
            candidate_node.append(nodev)
            candidate_com = list(set(C) | {nodev})
            com_inedge,com_outedge=Function.calculate_inedges_outedges(candidate_com, G)#计算候选社区的内边和外边
            Mnew=round((com_inedge/(com_outedge+1e-9)),4)#计算候选社区的模块度
            candidate_modularity.append(Mnew)
            candidate_com_graph=Function.spgraph(candidate_com, G)
            candidate_similarity = np.nan_to_num(global_sp.transform([candidate_com_graph]))
            candidate_similarity = candidate_similarity[0].tolist()
            Snew = round((sum(candidate_similarity) / len(candidate_similarity)),4)#候选社区和与已知社区相似性取均值来作为候选社区的相似性
            candidate_global_similarity.append(Snew)
            #当候选社区相似性都大于平均社区历史相似性，选择候选社区模块度最大的节点，来作为候选节点，来往社区C里面添加节点
            if (len(global_used_similarity) == 0):
                if Snew >= np.nan_to_num(
                        np.mean(global_used_similarity)) and Mnew >= Mmax:  # 一开始社区历史相似性列表为空，对空列表求均值的时候会出现nan np.nan_to_num(nan)含义为0
                    Mmax = Mnew
                    if len(record_global_node) != 0:
                        record_global_node[0] = nodev
                    else:
                        record_global_node.append(nodev)
            elif (len(global_used_similarity) <= 2 and len(global_used_similarity) >= 1):
                if Snew >= min(global_used_similarity) and Mnew >= Mmax:  # 一开始社区历史相似性列表为空，对空列表求均值的时候会出现nan
                    # np.nan_to_num(nan)含义为0
                    Mmax = Mnew
                    if len(record_global_node) != 0:
                        record_global_node[0] = nodev
                    else:
                        record_global_node.append(nodev)
            else:
                if Snew >= min(global_used_similarity[-3:]) and Mnew >= Mmax:  # 一开始社区历史相似性列表为空，对空列表求均值的时候会出现nan
                    # np.nan_to_num(nan)含义为0
                    Mmax = Mnew
                    if len(record_global_node) != 0:
                        record_global_node[0] = nodev
                    else:
                        record_global_node.append(nodev)
        if  len(record_global_node)!=0:
            node=record_global_node[0]
        else:
            break
        node1=node
        index_node1=candidate_node.index(node1)
        #在社区相似性阶段，计算完了候选节点的相似性之后，还要计算相应的节点相似性。
        if len(candidate_local_similarity1)==0:
            # 节点node1子图是由三部分组成，nodev，nodev对应的最大公共邻居数secondnode，以及N（secondnode）∩C
            local_com1=[]
            first_node = [node1]
            pick_second_node = list(set(list(G.neighbors(node1))) & set(C))
            common_nei_length = Function.cal_common_nei(pick_second_node, G)
            second_node = pick_second_node[common_nei_length.index(max(common_nei_length))]
            third_nodes = list(set(list(G.neighbors(second_node))) & set(C))
            local_com1=list(set(first_node).union(third_nodes).union([second_node]))
            local_subgraph1 = Function.spgraph(local_com1, G)
            weight_s1 = np.multiply(np.array(local_subgraph_weight),np.array(np.nan_to_num(local_sp.transform([local_subgraph1]))[0].tolist())).tolist()
            Similarity1 = round((sum(weight_s1) / sum(local_subgraph_weight)), 4)
            local_used_similarity.append(Similarity1)
            global_used_similarity.append(candidate_global_similarity[index_node1])
        else:
           break
        C.append(node1)
        N.remove(node1)
        for n in G.neighbors(node1):
            if n not in C:
                N.append(n)
        N = list(set(N))
    return C,N,local_used_similarity

def Alg7(C1,N1,local_com,G,maxsize,quanzhong,localsmi):
    "节点相似性阶段，节点相似性大于节点历史相似性的时候，选择最大节点相似性对应的节点"
    C=C1
    N=N1
    local_used_similarity=localsmi#记录往社区C每次添加节点所对应的每个节点的节点相似性
    local_sp = ShortestPath(normalize=True, with_labels=False)  # 存储候选社区的节点相似性
    local_sp.fit_transform(local_com)
    local_subgraph_weight = quanzhong  # 已知社区中节点子图对应的权重
    while True:
        if len(N)==0:
            break
        if len(C)>=maxsize:#判断找到的社区是否超过最大社区尺寸
            break
        candidate_local_similarity1=[]
        candidate_node=[]
        Smax=-10000
        record_local_node = []  # 记录节点相似性阶段从N中选择的节点
        for nodev in N:
            candidate_node.append(nodev)
            local_com1 = []
            first_node = [nodev]
            pick_second_node = list(set(list(G.neighbors(nodev))) & set(C))
            common_nei_length = Function.cal_common_nei(pick_second_node, G)
            second_node = pick_second_node[common_nei_length.index(max(common_nei_length))]
            third_nodes = list(set(list(G.neighbors(second_node))) & set(C))
            local_com1.extend(first_node)
            local_com1.append(second_node)
            local_com1.extend(third_nodes)
            local_com1 = list(set(local_com1))
            #节点nodev子图是由三部分组成，nodev，nodev对应的最大公共邻居数secondnode，以及N（secondnode）∩C
            local_subgraph1 = Function.spgraph(local_com1, G)#将候选社区的节点子图转换为最短路径形式表示的图
            weight_s1 = np.multiply(np.array(local_subgraph_weight),np.array(local_sp.transform([local_subgraph1])[0].tolist())).tolist()#计算候选社区的节点子图和已知社区节点子图的相似性，并且乘节点子图对应的权重
            Similarity1 = round((sum(weight_s1) / sum(local_subgraph_weight)), 4)#计算候选节点子图对应的平均节点相似性
            candidate_local_similarity1.append(Similarity1)
            # 当候选节点子图对应的节点相似性都大于平均历史节点相似性的时候，选择候选节点相似性值最大的节点，作为往社区C添加的节点
            if Similarity1 >= np.mean(local_used_similarity) and Similarity1>=Smax:
                Smax=Similarity1
                if len(record_local_node) != 0:
                    record_local_node[0] = nodev
                else:
                    record_local_node.append(nodev)
        if len(record_local_node) != 0:
            node = record_local_node[0]
        else:
            break
        node1 = node
        index_node1=candidate_node.index(node1)
        local_used_similarity.append(candidate_local_similarity1[index_node1])
        C.append(node1)
        N.remove(node1)
        for n in G.neighbors(node1):
            if n not in C:
                N.append(n)
        N = list(set(N))
    return C,N


def Go(para):
    i = para[0]
    fileedge, Ck, file, maxsize = para[1], para[2], para[3], para[4]
    f = open(file + str(i), "w", encoding="utf-8")
    G = Function.networkx(fileedge)
    i_knowncom=Alg5(i,G,Ck,fileedge)
    c_graph, n_subgraph, weight = Function.graph_subgraph_weight(i_knowncom, fileedge)
    # 根据node_konwncom来提取已知社区的社区子图，节点子图，及权重
    global_com = Function.com_trans_graph(c_graph, fileedge)  # 转换成最短路径形式表示的图
    local_com = Function.com_trans_graph(n_subgraph, fileedge)
    C1, N1 = select_first_node(i, G)  # 将给定种子节点公共邻居数最多的节点加入社区
    C2, N2, nodesim = Alg6(C1, N1, local_com, G, global_com, maxsize, weight)  # 社区相似性阶段
    C, N = Alg7(C2, N2, local_com, G, maxsize, weight, nodesim)  # 节点相似性阶段
    f.writelines(str(C) + "\n")
    f.close()


if __name__ == '__main__':
    seed = Function.big_dataset_seed()  # 数据集的种子节点
    known_com = ['dataset/amazon_knowcom', 'dataset/dblp_knowcom', 'dataset/facebook_knowcom', 'dataset/twitter_knowcom', 'dataset/lj_knowcom']
    file_edge = ['amazon-1.90.ungraph.txt', 'dblp-1.90.ungraph.txt', 'facebook-1.90.ungraph.txt', 'twitter-1.90.ungraph.txt', 'lj-1.90.ungraph.txt']
    path = ['amazon', 'dblp', 'facebook', 'twitter', 'lj']
    datasets = list(zip(seed, known_com, file_edge, path))
    for filename in datasets:
        print("数据集：", filename[3])
        dataset_knowcom = Function.get_knowcom(filename[1])
        for i in range(1, 6):
            print("第", i, len(dataset_knowcom[i - 1]), "组已知社区")
            start = datetime.datetime.now()
            Ck = dataset_knowcom[i - 1]
            maxsize = max([len(jjj) for jjj in Ck])
            filename_edge = "dataset/" + filename[2]  # 数据集对应边的路径
            nodelist = filename[0]  # 获取数据集的种子节点
            savefile = "Results/" + filename[3] + "/" + str(i) + "/"  # 把每个节点找到的社区写入文件中
            if os.path.isdir(os.path.dirname(savefile)) == False:  # 创建目录
                os.makedirs(os.path.dirname(savefile))
            plist = [[node, filename_edge, Ck, savefile, maxsize] for node in
                     nodelist]
            pool = multiprocessing.Pool(processes=1)  # 开启的进程数量
            pool.map(Go, plist)
            pool.close()
            pool.join()
            end = datetime.datetime.now()
            print("节点个数：", len(nodelist))
            print("耗时：{}".format(end - start))

