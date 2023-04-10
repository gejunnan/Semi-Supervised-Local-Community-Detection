import os
import Function

seed=Function.big_dataset_seed_top()
filecom = ["dataset/amazon-1.90.cmty.txt","dataset/dblp-1.90.cmty.txt","dataset/facebook-1.90.cmty.txt","dataset/twitter-1.90.cmty.txt","dataset/lj-1.90.cmty.txt"]
filenedge = ['dataset/amazon-1.90.ungraph.txt', 'dataset/dblp-1.90.ungraph.txt', 'dataset/facebook-1.90.ungraph.txt',
         'dataset/twitter-1.90.ungraph.txt','dataset/lj-1.90.ungraph.txt']
path=[["SLSS6.2_Results/amazon/1/","SLSS6.2_Results/amazon/2/","SLSS6.2_Results/amazon/3/","SLSS6.2_Results/amazon/4/","SLSS6.2_Results/amazon/5/"]
    ,["SLSS6.2_Results/dblp/1/","SLSS6.2_Results/dblp/2/","SLSS6.2_Results/dblp/3/","SLSS6.2_Results/dblp/4/","SLSS6.2_Results/dblp/5/"]
    ,["SLSS6.2_Results/facebook/1/","SLSS6.2_Results/facebook/2/","SLSS6.2_Results/facebook/3/","SLSS6.2_Results/facebook/4/","SLSS6.2_Results/facebook/5/"]
    ,["SLSS6.2_Results/twitter/1/","SLSS6.2_Results/twitter/2/","SLSS6.2_Results/twitter/3/","SLSS6.2_Results/twitter/4/","SLSS6.2_Results/twitter/5/"]
    ,["SLSS6.2_Results/lj/1/","SLSS6.2_Results/lj/2/","SLSS6.2_Results/lj/3/","SLSS6.2_Results/lj/4/","SLSS6.2_Results/lj/5/"]
    ]
dataname=['amazon','dblp','facebook','twitter',"lj"]
datasets=list(zip(filecom,filenedge,path,dataname))

for dataset in datasets[3:]:
    print("数据集：",dataset[3])
    for data in dataset[2]:
        print(data)
        know=[]
        node_list1, node, com_index, node_com_index = Function.com_node_index_list_list(dataset[0], dataset[1])
        for root, dirs, files in os.walk(data):
            for filename in files:
                file = open(data + filename, "r")
                lines = file.readlines()
                a1 = []
                for line in lines:
                    lines1 = line.strip("[]\n").split(',')
                    if len(lines1) == 1:
                        lines1 = []
                    else:
                        lines1 = [int(a) for a in lines1]
                    know.append(lines1)
        ff = []
        jj = []
        rr = []
        pp = []
        for i in know:
            p1 = []
            f1 = []
            r1 = []
            j1 = []
            a = node_com_index[i[0]]
            b = []
            for jjjj in a:
                b.append(com_index[jjjj])
            for jjj in b:
                r, p, f, j = Function.compare_comm(i, jjj)
                f1.append(f)
                r1.append(r)
                p1.append(p)
                j1.append(j)
            ff.append(round(sum(f1) / len(f1), 4))
            rr.append(round(sum(r1) / len(r1), 4))
            pp.append(round(sum(p1) / len(p1), 4))
            jj.append(round(sum(j1) / len(j1), 4))
        print(len(rr))
        print("avg_r:", round(sum(rr) / len(rr), 4))
        print("avg_p:", round(sum(pp) / len(pp), 4))
        print("avg_f:", round(sum(ff) / len(ff), 4))
        print("avg_j:", round(sum(jj) / len(jj), 4))
        print(round(sum(rr) / len(rr), 4))
        print(round(sum(pp) / len(pp), 4))
        print(round(sum(ff) / len(ff), 4))
        print(round(sum(jj) / len(jj), 4))