#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 15:39:08 2020

@author: shaoqianchen
"""

import networkx as nx
import matplotlib.pyplot as plt
import math
import os
import numpy as np
import statistics
"""
For  this homework, you will need to have the networkx package installed. If you are using pycharm, you can install
this package just like you installed numpy, sklearn etc. earlier. If not using pycharm, use pip to install the package in  your
interpreter. For more details, see  networkx's documentation on  installation. It is usually straightforward.
You will know you have succeeded when you do not see an error in  the first line (import networkx as nx).

NetworkX webpage: https://networkx.github.io/

As its name suggests, networkx is a 'graph theory' package that you can use to read, write and manipulate
networks of all kinds, including directed, undirected, weighted and attributed networks. It also gives you
many of the tools you'll ever need for getting diagnostic info on a network, including degree distribution,
connected components, and even community detection. I suggest looking at the resources in the tutorial section
to familiarize yourself with the basics: https://networkx.github.io/documentation/latest/tutorial.html
"""

"""
The network that we will be dealing with in this assignment is openly available in Stanford's SNAP ecosystem 
and involves Facebook data: http://snap.stanford.edu/data/ego-Facebook.html

"""

"""
As always, the first step is to download and familiarize yourself with the data. There should be three files at the
bottom of the html page I gave  you the link above for. The network we will be dealing with is facebook_combined.txt.gz,
which you should decompress to get the txt file. Note that sometimes
you have to refresh the page or try a different browser if you don't see the links to the files at the bottom. If none
of those work and you still don't see the files, write to me. 
"""

"""
The dataset that you see is what's called an edge-list. Each line is an edge e.g., 0 10 means that node 0 is linked to
node 10. Numerical IDs have been assigned to nodes for anonymization purposes. In this case, we will treat the network as undirected.
(WCC and SCC on the webpage stand for weakly connected components and strongly connected components. The metrics
are identical for them, as they always are if the network is undirected, since there is no distinction between
weakly connected/strongly connected components, which is why we only use the term 'connected components' in
undirected networks).  
"""
input_file = "facebook_combined.txt"
def read_in_network(input_path):
    """
    [5  points] In this function, which could be very short, you should read in  the facebook text file as a
    networkX graph object. See what I said above about the graph being undirected. Hint: there is a very nice
    and elegant way in networkX to read edge lists directly from file into a graph object.
    :param input_path: the path to the decompressed facebook_combined txt file.
    :return: a networkX graph object G
    """
    fh = open(input_path,"rb")
    G = nx.read_edgelist(fh)
    fh.close()
    return G

G = read_in_network(input_file)


def verify_webpage_numbers(G):
    """
    [25 points] In this function, write code (you can use networkX functionality to any extent that you wish) to verify
    the numbers on the SNAP webpage describing this dataset regarding the average clustering coefficient, 
    the number of triangles, the diameter, and the nodes and edges in the largest connected component. Is there any discrepancy?
    Please report if so.
    :param G: the networkX graph object
    :return: None
    """
    num_node = G.number_of_nodes()
    print("Total nodes in G: ", num_node)
    num_edge = G.number_of_edges()
    print("Total edges in G: ", num_edge)
    ACC = nx.average_clustering(G)
    print("Average clustering coefficient: ", ACC)
    G_tri = nx.triangles(G)
    num_tri_3 = 0
    for i in G_tri.values():
        num_tri_3+=i
    print("Number of triangles in G: ", num_tri_3/3)
    num_diameter = nx.diameter(G)
    print("Diameter of G: ", num_diameter)
    
verify_webpage_numbers(G)
"""
ANS:
    Total nodes in G:  4039
    Total edges in G:  88234
    Average clustering coefficient:  0.6055467186200876
    Number of triangles in G:  1612010.0
    Diameter of G:  8
    Verified that G is the same as the original network
"""





def degree_distribution(G):
    """
    [15 points] Plot the degree distribution on a log-log plot. You can use python (e.g., matplotlib and pyplot if you
    wish for the plot, or collect data and plot it somewhere else, like excel).
    Does the degree distribution obey the power law well? 
    If so, what is the power-law coefficient? 
    (Recall that a power law says that the prob(k) of degree k is proportional to k^{-gamma}. 
    Gamma is known as the power-law coefficient (it also has other names).

    The plot should be separately included with  your homework submission (a short PDF report, containing answers/figures,
    with captions to any questions that cannot be directly answered in file here)
    
    Hint: On a log-log plot, the power-law is linear. You can use linear regression (such as in sklearn) to deduce gamma.
    
    :param G: the networkX graph object
    :return: None
    """
    d = nx.degree_histogram(G)
    x = range(len(d))
    y = [i / float(sum(d)) for i in d]
    plt.figure()
    plt.loglog(x,y)
    plt.title("Degree Distribution Log-Log Plot")
    plt.savefig('Degree Distribution Loglog Plot.png')
    plt.show()
    
    



degree_distribution(G)




def find_max_degree(G):
    degrees = dict(G.degree())
    max_value = max(degrees.values())
    max_key = max(degrees, key=degrees.get)
    CC = nx.clustering(G,max_key)
    print("The highest degree node is: ",max_key)
    print("It has ",max_value, " friends")  
    print("Its clustering coefficient is: ", CC)
    #[5 points] How many friends does the node with the highest degree have? 
    # What is the clustering coefficient of that node? 
    # Interpret the result: do you believe this individual is a member of a cohesive and tightly knit
    # group or more like a central figure that people are connecting to for some reason (e.g., celebrity status
    # or because the person is popular, or just adding lots of friends on facebook for no reason, as some people do)

#find_max_degree(G)

"""
            The highest degree node is:  107
            It has  1045  friends
            Its clustering coefficient is:  0.049038479165520905
    ANS:
        Clustering coefficient tells how well connected the neighborhood of the node is.
        The coefficient of this node is pretty close to 0 which means that 
        there are hardly any connections in the neighborhood. So this individual
        could be a celebrity's account, some brands' official account, etc.
"""




    #[10 points extra credit]. Separately report the degree distributions of the ten individual networks on the dataset
    # webpage under file facebook.tar.gz. The folder (obtained upon decompressing) contains many files; 
    # edge lists are contained in [num].edges files.

def extra_credit_edge(path_of_facebooktargz):    
    dirs = os.listdir(path_of_facebooktargz)
    edge_list = []
    for i in dirs:
        if ".edges" in i:
            #plot loglog plot
            edge_list.append(i)
            temp_G = read_in_network(path_of_facebooktargz+"/"+i)
            d = nx.degree_histogram(temp_G)
            x = range(len(d))
            y = [i / float(sum(d)) for i in d]
            plt.figure()
            plt.loglog(x,y)
            plt.title("Degree Distribution Log-Log Plot "+i)
            #plt.savefig(i+'Degree Distributionloglog.png')
            plt.show()
            #plot distribution histagram
            d = dict(temp_G.degree())
            d = {int(k):int(v) for k,v in d.items()}
            plt.figure()
            plt.title("Degree Distribution Hist Plot"+i)
            #plt.savefig(i+'Degree Distribution Hist.png')
            plt.hist(list(d.values()), 50, density=True, facecolor='g', alpha=0.85)
            plt.show()

#extra_credit_edge("/Users/shaoqianchen/Desktop/2020 Fall/ISE 540/HW4/facebook")            
       






"""
Now, we will analyze what happens to the metrics if you only get to observe a 'fraction' of the network, as most of us do when
we are working  with these kinds of networks at very large-scale. 
"""

def sample_edges(G, num_edges):
    """
    [10 points] Complete the code. Make sure to use numpy or scipy for your sampling needs, as always. Note that because
    we are sampling edges, rather than nodes, nothing can be said about the number of nodes in the output graph. 
    In theory, because there are roughly 4000+ nodes, you may end up getting all nodes even if you sample only 5000 edges!
    :param G: the original graph
    :param num_edges: number of edges to sample. If -1 or greater than the total number of edges in G, just
    return the full graph G.
    :return: another graph H that contains num_edges randomly sampled edges from G
    """
    total_edge = G.number_of_edges()
    edge_list = list(G.edges())
    if (num_edges == -1 or num_edges>=G.number_of_edges()):
        return G
    else:
        sampling = list(np.random.choice(total_edge,num_edges))
        sampled_edges = []
        for i in sampling:
            sampled_edges.append(edge_list[i])
        new_G = G.edge_subgraph(sampled_edges)
        return new_G

#N = sample_edges(G, 20)




"""
[40 points]
Use  the function above  (including any others you need) to randomly sample edges 
from G to produce four graphs with 15000, 30000, 45000 and 60000 edges respectively.
Now re-compute the  
average clustering coefficient, 
fraction of closed triangles and 
diameter 
(note that if the graph is disconnected, the diameter is infinite. Please note if this happens) on these graphs. For
best results, you should sample (for each size) ten times and then compute averages and 
standard deviations of the network metrics noted above. Ignore infinite diameters
for the purpose of averaging, unless  the majority of diameters in  your 10-sample are infinite. 

In the report, tabulate these results and provide a succinct conclusion. Is there a sample size at which the
network starts looking like the original network? Which (if any) of the quantities you have computed above
are reliable and at what sampling  threshold? For example, you may conclude, 'fraction of closed triangles starts
approaching the level in  the original network even in the 15000 edge sample, but the diameter continues to be
untrustworthy till we see at least 45000 edges...'. 

I am looking for short, insightful reports. Statistical significance testing, if applied will boost your score. 
In particular, look to such testing for writing a decisive conclusion.   
"""
num_sample_edge = [15000,30000,45000,60000]
def sampling_compare(G,num_sample_edge):
    stand_deviations = 0
    for i in num_sample_edge:        
        ACC = [] 
        diameter= [] 
        temp_avg_diameter = 0
        triangle = []
        num_node = []
        for trail in range(10):
            K = sample_edges(G, i) 
            num_node.append(K.number_of_nodes())
            temp_ACC = nx.average_clustering(K)
            ACC.append(temp_ACC)            
            
            #triangles
            G_tri = nx.triangles(K)
            num_tri_3 = 0
            for m in G_tri.values():
                num_tri_3+=m
            triangle.append(num_tri_3)
            
            #diameter
            try:
                temp_avg_diameter += nx.diameter(K)
                diameter.append(nx.diameter(K))
            except:
                pass
        
        if len(diameter) == 0:
            temp_avg_diameter = 999999999
        else:
            temp_avg_diameter = temp_avg_diameter/len(diameter)
            for i in diameter:
                stand_deviations+=(i-temp_avg_diameter)**2
            stand_deviations = (stand_deviations/len(diameter))**(1/2)
            
        

        print("      Sampling ",i," Edges")
        print("Average clustering coefficient for ",i," sampling is: ",statistics.mean(ACC))
        print("                     Standard deviation: ", np.std(ACC))
        print("Number of triangles: ", statistics.mean(triangle)/3)
        print("                 Standard deviation: ", np.std([t/3 for t in triangle]))
        print("                 Fraction of closed triangles: ", statistics.mean(triangle)/(statistics.mean(num_node)*(statistics.mean(num_node)-3)))
        print("The average diameter of 10 trials for ",i," sampling is: ",temp_avg_diameter)
        print("The standard deviation of average diameter is: ", stand_deviations)
        
        print("\n")
        
#sampling_compare(G,num_sample_edge)




"""
[10 points extra credit]. You are trying to sample x edges, 
and the 
metric of success is the degree distribution
(both in shape and if the power-law applies, the power-law coefficient) 
and its similarity for your sampled network to the original network. 

Rounded to the nearest 5000, 
at what point does your sample bring you 'close enough' to the original 
(i.e. what is the value of x we should use)?
We are qualitatively unwilling to accept an 'error' of more than 10% e.g., 
the true coefficient must be within  10% of
the coefficient you get from your sampled network's degree distribution. 
"""



def close_enough(G,num_edge):
    #return: true if the point is close enough to the original, false if not
    G_new = sample_edges(G,num_edge)
    real = nx.degree_pearson_correlation_coefficient(G)
    current = nx.degree_pearson_correlation_coefficient(G_new)
  
    if(current<=real*1.1 and current>=real*0.9):
        return True
    else:
        return False

def find_X(G):
    maxi_edge = nx.number_of_edges(G)
    for i in range(5000,maxi_edge):
        print(i)
        if(close_enough(G,i)):
            print("It is close enough to the original at: ",i)
            break
find_X(G)
