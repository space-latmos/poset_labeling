# -*- coding: utf-8 -*-
"""
Éditeur de Spyder
%in http://michel.stainer.pagesperso-orange.fr/PSIx/Informatique/Cours/Ch2-Tris.pdf




"""

import os  
from PIL import Image  
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import matplotlib.pyplot as plt
import pickle
import random
import numpy as np
import copy    
import networkx as nx
import statistics as st #median for plotting
import networkx.algorithms.dag as DAG
import torch

os.chdir(r'C:\Users\Deep Yawner\Desktop\tri_images\training_on_AMOS\src')
from transforms_AMOS import same_crop_Transform
os.chdir(r'C:\Users\Deep Yawner\Desktop\tri_images')
#import config

#mode = ''  #intialisation de la variable globale

#%% Implémenter l'algo Poset-Mergesort pour des images:


#global variables
pause = 0.03
i = 0
j = 0

def invert_edges(edges):
    inverted_edges = set()
    for edge in edges:
        inverted_edges |={(edge[1], edge[0])}
    return inverted_edges

def edges_and_inverted_edges(edges):
    edges = set(edges)
    inverted_edges = invert_edges(edges)
    return edges.union(inverted_edges)


def true_edges_and_inverted_true_edges(graph):
    edges = {edge for edge in graph.edges if 'weight' in graph.edges[edge]}
    inverted_edges = invert_edges(edges)
    return edges.union(inverted_edges)


##############################################################
##############################################################
###################### tools for dicts      ##################
##############################################################
##############################################################

def count_dic(dic):
    count = 0
    for key in dic:
        count+=len([cpl for cpl in dic[key] if True ]) #cpl[0] == None])
    return count

def count_dic_None(dic):
    count = 0
    for key in dic:
        count+=len([cpl for cpl in dic[key] if cpl[0] == None])
    return count


def purge_dic(dic):
    for key in dic:
        toremove = {cpl for cpl in dic[key] if cpl == None}
        dic[key] = dic[key] - toremove
    
    
    for key in dic:
        toremove = {cpl for cpl in dic[key] if cpl[0] in [None, 4]}
        dic[key] = dic[key] - toremove
    return dic




def is_symmetric(**kwargs):
    dic_path = kwargs['dic_path']
    count = 0
    pickle_in = open(dic_path,"rb")
    dic_of_compas = pickle.load(pickle_in)
    pickle_in.close()
    for key in dic_of_compas:
        for cpl in dic_of_compas[key]:
            if (invert_POSET(cpl[0]), key) not in dic_of_compas[cpl[1]]:
                count+=1
    return count, count_dic(dic_of_compas)
    
    
def symmetrize(**kwargs):
    dic_path = kwargs['dic_path']
    pickle_in = open(dic_path,"rb")
    dic_of_compas = pickle.load(pickle_in)
    pickle_in.close()
    for key in dic_of_compas:
        for cpl in dic_of_compas[key]:
            dic_of_compas[cpl[1]].add((invert_POSET(cpl[0]), key))
    pickle_out = open(dic_path,"wb")
    pickle.dump(dic_of_compas, pickle_out)
    pickle_out.close()


##############################################################
##############################################################
###################### tools for  graph      #################
##############################################################
##############################################################


#names = sorted(os.listdir(images_dir))
def depth_of_cam(dg2):
    nodes = dg2.nodes
    lengths = []
    ks = list(range(10,len(nodes)))
    for k in ks:
        length = 0
        for n in range(0,10):
            subnodes = random.sample(nodes,k)
            length+= len(nx.dag_longest_path(dg2.subgraph(subnodes)))
        length = length/10
        lengths.append(length)
    plt.figure(1)
    plt.plot(ks, lengths)
    return lengths 


def restrict_to_names(names, **kwargs):    
    graphs = get_graphs(**kwargs)
    for graph in graphs:
        nodes = set(graph.nodes)
        nodes_to_remove = nodes - set(names)
        for node in nodes_to_remove:
            graph.remove_node(node)
    refresh_graphs(graphs, **kwargs)


def count_labels(**kwargs):
    dg, ug, eg = get_graphs(**kwargs)
    count = 0
    for edge in dg.edges:
        name0 = edge[0]
        name1 = edge[1]
        if 'weight' in dg[name0][name1]:
            if dg[name0][name1]['weight'] == 0:
                count+=1
                #print('yo')

    for edge in ug.edges:
        name0 = edge[0]
        name1 = edge[1]
        if 'weight' in ug[name0][name1]:
            if ug[name0][name1]['weight'] == 0:
                count+=1
 
    return count

def get_poset(root_cs, critere, subgroup, mode, **kwargs):
    suffixe = get_suffixe(critere,subgroup,mode)
    path_of_dg2 = os.path.join(root_cs, "poset" + suffixe + r".gpickle" )
    dg2 = nx.read_gpickle(path_of_dg2)    
    return dg2

def get_graph_by_eq(dg2, decomposition):  #réduit dg2 en divisant par la relation d'équibvlance
    nodes = []
    for chain in decomposition:
        for eqnodes in chain:
            nodes.append( random.choice(eqnodes))
            
    return dg2.subgraph(nodes).copy()


def count_22graph(**kwargs):
    
    decomposition = get_decomposition(**kwargs)
    dg2 = get_poset(**kwargs)

    sgraph = get_graph_by_eq(dg2, decomposition)
    comp = DAG.transitive_closure(sgraph).to_undirected()
    comp = nx.complement(comp)
    
    count = 0
    
    for a in sgraph.nodes:
        print(a)
        #get neighbors of successors
        nofs = set()
        for s in set(sgraph.successors(a)):
            nofs |= set(comp[s].keys())

        
        #get predecessors of neighbours
        pofn  = set()
        for d in comp[a]:
            pofn |= set(sgraph.predecessors(d))

        # count the intersection
        print(nofs.intersection(pofn))
        count += len(nofs.intersection(pofn))
#        if len(nofs.intersection(pofn)) >0:
#            break
    
    return count/2


def get_len(**kwargs):
    dg,ug,eg = get_graphs(**kwargs)
    print('total lengths :')
    print(len(dg.edges), len(ug.edges), len(eg.edges))
    tldg = len([e for e in dg.edges if dg.edges[e].get('weight') is not None])
    tlug = len([e for e in ug.edges if ug.edges[e].get('weight') is not None])
    tleg = len([e for e in eg.edges if eg.edges[e].get('weight') is not None])
    print('true lengths :')
    print(tldg, tlug, tleg)

##############################################################
##############################################################
###################### tools for decomposition      #################
##############################################################
##############################################################

def get_decomposition(root_cs, critere, subgroup, mode, **kwargs):
    suffixe = get_suffixe(critere,subgroup,mode)
    path_of_decomposition = os.path.join(root_cs, "decomposition" + suffixe + ".pickle" ) 

    pickle_in = open(path_of_decomposition,"rb")
    decomposition = pickle.load( pickle_in)
    pickle_in.close()
    return decomposition

def list_in_decomposition(name, decomposition):
    for a in decomposition:
        for b in a:
            if name in b:
                return b
            else:
                pass
    raise NameError('name not in decomposition')

       
def decomposition_to_names(decomposition):
    names = []
    for list_of_elements in decomposition:
        for element in list_of_elements:
            names+= element
    return names

##############################################################
##############################################################
###################### get prediction       ##################
##############################################################
##############################################################

def get_prediction(path0, path1,device,model,**kwargs):
    marginsup = 0.
    margininf = 0.19
    cropped_prop = 0.8

    size_in = 256+128+32
    size_out = 256+128-32 

    img0 = Image.open(path0).convert("RGB")    
    img1 = Image.open(path1).convert("RGB")
        
    tr = same_crop_Transform(marginsup, margininf, cropped_prop, size_in, size_out, zoom = 0, rotation = False, fixed_top = True)    
    [img0, img1] = tr([img0,img1])
    

    img0 = img0.to(device).unsqueeze(dim=0)
    img1 = img1.to(device).unsqueeze(dim=0)
    inputs = torch.cat((img0,img1), 1)

    # phase 1 get the comparison
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
    
    return outputs.cpu().numpy()
            
  
    

##############################################################
##############################################################
###################### plotting images      ##################
##############################################################
##############################################################


def show_image(name, title, images_dir, **kwargs):
    image_path = os.path.join(images_dir, name)
    plt.pause(pause) 
    
    plt.figure(num=0,figsize=(10,10))
    #plt.clf()
    im = Image.open(image_path)
    newsize = (1000, 1000) 
    im = im.resize(newsize) 
    plt.title(title)
    plt.imshow(im)
    
    
    plt.pause(pause) 



def compare(name1,name2, mode, images_dir, critere, test_model, **kwargs):
    global pause
#    critere = kwargs['critere']
    image_path1 = os.path.join(images_dir, name1)
    image_path2 = os.path.join(images_dir, name2)

    if test_model:
        predicted_probas = get_prediction(image_path1,image_path2,**kwargs)
        pbs = predicted_probas[0,:]
        pbs = np.exp(pbs)/np.sum(np.exp(pbs))
        pbs = str(np.round(pbs,decimals = 2))

    else:
        pbs = ''

    print(name1)
    print(name2)
    plt.figure(num=0, figsize=(10,10))
           
    im1 = Image.open(image_path1)
    im2 = Image.open(image_path2)
    newsize = (1000, 1000) 
    im1 = im1.resize(newsize) 
    im2 = im2.resize(newsize) 
    
    key=''
    while key == '':
        plt.pause(pause)
        plt.figure(num=0, figsize=(10,10))
        plt.title('image 0  ' + name1[-10:-4] + ' ' + pbs)
        plt.imshow(im1)
        plt.pause(pause)
        key = input("critere : " + critere + "\n" + 'mode : ' + mode + "\n" + "Press enter to change image")
        print(key)
        if key == '':
            plt.clf()
            if test_model:
                plt.title('image 1  ' + name2[-10:-4] + '   ' + pbs  )
                plt.imshow(im2)
            else:
                plt.title('image 1  ' + name2[-10:-4] + '   ' + critere + '  ' + mode )
                plt.imshow(im2)
        plt.pause(pause)
        key = input("critere : " + critere + "\n" +"Press enter to change image")
    print(key)
    #if key == 
    return(int(key))

"""
def show_image(name, images_dir, critere, **kwargs):
        global pause
        image_path1 = os.path.join(images_dir, name)
        im1 = Image.open(image_path1)
        newsize = (1000, 1000) 
        im1 = im1.resize(newsize)
        plt.pause(pause)
        plt.figure(num=0, figsize=(10,10))
        plt.title('image 0  ' + name[-9:-4])
        plt.imshow(im1)
        plt.pause(pause)
        input("critere : " + critere + "\n" + "Press enter to change image")
"""        

    
def show_list(list_of_names, **kwargs):
    for name in list_of_names:
        show_image(name, **kwargs)


##############################################################
##############################################################
###################### plotting graphs        ################
##############################################################
##############################################################
        
    
    
def draw_nodes_round(**kwargs):
    global pause
    dg, ug, eg = get_graphs(**kwargs)
    fig, ax = plt.subplots(num=100, figsize=(5,5))    
    fig.patch.set_visible(False)
    ax.axis('off')
    
    pos = {}  #nx.drawing.nx_agraph.graphviz_layout(dg, 'dot') #add pos in 2th pos1
    l = len(dg)
    nodes = sorted(list(dg.nodes))
    lab = {}
    pos_lab={}
    for i in range(l):
        lab[nodes[i]] = i #nodes[i][-15:-7]
        pos[nodes[i]] = (np.sin(i/l * 2 * np.pi),np.cos(i/l * 2 * np.pi))
        pos_lab[nodes[i]] = (1.08*np.sin(i/l * 2 * np.pi), 1.08* np.cos(i/l * 2 * np.pi))
        
    nx.draw_networkx_nodes(dg, pos, cmap = plt.get_cmap(), node_size = 20, alpha = .6, vmax=1, vmin =0) #node_color, node_list si l'on veut différencier    
    nx.draw_networkx_labels(dg, pos_lab, labels = lab, cmap = plt.get_cmap(), node_size = 1, font_size = 6) #node_color, node_list si l'on veut différencier    
    kwargs['pos']=pos
    return kwargs

def decomposition_to_dg2_and_plot(decomposition, **kwargs):
    global pause


    fig, ax = plt.subplots(num=200, figsize=(20,5))    
    fig.patch.set_visible(False)
    ax.axis('off')



    l = len(decomposition)

    #position of nodes
    nodes = []
    pos = {}  #nx.drawing.nx_agraph.graphviz_layout(dg, 'dot') #add pos in 2th pos1
    lab = {}
    pos_lab={}
    
    for i in range(l):
        chain = decomposition[i]
        lchain = len(chain)
        for j in range(lchain):
            list_of_nodes = chain[j]
            ln = len(list_of_nodes)
            for k in range(ln):
                node = list_of_nodes[k]
                nodes.append(node)
                lab[node] = str(i) + str(j) + str(k)#node[-15:-7]
                deltai = 0.5/ln
                median = st.median(range(ln))
                pos[node] = (i + (k-median)*deltai,-j)
                pos_lab[node] = (i + (k-median)*deltai,-j+0.1)

        
    dg2 = nx.DiGraph()
    dg2.add_nodes_from(nodes)
    nx.draw_networkx_nodes(nodes, pos, cmap = plt.get_cmap(), node_size = 20, alpha = .6, vmax=1, vmin =0) #node_color, node_list si l'on veut différencier    
    nx.draw_networkx_labels(nodes, pos_lab, labels = lab, cmap = plt.get_cmap(), node_size = 1, font_size = 6) #node_color, node_list si l'on veut différencier    
    kwargs['pos2']=pos
    return dg2, kwargs


def decomposition_to_dg2(decomposition):
    l = len(decomposition)

    #position of nodes
    nodes = []

    
    for i in range(l):
        chain = decomposition[i]
        lchain = len(chain)
        for j in range(lchain):
            list_of_nodes = chain[j]
            ln = len(list_of_nodes)
            for k in range(ln):
                node = list_of_nodes[k]
                nodes.append(node)

        
    dg2 = nx.DiGraph()
    dg2.add_nodes_from(nodes)
    return dg2


#decomposition = [[[1,2], [3], [5]], [[7],[8],[9,10]], [[12,13], [14], [15]]]
#edgelist=[(nodes[0],nodes[7])]        


def draw_new_edges(dg, new_edges, color, pos, root_cs, **kwargs):
    global pause
    global i

    plt.pause(pause)
    plt.figure(num=100)

#    fig, ax = plt.subplots(num=100, figsize=(5,5))    
#    fig.patch.set_visible(False)
#    ax.axis('off')


    plt.pause(pause)
    edgelist = list(new_edges)
    #pos= kwargs['pos']
    nx.draw_networkx_edges(dg, pos, edgelist = edgelist, arrows = True, width = 1, edge_color = 'k') #, arrowstyle = '-|>') #node_color, node_list si l'on veut différencier    
    plt.pause(pause)
    fig_graph1_path = os.path.join(root_cs, 'graphs1')
    address = os.path.join(fig_graph1_path, kwargs['critere'] + kwargs['subgroup']+'_' + kwargs['mode'] + '_graph1_' + str(i) + '.png')
    plt.savefig(address)
    i+=1
    print(address)
    
def draw_new_edges2(dg2, new_edges,color, pos2, root_cs, critere,subgroup, mode, **kwargs):
    global pause
    global j
    global fig_graph2_path
    plt.pause(pause)
    plt.figure(num=200)
    #fig, ax = plt.subplots(num=200, figsize=(20,5))    
    #fig.patch.set_visible(False)
    #ax.axis('off')
    plt.pause(pause)
    
    
    edgelist = list(new_edges)
    #pos= kwargs['pos']
    nx.draw_networkx_edges(dg2, pos2, edgelist = edgelist, arrows = True, width = 1, edge_color = color, arrowsize = 10, arrowstyle='-|>') #, arrowstyle = '-|>') #node_color, node_list si l'on veut différencier    
    plt.pause(pause)
    fig_graph2_path = os.path.join(root_cs, 'graphs2')
    address = os.path.join(fig_graph2_path, critere + subgroup+'_' + mode + '_graph2_' + str(j) + '.png')
    plt.savefig(address)
    j+=1
    print(address)





##############################################################
##############################################################
###################### get compa from label ##################
##############################################################
##############################################################

def invert_POSET(x):
    if x== 1:
        return 0
    elif x==0:
        return 1
    elif x== 2:
        return 2
    elif x==3:
        return 3
    elif x == 31:
        return 30
    elif x == 30:
        return 31
    elif x == 103:
        return 113
    elif x == 113:
        return 103
    elif x == 10:
        return 11
    elif x == 11:
        return 10
    else:
        raise NameError('bad label in invert')


def convert_cpl(cpl, mode, **kwargs):
    if mode == 'surface':
        OsupI = [0, 11, 30] #30: h1 ~ h0 but s0 > s1
        IsupO = [1,10,31]
        OeqI = [3,103,113]
    elif mode == 'height':
        OsupI = [0,10,103] #10: h0>h1 but s0<s1 103: h0>h1 and s1 ~s0
        IsupO =  [1,11,113]
        OeqI = [3,30,31]
    else:
        OsupI = [0] 
        IsupO = [1]
        OeqI =[3]
    #print(cpl[0])
    #print(OsupI)
    if cpl[0] in OsupI:
        return (0,cpl[1])
    elif cpl[0] in IsupO:
        return (1,cpl[1])
    elif cpl[0] in OeqI:
        return (3,cpl[1])
    elif  cpl[0] == 2:
        return (2, cpl[1])
    else:
        raise NameError(str(cpl) + ': bad label in convert_cpl')

    
def convert(x, mode, **kwargs):
    if mode == 'surface':
        OsupI = [0, 11, 30] #30: h1 ~ h0 but s0 > s1
        IsupO = [1,10,31]
        OeqI = [3,103,113]
    elif mode == 'height':
        OsupI = [0,10,103] #10: h0>h1 but s0<s1 103: h0>h1 and s1 ~s0
        IsupO =  [1,11,113]
        OeqI = [3,30,31]
    else:
        OsupI = [0] 
        IsupO = [1]
        OeqI =[3]
    #print(cpl[0])
    #print(OsupI)
    if x in OsupI:
        return 0
    elif x in IsupO:
        return 1
    elif x in OeqI:
        return 3
    elif  x == 2:
        return 2    
    else:
        raise NameError('bad label in convert')
        
        
def reduce_label(old_compas,**kwargs):
    #reduce to the actual names useless
    new_compas = copy.deepcopy(old_compas) 
    #for name in old_compas:
    #    if name not in names:
    #        del new_compas[name]
    #reduce to the actual mode
       
    for name in new_compas:
        new_set = set()
        for cpl in old_compas[name]:
            new_set.add(convert_cpl(cpl,**kwargs))
        new_compas[name] = new_set
    return new_compas

def reduce_name(old_compas,names):
    #reduce to the actual names useless
    new_compas = {} # copy.deepcopy(old_compas) 
    for name in old_compas:
        if name in names:
            new_compas[name] = set()
            for cpl in old_compas[name]:
                if cpl[1] in names:
                    new_compas[name].add(cpl)
        
    return new_compas


##############################################################
##############################################################
###################### get graphs from dict ##################
##############################################################
##############################################################

def dic_of_compas_to_dig(**kwargs):
    dic = get_dic(**kwargs)
    geq = dic_to_eq(dic)    
    G, edges = dic_to_dig(dic, geq)
    return G, geq


def dic_of_compas_to_dig2(**kwargs):
    dic = get_dic(**kwargs)
    geq = dic_to_eq(dic)    
    G, edges = dic_to_dig2(dic, geq)
    return G, geq

def dic_to_dig(dic, geq):  #make dg from dic (geq already made)
    dig = nx.DiGraph()
    dig.add_nodes_from(dic)
    
    edges = set()
    
    for key in dic:
        for cpl in dic[key]:
            if cpl[0] == 0 and cpl[1] in dic:
                set0 = nx.descendants(geq,key).union({key})
                set1 = nx.descendants(geq,cpl[1]).union({cpl[1]})
                for name0 in set0:
                    for name1 in set1:
                        edges.add((name0, name1))            
                
            if cpl[0] == 1 and cpl[1] in dic:
                set1 = nx.descendants(geq,key).union({key})
                set0 = nx.descendants(geq,cpl[1]).union({cpl[1]})
                for name0 in set0:
                    for name1 in set1:
                        edges.add((name0,name1))
    
    dig.add_edges_from(edges)
    
    return dig, edges

def dic_to_dig2(dic, geq): #make dg + sym(geq) from dic
    dig = nx.DiGraph()
    dig.add_nodes_from(dic)
    
    edges = set()
    
    for key in dic:
        for cpl in dic[key]:
            if cpl[0] == 0 and cpl[1] in dic:
                name0 = key
                name1 = cpl[1]
                edges.add((name0, name1)) 
                
            if cpl[0] == 1 and cpl[1] in dic:
                name1 = key
                name0 = cpl[1]
                edges.add((name0, name1)) 
                
            if cpl[0] == 2 and cpl[1] in dic:
                name1 = key
                name0 = cpl[1]
                edges.add((name0, name1)) 
                edges.add((name0, name1))
                
    dig.add_edges_from(edges)
    
    return dig, edges


def dic_to_eq(dic):
    g = nx.Graph()
    g.add_nodes_from(dic)
    for key in dic:
        for cpl in dic[key]:
            if cpl[0] == 2 and cpl[1] in dic:
                g.add_edge(*(key,cpl[1]))
    return g

##############################################################
##############################################################
###################### load, update and save dicts ##########
##############################################################
##############################################################



def get_dic(dic_path, **kwargs):

    
    pickle_in = open(dic_path,"rb")
    dic_of_compas = pickle.load(pickle_in)
    pickle_in.close()
    
    dic = reduce_label(dic_of_compas, **kwargs)
    return dic



def save_new_compa(name1,name2,compa,dic_path, **kwargs):
     
    #rec the new compa
    pickle_in = open(dic_path,"rb")
    dic_of_compas= pickle.load(pickle_in)
    pickle_in.close()
    
    dic_of_compas[name1].add((compa,name2))
    dic_of_compas[name2].add((invert_POSET(compa),name1))
    print("new compa saved")    
    print(str(count_dic(dic_of_compas)) + ' compas for   ' + str(len(dic_of_compas)) + ' images')
    
    pickle_out = open(dic_path,"wb")
    pickle.dump(dic_of_compas, pickle_out)
    pickle_out.close()





##############################################################
##############################################################
###################### load, update and save graphs ##########
##############################################################
##############################################################


def get_graphs(graphs_paths,**kwargs):
    dg = nx.read_gpickle(graphs_paths[0]) 
    ug = nx.read_gpickle(graphs_paths[1])
    eg =nx.read_gpickle(graphs_paths[2])
    return (dg, ug, eg)

def get_dg2(root_cs, critere,subgroup,mode,**kwargs):
    if mode != '':
        suffixe = '_' + critere+subgroup+r'_'+mode 
    else:
        suffixe = '_' + critere+subgroup    
    path_of_dg2 = os.path.join(root_cs, "poset" + suffixe + r".gpickle" )  
    dg2 = nx.read_gpickle(path_of_dg2)     
    return dg2



def save_graphs(graphs, graphs_paths, **kwargs):
    dg, ug, eg = graphs
    nx.write_gpickle(dg, graphs_paths[0])
    nx.write_gpickle(ug, graphs_paths[1])
    nx.write_gpickle(eg, graphs_paths[2])
    print("graphs saved")

def impact_new_dg_edge(graphs,dg_edge):
    dg, _, eg = graphs   
    name0,name1 = dg_edge
    
    #1 get connexe components from eg
    component0 = nx.node_connected_component(eg,name0)
    component1 = nx.node_connected_component(eg,name1)

    #build new links
    new_dg_edges = {(node0,node1) for node0 in component0 for node1 in component1}
    dg.add_edges_from(new_dg_edges)    
    
    
def impact_new_ug_edge(graphs,ug_edge):
    _, ug, eg = graphs   
    name0,name1 = ug_edge
        
    #1 get connexe components from eg
    component0 = nx.node_connected_component(eg,name0)
    component1 = nx.node_connected_component(eg,name1)

    #2build new links    
    new_ug_edges = {(node0,node1) for node0 in component0 for node1 in component1}
    ug.add_edges_from(new_ug_edges)    

    
def impact_new_eg_edge(graphs,eg_edge):
    
    dg, ug, eg = graphs   
    name0,name1 = eg_edge

    #update eg and ug
    eg.add_edge(name0,name1)
    ug.add_edge(name0,name1)

    
    #1get the new connected component
    component = nx.node_connected_component(eg,name0)

    #2 get the whole predecessors, successors and neighbors
    nodes_up = set()
    nodes_down= set()
    nodes_unr = set()
    for node in component:
        nodes_up |= set(dg.predecessors(node))
        nodes_down |= set(dg.successors(node))
        nodes_unr |= set(ug.neighbors(node))

    #get new edges
    new_dg_edges = set()
    new_ug_edges = set()
    
    new_dg_edges |= {(node0,node1) for node0 in nodes_up for node1 in component}
    new_dg_edges |= {(node0,node1) for node0 in component for node1 in nodes_down}
    new_ug_edges |= {(node0,node1) for node0 in component for node1 in nodes_unr}


    #add new edges
    dg.add_edges_from(new_dg_edges) 
    ug.add_edges_from(new_ug_edges)    
    
    
    

def update_graphs(graphs, name0, name1, reduced_compa, weight=0, **kwargs):
        
        dg,ug,eg = graphs
        
        if reduced_compa == 0:
            new_dg_edge = {(name0,name1, weight)}
            dg.add_weighted_edges_from(new_dg_edge)  #this edge is marked

            impact_new_dg_edge(graphs,(name0,name1))  #all other component to compnent edges are build


            
        elif reduced_compa==1:
            new_dg_edge = {(name1,name0, weight)}
            dg.add_weighted_edges_from(new_dg_edge)

            impact_new_dg_edge(graphs,(name1,name0))  #all other component to compnent edges are build

            
        elif reduced_compa==3:
            ug.add_edge(name0,name1,weight=weight)
            
            impact_new_ug_edge(graphs,(name1,name0))  #all other component to compnent edges are build

            
        elif reduced_compa==2:
            eg.add_edge(name0,name1, weight = weight)
            ug.add_edge(name0,name1, weight = weight)
            
            impact_new_eg_edge(graphs,(name0,name1))


        else:
            raise NameError('sortie de boucle')
        

        

    


##############################################################
##############################################################
###################### tools for label correction  ###########
##############################################################
##############################################################

def get_naked(graphs):
    for graph in graphs:
        edges_to_remove = set()
        for edge in graph.edges:
            if graph[edge[0]][edge[1]].get('weight') == None:
#                print('in weight there is :' +   str((graph[edge[0]][edge[1]].get('weight'))))
#                raise NameError('check if weights have been well done')
                edges_to_remove.add(edge) 
        graph.remove_edges_from(edges_to_remove)

def rebuild(graphs):
    dg, ug, eg = graphs
    for edge in set(dg.edges):
        impact_new_dg_edge(graphs,edge)
    for edge in set(ug.edges):
        impact_new_ug_edge(graphs,edge)

def refresh_graphs(graphs, **kwargs):
    get_naked(graphs)
    rebuild(graphs)
    save_graphs(graphs,**kwargs)    


def graphs_to_label(edge, **kwargs):
    graphs  = get_graphs(**kwargs)
    dg, ug, eg = graphs
    anti_edge = (edge[1], edge[0])
    label = ''
    cdg = nx.transitive_closure(dg)
    if edge in cdg.edges:
        if edge in dg.edges:
            if 'weight' in dg.edges[edge]:
                label += '0 '
            else:
                label += '0uw'
        else:
            label += '0 c'
    if anti_edge in cdg.edges:
        if anti_edge in dg.edges:
            if 'weight' in dg.edges[anti_edge]:
                label += '1 '
            else:
                label += '1uw'
        else:
            label += '1 c'
    
    if edge in ug.edges:
        if 'weight' in ug.edges[edge]:
            label += '3 '
        else:
            label += '3uw '
    
    if edge in eg.edges:
        if 'weight' in eg.edges[edge]:
            label += '2 '
        else:
            label += '2uw '
    return label
    
    
def remove_from_graphs(edge,**kwargs):
    
    graphs  = get_graphs(**kwargs)
    dg, ug, eg = graphs
    anti_edge = (edge[1], edge[0])
    #eg_edges0 = list(eg.edges(edge[0]))
    #eg_edges1 = list(eg.edges(edge[1]))
    
    if edge in dg.edges:
        dg.remove_edge(*edge)
        gr = 'dg' 
        print('removal of the edge: ' + str(edge))
        print('this edge was in ' + gr)


    if anti_edge in dg.edges:
        dg.remove_edge(*anti_edge)
        edge = anti_edge
        gr = 'dg' 
        print('removal of the edge: ' + str(edge))
        print('this edge was in ' + gr)

    if edge in ug.edges:
        ug.remove_edge(*edge)
        gr = 'ug'
        print('removal of the edge: ' + str(edge))
        print('this edge was in ' + gr)

    if edge in eg.edges:
        eg.remove_edge(*edge)
        gr = 'eg'
        print('removal of the edge: ' + str(edge))
        print('this edge was in ' + gr)
    
    #eg.remove_edges_from(eg_edges0+eg_edges1)
        
    refresh_graphs(graphs, **kwargs)

def remove_from_dg(edge,**kwargs):
    graphs  = get_graphs(**kwargs)
    dg, ug, eg = graphs

    gr = 'dg'     
    if edge in dg.edges:
        dg.remove_edge(*edge)
        print('removal of the edge: ' + str(edge))
        print('this edge was in ' + gr)
    else:
         print('this edge was not in ' + gr)       

    refresh_graphs(graphs, **kwargs)

"""
for edge in liste_edge:
    remove_from_dg(edge,**kwargs)
    

"""


def check_path(path, **kwargs):
    global pause

#    critere = kwargs['critere']
#    subgroup = kwargs['subgroup']
#    mode = kwargs['mode']
    
    print(5*'\n' + 'check true path starting')
   # print(5*'\n' + 'check true path starting.  \n Critere : ' + critere + '\n Subgroup :' + subgroup + '\n Mode : ' + mode )
    
    graph_changed = False
    indexp = 0
    key = ''

    
    while indexp< len(path):
        
            name = path[indexp]
            edge = (path[indexp-1], path[indexp])
            
            print('rank :' + str(indexp))
            #print('critere :' + critere + "\n" + 'mode : ' + mode + "\n")
            print(name)
            if indexp>0:
                title = 'rank in path :' + str(indexp) + "   " + "label :" + graphs_to_label(edge, **kwargs) + "  name:" + name
            else:
                title = 'rank in path :' + str(indexp) + "   " + "  name:" + name
            show_image(name, title, **kwargs)       
            key = input("press 9 to remove the last edge")
            
            if key == '9' and indexp != 0:
                graph_changed = True
                remove_from_graphs(edge, **kwargs) 
                print('graph changed')
                #except:
                #    print("no existing comp there")
            if key == '1':
                indexp -=1
            elif key == '':
                indexp += 1
            elif key == '4':
                raise NameError('sortie de boucle')
                
    print('check path ending' + 5*'\n')
    return graph_changed

def make_complete_dg(**kwargs):  #pour pouvoir passer par les eq dans un sens ou dans l'autre

    dg, ug, eg = get_graphs(**kwargs)
    a = copy.deepcopy(dg)
    get_naked([a])
    b = copy.deepcopy(eg)
    get_naked([b])
    b=b.to_directed()
    b = nx.compose( b ,   nx.reverse(b) )   #symmetrization of b

    complete_dg = nx.compose(a, b)
    
    return complete_dg

def make_close_complete_dg(**kwargs):  #pour pouvoir passer par les eq dans un sens ou dans l'autre

    dg, ug, eg = get_graphs(**kwargs)
    a = copy.deepcopy(dg)
#    get_naked([a])
    a=nx.transitive_closure(a)
    b = copy.deepcopy(eg)
    get_naked([b])
    b=b.to_directed()
    b = nx.compose( b ,   nx.reverse(b) )   #symmetrization of b

    complete_dg = nx.compose(a, b)
    
    return complete_dg

def find_path(list_of_nodes, **kwargs): #find a labelled path going through the nodes of list_of_nodes    
    complete_dg = make_complete_dg(**kwargs)    
    
    path = [list_of_nodes[0]]
    
    l = len(list_of_nodes)
    
    for k in range(l-1):
        node1 =  list_of_nodes[k]
        node2 = list_of_nodes[k+1]
        path+= nx.shortest_path(complete_dg, node1, node2)[1:]
        
    print('length =0 to check')
    return path 


def find_path_in_cdg(list_of_nodes, **kwargs): #find a labelled path going through the nodes of list_of_nodes    
    complete_dg = make_close_complete_dg(**kwargs)    
    
    path = [list_of_nodes[0]]
    
    l = len(list_of_nodes)
    
    for k in range(l-1):
        node1 =  list_of_nodes[k]
        node2 = list_of_nodes[k+1]
        path+= nx.shortest_path(complete_dg, node1, node2)[1:]
        
    print('length =0 to check')
    return path 


def correct_graphs(node1,node2,**kwargs):
#    node1 = list1[0]
#    node2 = list2[0]
    
    path = find_path([node1, node2], **kwargs)
    need_to_resort = check_path(path, **kwargs)
    return need_to_resort

#% random check
def check_decomposition(decomposition, **kwargs):
    global pause
    images_dir = kwargs['images_dir']
    critere = kwargs['critere']
    subgroup = kwargs['subgroup']
    mode = kwargs['mode']
    
    print(5*'\n' + 'check decomposition starting.  \n Critere : ' + critere + '\n Subgroup :' + subgroup + '\n Mode : ' + mode )
    need_to_resort = False
    ldec = len(decomposition)
    
    
    safe_piles = decomposition.copy()
    corrupted_piles = []
    
    ichain=0    
    
    need_to_resort = False
    
    while ichain < ldec:
        
        chain = decomposition[ichain]
        index = 0
        key = ''
        print('ichain n°' +str(ichain) + ' of  ' + str(len(decomposition))+ ' chains')
        while index >=0 and index< len(chain):
                listi = chain[index]
                name = random.choice(listi)
                print('rank :' + str(index))
                print(name)
                image_path = os.path.join(images_dir, name)
                plt.pause(pause) 
                
                plt.figure(num=0,figsize=(10,10))
                #plt.clf()
                im = Image.open(image_path)
                newsize = (1000, 1000) 
                im = im.resize(newsize) 

                plt.title('decomp. rank :' + str(index) + "   " + name + '   ' + critere + '  ' + mode )
                plt.imshow(im)
                
                
                plt.pause(pause)        
                key = input("Press 9 to check last transition")
#                print('ichain : ' +str(ichain))
                if key == '9':
                                     
                    
                    print('correction')
#                    print('ichain : ' +str(ichain))
                    if index-1 < 0:
                        raise NameError('rank 0')
                        
                    node0 = random.choice(chain[index-1])
                    ntr2 = correct_graphs(node0,name, **kwargs)

#                    get_len(**kwargs)
                    need_to_resort = need_to_resort or ntr2
#                    get_len(**kwargs)
                    print('ichain : ' +str(ichain) + ' ' + str(len(decomposition)))
                    
                    if chain in safe_piles:
                        safe_piles.remove(chain)
#                    if need_to_resort:
#                        corrupted_piles+=chain[:index-1]
#                        corrupted_piles+=chain[index:]
                    
                    
                if key == '1':
                    index -=1
                elif key == '':
                    index += 1
                elif key == '4':
                    raise NameError('sortie de boucle')
                elif key == '5':
                    break
        if key =='5':
            break
        if key =='1':
            ichain-=1
        else:
            ichain+=1
        
    print('check decomposition ended' + 5*'\n')
    return need_to_resort, safe_piles, corrupted_piles


def get_list_of_cycles(G):   
    cycles = nx.simple_cycles(G)
    list_cycles = []
    count = 0
    for cycle in cycles:
        if len(cycle)>1:
            list_cycles.append(cycle)
            #print('new cycle: ' + str(cycle))
            count+=1
            if count == 10:
                break
        
    return list_cycles




def check_cycles(**kwargs):
    graphs = get_graphs(**kwargs)
    dg, _, _ = graphs
    #get cycles of G with "classical" labels 
    list_cycles = get_list_of_cycles(nx.transitive_closure(dg))
    
    
    if list_cycles == []:
        return False
    else:
        print("check first cycle")
        cycle = list_cycles[0]
        path1 = find_path(cycle, **kwargs)
        path2 = find_path([cycle[-1],cycle[0]], **kwargs)
        path = path1 + path2[1:]
        check_path(path, **kwargs)

        return True     




##############################################################
##############################################################
###################### get decomposition in chains ###########
##############################################################
##############################################################

def init_with_eg(**kwargs):

    _,_,eg = get_graphs(**kwargs)
    l = sorted(eg.nodes)
    names_2 = []
    waste = set()
    i = 0
    for name in l:
        if name not in waste:
            i+=1
            ccname = nx.node_connected_component(eg,name)
            waste |= ccname
            names_2.append([sorted(ccname)])
    
    print(str(i) + ' components to sort')
    
    return names_2
#     

def make_cpls_of_index(l):
    cpls_of_index = []
    for i in range(l):
        for j in range(i+1,l):
            cpls_of_index.append((i,j))
    return cpls_of_index

def update_cursors_and_pointers(i,j,cursors, pointers):                    
    pointers[(i,cursors[i])] = (j,cursors[j])
    cursors[i]+=1
    return cursors, pointers

def InfOrUnc(graphs,name0,name1):
    dg, ug, eg = graphs
    
    ancestors0 = nx.ancestors(dg,name0).union({name0})
    descendants1 = nx.descendants(dg,name1).union({name1})
    possible_edges = {(node0,node1) for node0 in ancestors0 for node1 in descendants1}
    
    if len(possible_edges.intersection(set(ug.edges))) > 0:
        #print(possible_edges.intersection(set(ug.edges)))
        return True
    else:
        return False
    
def do_the_impossible(graphs, name0,name1,label, **kwargs):
    dg, ug, eg = graphs

    component0 = nx.node_connected_component(eg,name0)
    component1 = nx.node_connected_component(eg,name1)



    if label == 0:
        #get the ancestors of component0
        ancestors0 = set()
        for name in component0:
            ancestors0 |= nx.ancestors(dg,name).union({name})
        
        #get the descendants of component1
        descendants1 = set()
        for name in component1:
            descendants1 |= nx.descendants(dg,name).union({name})
        
        
        #look for the unwanted true edges
        ug_true_edges = true_edges_and_inverted_true_edges(ug)
        possible_edges = {(node0,node1) for node0 in ancestors0 for node1 in descendants1}.intersection(ug_true_edges)


        #find the unwanted true edge        
        for (node0,node1) in possible_edges:
            graph_changed0 = check_path([node0,node1], **kwargs)
            if not graph_changed0:                
                graph_changed0a = check_path(find_path([node0,name0],**kwargs), **kwargs)
                if not graph_changed0a:
                    graph_changed0b = check_path(find_path([name1,node1],**kwargs), **kwargs)
            if graph_changed0 or graph_changed0a or graph_changed0b:
                break
            
    elif label == 1:
        #get the ancestors of component1
        ancestors1 = set()
        for name in component1:
            ancestors1 |= nx.ancestors(dg,name).union({name})
        
        #get the descendants of component0
        descendants0 = set()
        for name in component0:
            descendants0 |= nx.descendants(dg,name).union({name})
        
        
        #look for the POSSIBLE unwanted true edges
        ug_true_edges = true_edges_and_inverted_true_edges(ug)
        possible_edges = {(node1,node0) for node1 in ancestors1 for node0 in descendants0}.intersection(ug_true_edges)

        #find the unwanted true edge
        for (node1,node0) in possible_edges:
            graph_changed1 = check_path([node1,node0],**kwargs)
            if not graph_changed1:
                graph_changed1a = check_path(find_path([name0,node0],**kwargs), **kwargs)
                if not graph_changed1a:                
                    graph_changed1b = check_path(find_path([node1,name1],**kwargs), **kwargs)
            if graph_changed1 or graph_changed1a or graph_changed1b:
                break
            

def get_possible_compas(graphs, name0, name1):
    #test if name0 >= name1
    possible_compas = {0,1,2,3}
    inforunc01 = InfOrUnc(graphs,name0,name1)
    
    if inforunc01:
        possible_compas -= {0,2}
        
    inforunc10 = InfOrUnc(graphs,name1,name0)
    
    if inforunc10:
        possible_compas -= {1,2}   

    return possible_compas
    


def oracle(graphs, name0,name1, **kwargs):  #use it for the second phase
#    root_cs = kwargs['root_cs']
#    critere = kwargs['critere']
#    images_dir   = kwargs['image_dir']
    mode   = kwargs['mode']
    
    
    dg,ug,eg = graphs
    new_edges = set()
    #open the dic of compas
    #new_edges = {(name0,name1)}
    if nx.has_path(dg, name0, name1):
        reduced_compa = 0
        new_edges.add((name0,name1))
        #color = 'k'
        
    elif nx.has_path(dg,name1,name0):
        reduced_compa = 1
        new_edges.add((name1,name0))
        #color = 'k'

    elif (name0,name1) in ug.edges:
        reduced_compa = 3
        #color = 'b'

    elif nx.has_path(eg, name0, name1):
        reduced_compa = 2
        #color = 'b'
        
    else:
        
        weight = 0
        
        possible_compas = get_possible_compas(graphs, name0,name1)
        
        if len(possible_compas) == 1:
            compa = 3
        else:
            compa = compare(name0,name1, **kwargs)
            weight = 1
        
        if compa == 4:  #on se donne un moyen de sortir
                raise NameError('sortie de boucle')
        
        if len(str(compa))>1 :
            weight = int(str(compa)[:-1])  #c'était pour le cas sh != ss
            compa = int(str(compa)[-1])
            
        
        while compa not in possible_compas:
            print('Conflict. Choose compa in : ' + str(possible_compas))
            compa = compare(name0,name1, **kwargs)
            if compa == 4:  #on se donne un moyen de sortir
                    raise NameError('sortie de boucle')
            if compa == 91:
#                kwargs['critere'] = critere
                do_the_impossible(graphs, name0, name1, 1,**kwargs)
                possible_compas = get_possible_compas(graphs, name0,name1)

            if compa == 90:
#                kwargs['critere'] = critere                
                do_the_impossible(graphs, name0, name1, 0, **kwargs)
                possible_compas = get_possible_compas(graphs, name0,name1)
                
        #rec the new compa
        #save_new_compa(name0,name1,compa,**kwargs)        
        #convert the compa
        reduced_compa = convert(compa, **kwargs)
        #update the graph:
        update_graphs(graphs, name0, name1, reduced_compa,weight=weight, **kwargs)

        #test la possibilité qu'on ait des arêtes communes à dg et ug        
        ug_edges = set(ug.edges) | invert_edges(set(ug.edges))
        to_remove = set(nx.transitive_closure(dg).edges).intersection(ug_edges)     
        

        if len(to_remove) >0:
            print('pb of intersection :' + str(len(set(nx.transitive_closure(dg).edges).intersection(ug.edges))))            
            ug.remove_edges_from(to_remove)
            dg.remove_edges_from(to_remove)
            eg.remove_edges_from(to_remove)
            refresh_graphs(graphs, **kwargs)



#            else:
            print('edges of intersection removed')

                
        #savings and plots:
        save_graphs(graphs,**kwargs )
        
    return reduced_compa, new_edges



#Codage de Poset-Mergesort:


def Poset_mergesort_im(t, **kwargs):
    #first: take into account all the equality cases
    
    #then launch the recursive loop
    n=len(t) 
    if n<2: 
        return t 
    else: 
        m=n//2 
        return Peeling_im(Poset_mergesort_im(t[:m],**kwargs),Poset_mergesort_im(t[m:], **kwargs), **kwargs)




def rebuild_decomposition(tsafe,tsick,**kwargs):
    new_decomposition = Peeling_im(tsafe,tsick,**kwargs)
    return new_decomposition






def Peeling_im(t1,t2, **kwargs):
    #step 1: fusion des listes ordonnées:
    graphs = get_graphs(**kwargs)
    
    Fail = False
    piles = t1 + t2
    while (not Fail) and (len(piles)>=max(len(t1),len(t2))):
        graphs, piles, Fail = kill_a_pile_im(graphs, piles, **kwargs)
    
    
    print('made the peeling of length: ' + str(len(decomposition_to_names(piles))))
    graphs = save_graphs(graphs, **kwargs)
    
    return piles
    
    #step 2: dépilage: l for an ordered lis

def refresh(piles):
    return [pile for pile in piles if len(pile)>0]

#%
def kill_a_pile_im(graphs, t, **kwargs):

    #dic = get_dic_trans(**kwargs)
    
    print('new murder')

   
    piles = copy.deepcopy(t)
    pointers = {}
    l = len(piles)
    cursors = np.zeros(l, dtype = int)
    lens = np.array([len(pile) for pile in piles])
    while np.sum(lens-cursors == 0) == 0 :
        #find a cpl
        Fail = True
        cpls_of_index = make_cpls_of_index(l)
        for (i,j) in cpls_of_index:
            compa, new_edges = oracle(graphs, piles[i][cursors[i]][0], piles[j][cursors[j]][0], **kwargs)
            #draw_new_edges(graphs[0], new_edges, 'k', **kwargs)
            print("reduced compa :"  + str(compa))
            #cursors and pointers:
            if compa == 0:    
                last_cpl = (i,cursors[i])
                cursors, pointers = update_cursors_and_pointers(i, j, cursors, pointers)
                Fail = False
                break
            elif compa == 1:  
                last_cpl = (j,cursors[j])
                cursors, pointers = update_cursors_and_pointers(j, i, cursors, pointers)
                Fail = False 
                break
            elif compa == 2: #cas d'"égalité" on ramène l'élément ds la liste de gauche
                piles[i][cursors[i]]+=piles[j][cursors[j]]
                del piles[j][cursors[j]]
                Fail = False
                print("equality: relaunch the murder")
                piles = refresh(piles)
                return graphs, piles, Fail
            else:
                pass
        if Fail:
            print("no possible comparisons")
            break
     
    
    
    if Fail:
        return graphs, piles, Fail
    else:        
        #step 3: modify the lists
        #path = []
        #path[0] = last_cpl
        cpl = last_cpl
        while cpl[1] >= 0:
            new_cpl = pointers[cpl]
            piles[cpl[0]]+=piles[new_cpl[0]][new_cpl[1]:]
            piles[new_cpl[0]] = piles[new_cpl[0]][:new_cpl[1]]
            cpl=(new_cpl[0],new_cpl[1]-1)  #on remonte la pile d'un cran
        
        return graphs, refresh(piles) , Fail

##############################################################
##############################################################
######################  decomposition to DiG       ###########
##############################################################
##############################################################
 


def comparison(graphs, name0,name1, **kwargs):
    dg,ug,eg = graphs
    #open the dic of compas
    #new_edges = {(name0,name1)}
    if nx.has_path(dg, name0, name1):
        reduced_compa = 0
    elif nx.has_path(dg,name1,name0):
        reduced_compa = 1
    elif (name0,name1) in ug.edges:
        reduced_compa= 3
    elif (name0,name1) in eg.edges:
        reduced_compa= 2
    else:
        reduced_compa = None
 
    return reduced_compa







def get_transversal_edges(graphs, elementij, listk, **kwargs):    
    lk = len(listk)
    print(kwargs['critere'])
    #get the bigger dominated if exists
    s=0
    stop = False    
    biggest_dominated = []
    index_bigger = lk

    
    while  stop == False and s<lk:
        #print(s)
        #t = lk-s-1 #on parcourt la liste en commençant par le dernier
        compa = comparison(graphs, elementij[0], listk[s][0], **kwargs)
        #print(compas12)
        if compa is not None:
            if compa == 0 :
                biggest_dominated = listk[s]
                index_bigger = s
                stop = True
        s+=1
    
    print('index_bigger :' + str(index_bigger))
    
    if index_bigger > 0:
        upper_compa = 0
        s = index_bigger-1
        while upper_compa not in [1,2,3] and s >=0:
            #print(s)
            upper_compa, _ = oracle(graphs, elementij[0], listk[s][0], **kwargs)
            if upper_compa == 0:
                biggest_dominated = listk[s]
            s-=1
            
        
    transversal_edges = []
    
    for name0 in elementij:
        for name1 in biggest_dominated:
            transversal_edges.append((name0,name1))
    
    return  transversal_edges   




def make_the_DiG(dg2, decomposition, **kwargs):
    graphs = get_graphs(**kwargs)
    dg,ug,eg = graphs
    ldec = len(decomposition)
    

    #vertical edges
    new_vertical_edges = set()
#    color_vertical = 'k'
    
    for list_of_elements in decomposition:
        ll = len(list_of_elements)
        if ll>0:
            for k in range(ll-1):
                trpls=set()
                for name0 in list_of_elements[k]:
                    for name1 in list_of_elements[k+1]:
                        trpls.add((name0,name1,0))
                new_vertical_edges |= trpls
    
    dg.add_weighted_edges_from(new_vertical_edges)   
    save_graphs(graphs,**kwargs )
        
    dg2.add_edges_from([edge[:-1] for edge in new_vertical_edges])
    #draw_new_edges2(dg2, new_vertical_edges,color_vertical, **kwargs)
    if len(set(dg.edges).intersection(set(ug.edges))) >0:
        print('caution: ' + str(len(set(dg.edges).intersection(set(ug.edges)))) + ' edges in dg and ug')
        print('get out of ug')
        ug.remove_edges_from((set(dg.edges).intersection(set(ug.edges))))
    #transversal edges
#    color_transversal = 'b'
    
    for i in range(ldec):
        listi = decomposition[i].copy()
        listi.reverse()
        ll = len(listi)
        for j in range(ll):
            element = listi[j]
            for k in range(ldec):
                if k != i:
                    listk = decomposition[k]
                    print("new links between element " +  str(ll - j) +' of chain ' + str(i) +  ' and chain ' + str(k))
                    transversal_edges = get_transversal_edges(graphs, element, listk, **kwargs)
                    print(transversal_edges)
                    #draw_new_edges2(dg2, transversal_edges, color_transversal, **kwargs)
#                    dg.add_edges_from(transversal_edges) 
                    dg2.add_edges_from(transversal_edges)    
        
    return dg2




##############################################################
##############################################################
###################### tools for  main               #########
##############################################################
##############################################################


def get_suffixe(critere,subgroup,mode):
    if mode != '':
        suffixe = '_' + critere+subgroup+r'_'+mode 
    else:
        suffixe = '_' + critere+subgroup 
    return suffixe



def labelling_mode(**kwargs):
    
    #some paths:
    images_dir = kwargs['images_dir']
    root_cs = kwargs['root_cs']
    mode = kwargs['mode']
    critere = kwargs['critere']
    subgroup = kwargs['subgroup']
    
    if mode != '':
        suffixe = '_' + critere+subgroup+r'_'+mode 
    else:
        suffixe = '_' + critere+subgroup
    
    path_of_decomposition = os.path.join(root_cs, "decomposition" + suffixe + ".pickle" ) 
    path_of_dg2 = os.path.join(root_cs, "poset" + suffixe + r".gpickle" )
    
    #check the previous jobs:

    try:
        dg2 = nx.read_gpickle(path_of_dg2)
        print("DG already found")
        need_to_resort = False
    except:
        print('need to find the dg2')
        need_to_resort = True

    try:
        pickle_in = open(path_of_decomposition,"rb")
        decomposition = pickle.load( pickle_in)
        pickle_in.close()
        
        print("decomposition already found")
        need_to_resort2 = False
    except:
        print("need to find a decomposition")
        need_to_resort2 = True


    #images to label    
    names = sorted(os.listdir(images_dir))
#    names_2 = [[[name]] for name in names]
    #plot the graph:
    #kwargs = draw_nodes_round(**kwargs)
    
    
    while need_to_resort:      
        need_to_resort = False
        while need_to_resort2:
            names_2 = init_with_eg(**kwargs)
            decomposition = Poset_mergesort_im(names_2, **kwargs) 
            print("need to (re)make the decomp :" + str(need_to_resort2))
            print("images to sort : " + str(len(names)))                           
            need_to_resort0, safe_piles, corrupted_piles = check_decomposition(decomposition, **kwargs) #check that chains are ok
#            while need_to_resort0:
#                print(safe_piles)
    #            print(corrupted_piles)
    #            decomposition = rebuild_decomposition(safe_piles, corrupted_piles, **kwargs)
#            need_to_resort0, safe_piles, corrupted_piles = check_decomposition(decomposition, **kwargs) #check that chains are ok

            need_to_resort1 = check_cycles(**kwargs) #check there's no cycle in our labels. no cycle a priori at this point
            need_to_resort2 = need_to_resort0  or need_to_resort1
            
        #save the decomposition
        pickle_out = open(path_of_decomposition,"wb")
        pickle.dump(decomposition, pickle_out)
        pickle_out.close()
        
        #get the naked DiG and amorce the plot:
        #dg2, kwargs = decomposition_to_dg2_and_plot(decomposition, **kwargs)        
        dg2 = decomposition_to_dg2(decomposition)
        
        #end the dig
        print("len(decomposition) = " + str(len(decomposition)) )
                               
        dg2 =  make_the_DiG(dg2, decomposition, **kwargs)

        had_cycles = check_cycles(**kwargs)
        if had_cycles:
            need_to_resort = True
#            need_to_resort2 = True
        else:
            if len(decomposition) > 1:
                print('no cycle')
                longest_path = nx.dag_longest_path(dg2)
                print('longest path of len : ' +str(len(longest_path)))
                true_path = find_path(longest_path,**kwargs)
                need_to_resort = check_path(true_path, **kwargs)
                need_to_resort2 = need_to_resort
                if need_to_resort2:
                    os.remove(path_of_decomposition)
        print("need to finish the DiG :" + str(need_to_resort))

            
#    longest_path = nx.dag_longest_path(dg2)
     
    nx.write_gpickle(dg2, path_of_dg2)
    
    return decomposition, dg2

        
        #check if no cycle:
"""


if cycles!=[]:
    print('kill the cycles')
    return decomposition, DG, []
    #
    #for path in cycles:
    #    path.append(path[0])
    #    print(path)
    #    check_path(path, **kwargs)
    #need_to_resort = True
"""

#if __name__ == '__main__':

def labelling_mode_without_dg2(**kwargs):
    
    #some paths:
    images_dir = kwargs['images_dir']
    root_cs = kwargs['root_cs']
    mode = kwargs['mode']
    critere = kwargs['critere']
    subgroup = kwargs['subgroup']
    
    if mode != '':
        suffixe = '_' + critere+subgroup+r'_'+mode 
    else:
        suffixe = '_' + critere+subgroup
    
    path_of_decomposition = os.path.join(root_cs, "decomposition" + suffixe + ".pickle" ) 
    path_of_dg2 = os.path.join(root_cs, "poset" + suffixe + r".gpickle" )
    
    #check the previous jobs:

    try:
        dg2 = nx.read_gpickle(path_of_dg2)
        print("DG already found")
        need_to_resort = False
    except:
        print('need to find the dg2')
        need_to_resort = True

    try:
        pickle_in = open(path_of_decomposition,"rb")
        decomposition = pickle.load( pickle_in)
        pickle_in.close()
        
        print("decomposition already found")
        need_to_resort2 = False
    except:
        print("need to find a decomposition")
        need_to_resort2 = True


    #images to label    
    names = sorted(os.listdir(images_dir))
#    names_2 = [[[name]] for name in names]
#    names_2 = init_with_eg(**kwargs)
    #plot the graph:
    #kwargs = draw_nodes_round(**kwargs)
    
    while need_to_resort2:
        names_2 = init_with_eg(**kwargs)
        decomposition = Poset_mergesort_im(names_2, **kwargs) 
        print("need to (re)make the decomp :" + str(need_to_resort2))
        print("images to sort : " + str(len(names)))                           
        need_to_resort0, safe_piles, corrupted_piles = check_decomposition(decomposition, **kwargs) #check that chains are ok
#            while need_to_resort0:
#                print(safe_piles)
#            print(corrupted_piles)
#            decomposition = rebuild_decomposition(safe_piles, corrupted_piles, **kwargs)
#            need_to_resort0, safe_piles, corrupted_piles = check_decomposition(decomposition, **kwargs) #check that chains are ok

        need_to_resort1 = check_cycles(**kwargs) #check there's no cycle in our labels. no cycle a priori at this point
        need_to_resort2 = need_to_resort0  or need_to_resort1
        print('need_to_resort0 :' + str(need_to_resort0)  )    
 
#        while need_to_resort0:
#            print(safe_piles)
##            print(corrupted_piles)
##            decomposition = rebuild_decomposition(safe_piles, corrupted_piles, **kwargs)
       
        #save the decomposition
    pickle_out = open(path_of_decomposition,"wb")
    pickle.dump(decomposition, pickle_out)
    pickle_out.close()
        
    
    return decomposition

        
        #check if no cycle:
"""


if cycles!=[]:
    print('kill the cycles')
    return decomposition, DG, []
    #
    #for path in cycles:
    #    path.append(path[0])
    #    print(path)
    #    check_path(path, **kwargs)
    #need_to_resort = True
"""

#if __name__ == '__main__':

#%%Correct the graph: clean nodes to remove

def remove_nodes(ntr,**kwargs):
    dg,ug,eg = get_graphs(**kwargs)
    dg.remove_nodes_from(ntr)
    ug.remove_nodes_from(ntr)
    eg.remove_nodes_from(ntr)
    save_graphs((dg,ug,eg),**kwargs) 
    
def clean_nodes(ntr, **kwargs):
    dg,ug,eg = get_graphs(**kwargs)
    
    dg.remove_nodes_from(ntr)
    ug.remove_nodes_from(ntr)
    eg.remove_nodes_from(ntr)
    
    dg.add_nodes_from(ntr)
    ug.add_nodes_from(ntr)
    eg.add_nodes_from(ntr)
    
    save_graphs((dg,ug,eg),**kwargs)
    
    
    
    