#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 19:23:05 2019
makes dic of comparisons without the 0 case
@author: lepetit
"""


import random
import shutil
import os
import matplotlib.pyplot as plt

import numpy as np

import pickle

import re
import json 
import copy

from utile import *
import networkx as nx

#%%

def exclude_from_graph(graph,exclusion_cases):
    nodes_to_remove = []
    for attribute in exclusion_cases:
        for node in graph:
            if graph.nodes[node][attribute] in exclusion_cases[attribute]:
                nodes_to_remove.append(node)
    graph.remove_nodes_from(nodes_to_remove)




def kill_Id_edges(graphs):
    for graph in graphs:
        edges_to_remove =[]
        for edge in graph.edges:
            if edge[0] == edge[1]:
                edges_to_remove.append(edge)
        graph.remove_edges_from(edges_to_remove)


def oracle_from_AMOSlbls(lbls, prev_name, name, param, mode):

    compa = lbls[name]['compa']
    visi = lbls[name]['visi']
    time = lbls[name]['time']
    ground = lbls[name]['ground']
    traces = lbls[name]['old snow_traces']
    noise = lbls[name]['noise']
    
    prev_time = lbls[prev_name]['time']
    prev_ground = lbls[prev_name]['ground']
    
    #hauteur de neige, cas général:
    if param == 'sh':

        if compa == 'snow_up':
            label = 1
        elif compa == 'snow_down':
            label = 0
        elif compa == 'snow_eq':
            label = 2
        elif compa == 'snow_eq3':
            label = 3
        elif compa == 'no_comp':
            label = 3

        #cas particulier 2 changement de couverture: dépendance au mode.
        if  prev_ground != ground:
            #épaisseur + mais surface -:
            if label == 1 and prev_ground == 'white_road' and ground == 'snow_road':
                if mode == 'surface':
                    label = 0
            if label == 1 and prev_ground == 'snow_road' and ground == 'snow_ground':
                if mode == 'surface':
                    label = 0        
            #épaisseur = mais surface +:
            if label in [2,3] and prev_ground == 'snow_road' and ground == 'white_road':
                if mode == 'surface':
                    label = 1 
            if label in [2,3] and prev_ground == 'snow_ground' and ground == 'snow_road':
                if mode == 'surface':
                    label = 1
            #cas d'omission labellisation
            if prev_ground in ['snow_ground','snow_road','white_road'] and ground in ['wet_road','dry_road']:
                    label = 0
            if prev_ground in ['wet_road','dry_road'] and ground in ['snow_ground','snow_road','white_road']:
                    label = 1
            if prev_ground in ['dry_road','wet_road'] and ground in ['wet_road','dry_road']:
                    label = 3
        
        #utilisation de new snow
        if label in [0,2,3] and traces == 'new_snow' and ground in ['snow_road','white_road']:
            if mode == 'height':
                label = 1 
    #        if label in [0] and traces == 'new_snow' and ground == 'snow_road':
    #            if mode == 'height':
    #                label = 1

        #cas particulier 3: en mode height, pas utiliser les changements de surface si snow road
        if  mode == 'height':
            if (prev_ground == 'snow_road') or (ground == 'snow_road'):
                if label != 2:
                    print('get rid of that fluctuation')
                    label = None

    #visibilité, cas général:
    if param == 'vv':
        
        if visi == 'farer':
            label = 1
        elif visi == 'lower':
            label = 0
        elif visi == 'eq':
            label = 2
        elif visi == 'eq3':
            label = 3
        elif visi == 'no_comp':
            label = 3
            
    #cas des changements de luminosité  eq -> eq3:
    if time != prev_time and label == 2:
        label = 3  #dégradation du label pour l'instant
        
    
        
    #cas des bruits:
    if noise in ['blurry','miss_rec', 'other'] and label == 2:
        label = 3  #dégradation du label pour l'instant        
#    if noise in ['surexp'] and and time = ['night'] and label == 2:
#        label = 3  #dégradation du label pour l'instant 
    return label


def oracle_from_TENEBRElbls(lbls, prev_name, name, param, mode):
    
    tresh_sh = 1  #ce qu'il faut de différence pour un label ordinal 
    tresh_vv = 0.25   # seuil en log10 -> 100 m 
    
    sh = lbls[name]['sh']
    vv = lbls[name]['vv']
    time = lbls[name]['time']
    ground = lbls[name]['ground']
    rr6 = lbls[name]['atmo']


    prev_sh = lbls[prev_name]['sh']
    prev_vv = lbls[prev_name]['vv']
    prev_rr6 = lbls[prev_name]['atmo']    
    prev_time = lbls[prev_name]['time']
    prev_ground = lbls[prev_name]['ground']




    
    #hauteur de neige, cas général:
    if param == 'sh':

        if prev_sh >= sh + tresh_sh:
            label = 0
        elif prev_sh + tresh_sh <= sh :
            label = 1
        elif prev_sh >= sh - tresh_sh and prev_sh <= sh + tresh_sh:
            label = 3
        else:
            label = None



    #visibilité, cas général:
    if param == 'vv':
        if np.log10(prev_vv) >= np.log10(vv) + tresh_vv:
            label = 0
        elif np.log10(prev_vv) + tresh_vv <= np.log10(vv) :
            label = 1
        elif np.log10(prev_vv) >= np.log10(vv) - tresh_vv and np.log10(prev_vv) <= np.log10(vv) + tresh_vv:
            label = 3
        else:
            label = None
            
    #on s'assure que les deux images sont de la même scène:
    if lbls[prev_name]['superframe'] != lbls[name]['superframe']:
        label = None

    return label



#Les trois fonctions suivantes sont les mêmes que dans tri_fonction
    

def rebuild(graphs):
    dg, ug, eg = graphs
    for edge in set(dg.edges):
        impact_new_dg_edge(graphs,edge)
    for edge in set(ug.edges):
        impact_new_ug_edge(graphs,edge)

#def impact_neweq(graphs,edge_eq):
#    
#    name0, name1 = edge_eq
#    dg, ug, eg = graphs                
#
#    component = nx.descendants(eg,name1)
#    component.add(name1)
#    nodes_up = set()
#    nodes_down= set()
#    nodes_unr = set()
#    for node in component:
#        nodes_up |= set(dg.predecessors(node))
#        nodes_down |= set(dg.successors(node))
#        nodes_unr |= set(ug.neighbors(node))
#    
#    new_dg_edges = set()
#    new_ug_edges = set()
#    
#    new_dg_edges |= {(node0,node1) for node0 in nodes_up for node1 in component}
#    new_dg_edges |= {(node0,node1) for node0 in component for node1 in nodes_down}
#    new_ug_edges |= {(node0,node1) for node0 in component for node1 in nodes_unr}
#
#    dg.add_edges_from(new_dg_edges)
#    ug.add_edges_from(new_ug_edges)
#
#    return new_dg_edges
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

def make_dg_from_eg(dg, eg):  #get the graph of true labels

    a = copy.deepcopy(dg)
#    get_naked([a])
    b = copy.deepcopy(eg).to_directed()
    b = nx.compose( b ,   nx.reverse(b) )   #symmetrization of b

    complete_dg = nx.compose(a, b)
    
    return complete_dg

def comprise_in(x,b0,b1):
    if x>b0 and x<= b1:
        return True

def get_sdg_from_TENEBRElbls(graphs,param,mode, tresh, size_by_interval = 50000): #pas codé pour vv
    
    if param == 'sh':
        dg = graphs[0]
        ug = graphs[1]
        eg = graphs[2]    
        names = sorted(list(dg.nodes))
    
        #adding edges:
        new_dg_edges = set()
        new_ug_edges = set()
        new_eg_edges = set()
        tresh_sh = tresh
        
        snow_names0 = [name for name in names if dg.nodes[name][param]<=2]
        snow_names1 = [name for name in names if comprise_in(dg.nodes[name][param],1,3)]
        snow_names2 = [name for name in names if comprise_in(dg.nodes[name][param],2,7)]
        snow_names5 = [name for name in names if dg.nodes[name][param] > 5]
    
        dic = dg.nodes
        
        for snow_names in [snow_names0,snow_names1,snow_names2,snow_names5]:
            print('new intervall')
            for name in snow_names:
                frame = dic[name]['frame']
    #            print('newb')
                tm = dic[name]['time']
                for name2 in snow_names:
                    if dic[name2]['frame'] == frame:
                         if dic[name2]['time'] == tm:
                             if dic[name][param] > tresh_sh + dic[name2][param]  and dic[name][param] < 2*tresh_sh + dic[name2][param]:
                                 new_dg_edges |= {(name,name2)}
            print(len(new_dg_edges))
            new_dg_edges = set(random.sample(new_dg_edges,size_by_interval))
            print(len(new_dg_edges))
            
            dg.add_edges_from(new_dg_edges)
            ug.add_weighted_edges_from(new_ug_edges)
            eg.add_weighted_edges_from(new_eg_edges)
    
        rebuild(graphs) #attention: upgrader si ug utile


    if param == 'vv':
        dg = graphs[0]
        ug = graphs[1]
        eg = graphs[2]    
        names = sorted(list(dg.nodes))
    
        #adding edges:
        new_dg_edges = set()
        new_ug_edges = set()
        new_eg_edges = set()
        tresh_vv = tresh
        
        POM_names0 = [name for name in names if dg.nodes[name][param]<=200]
        POM_names1 = [name for name in names if comprise_in(dg.nodes[name][param],100,500)]
        POM_names2 = [name for name in names if comprise_in(dg.nodes[name][param],250,1000)]
        POM_names3 = [name for name in names if comprise_in(dg.nodes[name][param],500,5000)]
        POM_names4 = [name for name in names if comprise_in(dg.nodes[name][param],2500,10000)]
        POM_names5 = [name for name in names if dg.nodes[name][param] > 5000]
    
        dic = dg.nodes
        
        for POM_names in [POM_names0 ,POM_names1,POM_names2,POM_names3, POM_names4,POM_names5]:
            print('new intervall')
            for name in POM_names:
                frame = dic[name]['frame']
    #            print('newb')
                tm = dic[name]['time']
                for name2 in POM_names:
                    if dic[name2]['frame'] == frame:
                         if dic[name2]['time'] == tm:
                             if np.log2(dic[name][param]) > tresh_vv + np.log2(dic[name2][param]) and np.log2(dic[name][param]) < 2*tresh_vv + np.log2(dic[name2][param]):
                                 new_dg_edges |= {(name,name2)}
            l = len(new_dg_edges)
            print(len(new_dg_edges))
            new_dg_edges = set(random.sample(new_dg_edges,min(size_by_interval,l)))
            print(len(new_dg_edges))
            
            dg.add_edges_from(new_dg_edges)
            ug.add_weighted_edges_from(new_ug_edges)
            eg.add_weighted_edges_from(new_eg_edges)
    
        rebuild(graphs) #attention: upgrader si ug utile
    

    return graphs






def get_sdg_from_TENEBRElbls(graphs,param,mode, tresh, size_by_interval = 50000): #pas codé pour vv
    
    if param == 'sh':
        dg = graphs[0]
        ug = graphs[1]
        eg = graphs[2]    
        names = sorted(list(dg.nodes))
    
        #adding edges:
        new_dg_edges = set()
        new_ug_edges = set()
        new_eg_edges = set()
        tresh_sh = tresh
        
        snow_names0 = [name for name in names if dg.nodes[name][param]<=2]
        snow_names1 = [name for name in names if comprise_in(dg.nodes[name][param],1,3)]
        snow_names2 = [name for name in names if comprise_in(dg.nodes[name][param],2,7)]
        snow_names5 = [name for name in names if dg.nodes[name][param] > 5]
    
        dic = dg.nodes
        
        for snow_names in [snow_names0,snow_names1,snow_names2,snow_names5]:
            print('new intervall')
            for name in snow_names:
                frame = dic[name]['frame']
    #            print('newb')
                tm = dic[name]['time']
                for name2 in snow_names:
                    if dic[name2]['frame'] == frame:
                         if dic[name2]['time'] == tm:
                             if dic[name][param] > tresh_sh + dic[name2][param]  and dic[name][param] < 2*tresh_sh + dic[name2][param]:
                                 new_dg_edges |= {(name,name2)}
            print(len(new_dg_edges))
            new_dg_edges = set(random.sample(new_dg_edges,size_by_interval))
            print(len(new_dg_edges))
            
            dg.add_edges_from(new_dg_edges)
            ug.add_weighted_edges_from(new_ug_edges)
            eg.add_weighted_edges_from(new_eg_edges)
    
        rebuild(graphs) #attention: upgrader si ug utile


    if param == 'vv':
        dg = graphs[0]
        ug = graphs[1]
        eg = graphs[2]    
        names = sorted(list(dg.nodes))
    
        #adding edges:
        new_dg_edges = set()
        new_ug_edges = set()
        new_eg_edges = set()
        tresh_vv = tresh
        
        POM_names0 = [name for name in names if dg.nodes[name][param]<=200]
        POM_names1 = [name for name in names if comprise_in(dg.nodes[name][param],100,500)]
        POM_names2 = [name for name in names if comprise_in(dg.nodes[name][param],250,1000)]
        POM_names3 = [name for name in names if comprise_in(dg.nodes[name][param],500,5000)]
        POM_names4 = [name for name in names if comprise_in(dg.nodes[name][param],2500,10000)]
        POM_names5 = [name for name in names if dg.nodes[name][param] > 5000]
    
        dic = dg.nodes
        
        for POM_names in [POM_names0 ,POM_names1,POM_names2,POM_names3, POM_names4,POM_names5]:
            print('new intervall')
            for name in POM_names:
                frame = dic[name]['frame']
    #            print('newb')
                tm = dic[name]['time']
                for name2 in POM_names:
                    if dic[name2]['frame'] == frame:
                         if dic[name2]['time'] == tm:
                             if np.log2(dic[name][param]) > tresh_vv + np.log2(dic[name2][param]) and np.log2(dic[name][param]) < 2*tresh_vv + np.log2(dic[name2][param]):
                                 new_dg_edges |= {(name,name2)}
            l = len(new_dg_edges)
            print(len(new_dg_edges))
            new_dg_edges = set(random.sample(new_dg_edges,min(size_by_interval,l)))
            print(len(new_dg_edges))
            
            dg.add_edges_from(new_dg_edges)
            ug.add_weighted_edges_from(new_ug_edges)
            eg.add_weighted_edges_from(new_eg_edges)
    
        rebuild(graphs) #attention: upgrader si ug utile
    

    return graphs



def get_sdg_from_AMOSlbls(graphs, lbls,param,mode, level = 'superframe'):
    names = sorted(list(lbls.keys()))
    dg = graphs[0]
    ug = graphs[1]
    eg = graphs[2]
    

    intraframe = True
    
    #adding edges:
    new_dg_edges = set()
    new_ug_edges = set()
    new_eg_edges = set()

    # in the time order
        
    for i in range(1,len(names)):
        prev_name = names[i-1]
        name= names[i]
        if lbls[name][level] == lbls[prev_name][level] or not intraframe:
            label = oracle_from_AMOSlbls(lbls, prev_name, name, param, mode)
            if label == 0:
                new_dg_edges |= {(prev_name, name,0)}
            elif label == 1:
                new_dg_edges |= {(name, prev_name,0)}
            elif label == 2:
                new_ug_edges |= {(name, prev_name,0)}       
                new_eg_edges |= {(name, prev_name,0)}
            elif label == 3:
                new_ug_edges |= {(name, prev_name,0)}
                
    #        if dg.nodes[name]['visi'] == 'no_comp' and label != None:
    #            print(label)

    
    dg.add_weighted_edges_from(new_dg_edges)
    ug.add_weighted_edges_from(new_ug_edges)
    eg.add_weighted_edges_from(new_eg_edges)
    
    #ajouter l'info d'égalité
    rebuild(graphs) #attention: upgrader qd ug utile
    kill_Id_edges(graphs)

    return graphs



def get_new_graphs(lbls):
    names = sorted(list(lbls.keys()))
    
    dg1 = nx.DiGraph()
    ug1 = nx.Graph()
    eg1 = nx.Graph()
    
    #adding nodes:
    dg1.add_nodes_from(names)
    ug1.add_nodes_from(names)
    eg1.add_nodes_from(names)
 
    #adding attributes on dg:
    nx.set_node_attributes(dg1,lbls)
    nx.set_node_attributes(ug1,lbls)
    nx.set_node_attributes(eg1,lbls) 
    
    return [dg1, ug1, eg1]





def new_get_wdg_from_AMOSlbls(graphs, param ,mode, intra_frame = False, nb_by_frame = 100, level = 'superframe'):
    dg = graphs[0]
    ug = graphs[1]
    eg = graphs[2]    
    
    frames = {dg.nodes[name][level] for name in dg}
    names = sorted(list(dg.nodes))
    lbls = dg.nodes
#    print(frames)
    #ug = graphs[1]
    #eg = graphs[2]
        
    #initialization:
    new_dg_edges = set()
    new_ug_edges = set()
    new_eg_edges = set()

    
    if param == 'sh':

        for frame in frames:
            new_frame_edges = set()
            
            
            names_to_pick = names  #ON NE VEUT PAS DEUX FOIS LA MËME IMAGE
            
            for name in names:
            
                
                
                if lbls[name][level] == frame and lbls[name]['ground'] in ['snow_ground','snow_road','white_road']:
                    for name2 in names_to_pick:
                        if (lbls[name2][level] == frame) or not intra_frame:  #intra_frame précise si on cherche dans ou en dehors de la frame
                           if lbls[name2]['ground'] in ['dry_road','wet_road']:                
                                new_frame_edges |= {(name, name2, 1)}
#                                names_to_pick.remove(name2)
            
            l = len(new_frame_edges)
            new_frame_edges = random.sample(new_frame_edges, min(nb_by_frame, l))
            new_dg_edges |= set(new_frame_edges)
#            print(len(new_dg_edges))
        
        if mode == 'surface':
            for frame in frames:
                new_frame_edges = set()
                
                
                names_to_pick = names  #ON NE VEUT PAS DEUX FOIS LA MËME IMAGE
                
                for name in names:
                
                    
                    
                    if lbls[name][level] == frame and lbls[name]['ground'] in ['snow_road','white_road']:
                        for name2 in names_to_pick:
                            if (lbls[name2][level] == frame) or not intra_frame:  #intra_frame précise si on cherche dans ou en dehors de la frame
                               if lbls[name2]['ground'] in ['snow_ground', 'no_snow_road']:                
                                    new_frame_edges |= {(name, name2, 1)}
    #                                names_to_pick.remove(name2)
                
                l = len(new_frame_edges)
                new_frame_edges = random.sample(new_frame_edges, min(nb_by_frame, l))
                new_dg_edges |= set(new_frame_edges)
    #            print(len(new_dg_edges))
            

            
            for frame in frames:
                new_frame_edges = set()
                
                for name in names:
                    if lbls[name][level] == frame and lbls[name]['ground'] in ['white_road']:
                        for name2 in names:
                            if (lbls[name2][level] == frame) or not intra_frame:  #intra_frame précise si on cherche dans ou en dehors de la frame
                               if lbls[name2]['ground'] in ['snow_road']:                
                                    new_frame_edges |= {(name, name2, 1)}
                l = len(new_frame_edges)                    
                new_frame_edges = random.sample(new_frame_edges, min(nb_by_frame,l))
                new_dg_edges |= set(new_frame_edges)                            
#            print(len(new_dg_edges))
            
        dg.add_weighted_edges_from(new_dg_edges)     

        #case of ug/eg

        for frame in frames:
            names_frame = [name for name in names if lbls[name][level] == frame]            
            new_frame_edges = {(n0,n1) for n0 in names_frame if (lbls[n0]['sh']==0) for n1 in names_frame if (lbls[n1]['sh']==0) }
#            print(len(new_frame_edges))
            l = len(new_frame_edges)   
            new_frame_edges = random.sample(new_frame_edges, min(nb_by_frame,l))
            new_ug_edges |= set(new_frame_edges)   
            new_eg_edges |= set(new_frame_edges)                          
            print(len(new_ug_edges))
            print(len(new_eg_edges))
            
        ug.add_edges_from(new_ug_edges)                          
        eg.add_edges_from(new_eg_edges)                                   

    if param == 'vv':

        for frame in frames:
            new_frame_edges = set()
            
            for name in names:
                if lbls[name][level] == frame and lbls[name]['atmo'] in ['no_precip']:
                    for name2 in names:
                        if (lbls[name2][level] == frame) or ( (lbls[name2][level] != frame) and not intra_frame):  #intra_frame précise si on cherche dans ou en dehors de la frame
                           if lbls[name2]['atmo'] in ['precip','rain','fog','snow','fog_or_snow']:                
                                new_frame_edges |= {(name, name2, 1)}
            l = len(new_frame_edges)                    
            new_frame_edges = random.sample(new_frame_edges, min(nb_by_frame,l))
            new_dg_edges |= set(new_frame_edges)                          
            print(len(new_dg_edges))         
        dg.add_weighted_edges_from(new_dg_edges)           
        
        
    return graphs



def get_wug_from_AMOSlbls(graphs, param ,mode, intra_frame = False, nb_by_frame = 100):
    
    ug = graphs[1]    
    frames = {ug.nodes[name]['superframe'] for name in ug}
    names = sorted(list(ug.nodes))
    lbls = ug.nodes

    new_ug_edges = set()

    
    if param == 'sh':

        for frame in frames:
            new_frame_edges = set()
            
            
            names_to_pick = names  #ON NE VEUT PAS DEUX FOIS LA MËME IMAGE
            
            for name in names:
            
                
                
                if lbls[name]['superframe'] == frame and lbls[name]['ground'] in ['dry_road','wet_road'] and lbls[name]['old snow_traces'] !='ground':
                    for name2 in names_to_pick:
                        if (lbls[name2]['superframe'] == frame) or not intra_frame:  #intra_frame précise si on cherche dans ou en dehors de la frame
                           if lbls[name2]['ground'] in ['dry_road','wet_road'] and lbls[name2]['old snow_traces'] !='ground':                
                                new_frame_edges |= {(name, name2,1)}
#                                names_to_pick.remove(name2)
            
            l = len(new_frame_edges)
            new_frame_edges = random.sample(new_frame_edges, min(nb_by_frame, l))
            new_ug_edges |= set(new_frame_edges)
#            print(len(new_dg_edges))


            
        ug.add_weighted_edges_from(new_ug_edges)
        kill_Id_edges(graphs)

    if param == 'vv':

        for frame in frames:
            new_frame_edges = set()
            
            for name in names:
                if lbls[name]['superframe'] == frame and lbls[name]['atmo'] in ['no_precip']:
                    for name2 in names:
                        if (lbls[name2]['superframe'] == frame) or not intra_frame:  #intra_frame précise si on cherche dans ou en dehors de la frame
                           if lbls[name2]['atmo'] in ['no_precip']:                
                                new_frame_edges |= {(name, name2,1)}
            l = len(new_frame_edges)                    
            new_frame_edges = random.sample(new_frame_edges, min(nb_by_frame,l))
            new_ug_edges |= set(new_frame_edges)                          
#            print(len(new_dg_edges))

        ug.add_weighted_edges_from(new_ug_edges)
        kill_Id_edges(graphs)
         
    return graphs


def add_self_edges(graph):
    for node in graph:
        graph.add_edge(node,node)


#%%
def touches_snow_road(dg,e):  #says if an edge has one of its node in snow_road        
    if dg.nodes[e[0]]['ground']=='snow_road' or dg.nodes[e[1]]['ground']=='snow_road':
        print('edge detroyed')
        return True
    else:
        return False
    
def fill_graph_from_splitted(graphs,splitted_dir, ext = 'vvday',forbidden_weights= None, only_eg = False, except_snow_road = False):
    sequences = os.listdir(splitted_dir)
    
    tdg,tug,teg = graphs
    
    for sequence in sequences:
        try:
            path = os.path.join(splitted_dir,sequence,'labels_ord')
            dg_path = os.path.join(path, 'dg_' + ext + '.gpickle')
            ug_path = os.path.join(path, 'ug_' + ext + '.gpickle')
            eg_path = os.path.join(path, 'eg_' + ext + '.gpickle')
            
            local_dg = nx.read_gpickle(dg_path) 
            local_ug = nx.read_gpickle(ug_path)
            local_eg =nx.read_gpickle(eg_path)
#            print(len(local_ug.edges))
            
            if forbidden_weights is not None:
                dg_edges_to_remove = [e for e in local_dg.edges if local_dg.edges[e].get('weight') in forbidden_weights]
                ug_edges_to_remove = [e for e in local_ug.edges if local_ug.edges[e].get('weight') in forbidden_weights]
                eg_edges_to_remove = [e for e in local_eg.edges if local_eg.edges[e].get('weight') in forbidden_weights]
                
                local_dg.remove_edges_from(dg_edges_to_remove)
                local_ug.remove_edges_from(ug_edges_to_remove)
                local_eg.remove_edges_from(eg_edges_to_remove)

            if except_snow_road:
                dg_edges_to_remove2 = [e for e in local_dg.edges if touches_snow_road(local_dg,e)]
                ug_edges_to_remove2 = [e for e in local_ug.edges if touches_snow_road(local_dg,e)]
                local_dg.remove_edges_from(dg_edges_to_remove2)
                local_ug.remove_edges_from(ug_edges_to_remove2)


            if not only_eg:
                tdg = nx.compose(tdg, local_dg)
                tug = nx.compose(tug, local_ug)    
            

                
            
            teg = nx.compose(teg, local_eg) 
            
            
        except:
            pass
#            print('nothing in sequence: ' + str(sequence))
            
    return (tdg, tug, teg)
    
        
def make_graph_from_splitted(splitted_dir, ext = 'vvday'):
    sequences = os.listdir(splitted_dir)
    
    tdg = nx.DiGraph()
    tug = nx.Graph()
    teg = nx.Graph()
    
    for sequence in sequences:
        try:
            path = os.path.join(splitted_dir,sequence,'labels_ord')
            dg_path = os.path.join(path, 'dg_' + ext + '.gpickle')
            ug_path = os.path.join(path, 'ug_' + ext + '.gpickle')
            eg_path = os.path.join(path, 'eg_' + ext + '.gpickle')
            
            local_dg = nx.read_gpickle(dg_path) 
            local_ug = nx.read_gpickle(ug_path)
            local_eg = nx.read_gpickle(eg_path)

            tdg = nx.compose(tdg, local_dg)
            tug = nx.compose(tug, local_ug)    
            teg = nx.compose(teg, local_eg) 
#            print(len(tdg.edges))
            
        except:
            print('nothing in sequence: ' + str(sequence))
            
    return (tdg, tug, teg)
            

#%% for qual_graph

def init_qual_graph(lbls):
    names = sorted(list(lbls.keys()))
    
    graph = nx.Graph()
    
    #adding nodes:
    graph.add_nodes_from(names)

    #adding attributes on dg:
    nx.set_node_attributes(graph,lbls)
    
    return graph


def fill_qual_graph(graph):
    level = 'superframe'
    long = 3
    names = sorted(list(graph.nodes))
    lbls = graph.nodes
#    print(frames)
    #ug = graphs[1]
    #eg = graphs[2]
        
    #initialization:
    new_graph_edges = set()
    L = len(names)
    for i in range(L):
        for k in range(i+1,min(i+long,L)):
            if lbls[names[i]][level] == lbls[names[k]][level]:
                if lbls[names[i]]['time'] == lbls[names[k]]['time'] or lbls[names[k]]['time'] == 'inter'  or lbls[names[i]]['time'] == 'inter':
                    new_graph_edges|= {(names[i],names[k])}
    

    print(str(len(new_graph_edges)) +' edges in the graph')
    graph.add_edges_from(new_graph_edges)
    




###################################################################################
###################################################################################
    ############################ learning with graph #########################



#%%
