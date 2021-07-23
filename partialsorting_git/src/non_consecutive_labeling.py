# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 18:59:32 2020

@author: Deep Yawner
"""

import random
import sys
import threading
from threading import Thread
#import time
import shutil
import os
import urllib
#import cdsapi
#import netCDF4
#from netCDF4 import Dataset
#import urllib.request
#from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import datetime
from datetime import timedelta, datetime, date
from matplotlib.pyplot import plot_date
from matplotlib.dates import drange
import pandas
import numpy as np
import networkx as nx
import pickle
import subprocess
import pytz

import re
import json 
from datetime import timedelta, datetime, date

path = r"D:\sniff_webcams\datasets\partialsorting_example"
path_code = os.path.join(path, 'src')
os.chdir(path_code)

from utile import *
from utile_to_get_the_graph import *
from fts_utiles_tri5_et_test import *



#%% 

labels_dir = os.path.join(path,"labels_ord") 
images_dir= os.path.join(path,"images") 


images = sorted(os.listdir(images_dir))



#%% Step 1: define critere and mode

level =  'sequence'
superframes = {lbls[name][level] for name in lbls}
 

critere = 'vv'
param = 'vv'
mode = ''
modes = [mode]
subgroup = 'day'

#%%step 1 init the graph (from scratch or pre-labeled images)
nodes = images

dg = nx.DiGraph()
dg.add_nodes_from(nodes)

ug = nx.Graph()
ug.add_nodes_from(nodes)

eg = nx.Graph()
eg.add_nodes_from(nodes)

graphs = (dg, ug, eg)


graphs_names = [os.path.join(graph + '_example' + r'.gpickle') for graph in ["dg","ug","eg"]]
graphs_paths = [os.path.join(labels_dir,graph_name) for graph_name in graphs_names]

nx.write_gpickle(dg, graphs_paths[0])
nx.write_gpickle(ug, graphs_paths[1])
nx.write_gpickle(eg, graphs_paths[2])

#%%if want to look at model prediction at the same time:
"""
PATH = ...
model.load_state_dict(torch.load(PATH))
model = model.to(device)
model.eval()
"""
#%%step 3: restriction to the nodes

images_dataset = images_dir
root_cs = os.path.join(path,r'labels_ord')    
temp_images_dir= images_dir


name_poset_vv = r'poset.pickle'
name_poset_snowsurface =r'poset_snow_surface.pickle'
name_poset_snowheight = r'poset_snow_height.pickle'



#%%
print('labelling of ' + str(len(nodes)) + ' nodes')


kwargs = {'root_cs': labels_dir,
          'critere':  critere,
          'subgroup': 'subgroup',
          'mode' : 'mode',
          'graphs_paths': graphs_paths,
          'images_dir' : images_dir,
          'test_model':False,
          'model': None,
          'device': None
          }

if len(nodes) <= 20:
    decomposition =  labelling_mode(**kwargs)
else:
    decomposition =  labelling_mode_without_dg2(**kwargs)

