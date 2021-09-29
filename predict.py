import json
import tempfile
from pathlib import Path

import cog
import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config
from generate_figures import *

class Predictor(cog.Predictor):
    def setup(self):
        """Load the model"""


    #@cog.input("input", type=Path, help="Audio file")
    def predict(self):
        """Compute prediction"""
        #tflib.init_tf()
        #_G, _D, Gs = pickle.load(open("pretrained_models/network-snapshot-020400.pkl", "rb"))
        #Gs.print_layers()
        #self.Gs = Gs
        results_dir = "prova"
        tflib.init_tf()
        os.makedirs(result_dir, exist_ok=True)
        draw_truncation_trick_figure(os.path.join(config.result_dir, 'truncation-trick.png'), load_Gs(model_place), w=128, h=128, seeds=[np.random.randint(0,100000),np.random.randint(0,100000), np.random.randint(0,100000), np.random.randint(0,100000), np.random.randint(0,12345), np.random.randint(0,12345),np.random.randint(0,12345)], psis=[1,0.9, 0.7, 0.5, 0, -0.5, -1], labels_exist = True)
