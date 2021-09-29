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
        tflib.init_tf()
        _G, _D, Gs = pickle.load(open("pretrained_models/network-snapshot-020400.pkl", "rb"))
        Gs.print_layers()
        self.Gs = Gs
        results_dir = "prova"
