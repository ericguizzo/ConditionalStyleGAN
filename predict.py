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

class Predictor(cog.Predictor):
    def setup(self):
        """Load the model"""

    #@cog.input("input", type=Path, help="Audio file")
    def predict(self):
        """Compute f0 plot"""
        print ("heeey")
