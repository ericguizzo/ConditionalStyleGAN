import json
import tempfile
from pathlib import Path
import shutil

import cog
from crepe.core import *
import crepe
from scipy.io import wavfile

class Predictor(cog.Predictor):
    def setup(self):
        """Load the model"""

    #@cog.input("input", type=Path, help="Audio file")
    def predict(self):
        """Compute f0 plot"""
        print ("heeey")
