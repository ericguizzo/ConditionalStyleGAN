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
        """Compute prediction"""
        tflib.init_tf()
        _G, _D, Gs = pickle.load(open("pretrained_models/network-snapshot-020400.pkl", "rb"))
        Gs.print_layers()
        self.Gs = Gs
        results_dir = "prova"
        for i in range(0,25):
            rnd = np.random.RandomState(None)
            latents = rnd.randn(1, Gs.input_shape[1])
            fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
            images = Gs.run(latents, None, truncation_psi=0.6, randomize_noise=True, output_transform=fmt, allow_soft_placement=True, log_device_placement=True))
            os.makedirs(result_dir, exist_ok=True)
            png_filename = os.path.join(result_dir, 'example-'+str(i)+'.png')
            PIL.Image.fromarray(images[0], 'RGB').save(png_filename)
