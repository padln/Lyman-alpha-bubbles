import numpy as np
import h5py
import os.path

class HdF5Saver:
    def __init__(self,x_gal, x_first_bubble, output_dir):
        """Saving is done by referencing the first galaxies x-position and
        the first bubbles positin"""
        self.x_gal = x_gal
        self.x_first_bubble = x_first_bubble
        self.output_dir = output_dir
        self.create = False
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        self.create_file()

    def create_file(self):
        self.fname = self.output_dir + str(
            self.x_gal) + str(self.x_first_bubble) + '.hdf5'
        self.f = h5py.File(self.fname, 'a')

        self.created=True

    def save_attrs(self, dict_gal):
        for (nam, val) in dict_gal.items():
            self.f.attrs[nam] = val

    def save_datasets(self, dict_dat):
        for (nam, val) in dict_dat.items():
            self.f.create_dataset(
                nam,
                dtype = "float",
                data = val
            )

    def close_file(self):
        self.f.close()