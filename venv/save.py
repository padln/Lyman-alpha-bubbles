import numpy as np
import h5py
import os.path

class HdF5Saver:
    def __init__(
            self,
            x_gal,
            x_first_bubble,
            output_dir,
            x_main = 0.0,
            y_main = 0.0,
            z_main = 0.0,
            r_main = 10.0
    ):
        """Saving is done by referencing the first galaxies x-position and
        the first bubbles positin"""
        self.x_gal = x_gal
        self.x_first_bubble = x_first_bubble
        self.output_dir = output_dir

        self.x_main = x_main
        self.y_main = y_main
        self.z_main = z_main
        self.r_main = r_main

        self.create = False
        if not os.path.isdir(output_dir):
            try:
                os.mkdir(output_dir)
            except FileExistsError:
                pass
        self.create_file()

    def create_file(self):
        self.fname = (self.output_dir +
                      f"{self.x_gal:.4f}" + "_" +
                      f"{self.x_first_bubble:.4f}" + "_" +
                      f"{self.x_main:.2f}" + "_" +
                      f"{self.y_main:.2f}" + "_" +
                      f"{self.z_main:.2f}" + "_" +
                      f"{self.r_main:.2f}" + "_" +
                      '.hdf5')
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