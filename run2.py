
import numpy as np
import tensorflow as tf

import model

if __name__ == "__main__":

    print("Making level configs...")
    level_configs = model.default_level_configs()

    print("Making filter variables...")
    filters = model.make_filters(tf.get_default_graph(), level_configs)

    print("Done")
