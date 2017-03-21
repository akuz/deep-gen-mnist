
import tensorflow as tf

class LevelConfig(object):

    def __init__(self, ndim, filter_size, filter_rate):

        self.ndim = ndim
        self.filter_size = filter_size
        self.filter_rate = filter_rate

def default_level_configs():

    level_configs = []

    # 8x8 tiles
    level_configs.append(
        LevelConfig(
            ndim=64,
            filter_size=2, 
            filter_rate=4))

    # 4x4 tiles
    level_configs.append(
        LevelConfig(
            ndim=32, 
            filter_size=2, 
            filter_rate=2))

    # 2x2 tiles
    level_configs.append(
        LevelConfig(
            ndim=16, 
            filter_size=2, 
            filter_rate=1))

    # gradations of the colour
    level_configs.append(
        LevelConfig(
            ndim=16, 
            filter_size=1, 
            filter_rate=1))

    # last level is an image
    level_configs.append(
        LevelConfig(
            ndim=256, 
            filter_size=None, 
            filter_rate=None))

    return level_configs

def make_filters(graph, level_configs):

    filters = []

    with graph.as_default():

        for i in range(len(level_configs)-1):

            with tf.variable_scope('level_{}'.format(i)):

                this_level_config = level_configs[i]
                next_level_config = level_configs[i+1]

                w = tf.get_variable(
                    shape=(
                        this_level_config.filter_size,
                        this_level_config.filter_size,
                        next_level_config.ndim,
                        this_level_config.ndim),
                    name='w')

                print(w)

                filters.append(w)

    return filters
