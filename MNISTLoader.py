import tensorflow as tf 
import numpy as np


class MNISTLoader:
    @staticmethod
    def load_data(data_path):
        read_data = np.loadtxt(data_path, delimiter=',', dtype=np.float32)
        y = read_data[:, 0]
        x = read_data[:, 1:]

        return (x, y)
    
    @staticmethod
    def one_hot_encode(y):   
        encode_data = tf.one_hot(y, depth=10)

        return encode_data
