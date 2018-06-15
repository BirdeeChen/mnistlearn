# import necessary libraries
import os
import gzip
import struct
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# main function
def read_data_set(filename, one_hot = False):
    with gzip.open(filename, 'rb') as f:
        f_content = f.read()
        
        if 'images' in filename:
            magic_num, num_samples, dim_row, dim_col = struct.unpack('>IIII', f_content[:16])
            sample_len = dim_row * dim_col
            imgs_gen = struct.iter_unpack('>{}'.format('B' * sample_len), f_content[16:])
            # perform feature normalization
            return np.array([np.array(img) / 255 for img in imgs_gen])

        elif 'labels' in filename:
            magic_num, num_samples = struct.unpack('>II', f_content[:8])
            labels = struct.iter_unpack('>B', f_content[8:])
            if not one_hot:
                return np.array([l[0] for l in labels])
            else:
                # perform one-hot transformation
                return pd.get_dummies(pd.Series([l for l in labels])).values.astype(float)

if __name__ == '__main__':
    # debug
    trImage = read_data_set('temp/train-images-idx3-ubyte.gz')
    trImage_0 = trImage[5000]
    # print(trImage_0)
    trImage_0 = np.resize(trImage_0, (28, 28))
    print(trImage.shape)
    trLabel = read_data_set('temp/train-labels-idx1-ubyte.gz', one_hot=True)
    trLabel_false = read_data_set('temp/train-labels-idx1-ubyte.gz')
    print(trLabel.shape)
    print(trLabel[5000])
    # print(trLabel_false.shape)
    print(trLabel_false[5000])
    plt.imshow(trImage_0, cmap = 'Greys_r')
    plt.show()



