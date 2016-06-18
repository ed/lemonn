import numpy as np
import utils
import nn as neural

def main():
    annotated = 'inputs/0016_17_45_04_1_100to400_goldstd_dbl_bs_cosumtvd.csv'
    metadata = 'inputs/0016_17_45_04_v3_4_9__breath_meta.csv'
    x, y = utils.preprocess(annotated,metadata)
    x = np.array(x)
    nn = neural.NN(learning_rate=.0001, hidden_size=128, input_size=22, output_size=1)
    scores = nn.cross_validate(x, y)
    # this_id = utils.id_gen()
    this_id = utils.simple_id()
    nn.save(this_id+'_weights')
    utils.save_score(this_id+'_stats', scores)


if __name__ == '__main__':
    main()
