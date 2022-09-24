from os.path import join
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import util
import visual_words
import visual_recog
from opts import get_opts
import time


def main():
    opts = get_opts()

    # Q1.1
    #img_path = join(opts.data_dir, 'kitchen/sun_aasmevtpkslccptd.jpg')
    #img = Image.open(img_path)
    #img = np.array(img).astype(np.float32) / 255
    #filter_responses = visual_words.extract_filter_responses(opts, img)
    #util.display_filter_responses(opts, filter_responses)

    # Q1.2
    # Q1.3
    #img_path = join(opts.data_dir, 'kitchen/sun_aasmevtpkslccptd.jpg')
    #img = Image.open(img_path)
    #img = np.array(img).astype(np.float32)/255
    #dictionary = np.load(join(opts.out_dir, 'dictionary.npy'))
    #wordmap = visual_words.get_visual_words(opts, img, dictionary)
    #util.visualize_wordmap(wordmap)

    # Q2.1-2.4
    breakpoint()
    path = Path(opts.out_dir)
    path.mkdir(exist_ok=False)
    hyperparams_file = path / "hyperparameters.txt"
    with open(hyperparams_file,'w') as f:
        res = ""
        res = res+"filter_scales: "+str(opts.filter_scales) + "\n"
        res = res +"K: "+str(opts.K)+"\n"
        res = res +"alpha: "+str(opts.alpha)+"\n"
        res = res +"L: "+str(opts.L)+"\n"
        f.write(res)

    start_time = time.time()
    n_cpu = util.get_num_CPU()
    visual_words.compute_dictionary(opts, n_worker=n_cpu)

    n_cpu = util.get_num_CPU()
    visual_recog.build_recognition_system(opts, n_worker=n_cpu)

    # Q2.5
    n_cpu = util.get_num_CPU()
    conf, accuracy = visual_recog.evaluate_recognition_system(opts, n_worker=n_cpu)

    np.savetxt(join(opts.out_dir, 'confmat.csv'), conf, fmt='%d', delimiter=',')
    np.savetxt(join(opts.out_dir, 'accuracy.txt'), [accuracy], fmt='%g')
    end_time = time.time()
    print(end_time - start_time)


if __name__ == '__main__':
    main()
