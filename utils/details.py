import os

import matplotlib.pyplot as plt
import numpy

def plt_loss(loss_train):
    plt.figure(figsize=(20, 10), dpi=100)
    x=[i+1 for i in range(len(loss_train))]
    y=loss_train
    plt.plot(x, y, 'r-o', label="Loss")
    # plt.yticks(range(0, int(max(loss_train)),int(max(loss_train)/10 )))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlabel("epoch", fontdict={'size': 16})
    plt.ylabel("Loss", fontdict={'size': 16})
    plt.title("LOSS_TRAIN", fontdict={'size': 20})
    plt.show()

def datasave(Loss_train, cl_f1s, processed_dir):
    Loss_train = numpy.array(Loss_train)
    cl_f1s=numpy.array(cl_f1s)
    numpy.save(os.path.join(processed_dir, 'Loss_train.npy'), Loss_train)
    numpy.save(os.path.join(processed_dir, 'cl_f1s.npy'), cl_f1s)