import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(cm, name, cmap=plt.cm.Greens):
    fig, ax = plt.subplots()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion matrix ' + name)
    plt.colorbar()
    plt.xticks(np.arange(4), ['1','2', '3', '4'], rotation=45)
    plt.yticks(np.arange(4), ['1','2', '3', '4'])
    plt.tight_layout()

    width, height = cm.shape

    for x in range(width):
        for y in range(height):
            plt.annotate(str(cm[x][y]), xy=(y, x), 
                        horizontalalignment='center',
                        verticalalignment='center')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('../results/cm.png')
    plt.close('all')
