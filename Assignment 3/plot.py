import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
    
def attention_heatmap(input_word, heatmap_data):

    mats = []
    dec_inputs = []

    for data in heatmap_data:
        dec_ind, attn  = data[0], data[1]
        mats.append(attn.reshape(-1)[:len(input_word)])
        dec_inputs.append(dec_ind)
    
    attention_mat = np.array(mats)

    fig, ax = plt.subplots()
    ax.imshow(attention_mat)

    ax.set_xticks(np.arange(attention_mat.shape[1]))
    ax.set_yticks(np.arange(attention_mat.shape[0]))

    ax.set_yticklabels([inp if inp != '\n' else "<e>" for inp in dec_inputs], fontproperties = FontProperties(fname = "Fonts/nirmala.ttf"))
    ax.set_xticklabels([char for char in input_word])

    ax.tick_params(labelsize = 15)
    ax.tick_params(axis = 'x', labelrotation =  45)

    return fig

# attention_heatmap(None, None)