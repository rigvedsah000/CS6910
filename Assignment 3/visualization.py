import numpy as np
from IPython.display import HTML
from IPython.display import display
import ast

def softmax(x):
    denom = sum([np.exp(p) for p in x])
    return [np.exp(p) / denom for p in x]

def cstr(s, color = 'black'):
    return "<text style=color:#000;padding-top:1.5px;padding-bottom:1.5px;padding-left:2.5px;padding-right:2.5px;background-color:{}>{} </text>".format(color, s)

def get_clr(value, mode):
    if(mode == 'l'):
        colors = ['#85c2e1', '#89c4e2', '#95cae5', '#99cce6', '#a1d0e8', '#b2d9ec', '#baddee', '#c2e1f0', '#eff7fb', '#f9e8e8', '#f9e8e8', '#f9d4d4', '#f9bdbd', '#f8a8a8', '#f68f8f', '#f47676', '#f45f5f', '#f34343', '#f33b3b', '#f42e2e']
        value = int((value * 100) / 5)
        return colors[value]
    else:
        # colors = ['#FFFFFF','#DFFFFF','#BFFFFF','#9FFFFF','#7FFFFF','#5FFFFF','#3FFFFF','#03FFFF','#00EFFF','#00DFFF','#00CFFF','#00BFFF','#00AFFF','#009FFF']
        # factor = 0.07142857142857142
        # color_index = int(value/factor)
        # return colors[color_index]
        colors = ['#ffffff', '#ecf7fb', '#daeff7', '#c7e7f3', '#b5dfef', '#a2d7eb', '#90cfe7', '#7dc7e3', '#6abfdf', '#58b7db', '#46afd7']
        value = int((value * 100) / 10)
        return colors[value]

def visualize_c(dec_char, text_colours):
    if (dec_char == "<e>"):
      display(HTML(''.join([cstr(ti, color = ci) for ti, ci in text_colours]) + " <b> &emsp; &lt; e &gt; </b>  &emsp; &nbsp; "))
    else:
      display(HTML(''.join([cstr(ti, color = ci) for ti, ci in text_colours]) + " <b> &emsp; {}</b>  &emsp; &emsp; ".format(dec_char)))

def visualize_l(dec_seq, prob):
    text_colours = []

    for c, p in zip(dec_seq, prob): 
        text = (c, get_clr(p, 'l'))
        text_colours.append(text)
    
    display(HTML(''.join([cstr(ti, color = ci) for ti, ci in text_colours])))

def visualize_connectivity(N):

    # Reading from conv_vis file
    with open("conn_vis.txt", "r", encoding='utf-8') as filepointer:
        
        lines = filepointer.readlines()

        i = 0
        words_visualized = 0

        while i < len(lines) and  words_visualized< N:
            line = lines[i]
            
            if line[:4] == "Next":
                words_visualized += 1
                i += 1
                continue

            if line[:4] != "Next": 
                true_word, dec_char_len = line.split('\t') 
                dec_word_len = int(dec_char_len)
                i += 1

                true_word_array = [c for c in true_word]

                for j in range(dec_word_len):
                    line = lines[i]
                    line = line.split('\t')
  
                    dec_char = line[0]
                    text_colours = []

                    prob = []
                    for prob_index in range(1,len(true_word)+1) :
                        p = float(line[prob_index])
                        prob.append(p)

                    line = softmax(prob)

                    
                    for prob_index in range(len(true_word)) :
                        p = float(line[prob_index])

                        true_char = true_word_array[prob_index]
                        text= (true_char, get_clr(p, 'c') )
                        text_colours.append(text)

                    visualize_c(dec_char, text_colours)
            
                    i += 1

            print("\n\n")

def visualize_lstm(N, neuron):

    for i in range(N):

        file = open("lstm_vis_" + str(i) + ".txt", "r")
        input_seq = file.readline()[:-1]
        
        dec_seq = []
        prob = []

        for line in file:
            temp = line.split('\t')
            dec_seq.append(temp[0])
            prob.append(ast.literal_eval(temp[1][:-1])[neuron - 1])

        visualize_l(dec_seq, prob)
        print()

# visualize_connectivity(10)
# visualize_lstm(10, 0)