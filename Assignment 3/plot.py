import plotly.express as px

def attention_heatmap(input_word, heatmap_data, input_characters_index, inverse_input_characters_index):

    # z = [[1, 2, 3], [4, 5, 6]]
    # x = [1, 2, 4]
    # y = [1, 2]

    _max = max([max([data[1][i] for i in range(len(input_word))]) for data in heatmap_data[:-1]])

    z = []
    x = [char for char in input_word]
    y = []
    
    for data in heatmap_data[:-1]:
        
        char, weight = data[0], data[1]
        y.append(char)
        z.append([weight[i] / _max for i in range(len(input_word))])
    
    fig = px.imshow(z, labels = dict(x = "Input Sequence", y = "Decoded Sequence"), x = x, y = y, color_continuous_scale='gray')
    
    fig.update_xaxes(side="top", tickfont = dict(size = 15))
    fig.update_yaxes(side="left", tickfont = dict(size = 15))

    # fig.update_layout(width = 850, height = 550)
    
    fig.show()

    return fig

# attention_heatmap(None, None, None, None)