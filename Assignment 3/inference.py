import numpy as np
from tensorflow import keras
from random import sample

# Function to calculate indices of layers
def get_layer_index(model, enc_latent_dims, cell_type):

    n_enc_layer = len(enc_latent_dims)

    e_emb_index, d_emb_index = -1, -1
    e_cell_index, d_cell_index = [], []
    dense_index = -1

    e_celllayer_count = 0
    
    for i, layer in enumerate(model.layers):
        # Dense Layer
        if "dense" in layer.name : dense_index = i

        # Embedding layer
        if "embedding" in layer.name:
            if e_emb_index == -1 : e_emb_index = i
            else: d_emb_index = i

        # Lstm layer 
        if cell_type in layer.name:
            if e_celllayer_count < n_enc_layer:
                e_cell_index.append(i)
                e_celllayer_count += 1
            else:
                d_cell_index.append(i)

    return e_emb_index, d_emb_index, e_cell_index, d_cell_index, dense_index

# Inference Function
def infer(encoder_test_input_data, test_input_words, test_target_words, num_decoder_characters, max_decoder_seq_length, target_characters_index, inverse_target_characters_index, enc_latent_dims, dec_latent_dims, cell_type, beam_size):
    
    model = keras.models.load_model("seq2seq")

    # print(model.summary())

    e_emb_index, d_emb_index, e_cell_index, d_cell_index, dense_index = get_layer_index(model, enc_latent_dims, cell_type)

    # Encoder
    encoder_inputs = model.input[0]  # input_1

    if cell_type == "rnn" or cell_type == "gru":
        encoder_outputs, state = model.layers[e_cell_index[-1]].output
        encoder_model = keras.Model(encoder_inputs, [state])
    
    elif cell_type == "lstm":
        encoder_outputs, state_h_enc, state_c_enc = model.layers[e_cell_index[-1]].output
        encoder_model = keras.Model(encoder_inputs, [state_h_enc, state_c_enc])
    
    else:
        return

    decoder_inputs = model.input[1]  # input_2
    decoder_outputs =  model.layers[d_emb_index](decoder_inputs)

    decoder_states_inputs =  []
    decoder_states = []

    # Decoder LSTM
    for dec in range(len(d_cell_index)):
        
        if cell_type == "rnn" or cell_type == "gru":
            state = keras.Input(shape = (dec_latent_dims[dec], ))
            current_states_inputs = [state]
            decoder_outputs, state = model.layers[d_cell_index[dec]](decoder_outputs, initial_state = current_states_inputs)
            decoder_states += [state]

        elif cell_type == "lstm":
            state_h_dec, state_c_dec = keras.Input(shape = (dec_latent_dims[dec],)),  keras.Input(shape = (dec_latent_dims[dec],))
            current_states_inputs = [state_h_dec, state_c_dec]
            decoder_outputs, state_h_dec,state_c_dec = model.layers[d_cell_index[dec]](decoder_outputs, initial_state = current_states_inputs)
            decoder_states += [state_h_dec, state_c_dec]
        
        decoder_states_inputs += current_states_inputs

    # Dense layer
    decoder_dense = model.layers[dense_index]
    decoder_outputs = decoder_dense(decoder_outputs)

    # Final decoder model
    decoder_model = keras.Model(
        [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
    )

    def sigmoid(x):
        return [1/(1 + np.exp(-z)) for z in x]

    def decode_sequence(input_seq):
        # Encode the input as state vectors.
        states_value = [encoder_model.predict(input_seq)] * len(d_cell_index)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros(( 1, 1))

        # Populate the first character of target sequence with the start character.
        target_seq[0, 0 ] = target_characters_index["\t"]
        
        stop_condition = False
        decoded_sentence = ""
        visualization_data = []

        while not stop_condition:
            output = decoder_model.predict([target_seq] + states_value)
            output_tokens, states_value = output[0], output[1:]

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])

            sampled_char = inverse_target_characters_index[sampled_token_index]

            decoded_sentence += sampled_char

            if sampled_char == "\n" or len(decoded_sentence) > max_decoder_seq_length:
                stop_condition = True

            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index 
            visualization_data.append((sampled_char, states_value[0]))

        return decoded_sentence, visualization_data

    def beam_search_decoder(input_seq, k):
        # Encode the input as state vectors.
        states_value = [encoder_model.predict(input_seq)] * len(d_cell_index)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1))

        # Populate the first character of target sequence with the start character.
        target_seq[0, 0 ] = target_characters_index["\t"]
        
        stop_condition = False
        decoded_sentence = ""

        # probabibility of sequence(prob), flag for end of word(eow), states_value, target_seq, sequence_token, sequence_in_word
        sequences = [[0.0, 0, states_value, target_seq,  list(),list()]]

        while not stop_condition:

            all_candidates = list()
            
            for i in range(len(sequences)):
              output = decoder_model.predict([sequences[i][3]] + sequences[i][2])
              output_tokens, states_value = output[0], output[1:]
              prob = output_tokens[0,-1,:]
              
              score, eow, sv, t_seq, seq, d_word = sequences[i]
              
              if eow == 0:
                for j in range(len(inverse_target_characters_index)):
                  char = inverse_target_characters_index[j]

                  target_seq = np.zeros((1, 1))
                  target_seq[0, 0] = j

                  candidate = [score - np.log(prob[j]), 0, states_value, target_seq,  seq + [j] , d_word + [char] ]
                  all_candidates.append(candidate)
            
            ordered = sorted(all_candidates, key=lambda x:x[0])

            minlen = min(k, len(ordered))

            sequences = ordered[:minlen]

            stop_condition = True
           
            for sequence in range(len(sequences)):
               
                score, eow, sv, t_seq, seq, d_word = sequences[sequence]

                if d_word[-1] == "\n": eow = 1

                if len(d_word) > max_decoder_seq_length : eow = 1

                sequences[sequence] = [score, eow, sv, t_seq, seq, d_word].copy()

                if eow == 0: stop_condition = False

            if sequences[0][-1][-1]=="\n": stop_condition = True

        best_possible_decoded_sentence = ''.join(sequences[0][5])

        return best_possible_decoded_sentence

    count, visual_count, test_size = 0, 0, len(test_input_words)

    predictions_vanilla = open("predictions_vanilla.csv", "w", encoding='utf-8')
    predictions_vanilla.write("Input Sentence,Predicted Sentence,Original Sentence\n")  

    visualisation_inputs = sample(range(5), 5)

    for seq_index in range(5):        
        input_seq = encoder_test_input_data[seq_index : seq_index + 1]

        if beam_size == 0:
            '''  Call to Simple Decode Sequence  '''
            decoded_word, visualization_data = decode_sequence(input_seq)
        else:
            '''  Call to Beam Search Decoder  '''
            decoded_word = beam_search_decoder(input_seq, beam_size)

        orig_word = test_target_words[seq_index][1:]

        # print("-")
        # print("Input sentence:", test_input_words[seq_index])
        # print("Decoded sentence:", decoded_word[:-1])
        # print("Original sentence:", orig_word[:-1])

        predictions_vanilla.write(test_input_words[seq_index] + "," + decoded_word[:-1] + "," + orig_word[:-1] + "\n")
        
        if(orig_word == decoded_word): count += 1

        if beam_size == 0 and seq_index in visualisation_inputs:
            
            # LSTM Visualization
            file = open("lstm_vis_" + str(visual_count) + ".txt", "w", encoding='utf-8')
            file.write(test_input_words[seq_index] + "\n")

            for i, data in enumerate(visualization_data):
        
                dec_char, neuron_activation  = data[0], sigmoid(data[1].reshape(-1))
                
                if i == len(visualization_data) - 1:
                    file.write("<e>" + "\t" + str(neuron_activation) + "\n")
                else:
                    file.write(dec_char + "\t" + str(neuron_activation) + "\n")
            
            visual_count += 1

    return count / test_size