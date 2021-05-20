import numpy as np
from tensorflow import keras
from keras.utils.vis_utils import plot_model

import load_data



# Function to calculate indexes of layers (useful in inference function)
def func( model, enc_latent_dims, cell_type ):
    n_enc_layer = len(enc_latent_dims)

    e_emb_index, d_emb_index = -1,-1
    e_cell_index, d_cell_index = [],[]
    dense_index = -1;

    celltype = cell_type[0]

    e_celllayer_count = 0
    print("cell type is :",celltype)
    for i,layer in enumerate(model.layers):

        # Dense Layer
        if "dense" in layer.name : dense_index = i

        # Embedding layer
        if "embedding" in layer.name :
            if e_emb_index == -1 : e_emb_index = i
            else : d_emb_index = i

        # Lstm layer 
        if celltype in layer.name :
            if e_celllayer_count  < n_enc_layer :
                e_cell_index.append(i)
                e_celllayer_count += 1
            else : d_cell_index.append(i)

    return e_emb_index, d_emb_index, e_cell_index, d_cell_index, dense_index






# Inference Function
def infer(encoder_test_input_data, test_input_words, test_target_words, num_decoder_characters, max_decoder_seq_length, target_characters_index, inverse_target_characters_index,  enc_latent_dims, dec_latent_dims, cell_type):
    
    # Configuration
    batch_size = 128
    epochs = 20
    latent_dim = 256

    model = keras.models.load_model("seq2seq")

    print(model.summary())


    e_emb_index, d_emb_index, e_cell_index, d_cell_index, dense_index = func( model, enc_latent_dims, cell_type )


    # Encoder

    encoder_inputs = model.input[0]  # input_1
    encoder_outputs, state_h_enc, state_c_enc = model.layers[e_cell_index[-1]].output


    # Final encoder model
    encoder_model = keras.Model(encoder_inputs, [state_h_enc, state_c_enc])


    decoder_inputs = model.input[1]  # input_2
    decoder_outputs =  model.layers[d_emb_index](decoder_inputs)

    decoder_states_inputs =  []
    decoder_states = []



    # Decoder LSTM
    for dec in range(len(d_cell_index)):
        state_h_dec, state_c_dec = keras.Input(shape = (dec_latent_dims[dec],)),  keras.Input(shape = (dec_latent_dims[dec],))
        current_states_inputs = [state_h_dec,state_c_dec]
        decoder_outputs, state_h_dec,state_c_dec = model.layers[d_cell_index[dec]](decoder_outputs, initial_state=current_states_inputs)
        decoder_states += [state_h_dec,state_c_dec]
        decoder_states_inputs += current_states_inputs

    # Dense layer
    decoder_dense = model.layers[dense_index]
    decoder_outputs = decoder_dense(decoder_outputs)


    # Final decoder model
    decoder_model = keras.Model(
        [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
    )


    def decode_sequence(input_seq):
        # Encode the input as state vectors.
        states_value = encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0] = target_characters_index['\t']

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ""
        while not stop_condition:
            output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = inverse_target_characters_index[sampled_token_index]
            decoded_sentence += sampled_char

            # Exit condition: either hit max length
            # or find stop character.
            if sampled_char == "\n" or len(decoded_sentence) > max_decoder_seq_length:
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, num_decoder_characters))
            target_seq[0, 0] = sampled_token_index

            # Update states
            states_value = [h, c]
        return decoded_sentence


    count = 0
    for seq_index in range(20):
        # Take one sequence (part of the training set)
        # for trying out decoding.
        input_seq = encoder_test_input_data[seq_index : seq_index + 1]
        decoded_word = decode_sequence(input_seq)
        print("-")
        print("Input sentence:", test_input_words[seq_index])
        print("Decoded sentence:", decoded_word)
        orig_word = test_target_words[seq_index][1:]
        print("Original sentence:", orig_word)

        if(orig_word == decoded_word): count += 1

    print("Accuracy: ", count / 20)
    
