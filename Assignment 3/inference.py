import numpy as np
from tensorflow import keras

import load_data

def infer(encoder_test_input_data, test_input_words, test_target_words, num_decoder_characters, max_decoder_seq_length, target_characters_index, inverse_target_characters_index):
    
    # Configuration
    batch_size = 128
    epochs = 20
    latent_dim = 256

    model = keras.models.load_model("seq2seq_2")

    print(model.summary())

    encoder_inputs = model.input[0]  # input_1
    encoder_outputs, state_h_enc, state_c_enc = model.layers[4].output  # lstm_1
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = keras.Model(encoder_inputs, encoder_states)

    decoder_inputs = model.input[1]  # input_2
    decoder_embedded_inputs = model.layers[3].output
    decoder_state_input_h = keras.Input(shape=(latent_dim,), name = "input_3")
    decoder_state_input_c = keras.Input(shape=(latent_dim,), name = "input_4")
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_lstm = model.layers[5]
    decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
        decoder_embedded_inputs, initial_state = decoder_states_inputs
    )
    decoder_states = [state_h_dec, state_c_dec]
    decoder_dense = model.layers[6]
    decoder_outputs = decoder_dense(decoder_outputs)
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
    