import numpy as np
from tensorflow import keras

# Inference Function
def infer(encoder_test_input_data, test_input_words, test_target_words, num_decoder_characters, max_decoder_seq_length, target_characters_index, inverse_target_characters_index, latent_dim, cell_type):
    
    model = keras.models.load_model("seq2seq_attention")

    print(model.summary())

    # Encoder
    encoder_inputs = model.input[0]  # input_1

    if cell_type == "rnn" or cell_type == "gru":
        encoder_outputs, state = model.layers[4].output
        encoder_model = keras.Model(encoder_inputs, [encoder_outputs] + [state])
    
    elif cell_type == "lstm":
        encoder_outputs, state_h_enc, state_c_enc = model.layers[4].output
        encoder_model = keras.Model(encoder_inputs, [encoder_outputs] + [state_h_enc, state_c_enc])
    
    else:
        return

    decoder_inputs = model.input[1]  # input_2
    decoder_outputs = model.layers[3](decoder_inputs)

    if cell_type == "rnn" or cell_type == "gru":
        state = keras.Input(shape = (latent_dim, ))
        decoder_states_inputs = [state]
        decoder_outputs, state = model.layers[5](decoder_outputs, initial_state = decoder_states_inputs)
        decoder_states = [state]

    elif cell_type == "lstm":
        state_h_dec, state_c_dec = keras.Input(shape = (latent_dim, )), keras.Input(shape = (latent_dim, ))
        decoder_states_inputs = [state_h_dec, state_c_dec]
        decoder_outputs, state_h_dec, state_c_dec = model.layers[5](decoder_outputs, initial_state = decoder_states_inputs)
        decoder_states = [state_h_dec, state_c_dec]
        
    attention_inputs = keras.Input(shape = (None, ))
    attention_output = model.layers[6]([decoder_outputs, attention_inputs])
    decoder_concat_input = model.layers[7]([decoder_outputs, attention_output])

    # Dense layer
    decoder_dense = model.layers[8]
    decoder_outputs = decoder_dense(decoder_concat_input)

    # Final decoder model
    decoder_model = keras.Model(
        [decoder_inputs] + decoder_states_inputs + [attention_inputs], [decoder_outputs] + decoder_states
    )

    def decode_sequence(input_seq):
        # Encode the input as state vectors.
        encoder_outputs = encoder_model.predict(input_seq)
        encoder_output, states_value = encoder_outputs[0], encoder_outputs[1:]
        
        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1))

        # Populate the first character of target sequence with the start character.
        target_seq[0, 0] = target_characters_index["\t"]
        
        stop_condition = False
        decoded_sentence = ""

        while not stop_condition:
            output = decoder_model.predict([target_seq] + states_value + [encoder_output])
            output_tokens, states_value = output[0], output[1:]

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = inverse_target_characters_index[sampled_token_index]
            decoded_sentence += sampled_char

            if sampled_char == "\n" or len(decoded_sentence) > max_decoder_seq_length:
                stop_condition = True

            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index 

        return decoded_sentence

    count, test_size = 0, len(test_input_words)

    for seq_index in range(test_size):
        # Take one sequence (part of the training set)
        # for trying out decoding.
        input_seq = encoder_test_input_data[seq_index : seq_index + 1]
        decoded_word = decode_sequence(input_seq)
        print("-")
        print("Input sentence:", test_input_words[seq_index])
        print("Decoded sentence:", decoded_word[:-1])
        orig_word = test_target_words[seq_index][1:]
        print("Original sentence:", orig_word[:-1])

        if(orig_word == decoded_word): count += 1

    return count / test_size