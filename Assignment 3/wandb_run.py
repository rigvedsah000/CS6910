import wandb
from wandb.keras import WandbCallback
from tensorflow import keras

import load_data, inference

# Load Data
(encoder_train_input_data, decoder_train_input_data, decoder_train_target_data), (encoder_val_input_data, decoder_val_input_data, decoder_val_target_data), (val_input_words, val_target_words), (encoder_test_input_data, test_input_words, test_target_words), (num_encoder_characters, num_decoder_characters, max_encoder_seq_length, max_decoder_seq_length), (input_characters_index, inverse_input_characters_index), (target_characters_index, inverse_target_characters_index) = load_data.load_data_prediction()

def main(config = None):
    run = wandb.init(config = config)
    config = wandb.config

    run.name = "Embedding Size: " + str(config.embedding_size) + " Cell Type: " + config.cell_type + " Dropout: " + str(config.dropout) + " Beam Size: " + str(config.beam_size) + " Encoder Layers: " + str(config.encoder_layers) + " Decoder Layers: " + str(config.decoder_layers) + " Hidder Layer Size: " + str(config.hidden_layer_size)

    # Configuration
    batch_size = 128
    epochs = 25
    embedding_size = config.embedding_size
    enc_latent_dims = [config.hidden_layer_size] * config.encoder_layers
    dec_latent_dims  = [config.hidden_layer_size] * config.decoder_layers
    cell_type = config.cell_type
    dropout = config.dropout
    beam_size = config.beam_size

    # Encoder
    encoder_inputs = keras.Input(shape = (None, ))
    encoder_outputs = keras.layers.Embedding(input_dim = num_encoder_characters, output_dim = embedding_size, input_length = max_encoder_seq_length)(encoder_inputs)

    # Encoder LSTM layers
    encoder_states = list()
    for j in range(len(enc_latent_dims)):
        if cell_type == "rnn":
            encoder_outputs, state = keras.layers.SimpleRNN(enc_latent_dims[j], dropout = dropout, return_state = True, return_sequences = True)(encoder_outputs)
            encoder_states = [state]
        if cell_type == "lstm":
            encoder_outputs, state_h, state_c = keras.layers.LSTM(enc_latent_dims[j], dropout = dropout, return_state = True, return_sequences = True)(encoder_outputs)
            encoder_states = [state_h,state_c]
        if cell_type == "gru":
            encoder_outputs, state = keras.layers.GRU(enc_latent_dims[j], dropout = dropout, return_state = True, return_sequences = True)(encoder_outputs)
            encoder_states = [state]

    # Decoder
    decoder_inputs = keras.Input(shape=(None, ))
    decoder_outputs = keras.layers.Embedding(input_dim = num_decoder_characters, output_dim = embedding_size, input_length = max_decoder_seq_length)(decoder_inputs)

    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_states = encoder_states.copy()

    for j in range(len(dec_latent_dims)):
        if cell_type == "rnn":
            decoder = keras.layers.SimpleRNN(dec_latent_dims[j], dropout = dropout, return_sequences = True, return_state = True)
            decoder_outputs, state = decoder(decoder_outputs, initial_state = decoder_states)
            # decoder_states = [state]
        if cell_type == "lstm":
            decoder = keras.layers.LSTM(dec_latent_dims[j], dropout = dropout, return_sequences = True, return_state = True)
            decoder_outputs, state_h, state_c = decoder(decoder_outputs, initial_state = decoder_states)
            # decoder_states = [state_h, state_c]
        if cell_type == "gru":
            decoder = keras.layers.GRU(dec_latent_dims[j], dropout = dropout, return_sequences = True, return_state = True)
            decoder_outputs, state = decoder(decoder_outputs, initial_state = decoder_states)
            # decoder_states = [state]

    decoder_dense = keras.layers.Dense(num_decoder_characters, activation = "softmax")
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # encoder_input_data & decoder_input_data into decoder_output_data
    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

    model.compile(
        optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    model.fit(
        [encoder_train_input_data, decoder_train_input_data],
        decoder_train_target_data,
        batch_size = batch_size,
        epochs = epochs,
        callbacks = [WandbCallback()]
    )

    # Save model
    model.save("seq2seq")

    # Inference Call for Validation Data
    val_accuracy = inference.infer(encoder_val_input_data, val_input_words, val_target_words, num_decoder_characters, max_decoder_seq_length, target_characters_index, inverse_target_characters_index, enc_latent_dims, dec_latent_dims, cell_type, beam_size)
    wandb.log( { "val_accuracy": val_accuracy})

    # Inference Call for Test Data
    # test_accuracy = inference.infer(encoder_test_input_data, test_input_words, test_target_words, num_decoder_characters, max_decoder_seq_length, target_characters_index, inverse_target_characters_index, enc_latent_dims, dec_latent_dims, cell_type, beam_size)
    # wandb.log( { "test_accuracy": test_accuracy} )

sweep_config = {

  "name": "Test Sweep 1",
  
  "method": "bayes",

  'metric': {
      'name': 'accuracy',
      'goal': 'maximize'
  },

  "parameters": {
        "embedding_size": {
            "values": [16, 32, 64, 256]
        },
        "encoder_layers" :{
            "values" : [1, 2, 3]
        },
        "decoder_layers": {
            "values": [1, 2, 3]
        },
        "hidden_layer_size": {
            "values": [16, 32, 64, 256]
        },
        "cell_type": {
            "values": ["rnn", "lstm", "gru"]
        },
        "dropout": {
            "values": [0.2, 0.3]
        },
        "beam_size": {
            "values": [0]
        }
    }
}

sweep_id = wandb.sweep(sweep_config, project="Assignment 3")
wandb.agent(sweep_id, project = "Assignment 3", function = main)