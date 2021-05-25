import wandb
from wandb.keras import WandbCallback
from tensorflow import keras

import load_data, attention_inference, attention_layer

# Load Data
(encoder_train_input_data, decoder_train_input_data, decoder_train_target_data), (encoder_val_input_data, decoder_val_input_data, decoder_val_target_data), (val_input_words, val_target_words), (encoder_test_input_data, test_input_words, test_target_words), (num_encoder_characters, num_decoder_characters, max_encoder_seq_length, max_decoder_seq_length), (target_characters_index, inverse_target_characters_index) = load_data.load_data_prediction()

def main(config = None):
    run = wandb.init(config = config)
    config = wandb.config

    run.name = "Embedding Size: " + str(config.embedding_size) + " Cell Type: " + config.cell_type + " Dropout: " + str(config.dropout) + " Beam Size: " + str(config.beam_size) + " Encoder Layers: " + str(config.encoder_layers) + " Decoder Layers: " + str(config.decoder_layers) + " Hidder Layer Size: " + str(config.hidden_layer_size)

    # Configuration
    batch_size = 256
    epochs = 25
    embedding_size = config.embedding_size
    latent_dim = config.hidden_layer_size
    cell_type = config.cell_type
    dropout = config.dropout
    
    # Encoder
    encoder_inputs = keras.Input(shape = (None, ))
    encoder_outputs = keras.layers.Embedding(input_dim = num_encoder_characters, output_dim = embedding_size, input_length = max_encoder_seq_length)(encoder_inputs)

    # Encoder LSTM layers
    encoder_states = list()
    if cell_type == "rnn":
        encoder_outputs, state = keras.layers.SimpleRNN(latent_dim, dropout = dropout, return_state = True, return_sequences = True)(encoder_outputs)
        encoder_states = [state]
    if cell_type == "lstm":
        encoder_outputs, state_h, state_c = keras.layers.LSTM(latent_dim, dropout = dropout, return_state = True, return_sequences = True)(encoder_outputs)
        encoder_states = [state_h,state_c]
    if cell_type == "gru":
        encoder_outputs, state = keras.layers.GRU(latent_dim, dropout = dropout, return_state = True, return_sequences = True)(encoder_outputs)
        encoder_states = [state]

    # Decoder
    decoder_inputs = keras.Input(shape=(None, ))
    decoder_outputs = keras.layers.Embedding(input_dim = num_decoder_characters, output_dim = embedding_size, input_length = max_decoder_seq_length)(decoder_inputs)

    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.

    if cell_type == "rnn":
        decoder = keras.layers.SimpleRNN(latent_dim, dropout = dropout, return_sequences = True, return_state = True)
        decoder_outputs, state = decoder(decoder_outputs, initial_state = encoder_states)
        decoder_states = [state]
    if cell_type == "lstm":
        decoder = keras.layers.LSTM(latent_dim, dropout = dropout, return_sequences = True, return_state = True)
        decoder_outputs, state_h, state_c = decoder(decoder_outputs, initial_state = encoder_states)
        decoder_states = [state_h, state_c]
    if cell_type == "gru":
        decoder = keras.layers.GRU(latent_dim, dropout = dropout, return_sequences = True, return_state = True)
        decoder_outputs, state = decoder(decoder_outputs, initial_state = encoder_states)
        decoder_states = [state]

    # Attention
    attention = attention_layer.AttentionLayer()
    attention_output, _ = attention([encoder_outputs, decoder_outputs])
    decoder_concat_input = keras.layers.Concatenate(axis = -1)([decoder_outputs, attention_output])

    decoder_dense = keras.layers.Dense(num_decoder_characters, activation = "softmax")
    decoder_outputs = decoder_dense(decoder_concat_input)

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
    model.save("seq2seq_attention")

    # Inference Call for Validation Data
    val_accuracy, heatmaps = attention_inference.infer(encoder_val_input_data, val_input_words, val_target_words, num_decoder_characters, max_decoder_seq_length, target_characters_index, inverse_target_characters_index, latent_dim, cell_type)
    wandb.log( {"val_accuracy": val_accuracy})
    
    for i, heatmap in enumerate(heatmaps):
        wandb.log( {"heatmap_" + str(i): heatmap})

    # Inference Call for Test Data
    # test_accuracy, heatmaps = attention_inference.infer(encoder_test_input_data, test_input_words, test_target_words, num_decoder_characters, max_decoder_seq_length, target_characters_index, inverse_target_characters_index, latent_dim, cell_type)
    # wandb.log( { "test_accuracy": test_accuracy, "attention_heatmaps": heatmaps} )

sweep_config = {

  "name": "Attention Sweep 1",
  
  "method": "grid",

  'metric': {
      'name': 'accuracy',
      'goal': 'maximize'
  },

  "parameters": {
        "embedding_size": {
            "values": [64, 256]
        },
        "encoder_layers" :{
            "values" : [1]
        },
        "decoder_layers": {
            "values": [1]
        },
        "hidden_layer_size": {
            "values": [64, 256]
        },
        "cell_type": {
            "values": ["lstm", "gru"]
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