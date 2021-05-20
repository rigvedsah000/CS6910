import numpy as np
from tensorflow import keras

import load_data, inference

# Load Data
(encoder_train_input_data, decoder_train_input_data, decoder_train_target_data), (encoder_val_input_data, decoder_val_input_data, decoder_val_target_data), (encoder_test_input_data, test_input_words, test_target_words), (num_encoder_characters, num_decoder_characters, max_encoder_seq_length, max_decoder_seq_length), (target_characters_index, inverse_target_characters_index) = load_data.load_data_prediction()

# Configuration
batch_size = 128
epochs = 1
embedding_size = 256
enc_latent_dims = [64,128,256]
dec_latent_dims  = [256]
cell_type = ["lstm","lstm","lstm"]
DROPOUT = 0                                         # Dropout in lstm layers


# Encoder
encoder_inputs = keras.Input(shape = (None, ))
encoder_outputs = keras.layers.Embedding(input_dim = num_encoder_characters, output_dim = embedding_size, input_length = max_encoder_seq_length)(encoder_inputs)

# Encoder LSTM layers
encoder_states = list()
for j in range(len(enc_latent_dims)):
    if cell_type[j] == "lstm" :
        encoder_outputs, state_h, state_c = keras.layers.LSTM(enc_latent_dims[j],dropout=DROPOUT, recurrent_dropout=0, return_state=True, return_sequences=True)(encoder_outputs)
        encoder_states = [state_h,state_c]



# Decoder
decoder_inputs = keras.Input(shape=(None, ))
decoder_outputs = keras.layers.Embedding(input_dim = num_decoder_characters, output_dim = embedding_size, input_length = max_decoder_seq_length)(decoder_inputs)

# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.



decoder_states = encoder_states.copy()

for j in range(len(dec_latent_dims)):

    if cell_type[j] == "lstm" :
        decoder = keras.layers.LSTM(dec_latent_dims[j], dropout=DROPOUT, recurrent_dropout=0, return_sequences=True, return_state=True)
        decoder_outputs, state_h, state_c = decoder(decoder_outputs, initial_state=decoder_states)
        decoder_states = [state_h, state_c]


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
    validation_data = ([encoder_val_input_data, decoder_val_input_data], decoder_val_target_data),
)


# Save model
model.save("seq2seq")

inference.infer(encoder_test_input_data, test_input_words, test_target_words, num_decoder_characters, max_decoder_seq_length, target_characters_index, inverse_target_characters_index, enc_latent_dims, dec_latent_dims, cell_type)
