import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_data_prediction():

    train_file = pd.read_csv("dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv", sep = '\t', header = None)
    val_file = pd.read_csv("dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.dev.tsv", sep = '\t', header = None)
    test_file = pd.read_csv("dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv", sep = '\t', header = None)

    train_input_words = [str(word) for word in list(train_file[1])]
    train_target_words = ["\t" + str(word) + "\n" for word in list(train_file[0])]

    val_input_words = [str(word) for word in list(val_file[1])]
    val_target_words = ["\t" + str(word) + "\n" for word in list(val_file[0])]

    test_input_words = [str(word) for word in list(test_file[1])]
    test_target_words = ["\t" + str(word) + "\n" for word in list(test_file[0])]

    input_characters = set()
    target_characters = set()

    for train_word in train_input_words:
        for char in train_word:
            input_characters.add(char)

    for test_word in val_input_words:
        for char in test_word:
            input_characters.add(char)

    for val_word in test_input_words:
        for char in val_word:
            input_characters.add(char)

    for train_word in train_target_words:
        for char in train_word:
            target_characters.add(char)

    for val_word in val_target_words:
        for char in val_word:
            target_characters.add(char)

    for test_word in test_target_words:
        for char in test_word:
            target_characters.add(char)

    input_characters       = sorted(list(input_characters))
    target_characters      = sorted(list(target_characters))
    num_encoder_characters = len(input_characters)
    num_decoder_characters = len(target_characters)

    max_encoder_seq_length = max(max([len(word) for word in train_input_words]), max([len(word) for word in val_input_words]), max([len(word) for word in test_input_words]))
    max_decoder_seq_length = max(max([len(word) for word in train_target_words]), max([len(word) for word in val_target_words]), max([len(word) for word in test_target_words]))

    # Summary
    print("Number of train words: ", len(train_input_words))
    print("Number of val words: ", len(val_input_words))
    print("Number of test words: ", len(test_input_words))
    print("Number of input characters: ", num_encoder_characters)
    print("Number of output characters: ", num_decoder_characters)
    print("Max sequence length for inputs: ", max_encoder_seq_length)
    print("Max sequence length for train outputs: ", max_decoder_seq_length)

    input_characters_index = dict([(char, i) for i, char in enumerate(input_characters)])
    target_characters_index = dict([(char, i) for i, char in enumerate(target_characters)])

    inverse_input_characters_index = dict((i, char) for char, i in input_characters_index.items())
    inverse_target_characters_index = dict((i, char) for char, i in target_characters_index.items())

    encoder_train_input_data = np.zeros(
    (len(train_input_words), max_encoder_seq_length), dtype="float32"
    )
    decoder_train_input_data = np.zeros(
        (len(train_input_words), max_decoder_seq_length), dtype="float32"
    )
    decoder_train_target_data = np.zeros(
        (len(train_input_words), max_decoder_seq_length, num_decoder_characters ), dtype="float32"
    )

    encoder_val_input_data = np.zeros(
    (len(val_input_words), max_encoder_seq_length), dtype="float32"
    )
    decoder_val_input_data = np.zeros(
        (len(val_input_words), max_decoder_seq_length), dtype="float32"
    )
    decoder_val_target_data = np.zeros(
        (len(val_input_words), max_decoder_seq_length, num_decoder_characters), dtype="float32"
    )

    encoder_test_input_data = np.zeros(
    (len(test_input_words), max_encoder_seq_length), dtype="float32"
    )

    for i, (input_word, target_word) in enumerate(zip(train_input_words, train_target_words)):
        for t, char in enumerate(input_word):
            encoder_train_input_data[i, t] = input_characters_index[char]
        encoder_train_input_data[i, t + 1 :] = input_characters_index[' ']
        
        for t, char in enumerate(target_word):
            decoder_train_input_data[i, t] = target_characters_index[char]
            if t > 0:
                # decoder_target_data will be ahead by one timestep
                # and will not include the start character.
                decoder_train_target_data[i, t - 1, target_characters_index[char]] = 1.0
        decoder_train_input_data[i, t + 1 :] = target_characters_index[' ']
        decoder_train_target_data[i, t :, target_characters_index[' ']] = 1.0

    for i, (input_word, target_word) in enumerate(zip(val_input_words, val_target_words)):
        for t, char in enumerate(input_word):
            encoder_val_input_data[i, t] = input_characters_index[char]
        encoder_val_input_data[i, t + 1 :] = input_characters_index[' ']
        
        for t, char in enumerate(target_word):
            decoder_val_input_data[i, t] = target_characters_index[char]
            if t > 0:
                # decoder_target_data will be ahead by one timestep
                # and will not include the start character.
                decoder_val_target_data[i, t - 1 :, target_characters_index[char]] = 1.0
        decoder_val_input_data[i, t + 1 :] =  target_characters_index[' ']
        decoder_val_target_data[i, t :, target_characters_index[' ']] = 1.0

    for i, input_word in enumerate(test_input_words):
        for t, char in enumerate(input_word):
            encoder_test_input_data[i, t] = input_characters_index[char]
        encoder_test_input_data[i, t + 1 :] = input_characters_index[' ']


    return (encoder_train_input_data, decoder_train_input_data, decoder_train_target_data), (encoder_val_input_data, decoder_val_input_data, decoder_val_target_data), (encoder_test_input_data, test_input_words, test_target_words), (num_encoder_characters, num_decoder_characters, max_encoder_seq_length, max_decoder_seq_length), (target_characters_index, inverse_target_characters_index)  