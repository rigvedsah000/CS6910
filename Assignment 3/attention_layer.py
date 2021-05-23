import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import backend as K

class AttentionLayer(Layer):

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        self.W_a = self.add_weight(name='W_a',
                                   shape = tf.TensorShape((input_shape[0][2], input_shape[0][2])),
                                   initializer = 'uniform',
                                   trainable = True)

        self.U_a = self.add_weight(name = 'U_a',
                                   shape = tf.TensorShape((input_shape[1][2], input_shape[0][2])),
                                   initializer = 'uniform',
                                   trainable = True)

        self.V_a = self.add_weight(name = 'V_a',
                                   shape = tf.TensorShape((input_shape[0][2], 1)),
                                   initializer = 'uniform',
                                   trainable = True)

        super(AttentionLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):
       
        """
        inputs: [encoder_output_sequence, decoder_output_sequence]
        """
        encoder_out_seq, decoder_out_seq = inputs

        def energy_step(inputs, states):
           
            """ Step function for computing energy for a single decoder state
            inputs: (batchsize * 1 * de_in_dim)
            states: (batchsize * 1 * de_latent_dim)
            """

            """ Some parameters required for shaping tensors"""
            en_seq_len, en_hidden = encoder_out_seq.shape[1], encoder_out_seq.shape[2]
            de_hidden = inputs.shape[-1]

            """ Computing S.Wa where S=[s0, s1, ..., si]"""
            # <= batch size * en_seq_len * latent_dim
            W_a_dot_s = K.dot(encoder_out_seq, self.W_a)

            """ Computing hj.Ua """
            U_a_dot_h = K.expand_dims(K.dot(inputs, self.U_a), 1)  # <= batch_size, 1, latent_dim

            """ tanh(S.Wa + hj.Ua) """
            # <= batch_size*en_seq_len, latent_dim
            Ws_plus_Uh = K.tanh(W_a_dot_s + U_a_dot_h)

            """ softmax(va.tanh(S.Wa + hj.Ua)) """
            # <= batch_size, en_seq_len
            e_i = K.squeeze(K.dot(Ws_plus_Uh, self.V_a), axis=-1)
            # <= batch_size, en_seq_len
            e_i = K.softmax(e_i)
            
            return e_i, [e_i]

        def context_step(inputs, states):
            """ Step function for computing ci using ei """

            # <= batch_size, hidden_size
            c_i = K.sum(encoder_out_seq * K.expand_dims(inputs, -1), axis=1)

            return c_i, [c_i]

        fake_state_c = K.sum(encoder_out_seq, axis=1)
        fake_state_e = K.sum(encoder_out_seq, axis=2)  # <= (batch_size, enc_seq_len, latent_dim

        """ Computing energy outputs """
        # e_outputs => (batch_size, de_seq_len, en_seq_len)
        last_out, e_outputs, _ = K.rnn(
            energy_step, decoder_out_seq, [fake_state_e],
        )

        """ Computing context vectors """
        last_out, c_outputs, _ = K.rnn(
            context_step, e_outputs, [fake_state_c],
        )

        return c_outputs, e_outputs

    # def compute_output_shape(self, input_shape):
        # """ Outputs produced by the layer """
        # return [
        #     tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[1][2])),
        #     tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[0][1]))
        # ]