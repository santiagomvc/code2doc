class Code2DocTrain():
    def __init__(self, enc_dim, dec_dim, enc_vocab_size, dec_vocab_size, emb_size = 512, lstm_units=512):
        # importing libraries
        import tensorflow as tf
        from tensorflow.keras import layers

        # Creating layers
        self.enc_dim = enc_dim
        self.dec_dim = dec_dim
        self.lstm_units = lstm_units
        self.enc_emb_layer = layers.Embedding(enc_vocab_size, emb_size)
        self.dec_emb_layer = layers.Embedding(dec_vocab_size, emb_size)
        self.enc_lstm_layer = layers.LSTM(lstm_units, return_sequences=True, return_state=True)
        self.dec_lstm_layer = layers.LSTM(lstm_units, return_sequences=True, return_state=True)
        self.dec_dense_layer = layers.Dense(dec_vocab_size, activation='softmax')

    def enc_dec_model(self):
        # importing libraries
        import tensorflow as tf
        from tensorflow.keras import layers

        # Load Models
        enc_model = self.enc_model_fun()
        dec_model = self.dec_model_fun()

        # Encoding phase 
        enc_input = tf.keras.Input(shape=(None,), name = 'enc_input')
        enc_states = enc_model(enc_input)
        
        # Decoding phase 
        dec_input = tf.keras.Input(shape=(None,), name = 'dec_input')
        dec_out, _, _ = dec_model([dec_input] + enc_states)
        
        # Generates model object
        model = tf.keras.Model([enc_input, dec_input], dec_out, name = 'enc_dec')

        return model

    def enc_model_fun(self):
        # importing libraries
        import tensorflow as tf

        # Creating the encoding model
        enc_input = tf.keras.Input(shape=(None,))
        enc_emb = self.enc_emb_layer(enc_input)
        _, enc_h, enc_c = self.enc_lstm_layer(enc_emb)
        enc_model = tf.keras.Model(enc_input, [enc_h, enc_c], name = 'encoder')

        return enc_model

    def dec_model_fun(self):
        # importing libraries
        import tensorflow as tf

        # Creating the decoding model
        dec_input = tf.keras.Input(shape=(None,))
        dec_input_state_h = tf.keras.Input(shape=(None,))
        dec_input_state_c = tf.keras.Input(shape=(None,))
        dec_input_states = [dec_input_state_h, dec_input_state_c]
        
        dec_emb = self.dec_emb_layer(dec_input)
        dec_out, dec_h, dec_c = self.dec_lstm_layer(dec_emb, initial_state=dec_input_states)
        dec_out = self.dec_dense_layer(dec_out)
        dec_states = [dec_h, dec_c]
        
        dec_model = tf.keras.Model([dec_input] + dec_input_states, [dec_out] + dec_states, name = 'decoder')
        return dec_model


class Code2DocInfer():
    def __init__(self, model, vocabs):
        # Getting model
        self.vocabs = vocabs
        self.enc_model = model.get_layer("encoder")
        self.dec_model = model.get_layer("decoder")

    def predict(self, enc_input, max_tokens = 20):
        # Importing libraries
        import numpy as np
        import pandas as pd

        # Extracting data from class
        enc_model = self.enc_model
        dec_model = self.dec_model
        vocabs = self.vocabs

        # Extracting parameters
        m, _ = enc_input.shape
        doc_t2i = vocabs['doc_t2i']
        doc_i2t = vocabs['doc_i2t']
        go_id = doc_t2i['<go>']
        eos_id = doc_t2i['<eos>']
        pad_id = doc_t2i['<pad>']

        # encoding the input values
        enc_states = enc_model(enc_input)
        enc_states = [enc_states[0].numpy(), enc_states[1].numpy()]
        
        # Preparing for decoding
        start_seq = np.full((m, 1), go_id) 

        # creating placeholders for results
        seqs = pd.Series(np.full((m), ''))
        seqs_idxs = np.full((m, max_tokens), pad_id)

        # Creating list to identify active sequences
        active_seqs_idxs = np.arange(m) #change to m

        # Defining inputs for decoder
        active_pred_tokens = start_seq
        dec_states = enc_states

        # Starts decoding one step at the time
        # Stops when max tokens is reached
        for t in range(max_tokens):
            # runs decoder on the step t
            dec_token_prob, dec_h, dec_c = dec_model.predict([active_pred_tokens] + dec_states)
            # gets the most likely token
            pred_tokens = np.argmax(dec_token_prob, axis=-1).reshape(-1)
            # defines which sequences ended by predicting the eos token
            active_seqs = (pred_tokens != eos_id) 
            active_seqs_idxs = active_seqs_idxs[active_seqs]

            # stops the loop if there aren't any active sequences left
            if not(any(active_seqs)):
                break

            # selects the active tokens and states 
            active_pred_tokens = pred_tokens[active_seqs]
            dec_states = [dec_h[active_seqs], dec_c[active_seqs]]

            # gets the values for the active tokens
            active_pred_vals = np.apply_along_axis(
                lambda example: [doc_i2t[idx] for idx in example],
                axis = -1,
                arr = active_pred_tokens,
                )

            # Updates the lists with the token and values sequences 
            seqs[active_seqs_idxs] = [f'{seq} {tok}' for seq, tok in zip(seqs[active_seqs_idxs], active_pred_vals)] 
            seqs_idxs[active_seqs_idxs, t] = active_pred_tokens

        return(seqs, seqs_idxs)
