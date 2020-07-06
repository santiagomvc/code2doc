def create_vocab(dataset, max_vocab_size=25000):
    import pandas as pd
    
    # Tokenize
    ## input comes tokenized

    # Vocabulary
    ## Extracting all tokens
    all_tokens = pd.Series([token for example in dataset for token in example])
    ## Counting tokens
    vocab = all_tokens.value_counts().reset_index()
    ## Selecting the top vocab_size tokens
    vocab = vocab[vocab.index < max_vocab_size]['index']
    ## Add unknown and end of sentence tokens
    vocab.loc[vocab.size] = "<unk>"
    vocab.loc[vocab.size] = "<pad>"
    vocab.loc[vocab.size] = "<go>"
    vocab.loc[vocab.size] = "<eos>"
    ## generating dictionaries
    idx2tkn = dict(zip(vocab.index.tolist(), vocab.tolist()))
    tkn2idx = dict(zip(vocab.tolist(), vocab.index.tolist()))

    return(idx2tkn, tkn2idx)


def transform_examples(X_dataset, Y_dataset, X_params, Y_params):
    import numpy as np
    import pandas as pd
        
    # indexing and padding the sequences
    X_raw = transform_sequences(X_dataset, X_params['tkn2idx'], X_params['min_len'], X_params['max_len'])
    Y_raw = transform_sequences(Y_dataset, Y_params['tkn2idx'], Y_params['min_len'], Y_params['max_len'])

    # Deleting useless examples 
    df_train = pd.DataFrame({'X': X_raw, 'Y': Y_raw})
    df_train = df_train[(df_train['X'] != 'delete') & (df_train['Y'] != 'delete')]
    X, Y = np.array(df_train['X'].tolist()), np.array(df_train['Y'].tolist())

    return(X, Y)


def transform_inputs(X_dataset, vocabs, x_min_len=3, x_max_len=100):
    import numpy as np
    import pandas as pd
    
    # indexing and padding the sequences
    X_raw = transform_sequences(X_dataset, x_min_len, x_max_len)
    # Deleting useless examples 
    X = np.array(X_raw[X_raw != 'delete'].tolist())
    
    return(X)


def transform_sequences(dataset, tkn2idx, min_len=3, max_len=100):
    import pandas as pd
    
    # Indexing and padding
    dataset_series = pd.Series(dataset)
    sequences = dataset_series.apply(lambda tokens: prep_sequence(tokens, tkn2idx, min_len, max_len)).tolist()
    return(sequences)


def prep_sequence(tokens, tkn2idx, min_len, max_len):
    # marks sequences shorter than min_len
    if len(tokens) < min_len:
        return "delete"
    
    # replacing tokens with indexes
    unk_id = tkn2idx['<unk>']
    idxs = [tkn2idx[token] if token in tkn2idx.keys() else unk_id for token in tokens]
    # truncating to max len
    seq = idxs[:max_len]
    # adding eos and go tokens
    seq = [tkn2idx['<go>']] + seq
    seq = seq + [tkn2idx['<eos>']]
    # adding padding when necessary
    pad_size = (max_len + 2) - len(seq) 
    seq = seq + ([tkn2idx['<pad>']] * pad_size) 
    
    return seq


