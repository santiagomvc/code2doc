import pandas as pd
import os
import yaml
from datetime import datetime

from utils.transform_text_utils import create_vocab, transform_examples
from utils.code2doc_utils import Code2DocTrain

class ReadParams:
    def __init__(self, config):
        super().__init__()
        self.config = config

    def run(self):
        # Reading files
        config = self.config
        with open(config['FILES']['base_params_file'], 'r') as base_params_file:
            params = yaml.safe_load(base_params_file)
        if os.path.exists(config['FILES']['train_params_file']):
            with open(config['FILES']['train_params_file'], 'r') as train_params_file:
                train_params = yaml.safe_load(train_params_file)
            params = {**params, **train_params}

        if 'run_date' in params.keys():
            params['run_date'] = datetime.strptime(params['run_date'], '%Y-%m-%d')
        else:
            params['run_date'] = datetime.now()

        # updating config with params
        language = params['language']
        config_names = config['NAMES']
        data_dir = config['PATHS']['data_dir']
        extra_data_path = config['PATHS']['extra_data_path']
        language_dir = f'{data_dir}/{language}'
        raw_data_dir = f'{language_dir}/{language}/{extra_data_path}'
        config['PATHS']['language_dir'] = language_dir
        config['PATHS']['raw_data_dir'] = raw_data_dir
        config['PATHS']['train_dir'] = f"{raw_data_dir}/{config_names['train_dir_name']}"
        config['PATHS']['valid_dir'] = f"{raw_data_dir}/{config_names['valid_dir_name']}"
        config['PATHS']['test_dir'] = f"{raw_data_dir}/{config_names['test_dir_name']}"

        return config, params


class DownloadData:
    def __init__(self, config, params):
        super().__init__()
        self.config = config
        self.params = params

    def run(self):
        # Importing liraries
        import os
        import shutil
        import zipfile
        import requests

        # Reading configuations
        language = self.params['language']
        language_dir = self.config['PATHS']['language_dir']
        zip_name = f'{language}.zip'
        raw_data_url = self.config['PATHS']['raw_data_url']
        url = f'{raw_data_url}/{zip_name}'

        # Deleting folder if already exists
        if os.path.exists(language_dir) and os.path.isdir(language_dir):
            shutil.rmtree(language_dir)

        # Creating folder to save the files
        os.makedirs(language_dir, exist_ok=True)

        # Download the zipped files to a temporal location
        r = requests.get(url = url)
        with open(zip_name,'wb') as fd:
            fd.write(r.content)

        # Unzipping files into defined folder
        with zipfile.ZipFile(zip_name, 'r') as zip_ref:
            zip_ref.extractall(language_dir)

        # removing temp file
        os.remove(zip_name)


class ReadData:
    def __init__(self, config, params):
        super().__init__()
        self.config = config
        self.params = params

    def run(self):
        # Reading folder paths
        train_dir = self.config['PATHS']['train_dir']
        valid_dir = self.config['PATHS']['valid_dir']
        test_dir = self.config['PATHS']['test_dir']

        training_data = []

        # iterates over each folder
        for data_dir in [train_dir, valid_dir, test_dir]:

            # Empty list to save results
            dfs_list = []

            # List files in the folder 
            data_dir_files = os.listdir(data_dir)
            
            # Saves each file as a DF in a list 
            for data_file in data_dir_files:
                train_df_partition = pd.read_json(f'{data_dir}/{data_file}', compression = 'gzip', lines=True)
                dfs_list.append(train_df_partition)

            # Creates ands saves a df with all the data in the folder 
            df = pd.concat(dfs_list, ignore_index=True)
            training_data.append(df)
        
        return training_data


class PrepareData:
    def __init__(self, config, params):
        super().__init__()
        self.config = config
        self.params = params

    def run(self, training_data):
    
        training_sets_list = []
        for df in training_data:
            training_sets_list.append(df.code_tokens.values.tolist())
            training_sets_list.append(df.docstring_tokens.values.tolist())

        training_sets_names = ['X_train', 'Y_train', 
                               'X_valid', 'Y_valid',
                               'X_test', 'Y_test']

        training_sets = dict(zip(training_sets_names, training_sets_list))

        return(training_sets)


class TransformText:
    def __init__(self, config, params):
        super().__init__()
        self.config = config
        self.params = params

    def run(self, training_sets):
        # import libraries
        import numpy as np

        # Extracting parameters
        min_input_len = self.params['min_input_len']
        min_output_len = self.params['min_output_len']
        max_input_len = self.params['max_input_len']
        max_output_len = self.params['max_output_len']

        # Extracting training sets
        code_train, doc_train = training_sets['X_train'], training_sets['Y_train']

        # Creating vocabulary
        (code_idx2tkn, code_tkn2idx) = create_vocab(code_train)
        (doc_idx2tkn, doc_tkn2idx) = create_vocab(doc_train)
        
        # defining parameters for transforming examples
        code_params = {'tkn2idx': code_tkn2idx, 'min_len': min_input_len, 'max_len': max_input_len}
        doc_params = {'tkn2idx': doc_tkn2idx, 'min_len': min_output_len, 'max_len': max_output_len}

        # indexing and padding the sequences
        X_train, Y_train = transform_examples(code_train, doc_train, code_params, doc_params)

        enc_input = X_train 
        dec_input = Y_train[:,:-1]
        dec_output = Y_train[:,1:]

        # Creating a dictionary with all the vocabs
        vocabs = {
            'code_i2t': code_idx2tkn,
            'code_t2i': code_tkn2idx,
            'doc_i2t': doc_idx2tkn,
            'doc_t2i': doc_tkn2idx
        }

        # Return X_train and Y_train 
        return(enc_input, dec_input, dec_output, vocabs)


class BuildModel:
    def __init__(self, config, params):
        super().__init__()
        self.config = config
        self.params = params

    def run(self, enc_input, dec_input, dec_output, vocabs):
        # Extracting needed params
        loss = self.params['loss']
        optimizer = self.params['optimizer']
        metrics = self.params['metrics']
        batch_size = self.params['batch_size']
        epochs = self.params['epochs']
        model_save_dir = self.config['PATHS']['trainings_dir']
        model_save_file = self.config['FILES']['model_file']

        # Creating folder to save model if it doesn't exists
        os.makedirs(model_save_dir, exist_ok=True)
        
        # Extracting needed data
        _, enc_dim = enc_input.shape
        _, dec_dim = dec_input.shape
        enc_vocab_size = len(vocabs['code_t2i'].keys())
        dec_vocab_size = len(vocabs['doc_t2i'].keys())
        
        # Creating the model class
        code2doc = Code2DocTrain(enc_dim, dec_dim, enc_vocab_size, dec_vocab_size)
        # Extracting the model object
        model = code2doc.enc_dec_model()
        # Defining hyperparameters for the model
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        # Fitting the model to the data
        model.fit([enc_input, dec_input], dec_output, batch_size=batch_size, epochs=epochs) 
        #Saving the model
        model.save(model_save_file)
        
        return(model)


class EvalModel:
    def __init__(self, config, params):
        super().__init__()
        self.config = config
        self.params = params

    def run(self, model, training_sets, vocabs):
        
        # Extracting parameters
        min_input_len = self.params['min_input_len']
        min_output_len = self.params['min_output_len']
        max_input_len = self.params['max_input_len']
        max_output_len = self.params['max_output_len']
        score_save_dir = self.config['PATHS']['trainings_dir']
        score_save_file = self.config['FILES']['model_score']

        # Creating folder to save score if it doesn't exists
        os.makedirs(score_save_dir, exist_ok=True)

        # getting dictionaries with vocabulary
        code_tkn2idx = vocabs['code_t2i']
        doc_tkn2idx = vocabs['doc_t2i']

        # defining parameters for transforming examples
        code_params = {'tkn2idx': code_tkn2idx, 'min_len': min_input_len, 'max_len': max_input_len}
        doc_params = {'tkn2idx': doc_tkn2idx, 'min_len': min_output_len, 'max_len': max_output_len}

        # Extracting eval data
        code_val, doc_val = training_sets['X_valid'], training_sets['Y_valid']

        # indexing and padding the sequences
        X_valid, Y_valid = transform_examples(code_val, doc_val, code_params, doc_params)

        # Defining inputs and outputs
        enc_input = X_valid 
        dec_input = Y_valid[:,:-1]
        dec_output = Y_valid[:,1:]

        # evaluating model performance
        score = model.evaluate([enc_input, dec_input], dec_output)
        
        # Saving model performance
        with open(score_save_file, 'w') as f:
            print(score, file=f)

        return(score)