# Loading Libraries
import os
import pandas as pd
import numpy as np
import torch
import torchtext
from configparser import ConfigParser, ExtendedInterpolation
from modules import ReadParams, DownloadData, ReadData, PrepareData
from modules import TransformText, BuildModel, EvalModel
print("Libraries loaded")

# Reading config
config = ConfigParser(interpolation=ExtendedInterpolation())
config.read('train/config.ini')
print("Configuration loaded")

# Reading Parameters for the training
read_params = ReadParams(config)
config, params = read_params.run()
print("Parameters loaded")

# Downloading data
download_data =  DownloadData(config, params)
download_data.run()
print("Downloaded data")

# Reading downloaded data
read_data =  ReadData(config, params)
training_data = read_data.run()
print("Data loaded")

# Generating training pairs
prepare_data = PrepareData(config, params)
training_sets = prepare_data.run(training_data)
print("Training pairs generated")

# Generates vocabs and transforms data for training
transform_text = TransformText(config, params)
(enc_input, dec_input, dec_output, vocabs) = transform_text.run(training_sets)
print("Transformed training data")

# Creates and trains the model
build_model = BuildModel(config, params)
code2doc_train = build_model.run(enc_input, dec_input, dec_output, vocabs)
print("Model trained")

# Evaluates model
eval_model = EvalModel(config, params)
model_score = eval_model.run(code2doc_train, training_sets, vocabs)
print("Model scored on validation data")


# test model inference
# from utils.code2doc_utils import Code2DocInfer
# model_infer = Code2DocInfer(code2doc_train, vocabs)
# model_infer.predict(enc_input[:10])