# Code2Doc


Code2doc is a python experiment to generate documentation from code using ML

## Steps to run!

  - Clone this repo
  - Build the python environment from train/environment.yml https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file
  - Activate the code2doc environment
  - Run train/code2doc.py 


## Notes:
  - The data used comes from Github's CodeSearchNet and its hosted in an S3 bucket, thanks to https://github.com/github/CodeSearchNet
  - The data used will be downloaded automatically
  - The only model currently available to train is a simple LSTM Seq2Seq model

License
----

MIT

