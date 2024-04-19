# Deep-Learning-Project-S24

## Install
- Install the conda environment with "conda env create -f environment.yaml"
- Install the package locally with "pip install -e ."
- Download the dataset from the [VITONHD Dropbox](https://www.dropbox.com/scl/fi/xu08cx3fxmiwpg32yotd7/zalando-hd-resized.zip?rlkey=ks83mdv2pvmrdl2oo2bmmn69w&e=1&dl=0). Extract the zip and put the data in the "data/" folder.


## Usage
- Populate a config in "configs/" with your parameters. "train_dense.yaml" is provided as an example and runs a dummy dense network for debugging.

### Command Line
- Train with "python dl_project/main.py --config-name=\<config name\>"

### Jupyter Notebook
- Open main.ipynb and run the cells. Replace the config name with your config.

## References
- [Pytorch-Lightning documentation](https://lightning.ai/docs/pytorch/stable/)
- [Hydra documentation](https://hydra.cc/docs/intro/)