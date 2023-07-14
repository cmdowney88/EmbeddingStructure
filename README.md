# Targeted Multilingual Language Models

## Environment Setup
To manage your Python environment, we recommend you [install anaconda/miniconda](https://docs.conda.io/en/latest/miniconda.html). Conda should then be used to create an environment with **Python 3.9**, using this command `conda create --name txlm python=3.9`.

After activating your new environment with `conda activate txlm` or `source activate txlm`, confirm that the result of the command `which pip` returns the path to the `pip` executable within your environment folder, e.g. `~/miniconda3/envs/txlm/bin`.

Next, use conda/pip to install the version of PyTorch that is compatible with your system / CUDA version. Original experiments were conducted with PyTorch version 1.13.1 for CUDA 11.7. The command to install this version is `conda install pytorch==1.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia`

Finally, in the main folder of the repository, run the command `pip install -r requirements.txt` to install the required packages.
