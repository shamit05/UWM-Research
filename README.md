# Research
Microstructure research

### Poetry setup
We are using poetry as the our dependency management and packaging tool.\
In order to create the virutalenv within the project root, use command:
```
poetry config virtualenvs.in-project true
```
Then create environment and install packages using:
```
poetry install
```
Basic usage guide for [poetry commands](https://plainenglish.io/blog/poetry-a-better-version-of-python-pipenv).
___
#### If you have CUDA enabled GPU
After poetry install, you need to remove pytorch-cpu and install pytorch with CUDA support:
```
poetry shell
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir
```
To check if CUDA installation worked:
```
python -c "import torch; print(torch.cuda.is_available())"
```

#### If you have Apple Silicon
After poetry install, you need to install openssl (raytune) for grpcio dependency:
```
pip uninstall grpcio
conda install grpcio
conda activate
```
Check out https://docs.ray.io/en/master/ray-overview/installation.html#m1-mac-apple-silicon-support for more details.
