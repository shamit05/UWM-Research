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

#### If you have CUDA enabled GPU
After poetry install, you need to remove pytorch-cpu and install pytorch with CUDA support:
```
poetry shell
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir
```
