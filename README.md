# hdmm
Source code for HDMM

# Setup

First, clone the [ektelo repo](https://github.com/dpcomp-org/ektelo.git) and follow setup instructions.

## Ubutnu
```bash
sudo apt update
sudo apt-get install python3-venv
```

## Environment
```bash
export HDMM_HOME=$HOME/Documents/hdmm
export PYTHON_HOME=$HOME/Virtualenvs/PyHDMM
export PYTHONPATH=$PYTHONPATH:$HDMM_HOME/src
export PYTHONPATH=$PYTHONPATH:$HOME/Documents/ektelo
```

## Initialization
```bash
python3 -m venv $PYTHON_HOME
source $PYTHON_HOME/bin/activate
pip install -r $HDMM_HOME/resources/requirements.txt
```

## Testing

```bash
nosetests
```
