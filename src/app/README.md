# priv-publish
Website for private data publication using Ektelo

# Setup

First, clone the following repos to `$HOME/Documents`.

* `https://github.com/zeeqy/hdmm.git`

Follow the setup instructions in the ektlo repository.

## Ubutnu
```bash
sudo apt update
sudo apt-get install python3-venv

## Environment
```bash
export PRIV_HOME=$HOME/Documents/priv-publish
export PRIV_DATA=$PRIV_HOME/data/uploads
export PYTHON_HOME=$HOME/Virtualenvs/PyPriv
export PYTHONPATH=$PYTHONPATH:PRIV_HOME/app
export PYTHONPATH=$PYTHONPATH:$HOME/Documents/hdmm
export PYTHONPATH=$PYTHONPATH:$HOME/Documents/ektelo
export PYTHONPATH=$PYTHONPATH:$HOME/Documents/private-pgm/src
export FLASK_APP=app.py
```

## Initialization
```bash
python3 -m venv $PYTHON_HOME
source $PYTHON_HOME/bin/activate
pip install -r $PRIV_HOME/resources/requirements.txt
```

# Running the site
```bash
mkdir /tmp/priv_publish
cd $PRIV_HOME/app
python app.py --mode production --ip 54.165.230.129 --port 8555
```
where X.X.X.X is the IP address of the server.
