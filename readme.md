# Tails: chasing comets with the Zwicky Transient Facility and Deep Learning

## Install and run Tails

Clone the repository:

```bash
git clone https://github.com/dmitryduev/tails-pub.git tails && cd tails
```

Before proceeding, you may want to create/activate a virtual environment, for example:

```bash
python -m venv tails-env
source tails-env/bin/activate
```

Install [`swarp`](https://www.astromatic.net/software/swarp). For example, with `conda`:

```bash
conda install -c conda-forge astromatic-swarp
```

Install `Tails`:

```bash
python setup.py install
```

Fetch pre-trained models:

```bash
mkdir models
gsutil -m cp -r -n gs://tails-models/* models/
```

Run Tails on a (publicly accessible) ZTF frame:

```bash
cd scripts
python run_tails.py run \
  --config=../config.defaults.yaml \
  --checkpoint=../models/tails-20210107/tails \
  --score_threshold=0.5 \
  --cleanup=none \
  --single_image=ztf_20191014495961_000570_zr_c05_o_q3
```

Check out the `runs/20191014` directory for the output.

## ZTF watchdog service

Fetch models from GCP:

```bash
./watchdog.py fetch-models
```

Spin up:

```bash
./watchdog.py run
```
