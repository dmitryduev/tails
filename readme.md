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

Run Tails on a (publicly accessible) ZTF observation of 2I/Borisov:

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

## ZTF sentinel service

A containerized service that monitors
[Kowalski](https://kowalski.caltech.edu)/[IRSA](https://irsa.ipac.caltech.edu/) for new ZTF data,
executes Tails on them, and posts the identified candidates to [Fritz](https://fritz.science).

Requires Kowalski and IRSA accounts, see the [config file](config.defaults.yaml).
Additionally, a Fritz account is required if `sentinel.app.post_to_fritz` is set to `true`.

Fetch models from GCP:

```bash
./sentinel.py fetch-models
```

Spin up:

```bash
./sentinel.py up
```

Shut down:

```bash
./sentinel.py down
```
