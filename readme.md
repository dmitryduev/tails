# Tails: chasing comets with the Zwicky Transient Facility and Deep Learning

## Install and run Tails

Clone the repository:

```bash
git clone https://github.com/dmitryduev/tails-pub.git tails && cd tails
```

You may want to create a virtual environment before proceeding.

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
gsutil cp -r -n gs://dmitryduev/tails/models models/
```

## ZTF watchdog service

todo:
