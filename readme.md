# Tails: chasing comets with the Zwicky Transient Facility and Deep Learning

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4563226.svg)](https://doi.org/10.5281/zenodo.4563226)
[![arXiv](https://img.shields.io/badge/arXiv-2102.13352-brightgreen)](https://arxiv.org/abs/2102.13352)

Tails is a deep-learning framework for the identification and localization of comets in the image data
of the [Zwicky Transient Facility (ZTF)](https://ztf.caltech.edu), a robotic optical sky survey
at the [Palomar Observatory](https://sites.astro.caltech.edu/palomar/about/) in California, USA.

Tails uses a custom EfficientDet-based architecture and is thus capable of finding comets in single images
in near real time, rather than requiring multiple epochs as with traditional methods.
In production, we have observed 99% recall, <0.01% false positive rate,
and 1-2 pixel root mean square error in the predicted position.

Tails enabled the first AI-assisted discovery of a comet -
[C/2020 T2](https://minorplanetcenter.net/mpec/K20/K20UH0.html).

## Install and run Tails

Clone the repository:

```bash
git clone https://github.com/dmitryduev/tails.git && cd tails
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
python run_tails.py \
  --config=../config.defaults.yaml \
  --checkpoint=../models/tails-20210107/tails \
  --score_threshold=0.5 \
  --cleanup=none \
  --single_image=ztf_20191014495961_000570_zr_c05_o_q3
```

Check out the `runs/20191014` directory for the output:
- A csv file with the detection metadata
- A 256x256 pix cutout image triplet (epochal, reference, and difference) containing the detection: an `.npy` file and a `.png` visualization:
![ztf_20191014495961_000570_zr_c05_o_q3_0](https://user-images.githubusercontent.com/7557205/105624917-34802880-5dda-11eb-959e-8386142ac4a4.png)

## ZTF sentinel service

A containerized service that monitors
[Kowalski](https://kowalski.caltech.edu)/[IRSA](https://irsa.ipac.caltech.edu/) for new ZTF data,
executes Tails on them, and optionally posts the identified candidates to [Fritz](https://fritz.science).

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

---

The Sentinel:
- Managed with the `sentinel.py` utility.
- Implemented as a containerized service, orchestrated with `docker-compose`.
- Monitors the `ZTF_ops` collection on `Kowalski` for new ZTF data (Twilight only by default).
- Uses `dask.distributed` to process individual ZTF image frames (ccd-quads).
Each worker is initialized with a `TailsWorker` instance that maintains a `Fritz` connection and preloads Tails.
The candidate comet detections, if any, are posted to `Fritz` together with auto-annotations
(cross-matches from the MPC and SkyBot) and auxiliary data.
  - By default, spins up a `dask.distributed.LocalCluster`, whose parameters are defined in `config.yaml`,
  and a sentinel process, both managed with `supervisor` inside the `tails_sentinel_1` Docker container.
  To use an external `dask.distributed` cluster, the user needs to simply provide its host and the scheduler port
  and remove the `dask-cluster` `supervisor` program in the config.


On Fritz, the posted results look like the following:

<table>
<thead>
<tr>
<th>Candidates page</th>
<th>Source page</th>
</tr>
</thead>
<tbody>
<tr>
<td>
<img width="1605" alt="fritz-candidates" src="https://user-images.githubusercontent.com/7557205/104243874-2283aa80-5416-11eb-98e1-8f5cc5224d9e.png">
</td>
<td>
<img width="1606" alt="fritz-source" src="https://user-images.githubusercontent.com/7557205/104243884-29122200-5416-11eb-9c76-3f727f22683b.png">
</td>
</tr>
</tbody>
</table>
