#!/usr/bin/env python

import argparse
from astropy.coordinates import SkyCoord
from astropy.time import Time
from copy import deepcopy
import datetime
import matplotlib.pyplot as plt
import multiprocessing as mp

# from multiprocessing.pool import ThreadPool
import numpy as np
import os
import pandas as pd
import pathlib
from penquins import Kowalski
import sys
from tqdm.auto import tqdm
import warnings

from tails.efficientdet import Tails
from tails.image import Dvoika, Troika
from tails.utils import load_config, plot_stack, preprocess_stack

os.environ[
    "LD_LIBRARY_PATH"
] = "/usr/local/cuda-10.1/lib64:/usr/local/cuda/extras/CUPTI/lib64"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# or run on CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

os.environ["DISPLAY"] = "localhost:0.0"

# ignore warnings
warnings.simplefilter("ignore")


N_CPU = mp.cpu_count()


def fetch_data(_nsp):
    try:
        name, secrets, p_base = _nsp
        path_base = (p_base / "data").resolve()
        Troika(
            path_base=str(path_base),
            name=name,
            secrets=secrets,
            verbose=False,
            fetch_data_only=True,
        )
    except Exception as e:
        print(e)
        pass


def process_ccd_quad(
    name,
    p_date,
    checkpoint,
    model,
    secrets,
    nthreads: int = N_CPU,
    score_threshold: float = 0.5,
    cleanup="none",
    **kwargs,
):
    # init Troika and fetch image data from IPAC or, if available, NERSC
    try:
        box_size_pix = 256  # @param

        # print(name)
        n_c = model.inputs[0].shape[-1]

        if n_c == 2:
            stack_class = Dvoika
        elif n_c == 3:
            stack_class = Troika
        else:
            raise ValueError(
                "bad n_c: only know how to operate on duplets and triplets"
            )

        path_base = pathlib.Path("./data").resolve()
        stack = stack_class(
            path_base=str(path_base), name=name, secrets=secrets, verbose=False
        )

        # reproject ref
        ref_projected = stack.reproject_ref2sci(how="swarp", nthreads=nthreads)
        # tessellate
        xboxes, yboxes = stack.tessellate_boxes(box_size_pix=box_size_pix, offset=20)

        tessellation = []

        for i, xbox in enumerate(xboxes):
            for j, ybox in enumerate(yboxes):
                # stack and preprocess triplet
                if n_c == 2:
                    s = np.stack(
                        [
                            stack.sci[xbox[0] : xbox[1], ybox[0] : ybox[1]],
                            ref_projected[xbox[0] : xbox[1], ybox[0] : ybox[1]],
                        ],
                        axis=2,
                    )
                elif n_c == 3:
                    s = np.stack(
                        [
                            stack.sci[xbox[0] : xbox[1], ybox[0] : ybox[1]],
                            ref_projected[xbox[0] : xbox[1], ybox[0] : ybox[1]],
                            stack.zogy[xbox[0] : xbox[1], ybox[0] : ybox[1]],
                        ],
                        axis=2,
                    )
                else:
                    raise ValueError(
                        "bad n_c: only know how to operate on duplets and triplets"
                    )

                s_raw = deepcopy(s)
                preprocess_stack(s)
                tessellation.append(
                    {
                        "i": i,
                        "j": j,
                        "xbox": xbox,
                        "ybox": ybox,
                        "stack": s,
                        "stack_raw": s_raw,
                    }
                )

        stacks = np.array([t["stack"] for t in tessellation])

        predictions = model.predict(stacks)
        # print(predictions[0], predictions.shape)

        # detections
        dets = []
        score_threshold = score_threshold
        ind = np.where(predictions[:, 0].flatten() > score_threshold)[0]
        for ni, ii in enumerate(ind):
            x_o, y_o = predictions[ii, 1:] * stacks.shape[1]
            # save png
            plot_stack(
                # tessellation[ii]['stack'],
                tessellation[ii]["stack_raw"],
                reticles=((x_o, y_o),),
                w=6,
                h=2,
                dpi=360,
                save=str(p_date / f"{name}_{ni}.png"),
                # cmap=cmr.arctic,
                # cmap=plt.cm.viridis,
                # cmap=plt.cm.cividis,
                cmap=plt.cm.bone,
            )
            # save npy:
            np.save(str(p_date / f"{name}_{ni}.npy"), tessellation[ii]["stack"])

            x, y = tessellation[ii]["ybox"][0] + x_o, tessellation[ii]["xbox"][0] + y_o
            ra, dec = stack.pix2world_sci(x, y)
            c = SkyCoord(ra=ra, dec=dec, unit="deg")
            radecstr = c.to_string(style="hmsdms")
            t = Time(stack.header_sci["OBSJD"], format="jd")
            jd = float(t.jd)
            dt = t.datetime
            det = {
                "id": name,
                "ni": ni,
                "jd": jd,
                "datestr": f"{dt.year} {dt.month} {dt.day + (dt.hour + dt.minute / 60.0 + dt.second / 3600.0)/24}",
                "p": float(predictions[ii, 0]),
                "x": x,
                "y": y,
                "ra": ra,
                "dec": dec,
                "radecstr": radecstr,
                "tails_v": checkpoint,
            }
            dets.append(det)

        if len(dets) > 0:
            # print(dets)
            df_dets = pd.DataFrame.from_records(dets)
            df_dets.to_csv(str(p_date / f"{name}.csv"), index=False)

        if cleanup.lower() != "none":
            cleanup_ref = True if cleanup.lower() in ("all", "ref") else False
            cleanup_sci = True if cleanup.lower() in ("all", "sci") else False
            stack.cleanup(ref=cleanup_ref, sci=cleanup_sci)

        return True

    except Exception as e:
        print(str(e))
        return False


def run(arguments):
    p_base = pathlib.Path(arguments.output_base_path)

    config = load_config(arguments.config)

    # build model and load weights
    # checkpoint = 'checkpoints-loc/tails'
    # checkpoint = 'checkpoints-loc-neg/tails'
    # checkpoint = 'checkpoints-loc-neg-2/tails'
    # checkpoint = 'checkpoints-loc-neg-3/tails'
    # checkpoint = 'checkpoints-loc-l1-loss/tails'
    checkpoint = arguments.checkpoint

    model = Tails()
    model.load_weights(checkpoint).expect_partial()

    score_threshold = arguments.score_threshold
    if not (0 <= score_threshold <= 1):
        raise ValueError("score_threshold must be (0 <= score_threshold <=1)")

    nthreads = arguments.nthreads
    if not (1 <= nthreads <= N_CPU):
        raise ValueError(f"nthreads must be (1 <= nthreads <={N_CPU})")

    cleanup = arguments.cleanup
    if cleanup not in ("all", "none", "ref", "sci"):
        raise ValueError("cleanup value not in ('all', 'none', 'ref', 'sci')")

    if arguments.single_image:
        datestr = arguments.single_image[4:12]
        date = datetime.datetime.strptime(datestr, "%Y%m%d")
        print(date)

        p_date = p_base / "runs" / datestr
        if not p_date.exists():
            p_date.mkdir(parents=True, exist_ok=True)

        names = [arguments.single_image]

    else:
        if arguments.date:
            datestr = arguments.date
        else:
            datestr = datetime.datetime.utcnow().strftime("%Y%m%d")

        date = datetime.datetime.strptime(datestr, "%Y%m%d")
        print(date)

        p_date = p_base / "runs" / datestr
        if not p_date.exists():
            p_date.mkdir(parents=True, exist_ok=True)

        # setup

        # from astroplan import download_IERS_A
        # download_IERS_A()

        kowalski = Kowalski(
            username=config["kowalski"]["username"],
            password=config["kowalski"]["password"],
        )

        q = {
            "query_type": "find",
            "query": {
                "catalog": "ZTF_ops",
                "filter": {
                    # 'qcomment': {'$regex': 'Twilight'},
                    # 'field': 773,
                    "jd_start": {"$gt": Time(date).jd, "$lt": Time(date).jd + 1}
                },
                "projection": {"_id": 0, "fileroot": 1},
            },
        }

        if arguments.twilight:
            q["query"]["filter"]["qcomment"] = {"$regex": "Twilight"}

        r = kowalski.query(q).get("data", dict())
        fileroots = sorted([e["fileroot"] for e in r])

        names = [
            f"{fileroot}_c{ccd:02d}_o_q{quad:1d}"
            for fileroot in fileroots
            for ccd in range(1, 17)
            for quad in range(1, 5)
        ]

    # fetch data first
    nsp = [(name, config, p_base) for name in names]
    with mp.Pool(processes=N_CPU) as pool:
        list(tqdm(pool.imap(fetch_data, nsp), total=len(nsp)))
    # fixme:
    # for nsp_i in tqdm(nsp[:3]):
    # for nsp_i in tqdm(nsp):
    #     fetch_data(nsp_i)

    # fixme:
    # for name in tqdm(names[:3]):
    for name in tqdm(names):
        process_ccd_quad(
            name=name,
            p_date=p_date,
            checkpoint=checkpoint,
            model=model,
            secrets=config,
            nthreads=nthreads,
            score_threshold=score_threshold,
            cleanup=cleanup,
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title="commands", dest="command")

    commands = [
        ("run", "🚀 Run Tails, run!"),
        ("help", "Print this message"),
    ]

    parsers = dict()
    for (cmd, desc) in commands:
        parsers[cmd] = subparsers.add_parser(cmd, help=desc)

    parsers["run"].add_argument(
        "--config",
        type=str,
        default="../config.yaml",
        help="path to yaml file with configs and secrets",
    )
    parsers["run"].add_argument(
        "--checkpoint",
        type=str,
        default="../nb/checkpoints-loc-l1-loss/tails",
        help="model weights",
    )
    parsers["run"].add_argument(
        "--score_threshold",
        type=float,
        default=0.6,
        help="score threshold for declaring a candidate plausible (0 <= score_threshold <= 1)",
    )
    parsers["run"].add_argument("--date", type=str, help="UTC date string YYYYMMDD")
    parsers["run"].add_argument(
        "--twilight", action="store_true", help="Run on the Twilight survey data only"
    )
    parsers["run"].add_argument(
        "--single_image",
        type=str,
        help="Run on single ccd-quad image, e.g. ztf_20200810193681_000635_zr_c09_o_q2",
    )
    parsers["run"].add_argument(
        "--output_base_path", type=str, default="./", help="base path for output"
    )
    parsers["run"].add_argument(
        "--nthreads",
        type=int,
        default=N_CPU,
        help="number of threads for image re-projecting",
    )
    parsers["run"].add_argument(
        "--cleanup", type=str, default="none", help="delete raw data: ref|sci|all|none"
    )

    args = parser.parse_args()
    if args.command is None or args.command == "help":
        parser.print_help()
    else:
        getattr(sys.modules[__name__], args.command)(args)
