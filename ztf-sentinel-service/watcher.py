import arrow
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.visualization import (
    # AsymmetricPercentileInterval,
    ZScaleInterval,
    # LinearStretch,
    # LogStretch,
    # ImageNormalize,
)
import base64
from copy import deepcopy
import dask.distributed
import fire
import io
import json
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import pathlib
from penquins import Kowalski
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import time
from typing import Mapping, Optional, Sequence, Union

from tails.efficientdet import Tails
from tails.image import Dvoika, Troika
from tails.utils import load_config, log, plot_stack, preprocess_stack, MPC, IMCCE

from utilities import init_db, Mongo, timer, uid


config = load_config(config_file="/app/config.yaml")


DEFAULT_TIMEOUT = 5  # seconds


class TimeoutHTTPAdapter(HTTPAdapter):
    """
    HTTPAdapter with built-in default timeouts
    """

    def __init__(self, *args, **kwargs):
        self.timeout = DEFAULT_TIMEOUT
        if "timeout" in kwargs:
            self.timeout = kwargs["timeout"]
            del kwargs["timeout"]
        super().__init__(*args, **kwargs)

    def send(self, request, **kwargs):
        timeout = kwargs.get("timeout")
        if timeout is None:
            kwargs["timeout"] = self.timeout
        return super().send(request, **kwargs)


class TailsWorker:
    """
    Helper class that is instantiated on dask.distributed.Worker's used for processing

    - Maintains MongoDB, MPC+IMCCE, and Fritz connections
    - Pre-loads Tails (the model itself)
    """

    def __init__(self, **kwargs):
        self.verbose = kwargs.get("verbose", 2)
        self.config = config

        # mongo connection
        self.mongo = Mongo(
            host=config["sentinel"]["database"]["host"],
            port=config["sentinel"]["database"]["port"],
            username=config["sentinel"]["database"]["username"],
            password=config["sentinel"]["database"]["password"],
            db=config["sentinel"]["database"]["db"],
            verbose=self.verbose,
        )

        # requests.Session to talk to Fritz
        self.session = requests.Session()
        self.session_headers = {
            "Authorization": f"token {config['sentinel']['fritz']['token']}"
        }

        retries = Retry(
            total=5,
            backoff_factor=2,
            status_forcelist=[405, 429, 500, 502, 503, 504],
            method_whitelist=["HEAD", "GET", "PUT", "POST", "PATCH"],
        )
        adapter = TimeoutHTTPAdapter(timeout=5, max_retries=retries)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        # init MPC and SkyBot - services used for cross-matching identified candidates with known Solar system objects
        self.mpc = MPC(verbose=self.verbose)
        self.imcce = IMCCE(verbose=self.verbose)

        # check Fritz connection
        try:
            with timer("Checking connection to Fritz", self.verbose > 1):
                response = self.api_fritz("GET", "/api/sysinfo")
            if response.json()["status"] == "success":
                log("Fritz connection OK")
            else:
                log("Failed to connect to Fritz")
                raise ValueError("Failed to connect to Fritz")
        except Exception as e:
            log(e)

        # load Tails
        self.path = pathlib.Path(config["sentinel"]["app"]["path"])

        self.checkpoint = f"/app/models/{config['sentinel']['app']['checkpoint']}/tails"

        self.model = Tails()
        self.model.load_weights(self.checkpoint).expect_partial()

        self.score_threshold = float(
            config["sentinel"]["app"].get("score_threshold", 0.5)
        )
        if not (0 <= self.score_threshold <= 1):
            raise ValueError(
                "sentinel.app.score_threshold must be (0 <= score_threshold <=1), check config"
            )

        self.cleanup = config["sentinel"]["app"]["cleanup"]
        if self.cleanup not in ("all", "none", "ref", "sci"):
            raise ValueError(
                "sentinel.app.cleanup value not in ('all', 'none', 'ref', 'sci'), check config"
            )

        self.num_threads = mp.cpu_count()

    def api_fritz(self, method: str, endpoint: str, data=None):
        """Make an API call to a Fritz/SkyPortal instance

        :param method:
        :param endpoint:
        :param data:
        :return:
        """
        method = method.lower()
        methods = {
            "head": self.session.head,
            "get": self.session.get,
            "post": self.session.post,
            "put": self.session.put,
            "patch": self.session.patch,
            "delete": self.session.delete,
        }

        if endpoint is None:
            raise ValueError("Endpoint not specified")
        if method not in ["head", "get", "post", "put", "patch", "delete"]:
            raise ValueError(f"Unsupported method: {method}")

        if method == "get":
            response = methods[method](
                f"{config['sentinel']['fritz']['protocol']}://"
                f"{config['sentinel']['fritz']['host']}:{config['sentinel']['fritz']['port']}"
                f"{endpoint}",
                params=data,
                headers=self.session_headers,
            )
        else:
            response = methods[method](
                f"{config['sentinel']['fritz']['protocol']}://"
                f"{config['sentinel']['fritz']['host']}:{config['sentinel']['fritz']['port']}"
                f"{endpoint}",
                json=data,
                headers=self.session_headers,
            )

        return response

    def process_frame(self, frame: str) -> Mapping[str, Union[Sequence, str]]:
        """
        Process a ZTF observation
        - Fetch the epochal science, reference, and difference image for a ZTF observation
        - Re-project the reference image onto the epochal science image
        - Stack all three together
        - Tessellate into a 13x13 grid of overlapping tiles of size 256x256 pix
        - Execute Tails on each tile
        - For each candidate detection, query MPC and IMCCE for cross-matches with known SS objects

        :param frame: ZTF frame id formatted as in ztf_20201114547512_000319_zr_c01_o_q1, where
                      20201114: date string
                      547512: time tag
                      000319: zero-padded field id
                      zr: filter code <zg|zr|zi>
                      c01: zero-padded CCD id c [1, 16]
                      q1: quadrant id c [1, 4]
        :return: detections (if any), packed into the result dict
        """
        path_base = self.path.resolve()

        date_string = frame.split("_")[1][:8]
        path_date = self.path / "runs" / date_string

        if not path_date.exists():
            path_date.mkdir(parents=True, exist_ok=True)

        results = {"detections": [], "status": "success"}

        try:
            box_size_pix = config["sentinel"]["app"]["box_size_pix"]

            dim_last = self.model.inputs[0].shape[-1]

            if dim_last == 2:
                stack_class = Dvoika
            elif dim_last == 3:
                stack_class = Troika
            else:
                raise ValueError(
                    "bad dim_last: only know how to operate on duplets and triplets"
                )

            with timer("Making stack", self.verbose > 1):
                stack = stack_class(
                    path_base=str(path_base),
                    name=frame,
                    secrets=self.config,
                    verbose=self.verbose,
                )

            # re-project ref
            with timer("Re-projecting", self.verbose > 1):
                ref_projected = stack.reproject_ref2sci(
                    how="swarp", nthreads=self.num_threads
                )
            # tessellate
            with timer("Tessellating", self.verbose > 1):
                xboxes, yboxes = stack.tessellate_boxes(
                    box_size_pix=box_size_pix, offset=20
                )

            tessellation = []

            with timer("Preprocessing tiles", self.verbose > 1):
                for i, xbox in enumerate(xboxes):
                    for j, ybox in enumerate(yboxes):
                        # stack and preprocess image stack
                        if dim_last == 2:
                            s = np.stack(
                                [
                                    stack.sci[xbox[0] : xbox[1], ybox[0] : ybox[1]],
                                    ref_projected[xbox[0] : xbox[1], ybox[0] : ybox[1]],
                                ],
                                axis=2,
                            )
                        elif dim_last == 3:
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
                                "bad dim_last: only know how to operate on duplets and triplets"
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

            with timer("Running Tails", self.verbose > 1):
                predictions = self.model.predict(stacks)
            # log(predictions[0], predictions.shape)

            # URL to fetch original images from IPAC
            sci_name = stack.name + "_sciimg.fits"
            tmp = sci_name.split("_")[1]
            y, p1, p2 = tmp[:4], tmp[4:8], tmp[8:]
            sci_ipac_url = os.path.join(
                config["irsa"]["url"], "sci", y, p1, p2, sci_name
            )

            # detections
            ind = np.where(predictions[:, 0].flatten() > self.score_threshold)[0]
            for ni, ii in enumerate(ind):
                x_o, y_o = predictions[ii, 1:] * stacks.shape[1]
                # save png
                with timer("Saving png", self.verbose > 1):
                    plot_stack(
                        # tessellation[ii]['stack'],
                        tessellation[ii]["stack_raw"],
                        reticles=((x_o, y_o),),
                        w=6,
                        h=2,
                        dpi=360,
                        save=str(path_date / f"{frame}_{ni}.png"),
                        # cmap=cmr.arctic,
                        # cmap=plt.cm.viridis,
                        # cmap=plt.cm.cividis,
                        cmap="bone",
                    )
                # save npy:
                with timer("Saving npy", self.verbose > 1):
                    np.save(
                        str(path_date / f"{frame}_{ni}.npy"), tessellation[ii]["stack"]
                    )

                x, y = (
                    tessellation[ii]["ybox"][0] + x_o,
                    tessellation[ii]["xbox"][0] + y_o,
                )
                ra, dec = stack.pix2world_sci(x, y)
                sky_coord = SkyCoord(ra=ra, dec=dec, unit="deg")
                radecstr = sky_coord.to_string(style="hmsdms")
                epoch = Time(stack.header_sci["OBSJD"], format="jd")
                jd = float(epoch.jd)
                dt = epoch.datetime

                # make cutouts for posting to Fritz
                with timer("Making cutouts", self.verbose > 1):
                    cutout_xbox, cutout_ybox, _, _ = stack.make_box(
                        ra=ra, dec=dec, box_size_pix=63, min_offset=0, random=False
                    )
                    if self.verbose:
                        log((cutout_xbox, cutout_ybox))

                    cutouts = np.stack(
                        [
                            stack.sci[
                                cutout_ybox[0] : cutout_ybox[1],
                                cutout_xbox[0] : cutout_xbox[1],
                            ],
                            ref_projected[
                                cutout_ybox[0] : cutout_ybox[1],
                                cutout_xbox[0] : cutout_xbox[1],
                            ],
                            stack.zogy[
                                cutout_ybox[0] : cutout_ybox[1],
                                cutout_xbox[0] : cutout_xbox[1],
                            ],
                        ],
                        axis=2,
                    )

                # query MPC and SkyBot
                x_match_query = {
                    "id": f"{frame}_{ni}",
                    "position": sky_coord,
                    "radius": 2,
                    "epoch": epoch,
                    "timeout": 30,
                }
                with timer("Querying MPC", self.verbose > 1):
                    x_match_mpc = self.mpc.query(query=x_match_query)
                with timer("Querying IMCCE", self.verbose > 1):
                    x_match_skybot = self.imcce.query(query=x_match_query)
                if self.verbose:
                    log(x_match_mpc)
                    log(x_match_skybot)

                detection = {
                    "id": frame,
                    "ni": ni,
                    "jd": jd,
                    "datestr": f"{dt.year} {dt.month} {dt.day + (dt.hour + dt.minute / 60 + dt.second / 3600) / 24}",
                    "p": float(predictions[ii, 0]),
                    "x": x,
                    "y": y,
                    "ra": ra,
                    "dec": dec,
                    "radecstr": radecstr,
                    "tails_v": self.config["sentinel"]["app"]["checkpoint"],
                    "sci_ipac_url": sci_ipac_url,
                    "dif_ipac_url": sci_ipac_url.replace(
                        "sciimg.fits", "scimrefdiffimg.fits.fz"
                    ),
                    "cutouts": cutouts,
                    "x_match_mpc": x_match_mpc,
                    "x_match_skybot": x_match_skybot,
                }
                results["detections"].append(detection)

            if len(results["detections"]) > 0:
                df_dets = pd.DataFrame.from_records(results["detections"])
                df_dets.to_csv(str(path_date / f"{frame}.csv"), index=False)

            if self.cleanup.lower() != "none":
                cleanup_ref = True if self.cleanup.lower() in ("all", "ref") else False
                cleanup_sci = True if self.cleanup.lower() in ("all", "sci") else False
                stack.cleanup(ref=cleanup_ref, sci=cleanup_sci)

        except Exception as e:
            log(e)
            results["status"] = "error"

        return results

    def post_candidate(self, oid: str, detection: Mapping):
        """
        Post a candidate comet detection to Fritz

        :param oid: candidate id
        :param detection: from self.process_frame
        :return:
        """
        candid = f"{detection['id']}_{detection['ni']}"
        meta = {
            "id": oid,
            "ra": detection["ra"],
            "dec": detection["dec"],
            "score": detection["p"],
            "filter_ids": [self.config["sentinel"]["fritz"]["filter_id"]],
            "origin": candid,
            "passed_at": arrow.utcnow().format("YYYY-MM-DDTHH:mm:ss.SSS"),
        }
        if self.verbose > 1:
            log(meta)

        with timer(
            f"Posting metadata of {oid} {candid} to Fritz",
            self.verbose > 1,
        ):
            response = self.api_fritz("POST", "/api/candidates", meta)
        if response.json()["status"] == "success":
            log(f"Posted {oid} {candid} metadata to Fritz")
        else:
            log(f"Failed to post {oid} {candid} metadata to Fritz")
            log(response.json())

    def post_annotations(self, oid: str, detection: Mapping):
        """
        Post candidate annotations to Fritz

        :param oid: candidate id
        :param detection: from self.process_frame
        :return:
        """
        candid = f"{detection['id']}_{detection['ni']}"
        data = {
            "candid": candid,
            "score": detection["p"],
            "jd": detection["jd"],
            "datestr": detection["datestr"],
            "x": detection["x"],
            "y": detection["y"],
            "ra": detection["ra"],
            "dec": detection["dec"],
            "radecstr": detection["radecstr"],
            "tails_v": self.config["sentinel"]["app"]["checkpoint"],
            "sci_ipac_url": detection["sci_ipac_url"],
            "dif_ipac_url": detection["dif_ipac_url"],
        }

        if (
            detection["x_match_mpc"]["status"] == "success"
            and len(detection["x_match_mpc"]["data"]) > 0
        ):
            nearest_match = detection["x_match_mpc"]["data"][0]
            data["mpc_nearest_id"] = nearest_match.get("designation")
            data["mpc_nearest_offset_arcsec"] = nearest_match.get("offset__arcsec")
            data["mpc_nearest_orbit"] = nearest_match.get("orbit")

        if (
            detection["x_match_skybot"]["status"] == "success"
            and len(detection["x_match_skybot"]["data"]) > 0
        ):
            nearest_match = detection["x_match_skybot"]["data"][0]
            data["imcce_nearest_id"] = str(nearest_match["Name"])
            data["imcce_nearest_offset_arcsec"] = float(
                nearest_match["centerdist"].value
            )
            data["imcce_nearest_Vmag"] = float(nearest_match["V"].value)

        annotations = {
            "obj_id": oid,
            "origin": "tails:twilight",
            "data": data,
            "group_ids": [self.config["sentinel"]["fritz"]["group_id"]],
        }

        if self.verbose:
            log(annotations)

        with timer(
            f"Posting annotation for {oid} {candid} to Fritz",
            self.verbose > 1,
        ):
            response = self.api_fritz("POST", "/api/annotation", annotations)
        if response.json()["status"] == "success":
            log(f"Posted {oid} {candid} annotation to Fritz")
        else:
            log(f"Failed to post {oid} {candid} annotation to Fritz")
            log(response.json())

    @staticmethod
    def make_thumbnail(oid: str, detection: Mapping, thumbnail_type: str):
        """Convert lossless FITS cutouts from ZTF images into PNGs

        :param oid: Fritz obj id
        :param detection: Tails detection dict
        :param thumbnail_type: <new|ref|sub>
        :return:
        """
        stack = deepcopy(detection["cutouts"])

        if thumbnail_type == "ref":
            index = 1
        elif thumbnail_type == "sub":
            index = 2
        else:
            index = 0
        cutout_data = stack[..., index]
        # flip up/down
        # cutout_data = np.flipud(cutout_data)
        buff = io.BytesIO()
        plt.close("all")
        fig = plt.figure()
        fig.set_size_inches(4, 4, forward=False)
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        fig.add_axes(ax)

        # replace nans with median:
        img = np.array(cutout_data)
        # replace dubiously large values
        xl = np.greater(np.abs(img), 1e20, where=~np.isnan(img))
        if img[xl].any():
            img[xl] = np.nan
        if np.isnan(img).any():
            median = float(np.nanmean(img.flatten()))
            img = np.nan_to_num(img, nan=median)

        interval = ZScaleInterval(nsamples=img.shape[0] * img.shape[1])
        limits = interval.get_limits(img)
        ax.imshow(img, origin="upper", cmap="bone", vmin=limits[0], vmax=limits[1])
        plt.savefig(buff, dpi=42)

        buff.seek(0)
        plt.close("all")

        thumb = {
            "obj_id": oid,
            "data": base64.b64encode(buff.read()).decode("utf-8"),
            "ttype": thumbnail_type,
        }

        return thumb

    def post_thumbnails(self, oid: str, detection: Mapping):
        """
        Post Fritz-style (~1'x1') cutout images centered around the detected candidate to Fritz

        :param oid: Fritz obj id
        :param detection: Tails detection dict
        :return:
        """
        candid = f"{detection['id']}_{detection['ni']}"

        for thumbnail_type in ("new", "ref", "sub"):
            with timer(
                f"Making {thumbnail_type} thumbnail for {oid} {candid}",
                self.verbose > 1,
            ):
                thumb = self.make_thumbnail(oid, detection, thumbnail_type)

            with timer(
                f"Posting {thumbnail_type} thumbnail for {oid} {candid} to Fritz",
                self.verbose > 1,
            ):
                response = self.api_fritz("POST", "/api/thumbnail", thumb)

            if response.json()["status"] == "success":
                log(f"Posted {oid} {candid} {thumbnail_type} cutout to Fritz")
            else:
                log(f"Failed to post {oid} {candid} {thumbnail_type} cutout to Fritz")
                log(response.json())

    def post_comments(self, oid: str, detection: Mapping):
        """
        Post auxiliary candidate info to Fritz as comments to the respective object

        :param oid: Fritz obj id
        :param detection: Tails detection dict
        :return:
        """
        candid = f"{detection['id']}_{detection['ni']}"

        # post full-sized cutouts
        date_string = detection["id"].split("_")[1][:8]
        path_date = self.path / "runs" / date_string

        with open(str(path_date / f"{candid}.png"), "rb") as f:
            file_content = f.read()
        cutouts_png = base64.b64encode(file_content).decode("utf-8")
        comment = {
            "obj_id": oid,
            "text": "Full-sized cutouts (256x256 px)",
            "group_ids": [self.config["sentinel"]["fritz"]["group_id"]],
            "attachment": {
                "body": cutouts_png,
                "name": f"{candid}.png",
            },
        }

        response = self.api_fritz("POST", "/api/comment", comment)
        if response.json()["status"] == "success":
            log(f"Posted {oid} {candid} png to Fritz")
        else:
            log(f"Failed to post {oid} {candid} png to Fritz")
            log(response.json())

        # post cross-matches from MPC and IMCCE
        cross_matches = {"MPC": None, "IMCCE": None}
        if (
            detection["x_match_mpc"]["status"] == "success"
            and len(detection["x_match_mpc"]["data"]) > 0
        ):
            cross_matches["MPC"] = detection["x_match_mpc"]["data"]
        if (
            detection["x_match_skybot"]["status"] == "success"
            and len(detection["x_match_skybot"]["data"]) > 0
        ):
            cross_matches["IMCCE"] = (
                detection["x_match_skybot"]["data"]
                .to_pandas(index=False, use_nullable_int=False)
                .fillna(value="null")
                .to_dict(orient="records")
            )
        comment = {
            "obj_id": oid,
            "text": "MPC and IMCCE cross-match",
            "group_ids": [self.config["sentinel"]["fritz"]["group_id"]],
            "attachment": {
                "body": base64.b64encode(json.dumps(cross_matches).encode()).decode(
                    "utf-8"
                ),
                "name": f"{candid}.json",
            },
        }

        response = self.api_fritz("POST", "/api/comment", comment)
        if response.json()["status"] == "success":
            log(f"Posted {oid} {candid} cross-matches to Fritz")
        else:
            log(f"Failed to post {oid} {candid} cross-matches to Fritz")
            log(response.json())

        # post SCI image URL on IPAC's IRSA
        comment = {
            "obj_id": oid,
            "text": f"[SCI image from IPAC]({detection['sci_ipac_url']})",
            "group_ids": [self.config["sentinel"]["fritz"]["group_id"]],
        }

        response = self.api_fritz("POST", "/api/comment", comment)
        if response.json()["status"] == "success":
            log(f"Posted {oid} {candid} sci image url to Fritz")
        else:
            log(f"Failed to post {oid} {candid} sci image url to Fritz")
            log(response.json())

    def post_detections_to_fritz(self, detections: Sequence[Mapping]):
        """
        Post a list of detections from self.process_frame to Fritz

        :param detections: results["detections"] from self.process_frame
        :return:
        """
        for detection in detections:
            # generate unique object id
            utc = arrow.utcnow()
            prefix = "ZTF" + utc.format("YYYYMMDD") + "_"
            oid = uid(length=6, prefix=prefix)

            self.post_candidate(oid, detection)
            self.post_annotations(oid, detection)
            self.post_thumbnails(oid, detection)
            self.post_comments(oid, detection)


class WorkerInitializer(dask.distributed.WorkerPlugin):
    """
    Create an instance of TailsWorker when initializing dask.distributed.Worker's
    """

    def __init__(self, *args, **kwargs):
        self.tails_worker = None

    def setup(self, worker: dask.distributed.Worker):
        self.tails_worker = TailsWorker()


def process_frame(frame: str):
    """
    Function that is executed on dask.distributed.Worker's to process a ZTF observation

    :param frame: ZTF frame id formatted as in ztf_20201114547512_000319_zr_c01_o_q1, where
                      20201114: date string
                      547512: time tag
                      000319: zero-padded field id
                      zr: filter code <zg|zr|zi>
                      c01: zero-padded CCD id c [1, 16]
                      q1: quadrant id c [1, 4]
    :return:
    """
    # get worker running current task
    worker = dask.distributed.get_worker()
    tails_worker = worker.plugins["worker-init"].tails_worker

    log(f"Processing {frame} on {worker.address}")

    collection = config["sentinel"]["database"]["collection"]

    try:
        # process frame, tessellate, run Tails on individual tiles
        results = tails_worker.process_frame(frame)

        # post results to Fritz, if any
        if results["status"] == "success":
            if config["sentinel"]["app"]["post_to_fritz"]:
                tails_worker.post_detections_to_fritz(results["detections"])
            tails_worker.mongo.db[collection].update_one(
                {"_id": frame}, {"$set": {"status": "success"}}, upsert=True
            )
        else:
            log(f"Processing of {frame} failed")
            tails_worker.mongo.db[collection].update_one(
                {"_id": frame}, {"$set": {"status": "error"}}, upsert=True
            )
    except Exception as e:
        log(e)
        tails_worker.mongo.db[collection].update_one(
            {"_id": frame}, {"$set": {"status": "error"}}, upsert=True
        )


def sentinel(
    utc_start: Optional[str] = None,
    utc_stop: Optional[str] = None,
    twilight: Optional[bool] = False,
    test: Optional[bool] = False,
    verbose: Optional[bool] = False,
):
    """
    ZTF Sentinel service

    - Monitors the ZTF_ops collection on Kowalski for new ZTF data (Twilight only by default).
    - Uses dask.distributed to process individual ZTF image frames (ccd-quads).
      Each worker is initialized with a TailsWorker instance that maintains a Fritz connection and preloads Tails.
      The candidate comet detections, if any, are posted to Fritz together with auto-annotations
      (cross-matches from the MPC and SkyBot) and auxiliary data.

    :param utc_start: UTC start date/time in arrow-parsable format. If not set, defaults to (now - 1h)
    :param utc_stop: UTC stop date/time in arrow-parsable format. If not set, defaults to (now + 1h).
                     If set, program runs once
    :param twilight: process only the data of the ZTF Twilight survey
    :param test: run in test mode
    :param verbose: verbose?
    :return:
    """
    if verbose:
        log("Setting up MongoDB connection")

    init_db(config=config, verbose=verbose)

    mongo = Mongo(
        host=config["sentinel"]["database"]["host"],
        port=config["sentinel"]["database"]["port"],
        username=config["sentinel"]["database"]["username"],
        password=config["sentinel"]["database"]["password"],
        db=config["sentinel"]["database"]["db"],
        verbose=verbose,
    )
    if verbose:
        log("Set up MongoDB connection")

    collection = config["sentinel"]["database"]["collection"]

    # remove dangling entries in the db at startup
    mongo.db[collection].delete_many({"status": "processing"})

    # Configure dask client
    if verbose:
        log("Initializing dask.distributed client")
    dask_client = dask.distributed.Client(
        address=f"{config['sentinel']['dask']['host']}:{config['sentinel']['dask']['scheduler_port']}"
    )

    # init each worker with Worker instance
    if verbose:
        log("Initializing dask.distributed workers")
    worker_initializer = WorkerInitializer()
    dask_client.register_worker_plugin(worker_initializer, name="worker-init")

    if test:
        frame = "ztf_20191014495961_000570_zr_c05_o_q3"
        with timer(f"Submitting frame {frame} for processing", verbose):
            mongo.db[collection].update_one(
                {"_id": frame}, {"$set": {"status": "processing"}}, upsert=True
            )
            future = dask_client.submit(process_frame, frame, pure=True)
            dask.distributed.fire_and_forget(future)
            future.release()
            del future
        return True

    if verbose:
        log("Setting up Kowalski connection")
    kowalski = Kowalski(
        token=config["kowalski"]["token"],
        protocol=config["kowalski"]["protocol"],
        host=config["kowalski"]["host"],
        port=config["kowalski"]["port"],
        verbose=verbose,
    )
    if verbose:
        log(f"Kowalski connection OK: {kowalski.ping()}")

    while True:
        try:
            # monitor the past 24 hours as sometimes there are data processing/posting delays at IPAC
            start = (
                arrow.get(utc_start)
                if utc_start is not None
                else arrow.utcnow().shift(hours=-24)
            )
            stop = (
                arrow.get(utc_stop)
                if utc_stop is not None
                else arrow.utcnow().shift(hours=1)
            )

            if (stop - start).total_seconds() < 0:
                raise ValueError("utc_stop must be greater than utc_start")

            if verbose:
                log(f"Looking for ZTF exposures between {start} and {stop}")

            kowalski_query = {
                "query_type": "find",
                "query": {
                    "catalog": "ZTF_ops",
                    "filter": {
                        "jd_start": {
                            "$gt": Time(start.datetime).jd,
                            "$lt": Time(stop.datetime).jd,
                        }
                    },
                    "projection": {"_id": 0, "fileroot": 1},
                },
            }

            if twilight:
                kowalski_query["query"]["filter"]["qcomment"] = {"$regex": "Twilight"}

            response = kowalski.query(query=kowalski_query).get("data", dict())
            file_roots = sorted([entry["fileroot"] for entry in response])

            frame_names = [
                f"{file_root}_c{ccd:02d}_o_q{quad:1d}"
                for file_root in file_roots
                for ccd in range(1, 17)
                for quad in range(1, 5)
            ]

            if verbose:
                log(f"Found {len(frame_names)} ccd-quad frames")
                log(frame_names)

            processed_frames = [
                frame["_id"]
                for frame in mongo.db[collection].find(
                    {
                        "_id": {"$in": frame_names},
                        "status": {"$in": ["processing", "success"]},
                    },
                    {"_id": 1},
                )
            ]
            if verbose:
                log(processed_frames)

            unprocessed_frames = set(frame_names) - set(processed_frames)

            for frame in unprocessed_frames:
                with timer(f"Submitting frame {frame} for processing", verbose):
                    mongo.db[collection].update_one(
                        {"_id": frame}, {"$set": {"status": "processing"}}, upsert=True
                    )
                    future = dask_client.submit(process_frame, frame, pure=True)
                    dask.distributed.fire_and_forget(future)
                    future.release()
                    del future

        except Exception as e:
            log(e)

        # run once if utc_stop is set
        if utc_stop is not None:
            break
        else:
            log("Heartbeat")
            time.sleep(60)


if __name__ == "__main__":
    fire.Fire(sentinel)
