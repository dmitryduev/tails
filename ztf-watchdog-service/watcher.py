import arrow
from astropy.coordinates import SkyCoord
from astropy.time import Time
from copy import deepcopy
import dask.distributed
import fire
import multiprocessing as mp
import numpy as np
import pandas as pd
import pathlib
from penquins import Kowalski
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import time

from tails.efficientdet import Tails
from tails.image import Dvoika, Troika
from tails.utils import load_config, log, plot_stack, preprocess_stack

from utilities import init_db, Mongo, timer


config = load_config(config_file="/app/config.yaml")


DEFAULT_TIMEOUT = 5  # seconds


class TimeoutHTTPAdapter(HTTPAdapter):
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
    def __init__(self, **kwargs):
        self.verbose = kwargs.get("verbose", 2)
        self.config = config

        # mongo connection
        self.mongo = Mongo(
            host=config["watchdog"]["database"]["host"],
            port=config["watchdog"]["database"]["port"],
            username=config["watchdog"]["database"]["username"],
            password=config["watchdog"]["database"]["password"],
            db=config["watchdog"]["database"]["db"],
            verbose=self.verbose,
        )

        # session to talk to Fritz
        self.session = requests.Session()
        self.session_headers = {
            "Authorization": f"token {config['watchdog']['fritz']['token']}"
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

        # todo: preload Tails
        self.path = pathlib.Path(config["watchdog"]["app"]["path"])

        self.checkpoint = pathlib.Path(
            f"/app/models/{config['watchdog']['app']['checkpoint']}/tails"
        )

        self.model = Tails()
        self.model.load_weights(self.checkpoint).expect_partial()

        self.score_threshold = float(
            config["watchdog"]["app"].get("score_threshold", 0.5)
        )
        if not (0 <= self.score_threshold <= 1):
            raise ValueError(
                "watchdog.app.score_threshold must be (0 <= score_threshold <=1), check config"
            )

        self.cleanup = config["watchdog"]["app"]["cleanup"]
        if self.cleanup not in ("all", "none", "ref", "sci"):
            raise ValueError(
                "watchdog.app.cleanup value not in ('all', 'none', 'ref', 'sci'), check config"
            )

        self.num_threads = mp.cpu_count()

    def api_fritz(self, method: str, endpoint: str, data=None):
        """Make an API call to a SkyPortal instance

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
                f"{config['watchdog']['fritz']['protocol']}://"
                f"{config['watchdog']['fritz']['host']}:{config['watchdog']['fritz']['port']}"
                f"{endpoint}",
                params=data,
                headers=self.session_headers,
            )
        else:
            response = methods[method](
                f"{config['watchdog']['fritz']['protocol']}://"
                f"{config['watchdog']['fritz']['host']}:{config['watchdog']['fritz']['port']}"
                f"{endpoint}",
                json=data,
                headers=self.session_headers,
            )

        return response

    def process_frame(self, frame):
        path_base = self.path.resolve()

        date_string = frame.split("_")[1][:8]
        path_date = self.path / "runs" / date_string

        if not path_date.exists():
            path_date.mkdir(parents=True, exist_ok=True)

        try:
            box_size_pix = config["watchdog"]["app"]["box_size_pix"]

            dim_last = self.model.inputs[0].shape[-1]

            if dim_last == 2:
                stack_class = Dvoika
            elif dim_last == 3:
                stack_class = Troika
            else:
                raise ValueError(
                    "bad dim_last: only know how to operate on duplets and triplets"
                )

            stack = stack_class(
                path_base=str(path_base),
                name=frame,
                secrets=self.config,
                verbose=self.verbose,
            )

            # reproject ref
            ref_projected = stack.reproject_ref2sci(
                how="swarp", nthreads=self.num_threads
            )
            # tessellate
            xboxes, yboxes = stack.tessellate_boxes(
                box_size_pix=box_size_pix, offset=20
            )

            tessellation = []

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

            predictions = self.model.predict(stacks)
            # log(predictions[0], predictions.shape)

            # detections
            dets = []
            ind = np.where(predictions[:, 0].flatten() > self.score_threshold)[0]
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
                    save=str(path_date / f"{frame}_{ni}.png"),
                    # cmap=cmr.arctic,
                    # cmap=plt.cm.viridis,
                    # cmap=plt.cm.cividis,
                    cmap="bone",
                )
                # save npy:
                np.save(str(path_date / f"{frame}_{ni}.npy"), tessellation[ii]["stack"])

                x, y = (
                    tessellation[ii]["ybox"][0] + x_o,
                    tessellation[ii]["xbox"][0] + y_o,
                )
                ra, dec = stack.pix2world_sci(x, y)
                c = SkyCoord(ra=ra, dec=dec, unit="deg")
                radecstr = c.to_string(style="hmsdms")
                t = Time(stack.header_sci["OBSJD"], format="jd")
                jd = float(t.jd)
                dt = t.datetime
                det = {
                    "id": frame,
                    "ni": ni,
                    "jd": jd,
                    "datestr": f"{dt.year} {dt.month} {dt.day + (dt.hour + dt.minute / 60.0 + dt.second / 3600.0) / 24}",
                    "p": float(predictions[ii, 0]),
                    "x": x,
                    "y": y,
                    "ra": ra,
                    "dec": dec,
                    "radecstr": radecstr,
                    "tails_v": self.config["watchdog"]["app"]["checkpoint"],
                }
                dets.append(det)

            if len(dets) > 0:
                # print(dets)
                df_dets = pd.DataFrame.from_records(dets)
                df_dets.to_csv(str(path_date / f"{frame}.csv"), index=False)

            if self.cleanup.lower() != "none":
                cleanup_ref = True if self.cleanup.lower() in ("all", "ref") else False
                cleanup_sci = True if self.cleanup.lower() in ("all", "sci") else False
                stack.cleanup(ref=cleanup_ref, sci=cleanup_sci)

            return True

        except Exception as e:
            print(str(e))
            return False


class WorkerInitializer(dask.distributed.WorkerPlugin):
    def __init__(self, *args, **kwargs):
        self.tails_worker = None

    def setup(self, worker: dask.distributed.Worker):
        self.tails_worker = TailsWorker()


def process_frame(frame):
    # get worker running current task
    worker = dask.distributed.get_worker()
    tails_worker = worker.plugins["worker-init"].tails_worker

    log(f"Processing {frame} on {worker.address}")

    collection = config["watchdog"]["database"]["collection"]

    try:
        tails_worker.mongo.db[collection].update_one(
            {"_id": frame}, {"$set": {"status": "processing"}}
        )
        # todo: process frame, tessellate, run Tails on individual tiles
        tails_worker.process_frame(frame)

        # todo: prepare and post results to Fritz, if any
        if config["watchdog"]["app"]["post_to_fritz"]:
            pass
        tails_worker.mongo.db[collection].update_one(
            {"_id": frame}, {"$set": {"status": "success"}}
        )
    except Exception as e:
        log(e)
        tails_worker.mongo.db[collection].update_one(
            {"_id": frame}, {"$set": {"status": "error"}}
        )


def watchdog(
    utc_start: str = None,
    utc_stop: str = None,
    twilight: bool = False,
    test: bool = False,
    verbose: bool = False,
):
    """

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
        host=config["watchdog"]["database"]["host"],
        port=config["watchdog"]["database"]["port"],
        username=config["watchdog"]["database"]["username"],
        password=config["watchdog"]["database"]["password"],
        db=config["watchdog"]["database"]["db"],
        verbose=verbose,
    )
    if verbose:
        log("Set up MongoDB connection")

    if verbose:
        log("Setting up Kowalski connection")
    kowalski = Kowalski(
        token=config["kowalski"]["token"],
        protocol=config["kowalski"]["protocol"],
        host=config["kowalski"]["host"],
        port=config["kowalski"]["port"],
        verbose=True,
    )
    if verbose:
        log(f"Kowalski connection OK: {kowalski.ping()}")

    collection = config["watchdog"]["database"]["collection"]

    # remove dangling entries in the db at startup
    mongo.db[collection].delete_many({"status": "processing"})

    # Configure dask client
    if verbose:
        log("Initializing dask.distributed client")
    dask_client = dask.distributed.Client(
        address=f"{config['watchdog']['dask']['host']}:{config['watchdog']['dask']['scheduler_port']}"
    )

    # init each worker with Worker instance
    if verbose:
        log("Initializing dask.distributed workers")
    worker_initializer = WorkerInitializer()
    dask_client.register_worker_plugin(worker_initializer, name="worker-init")

    while True:
        try:
            start = (
                arrow.get(utc_start)
                if utc_start is not None
                else arrow.utcnow().shift(hours=-1)
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

            # fixme:
            frame_names = ["ztf_20191014495961_000570_zr_c05"]

            if verbose:
                log(frame_names)

            processed_frames = list(
                mongo.db[collection].find(
                    {
                        "_id": {"$in": frame_names},
                        "status": {"$in": ["processing", "success"]},
                    },
                    {"_id": 1},
                )
            )
            if verbose:
                log(processed_frames)

            unprocessed_frames = set(frame_names) - set(processed_frames)

            for frame in unprocessed_frames:
                with timer(f"Submitting frame {frame} for processing", verbose):
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
            time.sleep(30)


if __name__ == "__main__":
    fire.Fire(watchdog)
