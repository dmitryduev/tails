import arrow
from astropy.time import Time
import dask.distributed
import fire
from penquins import Kowalski
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import time

from tails.utils import log, load_config

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
            host=config["database"]["host"],
            port=config["database"]["port"],
            username=config["database"]["username"],
            password=config["database"]["password"],
            db=config["database"]["db"],
            verbose=self.verbose,
        )

        # session to talk to Fritz
        self.session = requests.Session()
        self.session_headers = {"Authorization": f"token {config['fritz']['token']}"}

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
        self.model = None

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
                f"{config['fritz']['protocol']}://"
                f"{config['fritz']['host']}:{config['fritz']['port']}"
                f"{endpoint}",
                params=data,
                headers=self.session_headers,
            )
        else:
            response = methods[method](
                f"{config['fritz']['protocol']}://"
                f"{config['fritz']['host']}:{config['fritz']['port']}"
                f"{endpoint}",
                json=data,
                headers=self.session_headers,
            )

        return response

    def process_frame(self, frame):
        pass


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

    collection = config["database"]["collection"]

    try:
        tails_worker.mongo.db[collection].update_one(
            {"_id": frame}, {"$set": {"status": "processing"}}
        )
        # todo: process frame, tessellate, run Tails on individual tiles
        tails_worker.process_frame(frame)
        # todo: prepare and post results to Fritz, if any
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
        host=config["database"]["host"],
        port=config["database"]["port"],
        username=config["database"]["username"],
        password=config["database"]["password"],
        db=config["database"]["db"],
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

    collection = config["database"]["collection"]

    # remove dangling entries in the db at startup
    mongo.db[collection].delete_many({"status": "processing"})

    # Configure dask client
    if verbose:
        log("Initializing dask.distributed client")
    dask_client = dask.distributed.Client(
        address=f"{config['dask']['host']}:{config['dask']['scheduler_port']}"
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
