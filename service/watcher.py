import arrow
from astropy.time import Time
import fire
from penquins import Kowalski
import time

from tails.utils import log, load_config

from utilities import init_db, Mongo


config = load_config(config_file="/app/config.yaml")


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

            query = {
                "query_type": "find",
                "query": {
                    "catalog": "ZTF_ops",
                    "filter": {
                        "jd_start": {
                            "$gt": Time(start.datetime).jd,
                            "$lt": Time(stop.datetime).jd + 1,
                        }
                    },
                    "projection": {"_id": 0, "fileroot": 1},
                },
            }

            if twilight:
                query["query"]["filter"]["qcomment"] = {"$regex": "Twilight"}

            response = kowalski.query(query=query).get("data", dict())
            file_roots = sorted([entry["fileroot"] for entry in response])

            frame_names = [
                f"{file_root}_c{ccd:02d}_o_q{quad:1d}"
                for file_root in file_roots
                for ccd in range(1, 17)
                for quad in range(1, 5)
            ]

            log(frame_names)

            mongo.db[collection].find(
                {
                    "_id": {"$in": frame_names},
                    "processed": True,
                }
            )
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
