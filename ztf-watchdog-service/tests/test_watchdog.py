import yaml

from utilities import log, Mongo
from watcher import watchdog


with open("/app/config.yaml") as config_yaml:
    config = yaml.load(config_yaml, Loader=yaml.FullLoader)


class TestWatchdog:
    def test_watchdog(self):
        mongo = Mongo(
            host=config["watchdog"]["database"]["host"],
            port=config["watchdog"]["database"]["port"],
            username=config["watchdog"]["database"]["username"],
            password=config["watchdog"]["database"]["password"],
            db=config["watchdog"]["database"]["db"],
            verbose=True,
        )
        log("MongoDB connection OK")

        collection = config["watchdog"]["database"]["collection"]

        # execute watchdog on a single file, wait for it to be processed, then exit
        watchdog(test=True)

        # check status in db
        processed_frames = list(
            mongo.db[collection].find(
                {"status": "success"},
                {"_id": 1},
            )
        )
        log(processed_frames)
        # assert len(processed_frames) == 1

        mongo.db[collection].delete_one(
            {"_id": "ztf_20191014495961_000570_zr_c05_o_q3"}
        )
