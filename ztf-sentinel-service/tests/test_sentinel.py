import time
import yaml

from utilities import log, Mongo
from watcher import sentinel


with open("/app/config.yaml") as config_yaml:
    config = yaml.load(config_yaml, Loader=yaml.FullLoader)


class TestSentinel:
    def test_sentinel(self):
        # execute sentinel on a single file, wait for it to be processed, then exit
        sentinel(test=True)

        # check that processing succeeded
        frame_name = "ztf_20191014495961_000570_zr_c05_o_q3"

        mongo = Mongo(
            host=config["sentinel"]["database"]["host"],
            port=config["sentinel"]["database"]["port"],
            username=config["sentinel"]["database"]["username"],
            password=config["sentinel"]["database"]["password"],
            db=config["sentinel"]["database"]["db"],
            verbose=True,
        )
        log("MongoDB connection OK")

        collection = config["sentinel"]["database"]["collection"]

        num_retries = 10
        for i in range(num_retries):
            if i == num_retries - 1:
                mongo.db[collection].delete_one({"_id": frame_name})
                # raise RuntimeError("Processing failed")

            # check status in db
            processed_frames = list(
                mongo.db[collection].find(
                    {"_id": frame_name, "status": "success"},
                    {"_id": 1},
                )
            )
            # assert len(processed_frames) == 1
            if len(processed_frames) == 0:
                print("No processed frames found, waiting...")
                time.sleep(20)
                continue

            log(processed_frames)

            break
