from dask.distributed import LocalCluster
import time

from tails.utils import load_config, log


""" load config and secrets """
config = load_config(config_file="./config.yaml")


if __name__ == "__main__":

    cluster = LocalCluster(
        threads_per_worker=config["watchdog"]["dask"]["threads_per_worker"],
        n_workers=config["watchdog"]["dask"]["n_workers"],
        scheduler_port=config["watchdog"]["dask"]["scheduler_port"],
        lifetime=config["watchdog"]["dask"]["lifetime"],
        lifetime_stagger=config["watchdog"]["dask"]["lifetime_stagger"],
        lifetime_restart=config["watchdog"]["dask"]["lifetime_restart"],
    )
    log(cluster)

    while True:
        time.sleep(60)
        log("Heartbeat")