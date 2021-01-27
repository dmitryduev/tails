from dask.distributed import LocalCluster
import time

from tails.utils import load_config, log


""" load config and secrets """
config = load_config(config_file="./config.yaml")


if __name__ == "__main__":

    cluster = LocalCluster(
        threads_per_worker=config["sentinel"]["dask"]["threads_per_worker"],
        n_workers=config["sentinel"]["dask"]["n_workers"],
        scheduler_port=config["sentinel"]["dask"]["scheduler_port"],
        lifetime=config["sentinel"]["dask"]["lifetime"],
        lifetime_stagger=config["sentinel"]["dask"]["lifetime_stagger"],
        lifetime_restart=config["sentinel"]["dask"]["lifetime_restart"],
    )
    log(cluster)

    while True:
        time.sleep(60)
        log("Heartbeat")
