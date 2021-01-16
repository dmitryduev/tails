import math
import pandas as pd
import pathlib

import run_tails
from tails.utils import log


class TestRunTails:
    """
    Test run_tails.py on a ZTF observation of 2I/Borisov from 20191014
    """

    def test_run_tails(self):
        run_tails.run(
            cleanup="all",
            checkpoint="../models/tails-20210107/tails",
            config="../config.defaults.yaml",
            score_threshold=0.7,
            single_image="ztf_20191014495961_000570_zr_c05_o_q3",
        )

        # The image is tessellated into 169 tiles, and on one of the them, the comet should be detected,
        # which produces 3 files -- this is what we are checking here:
        assert (
            len(
                list(
                    pathlib.Path("./runs/20191014").glob(
                        "ztf_20191014495961_000570_zr_c05_o_q3*"
                    )
                )
            )
            == 3
        )

        # check the prediction:
        df = pd.read_csv("./runs/20191014/ztf_20191014495961_000570_zr_c05_o_q3.csv")
        log(df)

        # check the score and the predicted image plane position
        score = df.p.values[0]
        log(f"Predicted score: {score}")

        x_comet, y_comet = 558, 2639  # approximate position [pix]
        x, y = df.x.values[0], df.y.values[0]
        # call a prediction within 5 pix a success
        offset = math.sqrt((x - x_comet) ** 2 + (y - y_comet) ** 2)
        log(f"Predicted offset from the apriori position: {offset:.2f} pix")

        assert score > 0.9
        assert offset < 5
