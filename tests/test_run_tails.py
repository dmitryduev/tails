import pathlib

import run_tails


class TestRunTails:
    """
    Test run_tails.py
    """

    def test_run_tails(self):
        success = run_tails.run(
            cleanup="all",
            checkpoint="../models/tails-20210107/tails",
            config="../config.defaults.yaml",
            score_threshold=0.7,
            single_image="ztf_20191014495961_000570_zr_c05_o_q3",
        )

        assert success

        assert len(list(pathlib.Path("./runs/20191014").glob("*.*"))) == 3
