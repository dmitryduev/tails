__all__ = [
    "IMCCE",
    "leaky_relu",
    "load_config",
    "log",
    "make_triplet",
    "MPC",
    "reticle",
    "plot_stack",
    "preprocess_stack",
    "time_stamp",
    "query_horizons",
    "radec_str2deg",
    "relu",
]

from astropy.coordinates import SkyCoord
from astropy.io import fits
import astropy.units as u
from astropy.visualization import (
    # AsymmetricPercentileInterval,
    ZScaleInterval,
    LinearStretch,
    LogStretch,
    ImageNormalize,
)
from astroquery.imcce import Skybot
from bs4 import BeautifulSoup
from bson.json_util import dumps, loads
from copy import deepcopy
from cycler import cycler
import datetime
import gzip
import io

# from matplotlib.colors import LogNorm
from matplotlib.path import Path
import matplotlib.pyplot as plt
from multiprocessing.pool import ThreadPool
import numpy as np
import re
import requests
from requests_html import HTMLSession
import traceback
from tqdm.auto import tqdm
import yaml


s = requests.Session()


def load_config(config_file="./config.yaml"):
    """
    Load config and secrets
    """
    with open(config_file) as cyaml:
        config = yaml.load(cyaml, Loader=yaml.FullLoader)

    return config


def time_stamp():
    """

    :return: UTC time -> string
    """
    return datetime.datetime.utcnow().strftime("%Y%m%d_%H:%M:%S")


def log(message):
    print(f"{time_stamp()}: {message}")


def relu(a):
    a[a < 0] = 0
    return a


def leaky_relu(a, const=1e-7):
    a[a <= 0] = const
    return a


def radec_str2deg(_ra_str, _dec_str):
    """
    :param _ra_str: 'H:M:S'
    :param _dec_str: 'D:M:S'
    :return: ra, dec in deg
    """
    # convert to deg:
    _ra = list(map(float, _ra_str.split(":")))
    _ra = (_ra[0] + _ra[1] / 60.0 + _ra[2] / 3600.0) * np.pi / 12.0
    _dec = list(map(float, _dec_str.split(":")))
    _sign = -1 if _dec_str.strip()[0] == "-" else 1
    _dec = (
        _sign
        * (abs(_dec[0]) + abs(_dec[1]) / 60.0 + abs(_dec[2]) / 3600.0)
        * np.pi
        / 180.0
    )

    return _ra * 180.0 / np.pi, _dec * 180.0 / np.pi


def query_horizons(record_id: int, position: str = "I41@399", jd=None):
    try:
        url = (
            "http://ssd.jpl.nasa.gov/horizons_batch.cgi?batch=l"
            f"&COMMAND='{record_id}'"
            f"&CENTER='{position}'"
            "&MAKE_EPHEM='YES'"
            "&TABLE_TYPE='OBSERVER'"
            f"&START_TIME='JD {jd}'"
            f"&STOP_TIME='JD {jd + 1e-6}'"
            "&STEP_SIZE='1 m'"
            "&CAL_FORMAT='CAL'"
            "&TIME_DIGITS='MINUTES'"
            "&ANG_FORMAT='HMS'"
            "&OUT_UNITS='KM-S'"
            "&RANGE_UNITS='AU'"
            "&APPARENT='AIRLESS'"
            "&SUPPRESS_RANGE_RATE='NO'"
            "&SKIP_DAYLT='NO'"
            "&EXTRA_PREC='NO'"
            "&R_T_S_ONLY='NO'"
            "&REF_SYSTEM='J2000'"
            "&CSV_FORMAT='NO'"
            "&OBJ_DATA='NO'"
            "&QUANTITIES='1,2,3,4,9'"
        )

        r = s.get(url)

        if r.status_code == requests.codes.ok:

            resp = r.text.split("\n")
            i_start = [ir for ir, rr in enumerate(resp) if rr == "$$SOE"][0]
            i_stop = [ir for ir, rr in enumerate(resp) if rr == "$$EOE"][0]
            record = resp[i_start + 1 : i_stop]

            # print(record)

            tmp = record[0].split()

            raw_date = "_".join(tmp[:2]).strip()
            if "." in raw_date:
                dt = datetime.datetime.strptime(raw_date, "%Y-%b-%d_%H:%M:%S.%f")
            elif raw_date.count(":") == 2:
                dt = datetime.datetime.strptime(raw_date, "%Y-%b-%d_%H:%M:%S")
            elif raw_date.count(":") == 1:
                dt = datetime.datetime.strptime(raw_date, "%Y-%b-%d_%H:%M")
            else:
                dt = None

            if len(tmp) == 21:
                data = {
                    "t_utc": dt,
                    "ra_str": ":".join(tmp[3:6]),
                    "dec_str": ":".join(tmp[6:9]),
                    "ra_apparent_str": ":".join(tmp[9:12]),
                    "dec_apparent_str": ":".join(tmp[12:15]),
                    "dRA*cosD": float(tmp[15]),
                    "d(DEC)/dt": float(tmp[16]),
                    "az": float(tmp[17]),
                    "el": float(tmp[18]),
                    "T_mag": float(tmp[19]) if tmp[19] != "n.a." else None,
                    "N_mag": float(tmp[20]) if tmp[20] != "n.a." else None,
                }
            elif len(tmp) == 20:
                data = {
                    "t_utc": dt,
                    "ra_str": ":".join(tmp[2:5]),
                    "dec_str": ":".join(tmp[5:8]),
                    "ra_apparent_str": ":".join(tmp[8:11]),
                    "dec_apparent_str": ":".join(tmp[11:14]),
                    "dRA*cosD": float(tmp[14]),
                    "d(DEC)/dt": float(tmp[15]),
                    "az": float(tmp[16]),
                    "el": float(tmp[17]),
                    "T_mag": float(tmp[18]) if tmp[18] != "n.a." else None,
                    "N_mag": float(tmp[19]) if tmp[19] != "n.a." else None,
                }

            ra, dec = radec_str2deg(data["ra_str"], data["dec_str"])
            data["ra_jpl"], data["dec_jpl"] = ra, dec

            return data

        else:
            return None

    except Exception as e:
        print(e)
        traceback.print_exc()
        return None


def make_triplet(alert, normalize: bool = False):
    """
    Feed in alert packet
    """
    cutout_dict = dict()

    for cutout in ("science", "template", "difference"):
        cutout_data = loads(
            dumps([alert[f"cutout{cutout.capitalize()}"]["stampData"]])
        )[0]

        # unzip
        with gzip.open(io.BytesIO(cutout_data), "rb") as f:
            with fits.open(io.BytesIO(f.read())) as hdu:
                data = hdu[0].data
                # replace nans with zeros
                cutout_dict[cutout] = np.nan_to_num(data)
                # normalize
                if normalize:
                    cutout_dict[cutout] /= np.linalg.norm(cutout_dict[cutout])

        # pad to 63x63 if smaller
        shape = cutout_dict[cutout].shape
        if shape != (63, 63):
            cutout_dict[cutout] = np.pad(
                cutout_dict[cutout],
                [(0, 63 - shape[0]), (0, 63 - shape[1])],
                mode="constant",
                constant_values=1e-9,
            )

    triplet = np.zeros((63, 63, 3))
    triplet[:, :, 0] = cutout_dict["science"]
    triplet[:, :, 1] = cutout_dict["template"]
    triplet[:, :, 2] = cutout_dict["difference"]

    return triplet


def reticle(inner=0.5, outer=1.0, angle=0.0, which="lrtb"):
    """Create a reticle or crosshairs marker.

    Author: Leo P. Singer

    Parameters
    ----------
    inner : float
        Distance from the origin to the inside of the crosshairs.
    outer : float
        Distance from the origin to the outside of the crosshairs.
    angle : float
        Rotation in degrees; 0 for a '+' orientation and 45 for 'x'.
    Returns
    -------
    path : `matplotlib.path.Path`
        The new marker path, suitable for passing to Matplotlib functions
        (e.g., `plt.plot(..., marker=reticle())`)
    Examples
    --------
    .. plot::
       :context: reset
       :include-source:
       :align: center
        from matplotlib import pyplot as plt
        from ligo.skymap.plot.marker import reticle
        markers = [reticle(inner=0),
                   reticle(which='lt'),
                   reticle(which='lt', angle=45)]
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(-0.5, 0.5)
        for x, marker in enumerate(markers):
            ax.plot(x, 0, markersize=20, markeredgewidth=2, marker=marker)
    """
    angle = np.deg2rad(angle)
    x = np.cos(angle)
    y = np.sin(angle)
    rotation = [[x, y], [-y, x]]
    vertdict = {"l": [-1, 0], "r": [1, 0], "b": [0, -1], "t": [0, 1]}
    verts = [vertdict[direction] for direction in which]
    codes = [Path.MOVETO, Path.LINETO] * len(verts)
    verts = np.dot(verts, rotation)
    verts = np.swapaxes([inner * verts, outer * verts], 0, 1).reshape(-1, 2)
    return Path(verts, codes)


def plot_stack(stack, reticles=None, zscale=True, save=False, **kwargs):
    """

    :param stack: assuming "channels_last" shape
    :param reticles:
    :param zscale:
    :param save:
    :param kwargs:
    :return:
    """
    w = kwargs.get("w", 8)
    h = kwargs.get("h", 2)
    dpi = kwargs.get("dpi", 120)
    cmap = kwargs.get("cmap", plt.cm.cividis)
    origin = kwargs.get("origin", "lower")
    titles = kwargs.get("titles", None)  # should be of shape (n_i, )

    # number of images in the stack to plot
    n_i = stack.shape[-1]
    plt.close("all")
    # cmap: plt.cm.cividis, plt.cm.bone

    fig = plt.figure(figsize=(w, h), dpi=dpi)

    # number of rows and columns
    n_r = kwargs.get("n_r", 1)
    n_c = kwargs.get("n_c", n_i)

    for i in range(n_i):
        ax = fig.add_subplot(n_r, n_c, i + 1)
        ax.axis("off")

        img = deepcopy(stack[..., i])
        # print(img)

        # replace dubiously large values
        xl = np.greater(np.abs(img), 1e20, where=~np.isnan(img))
        if img[xl].any():
            img[xl] = np.nan
        img[np.abs(img) < 0.1] = np.nan
        if np.isnan(img).any():
            median = float(np.nanmean(img.flatten()))
            img = np.nan_to_num(img, nan=median)

        norm = ImageNormalize(
            img, stretch=LinearStretch() if i == n_i - 1 else LogStretch()
        )
        img_norm = norm(img)
        # normalizer = AsymmetricPercentileInterval(
        #     lower_percentile=1, upper_percentile=100
        # )
        normalizer = ZScaleInterval(
            nsamples=stack.shape[0] * stack.shape[1],
            contrast=kwargs.get("contrast", 0.2),
            krej=kwargs.get("krej", 2.5),
        )
        vmin, vmax = normalizer.get_limits(img_norm)
        ax.imshow(img_norm, cmap=cmap, origin=origin, vmin=vmin, vmax=vmax)

        if titles is not None:
            ax.title.set_text(titles[i])
        if reticles is not None:
            ax.set_prop_cycle(cycler("color", ["#fd6a21", "c", "m", "y", "k"]))
            for ri, (x, y) in enumerate(reticles):
                ax.plot(
                    x,
                    y,
                    marker=reticle(which="lt", inner=0.4),
                    alpha=1,
                    markersize=20,
                    markeredgewidth=1.2,
                )

    plt.tight_layout()

    if not save:
        plt.show()
    else:
        fig.savefig(save, dpi=dpi)


def preprocess_stack(
    stack,
    fix_diff_neg: bool = True,
    shape: tuple = (256, 256),
    bad_diff_neg: int = -100,
):
    ns = stack.shape[-1]
    for i in range(ns):
        ti = stack[:, :, i]
        if np.sum(np.isnan(ti)) == shape[0] * shape[1]:
            return None
        ti[np.isnan(ti)] = np.median(ti[~np.isnan(ti)])
        stack[:, :, i] = ti

    if fix_diff_neg and ns == 3:
        # replace large negative values (bad!) with zeros:
        dni = stack[..., 2] < bad_diff_neg
        stack[dni, 2] = 0

    # normalize
    for i in range(ns):
        if np.linalg.norm(stack[..., i]) == 0:
            return None
        stack[..., i] /= np.linalg.norm(stack[..., i])

    return stack


class MPC(object):
    """Utility to query Minor Planet Center's MPCheker service"""

    def __init__(self, checker="neocmt", verbose=False):
        """

        :param checker: enum: ['neocmt', 'cmt']
        :param verbose:
        """
        # initialize an HTTP session
        self.session = HTMLSession()
        self.url = f"https://cgi.minorplanetcenter.net/cgi-bin/{checker}check.cgi"

        self.v = verbose

    def batch_query(self, queries, n_treads: int = 4):
        """Run multiple queries in parallel

        :param queries:
        :param n_treads:
        :return:
        """
        n_treads = min(len(queries), n_treads)

        with ThreadPool(processes=n_treads) as pool:
            if self.v:
                return list(tqdm(pool.imap(self.query, queries), total=len(queries)))
            else:
                return list(pool.imap(self.query, queries))

    def query(self, query: dict, **kwargs):
        """Query MPCheker

        :param query
        :param kwargs:
        :return:
        """

        qid = query.get("id", None)

        # must be a astropy.coordinates.SkyCoord instance:
        position = query.get("position")
        # radius: [arcmin]
        radius = query.get("radius")
        # astropy.time.Time instance:
        epoch = query.get("epoch")
        year = epoch.datetime.year
        month = epoch.datetime.month
        day = (
            epoch.datetime.day
            + epoch.datetime.hour / 24
            + epoch.datetime.minute / 60 / 24
            + epoch.datetime.second / 3600 / 24
        )
        # observatory code, e.g. 'I41' for Palomar:
        observatory_code = query.get("observatory_code", "I41")
        # V magnitude limit:
        limit = query.get("limit", 24)

        post_data = {
            "year": str(year),
            "month": f"{month:02d}",
            "day": f"{day:07.4f}",
            "which": query.get("which", "pos"),
            "ra": f"{position.ra.hms.h:02.0f} {position.ra.hms.m:02.0f} {position.ra.hms.s:06.3f}",
            "decl": f"{'-' if position.dec.deg < 0 else '+'}{abs(position.dec.dms.d):02.0f} "
            f"{abs(position.dec.dms.m):02.0f} {abs(position.dec.dms.s):05.2f}",
            "TextArea": "",
            "radius": str(radius),
            "limit": str(limit),
            "oc": observatory_code,
            "sort": query.get("sort", "d"),
            "mot": query.get("mot", "h"),
            "tmot": query.get("tmot", "t"),
            "pdes": query.get("pdes", "u"),
            "needed": query.get("needed", "f"),
            "ps": query.get("ps", "n"),
            "type": query.get("type", "p"),
        }
        # print(post_data)

        try:
            resp = self.session.post(
                self.url,
                data=post_data,
                timeout=int(kwargs.get("timeout", 60)),
            )
            if resp.status_code != requests.codes.ok:
                response = {"status": "error"}
                if qid:
                    response["id"] = qid
                return response
        except Exception as e:
            response = {"status": "error", "message": str(e)}
            if qid:
                response["id"] = qid
            return response

        soup = BeautifulSoup(resp.text, "html.parser")
        # print(soup)

        if soup.pre is None:
            response = {"status": "success", "data": []}
            if qid:
                response["id"] = qid
            return response

        table_raw = soup.pre.get_text()

        data = []
        lines = table_raw.split("\n")[4:-1]

        for line in lines:
            p = re.compile("(?<!\d)\d{2}(?!\d)")
            i_0 = 25
            m = p.findall(line[i_0:])
            o = line[i_0:].find(m[0]) + i_0

            designation = line[:o].strip()
            c = SkyCoord(line[o : o + 20], unit=(u.hourangle, u.deg))
            v = line[o + 20 : o + 29]
            v = float(v) if len(v.strip()) > 0 else None
            rest = line[o + 29 :].split()
            offset = position.separation(c).arcsec
            offset_ra__min = rest[0]
            offset_dec__min = rest[1]
            motion_ra__min_per_h = (
                float(rest[2])
                if not rest[2].endswith("d")
                else float(rest[2][:-1]) * 60
            )
            motion_dec__min_per_h = (
                float(rest[3])
                if not rest[3].endswith("d")
                else float(rest[3][:-1]) * 60
            )
            orbit = rest[4]
            misc = " ".join(rest[5:])

            data.append(
                {
                    "designation": designation,
                    "ra": c.ra.deg,
                    "dec": c.dec.deg,
                    "radec_str": c.to_string(style="hmsdms"),
                    "v": v,
                    "offset__arcsec": offset.round(2),
                    "offset_ra__min": offset_ra__min,
                    "offset_dec__min": offset_dec__min,
                    "motion_ra__min_per_h": motion_ra__min_per_h,
                    "motion_dec__min_per_h": motion_dec__min_per_h,
                    "orbit": orbit,
                    "misc": misc,
                }
            )

        response = {"status": "success", "data": data}
        if qid:
            response["id"] = qid
        return response


class IMCCE(object):
    """Utility to query IMCCE's Skybot service"""

    def __init__(self, verbose=False):
        """

        :param verbose:
        """
        # initialize Skybot
        self.skybot = Skybot()

        self.v = verbose

    def batch_query(self, queries, n_treads: int = 4):
        """Run multiple queries in parallel

        :param queries:
        :param n_treads:
        :return:
        """
        n_treads = min(len(queries), n_treads)

        with ThreadPool(processes=n_treads) as pool:
            if self.v:
                return list(tqdm(pool.imap(self.query, queries), total=len(queries)))
            else:
                return list(pool.imap(self.query, queries))

    def query(self, query: dict, **kwargs):
        """Query Skybot

        :param query
        :param kwargs:
        :return:
        """

        qid = query.get("id", None)

        # must be a astropy.coordinates.SkyCoord instance:
        position = query.get("position")
        # radius: [arcmin]
        radius = query.get("radius") * u.arcmin
        # astropy.time.Time instance:
        epoch = query.get("epoch")
        # observatory code, e.g. 'I41' for Palomar:
        location = query.get("observatory_code", "I41")

        try:
            data = self.skybot.cone_search(
                coo=position, rad=radius, epoch=epoch, location=location
            )
        except Exception as e:
            response = {"status": "error", "message": e}
            if qid:
                response["id"] = qid
            return response

        response = {"status": "success", "data": data}
        if qid:
            response["id"] = qid
        return response
