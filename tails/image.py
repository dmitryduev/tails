__all__ = ["Dvoika", "Troika"]

from astropy.io import fits
from astropy.modeling import models, fitting
from astropy.stats import sigma_clip
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import requests
from reproject import reproject_interp
import subprocess

from .swarp import prepare_swarp_align

import matplotlib

matplotlib.use("Agg")


class Star(object):
    """Define a star by its coordinates and modelled FWHM
    Given the coordinates of a star within a 2D array, fit a model to the star and determine its
    Full Width at Half Maximum (FWHM).The star will be modelled using astropy.modelling. Currently
    accepted models are: 'Gaussian2D', 'Moffat2D'
    """

    _GAUSSIAN2D = "Gaussian2D"
    _MOFFAT2D = "Moffat2D"
    # _MODELS = set([_GAUSSIAN2D, _MOFFAT2D])

    def __init__(
        self, x0, y0, data, model_type=_GAUSSIAN2D, box=64, plate_scale=1.012, exp=0.0
    ):
        """Instantiation method for the class Star.
        The 2D array in which the star is located (data), together with the pixel coordinates (x0,y0) must be
        passed to the instantiation method. .
        """
        self.x = x0
        self.y = y0
        self._box = box
        # field of view in x in arcsec:
        self._plate_scale = plate_scale
        self._exp = exp
        self._XGrid, self._YGrid = self._grid_around_star(x0, y0, data)
        self.data = data[self._XGrid, self._YGrid]
        self.model_type = model_type

    def model(self):
        """ Fit a model to the star. """
        return self._fit_model()

    @property
    def model_psf(self):
        """ Return a modelled PSF for the given model  """
        return self.model()(self._XGrid, self._YGrid)

    @property
    def fwhm(self):
        """Extract the FWHM from the model of the star.
        The FWHM needs to be calculated for each model. For the Moffat, the FWHM is a function of the gamma and
        alpha parameters (in other words, the scaling factor and the exponent of the expression), while for a
        Gaussian FWHM = 2.3548 * sigma. Unfortunately, our case is a 2D Gaussian, so a compromise between the
        two sigmas (sigma_x, sigma_y) must be reached. We will use the average of the two.
        """
        model_dict = dict(zip(self.model().param_names, self.model().parameters))
        if self.model_type == self._MOFFAT2D:
            gamma, alpha = [model_dict[ii] for ii in ("gamma_0", "alpha_0")]
            FWHM = 2.0 * gamma * np.sqrt(2 ** (1 / alpha) - 1)
            FWHM_x, FWHM_y = None, None
        elif self.model_type == self._GAUSSIAN2D:
            sigma_x, sigma_y = [model_dict[ii] for ii in ("x_stddev_0", "y_stddev_0")]
            FWHM = 2.3548 * np.mean([sigma_x, sigma_y])
            FWHM_x, FWHM_y = 2.3548 * sigma_x, 2.3548 * sigma_y
        return FWHM, FWHM_x, FWHM_y

    # @memoize
    def _fit_model(self):
        fit_p = fitting.LevMarLSQFitter()
        model = self._initialize_model()
        _p = fit_p(model, self._XGrid, self._YGrid, self.data)
        return _p

    def _initialize_model(self):
        """Initialize a model with first guesses for the parameters.
        The user can select between several astropy models, e.g., 'Gaussian2D', 'Moffat2D'. We will use the data to get
        the first estimates of the parameters of each model. Finally, a Constant2D model is added to account for the
        background or sky level around the star.
        """
        max_value = self.data.max()

        if self.model_type == self._GAUSSIAN2D:
            model = models.Gaussian2D(
                x_mean=self.x, y_mean=self.y, x_stddev=1, y_stddev=1
            )
            model.amplitude = max_value

            # Establish reasonable bounds for the fitted parameters
            model.x_stddev.bounds = (0, self._box / 4)
            model.y_stddev.bounds = (0, self._box / 4)
            model.x_mean.bounds = (self.x - 5, self.x + 5)
            model.y_mean.bounds = (self.y - 5, self.y + 5)

        elif self.model_type == self._MOFFAT2D:
            model = models.Moffat2D()
            model.x_0 = self.x
            model.y_0 = self.y
            model.gamma = 2
            model.alpha = 2
            model.amplitude = max_value

            #  Establish reasonable bounds for the fitted parameters
            model.alpha.bounds = (1, 6)
            model.gamma.bounds = (0, self._box / 4)
            model.x_0.bounds = (self.x - 5, self.x + 5)
            model.y_0.bounds = (self.y - 5, self.y + 5)

        model += models.Const2D(self.fit_sky())
        model.amplitude_1.fixed = True
        return model

    def fit_sky(self):
        """Fit the sky using a Ring2D model in which all parameters but the amplitude are fixed."""
        min_value = self.data.min()
        ring_model = models.Ring2D(
            min_value, self.x, self.y, self._box * 0.4, width=self._box * 0.4
        )
        ring_model.r_in.fixed = True
        ring_model.width.fixed = True
        ring_model.x_0.fixed = True
        ring_model.y_0.fixed = True
        fit_p = fitting.LevMarLSQFitter()
        return fit_p(ring_model, self._XGrid, self._YGrid, self.data).amplitude

    def _grid_around_star(self, x0, y0, data):
        """ Build a grid of side 'box' centered in coordinates (x0,y0). """
        lenx, leny = data.shape
        xmin, xmax = max(x0 - self._box / 2, 0), min(x0 + self._box / 2 + 1, lenx - 1)
        ymin, ymax = max(y0 - self._box / 2, 0), min(y0 + self._box / 2 + 1, leny - 1)
        return np.mgrid[int(xmin) : int(xmax), int(ymin) : int(ymax)]

    def get_model(self, sigma=2.0):
        """ Make model and residuals. """
        data = self.data
        model = self.model()(self._XGrid, self._YGrid)
        _residuals = data - model

        # model = relu(relu(model) - np.min(relu(model)))

        # model = leaky_relu(model - np.min(model))
        # model[model < 1e-3] = 0

        # model_abs = np.abs(model)
        # model[model_abs < 1e-3] = 0

        clp = ~sigma_clip(model, sigma=sigma).mask
        model[clp] = 0

        return model, _residuals


class Dvoika(object):
    def __init__(
        self,
        path_base: str,
        name: str,
        download: bool = True,
        secrets=None,
        verbose: bool = False,
        **kwargs,
    ):
        self.path_base = path_base
        self.name = name
        self.verbose = verbose

        self.fetch_data_only = kwargs.get("fetch_only", False)

        self.path_tmp = pathlib.Path(self.path_base) / "tmp"
        self.path_tmp.mkdir(parents=True, exist_ok=True)

        path_sci_base = pathlib.Path(self.path_base) / "sci"
        sci_name = self.name + "_sciimg.fits"

        tmp = sci_name.split("_")[1]
        y, p1, p2 = tmp[:4], tmp[4:8], tmp[8:]

        path_sci = path_sci_base / y / p1 / p2 / sci_name
        path_sci.parents[0].mkdir(parents=True, exist_ok=True)
        self.path_sci = path_sci

        if download:
            if secrets["irsa"]["username"] not in (None, "anonymous"):
                ursa_login = (
                    f"{secrets['irsa']['url_login']}?josso_cmd=login"
                    f"&josso_username={secrets['irsa']['username']}&josso_password={secrets['irsa']['password']}"
                )
                cookies = requests.get(ursa_login).cookies
            else:
                cookies = None

            path_ursa_sci = os.path.join(
                secrets["irsa"]["url"], "sci", y, p1, p2, sci_name
            )

            if verbose:
                print(path_ursa_sci)

            for postfix in (
                "sciimg.fits",
                "scimrefdiffimg.fits.fz",
            ):
                if verbose:
                    print(f"fetching {name}_{postfix}")

                path_aux = str(path_sci).replace("sciimg.fits", postfix)
                if not os.path.exists(path_aux):
                    path_ursa_aux = path_ursa_sci.replace("sciimg.fits", postfix)
                    r = requests.get(
                        path_ursa_aux, allow_redirects=True, cookies=cookies
                    )
                    # print(r.text)
                    if r.status_code != 200:
                        raise Exception(f"download failed: {path_ursa_aux}")
                    # print(path_ursa_aux)
                    # print(r.text)
                    with open(path_aux, "wb") as f:
                        f.write(r.content)

        if not self.fetch_data_only:
            with fits.open(str(path_sci)) as hdulist:
                self.sci = hdulist[0].data
                # self.sci = np.nan_to_num(hdulist[0].data)
                # self.sci[np.isnan(self.sci)] = np.median(self.sci)
                self.sci[np.isnan(self.sci)] = np.median(self.sci[~np.isnan(self.sci)])
                self.header_sci = hdulist[0].header

            self.plate_scale = self.header_sci["PIXSCALE"]

            self.w_sci = WCS(str(path_sci))

        path_ref_base = pathlib.Path(os.path.join(self.path_base, "ref"))
        _, _, field, filt, ccd, _, quad = self.name.split("_")
        if self.verbose:
            print(field, filt, ccd, quad)

        path_rel = pathlib.Path(f"{field}/{ccd}/{quad}/{filt}/")
        path_ref = path_ref_base / path_rel
        path_ref_file = path_ref / f"ref.{field}_{ccd}_{quad}_{filt}.fits"
        # path_mask_file = path_ref / f"ref.{field}_{ccd}_{quad}_{filt}.mask.fits"
        self.path_ref = path_ref_file

        if verbose:
            print(path_ref)
        path_ref.mkdir(parents=True, exist_ok=True)

        if download:
            if verbose:
                print(f"fetching {pathlib.Path(path_ref_file).stem}")
            if not os.path.exists(path_ref_file):
                # get IPAC's/Frank's regular reference instead:
                if secrets["irsa"]["username"] not in (None, "anonymous"):
                    ursa_login = (
                        f"{secrets['irsa']['url_login']}?josso_cmd=login"
                        f"&josso_username={secrets['irsa']['username']}&josso_password={secrets['irsa']['password']}"
                    )
                    cookies = requests.get(ursa_login).cookies
                else:
                    cookies = None

                path_ursa_ref = os.path.join(
                    secrets["irsa"]["url"],
                    "ref",
                    field[:3],
                    f"field{field}",
                    filt,
                    f"ccd{ccd[1:]}",
                    quad,
                    f"ztf_{field}_{filt}_{ccd}_{quad}_refimg.fits",
                )
                if verbose:
                    print(path_ursa_ref)
                r = requests.get(path_ursa_ref, allow_redirects=True, cookies=cookies)
                if r.status_code != 200:
                    raise Exception(f"download failed: {path_ursa_ref}")

                with open(path_ref_file, "wb") as f:
                    f.write(r.content)

        if not self.fetch_data_only:
            with fits.open(str(path_ref_file)) as hdulist:
                self.ref = hdulist[0].data
                # self.ref = np.nan_to_num(hdulist[0].data)
                self.ref[np.isnan(self.ref)] = np.median(self.ref[~np.isnan(self.ref)])
                self.header_ref = hdulist[0].header

            self.w_ref = WCS(str(path_ref_file))

    def cleanup(self, sci: bool = True, ref: bool = True):
        """
        Delete large files to save space
        """
        if sci:
            tmp = self.name.split("_")[1]
            y, p1, p2 = tmp[:4], tmp[4:8], tmp[8:]

            path_sci_base = pathlib.Path(os.path.join(self.path_base, "sci"))
            path_sci = path_sci_base / os.path.join(y, p1, p2)
            fs = list(path_sci.glob(f"{self.name}*"))
            if self.verbose:
                print(fs)

            for ff in fs:
                try:
                    if self.verbose:
                        print(f"removing {str(ff)}")
                    os.remove(str(ff))
                except OSError:
                    if self.verbose:
                        print(f"failed to remove {str(ff)}")

        if ref:
            _, _, field, filt, ccd, _, quad = self.name.split("_")

            path_ref_base = pathlib.Path(os.path.join(self.path_base, "ref"))
            path_ref = path_ref_base / pathlib.Path(f"{field}/{ccd}/{quad}/{filt}/")
            fs = list(path_ref.glob(f"ref.{field}_{ccd}_{quad}_{filt}*"))
            if self.verbose:
                print(fs)

            for ff in fs:
                try:
                    if self.verbose:
                        print(f"removing {str(ff)}")
                    os.remove(str(ff))
                except OSError:
                    if self.verbose:
                        print(f"failed to remove {str(ff)}")

    def world2pix_sci(self, ra: float = None, dec: float = None, to_int: bool = False):

        # print(np.array([ra, dec]))
        pix_sci = self.w_sci.wcs_world2pix(np.array([[ra, dec]]), 0)
        xpos_sci, ypos_sci = pix_sci[0]
        if self.verbose:
            print("xpos_sci, ypos_sci:", xpos_sci, ypos_sci)

        xpos_sci_int = int(np.rint(xpos_sci))
        ypos_sci_int = int(np.rint(ypos_sci))

        return (xpos_sci, ypos_sci) if not to_int else (xpos_sci_int, ypos_sci_int)

    def pix2world_sci(self, x: float = None, y: float = None):

        radec_sci = self.w_sci.wcs_pix2world(np.array([[x, y]]), 0)
        ra_sci, dec_sci = radec_sci[0]
        if self.verbose:
            print("ra_sci, dec_sci:", ra_sci, dec_sci)

        return ra_sci, dec_sci

    def world2pix_ref(self, ra: float = None, dec: float = None, to_int: bool = False):

        pix_ref = self.w_ref.wcs_world2pix(np.array([[ra, dec]]), 0)
        xpos_ref, ypos_ref = pix_ref[0]
        if self.verbose:
            print("xpos_ref, ypos_ref:", xpos_ref, ypos_ref)

        xpos_ref_int = int(np.rint(xpos_ref))
        ypos_ref_int = int(np.rint(ypos_ref))

        return (xpos_ref, ypos_ref) if not to_int else (xpos_ref_int, ypos_ref_int)

    def reproject_ref2sci(self, how: str = "swarp", **kwargs):
        """

        :param how: 'swarp' or 'reproject'
        :return:
        """
        if how == "swarp":
            nthreads = kwargs.get("nthreads", 4)

            command, ref_reprojected, weight_name = prepare_swarp_align(
                self, nthreads=nthreads
            )

            _ = subprocess.run(command.split(), capture_output=True, check=True)

            # load the result
            with fits.open(ref_reprojected) as hdulist:
                ref_projected = hdulist[0].data

            # clean up
            keep_tmp = kwargs.get("keep_tmp", False)

            if not keep_tmp:
                _, _, field, filt, ccd, _, quad = self.name.split("_")

                path_ref_base = pathlib.Path(os.path.join(self.path_base, "ref"))
                path_ref = path_ref_base / pathlib.Path(f"{field}/{ccd}/{quad}/{filt}/")
                fs = list(path_ref.glob(f"ref.{field}_{ccd}_{quad}_{filt}*remap*"))

                if self.verbose:
                    print(fs)

                for ff in fs:
                    try:
                        if self.verbose:
                            print(f"removing {str(ff)}")
                        os.remove(str(ff))
                    except OSError:
                        if self.verbose:
                            print(f"failed to remove {str(ff)}")

        elif how == "reproject":
            order = kwargs.get("order", "bicubic")  # 'bilinear'

            ref_projected = reproject_interp(
                (self.ref, self.header_ref),
                self.header_sci,
                order=order,
                return_footprint=False,
            )
        else:
            raise ValueError("unknown reproject method")

        return ref_projected

    def tessellate_boxes(self, box_size_pix: int = 128, offset: int = 20):
        image_shape = self.sci.shape
        stride = box_size_pix - offset
        assert stride <= box_size_pix, 'there should be no holes in the "tessellation"!'

        xboxes = [
            [stride * i, stride * i + box_size_pix]
            for i in range(image_shape[0] // stride)
        ]
        xboxes[-1] = [image_shape[0] - box_size_pix, image_shape[0]]
        yboxes = [
            [stride * i, stride * i + box_size_pix]
            for i in range(image_shape[1] // stride)
        ]
        yboxes[-1] = [image_shape[1] - box_size_pix, image_shape[1]]

        return xboxes, yboxes

    def make_box(
        self,
        ra: float = None,
        dec: float = None,
        box_size_pix: int = 128,
        min_offset: int = 10,
        random: bool = True,
        seed: int = None,
    ):

        if random:
            if seed is not None:
                np.random.seed(seed)
            else:
                np.random.seed(None)

            xoffset, yoffset = np.random.RandomState().randint(
                0 + min_offset, box_size_pix - min_offset, 2
            )
            if self.verbose:
                print(xoffset, yoffset)
        else:
            xoffset, yoffset = box_size_pix // 2, box_size_pix // 2

        xpos_sci_int, ypos_sci_int = self.world2pix_sci(ra, dec, to_int=True)

        xbox = [xpos_sci_int - xoffset, xpos_sci_int - xoffset + box_size_pix]
        ybox = [ypos_sci_int - yoffset, ypos_sci_int - yoffset + box_size_pix]
        if self.verbose:
            print((xpos_sci_int, ypos_sci_int), xbox, ybox)

        if xbox[0] < 0:
            xbox = [0, box_size_pix]
        if xbox[1] > self.sci.shape[1]:
            # correct for pseudiff img
            xoffset += xbox[1] - self.sci.shape[1]
            # fix box
            xbox = [self.sci.shape[1] - box_size_pix, self.sci.shape[1]]

        if ybox[0] < 0:
            ybox = [0, box_size_pix]
        if ybox[1] > self.sci.shape[0]:
            # correct for pseudiff img
            yoffset += ybox[1] - self.sci.shape[0]
            # fix box
            ybox = [self.sci.shape[0] - box_size_pix, self.sci.shape[0]]

        if self.verbose:
            print(
                self.sci.shape,
                xbox,
                ybox,
                xbox[1] - xbox[0],
                ybox[1] - ybox[0],
                xoffset,
                yoffset,
            )

        # ra, dec of box center
        ra_box_center, dec_box_center = self.pix2world_sci(
            x=xbox[0] + (box_size_pix - 1) // 2, y=ybox[0] + (box_size_pix - 1) // 2
        )

        return xbox, ybox, ra_box_center, dec_box_center  # , xoffset, yoffset

    def alerts_in_box(
        self,
        kowalski,
        ra,
        dec,
        box_size_pix,
        pid,
        exclude_candids=(),
        circle="circumscribed",
        braai_gt=0.3,
    ):
        # alerts within radius arcsec from ((xbox[1]+xbox[0])/2, (ybox[1]+ybox[0])/2)
        if circle == "inscribed":
            # inscribed circle (to ignore corners):
            radius = 1.01 * self.plate_scale * box_size_pix / 2
        elif circle == "circumscribed":
            # circumscribed circle:
            radius = self.plate_scale * np.sqrt(box_size_pix ** 2 / 2)
        else:
            # circumscribed circle by default
            radius = self.plate_scale * np.sqrt(box_size_pix ** 2 / 2)

        q = {
            "query_type": "cone_search",
            "object_coordinates": {
                "radec": f"[({ra}, {dec})]",
                "cone_search_radius": f"{radius}",
                "cone_search_unit": "arcsec",
            },
            "catalogs": {
                "ZTF_alerts": {
                    "filter": {
                        "candidate.pid": pid,
                        "classifications.braai": {"$gt": braai_gt},
                    },
                    "projection": {
                        "_id": 0,
                        "candid": 1,
                        "objectId": 1,
                        "candidate": 1,
                        "classifications": 1,
                    },
                }
            },
        }
        if len(exclude_candids) > 0:
            q["catalogs"]["ZTF_alerts"]["filter"]["candid"] = {"$nin": exclude_candids}
            # pass
        r = kowalski.query(query=q)
        # print(r)

        o = list(r["result_data"]["ZTF_alerts"].keys())[0]
        alerts = r["result_data"]["ZTF_alerts"][o]

        return alerts


class Troika(Dvoika):
    def __init__(
        self,
        path_base: str,
        name: str,
        download: bool = True,
        secrets=None,
        verbose: bool = False,
        **kwargs,
    ):

        super().__init__(path_base, name, download, secrets, verbose, **kwargs)

        path_zogy = str(self.path_sci).replace("sciimg.fits", "scimrefdiffimg.fits.fz")
        self.path_zogy = path_zogy
        if verbose:
            print(path_zogy)

        if not self.fetch_data_only:
            with fits.open(path_zogy) as hdulist:
                self.zogy = hdulist[1].data
                # self.zogy = np.nan_to_num(hdulist[1].data)
                # self.zogy[np.isnan(self.zogy)] = np.median(self.zogy)
                self.zogy[np.isnan(self.zogy)] = np.median(
                    self.zogy[~np.isnan(self.zogy)]
                )
                self.header_zogy = hdulist[1].header

    def model_zogy(
        self,
        ra: float = None,
        dec: float = None,
        model_type: str = "Moffat2D",
        box_size_pix: int = 20,
        sigma: float = 2.0,
        plot: bool = False,
    ):
        """

        :param ra:
        :param dec:
        :param model_type:  enum ['Gaussian2D', 'Moffat2D']
        :param box_size_pix:
        :param sigma:
        :param plot:
        :return:
        """

        xpos, ypos = self.world2pix_sci(ra, dec)

        d = Star(
            ypos,
            xpos,
            self.zogy,
            model_type=model_type,
            box=box_size_pix,
            plate_scale=self.plate_scale,
        )
        model, residual = d.get_model(sigma=sigma)

        if plot:
            fig = plt.figure(figsize=(6, 2), dpi=100)
            # ax = fig.add_subplot(121, projection=w)
            ax = fig.add_subplot(121)
            ax.axis("off")
            im = ax.imshow(model, origin="lower", cmap=plt.cm.cividis)
            fig.colorbar(im, orientation="vertical")
            ax.set_title("zogy model", fontsize=6)
            ax2 = fig.add_subplot(122)
            ax2.axis("off")
            im2 = ax2.imshow(residual, origin="lower", cmap=plt.cm.cividis)
            fig.colorbar(im2, orientation="vertical")
            ax2.set_title("zogy minus model", fontsize=6)
            plt.show()

        return model, residual

    def pseudo_diff(
        self,
        xbox,
        ybox,
        model_type: str = "Moffat2D",
        model_box_size_pix: int = 20,
        sigma: float = 2.0,
        model_plot: bool = False,
        alerts=(),
    ):
        box_size_pix = xbox[1] - xbox[0]
        pseudo_diff = np.zeros((box_size_pix, box_size_pix))

        for alert in alerts:
            # positive = alert['candidate']['isdiffpos'] in (1, 't', '1')
            ra, dec = alert["candidate"]["ra"], alert["candidate"]["dec"]

            xpos, ypos = self.world2pix_sci(ra, dec, to_int=True)
            # alert falls into the box and not too close to the edge:
            if (
                xbox[0]
                < xpos - model_box_size_pix // 2
                < xpos + model_box_size_pix // 2
                < xbox[1]
            ) and (
                ybox[0]
                < ypos - model_box_size_pix // 2
                < ypos + model_box_size_pix // 2
                < ybox[1]
            ):
                xoffset = xpos - xbox[0]
                yoffset = ypos - ybox[0]
            else:
                continue

            m, r = self.model_zogy(
                ra=alert["candidate"]["ra"],
                dec=alert["candidate"]["dec"],
                model_type=model_type,
                box_size_pix=model_box_size_pix,
                sigma=sigma,
                plot=model_plot,
            )

            pseudo_diff[
                yoffset - m.shape[0] // 2 : yoffset + m.shape[0] // 2 + 1,
                xoffset - m.shape[1] // 2 : xoffset + m.shape[1] // 2 + 1,
            ] = m  # if positive else -m

        return pseudo_diff
