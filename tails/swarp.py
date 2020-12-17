"""
 Mostly borrowed from Danny Goldstein's
 https://github.com/zuds-survey/zuds-pipeline

 https://github.com/zuds-survey/zuds-pipeline/blob/db775bba6e2319163d1b8838db5eb5c073d1bfe7/LICENSE
"""

from astropy.wcs import WCS
import pathlib


BKG_BOX_SIZE = 128
CONF_DIR = pathlib.Path(__file__).parent / "conf"
SCI_CONF = CONF_DIR / "default.swarp"


def prepare_swarp_align(stack, nthreads=1):
    conf = SCI_CONF

    directory = stack.path_ref.parent

    # shutil.copy(triplet.path_ref, directory)
    # impath = str(directory / triplet.path_ref.name)
    impath = str(stack.path_ref)
    align_header = stack.header_sci

    # now get the WCS keys to align the header to
    head = WCS(align_header).to_header(relax=True)

    # and write the results to a file that swarp will read
    extension = f"_aligned_to_{stack.path_sci.name[:-5]}.remap"

    outname = impath.replace(".fits", f"{extension}.fits")
    headpath = impath.replace(".fits", f"{extension}.head")

    with open(headpath, "w") as f:
        for card in align_header.cards:
            if card.keyword.startswith("NAXIS"):
                f.write(f"{card.image}\n")
        for card in head.cards:
            f.write(f"{card.image}\n")

    # make a random file for the weightmap -> we dont want to use it
    weightname = directory / stack.path_ref.name.replace(
        ".fits", f"{extension}.weight.fits"
    )

    combtype = "CLIPPED"
    verbose_type = "QUIET"
    # verbose_type = 'FULL'

    syscall = (
        f"swarp -c {conf} {impath} "
        f"-BACK_SIZE {BKG_BOX_SIZE} "
        f"-IMAGEOUT_NAME {outname} "
        f"-NTHREADS {nthreads} "
        f"-VMEM_DIR {directory} "
        f"-RESAMPLE_DIR {directory} "
        f"-SUBTRACT_BACK N "
        f"-WEIGHTOUT_NAME {weightname} "
        f"-WEIGHT_TYPE NONE "
        f"-COMBINE_TYPE {combtype} "
        f"-VERBOSE_TYPE {verbose_type} "
    )

    return syscall, outname, weightname
