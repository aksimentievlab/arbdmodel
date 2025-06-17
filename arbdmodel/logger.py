
## Set up loggers
import logging
from pathlib import Path

def _get_username():
    import sys
    try:
        return sys.environ['USER']
    except:
        return None

logging.basicConfig(format='%(name)s: %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
_ch = logging.StreamHandler()
_ch.setFormatter(logging.Formatter('%(name)s: %(levelname)s: %(message)s'))
logger.addHandler(_ch)
logger.propagate = False

devlogger = logging.getLogger(__name__+'.dev')
# devlogger.setLevel(logging.DEBUG)
if _get_username() not in ('cmaffeo2',):
    devlogger.addHandler(logging.NullHandler())

_RESOURCE_DIR = Path(__file__).parent / 'resources'
def get_resource_path(relative_path):
    """
    Get the absolute path from a relative path by joining it with the resource directory.

    This function computes the absolute path by combining the base resource directory
    with the provided relative path.

    Parameters
    ----------
    relative_path : str or pathlib.Path
        The relative path to be combined with the resource directory.

    Returns
    -------
    pathlib.Path
        The absolute path to the resource.

    Examples
    --------
    >>> get_resource_path('data/config.json')
    PosixPath('/path/to/resources/data/config.json')
    """
    return _RESOURCE_DIR / relative_path
