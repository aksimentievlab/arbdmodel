
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
    return _RESOURCE_DIR / relative_path
