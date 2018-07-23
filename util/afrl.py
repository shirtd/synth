from urllib3 import exceptions, disable_warnings
from ayasdi.core.api import Api
from config import AFRL_PATH, USER, PASS
import os, sys, urllib3

PTM = os.path.join(AFRL_PATH,'PTM')
SEGS = os.path.join(AFRL_PATH,'Segmentations')
sys.path.insert(0, PTM)

NBHD_STR = 'NeighborhoodLens'
ISO_STR = 'Isomapcoord'
GRP_STR = 'GROUPDATA'

disable_warnings(exceptions.InsecureRequestWarning)

def upload(s, force=False):
    print(' > connecting as %s' % USER)
    connection = Api(USER,PASS)
    sources = connection.get_sources()
    if s in [src.name for src in sources]:
        print(' ! source %s exists.')
        if force:
            print(' > deleting source %s' % s)
            connection.delete_source(name=s)
    else:
        print(' > uploading source %s' % s)
        return connection.upload_source(s)

def delete_src(s):
    print(' > connecting as %s' % USER)
    connection = Api(USER,PASS)
    print(' > deleting source %s' % s)
    return connection.delete_source(name=s)

def source_t(source):
    file,ext = os.path.splitext(source)
    return file + '_t' + ext

class Suppress:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = None
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout

def batch_name(k,g):
    s = NBHD_STR if NBHD_STR in k else ISO_STR
    return k.replace('%s1_%s2' % (s,s),s).replace(GRP_STR,g)
