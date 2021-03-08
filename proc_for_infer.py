from constants import PREPROC_MINS, POSTPROC_MIN, PREPROC_MAXES, POSTPROC_MAX
from constants import PREPROC_INCRS_4b as pre_incrs
from constants  import POSTPROC_INCR_4b as post_incr
import numpy as np

def preproc(x):
    x = x - np.asarray(PREPROC_MINS)
    x = x / np.asarray(pre_incrs)
    x = x.round()
    x = x.astype("int8")
    return x

def postproc(x):
    x = x.astype("float32")
    x = x * np.asarray(post_incr)
    x = x + np.asarray(POSTPROC_MIN)
    return x