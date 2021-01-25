from constants import PREPROC_MINS, POSTPROC_MIN, PREPROC_MAXES, POSTPROC_MAX
from constants import PREPROC_INCRS_4b as pre_incrs
from constants  import POSTPROC_INCR_4b as post_incr
from torch import tensor, int8, float32
import torch

#OLD VERSION
#converts between UINT4 and actual vals
#def FACILE_preproc(x):
#    x = x - PREPROC_MINS
#    x = x / pre_incrs
#    x = torch.round(x)
#    x = x.type(int8)
#    x = x.type(float32)
#    return x
#
#def FACILE_postproc(x):
#    x = x.type(float32)
#    x = x * post_incr
#    x = x + POSTPROC_MIN
#    return x

#NEW VERSION
#converts between 0-1 float values and actual vals
def FACILE_preproc(x):
    r = PREPROC_MAXES - PREPROC_MINS
    x = x - PREPROC_MINS
    x = x / r
    return x

def FACILE_postproc(x):
    r = POSTPROC_MAX - POSTPROC_MIN
    x = x * r
    x = x + POSTPROC_MIN
    return x

def FACILE_preproc_out(x):
    r = POSTPROC_MAX - POSTPROC_MIN
    x = x - POSTPROC_MIN
    x = x / r
    return x