from torch.nn import Module
from torch import tensor, int8, float32
import torch

#converts from floating point tensors closer to integer tensors
#created with a tensor of minimum values and increment values
#performs the operation result = (original - min)/increment
class FINN_FACILE_Preproc(Module):
    #OLD VERSION
    #Converts to int4 data
    #def __init__(self, mins, incrs):
    #    super(FINN_FACILE_Preproc, self).__init__()
    #    
    #    self.incrs = incrs
    #    self.mins = mins
    #
    #def forward(self, x):
    #    x = x - self.mins
    #    x = x / self.incrs
    #    x = torch.round(x)
    #    x = x.type(int8)
    #    x = x.type(float32)
    #    return x
    
    #NEW VERSION
    #Converts values to 0-1 tensor vals
    def __init(self, mins, maxes):
        super(FINN_FACILE_Preproc, self).__init__()
        
        self.mins = mins
        self.range = maxes-mins
        
    def forward(self, x):
        x = x - self.mins
        x = x / self.range
        return x

#converts from integer FACILE result to correct energy regression result
#created with a tensor representing the minimum value and the increment value
#performs the operation result = minimum + (original * increment)
class FINN_FACILE_Postproc(Module):
    #OLD VERSION
    #Converts from increment to result
    #def __init__(self, minval, incrval):
    #    super(FINN_FACILE_Postproc, self).__init__()
    #    
    #    self.minval = minval
    #    self.incrval = incrval
    #    
    #def forward(self, x):
    #    x = x.type(float32)
    #    x = x * self.incrval
    #    x = x + self.minval
    #    return x
    
    #NEW VERSION
    #Converts from 0 to 1 float value to result value
    def __init(self, minval, maxval):
        super(FINN_FACILE_Postproc, self).__init__()
        
        self.min = minval
        self.range = maxval - minval
        
    def forward(self, x):
        x = x * self.range
        x = x + self.min
        return x