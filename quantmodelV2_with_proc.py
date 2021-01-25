from quant_modelV2 import QuantNet_opt
from processing import FINN_FACILE_Preproc as preproc
from processing import FINN_FACILE_Postproc as postproc
from torch.nn import Module
from constants import PREPROC_MINS, POSTPROC_MIN, PREPROC_INCRS_4b, POSTPROC_INCR_4b
from finn.util.pytorch import ToTensor

class QuantNet_opt_proc(Module):
    def __init__(self, n_features):
        super(QuantNet_opt_proc, self).__init__()
        self.net = QuantNet_opt(n_features)
        self.preproc = preproc(PREPROC_MINS, PREPROC_INCRS_4b)
        self.postproc = postproc(POSTPROC_MIN, POSTPROC_INCR_4b)
        #self.totensor = ToTensor()
        
    def forward(self, x):
        x = self.preproc(x)
        #x = self.totensor(x)
        x = self.net(x)
        x = self.postproc(x)
        return x
    
    def parameters(self):
        return self.net.parameters()
    
    def get_net(self):
        return self.net