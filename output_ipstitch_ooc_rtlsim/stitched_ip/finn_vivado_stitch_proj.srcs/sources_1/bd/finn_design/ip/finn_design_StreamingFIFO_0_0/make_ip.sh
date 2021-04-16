#!/bin/bash 
cd /tmp/finn_dev_mtrahms/code_gen_ipgen_StreamingFIFO_0_xl_srhbu/project_StreamingFIFO_0/sol1/impl/verilog
vivado -mode batch -source package_ip.tcl
cd /workspace/finn
