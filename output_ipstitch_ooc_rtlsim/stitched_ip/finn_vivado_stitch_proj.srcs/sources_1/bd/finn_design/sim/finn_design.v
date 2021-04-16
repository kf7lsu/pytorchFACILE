//Copyright 1986-2019 Xilinx, Inc. All Rights Reserved.
//--------------------------------------------------------------------------------
//Tool Version: Vivado v.2019.1.3 (lin64) Build 2644227 Wed Sep  4 09:44:18 MDT 2019
//Date        : Wed Apr 14 17:53:32 2021
//Host        : finn_dev_mtrahms running 64-bit unknown
//Command     : generate_target finn_design.bd
//Design      : finn_design
//Purpose     : IP block netlist
//--------------------------------------------------------------------------------
`timescale 1 ps / 1 ps

module StreamingFCLayer_Batch_0_imp_5FXKV0
   (ap_clk,
    ap_rst_n,
    in0_V_V_tdata,
    in0_V_V_tready,
    in0_V_V_tvalid,
    out_V_V_tdata,
    out_V_V_tready,
    out_V_V_tvalid);
  input ap_clk;
  input ap_rst_n;
  input [7:0]in0_V_V_tdata;
  output in0_V_V_tready;
  input in0_V_V_tvalid;
  output [7:0]out_V_V_tdata;
  input out_V_V_tready;
  output out_V_V_tvalid;

  wire [7:0]StreamingFCLayer_Batch_0_out_V_V_TDATA;
  wire StreamingFCLayer_Batch_0_out_V_V_TREADY;
  wire StreamingFCLayer_Batch_0_out_V_V_TVALID;
  wire [7:0]StreamingFCLayer_Batch_0_wstrm_m_axis_0_TDATA;
  wire StreamingFCLayer_Batch_0_wstrm_m_axis_0_TREADY;
  wire StreamingFCLayer_Batch_0_wstrm_m_axis_0_TVALID;
  wire ap_clk_1;
  wire ap_rst_n_1;
  wire [7:0]in0_V_V_1_TDATA;
  wire in0_V_V_1_TREADY;
  wire in0_V_V_1_TVALID;

  assign StreamingFCLayer_Batch_0_out_V_V_TREADY = out_V_V_tready;
  assign ap_clk_1 = ap_clk;
  assign ap_rst_n_1 = ap_rst_n;
  assign in0_V_V_1_TDATA = in0_V_V_tdata[7:0];
  assign in0_V_V_1_TVALID = in0_V_V_tvalid;
  assign in0_V_V_tready = in0_V_V_1_TREADY;
  assign out_V_V_tdata[7:0] = StreamingFCLayer_Batch_0_out_V_V_TDATA;
  assign out_V_V_tvalid = StreamingFCLayer_Batch_0_out_V_V_TVALID;
  finn_design_StreamingFCLayer_Batch_0_0 StreamingFCLayer_Batch_0
       (.ap_clk(ap_clk_1),
        .ap_rst_n(ap_rst_n_1),
        .in0_V_V_TDATA(in0_V_V_1_TDATA),
        .in0_V_V_TREADY(in0_V_V_1_TREADY),
        .in0_V_V_TVALID(in0_V_V_1_TVALID),
        .out_V_V_TDATA(StreamingFCLayer_Batch_0_out_V_V_TDATA),
        .out_V_V_TREADY(StreamingFCLayer_Batch_0_out_V_V_TREADY),
        .out_V_V_TVALID(StreamingFCLayer_Batch_0_out_V_V_TVALID),
        .weights_V_V_TDATA(StreamingFCLayer_Batch_0_wstrm_m_axis_0_TDATA),
        .weights_V_V_TREADY(StreamingFCLayer_Batch_0_wstrm_m_axis_0_TREADY),
        .weights_V_V_TVALID(StreamingFCLayer_Batch_0_wstrm_m_axis_0_TVALID));
  finn_design_StreamingFCLayer_Batch_0_wstrm_0 StreamingFCLayer_Batch_0_wstrm
       (.aclk(ap_clk_1),
        .araddr({1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0}),
        .aresetn(ap_rst_n_1),
        .arprot({1'b0,1'b0,1'b0}),
        .arvalid(1'b0),
        .awaddr({1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0}),
        .awprot({1'b0,1'b0,1'b0}),
        .awvalid(1'b0),
        .bready(1'b0),
        .m_axis_0_tdata(StreamingFCLayer_Batch_0_wstrm_m_axis_0_TDATA),
        .m_axis_0_tready(StreamingFCLayer_Batch_0_wstrm_m_axis_0_TREADY),
        .m_axis_0_tvalid(StreamingFCLayer_Batch_0_wstrm_m_axis_0_TVALID),
        .rready(1'b0),
        .wdata({1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0}),
        .wstrb({1'b1,1'b1,1'b1,1'b1}),
        .wvalid(1'b0));
endmodule

module StreamingFCLayer_Batch_1_imp_11SOJ3N
   (ap_clk,
    ap_rst_n,
    in0_V_V_tdata,
    in0_V_V_tready,
    in0_V_V_tvalid,
    out_V_V_tdata,
    out_V_V_tready,
    out_V_V_tvalid);
  input ap_clk;
  input ap_rst_n;
  input [7:0]in0_V_V_tdata;
  output in0_V_V_tready;
  input in0_V_V_tvalid;
  output [7:0]out_V_V_tdata;
  input out_V_V_tready;
  output out_V_V_tvalid;

  wire [7:0]StreamingFCLayer_Batch_1_out_V_V_TDATA;
  wire StreamingFCLayer_Batch_1_out_V_V_TREADY;
  wire StreamingFCLayer_Batch_1_out_V_V_TVALID;
  wire [7:0]StreamingFCLayer_Batch_1_wstrm_m_axis_0_TDATA;
  wire StreamingFCLayer_Batch_1_wstrm_m_axis_0_TREADY;
  wire StreamingFCLayer_Batch_1_wstrm_m_axis_0_TVALID;
  wire ap_clk_1;
  wire ap_rst_n_1;
  wire [7:0]in0_V_V_1_TDATA;
  wire in0_V_V_1_TREADY;
  wire in0_V_V_1_TVALID;

  assign StreamingFCLayer_Batch_1_out_V_V_TREADY = out_V_V_tready;
  assign ap_clk_1 = ap_clk;
  assign ap_rst_n_1 = ap_rst_n;
  assign in0_V_V_1_TDATA = in0_V_V_tdata[7:0];
  assign in0_V_V_1_TVALID = in0_V_V_tvalid;
  assign in0_V_V_tready = in0_V_V_1_TREADY;
  assign out_V_V_tdata[7:0] = StreamingFCLayer_Batch_1_out_V_V_TDATA;
  assign out_V_V_tvalid = StreamingFCLayer_Batch_1_out_V_V_TVALID;
  finn_design_StreamingFCLayer_Batch_1_0 StreamingFCLayer_Batch_1
       (.ap_clk(ap_clk_1),
        .ap_rst_n(ap_rst_n_1),
        .in0_V_V_TDATA(in0_V_V_1_TDATA),
        .in0_V_V_TREADY(in0_V_V_1_TREADY),
        .in0_V_V_TVALID(in0_V_V_1_TVALID),
        .out_V_V_TDATA(StreamingFCLayer_Batch_1_out_V_V_TDATA),
        .out_V_V_TREADY(StreamingFCLayer_Batch_1_out_V_V_TREADY),
        .out_V_V_TVALID(StreamingFCLayer_Batch_1_out_V_V_TVALID),
        .weights_V_V_TDATA(StreamingFCLayer_Batch_1_wstrm_m_axis_0_TDATA),
        .weights_V_V_TREADY(StreamingFCLayer_Batch_1_wstrm_m_axis_0_TREADY),
        .weights_V_V_TVALID(StreamingFCLayer_Batch_1_wstrm_m_axis_0_TVALID));
  finn_design_StreamingFCLayer_Batch_1_wstrm_0 StreamingFCLayer_Batch_1_wstrm
       (.aclk(ap_clk_1),
        .araddr({1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0}),
        .aresetn(ap_rst_n_1),
        .arprot({1'b0,1'b0,1'b0}),
        .arvalid(1'b0),
        .awaddr({1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0}),
        .awprot({1'b0,1'b0,1'b0}),
        .awvalid(1'b0),
        .bready(1'b0),
        .m_axis_0_tdata(StreamingFCLayer_Batch_1_wstrm_m_axis_0_TDATA),
        .m_axis_0_tready(StreamingFCLayer_Batch_1_wstrm_m_axis_0_TREADY),
        .m_axis_0_tvalid(StreamingFCLayer_Batch_1_wstrm_m_axis_0_TVALID),
        .rready(1'b0),
        .wdata({1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0}),
        .wstrb({1'b1,1'b1,1'b1,1'b1}),
        .wvalid(1'b0));
endmodule

module StreamingFCLayer_Batch_2_imp_1YXLQV7
   (ap_clk,
    ap_rst_n,
    in0_V_V_tdata,
    in0_V_V_tready,
    in0_V_V_tvalid,
    out_V_V_tdata,
    out_V_V_tready,
    out_V_V_tvalid);
  input ap_clk;
  input ap_rst_n;
  input [7:0]in0_V_V_tdata;
  output in0_V_V_tready;
  input in0_V_V_tvalid;
  output [7:0]out_V_V_tdata;
  input out_V_V_tready;
  output out_V_V_tvalid;

  wire [7:0]StreamingFCLayer_Batch_2_out_V_V_TDATA;
  wire StreamingFCLayer_Batch_2_out_V_V_TREADY;
  wire StreamingFCLayer_Batch_2_out_V_V_TVALID;
  wire [7:0]StreamingFCLayer_Batch_2_wstrm_m_axis_0_TDATA;
  wire StreamingFCLayer_Batch_2_wstrm_m_axis_0_TREADY;
  wire StreamingFCLayer_Batch_2_wstrm_m_axis_0_TVALID;
  wire ap_clk_1;
  wire ap_rst_n_1;
  wire [7:0]in0_V_V_1_TDATA;
  wire in0_V_V_1_TREADY;
  wire in0_V_V_1_TVALID;

  assign StreamingFCLayer_Batch_2_out_V_V_TREADY = out_V_V_tready;
  assign ap_clk_1 = ap_clk;
  assign ap_rst_n_1 = ap_rst_n;
  assign in0_V_V_1_TDATA = in0_V_V_tdata[7:0];
  assign in0_V_V_1_TVALID = in0_V_V_tvalid;
  assign in0_V_V_tready = in0_V_V_1_TREADY;
  assign out_V_V_tdata[7:0] = StreamingFCLayer_Batch_2_out_V_V_TDATA;
  assign out_V_V_tvalid = StreamingFCLayer_Batch_2_out_V_V_TVALID;
  finn_design_StreamingFCLayer_Batch_2_0 StreamingFCLayer_Batch_2
       (.ap_clk(ap_clk_1),
        .ap_rst_n(ap_rst_n_1),
        .in0_V_V_TDATA(in0_V_V_1_TDATA),
        .in0_V_V_TREADY(in0_V_V_1_TREADY),
        .in0_V_V_TVALID(in0_V_V_1_TVALID),
        .out_V_V_TDATA(StreamingFCLayer_Batch_2_out_V_V_TDATA),
        .out_V_V_TREADY(StreamingFCLayer_Batch_2_out_V_V_TREADY),
        .out_V_V_TVALID(StreamingFCLayer_Batch_2_out_V_V_TVALID),
        .weights_V_V_TDATA(StreamingFCLayer_Batch_2_wstrm_m_axis_0_TDATA),
        .weights_V_V_TREADY(StreamingFCLayer_Batch_2_wstrm_m_axis_0_TREADY),
        .weights_V_V_TVALID(StreamingFCLayer_Batch_2_wstrm_m_axis_0_TVALID));
  finn_design_StreamingFCLayer_Batch_2_wstrm_0 StreamingFCLayer_Batch_2_wstrm
       (.aclk(ap_clk_1),
        .araddr({1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0}),
        .aresetn(ap_rst_n_1),
        .arprot({1'b0,1'b0,1'b0}),
        .arvalid(1'b0),
        .awaddr({1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0}),
        .awprot({1'b0,1'b0,1'b0}),
        .awvalid(1'b0),
        .bready(1'b0),
        .m_axis_0_tdata(StreamingFCLayer_Batch_2_wstrm_m_axis_0_TDATA),
        .m_axis_0_tready(StreamingFCLayer_Batch_2_wstrm_m_axis_0_TREADY),
        .m_axis_0_tvalid(StreamingFCLayer_Batch_2_wstrm_m_axis_0_TVALID),
        .rready(1'b0),
        .wdata({1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0}),
        .wstrb({1'b1,1'b1,1'b1,1'b1}),
        .wvalid(1'b0));
endmodule

module StreamingFCLayer_Batch_3_imp_RR03E4
   (ap_clk,
    ap_rst_n,
    in0_V_V_tdata,
    in0_V_V_tready,
    in0_V_V_tvalid,
    out_V_V_tdata,
    out_V_V_tready,
    out_V_V_tvalid);
  input ap_clk;
  input ap_rst_n;
  input [7:0]in0_V_V_tdata;
  output in0_V_V_tready;
  input in0_V_V_tvalid;
  output [7:0]out_V_V_tdata;
  input out_V_V_tready;
  output out_V_V_tvalid;

  wire [7:0]StreamingFCLayer_Batch_3_out_V_V_TDATA;
  wire StreamingFCLayer_Batch_3_out_V_V_TREADY;
  wire StreamingFCLayer_Batch_3_out_V_V_TVALID;
  wire [7:0]StreamingFCLayer_Batch_3_wstrm_m_axis_0_TDATA;
  wire StreamingFCLayer_Batch_3_wstrm_m_axis_0_TREADY;
  wire StreamingFCLayer_Batch_3_wstrm_m_axis_0_TVALID;
  wire ap_clk_1;
  wire ap_rst_n_1;
  wire [7:0]in0_V_V_1_TDATA;
  wire in0_V_V_1_TREADY;
  wire in0_V_V_1_TVALID;

  assign StreamingFCLayer_Batch_3_out_V_V_TREADY = out_V_V_tready;
  assign ap_clk_1 = ap_clk;
  assign ap_rst_n_1 = ap_rst_n;
  assign in0_V_V_1_TDATA = in0_V_V_tdata[7:0];
  assign in0_V_V_1_TVALID = in0_V_V_tvalid;
  assign in0_V_V_tready = in0_V_V_1_TREADY;
  assign out_V_V_tdata[7:0] = StreamingFCLayer_Batch_3_out_V_V_TDATA;
  assign out_V_V_tvalid = StreamingFCLayer_Batch_3_out_V_V_TVALID;
  finn_design_StreamingFCLayer_Batch_3_0 StreamingFCLayer_Batch_3
       (.ap_clk(ap_clk_1),
        .ap_rst_n(ap_rst_n_1),
        .in0_V_V_TDATA(in0_V_V_1_TDATA),
        .in0_V_V_TREADY(in0_V_V_1_TREADY),
        .in0_V_V_TVALID(in0_V_V_1_TVALID),
        .out_V_V_TDATA(StreamingFCLayer_Batch_3_out_V_V_TDATA),
        .out_V_V_TREADY(StreamingFCLayer_Batch_3_out_V_V_TREADY),
        .out_V_V_TVALID(StreamingFCLayer_Batch_3_out_V_V_TVALID),
        .weights_V_V_TDATA(StreamingFCLayer_Batch_3_wstrm_m_axis_0_TDATA),
        .weights_V_V_TREADY(StreamingFCLayer_Batch_3_wstrm_m_axis_0_TREADY),
        .weights_V_V_TVALID(StreamingFCLayer_Batch_3_wstrm_m_axis_0_TVALID));
  finn_design_StreamingFCLayer_Batch_3_wstrm_0 StreamingFCLayer_Batch_3_wstrm
       (.aclk(ap_clk_1),
        .araddr({1'b0,1'b0,1'b0,1'b0}),
        .aresetn(ap_rst_n_1),
        .arprot({1'b0,1'b0,1'b0}),
        .arvalid(1'b0),
        .awaddr({1'b0,1'b0,1'b0,1'b0}),
        .awprot({1'b0,1'b0,1'b0}),
        .awvalid(1'b0),
        .bready(1'b0),
        .m_axis_0_tdata(StreamingFCLayer_Batch_3_wstrm_m_axis_0_TDATA),
        .m_axis_0_tready(StreamingFCLayer_Batch_3_wstrm_m_axis_0_TREADY),
        .m_axis_0_tvalid(StreamingFCLayer_Batch_3_wstrm_m_axis_0_TVALID),
        .rready(1'b0),
        .wdata({1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0}),
        .wstrb({1'b1,1'b1,1'b1,1'b1}),
        .wvalid(1'b0));
endmodule

(* CORE_GENERATION_INFO = "finn_design,IP_Integrator,{x_ipVendor=xilinx.com,x_ipLibrary=BlockDiagram,x_ipName=finn_design,x_ipVersion=1.00.a,x_ipLanguage=VERILOG,numBlks=15,numReposBlks=11,numNonXlnxBlks=0,numHierBlks=4,maxHierDepth=1,numSysgenBlks=0,numHlsBlks=5,numHdlrefBlks=0,numPkgbdBlks=0,bdsource=USER,synth_mode=OOC_per_IP}" *) (* HW_HANDOFF = "finn_design.hwdef" *) 
module finn_design
   (ap_clk,
    ap_rst_n,
    m_axis_0_tdata,
    m_axis_0_tready,
    m_axis_0_tvalid,
    s_axis_0_tdata,
    s_axis_0_tready,
    s_axis_0_tvalid);
  (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 CLK.AP_CLK CLK" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME CLK.AP_CLK, ASSOCIATED_BUSIF s_axis_0:m_axis_0, ASSOCIATED_RESET ap_rst_n, CLK_DOMAIN finn_design_ap_clk_0, FREQ_HZ 5000000.000000, INSERT_VIP 0, PHASE 0.000" *) input ap_clk;
  (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 RST.AP_RST_N RST" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME RST.AP_RST_N, INSERT_VIP 0, POLARITY ACTIVE_LOW" *) input ap_rst_n;
  (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 m_axis_0 " *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME m_axis_0, CLK_DOMAIN finn_design_ap_clk_0, FREQ_HZ 5000000.000000, HAS_TKEEP 0, HAS_TLAST 0, HAS_TREADY 1, HAS_TSTRB 0, INSERT_VIP 0, LAYERED_METADATA undef, PHASE 0.000, TDATA_NUM_BYTES 1, TDEST_WIDTH 0, TID_WIDTH 0, TUSER_WIDTH 0" *) output [7:0]m_axis_0_tdata;
  (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 m_axis_0 " *) input m_axis_0_tready;
  (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 m_axis_0 " *) output m_axis_0_tvalid;
  (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 s_axis_0 " *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME s_axis_0, CLK_DOMAIN finn_design_ap_clk_0, FREQ_HZ 5000000.000000, HAS_TKEEP 0, HAS_TLAST 0, HAS_TREADY 1, HAS_TSTRB 0, INSERT_VIP 0, LAYERED_METADATA undef, PHASE 0.000, TDATA_NUM_BYTES 1, TDEST_WIDTH 0, TID_WIDTH 0, TUSER_WIDTH 0" *) input [7:0]s_axis_0_tdata;
  (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 s_axis_0 " *) output s_axis_0_tready;
  (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 s_axis_0 " *) input s_axis_0_tvalid;

  wire [7:0]StreamingFCLayer_Batch_0_out_V_V_TDATA;
  wire StreamingFCLayer_Batch_0_out_V_V_TREADY;
  wire StreamingFCLayer_Batch_0_out_V_V_TVALID;
  wire [7:0]StreamingFCLayer_Batch_1_out_V_V_TDATA;
  wire StreamingFCLayer_Batch_1_out_V_V_TREADY;
  wire StreamingFCLayer_Batch_1_out_V_V_TVALID;
  wire [7:0]StreamingFCLayer_Batch_2_out_V_V_TDATA;
  wire StreamingFCLayer_Batch_2_out_V_V_TREADY;
  wire StreamingFCLayer_Batch_2_out_V_V_TVALID;
  wire [7:0]StreamingFCLayer_Batch_3_out_V_V_TDATA;
  wire StreamingFCLayer_Batch_3_out_V_V_TREADY;
  wire StreamingFCLayer_Batch_3_out_V_V_TVALID;
  wire [7:0]StreamingFIFO_0_out_V_V_TDATA;
  wire StreamingFIFO_0_out_V_V_TREADY;
  wire StreamingFIFO_0_out_V_V_TVALID;
  wire [7:0]StreamingFIFO_1_out_V_V_TDATA;
  wire StreamingFIFO_1_out_V_V_TREADY;
  wire StreamingFIFO_1_out_V_V_TVALID;
  wire [7:0]Thresholding_Batch_0_out_V_V_TDATA;
  wire Thresholding_Batch_0_out_V_V_TREADY;
  wire Thresholding_Batch_0_out_V_V_TVALID;
  wire ap_clk_0_1;
  wire ap_rst_n_0_1;
  wire [7:0]in0_V_V_0_1_TDATA;
  wire in0_V_V_0_1_TREADY;
  wire in0_V_V_0_1_TVALID;

  assign Thresholding_Batch_0_out_V_V_TREADY = m_axis_0_tready;
  assign ap_clk_0_1 = ap_clk;
  assign ap_rst_n_0_1 = ap_rst_n;
  assign in0_V_V_0_1_TDATA = s_axis_0_tdata[7:0];
  assign in0_V_V_0_1_TVALID = s_axis_0_tvalid;
  assign m_axis_0_tdata[7:0] = Thresholding_Batch_0_out_V_V_TDATA;
  assign m_axis_0_tvalid = Thresholding_Batch_0_out_V_V_TVALID;
  assign s_axis_0_tready = in0_V_V_0_1_TREADY;
  StreamingFCLayer_Batch_0_imp_5FXKV0 StreamingFCLayer_Batch_0
       (.ap_clk(ap_clk_0_1),
        .ap_rst_n(ap_rst_n_0_1),
        .in0_V_V_tdata(StreamingFIFO_0_out_V_V_TDATA),
        .in0_V_V_tready(StreamingFIFO_0_out_V_V_TREADY),
        .in0_V_V_tvalid(StreamingFIFO_0_out_V_V_TVALID),
        .out_V_V_tdata(StreamingFCLayer_Batch_0_out_V_V_TDATA),
        .out_V_V_tready(StreamingFCLayer_Batch_0_out_V_V_TREADY),
        .out_V_V_tvalid(StreamingFCLayer_Batch_0_out_V_V_TVALID));
  StreamingFCLayer_Batch_1_imp_11SOJ3N StreamingFCLayer_Batch_1
       (.ap_clk(ap_clk_0_1),
        .ap_rst_n(ap_rst_n_0_1),
        .in0_V_V_tdata(StreamingFIFO_1_out_V_V_TDATA),
        .in0_V_V_tready(StreamingFIFO_1_out_V_V_TREADY),
        .in0_V_V_tvalid(StreamingFIFO_1_out_V_V_TVALID),
        .out_V_V_tdata(StreamingFCLayer_Batch_1_out_V_V_TDATA),
        .out_V_V_tready(StreamingFCLayer_Batch_1_out_V_V_TREADY),
        .out_V_V_tvalid(StreamingFCLayer_Batch_1_out_V_V_TVALID));
  StreamingFCLayer_Batch_2_imp_1YXLQV7 StreamingFCLayer_Batch_2
       (.ap_clk(ap_clk_0_1),
        .ap_rst_n(ap_rst_n_0_1),
        .in0_V_V_tdata(StreamingFCLayer_Batch_1_out_V_V_TDATA),
        .in0_V_V_tready(StreamingFCLayer_Batch_1_out_V_V_TREADY),
        .in0_V_V_tvalid(StreamingFCLayer_Batch_1_out_V_V_TVALID),
        .out_V_V_tdata(StreamingFCLayer_Batch_2_out_V_V_TDATA),
        .out_V_V_tready(StreamingFCLayer_Batch_2_out_V_V_TREADY),
        .out_V_V_tvalid(StreamingFCLayer_Batch_2_out_V_V_TVALID));
  StreamingFCLayer_Batch_3_imp_RR03E4 StreamingFCLayer_Batch_3
       (.ap_clk(ap_clk_0_1),
        .ap_rst_n(ap_rst_n_0_1),
        .in0_V_V_tdata(StreamingFCLayer_Batch_2_out_V_V_TDATA),
        .in0_V_V_tready(StreamingFCLayer_Batch_2_out_V_V_TREADY),
        .in0_V_V_tvalid(StreamingFCLayer_Batch_2_out_V_V_TVALID),
        .out_V_V_tdata(StreamingFCLayer_Batch_3_out_V_V_TDATA),
        .out_V_V_tready(StreamingFCLayer_Batch_3_out_V_V_TREADY),
        .out_V_V_tvalid(StreamingFCLayer_Batch_3_out_V_V_TVALID));
  finn_design_StreamingFIFO_0_0 StreamingFIFO_0
       (.ap_clk(ap_clk_0_1),
        .ap_rst_n(ap_rst_n_0_1),
        .in0_V_V_TDATA(in0_V_V_0_1_TDATA),
        .in0_V_V_TREADY(in0_V_V_0_1_TREADY),
        .in0_V_V_TVALID(in0_V_V_0_1_TVALID),
        .out_V_V_TDATA(StreamingFIFO_0_out_V_V_TDATA),
        .out_V_V_TREADY(StreamingFIFO_0_out_V_V_TREADY),
        .out_V_V_TVALID(StreamingFIFO_0_out_V_V_TVALID));
  finn_design_StreamingFIFO_1_0 StreamingFIFO_1
       (.ap_clk(ap_clk_0_1),
        .ap_rst_n(ap_rst_n_0_1),
        .in0_V_V_TDATA(StreamingFCLayer_Batch_0_out_V_V_TDATA),
        .in0_V_V_TREADY(StreamingFCLayer_Batch_0_out_V_V_TREADY),
        .in0_V_V_TVALID(StreamingFCLayer_Batch_0_out_V_V_TVALID),
        .out_V_V_TDATA(StreamingFIFO_1_out_V_V_TDATA),
        .out_V_V_TREADY(StreamingFIFO_1_out_V_V_TREADY),
        .out_V_V_TVALID(StreamingFIFO_1_out_V_V_TVALID));
  finn_design_Thresholding_Batch_0_0 Thresholding_Batch_0
       (.ap_clk(ap_clk_0_1),
        .ap_rst_n(ap_rst_n_0_1),
        .in0_V_V_TDATA(StreamingFCLayer_Batch_3_out_V_V_TDATA),
        .in0_V_V_TREADY(StreamingFCLayer_Batch_3_out_V_V_TREADY),
        .in0_V_V_TVALID(StreamingFCLayer_Batch_3_out_V_V_TVALID),
        .out_V_V_TDATA(Thresholding_Batch_0_out_V_V_TDATA),
        .out_V_V_TREADY(Thresholding_Batch_0_out_V_V_TREADY),
        .out_V_V_TVALID(Thresholding_Batch_0_out_V_V_TVALID));
endmodule
