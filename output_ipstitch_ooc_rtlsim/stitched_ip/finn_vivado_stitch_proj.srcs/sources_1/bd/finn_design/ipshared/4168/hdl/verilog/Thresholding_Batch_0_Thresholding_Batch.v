// ==============================================================
// RTL generated by Vivado(TM) HLS - High-Level Synthesis from C, C++ and SystemC
// Version: 2019.1.3
// Copyright (C) 1986-2019 Xilinx, Inc. All Rights Reserved.
// 
// ===========================================================

`timescale 1 ns / 1 ps 

module Thresholding_Batch_0_Thresholding_Batch (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        in_V_V_TDATA,
        in_V_V_TVALID,
        in_V_V_TREADY,
        out_V_V_TDATA,
        out_V_V_TVALID,
        out_V_V_TREADY,
        in_V_V_TDATA_blk_n,
        out_V_V_TDATA_blk_n
);

parameter    ap_ST_fsm_state1 = 1'd1;

input   ap_clk;
input   ap_rst;
input   ap_start;
output   ap_done;
output   ap_idle;
output   ap_ready;
input  [7:0] in_V_V_TDATA;
input   in_V_V_TVALID;
output   in_V_V_TREADY;
output  [7:0] out_V_V_TDATA;
output   out_V_V_TVALID;
input   out_V_V_TREADY;
output   in_V_V_TDATA_blk_n;
output   out_V_V_TDATA_blk_n;

reg ap_done;
reg ap_idle;
reg ap_ready;
reg in_V_V_TREADY;
reg out_V_V_TVALID;
reg in_V_V_TDATA_blk_n;
reg out_V_V_TDATA_blk_n;

(* fsm_encoding = "none" *) reg   [0:0] ap_CS_fsm;
wire    ap_CS_fsm_state1;
reg    ap_block_state1;
wire   [4:0] tmp_fu_115_p4;
wire   [0:0] icmp_ln899_fu_125_p2;
wire   [3:0] tmp_1_fu_135_p4;
wire   [0:0] icmp_ln899_1_fu_145_p2;
wire   [5:0] tmp_V_1_fu_111_p1;
wire   [0:0] icmp_ln899_2_fu_155_p2;
wire   [2:0] tmp_2_fu_165_p4;
wire   [0:0] icmp_ln899_3_fu_175_p2;
wire   [0:0] icmp_ln899_4_fu_185_p2;
wire   [0:0] icmp_ln899_5_fu_195_p2;
wire   [0:0] icmp_ln899_6_fu_205_p2;
wire   [1:0] tmp_3_fu_215_p4;
wire   [0:0] icmp_ln899_7_fu_225_p2;
wire   [0:0] icmp_ln899_8_fu_235_p2;
wire   [0:0] icmp_ln899_9_fu_245_p2;
wire   [0:0] icmp_ln899_10_fu_255_p2;
wire   [0:0] icmp_ln899_11_fu_265_p2;
wire   [0:0] icmp_ln899_12_fu_275_p2;
wire   [0:0] icmp_ln899_13_fu_285_p2;
wire   [0:0] icmp_ln899_14_fu_295_p2;
wire   [0:0] tmp_4_fu_305_p3;
wire   [0:0] icmp_ln899_15_fu_317_p2;
wire   [0:0] icmp_ln899_16_fu_327_p2;
wire   [0:0] icmp_ln899_17_fu_337_p2;
wire   [0:0] icmp_ln899_18_fu_347_p2;
wire   [0:0] icmp_ln899_19_fu_357_p2;
wire   [0:0] icmp_ln899_20_fu_367_p2;
wire   [0:0] icmp_ln899_21_fu_377_p2;
wire   [0:0] icmp_ln899_22_fu_387_p2;
wire   [0:0] icmp_ln899_23_fu_397_p2;
wire   [0:0] icmp_ln899_24_fu_407_p2;
wire   [0:0] icmp_ln899_25_fu_417_p2;
wire   [0:0] icmp_ln899_26_fu_427_p2;
wire   [0:0] icmp_ln899_27_fu_437_p2;
wire   [0:0] icmp_ln899_28_fu_447_p2;
wire   [0:0] icmp_ln899_29_fu_457_p2;
wire   [1:0] zext_ln899_fu_131_p1;
wire   [1:0] zext_ln899_1_fu_151_p1;
wire   [1:0] add_ln700_fu_467_p2;
wire   [1:0] zext_ln899_28_fu_313_p1;
wire   [1:0] add_ln700_1_fu_473_p2;
wire   [1:0] zext_ln899_2_fu_161_p1;
wire   [1:0] zext_ln899_3_fu_181_p1;
wire   [1:0] add_ln700_2_fu_483_p2;
wire   [1:0] zext_ln899_4_fu_191_p1;
wire   [1:0] zext_ln899_5_fu_201_p1;
wire   [1:0] add_ln700_3_fu_493_p2;
wire   [2:0] zext_ln700_4_fu_499_p1;
wire   [2:0] zext_ln700_3_fu_489_p1;
wire   [2:0] add_ln700_4_fu_503_p2;
wire   [2:0] zext_ln700_2_fu_479_p1;
wire   [2:0] add_ln700_5_fu_509_p2;
wire   [1:0] zext_ln899_6_fu_211_p1;
wire   [1:0] zext_ln899_7_fu_231_p1;
wire   [1:0] add_ln700_6_fu_519_p2;
wire   [1:0] zext_ln899_8_fu_241_p1;
wire   [1:0] zext_ln899_9_fu_251_p1;
wire   [1:0] add_ln700_7_fu_529_p2;
wire   [2:0] zext_ln700_7_fu_535_p1;
wire   [2:0] zext_ln700_6_fu_525_p1;
wire   [2:0] add_ln700_8_fu_539_p2;
wire   [1:0] zext_ln899_10_fu_261_p1;
wire   [1:0] zext_ln899_11_fu_271_p1;
wire   [1:0] add_ln700_9_fu_549_p2;
wire   [1:0] zext_ln899_12_fu_281_p1;
wire   [1:0] zext_ln899_13_fu_291_p1;
wire   [1:0] add_ln700_10_fu_559_p2;
wire   [2:0] zext_ln700_10_fu_565_p1;
wire   [2:0] zext_ln700_9_fu_555_p1;
wire   [2:0] add_ln700_11_fu_569_p2;
wire   [3:0] zext_ln700_11_fu_575_p1;
wire   [3:0] zext_ln700_8_fu_545_p1;
wire   [3:0] add_ln700_12_fu_579_p2;
wire   [3:0] zext_ln700_5_fu_515_p1;
wire   [3:0] add_ln700_13_fu_585_p2;
wire   [1:0] zext_ln700_fu_301_p1;
wire   [1:0] zext_ln899_14_fu_323_p1;
wire   [1:0] add_ln700_14_fu_595_p2;
wire   [1:0] zext_ln899_15_fu_333_p1;
wire   [1:0] zext_ln899_16_fu_343_p1;
wire   [1:0] add_ln700_15_fu_605_p2;
wire   [2:0] zext_ln700_14_fu_611_p1;
wire   [2:0] zext_ln700_13_fu_601_p1;
wire   [2:0] add_ln700_16_fu_615_p2;
wire   [1:0] zext_ln899_17_fu_353_p1;
wire   [1:0] zext_ln899_18_fu_363_p1;
wire   [1:0] add_ln700_17_fu_625_p2;
wire   [1:0] zext_ln899_19_fu_373_p1;
wire   [1:0] zext_ln899_20_fu_383_p1;
wire   [1:0] add_ln700_18_fu_635_p2;
wire   [2:0] zext_ln700_17_fu_641_p1;
wire   [2:0] zext_ln700_16_fu_631_p1;
wire   [2:0] add_ln700_19_fu_645_p2;
wire   [3:0] zext_ln700_18_fu_651_p1;
wire   [3:0] zext_ln700_15_fu_621_p1;
wire   [3:0] add_ln700_20_fu_655_p2;
wire   [1:0] zext_ln899_21_fu_393_p1;
wire   [1:0] zext_ln899_22_fu_403_p1;
wire   [1:0] add_ln700_21_fu_665_p2;
wire   [1:0] zext_ln899_23_fu_413_p1;
wire   [1:0] zext_ln899_24_fu_423_p1;
wire   [1:0] add_ln700_22_fu_675_p2;
wire   [2:0] zext_ln700_21_fu_681_p1;
wire   [2:0] zext_ln700_20_fu_671_p1;
wire   [2:0] add_ln700_23_fu_685_p2;
wire   [1:0] zext_ln899_25_fu_433_p1;
wire   [1:0] zext_ln899_26_fu_443_p1;
wire   [1:0] add_ln700_24_fu_695_p2;
wire   [1:0] zext_ln899_27_fu_453_p1;
wire   [1:0] zext_ln700_1_fu_463_p1;
wire   [1:0] add_ln700_25_fu_705_p2;
wire   [2:0] zext_ln700_24_fu_711_p1;
wire   [2:0] zext_ln700_23_fu_701_p1;
wire   [2:0] add_ln700_26_fu_715_p2;
wire   [3:0] zext_ln700_25_fu_721_p1;
wire   [3:0] zext_ln700_22_fu_691_p1;
wire   [3:0] add_ln700_27_fu_725_p2;
wire   [4:0] zext_ln700_26_fu_731_p1;
wire   [4:0] zext_ln700_19_fu_661_p1;
wire   [4:0] add_ln700_28_fu_735_p2;
wire   [4:0] zext_ln700_12_fu_591_p1;
wire   [4:0] tmp_V_fu_741_p2;
reg   [0:0] ap_NS_fsm;

// power-on initialization
initial begin
#0 ap_CS_fsm = 1'd1;
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_fsm <= ap_ST_fsm_state1;
    end else begin
        ap_CS_fsm <= ap_NS_fsm;
    end
end

always @ (*) begin
    if ((((ap_start == 1'b0) & (1'b1 == ap_CS_fsm_state1)) | (~((ap_start == 1'b0) | (out_V_V_TREADY == 1'b0) | (in_V_V_TVALID == 1'b0)) & (1'b1 == ap_CS_fsm_state1)))) begin
        ap_done = 1'b1;
    end else begin
        ap_done = 1'b0;
    end
end

always @ (*) begin
    if (((ap_start == 1'b0) & (1'b1 == ap_CS_fsm_state1))) begin
        ap_idle = 1'b1;
    end else begin
        ap_idle = 1'b0;
    end
end

always @ (*) begin
    if ((~((ap_start == 1'b0) | (out_V_V_TREADY == 1'b0) | (in_V_V_TVALID == 1'b0)) & (1'b1 == ap_CS_fsm_state1))) begin
        ap_ready = 1'b1;
    end else begin
        ap_ready = 1'b0;
    end
end

always @ (*) begin
    if (((ap_start == 1'b1) & (1'b1 == ap_CS_fsm_state1))) begin
        in_V_V_TDATA_blk_n = in_V_V_TVALID;
    end else begin
        in_V_V_TDATA_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((ap_start == 1'b0) | (out_V_V_TREADY == 1'b0) | (in_V_V_TVALID == 1'b0)) & (1'b1 == ap_CS_fsm_state1))) begin
        in_V_V_TREADY = 1'b1;
    end else begin
        in_V_V_TREADY = 1'b0;
    end
end

always @ (*) begin
    if (((ap_start == 1'b1) & (1'b1 == ap_CS_fsm_state1))) begin
        out_V_V_TDATA_blk_n = out_V_V_TREADY;
    end else begin
        out_V_V_TDATA_blk_n = 1'b1;
    end
end

always @ (*) begin
    if ((~((ap_start == 1'b0) | (out_V_V_TREADY == 1'b0) | (in_V_V_TVALID == 1'b0)) & (1'b1 == ap_CS_fsm_state1))) begin
        out_V_V_TVALID = 1'b1;
    end else begin
        out_V_V_TVALID = 1'b0;
    end
end

always @ (*) begin
    case (ap_CS_fsm)
        ap_ST_fsm_state1 : begin
            ap_NS_fsm = ap_ST_fsm_state1;
        end
        default : begin
            ap_NS_fsm = 'bx;
        end
    endcase
end

assign add_ln700_10_fu_559_p2 = (zext_ln899_12_fu_281_p1 + zext_ln899_13_fu_291_p1);

assign add_ln700_11_fu_569_p2 = (zext_ln700_10_fu_565_p1 + zext_ln700_9_fu_555_p1);

assign add_ln700_12_fu_579_p2 = (zext_ln700_11_fu_575_p1 + zext_ln700_8_fu_545_p1);

assign add_ln700_13_fu_585_p2 = (add_ln700_12_fu_579_p2 + zext_ln700_5_fu_515_p1);

assign add_ln700_14_fu_595_p2 = (zext_ln700_fu_301_p1 + zext_ln899_14_fu_323_p1);

assign add_ln700_15_fu_605_p2 = (zext_ln899_15_fu_333_p1 + zext_ln899_16_fu_343_p1);

assign add_ln700_16_fu_615_p2 = (zext_ln700_14_fu_611_p1 + zext_ln700_13_fu_601_p1);

assign add_ln700_17_fu_625_p2 = (zext_ln899_17_fu_353_p1 + zext_ln899_18_fu_363_p1);

assign add_ln700_18_fu_635_p2 = (zext_ln899_19_fu_373_p1 + zext_ln899_20_fu_383_p1);

assign add_ln700_19_fu_645_p2 = (zext_ln700_17_fu_641_p1 + zext_ln700_16_fu_631_p1);

assign add_ln700_1_fu_473_p2 = (add_ln700_fu_467_p2 + zext_ln899_28_fu_313_p1);

assign add_ln700_20_fu_655_p2 = (zext_ln700_18_fu_651_p1 + zext_ln700_15_fu_621_p1);

assign add_ln700_21_fu_665_p2 = (zext_ln899_21_fu_393_p1 + zext_ln899_22_fu_403_p1);

assign add_ln700_22_fu_675_p2 = (zext_ln899_23_fu_413_p1 + zext_ln899_24_fu_423_p1);

assign add_ln700_23_fu_685_p2 = (zext_ln700_21_fu_681_p1 + zext_ln700_20_fu_671_p1);

assign add_ln700_24_fu_695_p2 = (zext_ln899_25_fu_433_p1 + zext_ln899_26_fu_443_p1);

assign add_ln700_25_fu_705_p2 = (zext_ln899_27_fu_453_p1 + zext_ln700_1_fu_463_p1);

assign add_ln700_26_fu_715_p2 = (zext_ln700_24_fu_711_p1 + zext_ln700_23_fu_701_p1);

assign add_ln700_27_fu_725_p2 = (zext_ln700_25_fu_721_p1 + zext_ln700_22_fu_691_p1);

assign add_ln700_28_fu_735_p2 = (zext_ln700_26_fu_731_p1 + zext_ln700_19_fu_661_p1);

assign add_ln700_2_fu_483_p2 = (zext_ln899_2_fu_161_p1 + zext_ln899_3_fu_181_p1);

assign add_ln700_3_fu_493_p2 = (zext_ln899_4_fu_191_p1 + zext_ln899_5_fu_201_p1);

assign add_ln700_4_fu_503_p2 = (zext_ln700_4_fu_499_p1 + zext_ln700_3_fu_489_p1);

assign add_ln700_5_fu_509_p2 = (add_ln700_4_fu_503_p2 + zext_ln700_2_fu_479_p1);

assign add_ln700_6_fu_519_p2 = (zext_ln899_6_fu_211_p1 + zext_ln899_7_fu_231_p1);

assign add_ln700_7_fu_529_p2 = (zext_ln899_8_fu_241_p1 + zext_ln899_9_fu_251_p1);

assign add_ln700_8_fu_539_p2 = (zext_ln700_7_fu_535_p1 + zext_ln700_6_fu_525_p1);

assign add_ln700_9_fu_549_p2 = (zext_ln899_10_fu_261_p1 + zext_ln899_11_fu_271_p1);

assign add_ln700_fu_467_p2 = (zext_ln899_fu_131_p1 + zext_ln899_1_fu_151_p1);

assign ap_CS_fsm_state1 = ap_CS_fsm[32'd0];

always @ (*) begin
    ap_block_state1 = ((ap_start == 1'b0) | (in_V_V_TVALID == 1'b0));
end

assign icmp_ln899_10_fu_255_p2 = ((tmp_V_1_fu_111_p1 > 6'd21) ? 1'b1 : 1'b0);

assign icmp_ln899_11_fu_265_p2 = ((tmp_V_1_fu_111_p1 > 6'd23) ? 1'b1 : 1'b0);

assign icmp_ln899_12_fu_275_p2 = ((tmp_V_1_fu_111_p1 > 6'd25) ? 1'b1 : 1'b0);

assign icmp_ln899_13_fu_285_p2 = ((tmp_V_1_fu_111_p1 > 6'd27) ? 1'b1 : 1'b0);

assign icmp_ln899_14_fu_295_p2 = ((tmp_V_1_fu_111_p1 > 6'd29) ? 1'b1 : 1'b0);

assign icmp_ln899_15_fu_317_p2 = ((tmp_V_1_fu_111_p1 > 6'd33) ? 1'b1 : 1'b0);

assign icmp_ln899_16_fu_327_p2 = ((tmp_V_1_fu_111_p1 > 6'd35) ? 1'b1 : 1'b0);

assign icmp_ln899_17_fu_337_p2 = ((tmp_V_1_fu_111_p1 > 6'd37) ? 1'b1 : 1'b0);

assign icmp_ln899_18_fu_347_p2 = ((tmp_V_1_fu_111_p1 > 6'd39) ? 1'b1 : 1'b0);

assign icmp_ln899_19_fu_357_p2 = ((tmp_V_1_fu_111_p1 > 6'd41) ? 1'b1 : 1'b0);

assign icmp_ln899_1_fu_145_p2 = ((tmp_1_fu_135_p4 != 4'd0) ? 1'b1 : 1'b0);

assign icmp_ln899_20_fu_367_p2 = ((tmp_V_1_fu_111_p1 > 6'd43) ? 1'b1 : 1'b0);

assign icmp_ln899_21_fu_377_p2 = ((tmp_V_1_fu_111_p1 > 6'd45) ? 1'b1 : 1'b0);

assign icmp_ln899_22_fu_387_p2 = ((tmp_V_1_fu_111_p1 > 6'd47) ? 1'b1 : 1'b0);

assign icmp_ln899_23_fu_397_p2 = ((tmp_V_1_fu_111_p1 > 6'd49) ? 1'b1 : 1'b0);

assign icmp_ln899_24_fu_407_p2 = ((tmp_V_1_fu_111_p1 > 6'd51) ? 1'b1 : 1'b0);

assign icmp_ln899_25_fu_417_p2 = ((tmp_V_1_fu_111_p1 > 6'd53) ? 1'b1 : 1'b0);

assign icmp_ln899_26_fu_427_p2 = ((tmp_V_1_fu_111_p1 > 6'd55) ? 1'b1 : 1'b0);

assign icmp_ln899_27_fu_437_p2 = ((tmp_V_1_fu_111_p1 > 6'd57) ? 1'b1 : 1'b0);

assign icmp_ln899_28_fu_447_p2 = ((tmp_V_1_fu_111_p1 > 6'd59) ? 1'b1 : 1'b0);

assign icmp_ln899_29_fu_457_p2 = ((tmp_V_1_fu_111_p1 > 6'd61) ? 1'b1 : 1'b0);

assign icmp_ln899_2_fu_155_p2 = ((tmp_V_1_fu_111_p1 > 6'd5) ? 1'b1 : 1'b0);

assign icmp_ln899_3_fu_175_p2 = ((tmp_2_fu_165_p4 != 3'd0) ? 1'b1 : 1'b0);

assign icmp_ln899_4_fu_185_p2 = ((tmp_V_1_fu_111_p1 > 6'd9) ? 1'b1 : 1'b0);

assign icmp_ln899_5_fu_195_p2 = ((tmp_V_1_fu_111_p1 > 6'd11) ? 1'b1 : 1'b0);

assign icmp_ln899_6_fu_205_p2 = ((tmp_V_1_fu_111_p1 > 6'd13) ? 1'b1 : 1'b0);

assign icmp_ln899_7_fu_225_p2 = ((tmp_3_fu_215_p4 != 2'd0) ? 1'b1 : 1'b0);

assign icmp_ln899_8_fu_235_p2 = ((tmp_V_1_fu_111_p1 > 6'd17) ? 1'b1 : 1'b0);

assign icmp_ln899_9_fu_245_p2 = ((tmp_V_1_fu_111_p1 > 6'd19) ? 1'b1 : 1'b0);

assign icmp_ln899_fu_125_p2 = ((tmp_fu_115_p4 != 5'd0) ? 1'b1 : 1'b0);

assign out_V_V_TDATA = tmp_V_fu_741_p2;

assign tmp_1_fu_135_p4 = {{in_V_V_TDATA[5:2]}};

assign tmp_2_fu_165_p4 = {{in_V_V_TDATA[5:3]}};

assign tmp_3_fu_215_p4 = {{in_V_V_TDATA[5:4]}};

assign tmp_4_fu_305_p3 = in_V_V_TDATA[32'd5];

assign tmp_V_1_fu_111_p1 = in_V_V_TDATA[5:0];

assign tmp_V_fu_741_p2 = (add_ln700_28_fu_735_p2 + zext_ln700_12_fu_591_p1);

assign tmp_fu_115_p4 = {{in_V_V_TDATA[5:1]}};

assign zext_ln700_10_fu_565_p1 = add_ln700_10_fu_559_p2;

assign zext_ln700_11_fu_575_p1 = add_ln700_11_fu_569_p2;

assign zext_ln700_12_fu_591_p1 = add_ln700_13_fu_585_p2;

assign zext_ln700_13_fu_601_p1 = add_ln700_14_fu_595_p2;

assign zext_ln700_14_fu_611_p1 = add_ln700_15_fu_605_p2;

assign zext_ln700_15_fu_621_p1 = add_ln700_16_fu_615_p2;

assign zext_ln700_16_fu_631_p1 = add_ln700_17_fu_625_p2;

assign zext_ln700_17_fu_641_p1 = add_ln700_18_fu_635_p2;

assign zext_ln700_18_fu_651_p1 = add_ln700_19_fu_645_p2;

assign zext_ln700_19_fu_661_p1 = add_ln700_20_fu_655_p2;

assign zext_ln700_1_fu_463_p1 = icmp_ln899_29_fu_457_p2;

assign zext_ln700_20_fu_671_p1 = add_ln700_21_fu_665_p2;

assign zext_ln700_21_fu_681_p1 = add_ln700_22_fu_675_p2;

assign zext_ln700_22_fu_691_p1 = add_ln700_23_fu_685_p2;

assign zext_ln700_23_fu_701_p1 = add_ln700_24_fu_695_p2;

assign zext_ln700_24_fu_711_p1 = add_ln700_25_fu_705_p2;

assign zext_ln700_25_fu_721_p1 = add_ln700_26_fu_715_p2;

assign zext_ln700_26_fu_731_p1 = add_ln700_27_fu_725_p2;

assign zext_ln700_2_fu_479_p1 = add_ln700_1_fu_473_p2;

assign zext_ln700_3_fu_489_p1 = add_ln700_2_fu_483_p2;

assign zext_ln700_4_fu_499_p1 = add_ln700_3_fu_493_p2;

assign zext_ln700_5_fu_515_p1 = add_ln700_5_fu_509_p2;

assign zext_ln700_6_fu_525_p1 = add_ln700_6_fu_519_p2;

assign zext_ln700_7_fu_535_p1 = add_ln700_7_fu_529_p2;

assign zext_ln700_8_fu_545_p1 = add_ln700_8_fu_539_p2;

assign zext_ln700_9_fu_555_p1 = add_ln700_9_fu_549_p2;

assign zext_ln700_fu_301_p1 = icmp_ln899_14_fu_295_p2;

assign zext_ln899_10_fu_261_p1 = icmp_ln899_10_fu_255_p2;

assign zext_ln899_11_fu_271_p1 = icmp_ln899_11_fu_265_p2;

assign zext_ln899_12_fu_281_p1 = icmp_ln899_12_fu_275_p2;

assign zext_ln899_13_fu_291_p1 = icmp_ln899_13_fu_285_p2;

assign zext_ln899_14_fu_323_p1 = icmp_ln899_15_fu_317_p2;

assign zext_ln899_15_fu_333_p1 = icmp_ln899_16_fu_327_p2;

assign zext_ln899_16_fu_343_p1 = icmp_ln899_17_fu_337_p2;

assign zext_ln899_17_fu_353_p1 = icmp_ln899_18_fu_347_p2;

assign zext_ln899_18_fu_363_p1 = icmp_ln899_19_fu_357_p2;

assign zext_ln899_19_fu_373_p1 = icmp_ln899_20_fu_367_p2;

assign zext_ln899_1_fu_151_p1 = icmp_ln899_1_fu_145_p2;

assign zext_ln899_20_fu_383_p1 = icmp_ln899_21_fu_377_p2;

assign zext_ln899_21_fu_393_p1 = icmp_ln899_22_fu_387_p2;

assign zext_ln899_22_fu_403_p1 = icmp_ln899_23_fu_397_p2;

assign zext_ln899_23_fu_413_p1 = icmp_ln899_24_fu_407_p2;

assign zext_ln899_24_fu_423_p1 = icmp_ln899_25_fu_417_p2;

assign zext_ln899_25_fu_433_p1 = icmp_ln899_26_fu_427_p2;

assign zext_ln899_26_fu_443_p1 = icmp_ln899_27_fu_437_p2;

assign zext_ln899_27_fu_453_p1 = icmp_ln899_28_fu_447_p2;

assign zext_ln899_28_fu_313_p1 = tmp_4_fu_305_p3;

assign zext_ln899_2_fu_161_p1 = icmp_ln899_2_fu_155_p2;

assign zext_ln899_3_fu_181_p1 = icmp_ln899_3_fu_175_p2;

assign zext_ln899_4_fu_191_p1 = icmp_ln899_4_fu_185_p2;

assign zext_ln899_5_fu_201_p1 = icmp_ln899_5_fu_195_p2;

assign zext_ln899_6_fu_211_p1 = icmp_ln899_6_fu_205_p2;

assign zext_ln899_7_fu_231_p1 = icmp_ln899_7_fu_225_p2;

assign zext_ln899_8_fu_241_p1 = icmp_ln899_8_fu_235_p2;

assign zext_ln899_9_fu_251_p1 = icmp_ln899_9_fu_245_p2;

assign zext_ln899_fu_131_p1 = icmp_ln899_fu_125_p2;

endmodule //Thresholding_Batch_0_Thresholding_Batch
