open_project -reset static_mux_test
add_files static_mux_tb_top.cpp
add_files -tb tb.cpp
set_top Mux
open_solution -reset sol1 -flow_target vitis
set_part  { xcu280-fsvh2892-2L-e }
create_clock -period 5.0


set_param hls.enable_hidden_option_error false
config_compile -disable_unroll_code_size_check -pipeline_style flp
config_interface -m_axi_addr64
config_rtl -module_auto_prefix
config_rtl -deadlock_detection none

csynth_design
csim_design
cosim_design -trace_level port
exit
