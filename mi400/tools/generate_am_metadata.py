import os


def generate_aqlplay(names: list[str], capfile_root: str):
    aqlplay_file = """define(`AQLPLAY', `
define $1_cap
{

    test.ini.test_args.tcore_log=enabled;
    test.ini.test_args.sh_allow_gpr_read_before_write=true;
    test.ini.test_args.tR600DumpTexturePulledData=enable;
    test.ini.test_args.tc_tglHasChildOfVidmem=1;
    test.ini.test_args.tc_MicrocodeLoadEnable=1;
    test.ini.test_args.rlcFrontDoorLoad=1;
    test.ini.test_args.tc_EnableInterrupts=1;
    test.ini.test_args.tc_LoadMesUCode=0;
    test.ini.test_args.tc_DisableCpCmpRings=0;
        test.ini.test_args.tc_NoMmhubUTCL2=1;
        test.ini.test_args.tc_DisableCpGfxRings=1;
        test.ini.test_args.tc_EnableVM=1;
        test.ini.test_args.tc_EnableHIQ=1;
        test.ini.test_args.tc_PageTableRegionBase=0x7f0000000000;
        test.ini.test_args.tc_PageTableRegionSize=0xf000000000;
        test.ini.test_args.fbSizeInMBytes=1048576;
        test.ini.test_args.tc_VidMemSizeInBytes=1099511627776;
        test.ini.test_args.aqlplay_loglevel=trace;
        test.ini.test_args.aqlplay_driver=TC2;
        test.ini.test_args.aqlplay_syncAfterSubmit=1;
    test.ini.test_args.tc_ulVidmemFragmentSize=4;
    $2
}
')
"""

    for name in names:
        capfile_path = os.path.join(capfile_root, f'{name}.cap')
        body = """
AQLPLAY({},
    test.file="{}";
    test.ini.test_args.aqlplay_tracefile={};
    test.ini.test_args.tc_PageTableRegionBase=0x100000000;
    test.ini.test_args.tc_PageTableRegionSize=0xf00000000;
    test.ini.test_args.tc_FBLocation=0x20000000000;
    test.ini.test_args.fb_base=0x20000000000;
)
""".format(name, capfile_path, capfile_path)
        aqlplay_file += body

    return aqlplay_file


def generate_group_file(names: list[str], num_xcc: int, group_name: str, enable_itrace: bool = False,
                        enable_ttrace: bool = False):
    pm4p2_args = [
        "make_mi400_16cu_2se_1xcc_cu_cache_l0_64k_lds_320k",  #
        "gfx11_pktplay_base_settings",  #
        "monitors.counters.perf.en_level=2",  #
        "monitors.counters.perf.dump_freq=1000",  #
        "test.force_flush_end_of_cb=true",  #
        "model.gpu.compute_only_model=true",  #
        "monitors.counters.perf.config_file=$ANCHOR_gfxperf/build/rhel7/perfmon/mi400_perfmon.yml",  #
        "monitors.counters.perf.config_file2=$STEM/gc/src/am/config/counters/mi400_miperf.yml",  #
        "model.gpu.sh.sa.tex.tcp.tcp_clause_enable=clause",  #
        "model.enable_multipipe=true",  #
        "make_mi400_xcd_ml_B0",  #
        "make_mi400_1XCC_umc_const_delay_rd_320_wr_64_capped_hbm4_2p5kw_bw"
    ]

    if num_xcc == 8:
        pm4p2_args[0] = "make_mi400_16cu_2se_8xcc_8cp_cu_cache_l0_64k_lds_320k"
        pm4p2_args[-1] = "make_mi400_8XCC_umc_const_delay_rd_320_wr_64_capped_hbm4_2p5kw_bw"

    test_args = ["-use_kmd=1", "-tc_BindAqlProcess=1", "-tc_EnableHIQ=0", "-tc_LoadMesUCode=1"]
    if num_xcc == 8:
        test_args = ["-tg_chunksize=1", "-num_xcds=8"] + test_args

    args = [
        "--model=tb_am_rs64_fw",
        f'--test-args "{" ".join(test_args)}"',
        f'--pm4p2-args-end="{" ".join(pm4p2_args)}"',
    ]
    if enable_itrace:
        args.insert(0, '--itrace on')
    if enable_ttrace:
        args.insert(0, '--ttrace')
    group_file = '''group mi400am_1CP_1xcc_UMC_Loopback_cpfw {}
    group {} --pm4p2-args-end="make_mi450_gclk1p7_gl2clk1p7"'''.format(' '.join(args), group_name)
    for name in names:
        body = '''
        {}_cap {{"lsf-machine" : "select[type==local && (gb128||csbatch)] rusage[mem=32000]"}}'''.format(name)
        group_file += body
    return group_file


def main(args):
    if len(args.output_dir) > 0:
        os.makedirs(args.output_dir, exist_ok=True)

    group_file = generate_group_file(args.names, args.num_xcc, args.group_name, args.enable_itrace, args.enable_ttrace)
    aqlplay_file = generate_aqlplay(args.names, args.capfile_root)

    with open(os.path.join(args.output_dir, 'group_file.txt'), 'w', encoding='utf-8', newline='\n') as f:
        f.write(group_file)

    with open(os.path.join(args.output_dir, 'aqlplay.txt'), 'w', encoding='utf-8', newline='\n') as f:
        f.write(aqlplay_file)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Generate aqlplay.txt and group_file.txt based on provided cap file names and directory.', epilog=
        'Usage: python3 generate_metadata.py -o ./output -r /proj/triton_regr/TRITON/MXFP_FA -n roccap_1 [roccap_2 [...]] [-it] [-g CU_Tile_GEMM]'
    )
    parser.add_argument('-o', '--output_dir', type=str, default='',
                        help='Directory to save generated files, i.e. aqlplay.txt and group_file.txt')
    parser.add_argument(
        '-n', '--names', type=str, nargs='+',
        help='Name of the run, for example, if the cap file is roccap_1.cap, the name would be roccap_1')
    parser.add_argument('-g', '--group_name', type=str, default='XCC_FA', help='Name of the group')
    parser.add_argument('-r', '--capfile_root', type=str,
                        help='Root directory on ETX keeping the cap files, e.g. /proj/triton_regr/TRITON/MXFP_FA')
    parser.add_argument('-it', '--enable_itrace', action='store_true', help='Enable itrace or not')
    parser.add_argument('-tt', '--enable_ttrace', action='store_true', help='Enable ttrace or not')
    parser.add_argument('--num_xcc', type=int, choices=[1, 8], help='Number of XCCs')
    args = parser.parse_args()
    main(args)
