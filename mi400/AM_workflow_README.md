# AM Workflow Guide for Pre-Silicon Simulation

## Table of Contents

1. [Introduction](#introduction)
2. [Creating GROUPFILES](#creating-groupfiles)
3. [Infrastructure Setup](#infrastructure-setup)
4. [Setting Up Your Environment](#setting-up-your-environment)
5. [Building AM with ROCcap 4.5.1](#building-am-with-roccap-451)
6. [Setting Up Kernel Simulation For Performance](#setting-up-kernel-simulation)
7. [Running and Monitoring Simulations](#running-and-monitoring-simulations)

---

## Introduction

**AM** is a tool used for pre-silicon GPU simulations. It requires:
- An SP3 file
- Memory configuration
- Register configuration

### Workflow Options

There are two ways to obtain the necessary setup files for AM simulations:

1. **Manual setup** (not recommended)
2. **CAP files** (recommended workflow)

**CAP files** are zipped archives containing everything needed for AM simulations. There are two workflows available:
- `cs_pm4` (which does not use cap files at all)
- `aqlplay` (uses CAP files and automates most of the workflow)

---

## Creating GROUPFILES

GROUPFILES configure how AM runs simulations. **Proper configuration is critical** - an incorrectly configured GROUPFILE will cause simulation failures.

### Configuration File Location

GROUPFILES are typically created in the test plan directory:
```
gc/src/am/test/suites/mi400/testplan/
```

### Key Configuration Options

#### Number of Shader Engines (SEs)
- Specify the number of Shader Engines in your simulation
- Example: `make_mi400_16cu_2se_1xcc` defines **2 shader engines**

#### Memory Modeling with GUMMI
- Enable GUMMI for more accurate memory modeling
- See the GUMMI Integration section below for parameters

### Example GROUPFILE Structure

```
group mi400am_1CP_1xcc_umc_loopback_cpfw --model=tb_am_rs64_fw --test-args "-use_kmd=1 -tc_BindAqlProcess=1 -fb_base=0x30000000000 -tc_EnableHIQ=0 -tc_LoadMesUCode=1" --pm4p2-args-end="make_mi400_16cu_2se_1xcc_cu_cache_l0_64k_lds_320k gfx11_pktplay_base_settings monitors.counters.perf.en_level=2 monitors.counters.perf.dump_freq=1000 test.force_flush_end_of_cb=true model.gpu.compute_only_model=true monitors.counters.perf.config_file=$ANCHOR_gfxperf/build/rhel7/perfmon/mi400_perfmon.yml monitors.counters.perf.config_file2=$STEM/gc/src/am/config/counters/mi400_miperf.yml model.gpu.sh.sa.tex.tcp.tcp_clause_enable=clause model.enable_multipipe=true make_mi400_xcd_ml_B0 make_mi400_1XCC_umc_const_delay_rd_320_wr_64_capped_hbm4_2p5kw_bw"
    group gclk1p7_gl2clk1p7 --pm4p2-args-end="make_mi450_gclk1p7_gl2clk1p7"
        group mtype_RW --pm4p2-args="model.gpu.sh.sa.tex.tcp.mtype_info=RW"
            <your_kernel_test>_cap {"lsf-machine" : "select[type==local && (gb128||csbatch)] rusage[mem=32000]"}
            #roc_capture_fp6_256x256x256_TT__pipelined_with_cluster_dispatch_cap {"lsf-machine" : "select[type==local && (gb128||csbatch)] rusage[mem=32000]"}
            #roc_capture_fp6_256x256x256_TT__pipelined_with_non_cluster_dispatch_cap {"lsf-machine" : "select[type==local && (gb128||csbatch)] rusage[mem=32000]"}
```

### Important Arguments

#### `--model`
Specifies the simulation model. Options include:
- `tb_am`
- `am`
- `tb_am_rs64_fw` (recommended for this workflow)

#### `--pm4p2-args-end`
Example: `make_mi400_16cu_2se_1xcc_cu_cache_l0_64k_lds_320k`

This defines the GPU configuration being simulated:
- **16** compute units (CU) per SE
- **2** shader engines (SE) per XCC
- **1** XCC (cross-chip connection)
- L0 cache: 64KB
- LDS: 320KB

#### GUMMI Integration
To enable GUMMI (for more accurate memory modeling), add these parameters to the group files:
- `--compass`
- `--compass-config`
- `--compass-arg`

USE THIS GROUPFILE FOR GUMMI RUNS!

group mi400am_1CP_1xcc_UMC_Loopback_cpfw --model=tb_am_rs64_fw --test-args "-use_kmd=1 -tc_BindAqlProcess=1 -fb_base=0x30000000000 -tc_EnableHIQ=0 -tc_LoadMesUCode=1" --pm4p2-args-end="make_mi400_16cu_2se_1xcc_cu_cache_l0_64k_lds_320k gfx11_pktplay_base_settings monitors.counters.perf.en_level=2 monitors.counters.perf.dump_freq=1000 test.force_flush_end_of_cb=true model.gpu.compute_only_model=true monitors.counters.perf.config_file=$ANCHOR_gfxperf/build/rhel7/perfmon/mi400_perfmon.yml monitors.counters.perf.config_file2=$STEM/gc/src/am/config/counters/mi400_miperf.yml model.gpu.sh.sa.tex.tcp.tcp_clause_enable=clause model.enable_multipipe=true make_mi400_xcd_ml_B0 make_mi400_1XCC_umc_const_delay_rd_320_wr_64_capped_hbm4_2p5kw_bw"
group mi400am_1CP_1xcc_COMPASS_cpfw --model=tb_am_rs64_fw --test-args "-use_kmd=1 -tc_BindAqlProcess=1 -fb_base=0x30000000000 -tc_EnableHIQ=0 -tc_LoadMesUCode=1" --compass  --compass-arg="wallclockdevice-0/progressStatsPs=100000" --compass-arg="outputTimeseriesFile=True" --compass-arg=".*/dfPoolSize=131072"   --compass-arg="generic_sdptrace_replay_device.*/trafficTurnedOff=True"  --compass-arg="abstractsdma.*/trafficTurnedOff=True" --compass-config=deviceFiles/lightsaberGasketdeviceFiles/configs/am-mi450.cfg  --compass-config=deviceFiles/lightsaberGasketdeviceFiles/configs/MI450_B0_2p5KW_overrides.cfg --compass-config=deviceFiles/lightsaberGasketdeviceFiles/configs/MI450_B0_2p5KW_overrides_bw_bound.cfg --pm4p2-args-end="make_mi400_16cu_2se_1xcc_cu_cache_l0_64k_lds_320k_gummi gfx11_pktplay_base_settings monitors.counters.perf.en_level=2 monitors.counters.perf.dump_freq=1000 test.force_flush_end_of_cb=true model.gpu.compute_only_model=true monitors.counters.perf.config_file=$ANCHOR_gfxperf/build/rhel7/perfmon/mi400_perfmon.yml monitors.counters.perf.config_file2=$STEM/gc/src/am/config/counters/mi400_miperf.yml model.gpu.sh.sa.tex.tcp.tcp_clause_enable=clause model.enable_multipipe=true make_mi400_xcd_ml_B0"
    group gclk1p7_gl2clk1p7 --pm4p2-args-end="make_mi450_gclk1p7_gl2clk1p7"
        group mtype_RW --pm4p2-args="model.gpu.sh.sa.tex.tcp.mtype_info=RW"
            roc_capture_fp6_256x256x256_TT__pipelined_with_cluster_dispatch_cap {"lsf-machine" : "select[type==local && (gb128||csbatch)] rusage[mem=32000]"}
            roc_capture_fp6_256x256x256_TT__pipelined_with_non_cluster_dispatch_cap {"lsf-machine" : "select[type==local && (gb128||csbatch)] rusage[mem=32000]"}
            <YOUR_TESTNAME_IN_AQLPLAY.TXT>_cap {"lsf-machine" : "select[type==local && (gb128||csbatch)] rusage[mem=32000]"}

### Creating Your GROUPFILE

1. Navigate to the test plan directory:
```bash
cd $STEM/gc/src/am/test/suites/mi400/testplan/
```

2. Create a new GROUPFILE (e.g., `my_kernel_tests.txt`):
```bash
vim my_kernel_tests.txt
```

3. Add your configuration using the example structure above, customizing:
   - The test name (e.g., `<your_kernel_test>_cap`)
   - GPU configuration parameters
   - Memory settings

### Expected Output After Creation

After creating your GROUPFILE, you should have:

**File created**:
```
$STEM/gc/src/am/test/suites/mi400/testplan/my_kernel_tests.txt
```

**To verify your GROUPFILE**:
```bash
ls -la $STEM/gc/src/am/test/suites/mi400/testplan/my_kernel_tests.txt
cat $STEM/gc/src/am/test/suites/mi400/testplan/my_kernel_tests.txt
```

**Expected output**:
```
-rw-r--r-- 1 <username> <group> <size> <date> my_kernel_tests.txt
```

You should see your GROUPFILE content displayed, confirming it's properly saved and ready to be used with `perf_runner` (see [Running and Monitoring Simulations](#running-and-monitoring-simulations)).

---

## Infrastructure Setup

### Overview

- **Simulations run on**: Atlanta server farm (has necessary infrastructure)
- **ETX machine**: Used to schedule jobs (do NOT run simulations directly on ETX)

### Environment Initialization

Run these commands to set up your environment on ETX:

```bash
source /proj/verif_release_ro/cbwa_initscript/current/cbwa_init.csh
bootenv -u AM:rel -v mi400
```

These commands initialize aliases, environment variables, and paths required for AM.

> **Note**: This setup works on both ETX machines and server farm nodes.

---

## Setting Up Your Environment

### Step 1: Generate CAP Files

Generate CAP files using the FFM documentation in Triton for your target kernels.

### Step 2: Access an ETX Machine

1. Go to https://myetx.amd.com:8443/etx/ and download the client
2. Select **XFCE 8G** or **FVWM** profile
3. After setup, you can SSH directly into the machine instead of using the GUI on your company laptop

> **Tip**: Type `bash` to switch from the default C shell to Bash (version 4.4).

### Step 3: Create Your Working Directory

```bash
cd /proj/mi450_runs/
mkdir -p /proj/mi450_runs/${USERNAME}_triton/
```

### Step 4: Initialize Infrastructure

For **csh**:
```bash
source /proj/verif_release_ro/cbwa_initscript/current/cbwa_init.csh
```

For **bash**:
```bash
source /proj/verif_release_ro/cbwa_initscript/current/cbwa_init.bash
```

### Step 5: Access the Codebase

```bash
p4_mkwa -codeline gfxip_mi400
```

> **Note**: This requires specific techaccess groups. Check the shared loop doc for required permissions.

### Step 6: Configure AM Environment

```bash
bootenv -u AM:rel -v mi400
```

> **Troubleshooting**: If `bootenv` is not available, re-run Step 4.

### Step 7: Build AM

Use `lsf_bsub` to schedule the build on a server farm node:

```bash
lsf_bsub -I -JUSER=$USER -q normal -P gfxip-am-mi \
  -R "rusage[mem=32000] select[type==RHEL7_64]" \
  dj -c -S -e gc.top.quick_am -DCMAKE -DTB_AM -m 12
```

**What this does**:
- Schedules a job on a RHEL7_64 machine
- Uses `dj` (AM team's compilation orchestration tool)
- Builds with 12 cores (adjust `-m` flag as needed)

> **Warning**: This step can take several hours.

---

## Building AM with ROCcap 4.5.1

### Why ROCcap 4.5.1?

ROCcap 4.5.1 fixes critical bugs. **Always use this version.**

### Step 1: Download ROCcap

Download from: https://atlartifactory.amd.com:8443/artifactory/HW-RocPlayCap-REL/

> **Note**: At the time of writing, it's in the pre-release section.

### Step 2: Extract ROCcap

```bash
tar -xvf rocplaycap-src-4.5.1.tar.gz
```

### Step 3: Configure Module File

1. Navigate to `_env/local/modulefile` in your AM directory
2. Find the ROCcap section
3. Modify it to use the extracted ROCcap version:

```bash
module load $STEM/rocplaycap-src-4.5.1/modulefile
```

4. The next time you run `bootenv`, it will validate the configuration

> **Note**: You may need to adjust for older shell compatibility, but fixes should be straightforward.

### Step 4: Patch Utility Code

Edit `gc/src/test/tools/utility/utility.cpp` and add this at the end of the file:

```cpp
DllImportExport_UTILITY UtilStatus utilGetLastParam(const char* param, int64_t& value)
{
    return utilGetLastParam(param, (int64&) value);
}

DllImportExport_UTILITY UtilStatus utilGetLastParam(const char* param, uint64_t& value)
{
    return utilGetLastParam(param, (uint64&) value);
}
```

### Step 5: Build AM with ROCcap

Ensure your environment is properly configured, then run:

```bash
dj -e gc::utility.build_self
dj -e rocplaycap::test_tools_roccap_aqlplay.build_self
cd $STEM/out/linux_3.10.0_64.VCS/tmp/gfxsim/
make -j30 all install
cd $STEM
```

---

## Setting Up Kernel Simulation

### Step 1: Generating Meta-Data for a run

To prevent tracking problems, every workload must carry its metadata directly; embedding it in the file name is undesirable since it makes names overly long and unreadable. Put your CAP file to a dedicated folder - e.g., ${USERNAME}-kernel-<int>, where <int> as an arbitrary integer number. You must include IRs of your kernel as well. Here is the structure of a workload folder:

```bash
$ tree ${USERNAME}-kernel-<int>
${USERNAME}-kernel-<int>
|-- README.md
|-- irs
|   |-- kernel.amdgcn
|   |-- kernel.hsaco
|   |-- kernel.llir
|   `-- kernel.ttgir
`-- kernel.cap
```

Execute your kernel before capturing with rocplay using the following Triton's env. variables to generate IRs:

```bash
$ TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=$(realpath ./tmp) python3 <triton-kernel>.py ...
$ cp ./tmp/*/* ${USERNAME}-kernel-<int>/irs
```

Write/generate the following `README.md` file:

```bash
$ cat ./README.md
# <> - fill out the field

# include Triton and LLVM commits
Triton commit: <>
LLVM commit: <>

# write the exact cmd command to run a kernel
CMD: <>

# include any extra used env. variables - e.g., TRITON_HIP_USE_ASYNC_COPY=OFF
ENV: <>

FFM verion/commit: <>
Rocplaycap version: <>

# Leave a comment (optional) - e.g., MXFP FA with e4m3 with fix `A`, `B` and `C`
Comment: <>
```

### Step 2: Transfer CAP Files

Copy your generated folder with the CAP file to the Atlanta filesystem:

```bash
# Example location:
/proj/mi450_runs/${USERNAME}_triton/work/
```

### Step 3: Configure AQLPLAY Test

Edit `gc/src/am/test/tests/dv/aqlplay.txt` and add a new entry:

```
AQLPLAY(gemm_gfx1250_warp8,
    test.file="/proj/mi450_runs/${USERNAME}_triton/work/${USERNAME}-kernel-<int>/kernel.cap";
    test.ini.test_args.aqlplay_tracefile=/proj/mi450_runs/${USERNAME}_triton/work/${USERNAME}-kernel-<int>/kernel.cap;
    test.ini.test_args.tc_PageTableRegionBase=0x100000000;
    test.ini.test_args.tc_PageTableRegionSize=0xf00000000;
    test.ini.test_args.tc_FBLocation=0x20000000000;
    test.ini.test_args.fb_base=0x20000000000;
)
```

### Step 3: Update Test Plan

Edit `gc/src/am/test/suites/mi400/testplan/aql_cluster_dispatch_bringup.txt` to use your GROUPFILE.

> **Tip**: Use the example GROUPFILE from earlier as a template with minor adjustments.

### Step 4: Rebuild AM (Optional)

This step may not be required, but if needed:

```bash
cd $STEM/out/linux_3.10.0_64.VCS/tmp/gfxsim/
make -j64 install
```

---

## Running and Monitoring Simulations

### Schedule Your Simulation

```bash
perf_runner --lsf \
  --lsf-machine="select[type==RHEL8_64] rusage[mem=32000]" \
  --outdir /proj/mi450_runs/${USERNAME}_triton/testing_fb_base_quick_check \
  --timeout 345600 \
  /proj/mi450_runs/${USERNAME}_triton/gc/src/am/test/suites/mi400/testplan/aql_cluster_dispatch_bringup.txt
```

> **Note**: Simulations can take considerable time to complete.

### Monitor Your Simulation

Access the log viewer at:

```
http://logviewer-atl.amd.com/proj/mi450_runs/${USERNAME}_triton/<output_directory>/.report/home.html
```

When it is completed, you will see a bunch of plots with the performance metrics you are interested in.

**Example**:
```
http://logviewer-atl.amd.com/proj/mi450_runs/djavady_triton/testing_fb_base_quick_check/.report/home.html
```

> **Important**: Update the URL to match your username and output directory name.

---

## Tips and Best Practices

- Always use CAP files workflow for easier setup
- Double-check your GROUPFILE configuration before running simulations
- Use ROCcap 4.5.1 or later for bug fixes
- Monitor your jobs through the log viewer
- ETX machines are for scheduling only - actual simulations run on server farm nodes
