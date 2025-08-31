# AOHW25_242
An implementation of the Smith Waterman algorithm using multiple hardware accelerators

Link to the youtube video: `https://youtu.be/mdAmF2yWk70?si=ioC9uYCZffKO_hrQ`


## Repository Layout
- `source-files/` — design files.
  - `src/lsal.cpp` — hardware kernel (`compute_matrices`) implementation of the Smith–Waterman algorithm.
  - `src/lsal_host.cpp` — host application that launches the kernel - it uses XRT API and a CPU code for correctness checks and speedup evaluation.
  - `create_files.c` — generates `query.txt` and `database.txt` with random DNA sequences.
  - `design.cfg`, `xrt.ini` — compilation and runtime configuration files.

## Sourcing the Project
To source the project, you need clone the repository. This can be done by running the following commands:

```bash
git clone https://github.com/kellynanou/AOHW25_242.git
``` 

Or you can download as a zip.

Then simply:
```bash
cd AOHW25_242
```


## Prerequisites
- Xilinx XRT and Vitis 2023.1 or compatible.
- Source the environment before building:
  ```bash
  source /opt/xilinx/xrt/setup.sh
  source /opt/xilinx/Vitis/2023.1/settings64.sh
  ```
## Kernel Build

Enter the source directory:

```bash
cd source-files
```
The main build command builds the hardware kernel/kernels. Replace `<platform>` with your target FPGA device and `<host_arch>` with your host architecture.

```bash
make build TARGET=hw DEVICE=<platform> HOST_ARCH=<host_arch> COMMON_REPO=../ 
```
 For example, for ALVEO U250:

```bash
make build TARGET=hw DEVICE=xilinx_u250_gen3x16_xdma_4_1_202210_1 HOST_ARCH=x86 COMMON_REPO=../
```

This will generate the required binaries for hardware execution. The final binary `lsal.xclbin` will be in `build_dir.hw.xilinx_u250_gen3x16_xdma_4_1_202210_1/` or a similarly named folder.

If you wish to build for another kernel size combination make sure to change the Tile_N and/or Tile_M both in `lsal.cpp` and in `lsal_host.cpp`.

### Design.cfg

This file contains the design configuration for the hardware implementation.
Currently it is configured for 8 Hardware accelerators. You can change the number of accelerators in the `connectivity`section. Please be aware that if you change the number of accelerators, you may also need to adjust other parameters such as the memory banks section and in the host code the NumofCUs. This should be done carefully to ensure proper functionality.

## Host Build
To build the host application, run:

```bash
make host TARGET=hw DEVICE=<platform> HOST_ARCH=<host_arch> COMMON_REPO=../
```
or for the ALVEO U250 where we run our experiments:

```bash
make host TARGET=hw DEVICE=xilinx_u250_gen3x16_xdma_4_1_202210_1 HOST_ARCH=x86 COMMON_REPO=../ 
```
By default the flag -Dsw_validation is set. With this flag on, the host application will perform software validation of the results against a CPU implementation and provide a speedup of the CPU vs the hw implementation.

If you disable this flag, the host application will only run the hardware implementation without any software validation. This can be done to perform more tests, on greater size sequences in less time (because CPU takes more time).

## Generate input data
In the makefile you can change the size of the Query and Database sequence. To generate the sequences with the specified sizes, run:

```bash
make db
```

## Run

In order to run the application, use the following command:

```bash
.lsal <folder_bin>/lsal.xclbin query.txt database.txt <GlobalQuerySize> <GlobalDatabaseSize>
```
Example
```bash
./lsal build_dir.hw.xilinx_u250_gen3x16_xdma_4_1_202210_1/lsal.xclbin query.txt database.txt 1024 1048576
```

## Clean
After you're done with the experiments, you can clean the build artifacts by running:

```bash
make cleanall
```

## SW emulation

If you wish to perform software emulation (sw_emu), you can do so by building a different 

After setting the enviroment like in the HW execution section, you should also :

```bash
  export XCL_EMULATION_MODE=sw_emu
```
to enable software emulation mode.

### Kernel SW Emu
Now the building of the binary is different, since we are not targeting the hardware. So the command you should run is :

```bash
make build TARGET=sw_emu DEVICE=xilinx_u250_gen3x16_xdma_4_1_202210_1 HOST_ARCH=x86 COMMON_REPO=./ 
```
This build the software emulation binary and places it in a file like `build_dir.sw_emu.xilinx_u250_gen3x16_xdma_4_1_202210_1`, or a similar name if you are using a different platform.

### Host SW Emu
For the host executable, you should run:
```bash
make host HOST_ARCH=x86 COMMON_REPO=../ 
```
### Sequences SW Emu

You have to create the sequences for this case also. You can do so, by using the same command as prebiously mentioned.

### Run SW Emu

Now to perform the Software Emulation, we use the following command:
```bash
./lsal <sw_emu_folder_bin>/lsal.xclbin query.txt database.txt <GlobalQuerySize> <GlobalDatabaseSize>
```
or for example

```bash
./lsal build_dir.sw_emu.xilinx_u250_gen3x16_xdma_4_1_202210_1/lsal.xclbin query.txt database.txt 1024 1048576
```
