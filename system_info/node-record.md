# Node record

Generated 2026-10-10T13:47Z by `system_info/collect.sh`. Measured values only; host names, GPU UUIDs, user names and paths are removed.

## Allocation
```
SLURM_JOB_ID=none  partition=none  gpus_on_node=none  CUDA_VISIBLE_DEVICES=unset
```

## GPUs
```
index, name, memory.total [MiB], driver_version, power.limit [W], pstate
0, NVIDIA H200, 143771 MiB, 570.211.01, 700.00 W, P0
1, NVIDIA H200, 143771 MiB, 570.211.01, 700.00 W, P0
2, NVIDIA H200, 143771 MiB, 570.211.01, 700.00 W, P0
3, NVIDIA H200, 143771 MiB, 570.211.01, 700.00 W, P0
4, NVIDIA H200, 143771 MiB, 570.211.01, 700.00 W, P0
5, NVIDIA H200, 143771 MiB, 570.211.01, 700.00 W, P0
6, NVIDIA H200, 143771 MiB, 570.211.01, 700.00 W, P0
7, NVIDIA H200, 143771 MiB, 570.211.01, 700.00 W, P0
```

## GPU driver and CUDA
```
| NVIDIA-SMI 570.211.01             Driver Version: 570.211.01     CUDA Version: 12.8     |
```

## GPU topology
```
	[4mGPU0	GPU1	GPU2	GPU3	GPU4	GPU5	GPU6	GPU7	NIC0	NIC1	NIC2	NIC3	NIC4	NIC5	NIC6	NIC7	CPU Affinity	NUMA Affinity	GPU NUMA ID[0m
GPU0	 X 	NV18	NV18	NV18	NV18	NV18	NV18	NV18	NODE	NODE	SYS	NODE	PIX	SYS	SYS	SYS	0-63,128-191	0		N/A
GPU1	NV18	 X 	NV18	NV18	NV18	NV18	NV18	NV18	NODE	PIX	SYS	NODE	NODE	SYS	SYS	SYS	0-63,128-191	0		N/A
GPU2	NV18	NV18	 X 	NV18	NV18	NV18	NV18	NV18	PIX	NODE	SYS	NODE	NODE	SYS	SYS	SYS	0-63,128-191	0		N/A
GPU3	NV18	NV18	NV18	 X 	NV18	NV18	NV18	NV18	NODE	NODE	SYS	PIX	NODE	SYS	SYS	SYS	0-63,128-191	0		N/A
GPU4	NV18	NV18	NV18	NV18	 X 	NV18	NV18	NV18	SYS	SYS	NODE	SYS	SYS	NODE	PIX	NODE	64-127,192-255	1		N/A
GPU5	NV18	NV18	NV18	NV18	NV18	 X 	NV18	NV18	SYS	SYS	NODE	SYS	SYS	NODE	NODE	PIX	64-127,192-255	1		N/A
GPU6	NV18	NV18	NV18	NV18	NV18	NV18	 X 	NV18	SYS	SYS	NODE	SYS	SYS	PIX	NODE	NODE	64-127,192-255	1		N/A
GPU7	NV18	NV18	NV18	NV18	NV18	NV18	NV18	 X 	SYS	SYS	PIX	SYS	SYS	NODE	NODE	NODE	64-127,192-255	1		N/A
NIC0	NODE	NODE	PIX	NODE	SYS	SYS	SYS	SYS	 X 	NODE	SYS	NODE	NODE	SYS	SYS	SYS				
NIC1	NODE	PIX	NODE	NODE	SYS	SYS	SYS	SYS	NODE	 X 	SYS	NODE	NODE	SYS	SYS	SYS				
NIC2	SYS	SYS	SYS	SYS	NODE	NODE	NODE	PIX	SYS	SYS	 X 	SYS	SYS	NODE	NODE	NODE				
NIC3	NODE	NODE	NODE	PIX	SYS	SYS	SYS	SYS	NODE	NODE	SYS	 X 	NODE	SYS	SYS	SYS				
NIC4	PIX	NODE	NODE	NODE	SYS	SYS	SYS	SYS	NODE	NODE	SYS	NODE	 X 	SYS	SYS	SYS				
NIC5	SYS	SYS	SYS	SYS	NODE	NODE	PIX	NODE	SYS	SYS	NODE	SYS	SYS	 X 	NODE	NODE				
NIC6	SYS	SYS	SYS	SYS	PIX	NODE	NODE	NODE	SYS	SYS	NODE	SYS	SYS	NODE	 X 	NODE				
NIC7	SYS	SYS	SYS	SYS	NODE	PIX	NODE	NODE	SYS	SYS	NODE	SYS	SYS	NODE	NODE	 X 				

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks

NIC Legend:

  NIC0: mlx5_0
  NIC1: mlx5_1
  NIC2: mlx5_2
  NIC3: mlx5_3
  NIC4: mlx5_4
  NIC5: mlx5_5
  NIC6: mlx5_6
  NIC7: mlx5_7

```

## CPU and memory
```
CPU(s):                                  256
Vendor ID:                               AuthenticAMD
Model name:                              AMD EPYC 9534 64-Core Processor
Thread(s) per core:                      2
Core(s) per socket:                      64
Socket(s):                               2
CPU(s) scaling MHz:                      58%
NUMA node(s):                            2
NUMA node0 CPU(s):                       0-63,128-191
NUMA node1 CPU(s):                       64-127,192-255

               total        used        free      shared  buff/cache   available
Mem:           1.5Ti       872Gi       164Gi       5.0Gi       497Gi       638Gi
Swap:          4.0Gi       4.0Gi       0.0Ki
```

## Storage
```
<shared-filesystem>   48T   11T zfs
```

## OS and kernel
```
PRETTY_NAME="Red Hat Enterprise Linux 9.7 (Plow)"
5.14.0-611.27.1.el9_7.x86_64
```

## Runtime
```
python: Python 3.12.15
vllm 0.19.1
torch 2.10.0+cu129 cuda 12.9
```

## Slurm partitions (limits only)
```
PARTITION TIMELIMIT NODES GRES
main* 2:00:00 1 gpu:idx0:1,gpu:idx1:1,gpu:idx2:1,gpu:idx3:1,gpu:idx4:1,gpu:idx5:1,gpu:idx6:1,gpu:idx7:1
exceptions 6:00:00 1 gpu:idx0:1,gpu:idx1:1,gpu:idx2:1,gpu:idx3:1,gpu:idx4:1,gpu:idx5:1,gpu:idx6:1,gpu:idx7:1
admin infinite 1 gpu:idx0:1,gpu:idx1:1,gpu:idx2:1,gpu:idx3:1,gpu:idx4:1,gpu:idx5:1,gpu:idx6:1,gpu:idx7:1
priority 30:00 1 gpu:idx0:1,gpu:idx1:1,gpu:idx2:1,gpu:idx3:1,gpu:idx4:1,gpu:idx5:1,gpu:idx6:1,gpu:idx7:1
preemptible infinite 1 gpu:idx0:1,gpu:idx1:1,gpu:idx2:1,gpu:idx3:1,gpu:idx4:1,gpu:idx5:1,gpu:idx6:1,gpu:idx7:1
```

## Hugging Face hub in use
```
HF_HUB_CACHE set: yes
HF_HUB_OFFLINE=unset
```
