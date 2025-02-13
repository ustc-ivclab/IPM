<div align="center">

<h1>Partition Map-Based Fast Block Partitioning for VVC Inter Coding</h1>

<div>
    <a href='https://zhexinliang.github.io/' target='_blank'>Xinmin Feng</a>&emsp;
    <a href='https://scholar.google.com/citations?user=PiyMuF4AAAAJ&hl=en&oi=ao' target='_blank'>Zhuoyuan Li</a>&emsp;
    <a href='https://faculty.ustc.edu.cn/lil1/en/index.htm' target='_blank'>Li Li</a>&emsp;
    <a href='https://faculty.ustc.edu.cn/dongeliu/en/index.htm' target='_blank'>Dong Liu</a>&emsp;
    <a href='https://scholar.google.com/citations?user=5bInRDEAAAAJ&hl=en&oi=ao' target='_blank'>Feng Wu</a>
</div>
<div>
    Intelligent Visual Lab, University of Science and Technology of China &emsp; 
</div>

<div>
   <strong>Under Peer Review</strong>
</div>
<div>
    <h4 align="center">
    </h4>
</div>

---

</div>


- [Training Dataset](#training-dataset)
- [Installation of Dependencies](#installation-of-dependencies)
- [Modified VTM Encoder](#modified-vtm-encoder)
- [Network Inference and Post-processing](#network-inference-and-post-processing)
- [Performance](#performance)

<!-- - [Quickstart](#quickstart) -->

##  :open_book: Training Dataset

The training dataset is available at [Baidu Cloud](https://pan.baidu.com/s/1ZMPZqOcQS_gri_pzSq2vGA?pwd=tmxn). We used 668 4K sequences with 32 frames from the [BVI-DVC](https://fan-aaron-zhang.github.io/BVI-DVC/) dataset, Tencent Video Dataset / [TVD](https://multimedia.tencent.com/resources/tvd/), and [UVG](https://github.com/ultravideo/UVG-4K-Dataset/) dataset. These sequences were cropped or downsampled to create datasets with four different resolutions: 3840x2160, 1920x1080, 960x544, and 480x272. We organized the training dataset using HDF5 format, which includes the following files:

- `train_seqs.h5`: Luma components of the original sequences.
- `train_qp22.h5`: Training dataset label for basic QP22.
- `train_qp27.h5`: Training dataset label for basic QP27.
- `train_qp32.h5`: Training dataset label for basic QP32.
- `train_qp37.h5`: Training dataset label for basic QP37.

To further support subsequent research, we also provide the code for generating the training dataset, which includes:

1. Modified VTM source code `codec/print_encoder` and the executable file `codec/exe/print_encoder.exe` for extracting block partitioning statistics from YUV sequences. Code `dataset_preparation.py` for extracting the statistics into `DepthSaving/` with multiple threads.
3. Code `depth2dataset.py` for converting the  statistics into partition maps.


## :wrench: Installation of Dependencies

In order to explore this project, it is needed to first install the libraries used in it.

The base image is `pytorch:2.0.0-cuda11.7-cudnn8-runtime`. To install the dependencies, use the following command:

```bash
pip install einops matplotlib tensorboard timm ipykernel h5py thop openpyxl palettable -i https://mirrors.aliyun.com/pypi/simple/
```


## :gear: Modified VTM Encoder

We provide the source code for the VTM 10.0 and 23.0 encoder with integrated fast algorithms in the folder `codec/source_code/inter_fast`, and the corresponding executable files for different acceleration levels in `codec/exe`. Specifically, `inter_fast` corresponds to acceleration for B-frames only, while `inter_intra_fast` uses the proposed method to accelerate B-frames and uses the method from [1] to accelerate I-frames.

To implement different acceleration levels, you can modify the parameters in `TypeDef.h`. For example, for the acceleration level $L_1(0.2,0.9)$, and the configuration for accelerating I-frames is as follows:

```C++
// Fast block partitioning for VVC inter coding
#define   INTER_PARTITION_MAP_ACCELERATION_FXM      1  // Accelerating B-frames, True: 1, False: 0
#define   Acceleration_Config_fxm                   1  // Acceleration level, options: 0, 1, 2, 3
#define   boundary_handling_fxm                     1  // Boundary handling based on granularity
#define   Mtt_mask_fxm                              1  // If config=0 and mtt_mask=1, the uncovered parts of the mtt mask are decided by RDO. If config>=1 and mtt_mask=1, the uncovered parts are decided by the network
#define   mtt_mask_thd                              20 // MTT mask threshold, true threshold = threshold / 100
#define   mtt_rdo_thd                               90 // MTT RDO threshold. Blocks with values below this will skip MTT fast partitioning

// Fast block partitioning for VVC intra coding
#define   INTRA_PARTITION_MAP_ACCELERATION_FAL      1  // Accelerating I-frames, True: 1, False: 0
#if INTRA_PARTITION_MAP_ACCELERATION_FAL
#define   Acceleration_Config_fal_intra             1  // 4 configuration options (0, 1, 2, 3)
#endif
```

The acceleration configurations for different acceleration levels are as follows, , corresponding to `inter_fast/VTM10_L0_0_100.exe`, `inter_fast/VTM10_L0_20_100.exe`, and `inter_fast/VTM10_L1_20_90.exe`.

| Macro                            | $L_0(0,1)$ | $L_0(0.2,1)$ | $L_0(0.2,0.9)$ |
|-----------------------------------|------------|--------------|----------------|
| `INTER_PARTITION_MAP_ACCELERATION_FXM` | 1          | 1            | 1              |
| `Acceleration_Config_fxm`           | 0          | 0            | 1              |
| `boundary_handling_fxm`             | 1          | 1            | 1              |
| `Mtt_mask_fxm`                      | 0          | 1            | 1              |
| `mtt_mask_thd`                      | 0          | 20           | 20             |
| `mtt_rdo_thd`                       | 100        | 100          | 90             |

In addition, we also provide a combination of the proposed method and previous work [1], where the former accelerates B-frames and the latter accelerates I-frames. This corresponds to `inter_intra_fast/VTM10_L0i_0_100.exe`, `inter_intra_fast/VTM10_L0i_20_100.exe`, and `inter_intra_fast/VTM10_L1i_20_90.exe`.

You can use the following command to run the encoder and accelerate B-frames, where `el` represents the path to the partition flags of B-frames, and `ip` represents the intra period.

```bash
VTM10_L1_20_90.exe -el D:\\PartitionMat\\f65_intra\\PartitionMat\\f65_gop16\\BasketballDrive_1920x1080_50_Luma_QP22_PartitionMat.txt -c D:\\VTM\\VVCSoftware_VTM-VTM-10.0\\cfg\\encoder_randomaccess_vtm.cfg -c D:\\VTM\\VVCSoftware_VTM-VTM-10.0\\cfg\\per-sequence\\BasketballDrive.cfg -i D:\\VVC_test\\BasketballDrive_1920x1080_50.yuv  -q 22 -f 65 -ip 48 -b res_L0.bin
```
Alternatively, you can use the following command to accelerate both B-frames and I-frames. In this case, `ac` and `al` represent the paths to the partition flags for the I-frame Luma components and chroma components, respectively.

```bash
VTM10_L1_20_90.exe -el D:\\PartitionMat\\f65_intra\\PartitionMat\\f65_gop16\\BasketballDrive_1920x1080_50_Luma_QP22_PartitionMat.txt -ac D:\\PartitionMat\\f65_intra\\PartitionMat\\f65_gop16\\BasketballDrive_1920x1080_50_Luma_QP22_PartitionMat.txt -al D:\\PartitionMat\\f65_intra\\PartitionMat\\f65_intra\\RitualDance_1920x1080_60fps_10bit_420_Luma_QP22_PartitionMat_intra.txt  -c D:\\VTM\\VVCSoftware_VTM-VTM-10.0\\VVCSoftware_VTM-VTM-10.0-fast\\cfg\\encoder_randomaccess_vtm.cfg -c D:\\VTM\\VVCSoftware_VTM-VTM-10.0\\VVCSoftware_VTM-VTM-10.0-fast\\cfg\\per-sequence\\RitualDance.cfg -i E:\\VVC_test\\RitualDance_1920x1080_60fps_10bit_420.yuv  -q 22 -f 65 -ip 64 -b res_L0.bin
```

We provide partition flags for 22 VVC CTC sequences in GOP16 and GOP32 on [Baidu Cloud](https://pan.baidu.com/s/1STnEpLJmxiVV8AoA3hptRA?pwd=dddr). You can download these files and replace the `el`, `ac`, and `al` paths above to reproduce our results without invoking model.





## :brain: Network Inference and Post-Processing

To obtain the partition flags for accelerating the modified VTM encoder, we process the raw sequence using the proposed neural network, and apply the post-processing algorithm to generate the text file for modified VTM encoder.


## :running_woman: TODO 

- [ ] Update the code for neural network inference.
- [ ] Update the code for training models.

## :chart_with_upwards_trend: Performance

![Comparison with Related Methods.](./Figure/RDTA.png)


## :bust_in_silhouette: Ackownledgement

We acknowledge the support of GPU and HPC cluster built by MCC Lab of Information Science and Technology Institution, USTC.


## References

1. [Partition Map Prediction for Fast Block Partitioning in VVC Intra-frame Coding](https://github.com/AolinFeng/PMP-VVC-TIP2023)