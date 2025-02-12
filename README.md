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

## Training Dataset

The training dataset is available at [Baidu Cloud](https://pan.baidu.com/s/1ZMPZqOcQS_gri_pzSq2vGA?pwd=tmxn). We used 668 4K sequences with 32 frames from the BVI-DVC dataset, Tencent Video Dataset, and UVG dataset. These sequences were cropped or downsampled to create datasets with four different resolutions: 3840x2160, 1920x1080, 960x544, and 480x272. We organized the training dataset using HDF5 format, which includes the following files:

- `train_seqs.h5`: Luma components of the original sequences.
- `train_qp22.h5`: Training dataset label for basic QP22.
- `train_qp27.h5`: Training dataset label for basic QP27.
- `train_qp32.h5`: Training dataset label for basic QP32.
- `train_qp37.h5`: Training dataset label for basic QP37.

To further support subsequent research, we also provide the code for generating the training dataset, which includes:

1. Modified VTM source code `codec/print_encoder` and the executable file `codec/exe/print_encoder.exe` for extracting block partitioning statistics from YUV sequences. Code `dataset_preparation.py` for extracting the statistics into `DepthSaving/` with multiple threads.
3. Code `depth2dataset.py` for converting the  statistics into partition maps.





<!-- ## :running_woman: TODO  -->

## References

1. [Partition Map Prediction for Fast Block Partitioning in VVC Intra-frame Coding](https://github.com/AolinFeng/PMP-VVC-TIP2023)

2. 