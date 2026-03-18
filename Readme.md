# Video World Models with Long-term Spatial Memory



[**Project page**](https://spmem.github.io/) | [**Paper**](https://arxiv.org/abs/2506.05284) | [**Data**](https://huggingface.co/datasets/ysmikey/spmem_megadata)  



[Tong Wu*](https://wutong16.github.io/), 
[Shuai Yang*](https://ys-imtech.github.io/), 
[Ryan Po](https://ryanpo.com/), 
[Yinghao Xu](https://justimyhxu.github.io), 
[Ziwei Liu](https://liuziwei7.github.io/), 
[Dahua Lin](http://dahua.me/),
[Gordon Wetzstein](https://stanford.edu/~gordonwz/)

<p style="font-size: 0.6em; margin-top: -1em">* Equal Contribution</p></p>



<p align="center">
<a href="https://arxiv.org/abs/2506.05284"><img src="https://img.shields.io/badge/arXiv-Paper-<color>"></a>
<a href="https://spmem.github.io/"><img src="https://img.shields.io/badge/Project-Website-red"></a>
<a href="ttps://huggingface.co/datasets/ysmikey/spmem_megadata"><img src="https://img.shields.io/badge/Data-spmem_megadata-ffcc00"></a>
<a href="" target='_blank'>
<img src="https://visitor-badge.laobi.icu/badge?page_id=spmem.spmem" />
</a>
</p>

## Install Environment:
```
conda create -n spmem python=3.10 -y
conda activate spmem

pip install -r requirements.txt
```

- Install PyTorch3D:
```
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```
- Depth-Anything-3 (submodule):
```
cd Depth-Anything-3
pip install -e .
```


## Dataset Preprocess:

- We processed web videos (from Miradata) into ~80K video clips, and annotated the original videos with **MegaSAM** (images, depth, and camera poses).
- The resulting dataset is available at [**`ysmikey/spmem_megadata`**](https://huggingface.co/datasets/ysmikey/spmem_megadata).
- To further convert our data into the TSDF (dynamic/static separation) training format similar to `datasets/train_data_example`, please refer to `tsdf/data_process.sh`.




## Inference:
### Download Weights:
Download required weights from [[Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)] and [[spmem_ckpt](https://huggingface.co/ysmikey/spmem_ckpt)].

- **Qwen2.5-VL-7B-Instruct** → `ckpt/Qwen2.5-VL-7B-Instruct/`
- **spmem_ckpt** → `ckpt/spmem_ckpt/`


### Quick start:
Run the example:
```
bash run_demo.sh
```
### Streaming Control:
Run streaming control demo:
```
bash run_stream.sh
```




## Training:
We provide an example training script that uses the example training data format.

- **Script (8x GPU)**: `bash train_example.sh`
- **Example data**: `datasets/train_data_example`
- **Example config**: `datasets/train_data_example_config`

Run:
```
bash train_example.sh
```

## Citation
If you find our work helpful for your research, please consider giving a star ⭐ and citation 📝

```bibtex
@article{wu2025video,
  title={Video world models with long-term spatial memory},
  author={Wu, Tong and Yang, Shuai and Po, Ryan and Xu, Yinghao and Liu, Ziwei and Lin, Dahua and Wetzstein, Gordon},
  journal={arXiv preprint arXiv:2506.05284},
  year={2025}
}
```
