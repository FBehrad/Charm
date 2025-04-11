# 💫 Charm: The Missing Piece in ViT fine-tuning for Image Aesthetic Assessment

> [**Accepted at CVPR 2025**](https://cvpr.thecvf.com/virtual/2025/poster/34423)<br><br>
> Arxiv: https://arxiv.org/abs/2504.02522 
<div align="center">
<a href="https://github.com/FBehrad/Charm">
    <img src="https://github.com/FBehrad/Charm/blob/main/Figures/MainFigure.jpg?raw=true" alt="Overall framework" width="400"/>
</a>
</div>


We introduce **Charm** , a novel tokenization approach that preserves **C**omposition, **H**igh-resolution,
**A**spect **R**atio, and **M**ulti-scale information simultaneously. By preserving critical aesthetic information, <em> Charm </em> achieves significant performance improvement across different image aesthetic assessment datasets.


## Introduction

The capacity of Vision transformers (ViTs) to handle variable-sized inputs is often constrained by computational
complexity and batch processing limitations. Consequently, ViTs are typically trained on small, fixed-size images obtained through downscaling or cropping. While reducing
computational burden, these methods result in significant information loss, negatively affecting tasks like image aesthetic assessment. We introduce <em> Charm </em> , a novel tokenization approach that preserves Composition, High-resolution,
Aspect Ratio, and Multi-scale information simultaneously. <em> Charm </em>  prioritizes high-resolution details in specific regions
while downscaling others, enabling shorter fixed-size input sequences for ViTs while incorporating essential information. <em> Charm </em>  is designed to be compatible with pre-trained
ViTs and their learned positional embeddings. By providing multiscale input and introducing variety to input tokens,
<em> Charm </em>  improves ViT performance and generalizability for image aesthetic assessment. We avoid cropping or changing
the aspect ratio to further preserve information. Extensive experiments demonstrate significant performance improvements on various image aesthetic and quality assessment
datasets (up to 8.1 %) using a lightweight ViT backbone. 


## Performance improvement across various datasets
Charm not only improves the performance of image aesthetic/quality assessment models but also accelerates their convergence.
<div align="center">
<img src=Figures/table1.jpg width="400" />
</div>
Performance improvement across different IAA and
IQA datasets by replacing the standard tokenization with Charm.
Dinov2-small is employed for all experiments except for *, which
shows the ViT-small backbone.
<div align="center">
<img src=Figures/convergence.jpg width="400" />
</div>
The epoch at which each model achieves its highest
validation performance across different datasets. Charm generally
leads to faster convergence. 


## Standard Installation

Clone the repository locally and install with:

```setup
git clone https://github.com/FBehrad/Charm.git
pip install -r requirements.txt
```

## Model checkpoints
Pretrained models are available at [**HuggingFace**](https://huggingface.co/FatemehBehrad/Charm) 🤗.


## Quick inference

:globe_with_meridians:	Google Colab: [Colab](charm.ipynb)

:desktop_computer: On your system: Please read [this page](ReadMe_Inference.md).


## Training 
Configure the values in the [config file](config.yaml) and execute [train.py](train.py) to start training.

## Citation
If you find this repo useful in your research, please star ⭐ this repository and consider citing 📝:

```bibtex
@Inproceedings{Behrad2025Charm,
  author    = {Fatemeh Behrad and Tinne Tuytelaars and Johan Wagemans},
  title     = {Charm: The Missing Piece in ViT fine-tuning for Image Aesthetic Assessment},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2025},
}
```

## Acknowledgments
Funded by the European Union (ERC AdG, GRAPPA, 101053925, awarded to Johan Wagemans) and 
the Research Foundation-Flanders (FWO, 1159925N, awarded to Fatemeh Behrad).
