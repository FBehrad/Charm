# Charm: The Missing Piece in ViT fine-tuning for Image Aesthetic Assessment

We introduce **Charm** , a novel tokenization approach that preserves **C**omposition, **H**igh-resolution,
**A**spect **R**atio, and **M**ulti-scale information simultaneously. By preserving critical aesthetic information, <em> Charm </em> achieves significant performance improvement across different image aesthetic and quality assessment datasets.


### Quick Inference

* Step 1) Check our [GitHub Page](https://github.com/FBehrad/Charm/) and install the requirements. 

```setup
pip install -r requirements.txt
```
___
* Step 2) Install Charm tokenizer.
```setup
pip install Charm-tokenizer
```
___
* Step 3) Tokenization + Position embedding preparation

<div align="center">
<a href="https://github.com/FBehrad/Charm">
    <img src="https://github.com/FBehrad/Charm/blob/main/Figures/charm.gif?raw=true" alt="Charm tokenizer" width="700"/>
</a>
</div>

```python
from Charm_tokenizer.ImageProcessor import Charm_Tokenizer

img_path = r"img.png"

charm_tokenizer = Charm_Tokenizer(patch_selection='frequency', training_dataset='tad66k', without_pad_or_dropping=True)
tokens, pos_embed, mask_token = charm_tokenizer.preprocess(img_path)
```
Charm Tokenizer has the following input args:
* patch_selection (str): The method for selecting important patches
  * Options: 'saliency', 'random', 'frequency', 'gradient', 'entropy', 'original'.
* training_dataset (str): Used to set the number of ViT input tokens to match a specific training dataset from the paper.
  * Aesthetic assessment datasets: 'aadb', 'tad66k', 'para', 'baid'.
  * Quality assessment datasets: 'spaq', 'koniq10k'.
* backbone (str): The ViT backbone model (default: 'facebook/dinov2-small').
* factor (float): The downscaling factor for less important patches (default: 0.5).
* scales (int): The number of scales used for multiscale processing (default: 2).
* random_crop_size (tuple): Used for the 'original' patch selection strategy (default: (224, 224)).
* downscale_shortest_edge (int): Used for the 'original' patch selection strategy (default: 256).
* without_pad_or_dropping (bool): Whether to avoid padding or dropping patches (default: True).

The output is the preprocessed tokens, their corresponding positional embeddings, and a mask token that indicates which patches are in high resolution and which are in low resolution.
___

* Step 4) Predicting aesthetic/quality score

```python
from Charm_tokenizer.Backbone import backbone

model = backbone(training_dataset='tad66k', device='cpu')
prediction = model.predict(tokens, pos_embed, mask_token)
```

**Note:**
1. While random patch selection during training helps avoid overfitting,for consistent results during inference, fully deterministic patch selection approaches should be used. 
2. For the training code, check our [GitHub Page](https://github.com/FBehrad/Charm/).