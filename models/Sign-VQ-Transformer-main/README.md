# Sign VQ Transformer 

## Overview

This repository contains the code for the paper "A data-driven representation for sign language production". The first step creates a spatial-temporal codebook using a VQ transformer. Learning a set of small motions that can be combined to create the full sign language sequence. Then, in the second step, it is used to perform translation on a dataset, from text to a set of these codebook tokens. To evaluate the performance, back translation alongside a range of other metrics is used.

The code is structured to allow for easy training and testing of the VQ model and the translation model. It uses PyTorch Lightning for training and evaluation, and it is designed to be modular, allowing for easy extension and modification.

For the demo page and a further explanation on the code, please see the following repository [GitHub Repository](https://github.com/walsharry/VQ_SLP_Demos).

## Codebook visualization of the VQ model 

### RWTH-PHOENIX-Weather-2014**T**

<div align="center">

<table style="border-collapse: collapse;">
<tr>
  <td><img src="/codebook_examples/phix/Codebook_432.gif" width="150" height="150" /></td>
  <td><img src="/codebook_examples/phix/Codebook_97.gif" width="150" height="150" /></td>
  <td><img src="/codebook_examples/phix/Codebook_127.gif" width="150" height="150" /></td>
  <td><img src="/codebook_examples/phix/Codebook_75.gif" width="150" height="150" /></td>
  <td><img src="/codebook_examples/phix/Codebook_575.gif" width="150" height="150" /></td>
</tr>
<tr>
  <td><img src="/codebook_examples/phix/Codebook_436.gif" width="150" height="150" /></td>
  <td><img src="/codebook_examples/phix/Codebook_296.gif" width="150" height="150" /></td>
  <td><img src="/codebook_examples/phix/Codebook_714.gif" width="150" height="150" /></td>
  <td><img src="/codebook_examples/phix/Codebook_51.gif" width="150" height="150" /></td>
  <td><img src="/codebook_examples/phix/Codebook_42.gif" width="150" height="150" /></td>
</tr>
<tr>
  <td><img src="/codebook_examples/phix/Codebook_279.gif" width="150" height="150" /></td>
  <td><img src="/codebook_examples/phix/Codebook_645.gif" width="150" height="150" /></td>
  <td><img src="/codebook_examples/phix/Codebook_0.gif" width="150" height="150" /></td>
  <td><img src="/codebook_examples/phix/Codebook_898.gif" width="150" height="150" /></td>
  <td><img src="/codebook_examples/phix/Codebook_180.gif" width="150" height="150" /></td>
</tr>
<tr>
  <td><img src="/codebook_examples/phix/Codebook_664.gif" width="150" height="150" /></td>
  <td><img src="/codebook_examples/phix/Codebook_249.gif" width="150" height="150" /></td>
  <td><img src="/codebook_examples/phix/Codebook_330.gif" width="150" height="150" /></td>
  <td><img src="/codebook_examples/phix/Codebook_383.gif" width="150" height="150" /></td>
  <td><img src="/codebook_examples/phix/Codebook_368.gif" width="150" height="150" /></td>
</tr>
<tr>
  <td><img src="/codebook_examples/phix/Codebook_742.gif" width="150" height="150" /></td>
  <td><img src="/codebook_examples/phix/Codebook_689.gif" width="150" height="150" /></td>
  <td><img src="/codebook_examples/phix/Codebook_321.gif" width="150" height="150" /></td>
  <td><img src="/codebook_examples/phix/Codebook_28.gif" width="150" height="150" /></td>
  <td><img src="/codebook_examples/phix/Codebook_449.gif" width="150" height="150" /></td>
</tr>
</table>

</div>

## Installation

### Requirements

```bash
# Create and activate a new conda environment (recommended)
conda create --name signVQ python=3.8
conda activate signVQ

# Install PyTorch with CUDA support
# Please change PyTorch CUDA version to match your system!
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r ./requirements.txt
```

## Running the code

### Step 1: VQ model

**Train:** 
```bash
python __main__.py train vq ./config/codebook/codebook_config.yaml
```

**Test:** 
```bash
python __main__.py test vq ./models/vq_models/phix_codebook/config.yaml
```

### Step 2: Translation model

**Train:** 
```bash
python __main__.py train translate ./config/translation/translation_config.yaml
```

**Test:** 
```bash
python __main__.py test translate ./models/translation_models/phix_translation/config.yaml
```

### Arguments

| Argument      | Description                                               | Default |
|---------------|-----------------------------------------------------------|---------|
| `mode`        | train or test                                             | Required |
| `type`        | vq or translate                                           | Required |
| `config_path` | Directory containing the pretrained back translation model | Required |


### Data Format

Inside the provided .pt files, you will find a dictionary with the following structure:
The data files should be placed inside the directory ./data (**data_path** specified in the config file.) as three separate files named dev.pt, test.pt and train.pt.

The contents of the files should follow the following dictionary structure:

```json
{
    "01April_2010_Thursday_heute-6704":{
      "name": (string),
      "text": (string),
      "gloss": (string),
      "poses_3d" : (N x K x D),
      "speaker" : (string),
    },
     "30September_2012_Sunday_tagesschau-4038": {
      "name": (string),
      "text": (string),
      "gloss": (string),
      "poses_3d" : (N x K x D),
      "speaker" : (string),
    },
    ...
    "27October_2009_Tuesday_tagesschau-6148":   {
      "name": (string),
      "text": (string),
      "gloss": (string),
      "poses_3d" : (N x K x D),
      "speaker" : (string),
    },
}
```

### Pre-processed Data

 Pre-processed skeleton pose for the Phoenix14T dataset and corresponding back translation
 model can be found from the following competitions repository, [GitHub Repository](https://github.com/walsharry/SLRTP-Sign-Production-Evaluation/tree/main).

## Citation

When using this codebase, please cite:

```
@inproceedings{walsh2024data,
  title={A data-driven representation for sign language production},
  author={Walsh, Harry and Ravanshad, Abolfazl and Rahmani, Mariam and Bowden, Richard},
  booktitle={2024 IEEE 18th International Conference on Automatic Face and Gesture Recognition (FG)},
  pages={1--10},
  year={2024},
  organization={IEEE}
}
```

When using the pre-processed data, please cite:

```
@inproceedings{walsh2025slrtp,
title={SLRTP2025 Sign Language Production Challenge: Methodology, Results, and Future Work},
author={Walsh, Harry and Fish, Ed and Sincan, Ozge Mercanoglu and Lakhal, Mohamed Ilyes and Bowden, Richard and Fox, Neil and Cormier, Kearsy and Woll, Bencie and Wu, Kepeng and Li, Zecheng and Zhao, Weichao and Wang, Haodong and Zhou, Wengang and Li, Houqiang and Tang, Shengeng and He, Jiayi and Wang, Xu and Zhang, Ruobei and Wang, Yaxiong and Cheng, Lechao and Tasyurek, Meryem and Kiziltepe, Tugce and Keles, Hacer Yalim}, 
booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
year={2025}
}
```

When using the back translation model, please cite:

```
@inproceedings{camgoz2020sign,
author = {Necati Cihan Camgoz and Oscar Koller and Simon Hadfield and Richard Bowden},
title = {Sign Language Transformers: Joint End-to-end Sign Language Recognition and Translation},
booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
year = {2020}
}
```

## Acknowledgements
<sub>This work was supported by Intel, the SNSF project ‘SMILE II’ (CRSII5 193686), the European Union’s Horizon2020 programme (‘EASIER’ grant agreement 101016982) and the Innosuisse IICT Flagship (PFFS-21-47). This work reflects only the author's view and the Commission is not responsible for any use that may be made of the information it contains. </sub>