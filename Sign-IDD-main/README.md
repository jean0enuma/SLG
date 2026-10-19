# Sign-IDD
Source code for "Sign-IDD: Iconicity Disentangled Diffusion for Sign Language Production" (Shengeng Tang, Jiayi He, Dan Guo, Yanyan Wei, Feng Li, Richang Hong - AAAI 2025)

# Usage
Install required packages using the requirements.txt file.
```text
pip install -r requirements.txt
```
# Data
PHOENIX14T

We use the same data as the [Progressive Transformer](https://github.com/BenSaunders27/ProgressiveTransformersSLP/tree/master/Data/tmp).

USTC-CSL

Dataset will be released soon.


# Training
```text
python __main__.py train ./Configs/Sign-IDD.yaml
```

# Inference
```text
python __main__.py test ./Configs/Sign-IDD.yaml
```

# SLT Model
We use the back translation [SLT](https://github.com/NaVi-start/Sign-IDD-SLT.git).
# Reference
If you use this code in your research, please cite the following [papers](https://arxiv.org/abs/2412.13609):

```bibtex
@inproceedings{tang2025sign,
  title={Sign-IDD: Iconicity Disentangled Diffusion for Sign Language Production},
  author={Tang, Shengeng and He, Jiayi and Guo, Dan and Wei, Yanyan and Li, Feng and Hong, Richang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={7},
  pages={7266--7274},
  year={2025}
}

@article{tang2024discrete,
  title={Discrete to Continuous: Generating Smooth Transition Poses from Sign Language Observation},
  author={Tang, Shengeng and He, Jiayi and Cheng, Lechao and Wu, Jingjing and Guo, Dan and Hong, Richang},
  journal={arXiv preprint arXiv:2411.16810},
  year={2024}
}

@article{tang2024GCDM,
  title={Gloss-Driven Conditional Diffusion Models for Sign Language Production},
  author={Tang, Shengeng and Xue, Feng and Wu, Jingjing and Wang, Shuo and Hong, Richang},
  journal={ACM Transactions on Multimedia Computing, Communications, and Applications},
  issn = {1551-6857},
  year={2024},
}
```


# Acknowledge
This work was supported by the National Natural Science Foundation of China (Grants No. U23B2031, 61932009, U20A20183, 62272144, 62302141, 62331003), the Anhui Provincial Natural Science Foundation, China (Grant No. 2408085QF191), the Major Project of Anhui Province (Grant No. 202423k09020001), and the Fundamental Research Funds for the Central Universities (Grants No. JZ2024HGTA0178, JZ2024HGTB0255).
