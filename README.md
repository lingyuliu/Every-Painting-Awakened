<div align="center">
<h2>👊🏻🎨🔥Every Painting Awakened: A Training-free Framework for Painting-to-Animation Generation</h2>

[Lingyu Liu](), [Yaxiong Wang](), [Li Zhu](), [Zhedong Zheng]()

<a href='https://arxiv.org/abs/2503.23736'><img src='https://img.shields.io/badge/ArXiv-2503.23736-red'></a>
<a href='https://painting-animation.github.io/animation/'><img src='https://img.shields.io/badge/Project-Page-Blue'></a>
<!-- views since 25.03 -->
</div>

<p align="center">
<img src="example/a.gif" width="900px"/>
<img src="example/b.gif" width="900px"/>
<img src="example/c.gif" width="900px"/>
</p>

***Abstract:*** We introduce a training-free framework specifically designed to bring real-world static paintings to life through image-to-video (I2V)
synthesis, addressing the persistent challenge of aligning these motions with textual guidance while preserving fidelity to the original artworks.
Existing I2V methods, primarily trained on natural video datasets, often struggle to generate dynamic outputs from static paintings.
It remains challenging to generate motion while maintaining visual consistency with real-world paintings. This results in two distinct failure modes:
either static outputs due to limited text-based motion interpretation or distorted dynamics caused by inadequate alignment with real-world artistic
styles.
We leverage the advanced text-image alignment capabilities of pre-trained image models to guide the animation process. Our approach introduces
synthetic proxy images through two key innovations:
**(1) Dual-path score distillation:** We employ a dual-path architecture to distill motion priors from both real and synthetic data, preserving
static details from the original painting while learning dynamic characteristics from synthetic frames.
**(2) Hybrid latent fusion:**  We integrate hybrid features extracted from real paintings and synthetic proxy images via spherical linear
interpolation in the latent space, ensuring smooth transitions and enhancing temporal consistency.
Experimental evaluations confirm that our approach significantly improves semantic alignment with text prompts while faithfully preserving the unique
characteristics and integrity of the original paintings. Crucially, by achieving enhanced dynamic effects without requiring any model training or
learnable parameters, our framework enables plug-and-play integration with existing I2V methods, making it an ideal solution for animating real-world
paintings.

## Requirements

* All experiments are conducted on a single NVIDIA 3090 GPU (24 GB).

## Getting Started

Our gneration is based on the official repo of [AnimateAnything](https://github.com/alibaba/animate-anything).

**Inference setup:** Prepare the environment and download the required weights in the AnimateAnything. Please download
the [pretrained model](https://cloudbook-public-production.oss-cn-shanghai.aliyuncs.com/animation/animate_anything_512_v1.02.tar) to
/animate_anything/latent .

```
├── animate_anything/
│   ├── latent
│      ├── animate_anything_512_v1.02
├── example                   
```

**Generating videos:** Please run the following command.

```bash
cd animate_anything
python eval.py --config ../config.yaml
```

In `config.yaml`, arguments for inference:

* `pretrained_model_path`: path to animate_anything_512_v1.02.
* `prompt_image`: path to the real painting that needs to be animated.
* `prompt`: text instruction.
* `proxy_image`: path to the proxy image synthesized using the I2I model,
  *i.e.*, [stable-diffusion-xl-refiner-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0)

## Citation

Please cite this paper if it helps your research:

```bibtex
@article{liu2025every,
  title={Every Painting Awakened: A Training-free Framework for Painting-to-Animation Generation},
  author={Liu, Lingyu and Wang, Yaxiong and Zhu, Li and Zheng, Zhedong},
  journal={arXiv preprint arXiv:2503.23736},
  year={2025}
}
```

## **Acknowledgement**

This repository is benefit from [AnimateAnything](https://github.com/alibaba/animate-anything). Thanks for the open-sourcing work! Any third-party
packages are
owned by their respective authors and must be used under their respective licenses. We would also like to thank to the great projects
in [ConsistI2V](https://github.com/TIGER-AI-Lab/ConsistI2V), [Cinemo](https://github.com/maxin-cn/Cinemo)
and [i4vgen](https://github.com/xiefan-guo/i4vgen).
