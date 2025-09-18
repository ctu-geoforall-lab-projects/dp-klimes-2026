# Training sensor-agnostic deep learning models for remote sensing: Achieving state-of-the-art cloud and cloud shadow identification with OmniCloudMask (2025)

**Article:** https://doi.org/10.1016/j.rse.2025.114694  
**Source code:** https://github.com/DPIRD-DMA/OmniCloudMask  
**Podcast:** https://www.youtube.com/watch?v=3spUZ2xR8Fo  

## Description
The authors build on their previous work (CloudS2Mask) and present how to train a model capable of accurate detection on data from multiple different sensors (despite being trained on data from only one). The authors achieved the best results among all compared methods.  
>OCM demonstrates robust state-of-the-art performance across various satellite platforms when classifying clear, cloud, and shadow classes, with balanced overall accuracy values across: Landsat (91.5 % clear, 91.5 % cloud, and 75.2 % shadow); Sentinel-2 (92.2 % clear, 91.2 % cloud, and 80.5 % shadow); and PlanetScope (96.9 % clear, 98.8 % cloud, and 97.4 % shadow).  
OCM achieves this accuracy while only being trained on a single Sentinel-2 dataset, employing spectral normalisation and mixed resolution training to address the spectral and spatial differences between satellite platforms. This approach allows the model to effectively handle imagery from different sensors within the 10 m to 50 m resolution range, as well as higher resolution imagery resampled to 10 m. The OCM library is available as an open-source Python package on PyPI.

---

# CloudS2Mask: A novel deep learning approach for improved cloud and cloud shadow masking in Sentinel-2 imagery (2024)

**Article:** https://doi.org/10.1016/j.rse.2024.114122  
**Source code:** https://github.com/DPIRD-DMA/CloudS2Mask  

**Installation**  
```bash
pip install clouds2mask
```

## Basic info
- **Architecture:** U-Net + RegNetY_006 backbone  
- **Datasets:**
  - 3 classes (clear/cloud/shadow) – CloudSEN12, KappaSet, custom CloudS2Mask dataset  
  - 4 classes (thick/thin cloud) – (finetuning on CloudSEN12)  
  - validation – CloudSEN12 validation  
  - test – CloudSEN12 testing, PixBox  
- **Model:**
  - 2 variants (10/20 m resolution) + additional modifications (Adaptive Test Time Augmentation, Model ensembling)  
- **Performance:**
  - compared with: UNetMobV2, Fmask, s2cloudless  
  - Metrics slightly better than UNetMobV2 (presented in CloudSEN12 as the then state-of-the-art) except for precision, where UNetMobV2 performs slightly better  

## Description
The authors present a new model for cloud detection in Sentinel-2 imagery, trained on a dataset created by combining existing datasets with a newly created three-class dataset, and then applied transfer learning for retraining to recognize four classes. Extensive preprocessing, various optimization methods, and four case studies for model comparison. A scene is classified in about 2.2 s. The model is available as an open-source Python package on PyPI. The authors note a potential issue with the datasets:  
>While considerable effort has been made by the authors of the training and validation datasets to produce accurate labels (Aybar et al., 2022), errors in these datasets are likely to still exist. Errors which are persistent among both training and validation datasets are likely to produce models which appear better in validation than in independent testing. While the CloudS2Mask model backbones are relatively small in parameter count (Radosavovic et al., 2020) it is possible that even smaller models could generalise better as they are less capable of overfitting to the specific errors in the training data as shown by Northcutt et al. (2021).

## Note
Since the article’s publication, the model has been further developed under the new name OmniCloudMask (see the other article).

---

# Convolutional Neural Network‑Driven Improvements in Global Cloud Detection for Landsat 8 and Transfer Learning on Sentinel‑2 Imagery (2023)

**Article:** https://www.mdpi.com/2072-4292/15/6/1706  

## Basic info
- **Architecture:** 4 CNN models – UNmask (U-Net), FCNmask (Fully Convolutional), SNmask (SegNet-based), DLmask (DeepLabv3+)  
- **Datasets:**
  - USGS Landsat 8 Biome Cloud-Cover Assessment Dataset (96 scenes from various biomes) – training + validation  
  - Sentinel‑2 imagery (used for transfer learning, without re-annotation)  
- **Model:**
  - trained on Landsat 8  
- **Performance:**
  - best results achieved by **UNmask** (U-Net):
    - OA 94.9 %, F1-score 94.1 %, CAD ±0.35 % (Landsat 8)  
  - transfer learning to Sentinel‑2:
    - OA 90.1 %, CAD (cloud amount difference) 5.85 %  

## Description
The authors tested 4 CNN models for cloud detection on Landsat 8. The best model (UNmask) was then applied to Sentinel‑2 imagery, which first had to be spectrally adjusted to match Landsat imagery (spectral response function). No fine-tuning was used, the model was directly applied to S2 data. The authors mention the possibility of further improvements:  
>Although the deep CNN model has significant advantages, some improvement methods can be considered for Landsat cloud detection. The digital elevation model and global surface coverage map can be included as the additional bands by layer stack, which can design appropriate thresholds for different surface types and altitudes to improve the performance of the model.  

---

# Using Convolutional Neural Networks for Cloud Detection on VENμS Images over Multiple Land-Cover Types (2022)

**Article:** https://doi.org/10.3390/rs14205210  
**Source code:** https://github.com/pesekon2/cloud-detection-venus  

## Basic info
- **Architecture:** UNet, SegNet, DeepLabv3  
- **Datasets:**
  - VENuS cloud mask training dataset  

## Description
The authors compare the performance of three CNNs for cloud detection on VENuS satellite imagery over Israel. All CNNs outperformed the existing cloud masking algorithm used for this satellite (MAJA algorithm). UNet achieved the best results. Various settings and input data were also tested:  
>Moreover, it investigates their performance over different settings, including diverse numbers of classes (binary vs. multiclass), diverse band and index variations (RGB vs. RGB + NIR vs. full-band), and the effect of overfitting avoidance strategies (dropout, data augmentation).  

---

# A hybrid generative adversarial network for weakly-supervised cloud detection in multispectral images (2022)

**Article:** https://doi.org/10.1016/j.rse.2022.113197  
**Source code:** https://github.com/Neooolee/GANCDM  

## Basic info
- **Architecture:** GAN-CDM  
- **Datasets:**
  - WHUL8-CDb – training (https://doi.org/10.5281/zenodo.6420027)  
  - L8 Biome dataset, S2 Cloud Mask Catalogue dataset – test  
- **Model:**
  - GAN combined with CDM (cloud distortion model) – cloud masks for training were only block-based instead of pixel-based, saving time  
- **Performance:**
  - Compared with supervised models. OA was better than other models, while other metrics were comparable.  

## Description
The authors present a new model for cloud detection, consisting of a GAN and CDM (cloud distortion model). The main advantage is that detailed cloud masks are not needed for training, only block-level annotations. Trained on L8 and tested on both L8 and S2. No fine-tuning was used, the model was directly applied to S2 data.  

---

# Benchmarking Deep Learning Models for Cloud Detection in Landsat-8 and Sentinel-2 Images (2021)

**Article:** https://doi.org/10.3390/rs13050992  
**Source code:** https://github.com/IPL-UV/DL-L8S2-UV  

## Basic info
- **Architecture:** UNet, SegNet, DeepLabv3  
- **Datasets:**
  - L8: L8-Biome, L8-SPARCS, L8-38Clouds  
  - S2: S2-Hollstein, S2-BaetensHagolle  

## Description
The authors compare DL methods with threshold-based methods for cloud detection. DL methods performed better. They used L8 and S2, applying transfer learning from L8 to S2. Only corresponding bands were used, and S2 data were resampled. They report that TL from L8 to S2 works well, with results comparable to other methods.  

---

# Transferring deep learning models for cloud detection between Landsat-8 and Proba-V (2020)

**Article:** https://doi.org/10.1016/j.isprsjprs.2019.11.024  

## Basic info
- **Architecture:** modified U-Net  

## Description
The authors describe different TL approaches and apply them to L8 and Proba-V data.  
>Three types of experiments are conducted to demonstrate that transfer learning can work in both directions: (a) from Landsat-8 to Proba-V, where we show that models trained only with Landsat-8 data produce cloud masks 5 points more accurate than the current operational Proba-V cloud masking method, (b) from Proba-V to Landsat-8, where models that use only Proba-V data for training have an accuracy similar to the operational FMask in the publicly available Biome dataset (87.79–89.77% vs 88.48%), and (c) jointly from Proba-V and Landsat-8 to Proba-V, where we demonstrate that using jointly both data sources the accuracy increases 1–10 points when few Proba-V labeled images are available.

They emphasize the importance of proper "Domain adaptation":  
>Our results suggest that it is important to use both the spatial and the spectral adaptation in order to fully exploit TL advantages.

They mention the potential for improvement using GANs to determine transformations between datasets – explored further in the article *Cross-Sensor Adversarial Domain Adaptation of Landsat-8 and Proba-V images for Cloud Detection*.  
>Our next steps are fostered to improve the cloud detection accuracy by using the generative adversarial networks (GANs) framework (Mateo-García et al., 2019) to learn a transformation between Landsat-8 and Proba-V data.

---

# Cross-Sensor Adversarial Domain Adaptation of Landsat-8 and Proba-V images for Cloud Detection (2020)

**Article:** https://doi.org/10.48550/arXiv.2006.05923  
**Source code:** https://github.com/IPL-UV/pvl8dagans  

## Basic info
- **Architecture:** modified U-Net for detection, GAN for Domain Adaptation  
- **Model:**
  - Cycle-GAN, 2 generators, 2 discriminators (detailed in the article)  
- **Performance:**
  - better results with DA  

## Description
>Summarizing, in this paper, a Cycle-GAN architecture has been proposed to train a domain adaptation method between Proba-V and upscaled Landsat images. The proposal includes two generators, two discriminators and four different penalties. The GAN generator is used to modify the Proba-V images to better resemble the upscaled Landsat images that have also been used to train a cloud detection algorithm. Results on original Proba-V images demonstrate that when using the proposed model for the adaptation a higher cloud detection accuracy is achieved.

---

Various resources on the topic: https://github.com/satellite-image-deep-learning
