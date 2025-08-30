# CloudS2Mask: A novel deep learning approach for improved cloud and cloud shadow masking in Sentinel-2 imagery (2024)

**Článek:** https://doi.org/10.1016/j.rse.2024.114122  

**Zdrojový kód** https://github.com/DPIRD-DMA/CloudS2Mask

 **Instalace**

```bash
pip install clouds2mask
```

## základní info
- **Architektura:** U-Net + RegNetY_006 backbone
- **Datasety:**
  - 3 třídy (clear/cloud/shadow)- CloudSEN12, KappaSet, vlastní CloudS2Mask dataset
  - 4 třídy (thick/thin cloud)- (finetuning na CloudSEN12)
  - validace - CloudSEN12 validation
  - test - CloudSEN12 testing, PixBox
- **Model:**
  - 2 varianty 10/20 m rozlišení + další úpravy (Adaptive Test Time Augmentation, Model ensembling)
- **Výkon:**
  - porovnáno s: UNetMobV2, Fmask, s2cloudless
  - Metriky mírně lepší než UNetMobV2 (prezentovaný dle autorů CloudSEN12 jako momentálně nejlepší) kromě precision kde je mírně lepší UNetMobV2

## Popis
Autoři představují nový model pro detekci mraků na snímcích S2 natrénovaný na datasetu, který vytvořili kombinací existujících a nově vytvořeného datasetu o třech třídách a následně použili transfer learning pro přetrénování na rozpoznání 4 tříd. Komplexní preprocessing, různé optimalizační metody, 4 případové studie k porovnání modelů. Scéna je klasifikována za cca 2.2 s. Model je dostupný jako open-source python knihovna dostupný přes PYPI. Autoři upozorňují na možný problém v použitých datasetech: 
>While considerable effort has been made by the authors of the training and validation datasets to produce accurate labels (Aybar et al., 2022), errors in these datasets are likely to still exist. Errors which are persistent among both training and validation datasets are likely to produce models which appear better in validation than in independent testing. While the CloudS2Mask model backbones are relatively small in parameter count (Radosavovic et al., 2020) it is possible that even smaller models could generalise better as they are less capable of overfitting to the specific errors in the training data as shown by Northcutt et al. (2021).

## Poznámka
Od vydání článku došlo k vývoji modelu, který je nově vyvíjen pod názvem OmniCloudMask (viz jiný článek)

# Training sensor-agnostic deep learning models for remote sensing: Achieving state-of-the-art cloud and cloud shadow identification with OmniCloudMask (2025)

**Článek:** https://doi.org/10.1016/j.rse.2025.114694

**Zdrojový kód** https://github.com/DPIRD-DMA/OmniCloudMask

**Podcast** https://www.youtube.com/watch?v=3spUZ2xR8Fo

## Popis
Autoři navazují na svou předchozí práci (CloudS2Mask) a představují jak trénovat model, který bude schopný přesné detekce na více různých datech. Autoři dosáhli nejlepších výsledků ze všech porovnávaných metod.
>OCM demonstrates robust state-of-the-art
performance across various satellite platforms when classifying clear, cloud, and shadow classes, with balanced
overall accuracy values across: Landsat (91.5 % clear, 91.5 % cloud, and 75.2 % shadow); Sentinel-2 (92.2 %
clear, 91.2 % cloud, and 80.5 % shadow); and PlanetScope (96.9 % clear, 98.8 % cloud, and 97.4 % shadow).
OCM achieves this accuracy while only being trained on a single Sentinel-2 dataset, employing spectral normalisation and mixed resolution training to address the spectral and spatial differences between satellite platforms. This approach allows the model to effectively handle imagery from different sensors within the 10 m to
50 m resolution range, as well as higher resolution imagery that has been resampled to 10 m. The OCM library is
available as an open source Python package on PyPI.

# Convolutional Neural Network‑Driven Improvements in Global Cloud Detection for Landsat 8 and Transfer Learning on Sentinel‑2 Imagery (2023)

**Článek:** https://www.mdpi.com/2072-4292/15/6/1706  

## základní info
- **Architektura:** 4 CNN modely – UNmask (U-Net), FCNmask (Fully Convolutional), SNmask (SegNet-based), DLmask (DeepLabv3+)
- **Datasety:**
  - USGS Landsat 8 Biome Cloud-Cover Assessment Dataset (96 scén z různých biomů) - trénink + validace
  - Sentinel‑2 imagery (použito pro transfer learning, bez reannotace)
- **Model:**
  - trénink na Landsat 8
- **Výkon:**
  - nejlepší výsledky dosáhl model **UNmask** (U-Net):
    - OA 94,9 %, F1-score 94,1 %, CAD ±0,35 % (Landsat 8)
  - transfer learning na Sentinel‑2:
    - OA 90,1 %, CAD (cloud amount difference) 5,85 %

## Popis
Autoři testovali 4 CNN modely pro detekci mraků na Landsat 8. Nejlepší model (UNmask) byl dále aplikován na Sentinel‑2 snímky, které nejprve bylo nutné spektrálně přiblížit Landsat snímkům (spectral response function). Nebyl použit fine-tuning, model byl rovnou aplikován na S2 data. Autoři zmiňují možnost dalšího zlepšení 
>Although the deep CNN model has significant advantages, some improvement methods can be considered for Landsat cloud detection. The digital elevation model and global
surface coverage map can be included as the additional bands by layer stack, which can
design appropriate thresholds for different surface types and altitudes to improve the
performance of the model. 


# A hybrid generative adversarial network for weakly-supervised cloud detection in multispectral images (2022)

**Článek:** https://doi.org/10.1016/j.rse.2022.113197

**Zdrojový kód** https://github.com/Neooolee/GANCDM


## základní info
- **Architektura:** GAN-CDM
- **Datasety:**
  WHUL8-CDb - trénink (https://doi.org/10.5281/zenodo.6420027)
  L8 Biome dataset, S2 Cloud Mask Catalogue dataset - test
- **Model:**
  - GAN v kombinaci s CDM (cloud distortion model) - masky mraků pro trénink řešeny jen blokově a ne pixelově - úspora času
- **Výkon:**
  - Porovnáno se supervised modely. OA je lepší než u ostatních modelů, zbylé metriky spíše srovnatelné.

## Popis
Autoři představují nový model pro detekci mraků, který se skládá z GAN a CDM (cloud distortion model). Hlavní výhodou je, že pro trénink nejsou potřeba detailní masky mraků, ale jen blokově označené mraky. Trénováno na L8 a testováno na L8 i S2. Nebyl použit fine-tuning, model byl rovnou aplikován na S2 data.

# Transferring deep learning models for cloud detection between Landsat-8 and Proba-V (2020)

**Článek:** https://doi.org/10.1016/j.isprsjprs.2019.11.024


## základní info
- **Architektura:** upravená U-Net


## Popis
Autoři popisují různé TL přístupy a aplikují je na data z L8 a Proba V. 
>Three types of experiments are conducted to demonstrate that transfer learning can work in both directions: (a) from Landsat-8 to Proba-V, where we show that models trained only with Landsat-8 data produce cloud masks 5 points more accurate than the current operational Proba-V cloud masking method, (b) from Proba-V to Landsat-8, where models that use only Proba-V data for training have an accuracy similar to the operational FMask in the publicly available Biome dataset (87.79–89.77% vs 88.48%), and (c) jointly from Proba-V and Landsat-8 to Proba-V, where we demonstrate that using jointly both data sources the accuracy increases 1–10 points when few Proba-V labeled images are available.

Zdůrazňují důležitost správně provést "Domain adaptation":
>Our results suggest that it is important to use both the spatial and the spectral adaptation in order to fully exploit TL advantages.

Možnost vylepšení při použití GAN právě na určení transformace mezi daty.
- tomu se věnují v rámci článku Cross-Sensor Adversarial Domain Adaptation of Landsat-8 and Proba-V images for Cloud Detection (viz níže)
>Our next steps are fostered to improve the cloud detection accuracy by using the generative adversarial networks (GANs) framework (Mateo-García et al., 2019) to learn a transformation between Landsat-8 and Proba-V data.

# Benchmarking Deep Learning Models for Cloud Detection in Landsat-8 and Sentinel-2 Images (2021)

**Článek:** https://doi.org/10.3390/rs13050992

**Zdrojový kód** https://github.com/IPL-UV/DL-L8S2-UV


## základní info
- **Architektura:** UNet, SegNet, DeepLabv3
- **Datasety:**
  - L8: L8-Biome, L8-SPARCS, L8-38Clouds
  - S2: S2-Hollstein, S2-BaetensHagolle
   

## Popis
Autoři srovnávají DL metody a threshold-based metody pro detekci mraků. DL metody měly lepší výsledky. Využili L8 a S2, transfer learning z L8 na S2. Použili jen odpovídající si pásma a převzorkovali S2. Uvádí, že TL z L8 na S2 funguje dobře a výsledky jsou srovnatelné s ostatními metodami.

# Cross-Sensor Adversarial Domain Adaptation of Landsat-8 and Proba-V images for Cloud Detection (2020)

**Článek:** https://doi.org/10.48550/arXiv.2006.05923

**Zdrojový kód** https://github.com/IPL-UV/pvl8dagans


## základní info
- **Architektura:** upravená UNet na detekci, GAN na Domain Adaptation
- **Model:**
  - Cycle-GAN, 2 generátory, 2 diskriminátory (popsáno podrobně v článku)
- **Výkon:**
  - lepší výsledky s DA

## Popis
>Summarizing, in this paper, a Cycle-GAN architecture
has been proposed to train a domain adaptation method
between Proba-V and upscaled Landsat images. The proposal includes two generators, two discriminators and four
different penalties. The GAN generator is used to modify the Proba-V images to better resemble the upscaled
Landsat images that have also been used to train a cloud
detection algorithm. Results on original Proba-V images
demonstrate that when using the proposed model for the
adaptation a higher cloud detection accuracy is achieved.



