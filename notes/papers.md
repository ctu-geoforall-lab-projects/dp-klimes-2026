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
