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
Autoři natrénovali svůj model na datasetu, který vytvořili kombinací existujících a nově vytvořeného datasetu o třech třídách a následně použili transfer learning pro přetrénování na rozpoznání 4 tříd. Komplexní preprocessing, různé optimalizační metody, 4 případové studie k porovnání modelů. Scéna je klasifikována za cca 2.2 s. Model je dostupný jako open-source python knihovna dostupný přes PYPI. Autoři upozorňují na možný problém v použitých datasetech: 
>While considerable effort has been made by the authors of the training and validation datasets to produce accurate labels (Aybar et al., 2022), errors in these datasets are likely to still exist. Errors which are persistent among both training and validation datasets are likely to produce models which appear better in validation than in independent testing. While the CloudS2Mask model backbones are relatively small in parameter count (Radosavovic et al., 2020) it is possible that even smaller models could generalise better as they are less capable of overfitting to the specific errors in the training data as shown by Northcutt et al. (2021).

## Poznámka
Od vydání článku došlo k vývoji modelu, který je nově vyvíjen pod názvem OmniCloudMask (viz jiný článek)
