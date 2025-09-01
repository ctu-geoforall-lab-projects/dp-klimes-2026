mám to zde v češtině (je to pro mě rychlejší)
# Poznámky
- Primárně bych rád během konzultace odprezentoval, co jsem již v problematice pochopil, jak chápu první kroky atd. (pokusím se rozepsat počáteční kroky, které jste připravil pro tento projekt)

1. Prvotní rešerše 
    - Snažil jsem se najít co nejnovější články zabývající se detekcí mraků - v tom nejnovějším například při tréninku využívají dynamickou Z-score normalizaci a trénují na datech v původním rozlišení a i převzorkované na horší rozlišení, aby generalizovali model i na jiné senzory.
    - Dále jsem hledal články využívající transfer learning a zároveň zabývající se detekcí mraků, ale většina z nich využívala jen velmi málo technik TL. Zaujal mě článek, kde byla využita domain adaptation (viz otázka 1).
    - Z datasetů S2 vypadá nejkomplexněji CloudSEN12/CloudSEN12+, ale našel jsem i další. Dataset na VENuS předpokládám k použití ten Váš (ani jsem jiný na detekci mraků nenašel)
2. Stažení VENuS datasetu
3. Trénink U-Net na VENuS
   - otázka 7
4. Spustit U-Net na S2
5. fine-tuning, layer freezing U-Net z VENuS a spustit na S2 (+ domain adaptation? viz otázka 1)
    - domain adaptation - výběr stejných pásem, převzorkování..
6. Pre-training na jiném datasetu (MS COCO, Pascal, ImageNet, ....)
    - otázka 3
7. Porovnání transfer learning a training from scratch (see [here](https://proceedings.neurips.cc/paper/2020/file/27e9661e033a73a6ad8cefcde965c54d-Paper.pdf))
8. Deep learning vs jiné algoritmy

# Otázky na konzultaci
1. Zaujala mě Domain adaptation jako možnost/nutnost pro dobré přenesení modelu mezi různými senzory. Rád bych se pobavil o tom, zda a do jaké míry ji provádět. Zda použít výběr stejných pásem, převzorkování, srovnání histogramů, multivariate matching, graph matching či v extrému GAN (viz článek z roku 2020 - Cross-Sensor Adversarial Domain Adaptation..)
    - Napadá mi k tomu i možnost srovnávat výkon modelu 1 (trénovaného čistě na VENuS) s modelem 2 (trénovaného na VENuS + fine-tune na S2) s modelem 3 (použít určitý stupeň domain adaptation + fine-tune na S2) atd včetně layer freezing a vlastně se vším, co jste psal v bodě 4 v basic outline of project
2. Nestálo by za to podívat se i na srovnání CNN a transformerů (nevím zda se překládá čistě takto :D)? Moc toho o nich nevím, ale našel jsem nějaké články, kde je také k detekci mraků používali (a úspěšně). Kdyžtak prosím o vysvětlení. Našel jsem i nějaké kombinace např. UNet s transformerem.
3. Chápu, že se využije toho, že model má již naučené hledání okrajů, vzorů atd. Zároveň nechápu, jak by takový model mohl být srovnatelný s modelem, který byl předtrénován na VENuS. Nebo jde o to, že si člověk ušetří práci a využije obecně dostupný předtrénovaný model?
4. Zmiňoval jste jako jeden z pokročilejších kroků možnost vyzkoušet jiné architektury než U-Net. Prozatím jsem nenašel žádnou metodu, která by dosahovala lepších výsledků než U-Net či její modifikace. Měl jste na mysli nějakou konkrétní architekturu?
5. Komunikace přes GitHub - mám čekat na review PR? dokážu si představit si je "schvalovat" sám
6. Bylo by možné se domluvit na pravidelných konzultacích?
7. Bylo by možné nějak zařídit přístup na server?
