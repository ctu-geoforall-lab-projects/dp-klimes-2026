# Poznámky
- Primárně bych rád během konzultace odprezentoval, co jsem již v problematice pochopil, jak chápu první kroky atd. (pokusím se to sem nějak sepsat během pondělí 1.9.2025)

# Otázky na konzultaci
1. Zaujala mě Domain adaptation jako možnost/nutnost pro dobré přenesení modelu mezi různými senzory. Rád bych se pobavil o tom, zda a do jaké míry ji provádět. Zda použít výběr stejných pásem, převzorkování, srovnání histogramů, multivariate matching, graph matching či v extrému GAN (viz článek z roku 2020 - Cross-Sensor Adversarial Domain Adaptation..)
    - Napadá mi k tomu i možnost srovnávat výkon modelu 1 (trénovaného čistě na VENuS) s modelem 2 (trénovaného na VENuS + fine-tune na S2) s modelem 3 (použít určitý stupeň domain adaptation + fine-tune na S2) atd včetně layer freezing a vlastně se vším, co jste psal v bodě 4 basic outline of project
2. Nestálo by za to podívat se i na srovnání CNN a transformerů (nevím zda se překládá čistě takto :D)? Moc toho o nich nevím, ale našel jsem nějaké články, kde je také k detekci mraků používali (a úspěšně). Kdyžtak prosím o vysvětlení. Našel jsem i nějaké kombinace např. UNet s transformerem.
3. Bylo by možné se domluvit na pravidelných konzultacích?
