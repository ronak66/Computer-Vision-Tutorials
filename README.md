---
author:
- 'Ronak Doshi, `IMT2017523`'
title: ' **Image Enhancement**'
---

Input Image
===========

![Input Image[]{data-label="fig"}](plots/girl.png)

1. Whitening
============

$$\mu = \dfrac{\sum_{i=1}^{I} \sum_{j=1}^{J} p_{ij}}{IJ}$$

$$\sigma^2 = \dfrac{\sum_{i=1}^{I} \sum_{j=1}^{J} (p_{ij} - \mu)^2}{IJ}$$

$$x_{ij} =     \dfrac{p_{ij} - \mu}{\sigma}$$

![After Whitening Image[]{data-label="fig"}](plots/w.png)

2. Histogram Equalization 
=========================

$$h_{k} = \sum_{i=1}^{I} \sum_{j=1}^{J} \delta[p_{ij} - k]$$

$$c_{k} = \dfrac{\sum_{l=1}^{k} h_{l}}{IJ}$$

$$x_{ij} = Kc_{p_{ij}}$$

![After Histogram Equalization Image[]{data-label="fig"}](plots/w1.png)

