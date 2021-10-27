# morphometricity
Morphometricity project aims at reproducing the result from Sabuncu et al. 2016 (https://www.pnas.org/content/113/39/E5749.short) with large UK Biobank data, and hopefully extend the method to brain age prediction. 

Morphometricity is defined as the proportion of phenotypic variance explained by the brain morphology. The method uses the anatomical similarity matrix as the correlation of random intercepts in the linear mixed effect model for predicting phenotype, and obtain an estimated morphometricity from optimizing the restricted likelihood (by expectation-maximization algorithm). 

The implementation is in Python.

Collaboration with Nikhil and Jerome from the lab, and Clara from Pasteur institute, Paris. 

