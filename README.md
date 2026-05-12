# seqADAGE

> ADAGE models trained on RNAseq compendia of microbial gene expression.

## Context

This repository is an implementation of and RNA-seq version of the ADAGE
framework from [Tan et al 2016](https://pubmed.ncbi.nlm.nih.gov/27822512/) and [Tan and Doing et al 2017](https://pubmed.ncbi.nlm.nih.gov/28711280/) and the Greene lab [adage repo](https://github.com/greenelab/adage).

The tools is this repository require input data in the format of a 
compendium of gene expression as in [Doing et al 2023](https://pubmed.ncbi.nlm.nih.gov/36541761/) 
or [Refine.bio](https://www.refine.bio/). You may also find [big-little-data](https://www.refine.bio/)
a useful repository for compendium building.

Downstream analysis of adage models is preliminarily done here, but can
be followed-up by using the [ADAGEpath](https://github.com/greenelab/ADAGEpath) R package.

## In progress (To Do)

* keep iterating trainings scripts in /Py/adage
* update/integrate capatability with [big-little-data](https://github.com/georgiadoing/big-little-data)
* update consensus clustering scripts and documentation in /ensemble_construction
* expand assessment framework (gene-gene nets) in /node_interpretation and /latent_interpretation
* transfer learning workflow for staph in /Py

## Setup

This project requires the keras library for python (python 3 as of ]
6/22/21) and includes a couple class definitions designed for model 
inspection and interpretation.

I've made an up-to-date environment for this project (tfk.yml, tfkg.yaml
for gpu compatible package installations)

## How to train a seqADAGE model

See jupyter notebook for test case example of training a model and 
assessing it breifly using pathway_enrichment.R

## Documentation

Data and models-in-progress available on OSF project: https://osf.io/2pvhj/

Vignettes and examples to come.

## FAQ

Information pointing to compendium creation to come.

## Support

## License
