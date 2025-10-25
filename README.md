# Replication package for "Corrected Support Vector Regression for intraday point forecasting of prices in the continuous power market"
German continuous electricity market analysis, work done as a part of PhD studies on Wroclaw University of Science and Technology

## Authors
Andrzej Puć, Joanna Janczura, 

Wrocław University of Science and Technology, Faculty of Pure and Applied Mathematics, Hugo Steinhaus Center, Wyb. Wyspiańskiego 27, Wrocław, 50-370, Poland

### Contact information
andrzej.puc@pwr.edu.pl

## Date of replication package creation
2025.10.26

## Overview & contents
The code in this replication material allows to recalculate the forecasting simulation which served as an illustration to forecasting methodology proposed in the paper "Corrected Support Vector Regression for intraday point forecasting of prices in the continuous power market". 
When simulation is recalculated, each figure can be generated using paper_figures_reproduction.ipynb file. The notebook saves generated figures in the Paper_Figures directory.

Please note that raw data used in the forecasting study is not fully publicly available. Thus, Figures 1 and 2 cannot be generated based solely on the contents of this repository.

## Software requirements
The computing environment, language(s), licence(s) and package(s) necessary to run the reproducibility check (as well as their version); If additional information is needed to emulate the necessary environment (e.g., with conda), it should also be provided.

Requirements for text rendering with LaTeX in Matplotlib can be found here: [link](https://matplotlib.org/stable/users/explain/text/usetex.html).

## Data availability and provenance
The data being used and its format. Any relevant information regarding access to the data, origin, pre-processing, usage restrictions, etc. is to be provided.
For each sharable dataset, mention whether it is directly included in the replication kit or available elsewhere (repository, website).
For each non-sharable dataset (copyright, NDA, restricted access), provide the following relevant information on how to obtain it: data provider, database identifier (name, DOI, vintage), application and registration procedures, monetary costs, time requirements, instructions on which range and variables to pick. Please indicate whether a third party can temporarily access the data (for reproduction purposes).

## Hardware requirements and expected runtime
The simulation relies on heavy usage of parallel computing.
It was performed using the resources of Wrocław Centre for Networking and Supercomputing (WCSS).
Specifically, CPU: 2 x Intel Xeon Platinum 8268 (24 cores, 2,9 GHz), RAM: 192 GB 2933 MHz ECC DDR4.
Runtime on such config, using 48 parallel workers, is around 60 hours for (c)SVR models simulation and [] for the limited LASSO and RF simulations respectively.

## Running the simulation
instruction
- how to preprocess the data
- how to run the simulation
- how to add the weighted averaging
- how to calculate the MAE aggregation (Note that the intel_avg_generator.py requires changing the configuration to )
