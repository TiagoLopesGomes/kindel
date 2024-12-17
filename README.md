# KinDEL Modified Implementation

This repository is based on the [insitro](https://github.com/insitro/kindel/tree/main) repository, which provides the original implementation and instructions for installation and execution. To use the ChemBERTa featurizer, please install the transformers library:
```
pip install transformers
```

## Dataset Setup

1. Download the dataset files from [42basepairs](https://42basepairs.com/browse/s3/kin-del-2024/data):
   - ddr1_1M.parquet
   - mapk14_1M.parquet

## Key Modifications

1. Enhanced Plotting Functionality [kindel/kindel/utils/plots.py] :
- Added regression metrics visualization plots
- Automatic saving of plots and CSV data

2. Extended Featurization:
- Added a combined featurizer with physicochemical descriptors that is combined with the original Morgan featurizer:

  - QED (drug-likeness)
  - Molecular Weight
  - ALogP
  - Aromatic Rings count
  - Rotatable Bonds count
  - Polar Surface Area
  - Fraction of SP3 Carbons
  - H-Bond Acceptors/Donors count

  This was retrieved form the original [paper](https://arxiv.org/pdf/2410.08938) and includes properties for accessing molecule druglikeness.

3. Added ChemBERTa Language Model Featurization:
- Implemented ChemBERTa-77M-MLM model for molecular representation
- Possible advantages over Morgan fingerprints:
  - Captures contextual chemical information through transformer architecture
  - Learns from large-scale pre-training on chemical data
  - Better handles substructure relationships
  - More robust to small structural changes
  - Can capture long-range dependencies in molecules

4. Results Handling:
- Added human-readable YAML output files


## Rationale

We focused on using the XGBoost model, which was one of the top-performing models in the original paper. XGBoost offers several advantages for this molecular prediction task:

1. Performance
   - Excellent balance between prediction accuracy and computational cost
   - Handles non-linear relationships in molecular data

2. Model Characteristics
   - Gradient boosting approach captures complex molecular traits
   - Built-in handling of missing values
   - Robust to outliers in binding data
   - Automatic feature selection and importance ranking

3. Practical Benefits
   - Faster training compared to deep learning models
   - Lower computational requirements than GNN or Transformer models
   - Easier to tune and interpret
   - Good scalability with large molecular datasets

The model was evaluated on both extended and in-library held-out sets, with metrics tracked for:
- MSE (Mean Squared Error) for prediction accuracy
- Spearman's ρ for rank correlation
- Pearson's r for linear correlation

Results are saved in the results folder for both binary (.yml) and human-readable formats, with visualization plots generated for:
- On-DNA predictions
- Off-DNA predictions
- Training/validation/test metrics


### Results
I have managed to compile the code and run it and have tested four user-cases using XGBoost. Reported at the table and image below are the train, valid and test sets MSE together with Spearman's ρ for the in-library and extended held-out set compounds:

- A) Morgan featurizer with random splits
- B) Morgan featurizer with disynthon splits 
- C) Morgan featurizer with disynthon splits and early stopping
- D) ChemBERTa featurizer with disynthon splits






![Kindel Modification Results](/results/kindel_mod_results.png)


From the above image, the results are retrieved and shown in the table below:

| Model Configuration | Train MSE | Valid MSE | Test MSE | Spearman ρ (In-Lib On-DNA) | Spearman ρ (In-Lib Off-DNA) | Spearman ρ (Ext On-DNA) |
|-------------------|-----------|-----------|-----------|-------------------|-------------------|-----------------|
| A) XGBoost + Morgan (Random) | 0.452 | 0.476 | 0.539 | 0.502 | 0.255 | 0.580 |
| B) XGBoost + Morgan (Disynthon) | 0.432| 0.603 | 0.840 | 0.632 | 0.300 | 0.637 |
| C) XGBoost + Morgan (Disynthon + Early Stop) | 0.451 | 0.606 | 0.844 | 0.637 | 0.308 | 0.655 |
| D) XGBoost + ChemBERTa (Random) | 0.495 | 0.613 | 0.680 | 0.349 | 0.154 | 0.497 |



The best overall performer regarding the Spearman correlation is Morgan featurizer with disynthon splits and early stopping, case (B), for the in-lib on-DNA targets. The test MSE for cases B and C seems to be under an overfitting scenario even with early-stopping (regularization did not seem to solve this issue (results tested but not shown). All models have a worse performance for the off-DNA held-out set, as expected. Adding the ChemBERTa featurizer also did not improve the results, indicating that the model not capturing the chemical space of the extended set, expecialy for off-DNA targets.

Overall the XGBoost model with early stopping showed 

Adding the Combined featurizer (Morgan plus descriptors) also did not improve MSE or Spearman correlations, regarding case A (tested but not shown).

Possible solutions could include testing the DELcompose model, the best performing model that the authors repoorted, with other featurizers. Also, the authors provide YAML files for hyperparameter tuning, wich could also be tested together with my implemented featurizers. One could also try the best performing model from my accompaning repository (Multitask regressor), found [here](https://github.com/TiagoLopesGomes/chemoinfo).


