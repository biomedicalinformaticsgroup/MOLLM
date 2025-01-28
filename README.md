# Combining Clinical Embeddings with Multi-Omic Features for Improved Patient Classification and Interpretability in Parkinson’s Disease

Link to preprint - [https://www.medrxiv.org/content/10.1101/2025.01.17.25320664v1](https://www.medrxiv.org/content/10.1101/2025.01.17.25320664v1)

## Abstract

This study demonstrates the integration of Large Language Model (LLM)-derived clinical text embeddings from the Movement Disorder Society Unified Parkinson’s Disease Rating Scale (MDS-UPDRS) questionnaire with molecular genomics data to enhance patient classification and interpretability in Parkinson’s disease (PD). By combining genomic modalities encoded using an interpretable biological architecture with a patient similarity network constructed from clinical text embeddings, our approach leverages both clinical and genomic information to provide a robust, interpretable model for disease classification and molecular insights. We benchmarked our approach using the baseline time point from the Parkinson’s Progression Markers Initiative (PPMI) dataset, identifying the Llama-3.2-1B text embedding model on Part III of the MDS-UPDRS as the most informative. We further validated the framework at years 1, 2, and 3 post-baseline, achieving significance in identifying PD associated genes from a random null set by year 2 and replicating the association of MAPK with PD in a heterogeneous cohort. Our findings demonstrate that combining clinical text embeddings with genomic features is critical for both classification and interpretation.

## Workflow

![Framework](./workflow_diagrams/pipeline.jpg)

### Panel A. Data Availability

Data was obtained from the publicly available [PPMI](https://www.ppmi-info.org/) website.

Clinical assessments were taken from the MDS-UPDRS questionnaire, and genomic modalities comprise CSF, mRNA and DNAm samples. Samples were matched to the corresponding time point of the clinical assessments. For more information, see preprint.

### Panel B. Generating LLM Embeddings

To generate embeddings, use the following command:

```
cd ./llm_embed
python main.py model.model_name={model_path} dataset.event_id={event_id} dataset.section={section}
```

You can also modify configurations directly in `./configs/config.yaml`.

*   `event_id`: Refers to the timepoint identifier in PPMI's data scheme. For example, V08 corresponds to year 3 post-baseline.
    *   `"BL"`: Baseline
    *   `"V04"`: 1 year after baseline
    *   `"V06"`: 2 years after baseline
    *   `"V08"`: 3 years after baseline
*   `section`: Specifies the MDS-UPDRS questionnaire section to generate embeddings for:
    *   `"2"`: Section 2 (Motor Aspects of Experiences of Daily Living)
    *   `"3"`: Section 3 (Motor Examination)
    *   `"23"`: Joint embedding for Sections 2 and 3.

The embeddings will be saved as `.pkl` files in `./embeddings`. The naming convention for the files is as follows:

```
{model_name}_section{section}_{event_id}_{timestamp}.pkl
```

For example, `./embeddings/Llama-3.2-1B_section2_BL_22jan134800.pkl`.

Saved embeddings contain both mean-pooled and last token embeddings for each `patient_id` and `event_id`.

### Panel C. Encoding Genomic Modalities

Modalities need to be pre-processed to ensure data quality standards. The `MOGDx/PreProcessing/Preprocessing.ipynb` jupyter notebook was used for processing modalities.

### Paenl D. Model Predictions

The framework can be run from the command line. A sample command is :

```
python MOGDx.py -i "data/PPMI/input/" -o "Results/MOLLM/Output/" --n-splits 5 -mod mRNA CSF DNAm -ld 32 32 16 --target "CONCOHORT_DEFINITION" --index-col "index" --epochs 2000 --lr 0.001 --h-feats 16 --decoder-dim 16 --pnet --model GCN --gen-net --emb-model Llama --section sec3 --interpret --tmpt "V08"
```

#### Description of arguments

*   `-i "data/PPMI/input/"`: Specifies the input directory containing the data to be processed.
*   `-o "Results/MOLLM/Output/"`: Designates the output directory where all the results of the execution will be saved.
*   `--n-splits 5`: Sets the number of splits for cross-validation
*   `-mod mRNA CSF DNAm`: Lists the modalities to be included in the analysis, specifically mRNA, CSF (Cerebrospinal Fluid), and DNAm (DNA methylation).
*   `-ld 32 32 16`: Specifies the latent dimensions for each modality: 32 for mRNA, 32 for CSF, and 16 for DNAm.
*   `--target "CONCOHORT_DEFINITION"`: Identifies the target variable for the analysis
*   `--index-col "index"`: Indicates the column in the dataset used as the index or key for data points.
*   `--epochs 2000`: Sets the number of training cycles/epochs the model will run to 2000.
*   `--lr 0.001`: Establishes the learning rate of the model at 0.001, a critical parameter influencing the optimization algorithm's step size.
*   `--h-feats 16`: Configures the number of features in the hidden layers of the graph neural network to 16.
*   `--decoder-dim 16`: Sets the dimension of the shared decoder in the neural network to 16.
*   `--pnet`: Activates the usage of pnet biological encoder
*   `--model GCN`: Specifies the type of model to use as GCN (Graph Convolutional Network)
*   `--gen-net`: Indicates that the script should generate patient similarity network
*   `--emb-model Llama`: Specifies the embedding model
*   `--section sec3`: Denotes a specific section of MDS-UPDRS from which embeddings were generated
*   `--tmpt "V08"`: Specifies the time point at which text embeddings were generated
*   `--interpret`: Switch to calculate the attribution of the hidden PNet layers

Two main requirements need to be met for successful execution of the model.

1.  PPMI data needs to be stored in `/data/PPMI/input`, processed and saved using the convention `{genomic}_processed.pkl`. This is done automatically using the provided `MOGDx/Preprocessing/Preprocessing.ipynb` jupyter notebook
2.  The text embeddings need to be extracted and stored in `/data/PPMI/ext_data` using the naming convention `{tmpt}_{emb_model}_{section}_{emb_strategy}.pkl`

## Contact

Barry Ryan, Chaeeun Lee

PhD Candidate Biomedical AI, University of Edinburgh   
School of Informatics, 10 Crichton Street

Email: barry.ryan@ed.ac.uk   
LinkedIn: https://www.linkedin.com/in/barry-ryan/

Email: chaeeun.lee@ed.ac.uk   
LinkedIn: https://www.linkedin.com/in/chaeeun-lee-274210143/