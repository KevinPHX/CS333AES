# Graphically Modeling Text Coherence for Automated Essay Scoring 
This is a replication study of a paper by Stab & Gurveych (2017) on persuasive essays from the Hewlett Foundation's Automated Student Assessment Prize (ASAP) dataset. 

- **Citation**: Stab, C., & Gurevych, I. (2017). Parsing Argumentation Structures in Persuasive Essays. *Computational Linguistics*, 43(3), 619–659. https://doi.org/10.1162/COLI_a_00295


## Datasets  
- ASAP Set 2: Censorship in libraries https://www.kaggle.com/c/asap-aes 
- Argument Annotated Essay (AAE) Dataset (Stab & Gurveych 2017): https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2422

## Main Code Files

### stab/lib
- argument_identification.py
    - Performs feature extraction for Argument Identification on a list of essay paths
- argument_classification.py
    - Performs feature extraction Argument Classification on the labeled essay tokens in the form of a list of dictionaries
- argument_relations_preprocess.py
    - Writes out to another file the lemmas that occur within components that have either incoming or outgoing relations in the training data, as well as the probability that a component is associated with a certain direction. The script also formats all of the ground truth labels for components and relations from the .ann files of the training data. The outputs of this script are the argument_relation_info.json, argument_relation_info_TEST_SET.json, relation_probabilities.json, and training_data_lemma.json files in the stab/models folder.  
- argument_relations.py
    - Performs pairwise feature extraction for Relation Identification based on the outputs of argument_classification.py and argument_relations_preprocess.py. 
- argumentILP_withJSON.py
    - Performs optimization of relations and components that result from the outputs of the base SVM classifiers for the classification and relation identification stage. 
- stance_recognition.py
    - Performs feature extraction for Stance Recognition using information extracted from Argument Classification as a list of dictionaries

### stab/scripts 
- train.py
    - Runs Argument Identification, Argument Classification, Argument Relation Identification, and Stance Identification on training data and saves files as a CSV or JSON in stab/outputs. We also extract and save dependency and probability features 
- train_models.py
    - Trains the CRF for Argument Identification, and SVMs for Argument Classification, Argument Relation Identification, and Stance Identification and saves them in stab/models
- test.py
    - Runs Argument Identification, Argument Classification, Argument Relation Identification, and Stance Identification on test data and saves files as a CSV or JSON in stab/outputs/test. Note that the dependency and probability features from train.py must be inputs
- test_models.py 
    - Runs the trained CRF and SVMs on test data and prints out classification reports and confusion matrices for each step
- run_ILP.py
    - Formats inputs and calls the argumentILP_withJSON.py script to perform ILP for the AAE dataset. Writes outputs to the stab/outputs/test_set_optimized_relations.json file. Prints classification reports for both revised relations and components, which we stored in the ILP_results.txt file. 
- asap.py 
    - Selects a subset of essays from the ASAP dataset and formats all relevant essays and domain 1 scores into the stab/assets/asap_essays.json file.  
- asap_run_ILP.py 
    - Formats inputs and calls the argumentILP_withJSON.py script to perform ILP for the ASAP dataset. Writes the output to stab/outputs/asap_set2/optimized_relations.json. 
- asap_classification.py
    - Performs Argument Classification for the ASAP dataset. Writes features to the stab/outputs/asap_set2/classification folder and predictions to the stab/outputs/asap_set2/classification_predictions.json file. 
- asap_identification.py
    - Performs Argument Identification for the ASAP dataset. Writes outputs to CSVs in the stab/outputs/asap_set2 folder. 
- asap_relations_and_stances.py
    - Performs Relation Identification and Stance Recognition for the ASAP dataset. Writes outputs to the stab/outputs/asap_set2/relations and stab/outputs/asap_set2/stance folder. 

### stab/assets 
The files in this folder contain the names of the essays in our training data, AAE test set, and the subset of ASAP essays that we are working with. 
### stab/archive
These are old code files from our initial explorations with CoreNLP in Java and organizing features using Python. 
### stab/models 
These files contain the preprocessed data that we use for further feature extraction. 
### stab/notebooks 
Some Jupyter Notebooks for initial exploration and evaluation. Other IPYNB files are located in the stab/archive folder. 
- asap_persuasive.ipynb
    - This file contains our code for essay scoring and visualization. 
### stab/outputs 
This folder contains files and subfolders of files storing the features and predictions we obtain for each step of the algorithm. 
- visualization.py 
    - This file visualizes argument trees as interactive graphs. 

