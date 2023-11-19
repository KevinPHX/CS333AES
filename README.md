# Graphically Modeling Text Coherence for Automated Essay Scoring 
Data Sets:
https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2422

## Repository Directory

### stab/lib
stab/lib/argument_identification.py
- Performs Argument Identification on a list of essay paths
stab/lib/argument_classification.py
- Performs Argument Classification on the labeled essay tokens in the form of a list of dictionaries
stab/lib/argument_relations_preprocess.py
- 
stab/lib/argument_relations.py

stab/lib/argumentILP_withJSON.py


stab/lib/stance_identification.py
- Performs Stance Identification using information extracted from Argument Classification as a list of dictionaries

### stab/scripts 
stab/scripts/train.py
- Runs Argument Identification, Argument Classification, Argument Relation Identification, and Stance Identification on training data and saves files as a CSV or JSON in stab/outputs. We also extract and save dependency and probability features 
stab/scripts/train_models.py
- Trains the CRF for Argument Identification, and SVMs for Argument Classification, Argument Relation Identification, and Stance Identification and saves them in stab/models
stab/scripts/test.py
- Runs Argument Identification, Argument Classification, Argument Relation Identification, and Stance Identification on test data and saves files as a CSV or JSON in stab/outputs/test. Note that the dependency and probability features from train.py must be inputs
stab/scripts/test_models.py 
- Runs the trained CRF and SVMs on test data and prints out classification reports and confusion matrices for each step
stab/scripts/run_ILP.py
stab/scripts/asap_run_ILP.py 
stab/scripts/asap_classification.py
stab/scripts/asap_identification.py
stab/scripts/asap_relations_and_stances.py

### stab/assets 
### stab/archive
### stab/outputs 
	
