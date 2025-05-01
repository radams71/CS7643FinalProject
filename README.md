# Graph Neural Networks
Robert Adams:
DeeperGCN training is done in training.py, with finetuning with random forest fingerprints found in finetuning_with_fp.py. rf_preds contains the random forest models trained on the fingerprints.

Captum analysis visualizations of molecules can be generated with the scripts in the visualizations directory.  

Pedram Gharani:
conv_gcn_enhanced_att.py contains our model which incorporates multi-headed pooling attention and residual connections.

visualization_script.py and visualize_results.py are used to help visualize training data from our custom GCN and produces visualizations of molecules showing the attention values.

