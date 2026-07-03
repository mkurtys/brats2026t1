

nnUNet_experiment = $1
fold = 0 # for now


validation_path = "$nnUNet_results/$nnUNet_experiment/fold_$fold/validation"
nnUNet_preprocessed =

brats-evaluate \
    --config mets \
    --ref_path /path/to/reference/niftis/ \
    --pred_path /path/to/prediction/niftis/ \
    --summary_json ./panoptica_evaluation_summary.json