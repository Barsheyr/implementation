# Output files (folder `results/`, zipped as `results.zip`)
config.json, dataset_summary.json, weather_meta.json, weather_profiles.csv, minmax.json, splits.csv, severity_summary.json,
tuning_results.csv (full mode), summary.json, run_log.txt, runs/<variant>_seed<seed>/{history.csv,test_metrics.json,predictions.csv,best.pt (M1)},
explainability/{shap_values.csv,shap_mean_abs.json,shap_check.json,explanations.json},
figures/{eda_class_distribution,sample_images,weather_profiles,curves_<variant>_seed<seed>,confusion_<variant>,comparison_macroF1,per_class_f1,gradcam_samples,explanation_panel,shap_summary,shap_beeswarm,severity_scatter}.png
severity_grading/{grading_sheet.csv,montage_*.jpg}
