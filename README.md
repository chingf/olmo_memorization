# Project Files:
- `00_olmo_demo.ipynb`: Example of how to load and run a single prompt with OLMo
- `01_compute_p_match.py`: Greedily prompts model on the midtraining Dolmino-FLAN dataset, records the memorization score for each prompt, and saves the results to pickle.
- `02_analyze_proportions.ipynb`: Plots histogram of memorization scores and prints examples
- `03_calc_component_gradients.py`: Calculates the gradient attribution for model component weights (for passages with low or high memorization score) and saves the results to pickle.
- `04_plot_grad_heatmaps.ipynb`: Plots heatmaps of attribution scores and PCA plots
- `05_logistic_regression_classifier.ipynb`: Trains a classifier for low/high memorization and tests on held-out dataset.
