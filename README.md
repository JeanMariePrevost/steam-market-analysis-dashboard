# Steps to Build the datasets

1. Run **merging_pipeline.py** (ensure paths to input datasets are correct) to create the merged set from all selected sources.
2. You will end up with the fully-merged "/merge_output/combined_df_final.csv"
3. Run **general_preprocessing.py** to preprocess and clean up the merged set.
4. You will end up with the "human-readable" versions of the sets (with NAs and without one-hot encoding) as CSV for consultation, and as parquet for further usage, as "/preprocessed_output/combined_df_preprocessed..."

Note: The "dense" versions eliminate very sparse rows like unreleased or retired titles.
You can now use this dataset as is for various analyses.


## For general use with machine learning, continue with:
5. Run **feature_engineering.py** to further process the dataset into a form that is prepared for various machine learning techniques (e.g. by converting various categorical variables and lists using one-hot encoding, by imputing missing values...)

## For inference / predictions, continue with:
6. Run **inference_dataset_preprocessing.py** to generate the inference dataset
7. Run **inference_baseline_element.py** to calculate the "baseline" element for inference.


