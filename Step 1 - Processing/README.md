# Steps to Build the datasets

1. Run merging_pipeline.py (ensure paths to input datasets are correct)
2. You will end up with the fully-merged "/merge_output/combined_df_final.csv"
3. Run general_preprocessing.py
4. You will end up with the "human-readable" versions of the sets (with NAs and without one-hot encoding) as CSV for consultation, and as parquet for further usage, as "/preprocessed_output/combined_df_preprocessed..."

Note: The "dense" versions eliminate very sparse rows like unreleased or retired titles.

You can now use
