# Structure for ML projects

## Directory structure
### Directories
- data: directory for ingested data
  - External: Data from third party sources
  - Interim: Intermediate data that has been transformed
  - Processed: Final and canonical data sets for modeling
  - Raw: Original and immutable data dumb

- models: Trained and serialized models, model predictions, or model summaries
- reports: Generated analysis as HTML, PDF, LaTeX, etc..
  - Figures: Generated graphics and figures to be used in reporting
- src: Source code for use in this project
  - __init__.py: Makes src a Python module
  - data: Scripts to download or generate data
  - features: Scripts to turn raw data into features or modeling
  - models: Scripts to train models and then use trained models to make predictions
  - visualization: Scripts to rceate exploratory and results oriented visualization

### Other files
- Makefile: Makefile with commands like 'make data' or 'make train'
- README.md: Top-level README for developers using this project
- requirements.txt: Requirements file for reproducing the analysis environment. Can be generated with 'pip freeze > requirements.txt'