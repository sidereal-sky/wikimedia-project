## Parse and analyze wikipedia dumps with Spark and Wikimedia API

### Prerequisites

- [Conda or Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

```bash
conda env create -f environment.yaml

conda activate wikimedia-project
```

### Run

```bash
python3 main.py
```

### Structure
- `main.py`: Main script to parse the wiki dumps and run the analysis
- `insights.py`: Contains the analysis functions using wikimedia API
- `cluster_titles.py`: Zero-shot classification of article titles
