## Parse and analyze wikipedia dumps with Spark and Wikimedia API

### Prerequisites

- [Conda or Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

```bash
conda env create -f environment.yaml

conda activate wikimedia-project
```

### Run

Basic analysis:
```bash
python3 main.py
```

Interactive dashboard (recommended):
```bash
python3 run_dashboard.py
```
This will guide you through processing the dumps and launch an interactive dashboard at http://127.0.0.1:8050

### Structure
- `main.py`: Main script to parse the wiki dumps and run the analysis
- `insights.py`: Contains the analysis functions using wikimedia API
- `cluster_titles.py`: Zero-shot classification of article titles
- `dashboard.py`: Interactive visualization dashboard for the analyzed data
- `run_dashboard.py`: Helper script to process dumps and launch the dashboard

### Dashboard Features

The dashboard provides visualizations for:

- Article trends: Top articles by pageviews and distribution by category
- Geographic analysis: Interactive world map showing pageview distribution
- Device analysis: Charts showing breakdown of pageviews by device

For detailed dashboard instructions, see `dashboard_readme.md`
