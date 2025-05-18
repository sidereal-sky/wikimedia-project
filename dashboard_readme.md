# Wikipedia Analysis Dashboard

This interactive dashboard visualizes Wikipedia data processed by the main project, providing insights into article trends, geographical distribution of pageviews, and device usage patterns.

## Features

- **Article Trends**: Bar charts showing top articles by pageviews and pie charts displaying category distribution
- **Geographical Analysis**: Interactive world map showing the distribution of pageviews by country
- **Device Analysis**: Charts showing the breakdown of pageviews by device type (Desktop, Mobile Web, Mobile App)

## Prerequisites

- Run the main project first to generate the `titles.txt` and categorized titles data
- Install the dashboard dependencies in your conda environment

```bash
conda env update -f environment.yaml
```

## Running the Dashboard

```bash
python dashboard.py
```

This will start a local web server at http://127.0.0.1:8050/

## Using the Dashboard

1. **Article Trends Tab**:
   - View the top articles by pageviews
   - See the distribution of articles by category

2. **Geographical Analysis Tab**:
   - Select an article from the dropdown
   - The map will display estimated pageview distribution by country

3. **Device Analysis Tab**:
   - Select an article from the dropdown
   - View the breakdown of pageviews by device type

## Notes

- The dashboard caches API results in the `cache` directory to improve performance
- Some geographical data is simulated since the public API doesn't provide country-level stats
- If no real data is available, the dashboard will generate placeholder data for demonstration 