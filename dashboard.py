import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Check for required dependencies and provide helpful error messages
missing_deps = []
try:
    import pandas as pd
except ImportError:
    missing_deps.append("pandas")

try:
    import numpy as np
except ImportError:
    missing_deps.append("numpy")

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    missing_deps.append("plotly")

try:
    import dash
    from dash import dcc, html, Input, Output, State, callback_context
except ImportError:
    missing_deps.append("dash")

try:
    import folium
    from folium.plugins import HeatMap
except ImportError:
    missing_deps.append("folium")

# Handle missing dependencies
if missing_deps:
    print("Error: Missing required dependencies.")
    print(f"Please install: {', '.join(missing_deps)}")
    print("Run: conda install -c conda-forge " + " ".join(missing_deps))
    sys.exit(1)

import json
import requests
from datetime import datetime, timedelta
import pickle

# File paths
TITLES_FILE = "titles.txt"
CATEGORIZED_TITLES_FILE = "categorized_titles"

# Cache directory
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Create example data if not available
def create_example_titles():
    """Create example titles file if it doesn't exist"""
    example_titles = [
        "Python (programming language)",
        "Machine learning",
        "Data science",
        "Artificial intelligence",
        "Solar System",
        "Quantum physics",
        "History of Rome",
        "Leonardo da Vinci",
        "World War II",
        "Renewable energy",
        "Human genome",
        "Climate change",
        "Greek mythology",
        "French Revolution",
        "Albert Einstein",
        "William Shakespeare",
        "United States Constitution",
        "Ancient Egypt",
        "Industrial Revolution",
        "Theory of relativity"
    ]
    
    with open(TITLES_FILE, "w") as f:
        for title in example_titles:
            f.write(f"{title}\n")
    
    return example_titles

# Check if we have necessary data files
if not os.path.exists(TITLES_FILE):
    print(f"Warning: {TITLES_FILE} not found. Creating example data.")
    create_example_titles()

if not os.path.exists(CATEGORIZED_TITLES_FILE):
    print(f"Warning: {CATEGORIZED_TITLES_FILE} not found. Dashboard will use simulated categories.")
    os.makedirs(CATEGORIZED_TITLES_FILE, exist_ok=True)

# Wikimedia API functions (modified from insights.py)
def fetch_pageviews_by_country(title, start_date=None, end_date=None):
    """Fetch pageviews by country for a given Wikipedia article"""
    if start_date is None:
        # Default to last 30 days
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
    
    # Use a consistent filename format
    safe_title = title.replace(' ', '_').replace('/', '_').replace('\\', '_')
    cache_file = os.path.join(CACHE_DIR, f"{safe_title}_geocounts.pkl")
    views_cache_file = os.path.join(CACHE_DIR, f"{safe_title}_views.pkl")
    
    # Check if cached data exists
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading geocounts cache for {title}: {e}")
    
    # Try to get the total views from the views cache
    total_views = None
    if os.path.exists(views_cache_file):
        try:
            with open(views_cache_file, 'rb') as f:
                total_views = pickle.load(f)
        except Exception as e:
            print(f"Error loading views cache for {title}: {e}")
    
    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/{title}/daily/{start_date}/{end_date}"
    headers = {
        'User-Agent': 'WikiDashboard/1.0 (victormmirica@gmail.com.com)'
    }
    
    # Since the API doesn't directly provide country data, we'll simulate it
    # In a real implementation, you'd need to use the pageview API and Geoip data
    countries = ["United States", "United Kingdom", "India", "Germany", "France", 
                "Canada", "Australia", "Japan", "Brazil", "Mexico"]
    
    try:
        # If we already have total views from cache, skip the API call
        if total_views is None:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                total_views = sum(item["views"] for item in data["items"])
                
                # Cache the total views for consistency
                with open(views_cache_file, 'wb') as f:
                    pickle.dump(total_views, f)
            else:
                print(f"API error for {title}: {response.status_code}")
                total_views = np.random.randint(1000, 10000)
        
        # Distribute views across countries (simulated)
        # Use fixed seed based on title for consistency
        seed = sum(ord(c) for c in title)
        np.random.seed(seed)
        weights = np.random.dirichlet(np.ones(len(countries))*1.5)
        country_data = {country: int(total_views * weight) for country, weight 
                       in zip(countries, weights)}
        
        # Cache the result
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(country_data, f)
        except Exception as e:
            print(f"Error caching country data for {title}: {e}")
            
        return country_data
    except Exception as e:
        print(f"Error fetching data for {title}: {e}")
        # Generate sample data
        total_views = np.random.randint(1000, 10000)
        np.random.seed(seed if 'seed' in locals() else 0)
        weights = np.random.dirichlet(np.ones(len(countries))*1.5)
        country_data = {country: int(total_views * weight) for country, weight 
                       in zip(countries, weights)}
        return country_data

def fetch_pageviews_by_device(title, start_date=None, end_date=None):
    """Fetch pageviews by device type for a given Wikipedia article"""
    if start_date is None:
        # Default to last 30 days
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
    
    cache_file = os.path.join(CACHE_DIR, f"{title.replace(' ', '_')}_devices.pkl")
    
    # Check if cached data exists
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except:
            pass
    
    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/{title}/daily/{start_date}/{end_date}"
    headers = {
        'User-Agent': 'WikiDashboard/1.0 (victormmirica@gmail.com.com)'
    }
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            total_views = sum(item["views"] for item in data["items"])
            
            # Simulate device breakdown (in real implementation, you'd use the access-site parameter)
            device_data = {
                "Desktop": int(total_views * 0.55),
                "Mobile Web": int(total_views * 0.35),
                "Mobile App": int(total_views * 0.10)
            }
            
            # Cache the result
            with open(cache_file, 'wb') as f:
                pickle.dump(device_data, f)
                
            return device_data
        else:
            print(f"API error: {response.status_code}")
            # Return sample data
            total_views = np.random.randint(1000, 10000)
            return {
                "Desktop": int(total_views * 0.55),
                "Mobile Web": int(total_views * 0.35),
                "Mobile App": int(total_views * 0.10)
            }
    except Exception as e:
        print(f"Error fetching data for {title}: {e}")
        # Return sample data
        total_views = np.random.randint(1000, 10000)
        return {
            "Desktop": int(total_views * 0.55),
            "Mobile Web": int(total_views * 0.35),
            "Mobile App": int(total_views * 0.10)
        }

# Data loading functions
def load_titles():
    """Load article titles from the titles.txt file"""
    if os.path.exists(TITLES_FILE):
        with open(TITLES_FILE, 'r') as f:
            return [line.strip() for line in f.readlines()]
    return create_example_titles()

def load_categorized_titles():
    """Load categorized titles from the output CSV file"""
    # First try to load the simpler CSV file
    simple_csv = os.path.join(CATEGORIZED_TITLES_FILE, "categories.csv")
    if os.path.exists(simple_csv):
        try:
            return pd.read_csv(simple_csv)
        except Exception as e:
            print(f"Error reading simple CSV: {e}")
    
    # Try fallback CSV
    fallback_csv = os.path.join(CATEGORIZED_TITLES_FILE, "fallback.csv")
    if os.path.exists(fallback_csv):
        try:
            return pd.read_csv(fallback_csv)
        except Exception as e:
            print(f"Error reading fallback CSV: {e}")
    
    # Try looking through Spark output
    spark_output = os.path.join(CATEGORIZED_TITLES_FILE, "spark_output")
    if os.path.exists(spark_output):
        csv_files = [f for f in os.listdir(spark_output) if f.endswith('.csv')]
        if csv_files:
            for file in csv_files:
                filepath = os.path.join(spark_output, file)
                try:
                    # Use a more robust parsing approach
                    df = pd.read_csv(filepath, engine='python', error_bad_lines=False, 
                                   warn_bad_lines=True)
                    return df
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")
    
    # Final fallback - try old files
    if os.path.exists(CATEGORIZED_TITLES_FILE):
        # Directory with CSVs
        csv_files = [f for f in os.listdir(CATEGORIZED_TITLES_FILE) if f.endswith('.csv')]
        if csv_files:
            for file in csv_files:
                filepath = os.path.join(CATEGORIZED_TITLES_FILE, file)
                try:
                    # Try with different delimiters
                    for delimiter in [',', '||', '|']:
                        try:
                            df = pd.read_csv(filepath, sep=delimiter, engine='python', 
                                          error_bad_lines=False)
                            if not df.empty:
                                return df
                        except:
                            continue
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")
    
    # Create dummy data if file doesn't exist
    categories = ["technology", "science", "history", "art", "sports", "politics"]
    dummy_data = []
    for title in load_titles()[:100]:  # First 100 titles
        category = np.random.choice(categories)
        confidence = round(np.random.uniform(0.6, 0.95), 2)
        dummy_data.append({
            "title": title,
            "category": f"{category} ({confidence})"
        })
    return pd.DataFrame(dummy_data)

def get_category_distribution():
    """Get distribution of articles by category"""
    df = load_categorized_titles()
    
    # Extract just the category name (removing confidence score)
    df['category_name'] = df['category'].apply(
        lambda x: x.split('(')[0].strip() if isinstance(x, str) and '(' in x else x
    )
    
    category_counts = df['category_name'].value_counts().reset_index()
    category_counts.columns = ['category', 'count']
    return category_counts

def get_top_articles_by_views(limit=20):
    """Get top articles by pageviews"""
    titles = load_titles()
    
    # Load cached data if available
    cache_file = os.path.join(CACHE_DIR, "top_articles.pkl")
    
    # Only use cache if explicitly requested or for testing
    use_cache = os.environ.get('USE_CACHE', 'false').lower() == 'true'
    
    if use_cache and os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                # Verify we have enough diverse data
                if len(cached_data) >= limit:
                    return cached_data
        except:
            pass
    
    # Essential titles that should always be included in our analysis
    # These are known high-traffic articles
    essential_titles = [
        "China", "United States", "India", "Russia", "Germany", "United Kingdom",
        "World War II", "COVID-19 pandemic", "Artificial intelligence", "Earth",
        "Solar System", "iPhone", "Netflix", "Google", "Facebook", "Climate change"
    ]
    
    # Filter to keep only essential titles that exist in our dataset
    essential_titles = [t for t in essential_titles if t in titles]
    
    # Process titles in batches to avoid memory issues
    # Sample from across the alphabet by taking every Nth item
    regular_sample_size = min(300, len(titles))
    step = max(1, len(titles) // regular_sample_size)
    sampled_titles = [titles[i] for i in range(0, len(titles), step)][:regular_sample_size]
    
    # Add essential titles to our sample, but avoid duplicates
    for title in essential_titles:
        if title not in sampled_titles:
            sampled_titles.append(title)
    
    # Process titles in batches
    article_views = []
    batch_size = 50
    for i in range(0, len(sampled_titles), batch_size):
        batch = sampled_titles[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(sampled_titles) + batch_size - 1)//batch_size}")
        
        for title in batch:
            # Use a consistent filename format for caching
            safe_title = title.replace(' ', '_').replace('/', '_').replace('\\', '_')
            cache_file = os.path.join(CACHE_DIR, f"{safe_title}_views.pkl")
            
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'rb') as f:
                        views = pickle.load(f)
                except Exception as e:
                    print(f"Error loading cache for {title}: {e}")
                    views = None
            else:
                views = None
                
            if views is None:
                try:
                    end_date = datetime.now().strftime("%Y%m%d")
                    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
                    
                    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/{title}/daily/{start_date}/{end_date}"
                    headers = {'User-Agent': 'WikiDashboard/1.0 (victormmirica@gmail.com.com)'}
                    
                    response = requests.get(url, headers=headers)
                    if response.status_code == 200:
                        data = response.json()
                        views = sum(item["views"] for item in data["items"])
                    else:
                        # Generate random data if API call fails
                        print(f"API error for {title}: {response.status_code}")
                        views = np.random.randint(100, 10000)
                except Exception as e:
                    print(f"Error fetching data for {title}: {e}")
                    views = np.random.randint(100, 10000)
                    
                # Cache the result
                try:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(views, f)
                except Exception as e:
                    print(f"Error caching data for {title}: {e}")
            
            article_views.append({'title': title, 'views': views})
    
    # Sort by views
    article_views = sorted(article_views, key=lambda x: x['views'], reverse=True)[:limit]
    
    # Cache the result
    with open(os.path.join(CACHE_DIR, "top_articles.pkl"), 'wb') as f:
        pickle.dump(article_views, f)
        
    return article_views

# Dashboard App
app = dash.Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([
    html.H1("Wikipedia Analysis Dashboard", style={'textAlign': 'center'}),
    
    # Store for pre-loaded data - loaded once when dashboard starts
    dcc.Store(id='titles-store', data=[
        {'label': title, 'value': title} for title in load_titles()[:200]  # Initial 200 titles
    ]),
    dcc.Store(id='all-titles-store', data=load_titles()),  # Full list for search
    
    dcc.Tabs([
        dcc.Tab(label='Article Trends', children=[
            html.Div([
                html.Div([
                    html.H2("Top Articles by Pageviews", style={'textAlign': 'center'}),
                    html.Button(
                        'Refresh Data', 
                        id='refresh-articles-button', 
                        n_clicks=0, 
                        style={
                            'marginBottom': '20px', 
                            'padding': '10px 20px', 
                            'backgroundColor': '#4CAF50', 
                            'color': 'white', 
                            'border': 'none',
                            'borderRadius': '4px',
                            'float': 'right'
                        }
                    ),
                ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center'}),
                dcc.Loading(
                    id="loading-top-articles",
                    type="circle",
                    children=dcc.Graph(id='top-articles-graph'),
                ),
                html.Hr(),
                html.H2("Articles by Category", style={'textAlign': 'center'}),
                dcc.Loading(
                    id="loading-categories",
                    type="circle",
                    children=dcc.Graph(id='category-distribution-graph'),
                )
            ])
        ]),
        
        dcc.Tab(label='Geographical Analysis', children=[
            html.Div([
                html.H2("Geographical Distribution of Pageviews", style={'textAlign': 'center'}),
                html.P("Select an article to view its geographical distribution:"),
                html.Div([
                    dcc.Dropdown(
                        id='geo-article-dropdown',
                        options=[],  # Will be populated by callback
                        value=load_titles()[0] if load_titles() else None,
                        style={'marginBottom': '10px', 'width': '100%'},
                        searchable=True,
                        placeholder="Search for an article... (type to see options)"
                    ),
                    html.Button('Update Map', id='update-map-button', n_clicks=0, 
                               style={'marginBottom': '20px', 'padding': '10px 20px', 'backgroundColor': '#4CAF50', 'color': 'white', 'border': 'none'}),
                ], style={'display': 'flex', 'flexDirection': 'row', 'alignItems': 'center', 'gap': '10px'}),
                dcc.Loading(
                    id="loading-map",
                    type="circle",
                    children=html.Iframe(id='map-iframe', width='100%', height='700', style={'border': 'none', 'borderRadius': '8px'})
                )
            ])
        ]),
        
        dcc.Tab(label='Device Analysis', children=[
            html.Div([
                html.H2("Pageviews by Device Type", style={'textAlign': 'center'}),
                html.P("Select an article to view device breakdown:"),
                dcc.Dropdown(
                    id='device-article-dropdown',
                    options=[],  # Will be populated by callback
                    value=load_titles()[0] if load_titles() else None,
                    searchable=True,
                    placeholder="Search for an article... (type to see options)"
                ),
                dcc.Loading(
                    id="loading-device",
                    type="circle",
                    children=dcc.Graph(id='device-distribution-graph')
                )
            ])
        ])
    ]),
    
    html.Div(id='map-storage', style={'display': 'none'})
])

# Callbacks
@app.callback(
    Output('top-articles-graph', 'figure'),
    [Input('top-articles-graph', 'id'),
     Input('refresh-articles-button', 'n_clicks')]
)
def update_top_articles_graph(_, n_clicks):
    # Check if this is a refresh click or initial load
    ctx = callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    
    # If it's a refresh click, clear the top articles cache
    if trigger_id == 'refresh-articles-button' and n_clicks > 0:
        cache_file = os.path.join(CACHE_DIR, "top_articles.pkl")
        if os.path.exists(cache_file):
            try:
                os.remove(cache_file)
                print("Cleared top articles cache for refresh")
            except Exception as e:
                print(f"Error clearing cache: {e}")
    
    articles = get_top_articles_by_views(20)
    df = pd.DataFrame(articles)
    
    fig = px.bar(
        df, 
        x='views', 
        y='title', 
        orientation='h',
        title='Top 20 Articles by Pageviews',
        labels={'views': 'Pageviews', 'title': 'Article Title'},
        color='views',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        height=600
    )
    
    return fig

@app.callback(
    Output('category-distribution-graph', 'figure'),
    Input('category-distribution-graph', 'id')
)
def update_category_distribution_graph(_):
    df = get_category_distribution()
    
    fig = px.pie(
        df, 
        names='category', 
        values='count',
        title='Distribution of Articles by Category',
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    
    return fig

@app.callback(
    Output('map-storage', 'children'),
    [Input('geo-article-dropdown', 'value'),
     Input('update-map-button', 'n_clicks')]
)
def generate_map(article, n_clicks):
    if not article:
        return ""
    
    # Get country data
    country_data = fetch_pageviews_by_country(article)
    
    # Create a simplified map - reduced detail for better performance
    m = folium.Map(location=[20, 0], zoom_start=2, prefer_canvas=True, control_scale=True, max_bounds=True)
    
    # Add country markers with simplified rendering
    for country, views in country_data.items():
        # Get approximate country coordinates (in a real app, use a geocoding API or database)
        coords = {
            "United States": [38.0, -97.0],
            "United Kingdom": [54.0, -2.0],
            "India": [20.0, 77.0],
            "Germany": [51.0, 10.0],
            "France": [46.0, 2.0],
            "Canada": [56.0, -96.0],
            "Australia": [-25.0, 135.0],
            "Japan": [36.0, 138.0],
            "Brazil": [-10.0, -55.0],
            "Mexico": [23.0, -102.0]
        }
        
        if country in coords:
            folium.CircleMarker(
                location=coords[country],
                radius=min(views / 2000, 50),  # Increased base size and maximum
                popup=f"{country}: {views} views",
                tooltip=f"{country}: {views:,} views",  # Added tooltip for hover info
                color='crimson',
                fill=True,
                fill_color='crimson',
                fill_opacity=0.7,  # Added fill opacity to make circles more visible
                weight=2,  # Increased border weight
                opacity=0.8  # Added border opacity
            ).add_to(m)
    
    # Save to HTML with reduced JS dependencies
    timestamp = datetime.now().timestamp()
    map_path = os.path.join("assets", f"map_{timestamp}.html")
    os.makedirs("assets", exist_ok=True)
    m.save(map_path)
    
    return map_path

@app.callback(
    Output('map-iframe', 'src'),
    Input('map-storage', 'children')
)
def update_map_iframe(map_path):
    if not map_path:
        return ""
    return map_path

@app.callback(
    Output('device-distribution-graph', 'figure'),
    Input('device-article-dropdown', 'value')
)
def update_device_distribution_graph(article):
    if not article:
        return {}
    
    # Get device data
    device_data = fetch_pageviews_by_device(article)
    df = pd.DataFrame(list(device_data.items()), columns=['Device', 'Views'])
    
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "pie"}, {"type": "bar"}]],
        subplot_titles=("Percentage Distribution", "Absolute Views")
    )
    
    # Add pie chart
    fig.add_trace(
        go.Pie(
            labels=df['Device'],
            values=df['Views'],
            textinfo='percent+label',
            hole=0.4
        ),
        row=1, col=1
    )
    
    # Add bar chart with proper naming to fix legend
    colors = ['#636EFA', '#EF553B', '#00CC96']
    for i, (device, views) in enumerate(zip(df['Device'], df['Views'])):
        fig.add_trace(
            go.Bar(
                x=[device],
                y=[views],
                name=device,
                marker_color=colors[i % len(colors)]
            ),
            row=1, col=2
        )
    
    fig.update_layout(
        title_text=f"Device Distribution for '{article}'",
        height=500,
        showlegend=False,  # Hide redundant legend
    )
    
    return fig

# Add callbacks to populate dropdown options efficiently
@app.callback(
    [Output('geo-article-dropdown', 'options'),
     Output('device-article-dropdown', 'options')],
    [Input('titles-store', 'data')]
)
def set_dropdown_options(options):
    return options, options

# Add callback for search functionality
@app.callback(
    [Output('geo-article-dropdown', 'options', allow_duplicate=True),
     Output('device-article-dropdown', 'options', allow_duplicate=True)],
    [Input('geo-article-dropdown', 'search_value'),
     Input('device-article-dropdown', 'search_value')],
    [State('all-titles-store', 'data')],
    prevent_initial_call=True
)
def update_options(geo_search, device_search, all_titles):
    ctx = callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    search_term = geo_search if trigger_id == 'geo-article-dropdown' else device_search
    
    if not search_term or len(search_term) < 2:
        # Default options (first 200)
        options = [{'label': title, 'value': title} for title in all_titles[:200]]
    else:
        # Filter based on search term
        filtered_titles = [title for title in all_titles if search_term.lower() in title.lower()][:100]
        options = [{'label': title, 'value': title} for title in filtered_titles]
    
    if trigger_id == 'geo-article-dropdown':
        return options, dash.no_update
    else:
        return dash.no_update, options

if __name__ == '__main__':
    print("Starting Wikipedia Analysis Dashboard")
    print("Dashboard will be available at: http://127.0.0.1:8050")
    print("Press Ctrl+C to stop the server")
    
    # Create required directories
    os.makedirs("assets", exist_ok=True)
    
    try:
        # Run the app
        app.run_server(debug=True, port=8050)
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    except Exception as e:
        print(f"Error starting dashboard: {e}")
        print("Please make sure all dependencies are installed:")
        print("conda install -c conda-forge plotly dash folium pandas numpy") 