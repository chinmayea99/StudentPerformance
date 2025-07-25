import dash
from dash import dcc, html, Input, Output, callback
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Initialize Dash app
app = dash.Dash(__name__)

# Sample data loading (replace with actual data loading)
def load_dashboard_data():
    # This would load from your data store
    # For demo purposes, creating sample data
    np.random.seed(42)
    n_students = 1000
    
    df = pd.DataFrame({
        'student_id': [f'STU_{i:04d}' for i in range(n_students)],
        'performance_score': np.random.normal(75, 15, n_students),
        'engagement_level': np.random.choice(['Low', 'Medium', 'High'], n_students),
        'course_completion': np.random.uniform(0, 1, n_students),
        'time_spent_hours': np.random.exponential(20, n_students),
        'assignment_score': np.random.normal(80, 12, n_students),
        'participation_score': np.random.normal(70, 18, n_students)
    })
    return df

df = load_dashboard_data()

# App layout
app.layout = html.Div([
    html.Div([
        html.H1("Student Performance Analytics Dashboard", 
                className="header-title"),
        html.P("Real-time insights into student performance and engagement",
               className="header-description")
    ], className="header"),
    
    html.Div([
        # Key metrics cards
        html.Div([
            html.Div([
                html.H3(f"{len(df):,}", className="metric-number"),
                html.P("Total Students", className="metric-label")
            ], className="metric-card"),
            
            html.Div([
                html.H3(f"{df['performance_score'].mean():.1f}", className="metric-number"),
                html.P("Avg Performance", className="metric-label")
            ], className="metric-card"),
            
            html.Div([
                html.H3(f"{(df['engagement_level'] == 'High').sum()}", className="metric-number"),
                html.P("High Engagement", className="metric-label")
            ], className="metric-card"),
            
            html.Div([
                html.H3(f"{df['course_completion'].mean():.1%}", className="metric-number"),
                html.P("Completion Rate", className="metric-label")
            ], className="metric-card")
        ], className="metrics-container"),
        
        # Charts section
        html.Div([
            # Performance distribution
            html.Div([
                dcc.Graph(id="performance-distribution")
            ], className="chart-container"),
            
            # Engagement analysis
            html.Div([
                dcc.Graph(id="engagement-analysis")
            ], className="chart-container")
        ], className="charts-row"),
        
        html.Div([
            # Time spent vs performance scatter
            html.Div([
                dcc.Graph(id="time-performance-scatter")
            ], className="chart-container"),
            
            # Performance trends
            html.Div([
                dcc.Graph(id="performance-trends")
            ], className="chart-container")
        ], className="charts-row"),
        
        # Filters section
        html.Div([
            html.H3("Filters"),
            dcc.Dropdown(
                id='engagement-filter',
                options=[{'label': level, 'value': level} for level in df['engagement_level'].unique()],
                value=df['engagement_level'].unique().tolist(),
                multi=True,
                placeholder="Select engagement levels"
            )
        ], className="filters-section")
        
    ], className="dashboard-content")
])

# Callbacks for interactivity
@app.callback(
    [Output('performance-distribution', 'figure'),
     Output('engagement-analysis', 'figure'),
     Output('time-performance-scatter', 'figure'),
     Output('performance-trends', 'figure')],
    [Input('engagement-filter', 'value')]
)
def update_dashboard(selected_engagement):
    # Filter data
    filtered_df = df[df['engagement_level'].isin(selected_engagement)]
    
    # Performance distribution histogram
    perf_dist = px.histogram(
        filtered_df, 
        x='performance_score', 
        nbins=30,
        title="Performance Score Distribution",
        color_discrete_sequence=['#1f77b4']
    )
    perf_dist.update_layout(xaxis_title="Performance Score", yaxis_title="Count")
    
    # Engagement analysis pie chart
    engagement_counts = filtered_df['engagement_level'].value_counts()
    engagement_pie = px.pie(
        values=engagement_counts.values,
        names=engagement_counts.index,
        title="Engagement Level Distribution"
    )
    
    # Time spent vs performance scatter
    time_perf_scatter = px.scatter(
        filtered_df,
        x='time_spent_hours',
        y='performance_score',
        color='engagement_level',
        title="Time Spent vs Performance Score",
        hover_data=['assignment_score', 'participation_score']
    )
    time_perf_scatter.update_layout(
        xaxis_title="Time Spent (Hours)",
        yaxis_title="Performance Score"
    )
    
    # Performance trends (simulated weekly data)
    weeks = pd.date_range('2024-01-01', periods=12, freq='W')
    weekly_performance = [filtered_df['performance_score'].mean() + np.random.normal(0, 2) for _ in weeks]
    
    trends_fig = go.Figure()
    trends_fig.add_trace(go.Scatter(
        x=weeks,
        y=weekly_performance,
        mode='lines+markers',
        name='Average Performance',
        line=dict(color='#ff7f0e', width=3)
    ))
    trends_fig.update_layout(
        title="Performance Trends Over Time",
        xaxis_title="Week",
        yaxis_title="Average Performance Score"
    )
    
    return perf_dist, engagement_pie, time_perf_scatter, trends_fig

# CSS styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body { margin: 0; font-family: 'Arial', sans-serif; background-color: #f5f5f5; }
            .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; text-align: center; }
            .header-title { margin: 0; font-size: 2.5em; font-weight: bold; }
            .header-description { margin: 10px 0 0 0; font-size: 1.2em; opacity: 0.9; }
            .dashboard-content { padding: 20px; max-width: 1400px; margin: 0 auto; }
            .metrics-container { display: flex; gap: 20px; margin-bottom: 30px; flex-wrap: wrap; }
            .metric-card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); flex: 1; min-width: 200px; text-align: center; }
            .metric-number { margin: 0; font-size: 2.5em; font-weight: bold; color: #667eea; }
            .metric-label { margin: 10px 0 0 0; color: #666; font-size: 1.1em; }
            .charts-row { display: flex; gap: 20px; margin-bottom: 30px; flex-wrap: wrap; }
            .chart-container { background: white; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); flex: 1; min-width: 500px; }
            .filters-section { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)
