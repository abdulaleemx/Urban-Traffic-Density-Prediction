import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from model import TrafficModel
from utils import preprocess_data, load_sample_data

# Page configuration
st.set_page_config(
    page_title="Traffic Density Predictor",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def show_data_format_guide():
    st.info("""
    ### Required Data Format
    Your CSV file should include these columns:
    - Time of Day: (0-23) Hour of the day
    - Day of Week: Monday through Sunday
    - Weather: Clear, Rain, Snow, or Cloudy
    - Temperature: Value in Celsius
    - Special Event: True or False

    Download our sample CSV format for reference.
    """)

def validate_data(df):
    required_columns = ['time_of_day', 'day_of_week', 'weather', 'temperature', 'special_event']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        show_data_format_guide()
        return False
    return True

def main():
    # Header section
    st.title("Traffic Density Predictor")

    # Welcome card
    st.markdown("""
    <div class="card">
        <h2>Welcome to the Traffic Density Prediction System</h2>
        <p>This advanced analytics tool helps you:</p>
        <ul>
            <li>Predict urban traffic density</li>
            <li>Visualize traffic patterns</li>
            <li>Analyze contributing factors</li>
            <li>Access detailed performance metrics</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Initialize model
    with st.spinner('Initializing prediction model...'):
        model = TrafficModel()

    # Sidebar
    st.sidebar.header("Prediction Controls")
    input_method = st.sidebar.radio(
        "Select Input Method",
        ["Upload Data", "Sample Data", "Manual Input"]
    )

    data = None
    if input_method == "Upload Data":
        st.sidebar.subheader("Data Upload")
        show_data_format_guide()
        uploaded_file = st.sidebar.file_uploader(
            "Select CSV file",
            type=['csv'],
            help="Upload your CSV file with traffic data"
        )

        if uploaded_file is not None:
            try:
                with st.spinner('Processing data file...'):
                    data = pd.read_csv(uploaded_file)
                if not validate_data(data):
                    data = None
                else:
                    st.success("Data successfully uploaded")
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                data = None

    elif input_method == "Sample Data":
        with st.spinner('Loading sample dataset...'):
            data = load_sample_data()
        st.success("Sample data loaded")

    else:  # Manual Input
        st.sidebar.subheader("Parameter Input")
        with st.sidebar.form("manual_input"):
            col1, col2 = st.columns(2)

            with col1:
                time_of_day = st.slider(
                    "Time of Day",
                    0, 23, 12,
                    help="Select hour (24-hour format)"
                )
                weather = st.selectbox(
                    "Weather",
                    ['Clear', 'Rain', 'Snow', 'Cloudy'],
                    help="Select weather condition"
                )

            with col2:
                day_of_week = st.selectbox(
                    "Day of Week",
                    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                    help="Select day of week"
                )
                temperature = st.slider(
                    "Temperature (°C)",
                    -10, 40, 20,
                    help="Select temperature"
                )

            special_event = st.checkbox(
                "Special Event",
                help="Check if there's a special event"
            )

            submit_button = st.form_submit_button("Generate Prediction")

            if submit_button:
                data = pd.DataFrame({
                    'time_of_day': [time_of_day],
                    'day_of_week': [day_of_week],
                    'weather': [weather],
                    'temperature': [temperature],
                    'special_event': [special_event]
                })

    # Results section
    if data is not None:
        # Preprocess and predict
        with st.spinner('Processing data...'):
            processed_data = preprocess_data(data)
            predictions = model.predict(processed_data)

        st.header("Prediction Results")

        # Results content
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            # Traffic density visualization
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=predictions.mean(),
                title={'text': "Average Traffic Density"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#3498DB"},
                    'steps': [
                        {'range': [0, 33], 'color': "#E8F8F5"},
                        {'range': [33, 66], 'color': "#D4E6F1"},
                        {'range': [66, 100], 'color': "#D6EAF8"}
                    ],
                    'threshold': {
                        'line': {'color': "#2980B9", 'width': 4},
                        'thickness': 0.75,
                        'value': predictions.mean()
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Model Performance Metrics")
            metrics = model.get_metrics()
            cols = st.columns(3)
            for i, (metric, value) in enumerate(metrics.items()):
                with cols[i]:
                    st.metric(
                        metric,
                        f"{value:.2f}",
                        help=f"Model performance metric: {metric}"
                    )
            st.markdown('</div>', unsafe_allow_html=True)

        # Analysis section
        if len(predictions) > 1:
            st.header("Traffic Analysis")
            st.markdown('<div class="viz-container">', unsafe_allow_html=True)

            # Create time indices for better x-axis representation
            if len(data) == 24:  # If we have 24 hours of data
                x_labels = [f"{hour:02d}:00" for hour in range(24)]
            else:
                x_labels = [f"Period {i+1}" for i in range(len(predictions))]

            # Create the line chart with enhanced features
            fig = go.Figure()

            # Add the main line with points
            fig.add_trace(go.Scatter(
                x=x_labels,
                y=predictions,
                mode='lines+markers',
                name='Traffic Density',
                line=dict(color='#3498DB', width=2),
                marker=dict(size=8),
                hovertemplate='Time: %{x}<br>Density: %{y:.1f}%<extra></extra>'
            ))

            # Add trend line
            z = np.polyfit(range(len(predictions)), predictions, 1)
            p = np.poly1d(z)
            fig.add_trace(go.Scatter(
                x=x_labels,
                y=p(range(len(predictions))),
                mode='lines',
                name='Trend',
                line=dict(color='rgba(255, 0, 0, 0.3)', dash='dash'),
                hovertemplate='Trend: %{y:.1f}%<extra></extra>'
            ))

            # Update layout for better readability
            fig.update_layout(
                title={
                    'text': 'Traffic Density Pattern Over Time',
                    'y':0.95,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                xaxis_title='Time Period',
                yaxis_title='Traffic Density (%)',
                hovermode='x unified',
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                ),
                plot_bgcolor='white',
                height=400
            )

            # Add grid lines for better readability
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

            st.plotly_chart(fig, use_container_width=True)

            # Add explanatory text
            st.markdown("""
                <div style='margin-top: 1rem;'>
                    <h4>Understanding the Traffic Pattern:</h4>
                    <ul>
                        <li>The blue line shows actual traffic density values</li>
                        <li>Red dashed line indicates the overall trend</li>
                        <li>Hover over points to see exact values</li>
                        <li>Higher percentages indicate heavier traffic</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Feature Impact section
        st.header("Feature Impact Analysis")
        st.markdown('<div class="viz-container">', unsafe_allow_html=True)
        importance_df = model.get_feature_importance()
        fig = px.bar(
            importance_df,
            x='importance',
            y='feature',
            orientation='h',
            title="Factors Influencing Traffic Density",
            labels={'importance': 'Impact', 'feature': 'Factor'}
        )
        fig.update_traces(marker_color='#3498DB')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
            <p>This visualization shows the relative importance of different factors in traffic density predictions.
            Longer bars indicate stronger influence on the prediction outcome.</p>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()