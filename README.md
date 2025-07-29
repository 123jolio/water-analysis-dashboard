# Water Analysis Dashboard

An interactive Streamlit application for analyzing water quality and climate data with advanced statistical methods and visualizations.

## Features

- Time series analysis with interactive plots
- Correlation analysis (Pearson, Spearman, Kendall)
- Time-varying correlation analysis
- Non-linear relationship analysis
- Interrupted Time Series (ITS) analysis
- Advanced visualizations with Plotly
- Data export capabilities

## Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/water-analysis-dashboard.git
   cd water-analysis-dashboard
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # On Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the app locally:
   ```bash
   streamlit run Perifereia.py
   ```

## Deployment

This app can be easily deployed to Streamlit Cloud:

1. Push your code to a GitHub repository
2. Go to [Streamlit Cloud](https://share.streamlit.io/)
3. Click "New app" and connect your GitHub repository
4. Select the main branch and set the main file to `Perifereia.py`
5. Click "Deploy!"

## Project Structure

```
.
├── Perifereia.py          # Main application
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── .streamlit/            # Streamlit config
    └── config.toml        # Configuration
```

## License

This project is licensed under the MIT License.
