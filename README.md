# Lightweight Molecular Visualization App

A modular, extensible Streamlit application for interactive molecular data analysis and visualization.

## Features

- 📊 **Interactive Plotting**: Multiple plot types (scatter, line, histogram, box, parity) with Plotly
- 🧬 **Molecule Visualization**: 3D molecular structures using ASE database integration  
- 📁 **CSV Data Upload**: Easy data import with automatic column detection
- 🔌 **Plugin Architecture**: Extensible system for adding new analysis capabilities
- 📈 **Statistical Analysis**: Built-in correlation analysis and summary statistics

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

3. **Upload Your Data**:
   - Use the file uploader to select a CSV file
   - The app will automatically detect column types
   - Select columns for plotting and analysis

## Project Structure

```
lite_viz_app/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── core/                       # Core functionality modules
│   ├── plotting.py             # Interactive plotting utilities
│   ├── molecule_viz.py         # 3D molecular visualization
│   └── utils.py               # CSV handling and data processing
├── plugins/                    # Extensible plugin system
│   ├── base_plugin.py          # Plugin architecture
│   └── statistics_plugin.py    # Statistical analysis plugin
└── examples/                   # Sample data and usage examples
    ├── sample_data.csv         # Example dataset
    └── usage_examples.py       # Code demonstrations
```

## Usage Examples

### Basic Plotting
Upload a CSV file and create interactive plots:
- **Scatter plots**: Explore relationships between variables
- **Parity plots**: Compare predicted vs experimental values  
- **Histograms**: View data distributions
- **Box plots**: Compare groups and identify outliers

### Molecule Visualization
If your CSV contains molecule names:
1. The app will detect molecule name columns
2. Click on data points in plots to visualize 3D structures
3. Supports ASE database connectivity for structure lookup



## Extension Path

This lightweight app provides a foundation for building more complex molecular analysis tools:

1. **Additional Plot Types**: 3D scatter, parallel coordinates, radar charts
2. **Advanced Analysis**: Machine learning, charge distribution, descriptor calculations  
3. **Data Integration**: Multiple file formats, database connectivity, API integration
4. **Enhanced Visualization**: Animation, property-based styling, charge overlays
5. **Export/Reporting**: PDF generation, batch processing, configuration management

Each feature can be added as a plugin without modifying the core application!

## Migration from Full Version

This app provides a simplified interface based on the full EXChargeGNN analysis suite:
- **Lighter**: Focused on essential plotting and visualization
- **Modular**: Plugin-based architecture for extensibility  
- **Interactive**: Streamlit web interface vs. command-line tools
- **Accessible**: No complex configuration or model training required

The plugin system provides a clear upgrade path to incorporate advanced features from the full version as needed.
