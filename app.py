"""Lightweight Molecular Visualization App - Main Application."""

import streamlit as st
import pandas as pd

from core.plotting import InteractivePlotter, create_plot_from_config
from core.molecule_viz import get_database_selector, display_molecule_from_csv_selection
from core.utils import (
    CSVHandler,
    DataProcessor,
    ConfigManager,
    fix_dataframe_for_display,
)
from core.csv_to_ase import csv_to_ase_interface


class LiteVizApp:
    """Main application class for the lightweight visualization app."""

    def __init__(self) -> None:
        """Initialize the app."""
        self.csv_handler = CSVHandler()
        self.plotter = InteractivePlotter()
        if "selected_molecules_dict" not in st.session_state:
            st.session_state.selected_molecules_dict = {}

        # Initialize session state
        if "df" not in st.session_state:
            st.session_state.df = None
        if "selected_point" not in st.session_state:
            st.session_state.selected_point = None

    def run(self) -> None:
        """Run the main application."""
        st.set_page_config(
            page_title="Lite Molecular Viz",
            page_icon="🧬",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        st.title("🧬 Lightweight Molecular Visualization App")
        st.markdown(
            "Upload your CSV data and create interactive plots with integrated molecular visualization"
        )

        # Sidebar configuration
        self._setup_sidebar()

        # Main content
        with st.expander("📂 Upload Data", expanded=True):
            st.session_state.df = self._handle_data_upload()

        if st.session_state.df is not None:
            CSVHandler()._display_dataframe_info(st.session_state.df)
            self._show_main_interface(st.session_state.df)
        else:
            self._show_welcome_screen()

    def _setup_sidebar(self) -> None:
        """Set up the sidebar configuration."""
        st.sidebar.title("⚙️ Configuration")

        # Database selection for molecule visualization
        self.database_url = get_database_selector()

        # App information
        with st.sidebar.expander("ℹ️ About This App"):
            st.markdown("""
            **Lightweight Molecular Visualization App**
            
            - Upload CSV files with experimental data
            - Create interactive plots from any columns  
            - Visualize molecules when CSV contains names
            - Convert CSV + XYZ files to ASE databases
            - Modular design for easy extension
            
            **Usage:**
            1. Upload your CSV file
            2. Configure plot settings
            3. Click points to see molecules (if enabled)
            4. Create ASE databases from your data
            
            **Features:**
            - Multiple plot types (scatter, line, histogram, etc.)
            - Automatic molecule name detection
            - 3D molecular visualization with ASE
            - Distance measurement between atoms
            - CSV to ASE database conversion
            - Support for ZIP files with XYZ structures
            """)

    def _handle_data_upload(self) -> pd.DataFrame | None:
        """Handle CSV file upload and processing."""
        df = self.csv_handler.create_upload_widget("main_csv")

        if df is not None:
            # Clean the data
            df = DataProcessor.clean_dataframe(df)
            st.session_state.df = df
            return df

        return st.session_state.df

    def _show_welcome_screen(self) -> None:
        """Show welcome screen when no data is loaded."""
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            st.markdown("""
            ### 👋 Welcome!
            
            To get started:
            
            1. **Upload a CSV file** using the file uploader above
            2. **Configure your plot** using the sidebar options  
            3. **Explore your data** with interactive visualizations
            4. **View molecules** by clicking on plot points (if your data contains molecule names)
            
            #### Sample Data Format:
            Your CSV should contain columns with experimental data. If you have molecule names, 
            the app will automatically detect them and enable 3D visualization.
            
            | molecule_name | property_1 | property_2 | ... |
            |---------------|------------|------------|-----|
            | ALANIN01      | 1.23       | 4.56       | ... |
            | GLYCIN02      | 2.34       | 5.67       | ... |
            """)

    def _show_main_interface(self, df: pd.DataFrame) -> None:
        """Show the main interface with data and plots."""
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(
            ["📊 Interactive Plots", "📋 Data Explorer", "🔄 CSV to ASE Database"]
        )

        with tab1:
            self._show_plotting_interface(df)

        with tab2:
            self._show_data_explorer(df)

        with tab3:
            self._show_csv_to_ase_interface()

        # with tab3:
        #     self._show_advanced_features(df)

    def _show_csv_to_ase_interface(self) -> None:
        """Show the CSV to ASE database conversion interface."""
        csv_to_ase_interface()

    def _show_plotting_interface(self, df: pd.DataFrame) -> None:
        """Show the interactive plotting interface."""

        # Get plot configuration
        plot_config = ConfigManager.get_plot_config_widget(df)

        # Get molecule visualization configuration
        mol_config = ConfigManager.get_molecule_config_widget(df)

        # if st.button("Create Plot", type="primary"):
        self._create_and_display_plot(df, plot_config, mol_config)

    def _create_and_display_plot(
        self, df: pd.DataFrame, plot_config: dict, mol_config: dict
    ) -> None:
        """Create and display the plot with optional molecule visualization."""
        # try:
        # Add molecule names to hover data if molecule viz is enabled
        if mol_config.get("enabled") and mol_config.get("molecule_col"):
            hover_data = plot_config.get("hover_data", [])
            if mol_config["molecule_col"] not in hover_data:
                hover_data.extend(mol_config["molecule_col"])
                plot_config["hover_data"] = hover_data

        # Create the plot
        fig = create_plot_from_config(df, plot_config)

        # Layout for plot and molecule viewer
        if mol_config.get("enabled"):
            col1, col2 = st.columns([1, 1])
        else:
            col1, col2 = st.columns([1, 0.001])  # Full width for plot

        with col1:
            # Display the plot with selection capability
            #def _selection_callback():
             #   return self._handle_plot_selection(selection, df, mol_config, col2)
            event = st.plotly_chart(
                fig,
                use_container_width=True,
                on_select="rerun",
            )

            # Handle point selection for molecule visualization
            if mol_config.get("enabled"):
                self._handle_plot_selection(event, df, mol_config, col2)

        if not mol_config.get("enabled"):
            st.info(
                "💡 Tip: Enable molecule visualization in the sidebar to see 3D structures when clicking plot points!"
            )

        # except Exception as e:
        #     st.error(f"Error creating plot: {e}")
        #     st.write("Please check your column selections and data format.")

    def _handle_plot_selection(
        self, event, df: pd.DataFrame, mol_config: dict, container
    ) -> None:
        """Handle plot point selection for molecule visualization."""
        if not event.get("selection") or not event["selection"].get("points"):
            return

        try:
            # Get selected point
            selected_point = event["selection"]["points"][0]

            # Extract molecule name from the selected point
            # st.write(selected_point)
            molecule_name = selected_point["text"] if "text" in selected_point else None

            #st.write(selected_point)

            if molecule_name and self.database_url:
                with container:
                    st.subheader("🧬 Molecule Structure")
                    st.write(f"**Selected:** {molecule_name}")
                    cols = st.columns(
                        len(molecule_name) if isinstance(molecule_name, list) else 1
                    )
                    for i, _molecule_name in enumerate(
                        molecule_name
                        if isinstance(molecule_name, list)
                        else [molecule_name]
                    ):
                        with cols[i]:
                            st.write(f"Displaying molecule: {_molecule_name}")
                            display_molecule_from_csv_selection(
                                self.database_url,
                                str(_molecule_name),
                                container=None,  # Already in container context
                                key_prefix="_" + str(_molecule_name),
                            )
            elif not self.database_url:
                with container:
                    st.warning("⚠️ Please configure a molecular database in the sidebar")

        except Exception as e:
            with container:
                st.error(f"Error displaying molecule: {e}")

    def _show_data_explorer(self, df: pd.DataFrame) -> None:
        """Show the data exploration interface."""
        st.subheader("📋 Data Explorer")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Dataset Overview**")
            st.write(f"- **Rows:** {len(df):,}")
            st.write(f"- **Columns:** {len(df.columns)}")

            # Column types
            numeric_cols = DataProcessor.get_numeric_columns(df)
            categorical_cols = DataProcessor.get_categorical_columns(df)
            text_cols = DataProcessor.get_text_columns(df)

            st.write(f"- **Numeric columns:** {len(numeric_cols)}")
            st.write(f"- **Text columns:** {len(text_cols)}")
            st.write(f"- **Categorical columns:** {len(categorical_cols)}")

            # Molecule detection
            mol_col = DataProcessor.detect_molecule_name_column(df)
            if mol_col:
                st.success(f"🧬 Detected molecule column: `{mol_col}`")
            else:
                st.info("No molecule name column detected")

        with col2:
            st.write("**Column Details**")
            selected_col = st.selectbox(
                "Select column to analyze",
                df.columns.tolist(),
                help="Choose a column to see detailed statistics",
            )

            if selected_col:
                stats = DataProcessor.get_column_stats(df, selected_col)

                for key, value in stats.items():
                    if key == "sample_values" and value:
                        st.write(f"**{key}:** {', '.join(map(str, value[:3]))}")
                    elif isinstance(value, float):
                        st.write(f"**{key}:** {value:.3f}")
                    else:
                        st.write(f"**{key}:** {value}")

        # Data preview
        st.write("**Data Preview**")

        # Sampling for large datasets
        max_display_rows = 1000
        if len(df) > max_display_rows:
            st.info(
                f"Showing first {max_display_rows:,} rows of {len(df):,} total rows"
            )
            display_df = df.head(max_display_rows)
        else:
            display_df = df

        # Fix dataframe for display
        display_df = fix_dataframe_for_display(display_df)

        st.dataframe(display_df, use_container_width=True, height=400)

    def _show_advanced_features(self, df: pd.DataFrame) -> None:
        """Show advanced features and customization options."""
        st.subheader("🔧 Advanced Features")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Data Processing**")

            if st.button("🧹 Clean Data"):
                original_shape = df.shape
                cleaned_df = DataProcessor.clean_dataframe(df)
                st.session_state.df = cleaned_df

                if cleaned_df.shape != original_shape:
                    st.success(f"Data cleaned: {original_shape} → {cleaned_df.shape}")
                    st.rerun()
                else:
                    st.info("No cleaning needed - data is already clean")

            if st.button("📊 Generate Summary Report"):
                self._generate_data_summary(df)

        with col2:
            st.write("**Export Options**")

            # Download processed data
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name="processed_data.csv",
                mime="text/csv",
            )

            # Sample configuration export
            if st.button("⚙️ Export Configuration"):
                self._export_configuration(df)

    def _generate_data_summary(self, df: pd.DataFrame) -> None:
        """Generate a comprehensive data summary."""
        st.write("**Data Summary Report**")

        summary = {}

        # Basic info
        summary["Dataset Info"] = {
            "Total rows": len(df),
            "Total columns": len(df.columns),
            "Memory usage": f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB",
        }

        # Column analysis
        numeric_cols = DataProcessor.get_numeric_columns(df)
        text_cols = DataProcessor.get_text_columns(df)

        summary["Column Types"] = {
            "Numeric": len(numeric_cols),
            "Text": len(text_cols),
            "Other": len(df.columns) - len(numeric_cols) - len(text_cols),
        }

        # Missing data
        missing_data = df.isnull().sum()
        missing_cols = missing_data[missing_data > 0]

        summary["Data Quality"] = {
            "Columns with missing data": len(missing_cols),
            "Total missing values": missing_data.sum(),
            "Missing data percentage": f"{(missing_data.sum() / (len(df) * len(df.columns))) * 100:.2f}%",
        }

        for section, data in summary.items():
            st.write(f"**{section}:**")
            for key, value in data.items():
                st.write(f"- {key}: {value}")
            st.write("")

    def _export_configuration(self, df: pd.DataFrame) -> None:
        """Export app configuration for reproducibility."""
        config = {
            "app_version": "1.0.0",
            "dataset_info": {
                "columns": df.columns.tolist(),
                "shape": df.shape,
                "dtypes": df.dtypes.to_dict(),
            },
            "suggested_molecule_column": DataProcessor.detect_molecule_name_column(df),
            "numeric_columns": DataProcessor.get_numeric_columns(df),
            "text_columns": DataProcessor.get_text_columns(df),
        }

        import json

        config_json = json.dumps(config, indent=2, default=str)

        st.download_button(
            label="📥 Download Configuration",
            data=config_json,
            file_name="app_config.json",
            mime="application/json",
        )


def main() -> None:
    """Main entry point."""
    app = LiteVizApp()
    app.run()


if __name__ == "__main__":
    main()
