"""Utility functions for CSV handling and data processing."""

from typing import Any
import io

import pandas as pd
import streamlit as st
import numpy as np


class CSVHandler:
    """Handles CSV file upload and processing."""

    def __init__(self) -> None:
        """Initialize the CSV handler."""
        self.supported_separators = {
            "Comma (,)": ",",
            "Semicolon (;)": ";",
            "Tab": "\t",
            "Pipe (|)": "|",
        }

    def create_upload_widget(
        self, 
        key: str = "csv_upload"
    ) -> pd.DataFrame | None:
        """Create CSV upload widget with options.
        
        Args:
            key: Unique key for the widget
            
        Returns:
            Loaded DataFrame or None
        """
        st.subheader("📁 Upload CSV Data")

        if st.selectbox(
            "How would you like to provide your data?",
            [ "Use example dataset","Upload CSV file"]
        ) == "Use example dataset":
            # Load example dataset
            uploaded_file = pd.read_csv("examples/sampled_molecules_descriptors.csv")
            return uploaded_file
        else:
            uploaded_file = st.file_uploader(
                "Choose a CSV file",
                type=["csv", "txt"],
                key=key,
                help="Upload a CSV file with your experimental data"

            )
        
        if uploaded_file is not None:
            return self._process_uploaded_file(uploaded_file)
        
        return None

    def _process_uploaded_file(self, uploaded_file) -> pd.DataFrame | None:
        """Process the uploaded CSV file with options."""
        try:
            # Read file content
            content = uploaded_file.getvalue()
            
            # Auto-detect encoding
            try:
                content_str = content.decode('utf-8')
            except UnicodeDecodeError:
                content_str = content.decode('latin-1')
            
            # Show file preview
            st.write("**File Preview (first 5 lines):**")
            preview_lines = content_str.split('\n')[:5]
            st.code('\n'.join(preview_lines))
            
            # CSV parsing options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                separator = st.selectbox(
                    "Separator",
                    list(self.supported_separators.keys()),
                    help="Choose the column separator"
                )
                sep_char = self.supported_separators[separator]
            
            with col2:
                header_row = st.number_input(
                    "Header Row",
                    min_value=0,
                    max_value=10,
                    value=0,
                    help="Row number containing column headers (0-indexed)"
                )
            
            with col3:
                skip_rows = st.number_input(
                    "Skip Rows",
                    min_value=0,
                    max_value=20,
                    value=0,
                    help="Number of rows to skip at the beginning"
                )
            
            # Parse CSV
            df = pd.read_csv(
                io.StringIO(content_str),
                sep=sep_char,
                header=header_row if header_row >= 0 else None,
                skiprows=skip_rows
            )
            
            # Show basic info
            #
            
            return df
            
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            return None

    def _display_dataframe_info(self, df: pd.DataFrame) -> None:
        """Display basic information about the DataFrame."""
        st.success(f"✅ Successfully loaded CSV with {len(df)} rows and {len(df.columns)} columns")
        
        # Show column info
        with st.expander("📊 Dataset Info"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Columns:**")
                for i, col in enumerate(df.columns):
                    dtype = df[col].dtype
                    non_null = df[col].count()
                    st.write(f"{i+1}. `{col}` ({dtype}) - {non_null} non-null")
            
            with col2:
                st.write("**Sample Data:**")
                st.dataframe(df.head(3), use_container_width=True)


class DataProcessor:
    """Handles data processing and cleaning operations."""

    @staticmethod
    def get_numeric_columns(df: pd.DataFrame) -> list[str]:
        """Get list of numeric columns.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            List of numeric column names
        """
        return df.select_dtypes(include=[np.number]).columns.tolist()

    @staticmethod
    def get_categorical_columns(
        df: pd.DataFrame, 
        max_unique: int = 50
    ) -> list[str]:
        """Get list of categorical columns.
        
        Args:
            df: DataFrame to analyze
            max_unique: Maximum unique values to consider categorical
            
        Returns:
            List of categorical column names
        """
        categorical_cols = []
        for col in df.columns:
            if (
                df[col].dtype == 'object' or 
                df[col].nunique() <= max_unique
            ):
                categorical_cols.append(col)
        return categorical_cols

    @staticmethod
    def get_text_columns(df: pd.DataFrame) -> list[str]:
        """Get list of text/string columns.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            List of text column names
        """
        return df.select_dtypes(include=['object', 'string']).columns.tolist()

    @staticmethod
    def detect_molecule_name_column(df: pd.DataFrame) -> str | None:
        """Attempt to detect which column contains molecule names.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Column name that likely contains molecule names, or None
        """
        # Common column names for molecules
        molecule_keywords = [
            'molecule', 'mol', 'name', 'id', 'compound', 'smiles', 
            'structure', 'csd', 'ccdc', 'entry', 'label'
        ]
        
        # Check column names
        for col in df.columns:
            col_lower = col.lower()
            for keyword in molecule_keywords:
                if keyword in col_lower:
                    return col
        
        # Check for string columns with alphanumeric values
        text_cols = DataProcessor.get_text_columns(df)
        for col in text_cols:
            sample_values = df[col].dropna().head(10)
            if len(sample_values) > 0:
                # Check if values look like molecule identifiers
                alphanumeric_count = sum(
                    1 for val in sample_values 
                    if isinstance(val, str) and val.replace('-', '').replace('_', '').isalnum()
                )
                if alphanumeric_count >= len(sample_values) * 0.8:  # 80% alphanumeric
                    return col
        
        return None

    @staticmethod
    def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Basic data cleaning operations.
        
        Args:
            df: DataFrame to clean
            
        Returns:
            Cleaned DataFrame
        """
        # Create a copy
        df_clean = df.copy()
        
        # Remove completely empty rows and columns
        df_clean = df_clean.dropna(how='all').dropna(axis=1, how='all')
        
        # Strip whitespace from string columns
        text_cols = DataProcessor.get_text_columns(df_clean)
        for col in text_cols:
            df_clean[col] = df_clean[col].astype(str).str.strip()
        
        return df_clean

    @staticmethod
    def get_column_stats(df: pd.DataFrame, column: str) -> dict[str, Any]:
        """Get statistics for a specific column.
        
        Args:
            df: DataFrame
            column: Column name
            
        Returns:
            Dictionary with column statistics
        """
        stats = {}
        col_data = df[column]
        
        stats['dtype'] = str(col_data.dtype)
        stats['count'] = len(col_data)
        stats['non_null'] = col_data.count()
        stats['null_count'] = col_data.isnull().sum()
        stats['unique_count'] = col_data.nunique()
        
        if pd.api.types.is_numeric_dtype(col_data):
            stats.update({
                'mean': col_data.mean(),
                'std': col_data.std(),
                'min': col_data.min(),
                'max': col_data.max(),
                'median': col_data.median()
            })
        else:
            # For non-numeric columns
            value_counts = col_data.value_counts()
            stats.update({
                'most_common': value_counts.index[0] if not value_counts.empty else None,
                'most_common_count': value_counts.iloc[0] if not value_counts.empty else 0,
                'sample_values': col_data.dropna().head(5).tolist()
            })
        
        return stats


class ConfigManager:
    """Manages app configuration and settings."""

    @staticmethod
    def get_plot_config_widget(df: pd.DataFrame) -> dict[str, Any]:
        """Create widget for plot configuration.
        
        Args:
            df: DataFrame with data
            
        Returns:
            Plot configuration dictionary
        """
        st.subheader("📊 Plot Configuration")
        
        # Get column types
        numeric_cols = DataProcessor.get_numeric_columns(df)
        categorical_cols = DataProcessor.get_categorical_columns(df)
        all_cols = df.columns.tolist()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            plot_type = st.selectbox(
                "Plot Type",
                ["scatter", "line", "histogram", "box", "parity"],
                help="Choose the type of plot to create"
            )
        
        with col2:
            if plot_type in ["scatter", "line", "parity"]:
                x_col = st.selectbox(
                    "X-axis Column",
                    numeric_cols if numeric_cols else all_cols,
                    help="Choose column for X-axis"
                )
        
        # Y-axis selection
        with col3:
            if plot_type in ["scatter", "line", "box", "parity"]:
                numeric_cols_y = [col for col in numeric_cols if col != x_col]
                y_col = st.selectbox(
                    "Y-axis Column", 
                    numeric_cols_y if numeric_cols_y else all_cols,
                    help="Choose column for Y-axis"
                )
            elif plot_type == "histogram":
                y_col = st.selectbox(
                    "Column to Plot",
                    numeric_cols if numeric_cols else all_cols,
                    help="Choose column for histogram"
                )
        
        # Optional styling columns
        color_col = None
        size_col = None
        
        if plot_type == "scatter":
            col1, col2 = st.columns(2)
            with col1:
                if categorical_cols:
                    use_color = st.checkbox("Color by column")
                    if use_color:
                        color_col = st.selectbox(
                            "Color Column",
                            [None] + categorical_cols,
                            help="Choose column for color coding"
                        )
            
            with col2:
                if numeric_cols:
                    use_size = st.checkbox("Size by column")
                    if use_size:
                        size_col = st.selectbox(
                            "Size Column",
                            [None] + numeric_cols,
                            help="Choose column for point sizes"
                        )
        
        # Build configuration
        config = {
            "type": plot_type,
        }
        
        if plot_type in ["scatter", "line"]:
            config.update({
                "x_col": x_col,
                "y_col": y_col,
                "color_col": color_col,
                "size_col": size_col if plot_type == "scatter" else None
            })
        elif plot_type == "histogram":
            config["col"] = y_col
            if categorical_cols:
                use_color = st.checkbox("Group by column")
                if use_color:
                    config["color_col"] = st.selectbox(
                        "Group Column",
                        [None] + categorical_cols
                    )
        elif plot_type == "box":
            config["y_col"] = y_col
            if categorical_cols:
                use_x = st.checkbox("Group by X-axis")
                if use_x:
                    config["x_col"] = st.selectbox(
                        "X-axis (Grouping)",
                        [None] + categorical_cols
                    )
        elif plot_type == "parity":
            config.update({
                "true_col": x_col,
                "pred_col": y_col
            })
        
        return config

    @staticmethod  
    def get_molecule_config_widget(df: pd.DataFrame) -> dict[str, Any]:
        """Create widget for molecule visualization configuration.
        
        Args:
            df: DataFrame with data
            
        Returns:
            Molecule configuration dictionary  
        """
        st.subheader("🧬 Molecule Visualization")
        
        config = {"enabled": False}
        
        # Try to detect molecule column
        suggested_col = DataProcessor.detect_molecule_name_column(df)
        
        enable_mol_viz = st.checkbox(
            "Enable molecule visualization",
            help="Show molecular structures when clicking on plot points"
        )
        
        if enable_mol_viz:
            text_cols = DataProcessor.get_text_columns(df)
            if not text_cols:
                st.warning("No text columns found for molecule names")
                return config
            
            default_idx = 0
            if suggested_col and suggested_col in text_cols:
                default_idx = text_cols.index(suggested_col)
            
            molecule_col = st.selectbox(
                "Molecule Name Column",
                text_cols,
                index=default_idx,
                help="Column containing molecule names/identifiers"
            )
            
            #if suggested_col:
             #   st.info(f"💡 Detected likely molecule column: `{suggested_col}`")
            
            config.update({
                "enabled": True,
                "molecule_col": molecule_col
            })
        
        return config


def fix_dataframe_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Fix DataFrame for Streamlit display compatibility.
    
    Args:
        df: DataFrame to fix
        
    Returns:
        Fixed DataFrame
    """
    df_fixed = df.copy()
    
    for col in df_fixed.columns:
        # Handle numeric columns with potential issues
        if df_fixed[col].dtype in ["float64", "float32"]:
            # Replace infinities with NaN
            df_fixed[col] = df_fixed[col].replace([np.inf, -np.inf], np.nan)
        
        elif df_fixed[col].dtype == "object":
            # Convert to string and clean up
            df_fixed[col] = df_fixed[col].astype(str)
            df_fixed[col] = df_fixed[col].replace(["nan", "None", "<NA>"], "")
    
    return df_fixed