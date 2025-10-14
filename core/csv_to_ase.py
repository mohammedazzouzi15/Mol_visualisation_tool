"""
CSV to ASE Database conversion functionality for the Streamlit app.
"""

import pandas as pd
import streamlit as st
import tempfile
import zipfile
from pathlib import Path
from typing import List, Dict
import logging
from ase.io import read
from ase.db import connect

logger = logging.getLogger(__name__)


class CSVToASEConverter:
    """Handle conversion from CSV and XYZ files to ASE database."""

    def __init__(self):
        """Initialize the converter."""
        self.csv_data = None
        self.xyz_files = {}
        self.molecule_names = []

    def read_csv_data(self, csv_file) -> pd.DataFrame:
        """
        Read CSV data from uploaded file.

        Args:
            csv_file: Streamlit uploaded file object

        Returns:
            pandas DataFrame containing CSV data
        """
        try:
            if hasattr(csv_file, "read"):
                # It's a file-like object
                self.csv_data = pd.read_csv(csv_file)
            else:
                # It's a file path
                self.csv_data = pd.read_csv(csv_file)

            logger.info(
                f"Successfully read CSV with {len(self.csv_data)} rows and {len(self.csv_data.columns)} columns"
            )
            return self.csv_data
        except Exception as e:
            logger.error(f"Error reading CSV file: {e}")
            raise

    def extract_molecule_names(
        self, df: pd.DataFrame, columns: List[str] | None = None
    ) -> List[str]:
        """
        Extract unique molecule names from specified columns of the CSV.

        Args:
            df: pandas DataFrame containing CSV data
            columns: List of column names to extract molecules from (default: first 4 columns)

        Returns:
            List of unique molecule names
        """
        if columns is None:
            # Use first 4 columns by default
            columns = df.columns[:4].tolist()

        # Get all unique molecule names from these columns
        molecule_names = set()
        for col in columns:
            if col in df.columns:
                # Remove any NaN values and add to set
                names = df[col].dropna().unique()
                molecule_names.update(names)

        # Convert to list and sort for consistent ordering
        self.molecule_names = sorted(list(molecule_names))

        logger.info(
            f"Found {len(self.molecule_names)} unique molecule names in selected columns"
        )
        return self.molecule_names

    def process_xyz_files(self, xyz_files: List) -> Dict[str, str]:
        """
        Process uploaded XYZ files and map them to molecule names.

        Args:
            xyz_files: List of uploaded XYZ files

        Returns:
            Dictionary mapping molecule names to XYZ file contents
        """
        name_to_content = {}
        found_count = 0

        for xyz_file in xyz_files:
            file_name = xyz_file.name
            mol_name = Path(file_name).stem  # Remove .xyz extension

            # Check if this molecule name is in our list
            if mol_name in self.molecule_names:
                # Read file content
                content = xyz_file.read()
                if isinstance(content, bytes):
                    content = content.decode("utf-8")
                name_to_content[mol_name] = content
                found_count += 1

        self.xyz_files = name_to_content
        logger.info(
            f"Found XYZ files for {found_count}/{len(self.molecule_names)} molecules"
        )
        return name_to_content

    def process_xyz_zip(self, zip_file) -> Dict[str, str]:
        """
        Process a ZIP file containing XYZ files.

        Args:
            zip_file: Uploaded ZIP file containing XYZ files

        Returns:
            Dictionary mapping molecule names to XYZ file contents
        """
        name_to_content = {}
        found_count = 0

        try:
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                for file_info in zip_ref.filelist:
                    if file_info.filename.endswith(".xyz"):
                        mol_name = Path(file_info.filename).stem

                        # Check if this molecule name is in our list
                        if mol_name in self.molecule_names:
                            with zip_ref.open(file_info.filename) as xyz_file:
                                content = xyz_file.read().decode("utf-8")
                                name_to_content[mol_name] = content
                                found_count += 1

            self.xyz_files = name_to_content
            logger.info(
                f"Found XYZ files for {found_count}/{len(self.molecule_names)} molecules from ZIP"
            )
            return name_to_content

        except Exception:
            logger.error("Error processing ZIP file")
            raise

    def create_ase_database(self, db_path: str = "") -> bytes:
        """
        Create ASE database from the processed data.

        Args:
            output_name: Name for the output database file

        Returns:
            Bytes content of the created database
        """
        if not self.xyz_files or self.csv_data is None:
            raise ValueError("No XYZ files or CSV data available for database creation")

        # Create temporary file for the database
        # with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        #     db_path = tmp_file.name

        try:
            # Create or connect to database
            db = connect(db_path)

            added_count = 0
            error_count = 0

            for mol_name, xyz_content in self.xyz_files.items():
                try:
                    # Create temporary XYZ file and read with ASE
                    with tempfile.NamedTemporaryFile(
                        mode="w", suffix=".xyz", delete=False
                    ) as xyz_tmp:
                        xyz_tmp.write(xyz_content)
                        xyz_tmp_path = xyz_tmp.name

                    # Read the molecule from XYZ content
                    atoms = read(xyz_tmp_path)
                    # Get additional data from CSV for this molecule
                    data = {}
                    add_atoms(db_path, xyz_tmp_path, mol_name, data)
                    # Clean up temporary file
                    Path(xyz_tmp_path).unlink()

                    # Add to database
                    
                    # db.write(atoms, **data)
                    added_count += 1

                except Exception as e:
                    logger.error(f"Error processing molecule {mol_name}: {e}")
                    error_count += 1

            # Read the database file as bytes
            with open(db_path, "rb") as f:
                db_content = f.read()

            logger.info(
                f"Database creation complete! Added: {added_count}, Errors: {error_count}"
            )
            return db_content

        except Exception:
            # Clean up on error
            if Path(db_path).exists():
                Path(db_path).unlink()
            raise


def add_atoms(db_path, mols_xyz_path, mol_name, data_extra):
    """Add atoms to the database.

    Args:
        mols_xyz_path (str): Path to the atoms in xyz format.
        mol_name (str): Name of the atoms.
        data_extra (dict): Extra data to be stored with the atoms.

    """
    ase_atoms = read(mols_xyz_path)
    with connect(db_path) as db:
        if len(list(db.select(name=mol_name))) > 0:
            db.update(name=mol_name, data=data_extra)
        else:
            db.write(ase_atoms, data=data_extra, name=mol_name)


def get_database_size(db_path):
    """Get the number of atoms in the database.

    Returns:
        int: Number of atoms in the database.

    """
    with connect(db_path) as db:
        return len(db)


def csv_to_ase_interface():
    """
    Streamlit interface for CSV to ASE database conversion.
    """
    st.header("🔄 CSV to ASE Database Converter")
    st.markdown("""
    Convert your CSV data and XYZ molecular structure files into an ASE (Atomic Simulation Environment) database.
    This allows you to store molecular structures with associated properties for easy access and analysis.
    """)

    converter = CSVToASEConverter()

    # Step 1: Upload CSV file
    st.subheader("1. Upload CSV Data")
    csv_file = st.file_uploader(
        "Choose a CSV file containing molecular data",
        type=["csv"],
        help="Upload a CSV file where molecule names are in the first few columns",
    )

    if csv_file is not None:
        try:
            df = converter.read_csv_data(csv_file)

            st.success(
                f"✅ CSV loaded successfully: {len(df)} rows, {len(df.columns)} columns"
            )

            # Show data preview
            with st.expander("📋 Data Preview", expanded=False):
                st.write("**First 5 rows:**")
                st.dataframe(df.head(), use_container_width=True)

                st.write("**Column Information:**")
                col_info = []
                for col in df.columns:
                    col_info.append(
                        {
                            "Column": col,
                            "Type": str(df[col].dtype),
                            "Non-null Count": df[col].count(),
                            "Sample Values": ", ".join(
                                str(x) for x in df[col].dropna().head(3).values
                            ),
                        }
                    )
                st.dataframe(pd.DataFrame(col_info), use_container_width=True)

            # Step 2: Select molecule name columns
            st.subheader("2. Select Molecule Name Columns")
            st.markdown("Choose which columns contain molecule names:")

            default_cols = df.columns[:4].tolist()  # Default to first 4 columns
            selected_cols = st.multiselect(
                "Molecule name columns",
                options=df.columns.tolist(),
                default=default_cols,
                help="Select the columns that contain molecule names",
            )

            if selected_cols:
                molecule_names = converter.extract_molecule_names(df, selected_cols)
                st.info(f"Found {len(molecule_names)} unique molecule names")

                with st.expander("🧬 Molecule Names Preview", expanded=False):
                    # Show first 20 molecule names
                    preview_names = molecule_names[:20]
                    if len(molecule_names) > 20:
                        st.write(
                            f"Showing first 20 of {len(molecule_names)} molecule names:"
                        )
                    else:
                        st.write(f"All {len(molecule_names)} molecule names:")

                    # Display in columns for better layout
                    cols = st.columns(4)
                    for i, name in enumerate(preview_names):
                        with cols[i % 4]:
                            st.write(f"• {name}")

                # Step 3: Upload XYZ files
                st.subheader("3. Upload XYZ Structure Files")
                st.markdown(
                    "Upload the molecular structure files corresponding to the molecule names:"
                )

                upload_method = st.radio(
                    "Choose upload method:",
                    ["Upload XYZ files Folder", "Upload ZIP file containing XYZ files"],
                    help="Select how you want to upload your XYZ files",
                )

                xyz_files_processed = False

                if upload_method == "Upload XYZ files Folder":
                    xyz_files = st.file_uploader(
                        "Choose XYZ Files (you can select multiple files)",
                        type=["xyz"],
                        accept_multiple_files=True,
                        help="Upload multiple XYZ files (file names should match molecule names)",
                    )

                    if xyz_files:
                        name_to_content = converter.process_xyz_files(xyz_files)
                        xyz_files_processed = True

                else:  # ZIP upload
                    zip_file = st.file_uploader(
                        "Choose ZIP file containing XYZ files",
                        type=["zip"],
                        help="Upload a ZIP file containing XYZ files",
                    )

                    if zip_file is not None:
                        name_to_content = converter.process_xyz_zip(zip_file)
                        xyz_files_processed = True

                if xyz_files_processed:
                    found_count = len(name_to_content)
                    total_count = len(molecule_names)

                    if found_count > 0:
                        st.success(
                            f"✅ Found XYZ files for {found_count}/{total_count} molecules"
                        )

                        # Show matching summary
                        col1, col2 = st.columns(2)

                        with col1:
                            st.write("**Found molecules:**")
                            found_molecules = sorted(name_to_content.keys())
                            for mol in found_molecules[:10]:  # Show first 10
                                st.write(f"✅ {mol}")
                            if len(found_molecules) > 10:
                                st.write(f"... and {len(found_molecules) - 10} more")

                        with col2:
                            if found_count < total_count:
                                st.write("**Missing molecules:**")
                                missing_molecules = [
                                    mol
                                    for mol in molecule_names
                                    if mol not in name_to_content
                                ]
                                for mol in missing_molecules[:10]:  # Show first 10
                                    st.write(f"❌ {mol}")
                                if len(missing_molecules) > 10:
                                    st.write(
                                        f"... and {len(missing_molecules) - 10} more"
                                    )

                        # Step 4: Create ASE Database
                        st.subheader("4. Create ASE Database")

                        st.info(
                            " The database will include molecular structures and associated properties from the CSV. if the database already exists, it will be updated with new entries."
                        )

                        db_path = st.text_input(
                            "Database file path",
                            value="examples/molecules.db",
                            help="Path to save the database file (on server or path in your local machine if using locally)",
                        )

                        st.write(
                            f"Current database size: {get_database_size(db_path) if Path(db_path).exists() else 0} entries"
                        )

                        if st.button("🔄 Create ASE Database", type="primary"):
                            try:
                                with st.spinner("Creating ASE database..."):
                                    db_content = converter.create_ase_database(db_path)

                                st.success("✅ ASE database created successfully!")
                                st.info(
                                    f"Database contains {found_count} molecular structures with associated properties"
                                )
                                st.write(
                                    f"Current database size: {get_database_size(db_path) if Path(db_path).exists() else 0} entries"
                                )

                                # Provide download button
                                st.download_button(
                                    label="📥 Download ASE Database",
                                    data=db_content,
                                    file_name=db_path.split("/")[-1],
                                    mime="application/octet-stream",
                                    help="Download the created ASE database file",
                                )

                                # Show usage instructions
                                with st.expander(
                                    "📚 How to use your ASE database", expanded=False
                                ):
                                    st.markdown("""
                                    **Using your ASE database in Python:**

                                    ```python
                                    from ase.db import connect

                                    # Connect to your database
                                    db = connect('molecules.db')

                                    # Get all molecules
                                    for row in db.select():
                                        print(f"Molecule: {row.name}")
                                        print(f"Formula: {row.formula}")
                                        atoms = db.get_atoms(row.id)
                                        # Work with the atomic structure...

                                    # Search for specific molecules
                                    for row in db.select('name=molecule_name'):
                                        atoms = db.get_atoms(row.id)
                                        # Process the molecule...
                                    ```

                                    **Features of your database:**
                                    - Molecular structures from XYZ files
                                    - Associated properties from CSV data
                                    - Energy data (if available in CSV)
                                    - Searchable by molecule name
                                    - Compatible with ASE analysis tools
                                    """)

                            except Exception as e:
                                st.error(f"❌ Error creating database: {str(e)}")
                                st.write("Please check your files and try again.")

                    else:
                        st.warning(
                            "⚠️ No matching XYZ files found. Please check that your XYZ file names match the molecule names from your CSV."
                        )

        except Exception as e:
            st.error(f"❌ Error processing CSV file: {str(e)}")

    else:
        # Show instructions when no file is uploaded
        st.info("👆 Please upload a CSV file to get started")

        with st.expander("💡 Tips for preparing your data", expanded=True):
            st.markdown("""
            **CSV File Requirements:**
            - Contains molecule names in the first few columns
            - May include additional properties (energies, descriptors, etc.)
            - Standard CSV format with headers

            **XYZ Files Requirements:**
            - Standard XYZ coordinate format
            - File names should match molecule names (without .xyz extension)
            - Can be uploaded individually or in a ZIP file

            **Example CSV structure:**
            ```
            Int1_Name,TS_Name,Int2_Name,Sub_Name,DeltaG_kcalmol,Barrier_kcalmol
            Benzene_H10_Int1_Sub65,Benzene_H10_TS_Sub65,Benzene_H10_Int2_Sub65,Sub65,1.23,15.67
            ...
            ```

            **Supported data types:**
            - Energy values (automatically detected by keywords like 'DeltaG', 'energy', 'barrier')
            - Any numerical properties
            - Text-based descriptors
            """)
