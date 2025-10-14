#!/usr/bin/env python3
"""
Script to read CSV files and add molecules to an ASE database.
The script reads molecule names from the first 4 columns of the CSV file,
finds corresponding XYZ files in the CMD_Int1 folder, and adds them to an ASE database.
"""

import pandas as pd
import sys
from ase.io import read
from ase.db import connect
from pathlib import Path
import argparse
import logging
from typing import List

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def read_csv_file(csv_path: str) -> pd.DataFrame:
    """
    Read CSV file and return DataFrame.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        pandas DataFrame containing CSV data
    """
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Successfully read CSV file: {csv_path}")
        logger.info(f"CSV contains {len(df)} rows and {len(df.columns)} columns")
        return df
    except Exception as e:
        logger.error(f"Error reading CSV file {csv_path}: {e}")
        raise


def get_molecule_names_from_csv(df: pd.DataFrame) -> List[str]:
    """
    Extract unique molecule names from the first 4 columns of the CSV.
    
    Args:
        df: pandas DataFrame containing CSV data
        
    Returns:
        List of unique molecule names
    """
    # Get the first 4 columns (assuming they contain molecule names)
    first_four_columns = df.iloc[:, :4]
    
    # Get all unique molecule names from these columns
    molecule_names = set()
    for col in first_four_columns.columns:
        # Remove any NaN values and add to set
        names = first_four_columns[col].dropna().unique()
        molecule_names.update(names)
    
    # Convert to list and sort for consistent ordering
    molecule_names = sorted(list(molecule_names))
    
    logger.info(f"Found {len(molecule_names)} unique molecule names in first 4 columns")
    return molecule_names


def find_xyz_files(xyz_folder: str, molecule_names: List[str]) -> dict:
    """
    Find XYZ files in the specified folder that match the molecule names.
    
    Args:
        xyz_folder: Path to folder containing XYZ files
        molecule_names: List of molecule names to search for
        
    Returns:
        Dictionary mapping molecule names to XYZ file paths
    """
    xyz_folder_path = Path(xyz_folder)
    if not xyz_folder_path.exists():
        raise FileNotFoundError(f"XYZ folder not found: {xyz_folder}")
    
    # Get all XYZ files in the folder
    xyz_files = list(xyz_folder_path.glob("*.xyz"))
    logger.info(f"Found {len(xyz_files)} XYZ files in {xyz_folder}")
    
    # Map molecule names to XYZ file paths
    name_to_file = {}
    found_count = 0
    
    for mol_name in molecule_names:
        # Look for XYZ file with matching name
        matching_files = [f for f in xyz_files if f.stem == mol_name]
        
        if matching_files:
            name_to_file[mol_name] = str(matching_files[0])
            found_count += 1
        else:
            logger.warning(f"No XYZ file found for molecule: {mol_name}")
    
    logger.info(f"Found XYZ files for {found_count}/{len(molecule_names)} molecules")
    return name_to_file


def create_ase_database(name_to_file: dict, db_path: str, csv_data: pd.DataFrame) -> None:
    """
    Create ASE database and add molecules from XYZ files.
    
    Args:
        name_to_file: Dictionary mapping molecule names to XYZ file paths
        db_path: Path for the output ASE database
        csv_data: Original CSV data for additional properties
    """
    # Create or connect to database
    db = connect(db_path)
    
    added_count = 0
    error_count = 0
    
    for mol_name, xyz_path in name_to_file.items():
        try:
            # Read the molecule from XYZ file
            atoms = read(xyz_path)
            
            # Prepare additional data to store
            data = {
                'name': mol_name,
                'xyz_file': xyz_path,
                'source': 'CMD_Int1_folder'
            }
            
            # Add any relevant CSV data for this molecule
            # Look for rows in CSV where this molecule name appears in first 4 columns
            csv_rows = []
            for col in csv_data.columns[:4]:
                matching_rows = csv_data[csv_data[col] == mol_name]
                if not matching_rows.empty:
                    csv_rows.extend(matching_rows.index.tolist())
            
            if csv_rows:
                # Add information about which CSV rows contain this molecule
                data['csv_rows_count'] = len(set(csv_rows))
                data['first_csv_row'] = min(set(csv_rows))
                
                # Add some example properties from the first matching row
                first_row = csv_data.iloc[csv_rows[0]]
                
                # Add energy data if available
                energy_cols = [col for col in csv_data.columns if 'DeltaG' in col or 'kcalmol' in col]
                for col in energy_cols:
                    if pd.notna(first_row[col]):
                        data[col.lower().replace('_', '')] = float(first_row[col])
            
            # Add to database
            db.write(atoms, **data)
            added_count += 1
            
            if added_count % 50 == 0:
                logger.info(f"Added {added_count} molecules to database...")
            
        except Exception as e:
            logger.error(f"Error processing molecule {mol_name}: {e}")
            error_count += 1
    
    logger.info("Database creation complete!")
    logger.info(f"Successfully added: {added_count} molecules")
    logger.info(f"Errors encountered: {error_count} molecules")
    logger.info(f"Database saved to: {db_path}")


def main():
    """Main function to orchestrate the CSV to ASE database conversion."""
    parser = argparse.ArgumentParser(description='Convert CSV and XYZ files to ASE database')
    parser.add_argument('csv_file', help='Path to the CSV file')
    parser.add_argument('xyz_folder', help='Path to folder containing XYZ files')
    parser.add_argument('--output', '-o', default='molecules.db', 
                       help='Output ASE database file name (default: molecules.db)')
    
    args = parser.parse_args()
    
    try:
        # Read CSV file
        df = read_csv_file(args.csv_file)
        
        # Extract molecule names from first 4 columns
        molecule_names = get_molecule_names_from_csv(df)
        
        # Find corresponding XYZ files
        name_to_file = find_xyz_files(args.xyz_folder, molecule_names)
        
        if not name_to_file:
            logger.error("No matching XYZ files found. Exiting.")
            return
        
        # Create ASE database
        create_ase_database(name_to_file, args.output, df)
        
        logger.info("Script completed successfully!")
        
    except Exception as e:
        logger.error(f"Script failed with error: {e}")
        raise


if __name__ == "__main__":
    # If run without arguments, use the example data
    if len(sys.argv) == 1:
        # Use the example files
        csv_file = "/media/mohammed/Work/Mol_visualisation_tool/examples/Example_with_csv_and_xyz_folder/CMD_TS_formatted_f.csv"
        xyz_folder = "/media/mohammed/Work/Mol_visualisation_tool/examples/Example_with_csv_and_xyz_folder/CMD_Int1"
        output_db = "molecules_from_csv.db"
        
        print("No arguments provided. Using example data:")
        print(f"CSV file: {csv_file}")
        print(f"XYZ folder: {xyz_folder}")
        print(f"Output database: {output_db}")
        print()
        
        try:
            # Read CSV file
            df = read_csv_file(csv_file)
            
            # Extract molecule names from first 4 columns
            molecule_names = get_molecule_names_from_csv(df)
            
            # Find corresponding XYZ files
            name_to_file = find_xyz_files(xyz_folder, molecule_names)
            
            if not name_to_file:
                logger.error("No matching XYZ files found. Exiting.")
            else:
                # Create ASE database
                create_ase_database(name_to_file, output_db, df)
                logger.info("Script completed successfully!")
                
        except Exception as e:
            logger.error(f"Script failed with error: {e}")
            raise
    else:
        main()