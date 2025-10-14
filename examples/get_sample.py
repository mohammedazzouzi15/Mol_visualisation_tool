
import pandas as pd
from ase.db import connect


def get_atoms_from_db(mol_name, db_path):
    with connect(db_path) as conn:
        for row in conn.select(name=mol_name):
            atoms = row.toatoms()
    return atoms

def save_small_db(molecule_names, original_db_path, new_db_path):
    with connect(original_db_path) as original_conn, connect(new_db_path) as new_conn:
        for mol_name in molecule_names:
            for row in original_conn.select(name=mol_name):
                new_conn.write(row.toatoms(), data=row.data, name=row.name)
                # save the data as well

    print(f"Saved small database to {new_db_path}")


def main():
    csv_file_path = "eh_distances.csv"

    csv_file_descriptor="cheminformatics_descriptors.csv"
    db_path = "postgresql://ase:pw@Md843ae2c6f28.dyn.epfl.ch:5432/formed"
    df = pd.read_csv(csv_file_path)
    print(df.head())
    # sample a few molecule names
    sampled_molecules = (
        df["molecule_name"].sample(50, random_state=42).tolist()
    )
    print("Sampled Molecules:", sampled_molecules)
    df_sampled = df[df["molecule_name"].isin(sampled_molecules)]
    df_sampled.to_csv("sampled_molecules_ehdist.csv", index=False)

    new_db_path = "sampled_molecules.db"
    save_small_db(sampled_molecules, db_path, new_db_path)
    db_descriptor = pd.read_csv(csv_file_descriptor)
    db_descriptor_sampled = db_descriptor[db_descriptor["molecule_name"].isin(sampled_molecules)]
    db_descriptor_sampled.to_csv("sampled_molecules_descriptors.csv", index=False)



if __name__ == "__main__":
    main()
