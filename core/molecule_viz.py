"""Simplified molecule visualization utilities."""

import streamlit as st
import streamlit.components.v1 as components
from ase.db import connect


class MoleculeVisualizer:
    """Handles molecule visualization from ASE database."""

    def __init__(self, database_url: str) -> None:
        """Initialize the visualizer.

        Args:
            database_url: URL or path to ASE database
        """
        self.database_url = database_url

    def create_3d_plot(
        self, molecule_name: str, show_info: bool = True
    ) -> None:
        """Create 3D molecular plot.

        Args:
            molecule_name: Name of molecule in database
            show_info: Whether to show molecule information
        """
        try:
            # Get molecule data
            db = connect(self.database_url)
            atoms = db.get(name=molecule_name)

            # Convert to XYZ format
            xyz_data = self._atoms_to_xyz(atoms.toatoms())

            if show_info:
                self._display_molecule_info(atoms.toatoms(), molecule_name)

            # Generate atom data for JavaScript
            atom_data_js = self._generate_atom_data_js(atoms.toatoms())

            # Create 3D viewer
            self._create_3d_viewer(
                xyz_data, atom_data_js, color_by_charges=False
            )

            with st.expander("🔧 Extra Visualization Options", expanded=False):
                
                charges_available_list = [ x for x in atoms.data.keys() if (x.startswith("charge") or x.startswith("charges")) ]
                st.checkbox("Color by Partial Charges", value=True)
                st.checkbox("Show Atom Info on Click", value=True)
                selected_charge = st.selectbox(
                    "Select charge to color by",
                    options=charges_available_list,
                    index=0,
                )
                self._create_3d_viewer(
                    xyz_data, atom_data_js, color_by_charges=True, charges=atoms.data.get(selected_charge, None)
                )

            with st.expander("multiple molecules", expanded=False):
                st.write("To visualize multiple molecules, select them from the sidebar and use the 'Display Molecule' button.")
                st.session_state.selected_molecules_dict[molecule_name] ={"xyz": xyz_data, "atom_data_js": atom_data_js, "charges": atoms.data.get(selected_charge, None)}
                cols = st.columns(len(st.session_state.selected_molecules_dict))
                for i, (mol_name, mol_data) in enumerate(st.session_state.selected_molecules_dict.items()):
                    with cols[i]:
                        st.write(f"**{mol_name}**")
                        if st.button(f"Display {mol_name}"):
                            self._create_3d_viewer(
                                mol_data["xyz"], mol_data["atom_data_js"], color_by_charges=True, charges=mol_data["charges"]
                            )
                if st.button("Clear Selection"):
                    st.session_state.selected_molecules_dict = {}
        except Exception as e:
            st.error(f"Error visualizing molecule '{molecule_name}': {e}")

    def _atoms_to_xyz(self, atoms) -> str:
        """Convert ASE atoms object to XYZ format string."""
        xyz_lines = [str(len(atoms)), ""]
        for atom in atoms:
            xyz_lines.append(
                f"{atom.symbol} {atom.position[0]:.6f} "
                f"{atom.position[1]:.6f} {atom.position[2]:.6f}"
            )
        return "\n".join(xyz_lines)

    def _display_molecule_info(self, atoms, molecule_name: str) -> None:
        """Display molecule information."""
        st.write(
            f"**Atoms:** {len(atoms)} | "
            f"**Formula:** {atoms.get_chemical_formula()} | "
            f"**Name:** {molecule_name}"
        )

        # Add link to CSD database if name looks like a CSD code
        if len(molecule_name) <= 10 and molecule_name.isalnum():
            csd_url = (
                f"https://www.ccdc.cam.ac.uk/structures/Search?"
                f"Ccdcid={molecule_name}&DatabaseToSearch=Published"
            )
            st.markdown(f"[View on CSD]({csd_url})")

    def _generate_atom_data_js(self, atoms) -> str:
        """Generate JavaScript atom data array."""
        atom_data = []
        for atom in atoms:
            atom_data.append(
                {
                    "element": atom.symbol,
                    "position": [
                        float(atom.position[0]),
                        float(atom.position[1]),
                        float(atom.position[2]),
                    ],
                }
            )

        return f"const atomDataArray = {atom_data};"

    def _generate_clickable_js(self) -> str:
        """Generate JavaScript for clickable atoms."""
        js_code = """
                // Click state management
                let selectedAtoms = [];
                let currentHighlight = null;
                
                // Function to calculate distance between two points
                function calculateDistance(pos1, pos2) {
                    const dx = pos1.x - pos2.x;
                    const dy = pos1.y - pos2.y;
                    const dz = pos1.z - pos2.z;
                    return Math.sqrt(dx*dx + dy*dy + dz*dz);
                }
                
                // Function to update info panel
                function updateInfoPanel() {
                    const panel = document.getElementById('info-panel');
                    
                    if (selectedAtoms.length === 0) {
                        panel.innerHTML = '<div>Click on an atom to see its information</div>';
                    } else if (selectedAtoms.length === 1) {
                        const atom = selectedAtoms[0];
                        const atomData = atomDataArray[atom.serial];
                        panel.innerHTML = `
                            <div class="atom-info selected-atom">Selected: ${atomData.element}${atom.serial + 1}</div>
                            <div class="atom-info">Position: (${atomData.position[0].toFixed(2)}, ${atomData.position[1].toFixed(2)}, ${atomData.position[2].toFixed(2)})</div>
                            <div style="margin-top: 5px; font-style: italic;">Click another atom to measure distance</div>
                        `;
                    } else if (selectedAtoms.length === 2) {
                        const atom1 = selectedAtoms[0];
                        const atom2 = selectedAtoms[1];
                        const atomData1 = atomDataArray[atom1.serial];
                        const atomData2 = atomDataArray[atom2.serial];
                        const distance = calculateDistance(atom1, atom2);
                        
                        panel.innerHTML = `
                            <div class="atom-info">Atom 1: ${atomData1.element}${atom1.serial + 1}</div>
                            <div class="atom-info">Atom 2: ${atomData2.element}${atom2.serial + 1}</div>
                            <div class="atom-info distance-info">Distance: ${distance.toFixed(3)} Å</div>
                            <div style="margin-top: 5px; font-style: italic;">Click to select new atoms</div>
                        `;
                    }
                }
                
                // Function to clear highlights
                function clearHighlights() {
                    viewer.removeAllLabels();
                    currentHighlight = null;
                }
                
                // Function to highlight selected atoms
                function highlightAtom(atom) {
                    const atomData = atomDataArray[atom.serial];
                    currentHighlight = viewer.addLabel(
                        '●',
                        {
                            position: atom,
                            fontSize: 2,
                            fontColor: 'red',
                            backgroundColor: 'rgba(255,0,0,0.3)',
                            borderRadius: 50,
                            alignment: 'center',
                            inFront: true
                        }
                    );
                    viewer.render();
                }
                
                // Click handler
                viewer.setClickable(
                    {},
                    true,
                    function(atom, viewer, event, container) {
                        // Clear previous highlights if we're starting fresh
                        if (selectedAtoms.length >= 2) {
                            selectedAtoms = [];
                            clearHighlights();
                        }
                        
                        // Add clicked atom to selection
                        selectedAtoms.push(atom);
                        
                        // Highlight the selected atom
                        highlightAtom(atom);
                        
                        // Update info panel
                        updateInfoPanel();
                    }
                );
                
                // Keep existing hover functionality
                viewer.setHoverable(
                    {}, 
                    true,
                    function(atom, viewer, event, container) {
                        if (!atom.label && selectedAtoms.length < 2) {
                            atom.label = viewer.addLabel(
                                atom.atom + (atom.serial + 1),
                                {
                                    position: atom, 
                                    backgroundColor: 'mintcream', 
                                    fontColor: 'black',
                                    fontSize: 12
                                }
                            );
                        }
                    },
                    function(atom, viewer) {
                        if (atom.label) {
                            viewer.removeLabel(atom.label);
                            delete atom.label;
                        }
                    }
                );
        """
        return js_code

    def _generate_individual_atom_colors(
        self, charges, color_by_charges: bool = True
    ) -> str:
        """Generate JavaScript for setting individual atom colors."""
        # Color by partial charges (if available)
        if not color_by_charges or not charges:
            return ""
        charges = [float(c) for c in charges]
        color_map = create_charge_colormap(charges)
        color_styles = {}
        for i, charge in enumerate(charges):
            color_styles[f"atom_{i}"] = {
                "color": color_map[i],
                "scale": 0.3,
                "radius": 0.1,
            }

        # Generate the JavaScript for individual atom coloring
        script_lines = []
        for atom_id, style in color_styles.items():
            atom_index = atom_id.split("_")[
                1
            ]  # Extract index from "atom_0", "atom_1", etc.
            script_lines.append(
                f"viewer.setStyle({{serial: {atom_index}}}, "
                f"{{stick: {{radius: {style['radius']}, color: '{style['color']}'}}, "
                f"sphere: {{scale: {style['scale']}, color: '{style['color']}'}} }});"
            )
        return "\n    ".join(script_lines)

    def _create_3d_viewer(
        self,
        xyz_data: str,
        atom_data_js: str,
        color_by_charges: bool,
        charges=None,
    ) -> None:
        """Create the 3DMol.js viewer."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
            <script src="https://3Dmol.org/build/3Dmol.ui-min.js"></script>     
            <style>
                #viewer {{
                    width: 100%;
                    height: 350px;
                    border: 1px solid #ddd;
                    margin: 0 auto;
                    cursor: pointer;
                }}
                #info-panel {{
                    margin-top: 10px;
                    padding: 10px;
                    background-color: #f8f9fa;
                    border: 1px solid #dee2e6;
                    border-radius: 0px;
                    font-family: Arial, sans-serif;
                    font-size: 14px;
                    min-height: 180px;
                }}
                .atom-info {{
                    margin: 3px 0;
                }}
                .selected-atom {{
                    font-weight: bold;
                    color: #007bff;
                }}
                .distance-info {{
                    color: #28a745;
                    font-weight: bold;
                }}
            </style>
        </head>
        <body>
            <div id="viewer"></div>
            <div id="info-panel">
                <div>Click on an atom to see its information</div>
            </div>
            <script>
                const viewer = $3Dmol.createViewer('viewer');
                viewer.addModel(`{xyz_data}`, 'xyz');
                viewer.setStyle({{}}, {{
                    stick: {{radius: 0.1}}, 
                    sphere: {{scale: 0.3}}
                }});
                
                // Atom data
                {atom_data_js}

                // Individual atom colors
                {self._generate_individual_atom_colors(charges, color_by_charges)}

                {self._generate_clickable_js()}

                viewer.zoomTo();
                viewer.render();
                // Initialize info panel
                updateInfoPanel();
            </script>
        </body>
        </html>
        """
        components.html(html_content, height=500, width=None)


def create_molecule_viewer(database_url: str) -> MoleculeVisualizer:
    """Create a molecule visualizer instance.

    Args:
        database_url: URL or path to ASE database

    Returns:
        MoleculeVisualizer instance
    """
    return MoleculeVisualizer(database_url)


def display_molecule_from_csv_selection(
    database_url: str, molecule_name: str, container=None
) -> None:
    """Display molecule visualization in a Streamlit container.

    Args:
        database_url: URL or path to ASE database
        molecule_name: Name of molecule to display
        container: Streamlit container (optional)
    """
    if container:
        with container:
            visualizer = create_molecule_viewer(database_url)
            visualizer.create_3d_plot(molecule_name)
    else:
        visualizer = create_molecule_viewer(database_url)
        visualizer.create_3d_plot(molecule_name)


# Configuration for common databases
DEFAULT_ASE_DATABASES = {
    "Example": "examples/sampled_molecules.db",
    #"Local SQLite": "molecules.db",
    "Custom": "",
}


def get_database_selector() -> str:
    """Create a database selector widget.

    Returns:
        Selected database URL
    """
    st.sidebar.subheader("🗄️ Molecule Database")

    db_option = st.sidebar.selectbox(
        "Select Database",
        list(DEFAULT_ASE_DATABASES.keys()),
        help="Choose the molecular database to use",
    )

    if db_option == "Custom":
        database_url = st.sidebar.text_input(
            "Database URL/Path",
            placeholder="postgresql://user:pass@host:port/db or /path/to/file.db",
            help="Enter custom database URL or file path",
        )
        st.sidebar.info("using database: " + database_url)
    else:
        database_url = DEFAULT_ASE_DATABASES[db_option]
        if db_option != "Local SQLite":
            st.sidebar.info(f"Using: {database_url}")

    return database_url


def create_charge_colormap(charges, min_charge=0, max_charge=0.5):
    """Create color map based on charge values and display color scale."""
    # Normalize charges to [0, 1] range
    normalized_charges = [
        (c - min_charge) / (max_charge - min_charge) for c in charges
    ]

    # Create color gradient: red (negative) -> white (neutral) -> blue (positive)
    colors = []
    for norm_charge in normalized_charges:
        # White to blue
        intensity = int(255 * (norm_charge))
        colors.append(f"rgb({255 - intensity}, {255 - intensity}, 255)")

    # Display color scale

    return colors
