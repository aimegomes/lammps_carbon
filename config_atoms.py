import numpy as np

def generate_simple_cubic_lattice(a=10.0):
    """
    Generate 27 points in simple cubic lattice arrangement
    - Central point at (0,0,0)
    - Points at -a, 0, +a in each dimension
    """
    coords = np.array([-a, 0, a])
    x, y, z = np.meshgrid(coords, coords, coords, indexing='ij')
    points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    return points

def generate_bonds():
    """
    Generate bonds for simple cubic lattice with periodic boundary conditions.
    Each atom has 6 bonds (3 positive directions, 3 negative directions).
    Total bonds = 27 * 6 / 2 = 81
    """
    bonds = []
    bond_index = 1

    # Create a mapping from 3D coordinates to atom indices
    coord_to_index = {}
    coords = [-1, 0, 1]  # indices in each dimension (-a, 0, +a)

    # Build coordinate to index mapping
    idx = 1
    for i in coords:
        for j in coords:
            for k in coords:
                coord_to_index[(i, j, k)] = idx
                idx += 1

    # Generate ALL bonds for each atom (6 per atom)
    all_bonds_set = set()  # Use set to avoid duplicates

    for i in coords:
        for j in coords:
            for k in coords:
                current_atom = coord_to_index[(i, j, k)]

                # Define all 6 neighbors with periodic boundary conditions
                neighbors = [
                    ((i+1) % 3 - 1, j, k),  # +x direction
                    ((i-1) % 3 - 1, j, k),  # -x direction
                    (i, (j+1) % 3 - 1, k),  # +y direction
                    (i, (j-1) % 3 - 1, k),  # -y direction
                    (i, j, (k+1) % 3 - 1),  # +z direction
                    (i, j, (k-1) % 3 - 1)   # -z direction
                ]

                # Add all 6 bonds for this atom
                for neighbor_coord in neighbors:
                    neighbor_atom = coord_to_index[neighbor_coord]
                    # Only add bond if atoms are different and avoid duplicates
                    if current_atom != neighbor_atom:
                        # Create sorted tuple to avoid duplicate bonds
                        bond_pair = tuple(sorted([current_atom, neighbor_atom]))
                        all_bonds_set.add(bond_pair)

    # Convert set to sorted list with bond indices
    sorted_bonds = sorted(all_bonds_set)
    bonds = [(i+1, bond[0], bond[1]) for i, bond in enumerate(sorted_bonds)]

    return bonds

def generate_configuration_file(a=10.0, fixed_val1=1, fixed_val2=1,
                               atom_filename="lattice_configuration.txt",
                               bond_filename="bond_configuration.txt"):
    """
    Generate configuration files for atoms and bonds
    """
    # Generate points and bonds
    points = generate_simple_cubic_lattice(a=a)
    bonds = generate_bonds()

    # Create and save the atom configuration file
    with open(atom_filename, 'w') as f:
        # Write header
        f.write(f"# Simple Cubic Lattice Configuration\n")
        f.write(f"# Lattice parameter a = {a}\n")
        f.write(f"# Fixed value 1 = {fixed_val1}\n")
        f.write(f"# Fixed value 2 = {fixed_val2}\n")
        f.write(f"# Number of points: {len(points)}\n")
        f.write(f"# Format: Index Fixed1 Fixed2 X Y Z\n")
        f.write(f"#\n")

        # Write data
        for i, point in enumerate(points):
            # Index from 1 to 27 (not 0 to 26)
            index = i + 1
            f.write(f"{index:3d} {fixed_val1:8.3f} {fixed_val2:8.3f} {point[0]:8.3f} {point[1]:8.3f} {point[2]:8.3f}\n")

    # Create and save the bond configuration file
    with open(bond_filename, 'w') as f:
        # Write header
        f.write(f"# Bond Configuration for Simple Cubic Lattice\n")
        f.write(f"# Lattice parameter a = {a}\n")
        f.write(f"# Number of bonds: {len(bonds)}\n")
        f.write(f"# Format: Bond_Index Atom1 Atom2\n")
        f.write(f"#\n")

        # Write bond data
        for bond in bonds:
            bond_idx, atom1, atom2 = bond
            f.write(f"{bond_idx:3d} {atom1:3d} {atom2:3d}\n")

    print(f"Atom configuration file generated: {atom_filename}")
    print(f"Bond configuration file generated: {bond_filename}")
    print(f"Parameters: a = {a}, Fixed1 = {fixed_val1}, Fixed2 = {fixed_val2}")
    print(f"Total atoms: {len(points)}")
    print(f"Total bonds: {len(bonds)}")

    # Verify bond count and no self-bonds
    self_bonds = [bond for bond in bonds if bond[1] == bond[2]]
    if self_bonds:
        print(f"WARNING: Found {len(self_bonds)} self-bonds!")
    else:
        print("All bonds connect different atoms - verification passed!")

    # Verify each atom has exactly 6 bonds
    bond_count_per_atom = {}
    for bond in bonds:
        atom1, atom2 = bond[1], bond[2]
        bond_count_per_atom[atom1] = bond_count_per_atom.get(atom1, 0) + 1
        bond_count_per_atom[atom2] = bond_count_per_atom.get(atom2, 0) + 1

    print(f"Bonds per atom: {set(bond_count_per_atom.values())}")

# Example usage with different parameters
if __name__ == "__main__":
    # You can change these values as needed
    a_value = 10          # Spacing between points
    fixed_value_1 = 1    # First fixed value (e.g., mass, type, etc.)
    fixed_value_2 = 1    # Second fixed value (e.g., charge, radius, etc.)
    atom_output_filename = "atom_configuration.txt"
    bond_output_filename = "bond_configuration.txt"

    generate_configuration_file(
        a=a_value,
        fixed_val1=fixed_value_1,
        fixed_val2=fixed_value_2,
        atom_filename=atom_output_filename,
        bond_filename=bond_output_filename
    )
