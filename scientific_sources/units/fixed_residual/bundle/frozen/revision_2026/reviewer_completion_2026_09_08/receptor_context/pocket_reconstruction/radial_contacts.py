"""Fixed marginal type/radius densities; no affinities or learned parameters."""
import numpy as np

ELEMENTS = ("C", "N", "O", "S")
RESIDUES = tuple("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split())
CENTERS = np.arange(2., 11., dtype=np.float64)
SIGMA_ANGSTROM = 1.0
CUTOFF_ANGSTROM = 10.0
DIMENSION = (len(ELEMENTS)+len(RESIDUES))*len(CENTERS)


def protein_types(atoms):
    if not atoms:
        raise ValueError("The retained protein atom list must be nonempty")
    out = np.zeros((len(atoms), len(ELEMENTS)+len(RESIDUES)), dtype=np.float64)
    for i, atom in enumerate(atoms):
        out[i, ELEMENTS.index(atom["element"].upper())] = 1.
        out[i, len(ELEMENTS)+RESIDUES.index(atom["residue"])] = 1.
    return out


def radial_density(ligand_xyz, protein_xyz, types):
    """Float64 output in type-major, center-minor order; input units are Angstrom."""
    x, z, t = (np.asarray(v, dtype=np.float64) for v in (ligand_xyz, protein_xyz, types))
    if x.ndim != 2 or z.ndim != 2 or x.shape[1:] != (3,) or z.shape[1:] != (3,):
        raise ValueError("Coordinate arrays must have shape [atoms,3]")
    if len(x) < 1 or len(z) < 1 or t.shape != (len(z), 24):
        raise ValueError("Nonempty aligned ligand/protein/type arrays are required")
    if not all(np.isfinite(v).all() for v in (x, z, t)):
        raise ValueError("Nonfinite coordinate or type value")
    if not (np.isin(t, (0., 1.)).all() and (t[:, :4].sum(1) == 1).all() and (t[:, 4:].sum(1) == 1).all()):
        raise ValueError("Each protein atom requires one element and one residue indicator")
    distance = np.sqrt(np.sum((x[:, None, :]-z[None, :, :])**2, axis=-1))
    cutoff = np.where(distance < CUTOFF_ANGSTROM,
                      .5*(1.+np.cos(np.pi*distance/CUTOFF_ANGSTROM)), 0.)
    weights = np.exp(-.5*((distance[:, :, None]-CENTERS)/SIGMA_ANGSTROM)**2)*cutoff[:, :, None]
    result = np.einsum("ijc,ja->iac", weights, t, optimize=False).reshape(len(x), DIMENSION)
    if not np.isfinite(result).all() or (result < 0).any():
        raise ValueError("Invalid radial density")
    return result


def from_payload(payload):
    """Canonical-reference atom order, independent of serialized ligand row order.

    The caller must first verify the frozen payload hash and chemical alignment.
    This routine checks its own array/type contract, not full source eligibility.
    """
    lig, pro = payload["ligand_atoms"], payload["protein_atoms"]
    if sorted(a["reference_atom_index"] for a in lig) != list(range(len(lig))):
        raise ValueError("Ligand reference indices must be a complete bijection")
    lig = sorted(lig, key=lambda a: a["reference_atom_index"])
    result = radial_density([a["xyz"] for a in lig], [a["xyz"] for a in pro], protein_types(pro))
    return result.astype(np.float32)
