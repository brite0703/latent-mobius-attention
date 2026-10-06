"""Independent scalar equations and input-contract checks for contact features."""
import hashlib
import json
import math
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import radial_contacts as r

HERE = Path(__file__).resolve().parent


def scalar_reference(x, z, types):
    # Deliberately uses scalar distances and sums rather than the production axes.
    result = np.zeros((len(x), 216), dtype=np.float64)
    for i, coordinate in enumerate(x):
        for j, neighbor in enumerate(z):
            distance = math.sqrt(sum((float(a)-float(b))**2 for a,b in zip(coordinate, neighbor)))
            if distance >= 10:
                continue
            taper = .5*(1+math.cos(math.pi*distance/10))
            for a in range(24):
                if types[j,a]:
                    for c in range(9):
                        result[i, 9*a+c] += math.exp(-((distance-(c+2))**2)/2)*taper
    return result


def main():
    target = HERE / "radial_contacts_development_audit.json"
    if target.exists():
        raise FileExistsError("Preserve the completed development audit")
    checks = []
    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)
    def rejects(name, function):
        try:
            function()
        except (ValueError, TypeError, KeyError):
            checks.append(name)
        else:
            raise AssertionError(name)
    atoms = [dict(element="C", residue="ALA"), dict(element="N", residue="GLY"),
             dict(element="O", residue="ASP"), dict(element="S", residue="CYS")]
    types = r.protein_types(atoms)
    check("explicit element/residue order and one-hot counts", types.shape == (4,24) and
          np.array_equal(types[:,:4], np.eye(4)) and np.array_equal(types[:,4:].sum(1), np.ones(4)))
    x = np.array([[0.,0.,0.], [1.,-2.,.5], [-4.,2.,1.]])
    z = np.array([[2.,0.,0.], [3.,4.,0.], [-2.,1.,2.], [0.,0.,10.]])
    observed = r.radial_density(x,z,types)
    expected = scalar_reference(x,z,types)
    discrepancy = float(np.max(np.abs(observed-expected)))
    check("independent scalar distances, kernel, summation and flattening", discrepancy < 3e-15)
    check("float64 nonnegative finite output", observed.dtype == np.float64 and observed.shape == (3,216)
          and np.isfinite(observed).all() and (observed >= 0).all())
    single = r.radial_density([[0,0,0]], [[2,0,0]], r.protein_types(atoms[:1]))
    check("known radial-center value in both active type coordinates", abs(single[0,0]-(1+math.cos(math.pi/5))/2) < 1e-15
          and single[0,0] == single[0,36] and np.count_nonzero(single) == 18)
    for distance, name in [(10.,"at cutoff"), (10.000001,"beyond cutoff"), (100.,"remote atom")]:
        check("zero contribution "+name, not r.radial_density([[0,0,0]], [[0,0,distance]], types[:1]).any())
    check("positive contribution just inside cutoff", r.radial_density([[0,0,0]], [[0,0,9.999]], types[:1]).sum() > 0)
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    shift = np.array([31.,-16.,8.])
    check("joint rigid-motion invariance", np.allclose(observed, r.radial_density(x@rotation+shift,z@rotation+shift,types), rtol=0,atol=3e-14))
    check("reflection invariance of the radial descriptor", np.allclose(observed, r.radial_density(-x,-z,types),rtol=0,atol=3e-14))
    check("protein permutation invariance", np.allclose(observed,r.radial_density(x,z[::-1],types[::-1]),rtol=0,atol=3e-14))
    check("ligand row permutation covariance", np.allclose(observed[::-1],r.radial_density(x[::-1],z,types),rtol=0,atol=3e-14))
    shaped = observed.reshape(3,24,9)
    check("element and residue marginals have the same radial total", np.allclose(shaped[:,:4].sum(1),shaped[:,4:].sum(1),rtol=0,atol=3e-15))
    # A manufactured co-located type example exposes loss of joint type association.
    left=r.protein_types([dict(element="C",residue="ALA"),dict(element="N",residue="GLY")])
    right=r.protein_types([dict(element="N",residue="ALA"),dict(element="C",residue="GLY")])
    check("marginal descriptor does not retain joint element/residue association", np.array_equal(
        r.radial_density([[0,0,0]],[[2,0,0],[2,0,0]],left), r.radial_density([[0,0,0]],[[2,0,0],[2,0,0]],right)))
    payload=dict(ligand_atoms=[dict(reference_atom_index=i,xyz=c.tolist()) for i,c in enumerate(x)][::-1],
                 protein_atoms=[a|dict(xyz=c.tolist()) for a,c in zip(atoms,z)])
    check("payload reference indices restore canonical order and float32 output", np.array_equal(r.from_payload(payload),observed.astype(np.float32)))
    bad=dict(payload,ligand_atoms=[dict(reference_atom_index=0,xyz=c.tolist()) for c in x])
    rejects("duplicate reference atom indices rejected",lambda:r.from_payload(bad))
    rejects("unknown protein element rejected",lambda:r.protein_types([dict(element="CL",residue="ALA")]))
    rejects("unsupported protein residue rejected",lambda:r.protein_types([dict(element="C",residue="UNK")]))
    rejects("empty protein inventory rejected",lambda:r.protein_types([]))
    rejects("missing type assignments rejected",lambda:r.radial_density(x,z,np.zeros((4,24))))
    rejects("nonfinite coordinate rejected",lambda:r.radial_density(x,np.full((4,3),np.nan),types))
    rejects("coordinate dimensions rejected",lambda:r.radial_density(x,z[:,:2],types))
    rejects("unmatched protein/type count rejected",lambda:r.radial_density(x,z,types[:3]))
    rejects("empty ligand rejected",lambda:r.radial_density(np.zeros((0,3)),z,types))
    rejects("negative type value rejected",lambda:r.radial_density(x,z,-types))
    result=dict(passed=True,completed_utc=datetime.now(timezone.utc).isoformat(),check_count=len(checks),checks=checks,
        independent_scalar_max_absolute_difference=discrepancy,
        source_sha256=hashlib.sha256(Path(r.__file__).read_bytes()).hexdigest(),
        audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope="Manufactured descriptor equations and contracts only; no full-model, GPU, cached-data or predictive certification.")
    target.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
