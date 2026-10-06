"""Recompute every eligible deterministic input and measure its separate cost.

No affinity targets or model predictions enter this calculation. The original
common tensors remain unchanged; regenerated arrays are compared by exact hash.
"""
from pathlib import Path
from datetime import datetime,timezone
import gzip,hashlib,json,sys,time
import numpy as np
import torch
from rdkit import Chem,rdBase
import radial_contacts as contact

HERE=Path(__file__).resolve().parent
REV=HERE.parents[2]
sys.path.insert(0,str(REV))
import prepare_lp_pdbbind as features

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def digest(array):return hashlib.sha256(array.tobytes()).hexdigest()

def main():
    destination=HERE/'matched_preprocessing_costs.json'
    if destination.exists():raise FileExistsError('Preserve the existing timing record')
    torch.set_num_threads(2)
    cohort=read(HERE/'eligible_cohort_v2/cohort_manifest.json')
    cache=read(HERE/'matched_inputs/tensors_v1/manifest.json')
    expected={r['pdbid']:r for r in cache['rows']}
    assert len(expected)==1612
    rows=[]
    for split in ('train','val','test'):
        for entry in cohort['eligible_by_split'][split]:
            key=entry['pdbid'];reference=expected[key]
            assert reference['split']==split
            payload_path=(HERE/'eligible_cohort_v2'/entry['payload']['path']).resolve()
            assert payload_path.is_relative_to((HERE/'eligible_cohort_v2').resolve())
            start=time.perf_counter();cpu=time.process_time()
            encoded=payload_path.read_bytes();raw=gzip.decompress(encoded);payload=json.loads(raw)
            loading_wall=time.perf_counter()-start;loading_cpu=time.process_time()-cpu
            # Source verification is outside the reported preprocessing timers.
            assert hashlib.sha256(encoded).hexdigest()==entry['payload']['compressed_sha256']
            assert hashlib.sha256(raw).hexdigest()==entry['payload']['uncompressed_sha256']
            assert payload['pdbid']==key and payload['no_affinity_fields']
            start=time.perf_counter();cpu=time.process_time()
            mol=Chem.RemoveHs(Chem.MolFromSmiles(reference['canonical_smiles']))
            Chem.AssignStereochemistry(mol,cleanIt=True,force=True)
            x=np.asarray([features.atom_features(atom) for atom in mol.GetAtoms()],dtype=np.float32)
            adjacency=Chem.GetAdjacencyMatrix(mol).astype(np.float32)+np.eye(len(x),dtype=np.float32)
            inv=1/np.sqrt(adjacency.sum(1));adjacency=adjacency*inv[:,None]*inv[None,:]
            graph_wall=time.perf_counter()-start;graph_cpu=time.process_time()-cpu
            start=time.perf_counter();cpu=time.process_time()
            descriptor=contact.from_payload(payload)
            contact_wall=time.perf_counter()-start;contact_cpu=time.process_time()-cpu
            assert digest(x)==reference['graph_X_sha256'] and digest(adjacency)==reference['graph_adjacency_sha256']
            assert digest(descriptor)==reference['contact_sha256']
            rows.append(dict(pdbid=key,split=split,ligand_atoms=len(x),protein_atoms=len(payload['protein_atoms']),
                payload_read_decompress_parse_wall_seconds=loading_wall,payload_read_decompress_parse_cpu_seconds=loading_cpu,
                graph_feature_wall_seconds=graph_wall,graph_feature_cpu_seconds=graph_cpu,
                contact_descriptor_wall_seconds=contact_wall,contact_descriptor_cpu_seconds=contact_cpu,
                exact_graph_and_contact_hashes=True))
            if len(rows)%200==0:print(json.dumps({'recomputed':len(rows),'total':1612}),flush=True)
    keys=[k for k in rows[0] if k.endswith('_seconds')]
    summary={split:{'records':sum(r['split']==split for r in rows),
        **{k:float(sum(r[k] for r in rows if r['split']==split)) for k in keys}} for split in ('train','val','test')}
    paths=[Path(__file__),Path(contact.__file__),Path(features.__file__),HERE/'matched_inputs/tensors_v1/manifest.json',
        HERE/'eligible_cohort_v2/cohort_manifest.json']
    result=dict(passed=True,completed_utc=datetime.now(timezone.utc).isoformat(),records=len(rows),rows=rows,summary=summary,
        source_closure=[dict(path=str(p),sha256=sha(p)) for p in paths],
        runtime=dict(python=sys.version,numpy=np.__version__,rdkit=rdBase.rdkitVersion,torch=torch.__version__,threads=torch.get_num_threads()),
        numerical_affinity_targets_accessed=False,model_predictions_computed=False,
        qualification='One serial pass over every eligible input; per-process CPU and elapsed wall time are separate. Payload read/decompression/JSON parsing, canonical graph features and radial descriptor calculation are separately timed. Integrity hashing, source download, eligibility processing, GPU transfer, training and inference are excluded. Warm/cold filesystem caching is uncontrolled; other work may affect elapsed time. These are local preprocessing measurements, not whole-system inference latency or comparative speedup estimates.')
    destination.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'passed':True,'records':len(rows),'summary':summary}),flush=True)

if __name__=='__main__':main()
