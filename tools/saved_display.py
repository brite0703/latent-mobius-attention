"""Saved-result display entry. Explicit archives; no model execution or data download."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import runpy
import statistics
import sys
import zipfile

REPO=Path(__file__).resolve().parents[1]
SYN='units/retained_synthetic/frozen/revision_2026/reviewer_completion_2026_09_08/'
PUBLIC_TABLES=(1,2,3,4,5,6,7,8,10)
EXTENSION_TABLES=(9,11,12,13,14)
SOURCE_MEMBERS={
 'sequence':'units/receptor_replay/bundle/frozen/revision_2026/reviewer_completion_2026_09_08/receptor_context/sequence_summary.json',
 'matched':'units/receptor_replay/bundle/frozen/revision_2026/reviewer_completion_2026_09_08/receptor_context/pocket_reconstruction/matched_study/summary.json',
 'core':'units/core_cardinality_molecular/combined_results_summary.json',
 'components':'units/components_width/summary.json'}
SOURCE_HASHES={
 'sequence':'56ae5f496042f350f867541880a9e28bcc09c71bdf0ba628d03533380e3d4659',
 'matched':'c12b3e461bc99b056625442d005b13b0baff2fcc61edb42502d0d60b0ca9c10d',
 'core':'18a2206a3e79fe6b5047e73e750e227e89f5eb81d6f8f59203ccb628bd525d02',
 'components':'fba320ff52029d63fb01ca0f2b36a2e9d37e4ef6b1b759c69eed12ba2cb5a0dd'}
MODEL_NAMES={'deepsets_ln':'Deep Sets + LN','deepsets_plain':'Deep Sets plain','deepsets_wide':'Deep Sets wide',
 'transformer':'Transformer','janossy2':'Janossy-2','cp_pool':'CP','lma1':r'LBOIA $k=1$',
 'lma2':r'LBOIA $k=2$','lma3':r'LBOIA $k=3$','lma3_clip1':r'LBOIA $k=3$, clip 1',
 'additive2':r'Additive $k=2$','additive3':r'Additive $k=3$','count_lookup':'Count lookup',
 'walsh_degree1':'Walsh degree one','walsh_degree3':'Walsh degree at most three'}


def sha(data):return hashlib.sha256(data).hexdigest()
def json_member(z,name,expected=None):
    data=z.read(name)
    if expected and sha(data)!=expected:raise ValueError('Historical member identity mismatch: '+name)
    return json.loads(data)


def extension(root):
    if root is None:raise ValueError('Current scalar extension is unavailable; see docs/SAVED_RESULT_INPUTS.md')
    root=root.resolve();manifest=json.loads((root/'manifest.json').read_text())
    expected={f'table_{n}.csv' for n in EXTENSION_TABLES}|{'reference_cells.json'}
    if manifest.get('schema_version')!=1 or set(manifest['files'])!=expected:raise ValueError('Unexpected extension schema')
    result={}
    for name,record in manifest['files'].items():
        p=root/name
        if p.is_symlink() or not p.is_file():raise ValueError('Missing or unsafe input: '+name)
        data=p.read_bytes()
        if len(data)!=record['bytes'] or sha(data)!=record['sha256']:raise ValueError('Changed input: '+name)
        result[name]=data
    return result


def scalars(data):
    import io
    rows=list(csv.DictReader(io.StringIO(data.decode())))
    text_fields={'campaign','setting','head','metric','scope','label','id','context','contrast','family','panel','initializer','precision'}
    for r in rows:
        for k,v in list(r.items()):
            if k in text_fields:continue
            r[k]=json.loads(v) if v else None
    return rows


def aggregate(rows,section,metric):
    if len(rows)!=10 or len({r['seed'] for r in rows})!=10:raise ValueError('Expected ten distinct retained seeds')
    values=[r[section][metric] for r in rows]
    return statistics.mean(values),statistics.stdev(values)


def formatter():
    path=REPO/'scientific_sources/table_export/export_retained_tables.py'
    # Only the existing standard-library display formatter is imported, never an analyzer/model.
    return runpy.run_path(str(path))


def table_cells(z,selected,extra,presentation):
    f=formatter();num=f['number'];pm=lambda a,b:'$'+num(a)+r'\pm'+num(b)+'$'
    layout=json.loads((REPO/'configuration/saved_display_layout.json').read_text())
    result={};evaluations={}
    for n in selected:
        if n in (4,5,6,7,8):
            key={4:'sequence',5:'matched',6:'core',7:'components',8:'components'}[n]
            data=json_member(z,SOURCE_MEMBERS[key],SOURCE_HASHES[key]);spec=layout['tables'][str(n)]
            builder={4:'receptor_rows',5:'receptor_rows',6:'molecular_rows',7:'component_rows',8:'width_rows'}[n]
            lines,_=f[builder](data,spec)
            result[n]=[[c.strip() for c in line.removesuffix('[2pt]').removesuffix(r'\\').split(' & ')] for line in lines]
            continue
        if n in (1,2,3):
            block={1:'first_cubic',2:'synthetic_parity',3:'hierarchical_sequence'}[n]
            data=json_member(z,SYN+block+'/evaluation.json');evaluations[n]=data
            if n==1:heads=['deepsets_ln','deepsets_plain','deepsets_wide','janossy2','cp_pool','lma1','lma2','lma3','additive2','additive3','walsh_degree1','walsh_degree3']
            elif n==2:heads=['deepsets_ln','deepsets_plain','deepsets_wide','transformer','janossy2','cp_pool','lma1','lma2','lma3','lma3_clip1','count_lookup']
            else:heads=['lma1','lma2','lma3','transformer','deepsets_wide','cp_pool','count_lookup']
            rows=[]
            for head in heads:
                source=data.get('count_lookups',data.get('references',[])) if head=='count_lookup' or head.startswith('walsh_') else data['rows']
                label=MODEL_NAMES[head]
                if n==1 and head=='janossy2':label=r'Janossy $k=2$'
                if n==3:label={'lma1':'LBOIA order one','lma2':'LBOIA order two','lma3':'LBOIA order three'}.get(head,label)
                cells=[label]
                if n in (1,2):
                    parameters={r['total_parameters'] for r in source if r['head']==head and 'total_parameters' in r}
                    if len(parameters)>1:raise ValueError('Inconsistent parameter count')
                    cells.append(str(next(iter(parameters))) if parameters else '--')
                for condition in (['first','cubic'] if n==1 else [10,20,40,80] if n==2 else ['depth2','depth3','depth4']):
                    key='n' if n==2 else 'task';chosen=[r for r in source if r['head']==head and r[key]==condition]
                    section='population' if n==2 else 'test';metric='mse' if n==1 else 'accuracy'
                    mean,sd=aggregate(chosen,section,metric)
                    if n==3 and presentation=='current':
                        import numpy as np
                        values=[r['test']['accuracy'] for r in chosen]
                        mean=float(np.mean(values));sd=float(np.std(values,ddof=1))
                    if n==2:
                        success=sum(r['population']['success_at_099'] for r in chosen)
                        cells.append(r'\shortstack[r]{$'+num(mean)+r'$\\$\pm'+num(sd)+r'\;('+str(success)+')$}')
                    else:cells.append(pm(mean,sd))
                rows.append(cells)
            result[n]=rows;continue
        if n==10:
            diagnostic=json_member(z,'units/components_width/routing_diagnostics.json')
            component=json_member(z,SOURCE_MEMBERS['components'],SOURCE_HASHES['components'])
            groups={r['head']:r for r in component['groups']};rows=[]
            fields=['penalty_at_global_atom_weighted_load','atom_weighted_entropy_over_log_M','graph_weighted_soft_coassignment_mean','graph_weighted_hard_argmax_coassignment_mean']
            for order in (1,2):
                for suffix,label in [('learned0',r'Learned, $\lambda=0$'),('learned001',r'Learned, $\lambda=0.01$'),('learned01',r'Learned, $\lambda=0.1$'),('fixed','Fixed random-feature'),('uniform','Uniform'),('noquery','Query-free')]:
                    head='m8_k'+str(order)+'_'+suffix
                    chosen=[r for r in diagnostic['rows'] if r['head']==head]
                    if len(chosen)!=5:raise ValueError('Missing retained routing group')
                    tests=[next(s for s in r['splits'] if s['split']=='test') for r in chosen]
                    values=[statistics.mean(t[k] for t in tests) for k in fields]+[groups[head]['rmse']['mean']]
                    rows.append([label,str(order)]+['$'+num(v)+'$' for v in values])
            result[n]=rows;continue
        if extra is None:raise ValueError('No current input for Table '+str(n))
        source=scalars(extra['table_'+str(n)+'.csv']);rows=[]
        for r in source:
            if n==9:
                spec=layout['tables']['9'];label=spec['heads'][r['head']]
                if presentation=='current' and r['head']=='mean_count':label='Mean + count'
                rows.append([spec['settings'][r['setting']],label,pm(r['mean'],r['sample_sd']),format(r['parameters'],','),f['three'](r['forward_median_ms']),f['three'](r['train_step_median_ms']),f['three'](r['maximum_incremental_train_MiB'])])
            elif n==11:rows.append([str(r['n']),'$'+num(r['mean_difference'])+'$','$'+num(r['paired_sample_sd'])+'$','$'+f['interval']([r['interval_low'],r['interval_high']])+'$',str(r['clip1_successes'])+'/'+str(r['clip10_successes'])])
            elif n==12:rows.append([r['family'].split('_')[0],r['scope'],str(r['m']),str(r['t_Holm_below_005']),str(r['sign_flip_Holm_below_005'])])
            elif n==13:rows.append([f['escape'](r['id']+' '+r['label']),r['family'].split('_')[0],str(r['n']),'$'+num(r['mean_difference'])+'$','$'+f['interval'](r['t_reference_interval95'])+'$']+['$'+f['pnumber'](r[k])+'$' for k in ['t_p_raw','family_t_p_Holm','family_sign_flip_p_Holm','global_t_p_Holm']])
            elif n==14:
                if r['panel']=='starts':rows.append([str(r['seed']),format(r['warm_initial_accuracy_percent'],'.4f')]+[format(r[k],'.6f') for k in ['warm_initial_ce','random_final_ce','warm_final_ce']]+['$'+format(r['warm_minus_random_final_ce'],'.6f')+'$'])
                else:rows.append([r['initializer'],r['precision']]+[format(r[k],'.2f') if r[k] is not None else '--' for k in ['seed100_percent','seed101_percent','seed102_percent']]+[str(r['successes'])+'/'+str(r['denominator'])])
        result[n]=rows
    return result


def fresh_output(path):
    if path is None:raise ValueError('--output-root is required')
    path=path.resolve()
    if path.exists():raise ValueError('Output must be new; previous artifacts are preserved')
    if path.is_relative_to(REPO) or REPO.is_relative_to(path):raise ValueError('Output must be outside the code checkout')
    path.mkdir(parents=True,exist_ok=False);return path


def compare_cells(actual,reference):
    import re
    def canonical(s):return re.sub(r'\s+','',re.sub(r'\\[!,;]','',s)).replace('$','')
    def tokens(s):
        s=canonical(s);s=re.sub(r'([-+]?\d+(?:\.\d+)?)\\times10\^\{([-+]?\d+)\}',lambda m:m[1]+'e'+m[2],s)
        return re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?',s)
    reports=[]
    for n,rows in actual.items():
        expected=reference['tables'][str(n)]['cells'];topology=len(rows)==len(expected) and all(len(a)==len(b) for a,b in zip(rows,expected))
        diffs=[];numeric=True
        for i,(a,b) in enumerate(zip(expected,rows),1):
            for j,(x,y) in enumerate(zip(a,b),1):
                if canonical(x)!=canonical(y):diffs.append(dict(row=i,column=j,reference=x,regenerated=y))
                if tokens(x)!=tokens(y):numeric=False
        reports.append(dict(table=n,topology_matches=topology,numeric_displays_exact=topology and numeric,all_normalized_cells_exact=topology and not diffs,differences=diffs))
    return reports


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation',choices=['plan','preflight','tables','curves'])
    p.add_argument('--historical-09',type=Path);p.add_argument('--historical-14',type=Path)
    p.add_argument('--current-inputs',type=Path);p.add_argument('--output-root',type=Path)
    p.add_argument('--tables',nargs='+',type=int,choices=range(1,15),default=list(PUBLIC_TABLES))
    p.add_argument('--presentation',choices=['original','current'],default='original')
    a=p.parse_args()
    if a.operation=='plan':
        print(json.dumps(dict(public_archive_tables=PUBLIC_TABLES,current_extension_tables=EXTENSION_TABLES,
            curve_source='Public09 curve reference + public14 selection/candidate JSONs',
            critical_current_extension_publicly_available=False,scope='Saved displays only; no full raw-to-results claim'),indent=2));return
    if a.historical_14 is None:raise ValueError('--historical-14 is required')
    extra=extension(a.current_inputs) if a.current_inputs else None
    if a.operation=='preflight':
        with zipfile.ZipFile(a.historical_14) as z:
            for key,name in SOURCE_MEMBERS.items():json_member(z,name,SOURCE_HASHES[key])
        if a.historical_09:
            with zipfile.ZipFile(a.historical_09) as z:json_member(z,'curves/curve_manifest.json')
        print(json.dumps(dict(preflight_pass=True,current_extension_present=extra is not None,no_scientific_execution=True),indent=2));return
    if a.operation=='tables':
        if any(n in EXTENSION_TABLES for n in a.tables) and extra is None:raise ValueError('Selected tables require the unavailable current extension')
        with zipfile.ZipFile(a.historical_14) as z:cells=table_cells(z,a.tables,extra,a.presentation)
        output=fresh_output(a.output_root)
        for n,rows in cells.items():
            with (output/f'table_{n}_cells.csv').open('x',newline='') as stream:
                w=csv.writer(stream);w.writerows(rows)
            columns=len(rows[0]);body=[r'\begin{tabular}{'+'l'*columns+'}']+[' & '.join(r)+r'\\' for r in rows]+[r'\end{tabular}']
            (output/f'table_{n}.tex').write_text('\n'.join(body)+'\n')
        comparison=compare_cells(cells,json.loads(extra['reference_cells.json'])) if extra else []
        receipt=dict(scope='Saved values/mean-SD aggregation and formatting only',tables=a.tables,presentation=a.presentation,
            reference_comparison=comparison,no_new_model_evaluation=True,new_inferential_statistics=False,
            full_manuscript_layout_equivalence=False,current_mode_adaptations='NumPy Table3 aggregation; two Table9 mean/count display labels' if a.presentation=='current' else 'None')
        (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2));return
    if a.historical_09 is None:raise ValueError('--historical-09 is required for curve references')
    output=fresh_output(a.output_root);completion=output/'work';renderer=completion/'learning_curves/build.py'
    renderer.parent.mkdir(parents=True)
    source=(REPO/'scientific_sources/retained_curves/build.py').read_text()
    if a.presentation=='current':
        for old,new in [('LMA, order one','LBOIA, order one'),('LMA, order two','LBOIA, order two'),('LMA, order three','LBOIA, order three'),('LMA three, clip one','LBOIA three, clip one')]:
            if source.count('"'+old+'"')!=1:raise ValueError('Unexpected renderer label')
            source=source.replace('"'+old+'"','"'+new+'"')
    renderer.write_text(source)
    with zipfile.ZipFile(a.historical_09) as z:
        reference=json_member(z,'curves/curve_manifest.json');reference_points=z.read('curves/curve_points.csv')
        if sha(reference_points)!=reference['points_csv_sha256']:raise ValueError('Curve reference CSV mismatch')
    with zipfile.ZipFile(a.historical_14) as z:
        for block in ['synthetic_parity','first_cubic','hierarchical_sequence']:
            folder=completion/block;folder.mkdir()
            for name in ['selection_lock.json','final_audit.json']:(folder/name).write_bytes(z.read(SYN+block+'/'+name))
            lock=json.loads((folder/'selection_lock.json').read_text())
            for record in lock['candidate_records']:
                rel=record['path'].replace('\\','/');part=Path(rel)
                if part.is_absolute() or '..' in part.parts or part.suffix!='.json':raise ValueError('Unsafe candidate path')
                data=z.read(SYN+block+'/'+rel)
                if sha(data)!=record['sha256']:raise ValueError('Candidate identity changed')
                dest=folder/part;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
    runpy.run_path(str(renderer),run_name='__main__')
    produced=renderer.parent/'curve_points.csv'
    receipt=dict(scope='Plot selected retained histories only',csv_byte_identical=produced.read_bytes()==reference_points,
        selected_runs=780,points=76392,figures=9,presentation=a.presentation,
        full_pixel_or_whole_SVG_identity_asserted=False,loss_tolerance=0.0,new_training=False,new_model_inference=False)
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
    if not receipt['csv_byte_identical']:raise ValueError('Saved curve points differ; outputs retained')


if __name__=='__main__':main()
