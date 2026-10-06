"""Fixed 150-candidate panel and 75 validation-only choices; no data access."""
import math,random
from itertools import combinations
from matched_models import configurations,HEADS,SETTINGS

SEEDS=tuple(range(42,47))
RATES=(.0003,.001)
CANDIDATE_ORDER_SEED=2026090842

def key(setting,head):return setting+'/'+head

def candidate_plan():
    rows=[dict(id=f'{setting}_{head}_seed{seed}_lr{i}',setting=setting,head=head,seed=seed,lr=lr,lr_index=i)
        for setting,head in configurations() for seed in SEEDS for i,lr in enumerate(RATES)]
    random.Random(CANDIDATE_ORDER_SEED).shuffle(rows)
    assert len(rows)==len({r['id'] for r in rows})==150
    return rows

def planned_contrasts():
    rows=[]
    for setting in SETTINGS:
        for right in ('lma1','additive2','deepsets_plain70','cp_pool'):
            rows.append(dict(id=f'{setting}_lma2_minus_{right}',
                terms=[dict(configuration=key(setting,'lma2'),weight=1),dict(configuration=key(setting,right),weight=-1)]))
    for earlier,later in combinations(SETTINGS,2):
        for head in HEADS:
            rows.append(dict(id=f'{head}_{later}_minus_{earlier}',
                terms=[dict(configuration=key(later,head),weight=1),dict(configuration=key(earlier,head),weight=-1)]))
        rows.append(dict(id=f'{later}_order_difference_minus_{earlier}_order_difference',terms=[
            dict(configuration=key(later,'lma2'),weight=1),dict(configuration=key(later,'lma1'),weight=-1),
            dict(configuration=key(earlier,'lma2'),weight=-1),dict(configuration=key(earlier,'lma1'),weight=1)]))
    assert len(rows)==len({r['id'] for r in rows})==30
    return rows

def select(records):
    expected={row['id']:row for row in candidate_plan()}
    actual={row['id']:row for row in records}
    if len(records)!=len(actual) or set(actual)!=set(expected):
        raise ValueError('Exactly all 150 prescribed terminal candidates are required')
    for identifier,row in actual.items():
        if any(row.get(field)!=value for field,value in expected[identifier].items()):raise ValueError('Candidate identity mismatch')
        if row.get('status') not in ('valid','failed'):raise ValueError('Incomplete candidate')
        if row['status']=='valid':
            score,epoch=row.get('best_validation_rmse'),row.get('best_epoch')
            if type(score) not in (int,float) or not math.isfinite(score) or score<0:raise ValueError('Invalid validation score')
            if type(epoch)!=int or not 1<=epoch<=100:raise ValueError('Invalid selected epoch')
    choices=[];nominations=[]
    for setting,head in configurations():
        procedure=[]
        for seed in SEEDS:
            candidates=sorted([r for r in actual.values() if (r['setting'],r['head'],r['seed'])==(setting,head,seed)],key=lambda r:r['lr'])
            assert len(candidates)==2
            valid=[r for r in candidates if r['status']=='valid']
            row=dict(configuration=key(setting,head),setting=setting,head=head,seed=seed,candidate_ids=[r['id'] for r in candidates])
            if valid:
                chosen=min(valid,key=lambda r:(r['best_validation_rmse'],r['lr']))
                row.update(status='valid',selected_id=chosen['id'],learning_rate=chosen['lr'],
                    best_epoch=chosen['best_epoch'],validation_rmse=chosen['best_validation_rmse'])
            else:row.update(status='all_candidates_failed',selected_id=None,learning_rate=None,best_epoch=None,validation_rmse=None)
            choices.append(row);procedure.append(row)
        scores=[r['validation_rmse'] for r in procedure if r['status']=='valid']
        nominations.append(dict(configuration=key(setting,head),prescribed_seeds=5,valid_seeds=len(scores),eligible=len(scores)==5,
            mean_selected_validation_rmse=math.fsum(scores)/5 if len(scores)==5 else None))
    eligible=[(i,r) for i,r in enumerate(nominations) if r['eligible']]
    nominated=min(eligible,key=lambda p:(p[1]['mean_selected_validation_rmse'],p[0]))[1]['configuration'] if eligible else None
    assert len(choices)==75
    return dict(selections=choices,procedure_validation=nominations,nominated_procedure=nominated,
        nomination_scope='Mean of all five selected validation scores; complete procedures only. A nomination is neither an ensemble nor a refit and does not change the primary comparison.',
        primary_information_setting='ligand_contact',primary_contrast='ligand_contact_lma2_minus_lma1',contrasts=planned_contrasts())
