"""Pinned public DNA corpus; deduplication and outcome-blind census order."""
import csv
import hashlib
import json
import random
from pathlib import Path

MODEL = 'Hengchang-Liu/D3LM-from-nt'
MODEL_REV = '71bb7164b14d832df4e723e19556ff638b8d02f4'
MODEL_SHA = 'fb6d54f9edb58388b58967d44ff751e950605afbca39a549ad01a0cfe78aed46'
DATASET = 'Zehui127127/latent-dna-diffusion'
DATA_REV = 'ab85ce9a7e29bac5a7fe823f2ed5b2b64b60c8eb'
DATA_SHA = '07677dc1831d38a9867a939aa936b1ef21162cedef5712df2cfe4659e3237fb3'
FOUR = {'Homo sapiens (Human).','Macaca mulatta (rhesus macaque).',
        'Mus musculus (Mouse).','Rattus Norvegicus (rat).'}
PILOT_ROWS = {75308,90333,100266,125206,91825}


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def atomic(path,value):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n'); temp.replace(path)


def seed(key,label):
    return int.from_bytes(hashlib.sha256(f'census081-v1:{label}:{key}'.encode()).digest()[:8],'big')%(2**62)


def records(source,scope='all_species'):
    assert sha(source) == DATA_SHA
    groups,pilot = {},set()
    total = excluded = 0
    with Path(source).open() as f:
        for i,r in enumerate(csv.DictReader(f)):
            total += 1; seq = r['Sequence'].upper()
            if len(seq)!=2048 or set(seq)-set('ACGT'):
                excluded += 1; continue
            key = hashlib.sha256(seq.encode()).hexdigest()
            if i in PILOT_ROWS: pilot.add(key)
            if key not in groups:
                groups[key] = dict(row_index=i,species=r['species'],sequence=seq,sha256=key,source_rows=[])
            groups[key]['source_rows'].append(i)
    assert total==159123 and excluded==2166 and len(groups)==156403 and len(pilot)==5
    pool = [r for r in groups.values() if r['species'] in FOUR and r['sha256'] not in pilot]
    previous = {r['sha256']:i for i,r in enumerate(random.Random(2026090601).sample(pool,500))}
    rows = [r for r in groups.values() if scope=='all_species' or r['species'] in FOUR]
    rows.sort(key=lambda r:(0,previous[r['sha256']]) if r['sha256'] in previous
              else (1,hashlib.sha256(('order081:'+r['sha256']).encode()).hexdigest()))
    for i,r in enumerate(rows):
        key = r['sha256']; old = previous.get(key)
        r.update(index=i,prior_500_index=old,prior_pilot=key in pilot,
                 in_original_four_species=r['species'] in FOUR,development=old is not None or key in pilot)
        if old is not None:
            assert old==i
            r.update(mask_seed=710000+i,candidate_seed=720000+i,random_seed=730000+i,
                     discovery_seeds=[800000+i*100+j for j in range(2)],
                     evaluation_seeds=[800000+i*100+j for j in range(20,28)])
        else:
            base = seed(key,'proposal')
            r.update({name+'_seed':seed(key,name) for name in ['mask','candidate','random']})
            r.update(discovery_seeds=[base,base+1],evaluation_seeds=[base+j for j in range(20,28)])
        r['split'] = 'development' if r['development'] else ('validation' if seed(key,'split')%10==0 else 'train')
    return rows,dict(source_rows=total,excluded_non_acgt_or_length=excluded,
                    eligible_unique=len(rows),scope=scope,source_sha256=DATA_SHA,
                    exact_dedup_only=True,homology_disjoint_split=False)


def fetch(destination):
    from huggingface_hub import snapshot_download,hf_hub_download
    destination = Path(destination); destination.mkdir(parents=True,exist_ok=True)
    model = Path(snapshot_download(MODEL,revision=MODEL_REV,local_dir=destination/'model',
                 allow_patterns=['*.json','*.py','vocab.txt','model.safetensors']))
    source = Path(hf_hub_download(DATASET,'sequence.csv',repo_type='dataset',revision=DATA_REV,
                  local_dir=destination/'dataset'))
    assert sha(model/'model.safetensors')==MODEL_SHA and sha(source)==DATA_SHA
    return model,source
