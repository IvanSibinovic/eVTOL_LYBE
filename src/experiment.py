from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from functools import lru_cache
from pathlib import Path
from statistics import mean

from .demand import generate_demand
from .results import compact_replication, write_compact_pair
from .simulation import run_replication, save_replication

CHECKPOINT='study_checkpoint.sqlite3'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def candidate_assessment(rows, expected, target):
    if len(rows)>expected:
        raise ValueError('More replications than requested.')
    fractions=[r['fraction_boarding_within_wait_limit'] for r in rows
               if r['status']=='completed' and r['fraction_boarding_within_wait_limit'] is not None]
    failed=any(r['status']!='completed' for r in rows)
    remaining=expected-len(rows)
    upper=(sum(fractions)+remaining)/(len(fractions)+remaining) if len(fractions)+remaining else None
    average=mean(fractions) if fractions else None
    complete=len(rows)==expected
    qualifies=complete and not failed and average is not None and average>=target
    can_qualify=not failed and upper is not None and upper+1e-12>=target
    return dict(observed_replications=len(rows),valid_fractions=len(fractions),failed=failed,
                observed_mean=average,best_possible_final_mean=upper,complete=complete,
                qualifies=qualifies,can_qualify=can_qualify)


class Checkpoint:
    def __init__(self,directory,metadata,resume=False):
        self.path=Path(directory)/CHECKPOINT
        if resume and not self.path.is_file():
            raise ValueError(f'Checkpoint not found: {self.path}')
        if not resume and self.path.exists():
            raise FileExistsError(self.path)
        self.connection=sqlite3.connect(self.path)
        self.connection.execute('CREATE TABLE IF NOT EXISTS metadata (id INTEGER PRIMARY KEY, content TEXT NOT NULL)')
        self.connection.execute('CREATE TABLE IF NOT EXISTS runs (scenario TEXT, fleet INTEGER, rep INTEGER, content TEXT NOT NULL, PRIMARY KEY(scenario,fleet,rep))')
        previous=self.connection.execute('SELECT content FROM metadata WHERE id=1').fetchone()
        if previous:
            if json.loads(previous[0])!=metadata:
                self.connection.close()
                raise ValueError('Checkpoint configuration, inputs, code or experiment settings differ. Use the original settings to resume.')
        else:
            self.connection.execute('INSERT INTO metadata VALUES (1,?)',(json.dumps(metadata,sort_keys=True),))
            self.connection.commit()

    def rows(self,scenario,fleet):
        records=self.connection.execute('SELECT rep,content FROM runs WHERE scenario=? AND fleet=? ORDER BY rep',(scenario,fleet)).fetchall()
        if [r[0] for r in records]!=list(range(len(records))):
            raise ValueError('Non-contiguous checkpoint replication indices.')
        return [json.loads(r[1]) for r in records]

    def save(self,scenario,fleet,rep,row):
        self.connection.execute('INSERT INTO runs VALUES (?,?,?,?)',(scenario,fleet,rep,json.dumps(row,allow_nan=False)))
        self.connection.commit()

    def close(self,remove=False):
        self.connection.close()
        if remove:self.path.unlink()


def read_checkpoint_settings(directory):
    path=Path(directory)/CHECKPOINT
    if not path.is_file():raise ValueError(f'Checkpoint not found: {path}')
    connection=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)
    try:return json.loads(connection.execute('SELECT content FROM metadata WHERE id=1').fetchone()[0])['settings']
    finally:connection.close()


def run_experiment(data,directory,*,scenarios=('low','medium','high'),replications=100,
                   target=.95,code_hash='',mode='fleet_study',fleet_size=None,
                   detailed=False,resume=False,simulate=None,progress=print):
    if type(replications) is not int or replications<1:raise ValueError('replications must be positive.')
    if not 0<target<=1:raise ValueError('service target must be in (0,1].')
    if mode not in ('fleet_study','single_configuration'):raise ValueError('Unknown experiment mode.')
    sizes=sorted(data.config['experiment']['fleet_sizes'])
    allocations=data.network['initial_fleet_allocation']
    if not sizes or len(set(sizes))!=len(sizes):raise ValueError('Invalid fleet search range.')
    for size in sizes:
        if str(size) not in allocations:raise ValueError(f'Missing initial allocation for fleet {size}.')
    if mode=='fleet_study' and sizes!=list(range(sizes[0],sizes[-1]+1)):
        raise ValueError('Fleet search range must be consecutive for the N-1 comparison.')
    if mode=='single_configuration' and str(fleet_size) not in allocations:
        raise ValueError(f'Missing initial allocation for fleet {fleet_size}.')
    scenarios=tuple(scenarios)
    if not scenarios or len(set(scenarios))!=len(scenarios):raise ValueError('Invalid scenarios.')
    if any(s not in data.config['demand']['adoption_share'] for s in scenarios):raise ValueError('Unknown demand scenario.')
    directory=Path(directory).resolve()
    if not resume:directory.mkdir(parents=True,exist_ok=False)
    elif not directory.is_dir():raise ValueError('Resume directory does not exist.')
    settings=dict(scenarios=list(scenarios),replications=replications,target=target,mode=mode,
                  fleet_size=fleet_size,detailed=detailed,base_seed=data.config['run']['base_seed'],
                  run_demand_scenario=data.config['run']['demand_scenario'])
    config_hash=digest(data.config)
    metadata=dict(settings=settings,config_sha256=config_hash,input_sha256=data.input_sha256,
                  code_sha256=code_hash,format_version=1)
    checkpoint=Checkpoint(directory,metadata,resume=resume)
    successful=False
    outcome={}
    confidence=data.config['outputs']['confidence_level']

    @lru_cache(maxsize=16)
    def demand_for(scenario,index):
        return generate_demand(data,scenario=scenario,replication_index=index)

    def evaluate(scenario,size,early=False):
        rows=checkpoint.rows(scenario,size)
        while len(rows)<replications:
            assessment=candidate_assessment(rows,replications,target)
            if early and not assessment['can_qualify']:break
            index=len(rows)
            config=dict(fleet_size=size,demand_scenario=scenario,replications=replications,
                        initial_allocation=dict(allocations[str(size)]))
            if simulate is None:
                result=run_replication(data,config,index,demand=demand_for(scenario,index),detailed=detailed)
                row=compact_replication(result,data,config_hash=config_hash,code_hash=code_hash,target=target)
                if detailed:
                    path=directory/'debug'/f'{scenario}_fleet{size}'/f'rep_{index:04d}'
                    # A crash after debug export but before checkpoint must not
                    # overwrite the earlier debug artifact on resume.
                    if not path.exists():save_replication(result,path,data)
            else:
                row=simulate(scenario,size,index)
            if (row['scenario'],row['fleet_size'],row['replication_index'])!=(scenario,size,index):
                raise ValueError('Simulation returned a different configuration or replication.')
            checkpoint.save(scenario,size,index,row)
            rows.append(row)
            fraction=row['fraction_boarding_within_wait_limit']
            value=f'{fraction:.2%}' if fraction is not None else 'недефинисано'
            progress(f'{scenario}, флота {size}, {index+1}/{replications}: услуга {value}; {row["status"]}.')
        return rows

    def report(scenario,role,size,opt,rows,status,evidence):
        return write_compact_pair(directory,scenario,role,size,opt,rows,expected_replications=replications,
                                  target=target,confidence=confidence,selection_status=status,
                                  search_evidence=json.dumps(evidence,sort_keys=True,separators=(',',':')))
    try:
        all_ok=True
        for scenario in scenarios:
            if mode=='single_configuration':
                rows=evaluate(scenario,fleet_size)
                report(scenario,f'n{fleet_size}',fleet_size,None,rows,'single_configuration_no_optimum_search',{})
                all_ok &= all(r['status']=='completed' for r in rows)
                outcome[scenario]=dict(status='single_configuration',fleet_size=fleet_size)
                continue
            evidence={}
            optimum=None
            for size in sizes:
                rows=evaluate(scenario,size,early=True)
                info=candidate_assessment(rows,replications,target)
                evidence[str(size)]=info
                if info['qualifies']:
                    optimum=size
                    break
                progress(f'{scenario}, флота {size}: циљ није достигнут или више није достижан у овом експерименту.')
            if optimum is None:
                # Keep the requested names, but never invent a qualifying fleet.
                status='target_not_met_within_tested_range'
                for role in ('n_opt','n_minus','n_plus'):
                    report(scenario,role,None,None,[],status,evidence)
                progress(f'{scenario}: циљ {target:.0%} није потврђен у испитаном распону {sizes[0]}–{sizes[-1]}.')
                outcome[scenario]=dict(status=status,n_opt=None,search=evidence)
                all_ok=False
                continue
            progress(f'{scenario}: N*={optimum}; довршавам понављања за суседне флоте.')
            role_rows={}
            for role,size in [('n_opt',optimum),('n_minus',optimum-1),('n_plus',optimum+1)]:
                if size<1 or str(size) not in allocations:
                    role_rows[role]=(size,[],'neighbor_allocation_not_defined')
                    all_ok=False
                else:
                    rows=evaluate(scenario,size)
                    role_rows[role]=(size,rows,'selected')
                    evidence[str(size)]=candidate_assessment(rows,replications,target)
                    all_ok &= all(r['status']=='completed' for r in rows)
            for role,(size,rows,status) in role_rows.items():
                report(scenario,role,size,optimum,rows,status,evidence)
            outcome[scenario]=dict(status='selected',n_opt=optimum,search=evidence)
            demand_for.cache_clear()
        successful=all_ok
        return outcome
    finally:
        # On interruption/error the one checkpoint retains every completed run.
        # It also retains search evidence if a target/neighbour is unavailable.
        checkpoint.close(remove=successful)
