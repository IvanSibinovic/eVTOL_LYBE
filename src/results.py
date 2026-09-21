from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from statistics import mean, stdev

import numpy as np

from .diagnostics import CAUSE_LABELS
from .entities import Passenger

PASSENGER_FIELDS = [*Passenger.__dataclass_fields__, 'boarding_end_s', 'waiting_time_min',
    'within_wait_limit', 'wait_exceedance_min', 'request_to_takeoff_min', 'extra_after_boarding_min',
    'journey_duration_min', 'terminal_target_lateness_min', 'wait_constraints',
    'request_time_local', 'boarding_start_local', 'boarding_end_local', 'takeoff_start_local',
    'landing_end_local', 'journey_end_local', 'terminal_arrival_local']
FLIGHT_FIELDS = ['flight_id','aircraft_id','operation','origin','destination','passenger_count',
    'passenger_ids','takeoff_start_s','landing_end_s','duration_min','distance_km','departure_energy_kwh',
    'landing_energy_kwh','planned_energy_kwh','departure_soc','landing_soc','boarding_end_s',
    'energy_ready_s','required_departure_energy_kwh','energy_context']
CANDIDATE_FIELDS = ['snapshot_id','time_s','group_id','passenger_ids','origin','destination','aircraft_id',
    'available_at_s','available_location','forecast_energy_kwh','status','reason','boarding_start_s',
    'takeoff_s','repositioning','energy_ready_s','resource_constraints','constraint_vertiport','selected']


def write_csv(path, rows, fields=None):
    rows = list(rows)
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _stats(values, prefix):
    values = [float(x) for x in values if x is not None and math.isfinite(x)]
    return {f'{prefix}_{key}': value for key, value in dict(
        mean=mean(values) if values else None,
        p95=float(np.quantile(values, .95)) if values else None,
        max=max(values) if values else None).items()}


def passenger_rows(result, data):
    by_person = defaultdict(set)
    for row in result.wait_intervals:
        if row['stage'] == 'before_boarding':
            by_person[row['passenger_id']].update(row['causes'].split('|'))
    prep = data.config['ground_handling']['departure_finalization_min']*60
    limit = data.config['service']['max_wait_to_boarding_min']*60
    eps = data.config['simulation']['time_comparison_tolerance_s']
    for p in result.passengers:
        row = asdict(p)
        row['status'] = p.status.value
        detail = result.flight_details.get(p.flight_id, {})
        b_end = detail.get('boarding_end_s')
        wait = p.waiting_time_s
        row.update(boarding_end_s=b_end,
            waiting_time_min=wait/60 if wait is not None else None,
            within_wait_limit=(wait <= limit+eps) if wait is not None else None,
            wait_exceedance_min=max(0, wait-limit)/60 if wait is not None else None,
            request_to_takeoff_min=(p.takeoff_start_s-p.request_time_s)/60 if p.takeoff_start_s is not None else None,
            extra_after_boarding_min=max(0, p.takeoff_start_s-b_end-prep)/60
                if p.takeoff_start_s is not None and b_end is not None else None,
            journey_duration_min=(p.journey_end_s-p.request_time_s)/60 if p.journey_end_s is not None else None,
            terminal_target_lateness_min=max(0, p.terminal_arrival_s-p.latest_terminal_arrival_s)/60
                if p.terminal_arrival_s is not None and p.latest_terminal_arrival_s is not None else None,
            wait_constraints='|'.join(sorted(by_person[p.id])))
        for key in ('request_time_s', 'boarding_start_s', 'boarding_end_s', 'takeoff_start_s',
                    'landing_end_s', 'journey_end_s', 'terminal_arrival_s'):
            row[key.removesuffix('_s')+'_local'] = result.demand.local_datetime(row[key]).isoformat() if row[key] is not None else ''
        yield row


def wait_rows(result, data):
    people = {p.id:p for p in result.passengers}
    limit = data.config['service']['max_wait_to_boarding_min']*60
    for raw in result.wait_intervals:
        row = dict(raw)
        p = people[row['passenger_id']]
        row['duration_min'] = (row['end_s']-row['start_s'])/60
        row['overdue_min'] = (max(0, row['end_s']-max(row['start_s'], p.request_time_s+limit))/60
                              if row['stage']=='before_boarding' else 0.0)
        row['extra_after_boarding_min'] = row['duration_min'] if row['stage']=='after_boarding' else 0.0
        row['start_local'] = result.demand.local_datetime(row['start_s']).isoformat()
        row['end_local'] = result.demand.local_datetime(row['end_s']).isoformat()
        yield row


def cause_reports(waits, people):
    """Per-cause rows overlap. Combination rows partition observed time exactly."""
    late = {p['id'] for p in people if p['within_wait_limit'] is False}
    totals, combinations = {}, {}
    for row in waits:
        for destination, codes in ((totals, row['causes'].split('|')), (combinations, [row['causes']])):
            for code in codes:
                key = (row['stage'], row['scope'], code)
                bucket = destination.setdefault(key, dict(passengers=set(), late_exposed=set(), late_blocked=set(),
                                                          observed_min=0., overdue_min=0., extra_min=0.))
                pid = row['passenger_id']
                bucket['passengers'].add(pid)
                if pid in late: bucket['late_exposed'].add(pid)
                if row['overdue_min'] > 1e-9: bucket['late_blocked'].add(pid)
                bucket['observed_min'] += row['duration_min']
                bucket['overdue_min'] += row['overdue_min']
                bucket['extra_min'] += row['extra_after_boarding_min']
    def rows(buckets):
        for (stage, scope, code), b in sorted(buckets.items()):
            yield dict(stage=stage, scope=scope, causes=code,
                       label=' + '.join(CAUSE_LABELS.get(c, c) for c in code.split('|')),
                       exposed_passengers=len(b['passengers']),
                       late_passengers_exposed=len(b['late_exposed']),
                       late_passengers_blocked_after_limit=len(b['late_blocked']),
                       fraction_of_late_passengers_exposed=len(b['late_exposed'])/len(late) if late else None,
                       observed_passenger_minutes=b['observed_min'], overdue_passenger_minutes=b['overdue_min'],
                       extra_after_boarding_passenger_minutes=b['extra_min'])
    return list(rows(totals)), list(rows(combinations))


def cohort_metrics(people):
    boarded = [p for p in people if p['within_wait_limit'] is not None]
    return dict(generated_passengers=len(people), boarded_passengers=len(boarded),
                completed_journeys=sum(p['journey_end_s'] is not None for p in people),
                on_time_passengers=sum(p['within_wait_limit'] is True for p in people),
                fraction_boarding_within_wait_limit=(sum(p['within_wait_limit'] is True for p in people)/len(people)
                    if people and len(boarded)==len(people) else None),
                **_stats([p['waiting_time_min'] for p in people], 'wait_to_boarding_min'),
                **_stats([p['extra_after_boarding_min'] for p in people], 'extra_after_boarding_min'),
                **_stats([p['journey_duration_min'] for p in people], 'journey_duration_min'))


def enrich_summary(result, data):
    people = list(passenger_rows(result, data))
    completed_flights = [f for f in result.flights if f.landing_end_s is not None]
    passenger = [f for f in completed_flights if f.passenger_ids]
    empty = [f for f in completed_flights if not f.passenger_ids]
    seconds = sum(f.estimate.duration_s for f in completed_flights)
    total_energy = sum(f.estimate.energy_kwh for f in completed_flights)
    carried = sum(len(f.passenger_ids) for f in passenger)
    s = result.summary
    s.update(schema_version=2,
        demand_fingerprint=hashlib.sha256(json.dumps([asdict(r) for r in result.demand.records],
            sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        wait_limit_min=data.config['service']['max_wait_to_boarding_min'],
        seat_load_factor_on_passenger_flights=carried/(len(passenger)*data.aircraft['capacity']['passenger_seats']) if passenger else None,
        repositioning_flight_fraction=len(empty)/len(completed_flights) if completed_flights else None,
        repositioning_time_fraction=sum(f.estimate.duration_s for f in empty)/seconds if seconds else None,
        repositioning_energy_fraction=sum(f.estimate.energy_kwh for f in empty)/total_energy if total_energy else None,
        flight_energy_per_completed_passenger_kwh=s['flight_energy_kwh']/s['completed_passenger_journeys'] if s['completed_passenger_journeys'] else None,
        total_boarding_exceedance_passenger_min=sum(p['wait_exceedance_min'] or 0 for p in people),
        total_extra_after_boarding_passenger_min=sum(p['extra_after_boarding_min'] or 0 for p in people),
        **_stats([p['extra_after_boarding_min'] for p in people], 'extra_after_boarding_min'),
        **_stats([p['journey_duration_min'] for p in people], 'journey_duration_min'),
        late_terminal_passengers=sum((p['terminal_target_lateness_min'] or 0)>1e-8 for p in people),
        terminal_passengers_observed=sum(p['terminal_target_lateness_min'] is not None for p in people),
        unexplained_wait_passenger_min=sum((r['end_s']-r['start_s'])/60 for r in result.wait_intervals if 'unexplained_wait' in r['causes']),
        queue_peak=max((r['waiting_passengers'] for r in result.queue_intervals), default=0),
        queue_passenger_minutes=sum((r['end_s']-r['start_s'])*r['waiting_passengers']/60 for r in result.queue_intervals))
    # Global peak is a simultaneous network count, not a sum of individual peaks.
    changes = defaultdict(int)
    for p in people:
        if p['request_time_s'] <= s['end_time_s']:
            changes[p['request_time_s']] += 1
            changes[p['boarding_start_s'] if p['boarding_start_s'] is not None else s['end_time_s']] -= 1
    current = peak = 0
    for _, delta in sorted(changes.items()):
        current += delta
        peak = max(peak, current)
    s['network_queue_peak'] = peak
    s.update(result.diagnostic_totals)
    wait_values = [p['waiting_time_min'] for p in people if p['waiting_time_min'] is not None]
    terminal_late = [p['terminal_target_lateness_min'] for p in people
                     if p['terminal_target_lateness_min'] is not None and p['terminal_target_lateness_min']>1e-8]
    terminal_count = s['terminal_passengers_observed']
    s.update(schema_version=3,
        generated_airport_to_city_passengers=sum(p['origin']=='LYBE' for p in people),
        generated_city_to_airport_passengers=sum(p['destination']=='LYBE' for p in people),
        late_boarding_passengers=sum(p['within_wait_limit'] is False for p in people),
        fraction_late_boarding=(1-s['fraction_boarding_within_wait_limit']) if s['fraction_boarding_within_wait_limit'] is not None else None,
        median_wait_to_boarding_min=float(np.median(wait_values)) if wait_values else None,
        passengers_with_extra_after_boarding_wait=sum((p['extra_after_boarding_min'] or 0)>1e-8 for p in people),
        p95_request_to_takeoff_min=_stats([p['request_to_takeoff_min'] for p in people],'takeoff')['takeoff_p95'],
        fraction_terminal_late=len(terminal_late)/terminal_count if terminal_count else None,
        mean_terminal_lateness_among_late_min=mean(terminal_late) if terminal_late else None,
        max_terminal_lateness_among_late_min=max(terminal_late) if terminal_late else None,
        mean_passengers_per_passenger_flight=carried/len(passenger) if passenger else None,
        full_passenger_flight_fraction=sum(len(f.passenger_ids)==data.aircraft['capacity']['passenger_seats']
                                         for f in passenger)/len(passenger) if passenger else None,
        simulation_duration_hours=(s['end_time_s']-s['start_time_s'])/3600)
    for direction in ('airport_to_city','city_to_airport'):
        subset = [p for p in people if (p['origin']=='LYBE') == (direction=='airport_to_city')]
        values = [p['journey_duration_min'] for p in subset if p['journey_duration_min'] is not None]
        s[f'{direction}_journey_duration_min_mean'] = mean(values) if values else None
        s[f'{direction}_journey_duration_min_p95'] = float(np.quantile(values,.95)) if values else None
    for port in (v['id'] for v in data.network['vertiports']):
        subset = [p for p in people if p['origin']==port]
        metrics = cohort_metrics(subset)
        s[f'{port}_requests'] = len(subset)
        s[f'{port}_on_time_fraction'] = metrics['fraction_boarding_within_wait_limit']
        s[f'{port}_wait_mean_min'] = metrics['wait_to_boarding_min_mean']
        s[f'{port}_wait_p95_min'] = metrics['wait_to_boarding_min_p95']
    for row in result.resource_metrics:
        s[f"{row['vertiport']}_{row['resource_kind']}_utilization"] = row['utilization']
        if row['resource_kind']=='stand':
            s[f"{row['vertiport']}_stands_fully_occupied_min"] = row['fully_occupied_minutes']
    if s['diagnostics_enabled']:
        observed_wait = sum(max(0, (p['boarding_start_s'] if p['boarding_start_s'] is not None
                                  else s['end_time_s'])-p['request_time_s'])/60 for p in people)
        interval_wait = s['diagnostic_before_boarding_min']
        interval_extra = s['diagnostic_after_boarding_min']
        for actual, expected in [(interval_wait,observed_wait),
                                 (s['queue_passenger_minutes'],observed_wait),
                                 (interval_extra,s['total_extra_after_boarding_passenger_min'])]:
            if not math.isclose(actual,expected,rel_tol=1e-9,abs_tol=1e-6):
                raise RuntimeError(f'Diagnostic time-accounting mismatch: {actual} != {expected}')
    else:
        for key in ('unexplained_wait_passenger_min','queue_peak','queue_passenger_minutes'):
            s[key] = None


def export_replication(result, directory, data):
    directory = Path(directory)
    people = list(passenger_rows(result, data))
    write_csv(directory/'passengers.csv', people, PASSENGER_FIELDS)
    flights = []
    capacity = data.aircraft['battery']['modeled_usable_capacity_kwh']
    for f in result.flights:
        flights.append(dict(flight_id=f.id, aircraft_id=f.aircraft_id, operation=f.estimate.operation.value,
            origin=f.estimate.origin, destination=f.estimate.destination, passenger_count=len(f.passenger_ids),
            passenger_ids='|'.join(f.passenger_ids), takeoff_start_s=f.takeoff_start_s, landing_end_s=f.landing_end_s,
            duration_min=f.estimate.duration_s/60, distance_km=f.estimate.total_distance_km,
            departure_energy_kwh=f.departure_energy_kwh,
            landing_energy_kwh=f.landing_energy_kwh, planned_energy_kwh=f.estimate.energy_kwh,
            departure_soc=f.departure_energy_kwh/capacity if f.departure_energy_kwh is not None else None,
            landing_soc=f.landing_energy_kwh/capacity if f.landing_energy_kwh is not None else None,
            **result.flight_details.get(f.id, {})))
    write_csv(directory/'flights.csv', flights, FLIGHT_FIELDS)
    waits = list(wait_rows(result, data))
    write_csv(directory/'delay_intervals.csv', waits, list(waits[0]) if waits else
              ['passenger_id','group_id','aircraft_id','origin','destination','stage','scope','causes','constraint_locations',
               'aircraft_activity','start_s','end_s','duration_min','overdue_min','extra_after_boarding_min','start_local','end_local'])
    write_csv(directory/'candidate_diagnostics.csv', result.candidate_diagnostics, CANDIDATE_FIELDS)
    causes, combinations = cause_reports(waits, people)
    cause_fields = ['stage','scope','causes','label','exposed_passengers','late_passengers_exposed',
        'late_passengers_blocked_after_limit','fraction_of_late_passengers_exposed','observed_passenger_minutes',
        'overdue_passenger_minutes','extra_after_boarding_passenger_minutes']
    write_csv(directory/'delay_causes.csv', causes, cause_fields)
    write_csv(directory/'delay_combinations.csv', combinations, cause_fields)
    write_csv(directory/'aircraft_activity.csv', result.aircraft_activities,
              ['start_s','end_s','aircraft_id','activity','charging','location','flight_id','operation'])
    write_csv(directory/'queue_intervals.csv', result.queue_intervals,
              ['start_s','end_s','vertiport','waiting_passengers'])
    write_csv(directory/'resource_utilization.csv', result.resource_metrics)
    routes, hours = defaultdict(list), defaultdict(list)
    for p in people:
        routes[p['origin'],p['destination']].append(p)
        hour = result.demand.local_datetime(p['request_time_s']).replace(minute=0, second=0, microsecond=0).isoformat()
        hours[hour,p['origin'],p['destination']].append(p)
    route_rows = [dict(origin=o,destination=d,**cohort_metrics(ps)) for (o,d),ps in sorted(routes.items())]
    hour_rows = [dict(request_hour_local=h,origin=o,destination=d,**cohort_metrics(ps)) for (h,o,d),ps in sorted(hours.items())]
    write_csv(directory/'route_summary.csv', route_rows, ['origin','destination',*cohort_metrics([])])
    write_csv(directory/'hourly_summary.csv', hour_rows, ['request_hour_local','origin','destination',*cohort_metrics([])])
    queue_rows = []
    horizon = result.summary['end_time_s']-result.summary['start_time_s']
    for port in (v['id'] for v in data.network['vertiports']):
        intervals = [r for r in result.queue_intervals if r['vertiport']==port]
        area = sum((r['end_s']-r['start_s'])*r['waiting_passengers'] for r in intervals)
        nonempty = sum(r['end_s']-r['start_s'] for r in intervals if r['waiting_passengers'])
        queue_rows.append(dict(vertiport=port, peak_queue=max((r['waiting_passengers'] for r in intervals),default=0),
                              mean_queue=area/horizon if horizon>0 else None,
                              queue_nonempty_min=nonempty/60, waiting_passenger_min=area/60))
    write_csv(directory/'queue_summary.csv', queue_rows)
    activity_rows = []
    buckets = defaultdict(float)
    for r in result.aircraft_activities:
        buckets[r['aircraft_id'],r['activity']] += r['end_s']-r['start_s']
        if r['charging']: buckets[r['aircraft_id'],'charging_overlay'] += r['end_s']-r['start_s']
    for (aircraft,activity), seconds in sorted(buckets.items()):
        activity_rows.append(dict(aircraft_id=aircraft,activity=activity,minutes=seconds/60,
                                 fraction_of_observed_time=seconds/horizon if horizon else None,
                                 is_overlay=activity=='charging_overlay'))
    write_csv(directory/'aircraft_summary.csv', activity_rows,
              ['aircraft_id','activity','minutes','fraction_of_observed_time','is_overlay'])


# Regularized incomplete beta via a continued fraction. Student t quantiles
# are obtained by bisection, avoiding an additional SciPy installation.
def _beta_fraction(a, b, x):
    tiny = 1e-300
    qab, qap, qam = a+b, a+1, a-1
    c, d = 1., 1.-qab*x/qap
    d = 1./(d if abs(d)>tiny else tiny)
    h = d
    for m in range(1, 401):
        for aa in (m*(b-m)*x/((qam+2*m)*(a+2*m)),
                   -(a+m)*(qab+m)*x/((a+2*m)*(qap+2*m))):
            d = 1.+aa*d
            if abs(d)<tiny: d=tiny
            c = 1.+aa/c
            if abs(c)<tiny: c=tiny
            d = 1./d
            delta = d*c
            h *= delta
        if abs(delta-1) < 3e-14: return h
    raise ArithmeticError('Incomplete beta did not converge.')


def _regularized_beta(x, a, b):
    if x <= 0: return 0.
    if x >= 1: return 1.
    front = math.exp(math.lgamma(a+b)-math.lgamma(a)-math.lgamma(b)+a*math.log(x)+b*math.log1p(-x))
    if x < (a+1)/(a+b+2): return front*_beta_fraction(a,b,x)/a
    return 1-front*_beta_fraction(b,a,1-x)/b


@lru_cache(maxsize=1024)
def t_critical(confidence, df):
    if not 0 < confidence < 1 or df < 1:
        raise ValueError('confidence must be in (0,1), df >= 1')
    target = (1+confidence)/2
    def cdf(t): return 1-0.5*_regularized_beta(df/(df+t*t),df/2,.5)
    low, high = 0., 1.
    while cdf(high) < target: high *= 2
    for _ in range(70):
        middle = (low+high)/2
        if cdf(middle)<target: low=middle
        else: high=middle
    return (low+high)/2


def estimate(values, confidence=.95):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    n = len(values)
    avg = mean(values) if n else None
    sd = stdev(values) if n>1 else None
    half = t_critical(confidence,n-1)*sd/math.sqrt(n) if n>1 else None
    return dict(n=n,mean=avg,std=sd,ci_low=avg-half if half is not None else None,
                ci_high=avg+half if half is not None else None,confidence=confidence)


def _numeric(value):
    if value in (None, '', 'True', 'False') or isinstance(value,bool): return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError,ValueError): return None


def aggregate_experiment(directory, *, output=None, confidence=None, service_target=None):
    """Reads saved replications only; does not need to rerun the simulation."""
    directory = Path(directory)
    snapshot = directory/'config_snapshot.json'
    config = json.loads(snapshot.read_text(encoding='utf-8')) if snapshot.exists() else {}
    confidence = confidence if confidence is not None else config.get('outputs',{}).get('confidence_level',.95)
    if not 0 < confidence < 1: raise ValueError('confidence must be in (0,1)')
    if service_target is not None and not 0 < service_target <= 1:
        raise ValueError('service_target must be in (0,1]')
    runs, grouped, seen = [], defaultdict(list), set()
    for file in sorted(directory.glob('*_fleet*/rep_*/summary.json')):
        payload = json.loads(file.read_text(encoding='utf-8'))
        s = payload['summary']
        ident = (s['scenario'],s['fleet_size'],s['replication_index'])
        if ident in seen: raise ValueError(f'Duplicate replication: {ident}')
        seen.add(ident)
        runs.append((file.parent,s))
        grouped[ident[:2]].append((file.parent,s))
    if not runs: raise ValueError('No replication summary.json files found.')
    output = Path(output) if output else directory/'analysis'
    output.mkdir(parents=True,exist_ok=False)
    (output/'methodology.json').write_text(json.dumps(dict(
        confidence=confidence, independent_unit='replication', interval='two_sided_student_t',
        service_fraction='arithmetic_mean_of_replication_fractions',
        single_replication_confidence_interval=None, service_target=service_target,
        failed_replications='reported_separately_and_excluded_from_completed_run_means',
        cause_minutes='overlap_possible; sum combination minutes instead',
        attribution='observed_constraints_not_counterfactual_causal_effects',
        cohort_time='request_datetime_with_local_date_and_offset'),indent=2)+'\n',encoding='utf-8')
    write_csv(output/'replications.csv', [s for _,s in runs])
    excluded = {'fleet_size','replication_index','schema_version','start_time_s','end_time_s','wait_limit_min'}
    rows = []
    for (scenario,fleet), members in sorted(grouped.items()):
        metrics = sorted({k for _,s in members for k,v in s.items()
                          if k not in excluded and (v is None or _numeric(v) is not None)})
        for metric in metrics:
            completed = [s for _,s in members if s['status']=='completed']
            stats = estimate([_numeric(s.get(metric)) for s in completed],confidence)
            rows.append(dict(scenario=scenario,fleet_size=fleet,metric=metric,total_replications=len(members),
                             completed_replications=len(completed),failed_replications=len(members)-len(completed),
                             zero_demand_replications=sum(s['generated_passengers']==0 for s in completed),**stats))
    write_csv(output/'scenario_summary.csv', rows)
    # Aggregate route/resource/diagnostic summaries with the replication as unit.
    tables = {
        'route_summary': ['origin','destination'],
        'hourly_summary': ['request_hour_local','origin','destination'],
        'resource_utilization': ['vertiport','resource_kind'],
        'queue_summary': ['vertiport'],
        'aircraft_summary': ['aircraft_id','activity'],
        'delay_causes': ['stage','scope','causes'],
        'delay_combinations': ['stage','scope','causes'],
    }
    for name, keys in tables.items():
        agg = []
        for (scenario,fleet), members in sorted(grouped.items()):
            tables_by_run, identities, metric_names = [], set(), set()
            for folder,s in members:
                if s['status'] != 'completed': continue
                file = folder/f'{name}.csv'
                if not file.exists(): continue  # missing old diagnostics are not zeros
                with file.open(encoding='utf-8-sig',newline='') as h: table=list(csv.DictReader(h))
                mapped = {tuple(r[k] for k in keys):r for r in table}
                tables_by_run.append((mapped,s))
                identities.update(mapped)
                metric_names.update(k for r in table for k,v in r.items()
                                    if k not in keys and (v == '' or _numeric(v) is not None))
            for ident in sorted(identities):
                for metric in sorted(metric_names):
                    values = []
                    for table,summary in tables_by_run:
                        if ident in table: values.append(_numeric(table[ident].get(metric)))
                        elif name in ('delay_causes','delay_combinations','aircraft_summary'):
                            # Absence of a cause/activity means zero occurrence;
                            # conditional ratios with no denominator remain undefined.
                            if metric == 'fraction_of_late_passengers_exposed':
                                late = summary['generated_passengers']-summary['boarded_within_limit']
                                values.append(0. if late else None)
                            elif metric == 'fraction_of_observed_time':
                                values.append(0. if summary['end_time_s']>summary['start_time_s'] else None)
                            else:
                                values.append(0.)
                        elif metric in ('generated_passengers','boarded_passengers','completed_journeys','on_time_passengers'):
                            values.append(0.)
                        else: values.append(None)
                    agg.append(dict(scenario=scenario,fleet_size=fleet,**dict(zip(keys,ident)),metric=metric,
                                    **estimate(values,confidence)))
        write_csv(output/f'{name}_aggregate.csv',agg,
                  ['scenario','fleet_size',*keys,'metric','n','mean','std','ci_low','ci_high','confidence'])
    paired = []
    metrics = ['fraction_boarding_within_wait_limit','mean_wait_to_boarding_min',
               'extra_after_boarding_min_mean','flight_energy_per_completed_passenger_kwh']
    for scenario in sorted({k[0] for k in grouped}):
        sizes = sorted(k[1] for k in grouped if k[0]==scenario)
        for a,b in zip(sizes,sizes[1:]):
            left = {s['replication_index']:s for _,s in grouped[scenario,a]}
            right = {s['replication_index']:s for _,s in grouped[scenario,b]}
            for metric in metrics:
                diffs = []
                for index in sorted(left.keys() & right.keys()):
                    x,y = left[index],right[index]
                    if x['status']!='completed' or y['status']!='completed': continue
                    if not x.get('demand_fingerprint') or x['demand_fingerprint']!=y.get('demand_fingerprint'):
                        continue
                    xv,yv = _numeric(x.get(metric)),_numeric(y.get(metric))
                    if xv is not None and yv is not None: diffs.append(yv-xv)
                paired.append(dict(scenario=scenario,fleet_a=a,fleet_b=b,metric=metric,
                                   difference='fleet_b_minus_fleet_a',**estimate(diffs,confidence)))
    write_csv(output/'paired_fleet_differences.csv',paired,
              ['scenario','fleet_a','fleet_b','metric','difference','n','mean','std','ci_low','ci_high','confidence'])
    if service_target is not None:
        choices = []
        for scenario in sorted({k[0] for k in grouped}):
            eligible = [r for r in rows if r['scenario']==scenario and r['metric']=='fraction_boarding_within_wait_limit'
                        and r['failed_replications']==0 and r['n']>0
                        and r['n']==r['total_replications']-r['zero_demand_replications']]
            for statistic,criterion in [('mean','mean_of_replication_service_fractions'),
                                        ('ci_low','lower_two_sided_t_confidence_bound')]:
                feasible = [r for r in eligible if r[statistic] is not None and r[statistic]>=service_target]
                choices.append(dict(scenario=scenario,service_target=service_target,
                                    minimum_tested_fleet=min((r['fleet_size'] for r in feasible),default=None),
                                    criterion=criterion,
                                    note='Exploratory comparison of tested sizes; not a final agreed fleet-selection rule.'))
        write_csv(output/'fleet_selection.csv',choices)
    return output


def main(argv=None):
    parser=argparse.ArgumentParser(description='CSV aggregation of completed eVTOL simulation runs.')
    parser.add_argument('directory',type=Path)
    parser.add_argument('--output',type=Path,help='New analysis folder; existing folders are preserved.')
    parser.add_argument('--confidence',type=float)
    parser.add_argument('--service-target',type=float,help='Explicit target as fraction, e.g. 0.95; no default.')
    args=parser.parse_args(argv)
    try:
        output=aggregate_experiment(args.directory,output=args.output,confidence=args.confidence,service_target=args.service_target)
        print(f'CSV reports: {output.resolve()}')
        return 0
    except (ValueError,OSError,KeyError,ArithmeticError) as exc:
        parser.exit(2,f'ERROR: {exc}\n')

# Compact report API. Legacy detailed exporters above are available only when
# explicitly requested by the runner; the default study calls this API.
COMPACT_META = ('scenario','fleet_role','fleet_size','n_opt','replication_index',
    'base_seed','seed_components','numpy_version','status','schema_version',
    'diagnostics_enabled','demand_fingerprint','wait_limit_min','service_target',
    'start_datetime','end_datetime','input_sha256','config_sha256','code_sha256')


def excel_value(value):
    """Serbian Excel CSV: semicolon columns, decimal comma, UTF-8 BOM."""
    if value is None: return ''
    if isinstance(value,bool): return '1' if value else '0'
    if isinstance(value,(float,np.floating)):
        if not math.isfinite(value): return ''
        return repr(float(value)).replace('.',',')
    return value


def write_excel_csv(path, rows, fields):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    try:
        with temporary.open('w',encoding='utf-8-sig',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=fields,delimiter=';',lineterminator='\r\n')
            writer.writeheader()
            for row in rows:
                writer.writerow({key:excel_value(value) for key,value in row.items()})
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def compact_replication(result, data, *, config_hash='', code_hash='', target=.95):
    row=dict(result.summary)
    row.update(base_seed=data.config['run']['base_seed'],
               seed_components=json.dumps(dict(result.demand.seed_components),separators=(',',':')),
               numpy_version=result.demand.numpy_version,service_target=target,
               input_sha256=json.dumps(data.input_sha256,sort_keys=True,separators=(',',':')),
               config_sha256=config_hash,code_sha256=code_hash)
    return row


def compact_summary(rows, *, scenario, role, fleet_size, n_opt, expected_replications,
                    target, confidence=.95, selection_status='selected', search_evidence=''):
    completed=[r for r in rows if r['status']=='completed']
    result=dict(scenario=scenario,fleet_role=role,fleet_size=fleet_size,n_opt=n_opt,
                service_target=target,selection_status=selection_status,
                expected_replications=expected_replications,observed_replications=len(rows),
                completed_replications=len(completed),
                failed_replications=len(rows)-len(completed),
                zero_demand_replications=sum(r['generated_passengers']==0 for r in completed),
                confidence=confidence,search_evidence=search_evidence,
                service_statistic='mean_of_replication_on_time_fractions')
    for field in ('base_seed','seed_components','numpy_version','input_sha256','config_sha256','code_sha256'):
        result[field] = rows[0].get(field) if rows else None
    # Seed components vary by replication: do not present replication 0 as a
    # configuration-wide seed. The base seed plus the rule is sufficient here.
    result['seed_components']='base_seed,replication_index,scenario_id,stream_id'
    if rows:
        metrics=[key for key in rows[0] if key not in COMPACT_META]
        for metric in metrics:
            stat=estimate([_numeric(row.get(metric)) for row in completed],confidence)
            for suffix in ('mean','std','ci_low','ci_high','n'):
                result[f'{metric}__{suffix}']=stat[suffix]
            if 'max' in metric or 'queue_peak' in metric:
                values=[_numeric(row.get(metric)) for row in completed]
                result[f'{metric}__worst']=max((v for v in values if v is not None),default=None)
    fractions=[r['fraction_boarding_within_wait_limit'] for r in completed
               if r['fraction_boarding_within_wait_limit'] is not None]
    result['meets_service_target']=(mean(fractions)>=target if fractions and len(rows)==expected_replications
                                    and len(completed)==len(rows) else None)
    return result


def write_compact_pair(directory, scenario, role, fleet_size, n_opt, rows, *, expected_replications,
                       target, confidence=.95, selection_status='selected', search_evidence=''):
    """One CSV row per replication; exactly one wide summary row."""
    directory=Path(directory)
    prepared=[dict(r,fleet_role=role,n_opt=n_opt) for r in rows]
    leading=('scenario','fleet_role','fleet_size','n_opt','replication_index','status')
    fields=[*leading,*[k for k in (prepared[0] if prepared else {}) if k not in COMPACT_META],
            *[k for k in COMPACT_META if k not in leading]]
    stem=f'{scenario}_dem_{role}_reps'
    write_excel_csv(directory/f'{stem}.csv',prepared,fields)
    summary=compact_summary(prepared,scenario=scenario,role=role,fleet_size=fleet_size,n_opt=n_opt,
        expected_replications=expected_replications,target=target,confidence=confidence,
        selection_status=selection_status,search_evidence=search_evidence)
    write_excel_csv(directory/f'{stem}_summary.csv',[summary],list(summary))
    return summary


if __name__=='__main__':
    raise SystemExit(main())
