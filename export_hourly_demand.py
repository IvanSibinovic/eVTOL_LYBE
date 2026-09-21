#!/usr/bin/env python3
"""Export the existing simulation's demand to Excel-compatible CSV files.

Place beside run_sim.py and run: python export_hourly_demand.py
Uses the original generator, config.json and all three JSON inputs.
Does not run dispatch/service simulation or modify any model/input file.
Defaults: low scenario, 6 aircraft, 100 replications (indices 0..99).
CSV: UTF-8 BOM, semicolon separator, decimal comma for Serbian Excel.
Optional exact validation: --reference "low_dem_n6_reps(3).csv"
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path


def write_csv(path, headers, rows):
    def excel_value(value):
        return repr(value).replace('.', ',') if isinstance(value, float) else value
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.writer(handle, delimiter=';')
        writer.writerow(headers)
        writer.writerows([[excel_value(v) for v in row] for row in rows])


def read_reference(path, scenario, fleet, replications):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        sample = handle.read(16384)
        handle.seek(0)
        delimiter = ';' if sample.splitlines()[0].count(';') > sample.splitlines()[0].count(',') else ','
        rows = list(csv.DictReader(handle, delimiter=delimiter))
    selected = [r for r in rows if r['scenario'] == scenario and int(r['fleet_size']) == fleet]
    indexed = {}
    for row in selected:
        index = int(row['replication_index'])
        if index in indexed:
            raise ValueError(f'Duplicate reference replication: {index}')
        indexed[index] = row
    if set(indexed) != set(range(replications)):
        raise ValueError('Reference must contain exactly the requested replication indices (0..N-1).')
    return indexed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--config', type=Path, help='Default: PROJECT/config.json')
    parser.add_argument('--replications', type=int, default=100)
    parser.add_argument('--scenario', default='low')
    parser.add_argument('--fleet-size', type=int, default=6)
    parser.add_argument('--reference', type=Path, help='Original per-replication results CSV; mismatch stops export')
    parser.add_argument('--allow-fingerprint-mismatch', action='store_true',
                        help='Export a clearly marked regeneration even if exact fingerprint differs; counts must still match')
    parser.add_argument('--output-dir', type=Path, help='Default: PROJECT/outputs/demand_timeline')
    args = parser.parse_args()
    if args.replications < 1:
        parser.error('--replications must be positive')
    project = args.project_dir.resolve()
    sys.path.insert(0, str(project))
    from src.input_data import load_inputs
    from src.demand import generate_demand

    data = load_inputs(args.config or project / 'config.json', overrides={
        'mode': 'single_configuration', 'fleet_size': args.fleet_size,
        'replications': args.replications, 'demand_scenario': args.scenario})
    reference = read_reference(args.reference, args.scenario, args.fleet_size, args.replications) if args.reference else None
    hourly_counts = Counter()
    requests, checks = [], []
    first_hour, last_hour = 0, 23  # Keep all hours of the schedule day, including empty ones.
    epoch = None
    matched_fingerprints = 0
    for rep in range(args.replications):
        demand = generate_demand(data, scenario=args.scenario, replication_index=rep)
        epoch = demand.epoch
        fingerprint = hashlib.sha256(json.dumps([asdict(r) for r in demand.records],
            sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if reference:
            row = reference[rep]
            if len(demand.records) != int(row['generated_passengers']):
                raise ValueError(f'Passenger count differs at replication {rep}; no output written.')
            matched_fingerprints += fingerprint == row['demand_fingerprint']
            if fingerprint != row['demand_fingerprint'] and not args.allow_fingerprint_mismatch:
                raise ValueError(f'Demand differs from saved results at replication {rep}; no output written.')
        airport_count = city_count = 0
        for record in demand.records:
            if record.direction not in ('arrival', 'departure'):
                raise ValueError(f'Unknown direction: {record.direction}')
            direction = 0 if record.direction == 'arrival' else 1
            hour = math.floor(record.request_time_s / 3600)
            first_hour, last_hour = min(first_hour, hour), max(last_hour, hour)
            hourly_counts[rep, hour, direction] += 1
            airport_count += direction == 0
            city_count += direction == 1
            requests.append([rep, record.passenger_id, record.airline_flight_id,
                record.origin, record.destination,
                demand.local_datetime(record.request_time_s).isoformat(timespec='microseconds'),
                record.request_time_s, hour])
        if reference and (airport_count != int(row['generated_airport_to_city_passengers'])
                          or city_count != int(row['generated_city_to_airport_passengers'])):
            raise ValueError(f'Directional passenger counts differ at replication {rep}; no output written.')
        status = ('Подудара се' if fingerprint == row['demand_fingerprint'] else 'Бројеви се подударају; SHA256 се разликује') if reference else 'Није упоређено'
        checks.append([rep, len(demand.records), airport_count, city_count, fingerprint, status])
    # Elapsed-hour bins are valid local clock hours for this date (no DST transition).
    if epoch.utcoffset() != (epoch + timedelta(hours=last_hour + 1)).utcoffset() or epoch.utcoffset() != (epoch + timedelta(hours=first_hour)).utcoffset():
        raise ValueError('This export requires a period without a daylight-saving transition.')
    hours, per_rep = [], []
    for hour in range(first_hour, last_hour + 1):
        start = demand.local_datetime(hour * 3600)
        end = demand.local_datetime((hour + 1) * 3600)
        a = sum(hourly_counts[rep, hour, 0] for rep in range(args.replications))
        c = sum(hourly_counts[rep, hour, 1] for rep in range(args.replications))
        label = start.strftime('%d.%m. %H:%M') + '–' + end.strftime('%H:%M')
        hours.append([label, a / args.replications, c / args.replications,
                      (a + c) / args.replications, a, c, args.replications])
        for rep in range(args.replications):
            x, y = hourly_counts[rep, hour, 0], hourly_counts[rep, hour, 1]
            per_rep.append([rep, label, x, y, x + y])
    total = len(requests)
    if sum(r[4] + r[5] for r in hours) != total:
        raise AssertionError('Hourly counts do not reconcile')
    output = (args.output_dir or project / 'outputs' / 'demand_timeline').resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'hourly_demand.csv',
        ['Временски интервал', 'Аеродром → град (просек)', 'Град → аеродром (просек)',
         'Укупно (просек)', 'Аеродром → град (збир)', 'Град → аеродром (збир)', 'Број репликација'], hours)
    write_csv(output / 'hourly_by_replication.csv',
        ['Репликација (индекс)', 'Временски интервал', 'Аеродром → град', 'Град → аеродром', 'Укупно'], per_rep)
    write_csv(output / 'passenger_requests.csv',
        ['Репликација (индекс)', 'Путник', 'Авионски лет', 'Полазиште', 'Одредиште',
         'Локално време захтева (ISO 8601)', 'Време захтева (s од поноћи)', 'Сат од поноћи'], requests)
    write_csv(output / 'replication_checks.csv',
        ['Репликација (индекс)', 'Путници', 'Аеродром → град', 'Град → аеродром', 'SHA256 тражње', 'Провера'], checks)
    metadata = {
        'description': 'Passenger requests, not boarding or flight departure times. Hour bins are [start, end).',
        'method': 'Original src.demand.generate_demand; all replications included in every hourly mean, including zeros.',
        'schedule_date': data.schedule['date'], 'timezone': data.schedule['timezone'],
        'scenario': args.scenario, 'replications': args.replications,
        'base_seed': data.config['run']['base_seed'], 'numpy_version': demand.numpy_version,
        'input_sha256': data.input_sha256,
        'reference': args.reference.name if args.reference else None,
        'matched_fingerprints': matched_fingerprints if reference else None,
        'matched_count_replications': args.replications if reference else None,
        'total_passengers': total, 'mean_passengers': total / args.replications,
        'hours': hours, 'checks': checks,
    }
    (output / 'demand_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Export complete: {output}')
    print(f'Replications: {args.replications}; passengers: {total}; mean: {total / args.replications:.2f}')
    print(f'Exact fingerprint matches: {matched_fingerprints}/{args.replications}; directional counts match in all replications.' if reference else 'No reference CSV supplied; exact match not checked.')
    print('Open hourly_demand.csv in Excel. Plot the first three columns as stacked columns.')


if __name__ == '__main__':
    main()
