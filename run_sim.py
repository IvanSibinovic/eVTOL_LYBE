"""Провера улаза и покретање независних репликација eVTOL симулације."""

from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.input_data import InputData, InputDataError, load_inputs


def print_summary(data: InputData) -> None:
    c, n, a = data.config, data.network, data.aircraft
    flights = data.flights
    arrivals = sum(f['direction'] == 'arrival' for f in flights)
    print('eVTOLs_LYBE — провера улазних података')
    print(f"Улази су исправни. Датум: {data.schedule['date']}; зона: {data.schedule['timezone']}.")
    print('Ред летења: синтетички сценарио.')
    print(f'Авионски летови: {len(flights)} ({arrivals} долазака, {len(flights)-arrivals} одлазака).')
    print(f"Седишта: {data.calibration['total_seats']:,}; рачунска попуњеност: {data.calibration['load_factor']:.4%}.")
    print(f"Мрежа: {len(n['vertiports'])} вертипортова, {len(n['transit_nodes'])} транзитна чвора, {len(n['routes'])} двосмерних рута.")
    print(f"Летелица: {a['manufacturer']} {a['model']}; {a['capacity']['passenger_seats']} путничка места; "
          f"{a['battery']['modeled_usable_capacity_kwh']:g} kWh; {a['performance']['operational_cruise_speed_kmh']:g} km/h.")
    print(f"Режим: {c['run']['mode']}; понављања по конфигурацији: {c['run']['replications']}; seed: {c['run']['base_seed']}.")
    print('Планиране конфигурације:')
    for item in data.configurations:
        scenario = item['demand_scenario']
        share = c['demand']['adoption_share'][scenario]
        expected = data.calibration['expected_evtol_by_scenario'][scenario]
        allocation = ', '.join(f'{key}: {count}' for key, count in item['initial_allocation'].items())
        print(f"  Флота {item['fleet_size']}, {scenario} ({share:.0%} O&D): очекивано {expected:.3f} путничких вожњи; {allocation}.")
    total_runs = sum(item['replications'] for item in data.configurations)
    print(f'Укупно планирано понављања: {total_runs}.')
    print('Ово су очекиване вредности тражње; стварни број се генерише за сваку репликацију.')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Компактни eVTOL експеримент: N−1, N*, N+1 за три нивоа тражње.')
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parent / 'config.json')
    parser.add_argument('--mode', choices=('fleet_study', 'single_configuration'),
                        help='Подразумевано fleet_study: претрага и три суседне флоте.')
    parser.add_argument('--fleet-size', type=int, help='Само за single_configuration.')
    parser.add_argument('--scenario', dest='demand_scenario', choices=('low','medium','high'),
                        help='Ограничи претрагу на један ниво; подразумевано сва три.')
    parser.add_argument('--replications', type=int)
    parser.add_argument('--seed', dest='base_seed', type=int)
    parser.add_argument('--service-target', type=float, help='Подразумевано циљ из config.json: 0.95.')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--detailed', action='store_true', default=None,
                        help='Укључи подробне CSV дневнике у debug; подразумевано искључено.')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--resume', type=Path, help='Настави прекинути експеримент из његовог директоријума.')
    args = parser.parse_args(argv)
    from src.experiment import read_checkpoint_settings, run_experiment, digest, CHECKPOINT
    settings = {}
    directory = None
    try:
        if args.resume:
            if args.output_dir: parser.error('--resume и --output-dir не користе се заједно.')
            settings = read_checkpoint_settings(args.resume.resolve())
        mode = args.mode or settings.get('mode','fleet_study')
        fleet = args.fleet_size if args.fleet_size is not None else settings.get('fleet_size')
        if fleet is not None and mode!='single_configuration':
            parser.error('--fleet-size користити уз --mode single_configuration.')
        repetitions = args.replications if args.replications is not None else settings.get('replications')
        overrides = {'mode':'experiment_grid' if mode=='fleet_study' else 'single_configuration'}
        for key,value in [('fleet_size',fleet),('replications',repetitions),
                          ('base_seed',args.base_seed if args.base_seed is not None else settings.get('base_seed')),
                          ('demand_scenario',args.demand_scenario or settings.get('run_demand_scenario'))]:
            if value is not None: overrides[key]=value
        data = load_inputs(args.config, overrides=overrides)
        repetitions = data.config['run']['replications']
        fleet = data.config['run']['fleet_size'] if mode=='single_configuration' else None
        scenarios = ([args.demand_scenario] if args.demand_scenario else settings.get('scenarios')
                     or (['low','medium','high'] if mode=='fleet_study' else [data.config['run']['demand_scenario']]))
        target = args.service_target if args.service_target is not None else settings.get('target',data.config['service']['target_on_time_fraction'])
        if not 0<target<=1: parser.error('--service-target мора бити у интервалу (0,1].')
        detailed = args.detailed if args.detailed is not None else settings.get('detailed',False)
        print('eVTOLs_LYBE — компактни експеримент')
        print(f'Улази исправни; сценарији: {", ".join(scenarios)}; понављања: {repetitions}; seed: {data.config["run"]["base_seed"]}.')
        print(f'Циљ: средњи удео услуге у року {target:.0%}; чекање до B ≤ {data.config["service"]["max_wait_to_boarding_min"]:g} min.')
        if mode=='fleet_study':
            print(f'Претрага флота: {data.config["experiment"]["fleet_sizes"]}; извоз N−1, N*, N+1.')
            print(f'Коначни извештаји: {len(scenarios)*6} CSV датотека, без директоријума по репликацији.')
        else:
            print(f'Проба флоте {fleet}; два CSV фајла по нивоу тражње, без проглашавања N*.')
        print('CSV: тачка-зарез за колоне, децимални зарез; детаљни дневници '+('укључени.' if detailed else 'искључени.'))
        if args.check_only: return 0
        directory = (args.resume or args.output_dir or data.output_directory /
                     datetime.now(timezone.utc).strftime('study_%Y%m%dT%H%M%S_%fZ')).resolve()
        project = Path(__file__).resolve().parent
        hashes = {str(p.relative_to(project)):hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in [Path(__file__).resolve(),*sorted((project/'src').glob('*.py'))]}
        print(f'Резултати: {directory}',flush=True)
        outcome = run_experiment(data,directory,scenarios=scenarios,replications=repetitions,target=target,
            code_hash=digest(hashes),mode=mode,fleet_size=fleet,detailed=detailed,resume=bool(args.resume))
        pending = (directory/CHECKPOINT).exists()
        print('Завршено. '+('Погледати статусе: циљ/суседна флота нису потврђени или постоји незавршена репликација.'
                           if pending else 'Сачувани су само договорени парови CSV извештаја.'))
        return 3 if pending else 0
    except KeyboardInterrupt:
        if directory is not None:
            print(f'Прекинуто. Сачувана понављања наставити командом: python run_sim.py --resume "{directory}"',file=sys.stderr)
        else:
            print('Прекинуто пре покретања експеримента.',file=sys.stderr)
        return 130
    except (InputDataError,ValueError,RuntimeError,OSError) as exc:
        print(f'ГРЕШКА: {exc}',file=sys.stderr)
        return 2


if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    raise SystemExit(main())
