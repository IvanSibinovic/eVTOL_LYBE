from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from statistics import NormalDist
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import numpy as np

from .entities import Passenger
from .flight_model import FlightModel

if TYPE_CHECKING:
    from .input_data import InputData


@dataclass(frozen=True, slots=True)
class DemandRecord:
    """Непроменљив план једног путника, без резултата опслуживања."""
    passenger_id: str
    airline_flight_id: str
    direction: str
    origin: str
    destination: str
    scheduled_time_s: float
    request_time_s: float
    terminal_access_duration_s: float
    terminal_processing_s: float | None
    terminal_earliness_s: float | None
    latest_terminal_arrival_s: float | None
    latest_desired_takeoff_s: float | None
    planned_boarding_s: float
    planned_disembarking_s: float
    planned_flight_s: float

    def to_passenger(self) -> Passenger:
        return Passenger(
            id=self.passenger_id, airline_flight_id=self.airline_flight_id,
            origin=self.origin, destination=self.destination,
            request_time_s=self.request_time_s,
            latest_desired_takeoff_s=self.latest_desired_takeoff_s,
            latest_terminal_arrival_s=self.latest_terminal_arrival_s,
            terminal_access_duration_s=self.terminal_access_duration_s)


@dataclass(frozen=True, slots=True)
class DemandRealization:
    scenario: str
    replication_index: int
    epoch: datetime
    records: tuple[DemandRecord, ...]
    expected_passengers: float
    flight_counts: tuple[tuple[str, int], ...]
    seed_components: tuple[tuple[str, tuple[int, ...]], ...]
    numpy_version: str

    @property
    def start_time_s(self) -> float:
        """Први захтев; за празну реализацију почетак је поноћ (0)."""
        return self.records[0].request_time_s if self.records else 0.0

    def local_datetime(self, seconds: float) -> datetime:
        # Сабирање по UTC оси чува протекло време и при промени UTC помака.
        return (self.epoch.astimezone(timezone.utc) + timedelta(seconds=seconds)).astimezone(self.epoch.tzinfo)

    def new_passengers(self) -> list[Passenger]:
        """Нови објекти за сваку флоту: нема дељења променљивог стања."""
        return [record.to_passenger() for record in self.records]


def _nonnegative_integer(value: int, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f'{label}: очекиван ненегативан цео број.')
    return value


def _validate_rules(data: InputData) -> None:
    c = data.config
    rng = c['randomness']
    expected = {
        'generator': 'numpy_PCG64', 'seed_strategy': 'SeedSequence',
        'flight_iteration_order': 'sorted_by_flight_id',
        'seed_components': ['base_seed', 'replication_index', 'scenario_id', 'stream_id'],
    }
    for key, value in expected.items():
        if rng[key] != value:
            raise ValueError(f'randomness.{key}: неподржана поставка.')
    if (rng['fleet_size_in_seed'] is not False or rng['independent_replications'] is not True
            or rng['reuse_same_generated_demand_across_fleet_sizes'] is not True):
        raise ValueError('Тражња мора бити независна од величине флоте, уз независне репликације.')
    d = c['demand']
    if (d['flight_delays_enabled'] is not False
            or d['passenger_generation']['distribution'] != 'binomial'
            or d['passenger_generation']['trials_field'] != 'seat_capacity'
            or d['passenger_generation']['success_probability_rule'] !=
                'load_factor * (1 - transfer_share) * selected_adoption_share'
            or d['passenger_generation']['round_expected_passenger_counts'] is not False
            or d['city_assignment']['distribution'] != 'categorical_per_passenger'
            or d['city_assignment']['same_shares_in_both_directions'] is not True):
        raise ValueError('Неподржано правило генерисања броја/локације путника или кашњења авиона.')
    planning = d['city_to_airport']['journey_planning']
    for key, expected in {
        'planning_passenger_count': 'aircraft_capacity',
        'planned_access_time': 'use_same_sample_as_realized_terminal_access',
        'desired_terminal_arrival_rule': 'scheduled_off_block - terminal_earliness',
        'latest_desired_takeoff_rule': 'desired_terminal_arrival - terminal_access - planned_disembarking - flight_duration',
        'request_time_rule': 'desired_terminal_arrival - terminal_access - planned_disembarking - flight_duration - departure_finalization - planned_boarding - planned_wait',
    }.items():
        if planning[key] != expected:
            raise ValueError(f'journey_planning.{key}: неподржано правило.')
    if d['airport_to_city']['request_time_rule'] != 'scheduled_on_block + terminal_processing + terminal_access':
        raise ValueError('Неподржано правило захтева на аеродрому.')
    if d['city_to_airport']['terminal_earliness_min']['sampling_method'] != 'inverse_cdf_or_rejection':
        raise ValueError('Неподржан метод узорковања одсечене нормалне расподеле.')


def _triangular_seconds(generator: np.random.Generator, profile: dict, count: int) -> np.ndarray:
    return 60 * generator.triangular(profile['minimum'], profile['mode'], profile['maximum'], size=count)


def _truncated_normal_seconds(generator: np.random.Generator, profile: dict, count: int) -> np.ndarray:
    """Инверзна CDF: одсецање расподеле, без лепљења узорака на границе."""
    normal = NormalDist(profile['mean'], profile['stddev'])
    lo, hi = normal.cdf(profile['minimum']), normal.cdf(profile['maximum'])
    if not 0 < lo < hi < 1:
        raise ValueError('Границе одсечене нормалне расподеле нису нумерички разрешиве.')
    uniforms = generator.uniform(lo, hi, size=count)
    return np.fromiter((60 * normal.inv_cdf(float(u)) for u in uniforms), dtype=float, count=count)


def generate_demand(data: InputData, *, scenario: str | None = None,
                    replication_index: int = 0) -> DemandRealization:
    """Улаз је резултат load_inputs(); један позив генерише једну реализацију.

    Подразумевани сценарио је run.demand_scenario. Ни величина флоте ни број
    репликација не улазе у seed. Резултат се може поново користити преко
    new_passengers(), без преношења стања опслуживања између флота.
    """
    _validate_rules(data)
    replication_index = _nonnegative_integer(replication_index, 'replication_index')
    c, d, random = data.config, data.config['demand'], data.config['randomness']
    scenario = c['run']['demand_scenario'] if scenario is None else scenario
    if scenario not in d['adoption_share'] or scenario not in random['scenario_ids']:
        raise ValueError(f'Непознат сценарио: {scenario}.')
    base_seed = _nonnegative_integer(c['run']['base_seed'], 'base_seed')
    scenario_id = _nonnegative_integer(random['scenario_ids'][scenario], 'scenario_id')
    streams, seed_components = {}, []
    for name in ('passenger_counts', 'city_assignment', 'arrival_terminal_processing',
                 'terminal_access', 'departure_terminal_earliness'):
        stream_id = _nonnegative_integer(random['stream_ids'][name], f'stream_ids.{name}')
        components = (base_seed, replication_index, scenario_id, stream_id)
        seed_components.append((name, components))
        streams[name] = np.random.Generator(np.random.PCG64(np.random.SeedSequence(components)))

    airports = [v['id'] for v in data.network['vertiports'] if v['role'] == 'airport']
    if len(airports) != 1:
        raise ValueError('Потребан је тачно један аеродромски вертипорт.')
    airport = airports[0]
    city_ids = sorted(d['city_assignment']['shares'])
    shares = [d['city_assignment']['shares'][city] for city in city_ids]
    flight_model = FlightModel.from_inputs(data)
    flight_times = {(a, b): flight_model.estimate(a, b).duration_s
                    for city in city_ids for a, b in ((airport, city), (city, airport))}
    seats = data.aircraft['capacity']['passenger_seats']
    handling = c['ground_handling']
    boarding = 60 * (handling['boarding']['fixed_min'] + seats * handling['boarding']['per_passenger_min'])
    disembarking = 60 * (handling['disembarking']['fixed_min'] + seats * handling['disembarking']['per_passenger_min'])
    preparation = 60 * handling['departure_finalization_min']
    planned_wait = 60 * c['service']['max_wait_to_boarding_min']
    zone = ZoneInfo(data.schedule['timezone'])
    epoch = datetime.combine(datetime.fromisoformat(data.schedule['date']).date(), time(), tzinfo=zone)
    epoch_timestamp = epoch.timestamp()
    records, counts, expectations = [], [], []
    for flight in sorted(data.flights, key=lambda f: f['flight_id']):
        probability = data.calibration['load_factor'] * (1 - flight['transfer_share']) * d['adoption_share'][scenario]
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError(f"{flight['flight_id']}: неисправна вероватноћа избора путника.")
        count = int(streams['passenger_counts'].binomial(flight['seat_capacity'], probability))
        counts.append((flight['flight_id'], count))
        expectations.append(flight['seat_capacity'] * probability)
        cities = streams['city_assignment'].choice(city_ids, size=count, p=shares)
        access = _triangular_seconds(streams['terminal_access'], d['terminal_access_min'], count)
        scheduled_s = flight['scheduled_datetime'].timestamp() - epoch_timestamp
        is_arrival = flight['direction'] == 'arrival'
        if is_arrival:
            times = _triangular_seconds(streams['arrival_terminal_processing'],
                                       d['airport_to_city']['terminal_processing_min'], count)
        else:
            profile = d['city_to_airport']['terminal_earliness_min']['profiles'][flight['haul_category']]
            times = _truncated_normal_seconds(streams['departure_terminal_earliness'], profile, count)
        for index in range(count):
            city, access_s, sampled_s = str(cities[index]), float(access[index]), float(times[index])
            origin, destination = (airport, city) if is_arrival else (city, airport)
            duration = flight_times[origin, destination]
            if is_arrival:
                request = scheduled_s + sampled_s + access_s
                desired_terminal = desired_takeoff = None
            else:
                desired_terminal = scheduled_s - sampled_s
                desired_takeoff = desired_terminal - access_s - disembarking - duration
                request = desired_takeoff - preparation - boarding - planned_wait
            records.append(DemandRecord(
                passenger_id=f"{flight['flight_id']}:P{index + 1:04d}",
                airline_flight_id=flight['flight_id'], direction=flight['direction'],
                origin=origin, destination=destination, scheduled_time_s=scheduled_s,
                request_time_s=request, terminal_access_duration_s=access_s,
                terminal_processing_s=sampled_s if is_arrival else None,
                terminal_earliness_s=None if is_arrival else sampled_s,
                latest_terminal_arrival_s=desired_terminal,
                latest_desired_takeoff_s=desired_takeoff,
                planned_boarding_s=boarding, planned_disembarking_s=disembarking,
                planned_flight_s=duration))
    records.sort(key=lambda r: (r.request_time_s, r.passenger_id))
    return DemandRealization(scenario, replication_index, epoch, tuple(records), math.fsum(expectations),
                             tuple(counts), tuple(seed_components), np.__version__)


def export_demand(realization: DemandRealization, path: Path, data: InputData) -> None:
    """CSV плана и суседни .metadata.json са seed-овима и верзијом NumPy."""
    path = Path(path)
    if path.suffix.lower() != '.csv':
        raise ValueError('Излаз мора имати наставак .csv.')
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(DemandRecord.__dataclass_fields__) + ['request_datetime', 'desired_terminal_datetime', 'desired_takeoff_datetime']
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in realization.records:
            row = asdict(record)
            for key, seconds in (('request_datetime', record.request_time_s),
                                 ('desired_terminal_datetime', record.latest_terminal_arrival_s),
                                 ('desired_takeoff_datetime', record.latest_desired_takeoff_s)):
                row[key] = realization.local_datetime(seconds).isoformat() if seconds is not None else ''
            writer.writerow(row)
    metadata = {
        'scenario': realization.scenario, 'replication_index': realization.replication_index,
        'epoch': realization.epoch.isoformat(), 'numpy_version': realization.numpy_version,
        'generator': 'PCG64', 'seed_components': dict(realization.seed_components),
        'generated_passengers': len(realization.records), 'expected_passengers': realization.expected_passengers,
        'flight_counts': dict(realization.flight_counts), 'input_sha256': data.input_sha256,
        'effective_config': data.config,
        'note': 'План тражње; временски рокови нису стварна времена извршења нити доказ пропуштеног авионског лета.',
    }
    path.with_suffix('.metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main(argv: list[str] | None = None) -> int:
    from .input_data import InputDataError, load_inputs

    parser = argparse.ArgumentParser(description='Генерисање тражње за једну репликацију, без опслуживања.')
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'config.json')
    parser.add_argument('--scenario', help='low=1%%, medium=3%%, high=5%%; иначе run.demand_scenario.')
    parser.add_argument('--replication', type=int, default=0)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--output', type=Path, help='Опциони CSV; додаје се и .metadata.json.')
    args = parser.parse_args(argv)
    try:
        overrides = {}
        if args.scenario is not None:
            overrides['demand_scenario'] = args.scenario
        if args.seed is not None:
            overrides['base_seed'] = args.seed
        data = load_inputs(args.config, overrides=overrides)
        result = generate_demand(data, replication_index=args.replication)
        share = data.config['demand']['adoption_share'][result.scenario]
        print(f'Сценарио: {result.scenario} ({share:.0%} O&D); репликација: {result.replication_index}.')
        print(f'Очекивана тражња: {result.expected_passengers:.3f}; генерисано: {len(result.records)} путника.')
        directions = Counter(r.direction for r in result.records)
        print(f"Аеродром → град: {directions['arrival']}; град → аеродром: {directions['departure']}.")
        if result.records:
            print(f'Први захтев: {result.local_datetime(result.start_time_s).isoformat(timespec="seconds")}.')
            print(f'Последњи захтев: {result.local_datetime(result.records[-1].request_time_s).isoformat(timespec="seconds")}.')
        else:
            print('Нема захтева; показатељ нивоа услуге за ову репликацију није дефинисан.')
        print('Градски вертипорт    Аеродром → град    Град → аеродром')
        for city in sorted(data.config['demand']['city_assignment']['shares']):
            arrivals = sum(r.direction == 'arrival' and r.destination == city for r in result.records)
            departures = sum(r.direction == 'departure' and r.origin == city for r in result.records)
            print(f'{city:18} {arrivals:15} {departures:18}')
        if args.output:
            export_demand(result, args.output, data)
            print(f'Сачувано: {args.output} и {args.output.with_suffix(".metadata.json")}.')
        print('Путници су генерисани; њихово опслуживање још није симулирано.')
    except (InputDataError, ValueError, KeyError, OSError) as exc:
        print(f'ГРЕШКА: {exc}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    raise SystemExit(main())
