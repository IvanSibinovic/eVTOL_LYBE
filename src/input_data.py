from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


class InputDataError(ValueError):
    """Грешка у улазном фајлу, намењена приказу кориснику."""


@dataclass
class InputData:
    config: dict
    aircraft: dict
    network: dict
    schedule: dict
    flights: list[dict]
    configurations: list[dict]
    calibration: dict
    input_paths: dict[str, Path]
    input_sha256: dict[str, str]
    output_directory: Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise InputDataError(message)


def field(obj: dict, key: str, label: str) -> Any:
    require(isinstance(obj, dict), f"{label}: очекиван је JSON објекат.")
    require(key in obj, f"{label}: недостаје поље '{key}'.")
    return obj[key]


def number(value: Any, label: str, minimum: float = 0,
           maximum: float | None = None, integer: bool = False) -> float:
    require(type(value) in (int, float) and math.isfinite(value),
            f"{label}: очекиван је коначан број.")
    require(not integer or type(value) is int, f"{label}: очекиван је цео број.")
    require(value >= minimum and (maximum is None or value <= maximum),
            f"{label}: вредност {value} је ван дозвољеног опсега.")
    return value


def text(value: Any, label: str) -> str:
    require(isinstance(value, str) and bool(value.strip()), f"{label}: очекиван је непразан текст.")
    return value


def sequence(value: Any, label: str, nonempty: bool = True) -> list:
    require(isinstance(value, list) and (bool(value) or not nonempty), f"{label}: очекивана је листа.")
    return value


def index_records(records: Any, label: str) -> dict[str, dict]:
    result = {}
    for record in sequence(records, label):
        ident = text(field(record, 'id', label), label + '.id')
        require(ident not in result, f"{label}: поновљени идентификатор '{ident}'.")
        result[ident] = record
    return result


def _unique_object(pairs: list) -> dict:
    result = {}
    for key, value in pairs:
        require(key not in result, f"Поновљено JSON поље '{key}'.")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise InputDataError(f"Недозвољена JSON бројчана вредност: {value}.")


def read_json(path: Path) -> tuple[dict, str]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode('utf-8-sig'), object_pairs_hook=_unique_object,
                           parse_constant=_invalid_constant)
    except json.JSONDecodeError as exc:
        raise InputDataError(f"{path.name}: неисправан JSON, ред {exc.lineno}, колона {exc.colno}: {exc.msg}") from exc
    except (OSError, UnicodeError, InputDataError) as exc:
        raise InputDataError(f"{path}: {exc}") from exc
    require(isinstance(value, dict), f"{path.name}: корен мора бити JSON објекат.")
    return value, hashlib.sha256(raw).hexdigest()


def _check_config(c: dict) -> None:
    for key in ('paths', 'run', 'experiment', 'simulation', 'fleet', 'randomness', 'demand',
                'service', 'dispatch', 'ground_handling', 'infrastructure_operations',
                'energy', 'charging', 'outputs', 'validation'):
        require(isinstance(field(c, key, 'config'), dict), f"config.{key}: очекиван је објекат.")
    require(field(c, 'schema_version', 'config') == '1.0', 'Неподржана верзија config.json.')
    require(field(c['paths'], 'relative_paths_base', 'paths') == 'config_file_directory',
            'Путање морају бити задате у односу на директоријум config.json.')
    require(field(c['run'], 'mode', 'run') in ('single_configuration', 'experiment_grid'),
            'run.mode: користити single_configuration или experiment_grid.')
    for key, lower in (('fleet_size', 1), ('replications', 1), ('base_seed', 0)):
        number(field(c['run'], key, 'run'), f'run.{key}', lower, integer=True)
    for key, expected in (('engine', 'discrete_event'), ('internal_time_unit', 'seconds')):
        require(field(c['simulation'], key, 'simulation') == expected, f'simulation.{key}: очекивано {expected}.')
    number(field(c['simulation'], 'time_comparison_tolerance_s', 'simulation'), 'time tolerance', 0)
    for key in ('initial_soc',):
        number(field(c['fleet'], key, 'fleet'), 'fleet.' + key, 0, 1)
    number(field(c['service'], 'max_wait_to_boarding_min', 'service'), 'max_wait_to_boarding_min', 0)
    number(field(c['service'], 'target_on_time_fraction', 'service'), 'target_on_time_fraction', 0, 1)
    number(field(c['outputs'], 'confidence_level', 'outputs'), 'confidence_level', 0, 1)
    require(0 < c['outputs']['confidence_level'] < 1, 'confidence_level мора бити између 0 и 1.')
    for key in ('landing_reserve_kwh', 'mission_contingency_fraction'):
        number(field(c['energy'], key, 'energy'), 'energy.' + key)
    # Строга провера улаза је обавезна у овој верзији учитавача.
    for key, value in c['validation'].items():
        if key != 'missing_input_policy':
            require(value is True, f'validation.{key}: ова верзија захтева укључену проверу.')
    require(field(c['validation'], 'missing_input_policy', 'validation') == 'raise_error_without_silent_defaults',
            'Није дозвољена тиха замена недостајућих параметара.')
    for section in ('boarding', 'disembarking'):
        model = field(c['ground_handling'], section, 'ground_handling')
        require(field(model, 'model', section) == 'linear_in_passenger_count', f'{section}: неподржан модел.')
        for key in ('fixed_min', 'per_passenger_min', 'duration_for_zero_passengers_min'):
            number(field(model, key, section), f'{section}.{key}')
        require(model['duration_for_zero_passengers_min'] == 0, f'{section}(0) мора бити нула.')
    number(field(c['ground_handling'], 'departure_finalization_min', 'ground_handling'), 'departure_finalization_min')
    clean = field(c['ground_handling'], 'cleaning', 'ground_handling')
    for key in ('every_completed_legs', 'crews_per_vertiport'):
        number(field(clean, key, 'cleaning'), 'cleaning.' + key, 1, integer=True)
    number(field(clean, 'duration_min', 'cleaning'), 'cleaning.duration_min')
    d = c['demand']
    triangular = [field(d, 'terminal_access_min', 'demand'),
                  field(field(d, 'airport_to_city', 'demand'), 'terminal_processing_min', 'airport_to_city')]
    for model in triangular:
        require(field(model, 'distribution', 'triangular') == 'triangular', 'Очекивана троугаона расподела.')
        for key in ('minimum', 'mode', 'maximum'):
            number(field(model, key, 'triangular'), 'triangular.' + key)
        require(model['minimum'] <= model['mode'] <= model['maximum'] and model['minimum'] < model['maximum'],
                'Неисправне границе троугаоне расподеле.')
    departure = field(d, 'city_to_airport', 'demand')
    normal = field(departure, 'terminal_earliness_min', 'city_to_airport')
    require(field(normal, 'distribution', 'terminal_earliness_min') == 'truncated_normal', 'Очекивана одсечена нормална расподела.')
    for name in ('short', 'medium', 'long'):
        profile = field(field(normal, 'profiles', 'terminal_earliness_min'), name, 'profiles')
        for key in ('minimum', 'mean', 'maximum', 'stddev'):
            number(field(profile, key, name), f'{name}.{key}')
        require(profile['minimum'] <= profile['mean'] <= profile['maximum'] and
                profile['minimum'] < profile['maximum'] and profile['stddev'] > 0,
                f'Неисправни параметри одсечене нормалне расподеле: {name}.')
    haul = field(departure, 'haul_classification', 'city_to_airport')
    a = number(field(haul, 'short_upper_inclusive_min', 'haul'), 'short threshold', 1)
    b = number(field(haul, 'medium_upper_inclusive_min', 'haul'), 'medium threshold', 1)
    require(a < b, 'Граница short мора бити мања од границе medium.')
    for key in ('stream_ids', 'scenario_ids'):
        values = field(c['randomness'], key, 'randomness')
        require(isinstance(values, dict) and bool(values), f'randomness.{key}: очекиван непразан објекат.')
        for name, value in values.items():
            number(value, f'{key}.{name}', 0, integer=True)
        require(len(set(values.values())) == len(values), f'randomness.{key}: идентификатори морају бити јединствени.')


def _check_aircraft(a: dict, c: dict) -> tuple[float, float]:
    require(field(a, 'schema_version', 'aircraft') == '1.1', 'Очекивана је eVTOL_spec.json верзија 1.1 без криве пуњења.')
    require(field(a, 'aircraft_id', 'aircraft') == field(c['fleet'], 'aircraft_id', 'fleet'), 'Неусклађен aircraft_id.')
    cap = field(a, 'capacity', 'aircraft')
    number(field(cap, 'passenger_seats', 'capacity'), 'passenger_seats', 1, integer=True)
    number(field(cap, 'minimum_flight_crew', 'capacity'), 'minimum_flight_crew', 1, integer=True)
    battery = field(a, 'battery', 'aircraft')
    require('charging' not in battery, 'Крива пуњења се задаје само у config.json; уклонити battery.charging из спецификације.')
    capacity = number(field(battery, 'modeled_usable_capacity_kwh', 'battery'), 'battery capacity', 0.000001)
    require(field(battery, 'soc_min', 'battery') == 0 and field(battery, 'soc_max', 'battery') == 1,
            'Овај модел користи SOC у опсегу 0–1.')
    performance = field(a, 'performance', 'aircraft')
    speed = number(field(performance, 'operational_cruise_speed_kmh', 'performance'), 'cruise speed', 0.000001)
    maximum = number(field(performance, 'manufacturer_target_max_speed_kmh', 'performance'), 'maximum speed', speed)
    require(speed <= maximum, 'Оперативна брзина премашује максималну.')
    require(c['energy']['landing_reserve_kwh'] < capacity, 'Резерва мора бити мања од капацитета батерије.')
    model = field(a, 'flight_energy_model', 'aircraft')
    power = field(model, 'power_kw', 'flight_energy_model')
    for key in ('hover', 'transition', 'cruise'):
        number(field(power, key, 'power_kw'), 'power_kw.' + key, 0.000001)
    phase_energy = 0.0
    for phase in ('takeoff', 'landing'):
        duration = 0.0
        for segment in sequence(field(model, phase + '_segments', 'flight_energy_model'), phase):
            seconds = number(field(segment, 'duration_s', phase), phase + '.duration_s', 0.000001)
            start = field(segment, 'power_start_key', phase)
            end = field(segment, 'power_end_key', phase)
            require(isinstance(start, str) and isinstance(end, str) and start in power and end in power,
                    f'{phase}: непознат кључ снаге.')
            phase_energy += (power[start] + power[end]) * 0.5 * seconds / 3600
            duration += seconds
        narrow = field(field(c['infrastructure_operations'], 'narrow_occupancy_for_reporting_s', 'infrastructure'), phase, 'occupancy')
        number(narrow, 'narrow_occupancy.' + phase, 0, duration)
    charge = c['charging']
    curve = sequence(field(charge, 'curve', 'charging'), 'charging.curve')
    require(len(curve) >= 2, 'Крива пуњења мора имати најмање две тачке.')
    previous = None
    for point in curve:
        soc = number(field(point, 'soc', 'curve'), 'curve.soc', 0, 1)
        kw = number(field(point, 'power_kw', 'curve'), 'curve.power_kw', 0.000001)
        if previous is not None:
            require(soc > previous[0] and kw <= previous[1], 'SOC мора расти, а снага пуњења не сме расти дуж криве.')
        previous = (soc, kw)
    require(curve[0]['soc'] == 0 and curve[-1]['soc'] == 1, 'Крива пуњења мора покривати SOC 0–1.')
    number(field(charge, 'charger_output_to_stored_energy_efficiency', 'charging'), 'charging efficiency', 0.000001, 1)
    number(field(charge, 'idle_target_soc', 'charging'), 'idle_target_soc', 0, 1)
    number(field(charge, 'start_delay_after_on_stand_min', 'charging'), 'charging start delay')
    require(field(charge, 'apply_efficiency_again_to_curve', 'charging') is False,
            'Ефикасност се не сме поново применити на ефективну криву пуњења.')
    return phase_energy, power['cruise'] / speed


def _coordinate(record: dict, label: str) -> list[float]:
    lon = number(field(record, 'longitude_deg', label), label + '.longitude', -180, 180)
    lat = number(field(record, 'latitude_deg', label), label + '.latitude', -90, 90)
    elevation = field(record, 'elevation_m_amsl', label)
    if elevation is not None:
        number(elevation, label + '.elevation', -500, 10000)
    return [lon, lat]


def _distance(a: list, b: list, radius: float) -> float:
    lon1, lat1, lon2, lat2 = map(math.radians, (*a, *b))
    h = math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
    return 2 * radius * math.asin(math.sqrt(min(1.0, max(0.0, h))))


def _check_network(n: dict, c: dict, a: dict, phase_energy: float, cruise_kwh_km: float) -> dict:
    require(field(n, 'schema_version', 'network') == '1.1', 'Неподржана верзија vertiport_network.json.')
    require(field(n, 'network_id', 'network') == field(c, 'project_id', 'config'), 'Неусклађени project_id и network_id.')
    ports = index_records(field(n, 'vertiports', 'network'), 'vertiports')
    transit = index_records(field(n, 'transit_nodes', 'network'), 'transit_nodes')
    require(not (ports.keys() & transit.keys()), 'Вертипорти и транзитни чворови морају имати различите идентификаторе.')
    coords = {key: _coordinate(v, key) for key, v in (ports | transit).items()}
    airports = [key for key, v in ports.items() if field(v, 'role', key) == 'airport']
    require(len(airports) == 1, 'Мрежа мора имати тачно један аеродромски вертипорт.')
    for key, port in ports.items():
        require(port['role'] in ('airport', 'city'), f'{key}: непозната улога вертипорта.')
        stands = number(field(port, 'parking_stands', key), key + '.parking_stands', 1, integer=True)
        charger = field(port, 'charging', key)
        count = number(field(charger, 'charger_count', key), key + '.charger_count', 1, stands, integer=True)
        number(field(charger, 'power_per_charger_kw', key), key + '.charger_power', 0.000001)
        require(field(charger, 'chargers_per_stand', key) == 1 and count == stands,
                f'{key}: очекиван је по један пуњач на свакој позицији.')
        number(field(charger, 'max_simultaneous_charging_aircraft', key), key + '.simultaneous_charging', 1, count, integer=True)
        movement = field(port, 'movement_area', key)
        for name in ('tlof_count', 'fato_count', 'max_simultaneous_movements'):
            require(field(movement, name, key) == 1, f'{key}.{name}: основни модел захтева један заједнички ресурс.')
        require(field(movement, 'shared_resource', key) is True, f'{key}: TLOF/FATO мора бити заједнички ресурс.')
    shares = field(field(c['demand'], 'city_assignment', 'demand'), 'shares', 'city_assignment')
    require(isinstance(shares, dict) and set(shares) == set(ports) - set(airports), 'Удели тражње морају покривати тачно градске вертипорте.')
    for key, value in shares.items():
        number(value, 'city share.' + key, 0, 1)
    require(math.isclose(sum(shares.values()), 1, abs_tol=1e-9), 'Збир удела тражње по градским вертипортовима мора бити 1.')
    ranks = field(c['dispatch'], 'origin_priority_rank', 'dispatch')
    require(isinstance(ranks, dict) and set(ranks) == set(ports), 'Приоритети морају покривати све вертипорте.')
    for key, value in ranks.items():
        number(value, 'priority.' + key, 1, integer=True)
    allocations = field(n, 'initial_fleet_allocation', 'network')
    require(isinstance(allocations, dict) and bool(allocations), 'Недостаје почетни распоред флоте.')
    for size, values in allocations.items():
        require(size.isdigit() and int(size) > 0 and str(int(size)) == size, 'Неисправна величина флоте у почетном распореду.')
        require(isinstance(values, dict) and set(values) <= set(ports), f'Флота {size}: непознати вертипорт.')
        for key, value in values.items():
            number(value, f'allocation[{size}].{key}', 0, ports[key]['parking_stands'], integer=True)
        require(sum(values.values()) == int(size), f'Флота {size}: збир почетног распореда није једнак величини флоте.')
    dm = field(n, 'distance_model', 'network')
    radius = number(field(dm, 'earth_mean_radius_km', 'distance_model'), 'earth radius', 1)
    deduction = sum(number(field(dm, key, 'distance_model'), key) for key in ('departure_segment_km', 'arrival_segment_km'))
    edges = index_records(field(n, 'edges', 'network'), 'edges')
    for key, edge in edges.items():
        start, end = field(edge, 'from_node', key), field(edge, 'to_node', key)
        require(isinstance(start, str) and isinstance(end, str) and start in coords and end in coords, f'{key}: непознати крајеви деонице.')
        require(type(field(edge, 'bidirectional', key)) is bool, f'{key}.bidirectional: очекивана bool вредност.')
        geometry = sequence(field(edge, 'geometry_lon_lat', key), key + '.geometry')
        require(len(geometry) >= 2, f'{key}: недовољно геометријских тачака.')
        for point in geometry:
            require(isinstance(point, list) and len(point) == 2, f'{key}: координата мора бити [lon, lat].')
            number(point[0], key + '.lon', -180, 180)
            number(point[1], key + '.lat', -90, 90)
        require(geometry[0] == coords[start] and geometry[-1] == coords[end], f'{key}: крајеви геометрије не одговарају чворовима.')
        length = number(field(edge, 'length_km', key), key + '.length_km')
        computed = sum(_distance(p, q, radius) for p, q in zip(geometry, geometry[1:]))
        require(math.isclose(length, computed, rel_tol=0, abs_tol=1e-5), f'{key}: length_km не одговара геометрији; поново израчунати дужину.')
    pairs = set()
    for key, route in index_records(field(n, 'routes', 'network'), 'routes').items():
        start, end = field(route, 'origin', key), field(route, 'destination', key)
        require(isinstance(start, str) and isinstance(end, str) and start in ports and end in ports and start != end, f'{key}: неисправна рута.')
        pair = frozenset((start, end))
        require(pair not in pairs, f'{key}: поновљена двосмерна рута.')
        pairs.add(pair)
        require(field(route, 'bidirectional', key) is True, f'{key}: тренутна мрежа користи двосмерне руте.')
        nodes = sequence(field(route, 'node_path', key), key + '.node_path')
        path = sequence(field(route, 'edge_path', key), key + '.edge_path')
        require(all(isinstance(x, str) for x in nodes + path), f'{key}: путања мора садржати текстуалне идентификаторе.')
        require(len(nodes) == len(path)+1 and nodes[0] == start and nodes[-1] == end, f'{key}: неусклађена путања.')
        require(len(set(nodes)) == len(nodes) and not (set(nodes[1:-1]) & set(ports)), f'{key}: петља или међуслетање у путањи.')
        total = 0.0
        for u, v, eid in zip(nodes, nodes[1:], path):
            require(eid in edges, f'{key}: непозната деоница {eid}.')
            edge = edges[eid]
            forward = edge['from_node'] == u and edge['to_node'] == v
            reverse = edge['bidirectional'] and edge['from_node'] == v and edge['to_node'] == u
            require(forward or reverse, f'{key}: деоница {eid} не повезује {u} и {v}.')
            total += edge['length_km']
        stored_total = number(field(route, 'total_distance_km', key), key + '.total_distance_km')
        cruise = number(field(route, 'cruise_distance_km', key), key + '.cruise_distance_km')
        require(math.isclose(total, stored_total, rel_tol=0, abs_tol=1e-5), f'{key}: укупна дужина не одговара збиру деоница.')
        require(math.isclose(cruise, total-deduction, rel_tol=0, abs_tol=1e-5), f'{key}: неправилно издвојене крајње фазе из крстарења.')
        operations = sequence(field(route, 'allowed_operations', key), key + '.operations')
        expected = {'passenger', 'repositioning'} if airports[0] in (start, end) else {'repositioning'}
        require(all(isinstance(x, str) for x in operations) and set(operations) == expected, f'{key}: неусклађене дозвољене операције.')
        required = (phase_energy + cruise*cruise_kwh_km)*(1+c['energy']['mission_contingency_fraction']) + c['energy']['landing_reserve_kwh']
        require(required <= a['battery']['modeled_usable_capacity_kwh'], f'{key}: лет са резервом захтева {required:.2f} kWh, више од капацитета батерије.')
    require(len(pairs) == len(ports)*(len(ports)-1)//2, 'Недостају руте између неких вертипортова.')
    return allocations


def _prepare_flights(s: dict, c: dict) -> tuple[list[dict], dict]:
    require(field(s, 'schema_version', 'schedule') == '1.2', 'Неподржана верзија реда летења.')
    require(field(s, 'dataset_status', 'schedule') == c['demand']['schedule_type'], 'Неусклађен тип скупа података о реду летења.')
    zone_name = text(field(s, 'timezone', 'schedule'), 'schedule.timezone')
    require(zone_name == field(c['simulation'], 'timezone', 'simulation'), 'Временске зоне реда летења и конфигурације се разликују.')
    try:
        zone = ZoneInfo(zone_name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise InputDataError('Временска зона није доступна. Проверите назив и инсталирајте: python -m pip install tzdata') from exc
    try:
        flight_date = date.fromisoformat(text(field(s, 'date', 'schedule'), 'schedule.date'))
    except ValueError as exc:
        raise InputDataError('schedule.date: очекиван датум YYYY-MM-DD.') from exc
    airport = text(field(s, 'airport', 'schedule'), 'schedule.airport')
    cal = field(c['demand'], 'calibration', 'demand')
    regional_airports = sequence(field(cal, 'regional_airports', 'calibration'), 'regional_airports')
    require(all(isinstance(x, str) and re.fullmatch('[A-Z]{3}', x) for x in regional_airports), 'Неисправан IATA код у регионалној групи.')
    haul = c['demand']['city_to_airport']['haul_classification']
    records_key = text(field(c['demand'], 'schedule_records_key', 'demand'), 'schedule_records_key')
    rows = sequence(field(s, records_key, 'schedule'), 'schedule.flights')
    result, seen = [], set()
    for row in rows:
        ident = text(field(row, 'flight_id', 'flight'), 'flight_id')
        require(ident not in seen, f'Поновљен flight_id: {ident}.')
        seen.add(ident)
        direction = field(row, 'direction', ident)
        require(direction in ('arrival', 'departure'), f'{ident}: direction мора бити arrival или departure.')
        origin = text(field(row, 'origin', ident), ident + '.origin').rsplit(' ', 1)[-1]
        destination = text(field(row, 'destination', ident), ident + '.destination').rsplit(' ', 1)[-1]
        require(bool(re.fullmatch('[A-Z]{3}', origin) and re.fullmatch('[A-Z]{3}', destination)), f'{ident}: неисправан IATA код.')
        require((destination if direction == 'arrival' else origin) == airport, f'{ident}: смер није усаглашен са аеродромом {airport}.')
        other_airport = origin if direction == 'arrival' else destination
        require(other_airport != airport, f'{ident}: полазиште и одредиште авиона не могу бити исти аеродром.')
        capacity = number(field(row, 'seat_capacity', ident), ident + '.seat_capacity', 1, integer=True)
        block = number(field(row, 'block_time_min', ident), ident + '.block_time_min', 0.000001)
        clock = text(field(row, 'scheduled_local_time', ident), ident + '.scheduled_local_time')
        require(bool(re.fullmatch(r'(?:[01]\d|2[0-3]):[0-5]\d', clock)), f'{ident}: време мора бити HH:MM, од 00:00 до 23:59.')
        scheduled = datetime.combine(flight_date, datetime.strptime(clock, '%H:%M').time(), tzinfo=zone)
        category = 'regional' if other_airport in regional_airports else 'other'
        category_haul = 'short' if block <= haul['short_upper_inclusive_min'] else ('medium' if block <= haul['medium_upper_inclusive_min'] else 'long')
        result.append(dict(row, seat_capacity=capacity, category=category, haul_category=category_haul,
                           scheduled_datetime=scheduled, other_airport=other_airport))
    result.sort(key=lambda x: x['flight_id'])
    total_seats = sum(f['seat_capacity'] for f in result)
    regional_seats = sum(f['seat_capacity'] for f in result if f['category'] == 'regional')
    target = number(field(cal, 'target_passenger_movements', 'calibration'), 'target passenger movements', 0, total_seats)
    target_transfer = number(field(cal, 'transfer_share_target', 'calibration'), 'transfer target', 0, 1)
    regional_share = number(field(cal, 'regional_transfer_share', 'calibration'), 'regional transfer share', 0, 1)
    require(total_seats > regional_seats, 'За калибрацију other_transfer_share потребан је бар један лет ван регионалне групе.')
    lf = target/total_seats
    other_share = (target_transfer*total_seats-regional_share*regional_seats)/(total_seats-regional_seats)
    number(other_share, 'Израчунати other_transfer_share', 0, 1)
    adoption = field(c['demand'], 'adoption_share', 'demand')
    require(isinstance(adoption, dict) and bool(adoption), 'Недостају сценарији прихватања eVTOL услуге.')
    for scenario, probability in adoption.items():
        number(probability, 'adoption_share.' + scenario, 0, 1)
    expected_by_direction = dict(arrival=0.0, departure=0.0)
    for flight in result:
        transfer = regional_share if flight['category'] == 'regional' else other_share
        flight['transfer_share'] = transfer
        flight['expected_non_transfer_passengers'] = flight['seat_capacity']*lf*(1-transfer)
        expected_by_direction[flight['direction']] += flight['expected_non_transfer_passengers']
    derived = {'total_seats': total_seats, 'load_factor': lf, 'other_transfer_share': other_share,
               'expected_non_transfer_by_direction': expected_by_direction,
               'expected_evtol_by_scenario': {k: sum(expected_by_direction.values())*v for k, v in adoption.items()}}
    return result, derived


def _configurations(c: dict, allocations: dict) -> list[dict]:
    sizes = sequence(field(c['experiment'], 'fleet_sizes', 'experiment'), 'experiment.fleet_sizes')
    scenarios = sequence(field(c['experiment'], 'demand_scenarios', 'experiment'), 'experiment.demand_scenarios')
    for size in sizes + [c['run']['fleet_size']]:
        number(size, 'fleet size', 1, integer=True)
        require(str(size) in allocations, f'Нема почетног распореда за флоту величине {size}.')
    for scenario in scenarios + [field(c['run'], 'demand_scenario', 'run')]:
        text(scenario, 'demand scenario')
        require(scenario in c['demand']['adoption_share'] and scenario in c['randomness']['scenario_ids'],
                f'Непознат сценарио тражње: {scenario}.')
    require(len(set(sizes)) == len(sizes) and len(set(scenarios)) == len(scenarios), 'Поновљена конфигурација у мрежи експеримената.')
    if c['run']['mode'] == 'single_configuration':
        combinations = [(c['run']['fleet_size'], c['run']['demand_scenario'])]
    else:
        combinations = itertools.product(sizes, scenarios)
    return [{'fleet_size': size, 'demand_scenario': scenario, 'replications': c['run']['replications'],
             'initial_allocation': dict(allocations[str(size)])} for size, scenario in combinations]


def load_inputs(config_path: str | Path, *, overrides: dict | None = None) -> InputData:
    """Учитај фајлове и врати проверене податке. Не мења улазне фајлове.

    Путање из config.json разрешавају се у односу на његов директоријум.
    overrides мења само run поља у меморији (CLI параметри имају предност).
    """
    config_path = Path(config_path).expanduser().resolve()
    c, config_hash = read_json(config_path)
    if overrides:
        run = field(c, 'run', 'config')
        require(isinstance(run, dict), 'config.run: очекиван објекат.')
        require(set(overrides) <= {'mode', 'fleet_size', 'demand_scenario', 'replications', 'base_seed'}, 'Непознато CLI подешавање.')
        run.update(overrides)
    _check_config(c)
    paths, hashes, docs = {'config': config_path}, {'config': config_hash}, {}
    for key in ('aircraft_spec', 'vertiport_network', 'flight_schedule'):
        path = (config_path.parent / text(field(c['paths'], key, 'paths'), 'paths.' + key)).resolve()
        docs[key], hashes[key] = read_json(path)
        paths[key] = path
    a, n, s = (docs[k] for k in ('aircraft_spec', 'vertiport_network', 'flight_schedule'))
    phase_energy, cruise_kwh_km = _check_aircraft(a, c)
    allocations = _check_network(n, c, a, phase_energy, cruise_kwh_km)
    flights, calibration = _prepare_flights(s, c)
    configurations = _configurations(c, allocations)
    output_directory = (config_path.parent / text(field(c['paths'], 'output_directory', 'paths'), 'output_directory')).resolve()
    return InputData(c, a, n, s, flights, configurations, calibration, paths, hashes, output_directory)
