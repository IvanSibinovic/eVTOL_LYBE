from __future__ import annotations

import argparse
import math
import sys
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

from .entities import Aircraft, AircraftSpec, FlightEstimate, FlightOperation, FlightPhase

if TYPE_CHECKING:
    from .input_data import InputData


def _number(value: float, label: str, *, positive: bool = False) -> float:
    if (type(value) not in (int, float) or not math.isfinite(value)
            or value < 0 or (positive and value == 0)):
        raise ValueError(f'{label}: очекиван коначан {"позитиван" if positive else "ненегативан"} број.')
    return float(value)


class FlightModel:
    """Користити FlightModel.from_inputs(data) после load_inputs().

    Потрошња не зависи од броја путника, висине терена или смера лета.
    Позициони и путнички лет имају исти модел снаге. Провера енергије
    не проверава доступност летелице, TLOF/FATO или паркинг позиција.
    """

    def __init__(self, aircraft: dict, network: dict, config: dict) -> None:
        model = aircraft['flight_energy_model']
        energy = config['energy']
        if (model['model'] != 'phase_based'
                or model['power_basis'] != 'battery_energy_draw'
                or model['segment_power_interpolation'] != 'linear_in_time'
                or model['cruise_duration_basis'] != 'cruise_distance_divided_by_operational_cruise_speed'):
            raise ValueError('Неподржан модел потрошње или интерполације снаге.')
        if (energy['payload_dependence_enabled']
                or energy['ground_energy_consumption_enabled']
                or energy['contingency_is_consumed_automatically']
                or not energy['repositioning_uses_same_flight_power_model']
                or energy['actual_flight_energy_rule'] != 'phase_based_planned_flight_energy'
                or energy['departure_requirement_rule'] !=
                    'planned_flight_energy * (1 + mission_contingency_fraction) + landing_reserve_kwh'):
            raise ValueError('Ова верзија подржава номиналну потрошњу лета без аутоматског трошења резерве.')
        self.aircraft_spec = AircraftSpec(
            aircraft['aircraft_id'], aircraft['capacity']['passenger_seats'],
            aircraft['battery']['modeled_usable_capacity_kwh'])
        self.cruise_speed_kmh = _number(
            aircraft['performance']['operational_cruise_speed_kmh'], 'cruise_speed', positive=True)
        self.cruise_power_kw = _number(model['power_kw']['cruise'], 'cruise_power')
        self.reserve_kwh = _number(energy['landing_reserve_kwh'], 'landing_reserve_kwh')
        self.contingency_fraction = _number(energy['mission_contingency_fraction'], 'contingency')
        self.takeoff = self._phase('takeoff', model['takeoff_segments'], model['power_kw'])
        self.landing = self._phase('landing', model['landing_segments'], model['power_kw'])
        self._routes: dict[tuple[str, str], dict] = {}
        self._forward_pairs: list[tuple[str, str]] = []
        distance = network['distance_model']
        deduction = (_number(distance['departure_segment_km'], 'departure_segment_km')
                     + _number(distance['arrival_segment_km'], 'arrival_segment_km'))
        for original in network['routes']:
            route = deepcopy(original)
            origin, destination = route['origin'], route['destination']
            total = _number(route['total_distance_km'], 'total_distance_km')
            cruise = _number(route['cruise_distance_km'], 'cruise_distance_km')
            if total < deduction or not math.isclose(total - deduction, cruise, rel_tol=0, abs_tol=1e-5):
                raise ValueError(f"{route['id']}: неусаглашена дужина крстарења.")
            if route.get('intermediate_landings'):
                raise ValueError('Лет са међуслетањем мора се поделити на засебне етапе.')
            self._add_route(origin, destination, route)
            self._forward_pairs.append((origin, destination))
            if route['bidirectional']:
                reverse = deepcopy(route)
                reverse.update(origin=destination, destination=origin,
                               node_path=list(reversed(route['node_path'])),
                               edge_path=list(reversed(route['edge_path'])))
                self._add_route(destination, origin, reverse)

    @classmethod
    def from_inputs(cls, data: InputData) -> FlightModel:
        return cls(data.aircraft, data.network, data.config)

    def _add_route(self, origin: str, destination: str, route: dict) -> None:
        if origin == destination or (origin, destination) in self._routes:
            raise ValueError(f'Неисправна или поновљена рута {origin}–{destination}.')
        self._routes[origin, destination] = route

    @staticmethod
    def _phase(name: str, segments: list[dict], power: dict) -> FlightPhase:
        if not segments:
            raise ValueError(f'{name}: недостају сегменти фазе.')
        duration = energy = 0.0
        for segment in segments:
            seconds = _number(segment['duration_s'], 'segment.duration_s')
            p0 = _number(power[segment['power_start_key']], 'power_start')
            p1 = _number(power[segment['power_end_key']], 'power_end')
            duration += seconds
            energy += (p0 + p1) / 2 * seconds / 3600
        return FlightPhase(name, duration, energy)

    def estimate(self, origin: str, destination: str,
                 operation: FlightOperation | str = FlightOperation.PASSENGER) -> FlightEstimate:
        operation = FlightOperation(operation)
        try:
            route = self._routes[origin, destination]
        except KeyError as exc:
            raise ValueError(f'Нема руте {origin}–{destination}.') from exc
        if operation.value not in route['allowed_operations']:
            raise ValueError(f'{origin}–{destination}: није дозвољена операција {operation.value}.')
        # У JSON-у је већ одузето по 0,5 km на оба краја. Не одузимати поново.
        cruise_distance = route['cruise_distance_km']
        cruise_s = cruise_distance / self.cruise_speed_kmh * 3600
        cruise = FlightPhase('cruise', cruise_s, self.cruise_power_kw * cruise_s / 3600)
        nominal = self.takeoff.energy_kwh + cruise.energy_kwh + self.landing.energy_kwh
        return FlightEstimate(
            route['id'], origin, destination, operation,
            route['total_distance_km'], cruise_distance,
            tuple(route['node_path']), tuple(route['edge_path']),
            self.takeoff, cruise, self.landing,
            nominal * (1 + self.contingency_fraction) + self.reserve_kwh)

    def all_routes(self) -> tuple[FlightEstimate, ...]:
        """Један смер сваке руте: путничка где је дозвољено, иначе позициона."""
        estimates = []
        for origin, destination in self._forward_pairs:
            operations = self._routes[origin, destination]['allowed_operations']
            operation = ('passenger' if 'passenger' in operations else 'repositioning')
            estimates.append(self.estimate(origin, destination, operation))
        return tuple(estimates)

    def can_depart(self, estimate: FlightEstimate, available_energy_kwh: float) -> bool:
        """Само енергетска изводљивост, за процену направљену овим моделом."""
        available = self._check_energy(available_energy_kwh)
        return available >= estimate.required_departure_energy_kwh

    def energy_after_flight(self, estimate: FlightEstimate, departure_energy_kwh: float) -> float:
        """Чист прорачун без измене летелице; захтева испуњен услов полетања."""
        if not self.can_depart(estimate, departure_energy_kwh):
            raise ValueError('Недовољна енергија за полетање са усвојеном резервом.')
        return departure_energy_kwh - estimate.energy_kwh

    def _check_energy(self, energy_kwh: float) -> float:
        energy = _number(energy_kwh, 'available_energy_kwh')
        if energy > self.aircraft_spec.battery_capacity_kwh:
            raise ValueError('Енергија премашује капацитет батерије.')
        return energy


def create_initial_fleet(data: InputData, configuration: dict,
                         start_time_s: float = 0.0) -> list[Aircraft]:
    """Свако позивање прави нову флоту за једну независну репликацију.

    configuration је један члан data.configurations; почетак се прослеђује
    након генерисања путника. Не користи тврдо задате величине или локације.
    """
    if type(start_time_s) not in (int, float) or not math.isfinite(start_time_s):
        raise ValueError('Почетно време мора бити коначан број.')
    allocation = configuration['initial_allocation']
    fleet_size = configuration['fleet_size']
    if (type(fleet_size) is not int or fleet_size < 1
            or any(type(n) is not int or n < 0 for n in allocation.values())
            or sum(allocation.values()) != fleet_size):
        raise ValueError('Неисправан почетни распоред флоте.')
    known = {v['id'] for v in data.network['vertiports']}
    if not set(allocation) <= known:
        raise ValueError('Непознат вертипорт у почетном распореду.')
    stands = {v['id']: v['parking_stands'] for v in data.network['vertiports']}
    if any(count > stands[location] for location, count in allocation.items()):
        raise ValueError('Почетни распоред премашује број паркинг позиција.')
    spec = AircraftSpec(data.aircraft['aircraft_id'], data.aircraft['capacity']['passenger_seats'],
                        data.aircraft['battery']['modeled_usable_capacity_kwh'])
    soc = _number(data.config['fleet']['initial_soc'], 'initial_soc')
    if soc > 1:
        raise ValueError('Почетни SOC премашује 1.')
    legs = data.config['fleet']['initial_completed_legs_since_cleaning']
    fleet = []
    for location, count in allocation.items():
        for _ in range(count):
            fleet.append(Aircraft(
                id=f'EVTOL_{len(fleet) + 1:03d}', spec=spec, location=location,
                energy_kwh=soc * spec.battery_capacity_kwh,
                completed_legs_since_cleaning=legs, total_completed_legs=legs,
                energy_updated_at_s=start_time_s))
    return fleet


def main(argv: list[str] | None = None) -> int:
    from .input_data import InputDataError, load_inputs

    parser = argparse.ArgumentParser(description='Провера времена и енергије свих рута; без симулације.')
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'config.json')
    args = parser.parse_args(argv)
    try:
        model = FlightModel.from_inputs(load_inputs(args.config))
        print('Рута            Тип             km крст.   Лет [min]  E [kWh]  Услов [kWh]')
        for estimate in model.all_routes():
            print(f'{estimate.route_id:15} {estimate.operation.value:15} '
                  f'{estimate.cruise_distance_km:8.3f} {estimate.duration_s / 60:10.3f} '
                  f'{estimate.energy_kwh:8.3f} {estimate.required_departure_energy_kwh:12.3f}')
        print(f'Услов = {1 + model.contingency_fraction:g} × E + {model.reserve_kwh:g} kWh (из config.json).')
        print('Приказано време обухвата полетање, крстарење и слетање; без земаљског опслуживања.')
    except (InputDataError, ValueError, KeyError) as exc:
        print(f'ГРЕШКА: {exc}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    raise SystemExit(main())
