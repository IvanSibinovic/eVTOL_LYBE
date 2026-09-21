from __future__ import annotations

import argparse
import math
import sys
from copy import copy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from .entities import FlightEstimate

if TYPE_CHECKING:
    from .input_data import InputData


class ResourceConflict(ValueError):
    """Резервација није изводљива; претходно стање је сачувано."""

    def __init__(self, message: str, code: str = 'resource_conflict', **details):
        super().__init__(message)
        self.code, self.details = code, details


def _time(value: float, *, allow_infinity: bool = False) -> float:
    if type(value) not in (int, float) or not (math.isfinite(value) or (allow_infinity and value == math.inf)):
        raise ValueError('Време мора бити коначан број; само крај интервала може бити +inf.')
    return float(value)


def _shift_to_event(base: float, event: float, offset: float) -> float:
    """Помери базно време тако да base+offset не буде пре event због заокруживања."""
    candidate = max(base, event - offset)
    if candidate + offset < event:
        candidate = math.nextafter(candidate, math.inf)
    return candidate


@dataclass(frozen=True, slots=True)
class Reservation:
    id: str
    resource_id: str
    owner: str  # aircraft_id
    start_s: float
    end_s: float
    purpose: str
    stand_visit_id: str | None = None

    def __post_init__(self) -> None:
        _time(self.start_s)
        _time(self.end_s, allow_infinity=True)
        if self.end_s < self.start_s or not all((self.id, self.resource_id, self.owner)):
            raise ValueError('Неисправна резервација.')

    def active_at(self, time_s: float) -> bool:
        return self.start_s <= time_s < self.end_s


class ResourceCalendar:
    """Један физички ресурс капацитета 1; остали капацитети су више јединица."""

    def __init__(self, resource_id: str, reservations: tuple[Reservation, ...] = ()) -> None:
        self.resource_id = resource_id
        self.reservations = tuple(sorted(reservations, key=lambda r: (r.start_s, r.end_s, r.id)))
        if len({r.id for r in reservations}) != len(reservations):
            raise ValueError('Поновљен идентификатор резервације.')
        end = -math.inf
        for booking in self.reservations:
            if booking.resource_id != resource_id:
                raise ValueError('Резервација припада другом ресурсу.')
            if booking.start_s == booking.end_s:
                continue
            if booking.start_s < end:
                raise ResourceConflict(f'Преклапање резервација на {resource_id}.')
            end = booking.end_s

    def earliest_start(self, after_s: float, duration_s: float) -> float | None:
        candidate = _time(after_s)
        duration = _time(duration_s, allow_infinity=True)
        if duration <= 0:
            raise ValueError('Трајање мора бити позитивно.')
        for booking in self.reservations:
            if booking.start_s == booking.end_s or booking.end_s <= candidate:
                continue
            if candidate + duration <= booking.start_s:
                return candidate
            if booking.end_s == math.inf:
                return None
            candidate = booking.end_s
        return candidate


@dataclass(frozen=True, slots=True)
class FlightSlot:
    origin: str
    destination: str
    takeoff_s: float
    takeoff_end_s: float
    landing_start_s: float
    landing_end_s: float
    destination_stand_id: str


@dataclass(frozen=True, slots=True)
class FlightReservation:
    flight_id: str
    aircraft_id: str
    slot: FlightSlot
    origin_visit_id: str
    origin_previous_end_s: float
    takeoff_reservation_id: str
    landing_reservation_id: str
    destination_visit_id: str


class NetworkResources:
    """Једна независна инстанца по репликацији; све измене иду кроз овај објекат.

    Подржано је по једно заједничко TLOF/FATO и по један пуњач на свакој
    позицији, према усвојеној мрежи. Непознат тренутак ослобађања позиције
    није претпостављен: таква позиција остаје недоступна другим доласцима.
    Једна летелица може имати једну резервисану, незавршену летну етапу.
    """

    def __init__(self, data: InputData) -> None:
        c = data.config
        rules = c['infrastructure_operations']
        for key, expected in {
            'movement_resource': 'shared_TLOF_FATO',
            'takeoff_blocking_window': 'entire_takeoff_phase',
            'landing_blocking_window': 'entire_landing_phase',
            'origin_stand_release_event': 'takeoff_phase_start',
            'destination_stand_occupancy_start': 'landing_phase_end',
            'resource_intervals': 'half_open', 'airborne_holding_enabled': False,
            'add_blocking_durations_to_flight_time': False,
            'destination_stand_reservation_required': True,
            'destination_movement_slot_reservation_required': True,
            'departure_allowed_only_if_arrival_resources_feasible': True,
        }.items():
            if rules[key] != expected:
                raise ValueError(f'infrastructure_operations.{key}: неподржана поставка.')
        self.start_charge_delay_s = 60 * c['charging']['start_delay_after_on_stand_min']
        self.stop_charge_lead_s = 60 * c['ground_handling']['departure_finalization_min']
        self.charging_enabled = c['charging']['enabled']
        self.charging_can_overlap_cleaning = c['charging']['can_overlap_cleaning']
        self._units: dict[str, tuple[str, str]] = {}  # resource_id -> (vertiport, kind)
        self._stands: dict[str, tuple[str, ...]] = {}
        self._chargers: dict[str, str] = {}
        self._bookings: dict[str, Reservation] = {}
        self._calendar_cache: dict[str, ResourceCalendar] = {}
        self._physical_stands: dict[str, str] = {}
        self._flights: dict[str, FlightReservation] = {}
        self._flight_state: dict[str, str] = {}
        self._serial = 0
        clean = c['ground_handling']['cleaning']
        for port in data.network['vertiports']:
            ident, movement, charging = port['id'], port['movement_area'], port['charging']
            if (movement['tlof_count'] != 1 or movement['fato_count'] != 1
                    or movement['shared_resource'] is not True or movement['max_simultaneous_movements'] != 1):
                raise ValueError('Ова верзија захтева један заједнички TLOF/FATO по вертипорту.')
            if (charging['charger_count'] != port['parking_stands'] or charging['chargers_per_stand'] != 1
                    or charging['max_simultaneous_charging_aircraft'] != port['parking_stands']):
                raise ValueError('Ова верзија захтева по један независан пуњач на свакој позицији.')
            self._units[f'{ident}:movement'] = (ident, 'movement')
            stands = []
            for index in range(1, port['parking_stands'] + 1):
                stand, charger = f'{ident}:stand:{index}', f'{ident}:charger:{index}'
                stands.append(stand)
                self._units[stand] = (ident, 'stand')
                self._units[charger] = (ident, 'charger')
                self._chargers[stand] = charger
            self._stands[ident] = tuple(stands)
            for index in range(1, clean['crews_per_vertiport'] + 1 if clean['enabled'] else 1):
                self._units[f'{ident}:cleaner:{index}'] = (ident, 'cleaner')

    def _id(self, prefix: str) -> str:
        self._serial += 1
        return f'{prefix}:{self._serial:06d}'

    def reservation(self, reservation_id: str) -> Reservation:
        return self._bookings[reservation_id]

    def services_for_visit(self, visit_id: str) -> tuple[Reservation, ...]:
        """Непроменљив снимак услуга везаних за боравак на позицији."""
        self.reservation(visit_id)
        return tuple(r for r in self._bookings.values() if r.stand_visit_id == visit_id)

    def planning_snapshot(self, at_s: float) -> NetworkResources:
        """Копија са спроведеним већ резервисаним кретањима до at_s.

        Позивалац задаје време које није пре тренутног симулационог сата.
        Нема нове тражње, промене SOC или путничких активности. При истом
        времену ослобађање позиције претходи доласку. Оригинал се не мења.
        """
        _time(at_s)
        # Резервације и летни слотови су frozen dataclasses. Делимо те
        # непроменљиве записе, а копирамо све речнике који се мењају.
        result = copy(self)
        result._bookings = dict(self._bookings)
        result._calendar_cache = dict(self._calendar_cache)
        result._physical_stands = dict(self._physical_stands)
        result._flights = dict(self._flights)
        result._flight_state = dict(self._flight_state)
        events = []
        for ident, flight in result._flights.items():
            state = result._flight_state[ident]
            if state == 'reserved' and flight.slot.takeoff_s <= at_s:
                events.append((flight.slot.takeoff_s, 0, ident))
            if state in ('reserved', 'airborne') and flight.slot.landing_end_s <= at_s:
                events.append((flight.slot.landing_end_s, 1, ident))
        for event_time, kind, ident in sorted(events):
            if kind == 0:
                result.depart(ident, event_time)
            else:
                result.arrive(ident, event_time)
        return result

    def calendar(self, resource_id: str) -> ResourceCalendar:
        if resource_id not in self._units:
            raise ValueError(f'Непознат ресурс: {resource_id}.')
        if resource_id not in self._calendar_cache:
            self._calendar_cache[resource_id] = ResourceCalendar(resource_id,
                tuple(r for r in self._bookings.values() if r.resource_id == resource_id))
        return self._calendar_cache[resource_id]

    def resource_ids(self, vertiport: str, kind: str) -> tuple[str, ...]:
        if vertiport not in self._stands or kind not in ('stand', 'movement', 'charger', 'cleaner'):
            raise ValueError('Непознат вертипорт или врста ресурса.')
        return tuple(key for key, value in self._units.items() if value == (vertiport, kind))

    def _commit(self, additions: tuple[Reservation, ...] = (), removals: tuple[str, ...] = ()) -> None:
        """Провера целог кандидата пре измене; замена = уклањање + додавање."""
        candidate = dict(self._bookings)
        affected = {self._bookings[ident].resource_id for ident in removals}
        for ident in removals:
            del candidate[ident]
        for booking in additions:
            if booking.id in candidate or booking.resource_id not in self._units:
                raise ResourceConflict('Поновљена резервација или непознат ресурс.')
            candidate[booking.id] = booking
            affected.add(booking.resource_id)
        # The current state is already valid. Only changed resource calendars
        # can acquire overlaps; validate those before committing anything.
        calendars = dict(self._calendar_cache)
        for resource_id in affected:
            calendars[resource_id] = ResourceCalendar(resource_id,
                tuple(r for r in candidate.values() if r.resource_id == resource_id))
        for booking in candidate.values():
            if booking.stand_visit_id is None:
                continue
            visit = candidate.get(booking.stand_visit_id)
            if (visit is None or visit.purpose != 'stand' or visit.owner != booking.owner
                    or not visit.start_s <= booking.start_s <= booking.end_s <= visit.end_s
                    or self._units[visit.resource_id][0] != self._units[booking.resource_id][0]):
                raise ResourceConflict('Земаљска активност мора бити унутар одговарајућег боравка на позицији.')
        services: dict[tuple[str, str], list[Reservation]] = {}
        for booking in candidate.values():
            if booking.stand_visit_id is not None and booking.end_s > booking.start_s:
                key = (booking.owner, booking.purpose if self.charging_can_overlap_cleaning else 'all')
                services.setdefault(key, []).append(booking)
        for group in services.values():
            previous_end = -math.inf
            for service in sorted(group, key=lambda r: r.start_s):
                if service.start_s < previous_end:
                    raise ResourceConflict('Несагласне истовремене услуге исте летелице.')
                previous_end = service.end_s
        self._bookings = candidate
        self._calendar_cache = calendars

    def park_initial_aircraft(self, aircraft_id: str, vertiport: str, at_s: float) -> Reservation:
        """За почетак репликације; физички заузима позицију до будућег полетања."""
        _time(at_s)
        if aircraft_id in self._physical_stands.values() or any(r.owner == aircraft_id for r in self._bookings.values()):
            raise ResourceConflict('Летелица је већ регистрована у овој репликацији.')
        for stand in self.resource_ids(vertiport, 'stand'):
            if self.calendar(stand).earliest_start(at_s, math.inf) == at_s:
                booking = Reservation(self._id('initial'), stand, aircraft_id, at_s, math.inf, 'stand')
                self._commit((booking,))
                self._physical_stands[stand] = aircraft_id
                return booking
        raise ResourceConflict(f'Нема почетне паркинг позиције на {vertiport}.')

    def physical_stand_occupancy(self, vertiport: str) -> dict[str, str]:
        """Снимак стварно присутних летелица; будуће резервације се не додају."""
        stands = self.resource_ids(vertiport, 'stand')
        return {s: self._physical_stands[s] for s in stands if s in self._physical_stands}

    def reserved_count(self, vertiport: str, kind: str, at_s: float) -> int:
        """Планирана заузетост у датом тренутку, засебно од физичког присуства."""
        _time(at_s)
        units = set(self.resource_ids(vertiport, kind))
        return sum(r.resource_id in units and r.active_at(at_s) for r in self._bookings.values())

    def find_flight_slot(self, estimate: FlightEstimate, earliest_takeoff_s: float,
                         latest_takeoff_s: float = math.inf, *, trace: list | None = None) -> FlightSlot | None:
        """Чиста претрага инфраструктуре; нема измене календара ни чекања у ваздуху.

        Сваки помак долазног слота помера цело полетање на земљи. Позиција на
        одредишту се тражи за [крај слетања, +inf), јер наредни лет још није познат.
        Не проверава летелицу, путнике или SOC.
        """
        t = _time(earliest_takeoff_s)
        latest = _time(latest_takeoff_s, allow_infinity=True)
        origin = self.calendar(f'{estimate.origin}:movement')
        destination = self.calendar(f'{estimate.destination}:movement')
        takeoff, landing, duration = estimate.takeoff.duration_s, estimate.landing.duration_s, estimate.duration_s
        if estimate.origin == estimate.destination or takeoff <= 0 or landing <= 0 or duration < takeoff + landing:
            raise ValueError('Неисправан профил лета.')
        offset = duration - landing
        def note(kind, start, end, port):
            if trace is not None and end > start:
                trace.append(dict(kind=kind, start_s=start, end_s=end, vertiport=port))
        while t <= latest:
            departure = origin.earliest_start(t, takeoff)
            if departure is None:
                note('origin_tlof', t, math.inf, estimate.origin)
                return None
            note('origin_tlof', t, departure, estimate.origin)
            arrival = destination.earliest_start(departure + offset, landing)
            if arrival is None:
                note('destination_tlof', departure, math.inf, estimate.destination)
                return None
            candidate = _shift_to_event(departure, arrival, offset)
            note('destination_tlof', departure, candidate, estimate.destination)
            choices = []
            for stand in self.resource_ids(estimate.destination, 'stand'):
                free = self.calendar(stand).earliest_start(candidate + duration, math.inf)
                if free is not None:
                    choices.append((free, stand))
            if not choices:
                note('destination_stand', candidate, math.inf, estimate.destination)
                return None
            free, stand = min(choices)
            shifted = _shift_to_event(candidate, free, duration)
            note('destination_stand', candidate, shifted, estimate.destination)
            candidate = shifted
            if candidate == t:
                return FlightSlot(estimate.origin, estimate.destination, t, t + takeoff,
                                  t + offset, t + duration, stand)
            t = candidate
        return None

    def reserve_flight(self, flight_id: str, aircraft_id: str, estimate: FlightEstimate,
                       origin_visit_id: str, earliest_takeoff_s: float,
                       latest_takeoff_s: float = math.inf) -> FlightReservation:
        if flight_id in self._flights or any(f.aircraft_id == aircraft_id and self._flight_state[key] in ('reserved', 'airborne')
                                           for key, f in self._flights.items()):
            raise ResourceConflict('Лет већ постоји или летелица има незавршену резервисану етапу.')
        visit = self.reservation(origin_visit_id)
        if (visit.purpose != 'stand' or visit.owner != aircraft_id
                or self._units[visit.resource_id] != (estimate.origin, 'stand')
                or self._physical_stands.get(visit.resource_id) != aircraft_id):
            raise ResourceConflict('Летелица мора бити на полазној позицији.')
        earliest = max(_time(earliest_takeoff_s), visit.start_s)
        # Већ резервисане услуге морају се завршити. Пуњење се може претходно
        # скратити truncate_service(), када је летелица енергетски спремна.
        for service in self._bookings.values():
            if service.stand_visit_id == origin_visit_id and service.end_s > service.start_s:
                lead = self.stop_charge_lead_s if service.purpose == 'charging' else 0
                earliest = max(earliest, service.end_s + lead)
        slot = self.find_flight_slot(estimate, earliest, min(latest_takeoff_s, visit.end_s))
        if slot is None:
            raise ResourceConflict('Нема усклађеног полетања, слетања и одредишне позиције.')
        takeoff = Reservation(f'{flight_id}:takeoff', f'{estimate.origin}:movement', aircraft_id,
                              slot.takeoff_s, slot.takeoff_end_s, 'takeoff')
        landing = Reservation(f'{flight_id}:landing', f'{estimate.destination}:movement', aircraft_id,
                              slot.landing_start_s, slot.landing_end_s, 'landing')
        destination = Reservation(f'{flight_id}:stand', slot.destination_stand_id, aircraft_id,
                                  slot.landing_end_s, math.inf, 'stand')
        result = FlightReservation(flight_id, aircraft_id, slot, origin_visit_id, visit.end_s,
                                   takeoff.id, landing.id, destination.id)
        self._commit((replace(visit, end_s=slot.takeoff_s), takeoff, landing, destination), (visit.id,))
        self._flights[flight_id] = result
        self._flight_state[flight_id] = 'reserved'
        return result

    def cancel_flight(self, flight_id: str) -> None:
        """Само пре полетања. Ако постоје зависне резервације, отказ се одбија
        без измена; прво отказати њих, па поновити овај позив.
        """
        f = self._flights[flight_id]
        if self._flight_state[flight_id] != 'reserved':
            raise ResourceConflict('Може се отказати само лет који још није започет.')
        visit = self.reservation(f.origin_visit_id)
        self._commit((replace(visit, end_s=f.origin_previous_end_s),),
                     (visit.id, f.takeoff_reservation_id, f.landing_reservation_id, f.destination_visit_id))
        self._flight_state[flight_id] = 'cancelled'

    def depart(self, flight_id: str, at_s: float) -> None:
        f = self._flights[flight_id]
        origin = self.reservation(f.origin_visit_id).resource_id
        if (self._flight_state[flight_id] != 'reserved' or at_s != f.slot.takeoff_s
                or self._physical_stands.get(origin) != f.aircraft_id):
            raise ResourceConflict('Неисправан догађај полетања.')
        del self._physical_stands[origin]
        self._flight_state[flight_id] = 'airborne'

    def arrive(self, flight_id: str, at_s: float) -> None:
        f = self._flights[flight_id]
        stand = f.slot.destination_stand_id
        if (self._flight_state[flight_id] != 'airborne' or at_s != f.slot.landing_end_s
                or stand in self._physical_stands):
            raise ResourceConflict('Неисправан долазак или физички заузета одредишна позиција.')
        self._physical_stands[stand] = f.aircraft_id
        self._flight_state[flight_id] = 'arrived'

    def reserve_service(self, visit_id: str, kind: str, earliest_start_s: float,
                        duration_s: float, *, latest_end_s: float = math.inf) -> Reservation:
        """Резервише пуњење или екипу за чишћење у оквиру боравка на позицији.

        За чишћење earliest_start_s мора бити крај искрцавања или касније.
        За почетну флоту delay пуњења нема ефекта при усвојеном SOC=100%.
        Преклапање са B/D проверава simulation; пуњач и чистач су независни
        ресурси и могу се преклапати када конфигурација то дозвољава.
        """
        visit = self.reservation(visit_id)
        if visit.purpose != 'stand' or kind not in ('charging', 'cleaning'):
            raise ValueError('Потребан је боравак на позицији и врста charging/cleaning.')
        earliest = max(_time(earliest_start_s), visit.start_s)
        duration = _time(duration_s)
        if duration <= 0:
            raise ValueError('Услуга мора трајати позитивно, коначно време.')
        end_limit = min(_time(latest_end_s, allow_infinity=True), visit.end_s)
        port = self._units[visit.resource_id][0]
        if kind == 'charging':
            if not self.charging_enabled:
                raise ResourceConflict('Пуњење је искључено у конфигурацији.')
            earliest = max(earliest, visit.start_s + self.start_charge_delay_s)
            end_limit = min(end_limit, visit.end_s - self.stop_charge_lead_s)
            units = (self._chargers[visit.resource_id],)
        else:
            units = self.resource_ids(port, 'cleaner')
        choices = []
        for unit in units:
            start = self.calendar(unit).earliest_start(earliest, duration)
            if start is not None and start + duration <= end_limit:
                choices.append((start, unit))
        if not choices:
            raise ResourceConflict('Нема слободног ресурса за услугу у задатом интервалу.')
        start, unit = min(choices)
        booking = Reservation(self._id(kind), unit, visit.owner, start, start + duration, kind, visit_id)
        self._commit((booking,))
        return booking

    def cancel_service(self, reservation_id: str) -> None:
        """За још незапочету услугу; за започету користити truncate_service.
        Позивалац је одговоран за статус догађаја у симулационом сату.
        """
        if self.reservation(reservation_id).stand_visit_id is None:
            raise ValueError('Ово није резервација земаљске услуге.')
        self._commit(removals=(reservation_id,))

    def truncate_service(self, reservation_id: str, at_s: float) -> Reservation:
        """Прекид услуге уз очување протеклог интервала за касније показатеље."""
        booking = self.reservation(reservation_id)
        if booking.stand_visit_id is None or not booking.start_s <= _time(at_s) <= booking.end_s:
            raise ValueError('Неисправан тренутак прекида услуге.')
        shortened = replace(booking, end_s=at_s)
        self._commit((shortened,), (booking.id,))
        return shortened


def main(argv: list[str] | None = None) -> int:
    from .flight_model import FlightModel, create_initial_fleet
    from .input_data import InputDataError, load_inputs

    parser = argparse.ArgumentParser(description='Провера инфраструктуре и пример резервације лета; без симулације.')
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'config.json')
    args = parser.parse_args(argv)
    try:
        data = load_inputs(args.config)
        resources = NetworkResources(data)
        configuration = data.configurations[0]
        fleet = create_initial_fleet(data, configuration)
        visits = {a.id: resources.park_initial_aircraft(a.id, a.location, 0) for a in fleet}
        print('Вертипорт   TLOF/FATO  Позиције  Пуњачи  Екипе  Почетно на позицијама')
        for port in data.network['vertiports']:
            ident = port['id']
            counts = [len(resources.resource_ids(ident, k)) for k in ('movement', 'stand', 'charger', 'cleaner')]
            print(f'{ident:10} {counts[0]:9} {counts[1]:9} {counts[2]:7} {counts[3]:6} '
                  f'{len(resources.physical_stand_occupancy(ident)):21}')
        aircraft = next(a for a in fleet if a.location == 'LYBE')
        estimate = FlightModel.from_inputs(data).estimate('LYBE', 'BW')
        flight = resources.reserve_flight('DEMO', aircraft.id, estimate, visits[aircraft.id].id, 600)
        print(f'Пример LYBE → BW: полетање {flight.slot.takeoff_s:.3f} s; '
              f'слетање [{flight.slot.landing_start_s:.3f}, {flight.slot.landing_end_s:.3f}) s.')
        print(f'Резервисана позиција: {flight.slot.destination_stand_id}; трајање лета {estimate.duration_s / 60:.3f} min.')
        print('Ово је провера ресурса; путници, SOC и опслуживање нису симулирани.')
    except (InputDataError, ValueError, KeyError) as exc:
        print(f'ГРЕШКА: {exc}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    raise SystemExit(main())
