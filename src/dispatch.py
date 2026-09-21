from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from .charging import ChargingModel
from .entities import Aircraft, AircraftStatus, FlightEstimate, Passenger, PassengerGroup, PassengerStatus
from .flight_model import FlightModel
from .resources import FlightSlot, NetworkResources, ResourceConflict

if TYPE_CHECKING:
    from .input_data import InputData


def _finite(value: float, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f'{name}: очекиван коначан број.')
    return float(value)


@dataclass(frozen=True, slots=True)
class AircraftAvailability:
    """Стање у тренутку ослобађања летелице, које обезбеђује симулација.

    energy_kwh је енергија баш у ready_s, после већ заказаних летова и
    опслуживања. За заузету летелицу ready_s и visit_id морају одговарати
    њеним познатим активностима; диспечер их не нагађа из activity_end_s.
    reserved_group_id задржава започето позиционирање/прихваћену преддоделу.
    Привремену преддоделу симулација преиспитује на свакој промени стања.
    """
    aircraft_id: str
    location: str
    visit_id: str
    ready_s: float
    energy_kwh: float
    completed_legs_since_cleaning: int
    reserved_group_id: str | None = None

    @classmethod
    def from_idle(cls, aircraft: Aircraft, visit_id: str, now_s: float) -> AircraftAvailability:
        _finite(now_s, 'now_s')
        if aircraft.status != AircraftStatus.IDLE or aircraft.location is None:
            raise ValueError('За заузету летелицу потребна је експлицитна прогноза доступности.')
        if aircraft.energy_updated_at_s > now_s or (aircraft.is_charging and aircraft.energy_updated_at_s != now_s):
            raise ValueError('Пре избора ажурирати енергију летелице на тренутни симулациони сат.')
        return cls(aircraft.id, aircraft.location, visit_id, now_s, aircraft.energy_kwh,
                   aircraft.completed_legs_since_cleaning,
                   aircraft.next_group_id or aircraft.assigned_group_id)


@dataclass(frozen=True, slots=True)
class GroundTask:
    kind: str
    start_s: float
    end_s: float
    resource_id: str
    already_reserved: bool = False


@dataclass(frozen=True, slots=True)
class LegPlan:
    estimate: FlightEstimate
    slot: FlightSlot
    boarding_start_s: float | None
    boarding_end_s: float | None
    departure_energy_kwh: float
    landing_energy_kwh: float
    ground_tasks: tuple[GroundTask, ...]
    energy_ready_s: float | None = None
    resource_delays: tuple[GroundTask, ...] = ()
    charging_required: bool = False


@dataclass(frozen=True, slots=True)
class DispatchDecision:
    group_id: str
    passenger_ids: tuple[str, ...]
    aircraft_id: str
    reason: str  # known_passenger_demand / parking_capacity_relief
    generated_at_s: float
    aircraft_ready_s: float
    legs: tuple[LegPlan, ...]  # optional repositioning, then passenger flight
    boarding_start_s: float | None

    @property
    def is_preassignment(self) -> bool:
        return self.aircraft_ready_s > self.generated_at_s

    @property
    def next_action_s(self) -> float:
        first = self.legs[0]
        times = [first.slot.takeoff_s]
        if first.boarding_start_s is not None:
            times.append(first.boarding_start_s)
        times.extend(task.start_s for task in first.ground_tasks if not task.already_reserved)
        return max(self.generated_at_s, min(times))


class Dispatcher:
    def __init__(self, data: InputData) -> None:
        self.data = data
        self.rules = data.config['dispatch']
        self._validate_rules()
        self.flight_model = FlightModel.from_inputs(data)
        self.charging = {p['id']: ChargingModel.from_inputs(data, p['id']) for p in data.network['vertiports']}
        self.capacity = data.aircraft['capacity']['passenger_seats']
        self.battery_capacity = data.aircraft['battery']['modeled_usable_capacity_kwh']
        self.preparation_s = 60 * data.config['ground_handling']['departure_finalization_min']
        self.wait_limit_s = 60 * data.config['service']['max_wait_to_boarding_min']
        self.cleaning = data.config['ground_handling']['cleaning']
        self.airport = next(p['id'] for p in data.network['vertiports'] if p['role'] == 'airport')
        self.passengers: dict[str, Passenger] = {}

    def _validate_rules(self) -> None:
        for key, expected in {
            'grouping_keys': ['origin_vertiport', 'destination_vertiport'],
            'within_queue_order': 'request_time_then_passenger_id',
            'boarding_trigger': 'full_group_or_oldest_passenger_wait_reaches_limit',
            'group_lock_event': 'boarding_start', 'allow_partial_groups': True,
            'aircraft_selection': 'earliest_feasible_boarding_start',
            'competing_group_order': ['origin_priority_rank_ascending', 'oldest_request_time_ascending', 'group_id_ascending'],
            'aircraft_tie_breakers': ['repositioning_duration_ascending', 'aircraft_id_ascending'],
            'preassignment_depth_per_aircraft': 1, 'preassignment_recheck_on_state_change': True,
            'future_passenger_request_visibility': False,
        }.items():
            if self.rules[key] != expected:
                raise ValueError(f'dispatch.{key}: неподржано правило.')
        r = self.rules['repositioning']
        if (r['may_start_before_group_boarding_trigger'] is not True
                or r['check_energy_for_reposition_and_onward_passenger_service'] is not True
                or r['may_charge_at_pickup_vertiport'] is not True):
            raise ValueError('Неподржана варијанта правила позиционирања.')
        charge = self.data.config['charging']
        if not all(charge[k] for k in ('can_overlap_boarding', 'can_overlap_disembarking',
                                       'can_overlap_cleaning', 'interrupt_for_feasible_assigned_departure')):
            raise ValueError('Ова верзија користи усвојено преклапање и прекид пуњења.')

    def handling_seconds(self, kind: str, passenger_count: int) -> float:
        if kind not in ('boarding', 'disembarking') or type(passenger_count) is not int or not 0 <= passenger_count <= self.capacity:
            raise ValueError('Неисправна врста опслуживања или број путника.')
        model = self.data.config['ground_handling'][kind]
        return 60 * (model['duration_for_zero_passengers_min'] if passenger_count == 0
                     else model['fixed_min'] + passenger_count * model['per_passenger_min'])

    def admit(self, passengers: Iterable[Passenger], now_s: float) -> None:
        """Симулација позива само за стварно пристигле захтеве; атомска провера серије."""
        _finite(now_s, 'now_s')
        batch = tuple(passengers)
        if len({p.id for p in batch}) != len(batch):
            raise ValueError('Поновљен путник у серији.')
        for p in batch:
            if (p.id in self.passengers or p.request_time_s > now_s
                    or p.status not in (PassengerStatus.NOT_ARRIVED, PassengerStatus.WAITING)):
                raise ValueError('Путник је већ пријављен, још није стигао или је већ у опслуживању.')
            self.flight_model.estimate(p.origin, p.destination)
        for p in batch:
            p.status = PassengerStatus.WAITING
            self.passengers[p.id] = p

    def groups(self, now_s: float) -> tuple[PassengerGroup, ...]:
        """FIFO по смеру, највише capacity путника. Незакључане групе се допуњују."""
        _finite(now_s, 'now_s')
        queues: dict[tuple[str, str], list[Passenger]] = {}
        for p in self.passengers.values():
            if p.status == PassengerStatus.WAITING:
                if p.request_time_s > now_s:
                    raise ValueError('Диспечер не може враћати време пре већ примљених захтева.')
                queues.setdefault((p.origin, p.destination), []).append(p)
        groups = []
        for (origin, destination), queue in sorted(queues.items()):
            queue.sort(key=lambda p: (p.request_time_s, p.id))
            for start in range(0, len(queue), self.capacity):
                members = queue[start:start + self.capacity]
                ident = f'G:{origin}:{destination}:{members[0].id}'
                group = PassengerGroup(ident, origin, destination, self.capacity)
                for p in members:
                    # Смена незакључане групе дозвољена је при поновном FIFO груписању.
                    p.group_id = None
                    group.add_passenger(p)
                groups.append(group)
        return tuple(sorted(groups, key=lambda g: (
            self.rules['origin_priority_rank'][g.origin], g.oldest_request_time_s, g.id)))

    def trigger_time(self, group: PassengerGroup) -> float:
        if not group.passenger_ids or group.oldest_request_time_s is None or group.latest_request_time_s is None:
            raise ValueError('Празна или неисправна група.')
        trigger = (group.latest_request_time_s if len(group.passenger_ids) == self.capacity
                   else group.oldest_request_time_s + self.wait_limit_s)
        deadline = self.rules['deadline_override']
        if deadline['enabled'] and group.destination == self.airport and group.latest_desired_takeoff_s is not None:
            trigger = min(trigger, group.latest_desired_takeoff_s
                          - self.handling_seconds('boarding', len(group.passenger_ids)) - self.preparation_s)
        return max(trigger, group.latest_request_time_s)

    def next_group_wakeup(self, now_s: float) -> float | None:
        times = [self.trigger_time(g) for g in self.groups(now_s)]
        return min((t for t in times if t > now_s), default=None)

    def _validate_availability(self, availability: tuple[AircraftAvailability, ...], now_s: float) -> None:
        if len({a.aircraft_id for a in availability}) != len(availability):
            raise ValueError('Иста летелица има више прогноза доступности.')
        assigned = [a.reserved_group_id for a in availability if a.reserved_group_id is not None]
        if len(set(assigned)) != len(assigned):
            raise ValueError('Иста група је већ додељена различитим летелицама.')
        for a in availability:
            if (_finite(a.ready_s, 'ready_s') < now_s or a.location not in self.charging
                    or not 0 <= _finite(a.energy_kwh, 'energy_kwh') <= self.battery_capacity
                    or type(a.completed_legs_since_cleaning) is not int or a.completed_legs_since_cleaning < 0):
                raise ValueError('Неисправна или застарела прогноза доступности летелице.')

    def _leg(self, available: AircraftAvailability, estimate: FlightEstimate, passenger_count: int,
             trigger_s: float, resources: NetworkResources, preview_id: str) -> tuple[LegPlan, NetworkResources, int]:
        """Разматрање на копији; враћа резервисану копију и број етапа после лета."""
        shadow = resources.planning_snapshot(available.ready_s)
        visit = shadow.reservation(available.visit_id)
        if (visit.owner != available.aircraft_id or visit.purpose != 'stand'
                or visit.resource_id not in shadow.resource_ids(available.location, 'stand')
                or shadow.physical_stand_occupancy(available.location).get(visit.resource_id) != available.aircraft_id):
            raise ResourceConflict('Прогноза нема одговарајућу позицију по ослобађању летелице.')
        tasks = []
        # Постојеће пуњење до 100% не сме да блокира ранији задатак.
        for service in shadow.services_for_visit(visit.id):
            if service.purpose == 'charging' and service.end_s > available.ready_s:
                if service.start_s >= available.ready_s:
                    shadow.cancel_service(service.id)
                else:
                    shadow.truncate_service(service.id, available.ready_s)
        ready = available.ready_s
        legs = available.completed_legs_since_cleaning
        existing_clean = [s for s in shadow.services_for_visit(visit.id)
                          if s.purpose == 'cleaning' and s.end_s > ready]
        if existing_clean:
            ready = max(s.end_s for s in existing_clean)
            tasks.extend(GroundTask('cleaning', s.start_s, s.end_s, s.resource_id, True) for s in existing_clean)
            legs = 0
        elif self.cleaning['enabled'] and legs >= self.cleaning['every_completed_legs']:
            service = shadow.reserve_service(visit.id, 'cleaning', ready, 60 * self.cleaning['duration_min'])
            ready = service.end_s
            tasks.append(GroundTask('cleaning', service.start_s, service.end_s, service.resource_id))
            legs = 0
        charger = self.charging[available.location]
        charge_start = max(available.ready_s, visit.start_s + charger.start_delay_s)
        requirement = estimate.required_departure_energy_kwh
        needed = charger.time_to_energy(available.energy_kwh, requirement)
        if math.isinf(needed):
            raise ResourceConflict('Потребна енергија није достижна.', 'energy_unreachable',
                                   operation=estimate.operation.value, vertiport=available.location)
        energy_ready = available.ready_s if needed == 0 else charge_start + needed
        boarding = self.handling_seconds('boarding', passenger_count)
        # KPI је почетак B. После B може уследити чекање пуњења/слота;
        # тај интервал не сме прећутно да се дода чекању ПРЕ укрцавања.
        start_boarding = max(ready, trigger_s) if passenger_count else None
        end_boarding = start_boarding + boarding if passenger_count else None
        earliest_takeoff = max(max(ready, trigger_s) + boarding, energy_ready) + self.preparation_s
        trace = []
        slot = shadow.find_flight_slot(estimate, earliest_takeoff, visit.end_s, trace=trace)
        if slot is None:
            raise ResourceConflict('Нема изводљивог слота и одредишне позиције.',
                                   trace[-1]['kind'] if trace else 'resource_conflict',
                                   operation=estimate.operation.value,
                                   vertiport=trace[-1]['vertiport'] if trace else estimate.destination)
        charge_end = slot.takeoff_s - self.preparation_s
        charge_duration = max(0.0, charge_end - charge_start)
        departure_energy = charger.energy_after(available.energy_kwh, charge_duration, target_soc=1)
        if departure_energy < requirement:
            if requirement - departure_energy > 1e-8:
                raise ResourceConflict('Недовољна енергија после предвиђеног пуњења.')
            departure_energy = requirement  # само машинска грешка на граници
        time_to_full = charger.time_to_energy(available.energy_kwh, self.battery_capacity)
        active_charge = min(charge_duration, time_to_full)
        if charger.enabled and departure_energy > available.energy_kwh and active_charge > 0:
            service = shadow.reserve_service(visit.id, 'charging', charge_start, active_charge,
                                             latest_end_s=charge_end)
            if service.start_s != charge_start:
                raise ResourceConflict('Пуњач није доступан у израчунатом интервалу.')
            tasks.append(GroundTask('charging', service.start_s, service.end_s, service.resource_id))
        reserved = shadow.reserve_flight(preview_id, available.aircraft_id, estimate, visit.id,
                                         slot.takeoff_s, slot.takeoff_s)
        tasks.append(GroundTask('departure_finalization', slot.takeoff_s - self.preparation_s,
                                slot.takeoff_s, visit.resource_id))
        result = LegPlan(estimate, reserved.slot, start_boarding, end_boarding, departure_energy,
                         departure_energy - estimate.energy_kwh,
                         tuple(sorted(tasks, key=lambda task: (task.start_s, task.kind))), energy_ready,
                         tuple(GroundTask(row['kind'], row['start_s'] - self.preparation_s,
                                          row['end_s'] - self.preparation_s, row['vertiport']) for row in trace), needed > 0)
        increment = int(estimate.operation.value == 'passenger' or self.cleaning['include_repositioning_legs'])
        return result, shadow, legs + increment

    def _candidate(self, group: PassengerGroup, available: AircraftAvailability,
                   resources: NetworkResources, now_s: float) -> DispatchDecision:
        prefix = f'preview:{group.id}:{available.aircraft_id}'
        legs = []
        shadow = resources
        current = available
        if current.location != group.origin:
            if not self.rules['repositioning']['enabled']:
                raise ResourceConflict('Позиционирање је искључено.')
            estimate = self.flight_model.estimate(current.location, group.origin, 'repositioning')
            reposition, shadow, count = self._leg(current, estimate, 0, now_s, shadow, prefix + ':empty')
            legs.append(reposition)
            shadow = shadow.planning_snapshot(reposition.slot.landing_end_s)
            current = AircraftAvailability(current.aircraft_id, group.origin, prefix + ':empty:stand',
                                           reposition.slot.landing_end_s, reposition.landing_energy_kwh, count,
                                           group.id)
        estimate = self.flight_model.estimate(group.origin, group.destination)
        passenger, _, _ = self._leg(current, estimate, len(group.passenger_ids), self.trigger_time(group),
                                    shadow, prefix + ':passenger')
        legs.append(passenger)
        return DispatchDecision(group.id, group.passenger_ids, available.aircraft_id, 'known_passenger_demand',
                                now_s, available.ready_s, tuple(legs), passenger.boarding_start_s)

    def choose(self, now_s: float, availability: Iterable[AircraftAvailability],
               resources: NetworkResources, *, excluded_group_ids: frozenset[str] = frozenset(),
               audit: list | None = None) -> DispatchDecision | None:
        """Једна одлука по приоритету група, па најранијем почетку B.

        За заузете летелице користи се позната прогноза ослобађања. Извршилац
        прихвата највише једну наредну групу по летелици и прослеђује је као
        reserved_group_id у следећем позиву. Без кандидата путници остају у реду.
        """
        _finite(now_s, 'now_s')
        availability = tuple(availability)
        self._validate_availability(availability, now_s)
        groups = tuple(g for g in self.groups(now_s) if g.id not in excluded_group_ids)
        pinned = {a.reserved_group_id: a.aircraft_id for a in availability if a.reserved_group_id is not None}
        for group in groups:
            choices = []
            for aircraft in availability:
                row = dict(time_s=now_s, group_id=group.id, passenger_ids='|'.join(group.passenger_ids),
                           origin=group.origin, destination=group.destination, aircraft_id=aircraft.aircraft_id,
                           available_at_s=aircraft.ready_s, available_location=aircraft.location,
                           forecast_energy_kwh=aircraft.energy_kwh, status='excluded', reason='',
                           boarding_start_s=None, takeoff_s=None, repositioning=False,
                           energy_ready_s=None, resource_constraints='', constraint_vertiport='', selected=False)
                if (aircraft.reserved_group_id not in (None, group.id)
                        or (group.id in pinned and pinned[group.id] != aircraft.aircraft_id)
                        or (aircraft.ready_s > now_s and not self.rules['preassignment_enabled'])):
                    row['reason'] = 'reserved_for_other_group' if aircraft.reserved_group_id not in (None, group.id) else 'pinned_aircraft_policy'
                    if audit is not None: audit.append(row)
                    continue
                try:
                    plan = self._candidate(group, aircraft, resources, now_s)
                    choices.append(plan)
                    row.update(status='feasible', boarding_start_s=plan.boarding_start_s,
                               takeoff_s=plan.legs[-1].slot.takeoff_s, repositioning=len(plan.legs)>1,
                               energy_ready_s=plan.legs[0].energy_ready_s,
                               resource_constraints='|'.join(sorted({t.kind for leg in plan.legs for t in leg.resource_delays})))
                except ResourceConflict as exc:
                    row.update(status='infeasible', reason=exc.code,
                               constraint_vertiport=exc.details.get('vertiport', ''))
                    if audit is not None: audit.append(row)
                    continue
                if audit is not None: audit.append(row)
            if choices:
                selected = min(choices, key=lambda p: (p.boarding_start_s,
                    sum(leg.estimate.duration_s for leg in p.legs if leg.estimate.operation.value == 'repositioning'),
                    p.aircraft_id))
                if audit is not None:
                    for row in audit:
                        if row['group_id'] == group.id and row['aircraft_id'] == selected.aircraft_id:
                            row['selected'] = True
                return selected
        return self._parking_relief(groups, availability, resources, now_s)

    def _parking_relief(self, groups: tuple[PassengerGroup, ...], availability: tuple[AircraftAvailability, ...],
                        resources: NetworkResources, now_s: float) -> DispatchDecision | None:
        """Резервна одлука: једна празна етапа ради ослобађања пуног одредишта.

        Користи се само када нема изводљиве путничке доделе. Узима слободну,
        недодељену летелицу са блокираног вертипорта. После прихватања ове
        одлуке choose() се понавља; не гарантује решење сваке могуће блокаде.
        """
        repositioning = self.rules['repositioning']
        if not repositioning['enabled'] or 'parking_capacity_relief' not in repositioning['reasons']:
            return None
        for group in groups:
            if any(resources.calendar(s).earliest_start(now_s, math.inf) is not None
                   for s in resources.resource_ids(group.destination, 'stand')):
                continue
            choices = []
            for aircraft in availability:
                if aircraft.location != group.destination or aircraft.ready_s != now_s or aircraft.reserved_group_id is not None:
                    continue
                for destination in sorted(self.charging):
                    if destination == aircraft.location:
                        continue
                    estimate = self.flight_model.estimate(aircraft.location, destination, 'repositioning')
                    try:
                        leg, _, _ = self._leg(aircraft, estimate, 0, now_s, resources,
                                              f'preview:relief:{aircraft.aircraft_id}:{destination}')
                    except ResourceConflict:
                        continue
                    choices.append(DispatchDecision(group.id, (), aircraft.aircraft_id, 'parking_capacity_relief',
                                                    now_s, now_s, (leg,), None))
            if choices:
                return min(choices, key=lambda p: (p.legs[0].slot.takeoff_s,
                            p.legs[0].estimate.duration_s, p.aircraft_id, p.legs[0].estimate.destination))
        return None

    def lock_for_boarding(self, decision: DispatchDecision, now_s: float) -> PassengerGroup:
        """После успешне стварне резервације ресурса, у догађају почетка B.

        Извршилац пре позива мора потврдити ресурсни/енергетски план. Овде се
        проверава свежа одлука и непромењен списак већ пристиглих путника.
        Не позивати за стари план који је садржао још неизвршен позициони лет.
        """
        if (decision.reason != 'known_passenger_demand' or len(decision.legs) != 1
                or decision.generated_at_s != now_s or decision.boarding_start_s != now_s):
            raise ValueError('Потребна је свежа одлука за почетак укрцавања у овом тренутку.')
        group = next((g for g in self.groups(now_s) if g.id == decision.group_id), None)
        if group is None or group.passenger_ids != decision.passenger_ids or self.trigger_time(group) > now_s:
            raise ValueError('Група се променила или још није спремна за укрцавање.')
        group.assigned_aircraft_id = decision.aircraft_id
        group.lock(now_s)
        for ident in group.passenger_ids:
            passenger = self.passengers[ident]
            passenger.status = PassengerStatus.BOARDING
            passenger.boarding_start_s = now_s
        return group


def main(argv: list[str] | None = None) -> int:
    from .demand import generate_demand
    from .flight_model import create_initial_fleet
    from .input_data import InputDataError, load_inputs

    parser = argparse.ArgumentParser(description='Прва одлука диспечера за пристигли захтев; без пуне симулације.')
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'config.json')
    parser.add_argument('--scenario', default='low', help='Подразумевано low (1%%).')
    args = parser.parse_args(argv)
    try:
        data = load_inputs(args.config, overrides={'demand_scenario': args.scenario})
        demand = generate_demand(data)
        dispatcher = Dispatcher(data)
        if not demand.records:
            print('Нема путника; нема диспечерске одлуке.')
            return 0
        now = demand.start_time_s
        resources = NetworkResources(data)
        fleet = create_initial_fleet(data, data.configurations[0], now)
        visits = {a.id: resources.park_initial_aircraft(a.id, a.location, now) for a in fleet}
        dispatcher.admit((p for p in demand.new_passengers() if p.request_time_s <= now), now)
        available = [AircraftAvailability.from_idle(a, visits[a.id].id, now) for a in fleet]
        decision = dispatcher.choose(now, available, resources)
        print(f'Познатих захтева: {len(dispatcher.passengers)}; време: {demand.local_datetime(now).isoformat(timespec="seconds")}.')
        if decision is None:
            print('Тренутно нема изводљиве одлуке; путници остају у реду.')
        else:
            print(f'Летелица: {decision.aircraft_id}; путника у групи: {len(decision.passenger_ids)}; разлог: {decision.reason}.')
            for leg in decision.legs:
                print(f'  {leg.estimate.origin} → {leg.estimate.destination} ({leg.estimate.operation.value}): '
                      f'полетање {demand.local_datetime(leg.slot.takeoff_s).isoformat(timespec="seconds")}; '
                      f'енергија пре/после лета {leg.departure_energy_kwh:.3f}/{leg.landing_energy_kwh:.3f} kWh.')
            if decision.boarding_start_s is not None:
                print(f'Почетак укрцавања: {demand.local_datetime(decision.boarding_start_s).isoformat(timespec="seconds")}.')
        print('Приказан је предлог; стварне резервације и летови нису извршени. Пуна симулација следи у simulation.py.')
    except (InputDataError, ValueError, KeyError) as exc:
        print(f'ГРЕШКА: {exc}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    raise SystemExit(main())
