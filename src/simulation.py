from __future__ import annotations

import argparse
import heapq
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .demand import DemandRealization, generate_demand
from .diagnostics import Diagnostics, resource_utilization
from .dispatch import AircraftAvailability, DispatchDecision, Dispatcher, LegPlan
from .entities import Aircraft, AircraftStatus, Flight, Passenger, PassengerStatus
from .flight_model import create_initial_fleet
from .resources import FlightReservation, NetworkResources, Reservation, ResourceConflict

if TYPE_CHECKING:
    from .input_data import InputData


@dataclass(order=True, slots=True)
class _Event:
    time_s: float
    priority: int
    sequence: int
    kind: str = field(compare=False)
    entity_id: str = field(compare=False, default='')
    token: str = field(compare=False, default='')


@dataclass(slots=True)
class _Charge:
    reservation_id: str
    start_s: float
    end_s: float
    initial_energy_kwh: float
    vertiport: str
    target_soc: float


@dataclass(slots=True)
class _Execution:
    flight: Flight
    reservation: FlightReservation
    plan: LegPlan
    disembarking_end_s: float
    ready_s: float
    legs_after_landing: int
    cleaning: Reservation | None


@dataclass(slots=True)
class _Runtime:
    aircraft: Aircraft
    visit_id: str
    charge: _Charge | None = None
    active: _Execution | None = None
    ground_cleaning: Reservation | None = None
    pinned_group_id: str | None = None
    last_leg_operation: str | None = None


@dataclass(slots=True)
class SimulationResult:
    summary: dict
    passengers: list[Passenger]
    flights: list[Flight]
    event_log: list[dict]
    demand: DemandRealization
    wait_intervals: list[dict] = field(default_factory=list)
    candidate_diagnostics: list[dict] = field(default_factory=list)
    aircraft_activities: list[dict] = field(default_factory=list)
    queue_intervals: list[dict] = field(default_factory=list)
    resource_metrics: list[dict] = field(default_factory=list)
    flight_details: dict = field(default_factory=dict)
    diagnostic_totals: dict = field(default_factory=dict)


class Simulation:
    """Нова инстанца за сваку репликацију/величину флоте.

    Исти непроменљив DemandRealization може се делити између флота;
    new_passengers() и create_initial_fleet() стварају ново променљиво стање.
    """

    def __init__(self, data: InputData, configuration: dict, demand: DemandRealization,
                 *, record_events: bool = True, max_events: int = 1_000_000,
                 collect_diagnostics: bool = True, retain_details: bool = True) -> None:
        if configuration['demand_scenario'] != demand.scenario:
            raise ValueError('Сценарио реализације тражње не одговара конфигурацији.')
        if type(max_events) is not int or max_events < 1:
            raise ValueError('max_events мора бити позитиван цео број.')
        self.data, self.configuration, self.demand = data, configuration, demand
        self.now = demand.start_time_s
        self.start_s = self.now
        self.dispatcher = Dispatcher(data)
        self.resources = NetworkResources(data)
        self.passengers = {p.id: p for p in demand.new_passengers()}
        self.runtime: dict[str, _Runtime] = {}
        self.flights: list[Flight] = []
        self.executions: dict[str, _Execution] = {}
        self.log: list[dict] = []
        self.record_events = record_events
        self.max_events = max_events
        self.queue: list[_Event] = []
        self._sequence = 0
        self._wake_times: set[float] = set()
        self.completed = 0
        self.event_count = 0
        self.cleanings_completed = 0
        self.energy_added_kwh = 0.0
        self.energy_used_kwh = 0.0
        self.eps = data.config['simulation']['time_comparison_tolerance_s']
        self.minimum_soc = data.config['fleet']['initial_soc']
        self._has_run = False
        self.diagnostics = Diagnostics(retain_details) if collect_diagnostics else None
        self.waiting_groups, self.pending_decisions, self.last_candidate_rows = [], {}, {}
        self.flight_energy_context = {}
        for aircraft in create_initial_fleet(data, configuration, self.now):
            visit = self.resources.park_initial_aircraft(aircraft.id, aircraft.location, self.now)
            aircraft.stand_id = visit.resource_id
            self.runtime[aircraft.id] = _Runtime(aircraft, visit.id)
        self.initial_energy_kwh = sum(r.aircraft.energy_kwh for r in self.runtime.values())
        for p in self.passengers.values():
            self._schedule(p.request_time_s, 4, 'request', p.id)
        if self.passengers:
            for rt in self.runtime.values():
                self._start_idle_charging(rt)
                self._ensure_initial_cleaning(rt)
            self._wake(self.now)

    def _schedule(self, at_s: float, priority: int, kind: str, ident: str = '', token: str = '') -> None:
        if not math.isfinite(at_s) or at_s < self.now - self.eps:
            raise RuntimeError(f'Неисправно заказивање {kind}: {at_s} при сату {self.now}.')
        at_s = max(at_s, self.now)
        self._sequence += 1
        heapq.heappush(self.queue, _Event(at_s, priority, self._sequence, kind, ident, token))

    def _wake(self, at_s: float) -> None:
        if at_s not in self._wake_times:
            self._wake_times.add(at_s)
            self._schedule(at_s, 9, 'dispatch')

    def _record(self, kind: str, aircraft: Aircraft | None = None, **details) -> None:
        if self.record_events:
            row = dict(time_s=self.now, kind=kind, **details)
            if aircraft is not None:
                row.update(aircraft_id=aircraft.id, location=aircraft.location,
                           energy_kwh=aircraft.energy_kwh, soc=aircraft.soc,
                           completed_legs_since_cleaning=aircraft.completed_legs_since_cleaning)
            self.log.append(row)

    def _sync_energy(self, at_s: float) -> None:
        for rt in self.runtime.values():
            a, charge = rt.aircraft, rt.charge
            if charge is not None:
                seconds = max(0.0, min(at_s, charge.end_s) - charge.start_s)
                energy = self.dispatcher.charging[charge.vertiport].energy_after(
                    charge.initial_energy_kwh, seconds, target_soc=charge.target_soc)
                delta = energy - a.energy_kwh
                if delta < -1e-7:
                    raise RuntimeError('Пуњење је смањило енергију.')
                self.energy_added_kwh += max(0.0, delta)
                a.set_energy(energy)
                a.charging_since_s = charge.start_s if charge.start_s <= at_s < charge.end_s else None
            a.energy_updated_at_s = at_s

    def _install_charge(self, rt: _Runtime, reservation: Reservation, target_soc: float) -> None:
        rt.charge = _Charge(reservation.id, reservation.start_s, reservation.end_s,
                            rt.aircraft.energy_kwh, rt.aircraft.location, target_soc)
        rt.aircraft.charger_id = reservation.resource_id
        if reservation.start_s <= self.now:
            rt.aircraft.charging_since_s = self.now
            self._record('charging_start', rt.aircraft, reservation_id=reservation.id)
        else:
            self._schedule(reservation.start_s, 3, 'charging_start', rt.aircraft.id, reservation.id)
        self._schedule(reservation.end_s, 2, 'charging_end', rt.aircraft.id, reservation.id)

    def _start_idle_charging(self, rt: _Runtime) -> None:
        if rt.charge is not None or rt.aircraft.location is None:
            return
        model = self.dispatcher.charging[rt.aircraft.location]
        duration = model.time_to_energy(rt.aircraft.energy_kwh, model.idle_target_soc * model.capacity_kwh)
        if not model.enabled or duration <= 1e-8 or math.isinf(duration):
            return
        booking = self.resources.reserve_service(rt.visit_id, 'charging', self.now, duration)
        self._install_charge(rt, booking, model.idle_target_soc)

    @staticmethod
    def _end_charge_booking(resources: NetworkResources, charge: _Charge | None, at_s: float) -> None:
        if charge is not None and charge.end_s > at_s:
            if charge.start_s >= at_s:
                resources.cancel_service(charge.reservation_id)
            else:
                resources.truncate_service(charge.reservation_id, at_s)

    def _ensure_initial_cleaning(self, rt: _Runtime) -> None:
        rule = self.dispatcher.cleaning
        if rule['enabled'] and rt.aircraft.completed_legs_since_cleaning >= rule['every_completed_legs']:
            booking = self.resources.reserve_service(rt.visit_id, 'cleaning', self.now, rule['duration_min'] * 60)
            rt.ground_cleaning = booking
            self._schedule(booking.start_s, 3, 'cleaning_start', rt.aircraft.id, booking.id)
            self._schedule(booking.end_s, 2, 'cleaning_end', rt.aircraft.id, booking.id)

    def _forecast(self, rt: _Runtime) -> AircraftAvailability:
        a = rt.aircraft
        if rt.active is not None:
            execution = rt.active
            ready = max(self.now, execution.ready_s)
            location = execution.plan.estimate.destination
            charger = self.dispatcher.charging[location]
            seconds = max(0.0, ready - execution.plan.slot.landing_end_s - charger.start_delay_s)
            energy = charger.energy_after(execution.plan.landing_energy_kwh, seconds,
                                           target_soc=charger.idle_target_soc)
            return AircraftAvailability(a.id, location, execution.reservation.destination_visit_id,
                                         ready, energy, 0 if execution.cleaning else execution.legs_after_landing,
                                         rt.pinned_group_id)
        if rt.ground_cleaning is not None:
            ready = rt.ground_cleaning.end_s
            energy = a.energy_kwh
            if rt.charge:
                charge = rt.charge
                energy = self.dispatcher.charging[a.location].energy_after(charge.initial_energy_kwh,
                    max(0, min(ready, charge.end_s) - charge.start_s), target_soc=charge.target_soc)
            return AircraftAvailability(a.id, a.location, rt.visit_id, ready, energy, 0, rt.pinned_group_id)
        return AircraftAvailability(a.id, a.location, rt.visit_id, self.now, a.energy_kwh,
                                     a.completed_legs_since_cleaning, rt.pinned_group_id)

    def _dispatch(self) -> None:
        groups = self.dispatcher.groups(self.now)
        self.waiting_groups = list(groups)
        self.pending_decisions = {}
        self.last_candidate_rows = {}
        audit_rows = []
        waiting_groups = {g.id for g in groups}
        for rt in self.runtime.values():
            rt.aircraft.next_group_id = None
            if rt.pinned_group_id not in waiting_groups:
                rt.pinned_group_id = None
        if not groups:
            return
        used, claimed = set(), set()
        for _ in range(len(self.runtime)):
            available = [self._forecast(rt) for key, rt in self.runtime.items() if key not in used]
            if not available:
                break
            decision = self.dispatcher.choose(self.now, available, self.resources,
                                               excluded_group_ids=frozenset(claimed),
                                               audit=audit_rows if self.diagnostics else None)
            if decision is None:
                break
            used.add(decision.aircraft_id)
            claimed.add(decision.group_id)
            self.pending_decisions[decision.group_id] = decision
            rt = self.runtime[decision.aircraft_id]
            first = decision.legs[0]
            action_time = (first.boarding_start_s if first.boarding_start_s is not None
                           else first.slot.takeoff_s - self.dispatcher.preparation_s)
            if rt.active is None and rt.ground_cleaning is None and action_time <= self.now + self.eps:
                self._commit_first_leg(rt, decision)
            else:
                rt.aircraft.next_group_id = decision.group_id
                for wake in (decision.aircraft_ready_s, action_time):
                    if wake > self.now + self.eps:
                        self._wake(wake)
        deadline = self.dispatcher.next_group_wakeup(self.now)
        if deadline is not None:
            self._wake(deadline)
        if self.diagnostics:
            for group in groups:
                if group.id not in claimed:
                    for aircraft_id in sorted(used):
                        rt = self.runtime[aircraft_id]
                        audit_rows.append(dict(time_s=self.now, group_id=group.id,
                            passenger_ids='|'.join(group.passenger_ids), origin=group.origin,
                            destination=group.destination, aircraft_id=aircraft_id,
                            available_at_s=None, available_location=rt.aircraft.location,
                            forecast_energy_kwh=None, status='excluded', reason='priority_other_group',
                            boarding_start_s=None, takeoff_s=None, repositioning=False,
                            energy_ready_s=None, resource_constraints='', constraint_vertiport='', selected=False))
            latest = {(r['group_id'], r['aircraft_id']): r for r in audit_rows}
            for row in latest.values():
                self.last_candidate_rows.setdefault(row['group_id'], []).append(row)
            self.diagnostics.audit(audit_rows)

    def _commit_first_leg(self, rt: _Runtime, decision: DispatchDecision) -> None:
        """Измена на копији календара, па једна замена после свих провера."""
        a, plan = rt.aircraft, decision.legs[0]
        passenger_ids = decision.passenger_ids if plan.estimate.operation.value == 'passenger' else ()
        if passenger_ids and (plan.boarding_start_s != self.now or len(decision.legs) != 1):
            raise RuntimeError('Група се може закључати само на стварном почетку B.')
        shadow = self.resources.planning_snapshot(self.now)
        self._end_charge_booking(shadow, rt.charge, self.now)
        new_charge = None
        for task in plan.ground_tasks:
            if task.kind == 'cleaning':
                raise RuntimeError('Летелица мора завршити чишћење пре прихватања овог задатка.')
            if task.kind == 'charging' and task.end_s > task.start_s:
                new_charge = shadow.reserve_service(rt.visit_id, 'charging', task.start_s,
                    task.end_s - task.start_s, latest_end_s=task.end_s)
                if abs(new_charge.start_s - task.start_s) > self.eps:
                    raise ResourceConflict('Променила се расположивост пуњача.')
        ident = f'F{len(self.flights) + 1:06d}'
        reservation = shadow.reserve_flight(ident, a.id, plan.estimate, rt.visit_id,
                                             plan.slot.takeoff_s, plan.slot.takeoff_s)
        d_end = reservation.slot.landing_end_s + self.dispatcher.handling_seconds('disembarking', len(passenger_ids))
        cleaning = None
        rule = self.dispatcher.cleaning
        count = a.completed_legs_since_cleaning + int(bool(passenger_ids) or rule['include_repositioning_legs'])
        if rule['enabled'] and count >= rule['every_completed_legs']:
            cleaning = shadow.reserve_service(reservation.destination_visit_id, 'cleaning', d_end, rule['duration_min'] * 60)
        ready = cleaning.end_s if cleaning else d_end
        flight = Flight(ident, a.id, plan.estimate, passenger_ids,
                        decision.group_id if passenger_ids else None,
                        planned_takeoff_s=reservation.slot.takeoff_s)
        execution = _Execution(flight, reservation, plan, d_end, ready, count, cleaning)
        self.flight_energy_context[ident] = ('battery_after_repositioning'
            if rt.last_leg_operation == 'repositioning' else 'battery_at_origin')
        self.resources = shadow
        if rt.charge:
            self._record('charging_interrupted', a, reservation_id=rt.charge.reservation_id)
        rt.charge = None
        a.charging_since_s = a.charger_id = None
        if new_charge:
            self._install_charge(rt, new_charge, 1.0)
        rt.active = execution
        self.flights.append(flight)
        self.executions[ident] = execution
        a.current_flight_id = ident
        a.next_group_id = None
        if passenger_ids:
            self.dispatcher.lock_for_boarding(decision, self.now)
            rt.pinned_group_id = None
            a.assigned_group_id = decision.group_id
            a.status = AircraftStatus.BOARDING
            a.activity_end_s = plan.boarding_end_s
            for pid in passenger_ids:
                self.passengers[pid].flight_id = ident
            self._record('boarding_start', a, flight_id=ident, passenger_ids=list(passenger_ids))
            self._schedule(plan.boarding_end_s, 2, 'boarding_end', ident)
        elif decision.reason == 'known_passenger_demand':
            rt.pinned_group_id = decision.group_id
            if self.diagnostics:
                self.diagnostics.reposition_for_group[decision.group_id].append(ident)
        self._record('flight_reserved', a, flight_id=ident, operation=plan.estimate.operation.value,
                     origin=plan.estimate.origin, destination=plan.estimate.destination,
                     takeoff_s=plan.slot.takeoff_s, landing_end_s=plan.slot.landing_end_s,
                     future_cleaning_end_s=cleaning.end_s if cleaning else None)
        self._schedule(plan.slot.takeoff_s - self.dispatcher.preparation_s, 3, 'preparation', ident)
        self._schedule(plan.slot.takeoff_s, 0, 'takeoff', ident)
        self._schedule(plan.slot.takeoff_end_s, 1, 'takeoff_end', ident)
        self._schedule(plan.slot.landing_start_s, 1, 'landing_start', ident)
        self._schedule(plan.slot.landing_end_s, 1, 'landing_end', ident)
        self._schedule(d_end, 2, 'disembarking_end', ident)
        if cleaning:
            self._schedule(cleaning.start_s, 3, 'cleaning_start', a.id, cleaning.id)
            self._schedule(cleaning.end_s, 2, 'cleaning_end', a.id, cleaning.id)

    def _consume(self, a: Aircraft, kwh: float) -> None:
        if a.energy_kwh + 1e-8 < kwh:
            raise RuntimeError('Негативна енергија током лета.')
        a.set_energy(max(0, a.energy_kwh - kwh))
        self.energy_used_kwh += kwh
        self.minimum_soc = min(self.minimum_soc, a.soc)

    def _complete_passenger(self, p: Passenger) -> None:
        if p.status == PassengerStatus.COMPLETED:
            raise RuntimeError('Путник је завршен два пута.')
        p.status = PassengerStatus.COMPLETED
        p.journey_end_s = self.now
        self.completed += 1
        self._record('passenger_completed', passenger_id=p.id)

    def _release_aircraft(self, rt: _Runtime) -> None:
        rt.active = None
        rt.aircraft.status = AircraftStatus.IDLE
        rt.aircraft.current_flight_id = None
        rt.aircraft.assigned_group_id = None
        rt.aircraft.activity_end_s = None

    def _handle(self, event: _Event) -> None:
        kind = event.kind
        if kind == 'dispatch':
            self._wake_times.discard(event.time_s)
            self._dispatch()
            return
        if kind == 'request':
            p = self.passengers[event.entity_id]
            self.dispatcher.admit((p,), self.now)
            self._record('passenger_request', passenger_id=p.id, origin=p.origin, destination=p.destination)
        elif kind in ('charging_start', 'charging_end'):
            rt = self.runtime[event.entity_id]
            if rt.charge is None or rt.charge.reservation_id != event.token:
                return  # стари догађај прекинутог пуњења
            if kind == 'charging_end':
                rt.charge = None
                rt.aircraft.charging_since_s = rt.aircraft.charger_id = None
            self._record(kind, rt.aircraft, reservation_id=event.token)
        elif kind in ('cleaning_start', 'cleaning_end'):
            rt = self.runtime[event.entity_id]
            booking = self.resources.reservation(event.token)
            if kind == 'cleaning_start':
                rt.aircraft.status = AircraftStatus.CLEANING
                rt.aircraft.activity_end_s = booking.end_s
            else:
                rt.aircraft.completed_legs_since_cleaning = 0
                self.cleanings_completed += 1
                rt.ground_cleaning = None
                self._release_aircraft(rt)
            self._record(kind, rt.aircraft, reservation_id=event.token)
        elif kind == 'terminal_arrival':
            p = self.passengers[event.entity_id]
            p.terminal_arrival_s = self.now
            self._complete_passenger(p)
        else:
            execution = self.executions[event.entity_id]
            flight, plan = execution.flight, execution.plan
            rt = self.runtime[flight.aircraft_id]
            a = rt.aircraft
            if kind == 'boarding_end':
                for pid in flight.passenger_ids:
                    self.passengers[pid].status = PassengerStatus.ON_BOARD
                a.status = AircraftStatus.WAITING_FOR_TLOF
            elif kind == 'preparation':
                if rt.charge:
                    self._end_charge_booking(self.resources, rt.charge, self.now)
                    rt.charge = None
                    a.charging_since_s = a.charger_id = None
                a.status = AircraftStatus.DEPARTURE_FINALIZATION
                a.activity_end_s = plan.slot.takeoff_s
            elif kind == 'takeoff':
                if a.energy_kwh + 1e-8 < plan.estimate.required_departure_energy_kwh:
                    raise RuntimeError(f'{flight.id}: недовољна енергија пре полетања.')
                if self.dispatcher.cleaning['enabled'] and a.completed_legs_since_cleaning >= self.dispatcher.cleaning['every_completed_legs']:
                    raise RuntimeError('Полетање пре обавезног чишћења.')
                self.resources.depart(flight.id, self.now)
                flight.takeoff_start_s = self.now
                flight.departure_energy_kwh = a.energy_kwh
                a.location = a.stand_id = None
                a.status = AircraftStatus.TAKEOFF
                a.activity_end_s = plan.slot.takeoff_end_s
                for pid in flight.passenger_ids:
                    self.passengers[pid].takeoff_start_s = self.now
            elif kind == 'takeoff_end':
                self._consume(a, plan.estimate.takeoff.energy_kwh)
                a.status = AircraftStatus.CRUISE
                a.activity_end_s = plan.slot.landing_start_s
            elif kind == 'landing_start':
                self._consume(a, plan.estimate.cruise.energy_kwh)
                a.status = AircraftStatus.LANDING
                a.activity_end_s = plan.slot.landing_end_s
            elif kind == 'landing_end':
                self._consume(a, plan.estimate.landing.energy_kwh)
                self.resources.arrive(flight.id, self.now)
                flight.landing_end_s = self.now
                flight.landing_energy_kwh = a.energy_kwh
                if abs(a.energy_kwh - plan.landing_energy_kwh) > 1e-6:
                    raise RuntimeError('Стварна енергија слетања не одговара прихваћеном плану.')
                a.location = plan.estimate.destination
                rt.visit_id = execution.reservation.destination_visit_id
                a.stand_id = execution.reservation.slot.destination_stand_id
                a.total_completed_legs += 1
                rt.last_leg_operation = flight.estimate.operation.value
                a.completed_legs_since_cleaning = execution.legs_after_landing
                a.status = AircraftStatus.DISEMBARKING
                a.activity_end_s = execution.disembarking_end_s
                for pid in flight.passenger_ids:
                    self.passengers[pid].landing_end_s = self.now
                    self.passengers[pid].status = PassengerStatus.DISEMBARKING
                self._start_idle_charging(rt)
            elif kind == 'disembarking_end':
                for pid in flight.passenger_ids:
                    p = self.passengers[pid]
                    p.disembarking_end_s = self.now
                    if p.destination == self.dispatcher.airport:
                        p.status = PassengerStatus.TERMINAL_TRANSFER
                        self._schedule(self.now + p.terminal_access_duration_s, 2, 'terminal_arrival', pid)
                    else:
                        self._complete_passenger(p)
                if execution.cleaning is None:
                    self._release_aircraft(rt)
                else:
                    a.status = AircraftStatus.IDLE  # чека резервисану екипу; active и даље блокира доступност
                    a.activity_end_s = execution.cleaning.end_s
            else:
                raise RuntimeError(f'Непознат догађај: {kind}.')
            self._record(kind, a, flight_id=flight.id)
        self._wake(self.now)

    def run(self) -> SimulationResult:
        if self._has_run:
            raise RuntimeError('За нову репликацију направити нову Simulation инстанцу.')
        self._has_run = True
        while self.queue and self.completed < len(self.passengers):
            if self.event_count >= self.max_events:
                raise RuntimeError('Достигнута заштитна граница догађаја; могућа петља распоређивања.')
            event = heapq.heappop(self.queue)
            if self.diagnostics:
                self.diagnostics.observe(self, self.now, event.time_s)
            self._sync_energy(event.time_s)
            self.now = event.time_s
            self._handle(event)
            self.event_count += 1
        status = 'completed' if self.completed == len(self.passengers) else 'infeasible_deadlock'
        self._sync_energy(self.now)
        ending_energy = sum(rt.aircraft.energy_kwh for rt in self.runtime.values())
        residual = self.initial_energy_kwh + self.energy_added_kwh - self.energy_used_kwh - ending_energy
        if abs(residual) > 1e-6:
            raise RuntimeError(f'Нарушен биланс енергије: {residual} kWh.')
        result = self._result(status)
        result.summary['energy_balance_residual_kwh'] = residual
        from .results import enrich_summary
        enrich_summary(result, self.data)
        return result

    def _result(self, status: str) -> SimulationResult:
        people = list(self.passengers.values())
        waits = [p.waiting_time_s for p in people if p.boarding_start_s is not None]
        on_time = sum(w <= self.dispatcher.wait_limit_s + self.eps for w in waits)
        completed_flights = [f for f in self.flights if f.landing_end_s is not None]
        departures = [p.takeoff_start_s - p.request_time_s for p in people if p.takeoff_start_s is not None]
        lateness = [max(0, p.terminal_arrival_s - p.latest_terminal_arrival_s) for p in people
                    if p.terminal_arrival_s is not None and p.latest_terminal_arrival_s is not None]
        summary = dict(
            scenario=self.demand.scenario, fleet_size=self.configuration['fleet_size'],
            replication_index=self.demand.replication_index, status=status,
            generated_passengers=len(people), completed_passenger_journeys=self.completed,
            unfinished_passengers=len(people)-self.completed, boarded_within_limit=on_time,
            fraction_boarding_within_wait_limit=(on_time/len(people) if people and status == 'completed' else None),
            mean_wait_to_boarding_min=float(np.mean(waits))/60 if waits else None,
            p95_wait_to_boarding_min=float(np.quantile(waits, .95))/60 if waits else None,
            max_wait_to_boarding_min=max(waits)/60 if waits else None,
            mean_request_to_takeoff_min=float(np.mean(departures))/60 if departures else None,
            mean_terminal_target_lateness_min=float(np.mean(lateness))/60 if lateness else None,
            passenger_flights=sum(bool(f.passenger_ids) for f in completed_flights),
            repositioning_flights=sum(not f.passenger_ids for f in completed_flights),
            flight_energy_kwh=self.energy_used_kwh, battery_energy_added_kwh=self.energy_added_kwh,
            minimum_soc=self.minimum_soc, cleanings_completed=self.cleanings_completed,
            start_time_s=self.start_s, end_time_s=self.now,
            start_datetime=self.demand.local_datetime(self.start_s).isoformat(),
            end_datetime=self.demand.local_datetime(self.now).isoformat(), event_count=self.event_count,
        )
        d = self.diagnostics
        summary['diagnostics_enabled'] = d is not None
        return SimulationResult(summary, people, list(self.flights), self.log, self.demand,
            d.waits if d else [], d.candidates if d else [], d.activities if d else [],
            d.queues if d else [], resource_utilization(self.resources, self.dispatcher.charging, self.start_s, self.now),
            {ident: dict(boarding_end_s=ex.plan.boarding_end_s,
                         energy_ready_s=ex.plan.energy_ready_s,
                         required_departure_energy_kwh=ex.plan.estimate.required_departure_energy_kwh,
                         energy_context=self.flight_energy_context[ident]) for ident, ex in self.executions.items()},
            d.compact_metrics(self) if d else {})


def run_replication(data: InputData, configuration: dict, replication_index: int = 0,
                    *, demand: DemandRealization | None = None, record_events: bool | None = None,
                    detailed: bool = False) -> SimulationResult:
    if demand is None:
        demand = generate_demand(data, scenario=configuration['demand_scenario'], replication_index=replication_index)
    if demand.replication_index != replication_index:
        raise ValueError('Неусаглашен индекс реализације тражње.')
    if record_events is None:
        record_events = detailed and replication_index in data.config['outputs']['event_log_replications']
    return Simulation(data, configuration, demand, record_events=record_events, retain_details=detailed).run()


def save_replication(result: SimulationResult, directory: Path, data: InputData) -> None:
    """Сачувај сирове податке и дијагностику сваке репликације."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    metadata = dict(summary=result.summary, numpy_version=result.demand.numpy_version,
                    seed_components=dict(result.demand.seed_components), input_sha256=data.input_sha256)
    (directory/'summary.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    from .results import export_replication
    export_replication(result, directory, data)
    if result.event_log:
        with (directory/'events.jsonl').open('w', encoding='utf-8') as handle:
            for row in result.event_log:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')


def main(argv: list[str] | None = None) -> int:
    from .input_data import InputDataError, load_inputs

    parser = argparse.ArgumentParser(description='Једна цела репликација eVTOL симулације.')
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1]/'config.json')
    parser.add_argument('--scenario', default='low')
    parser.add_argument('--fleet-size', type=int, default=4)
    parser.add_argument('--replication', type=int, default=0)
    parser.add_argument('--output', type=Path, help='Нов директоријум за сирове резултате.')
    args = parser.parse_args(argv)
    try:
        data = load_inputs(args.config, overrides={'mode':'single_configuration', 'demand_scenario':args.scenario,
                                                   'fleet_size':args.fleet_size, 'replications':1})
        result = run_replication(data, data.configurations[0], args.replication, detailed=args.output is not None)
        print(json.dumps(result.summary, ensure_ascii=False, indent=2, allow_nan=False))
        if args.output:
            save_replication(result, args.output, data)
        return 0 if result.summary['status'] == 'completed' else 3
    except (InputDataError, ValueError, RuntimeError, OSError) as exc:
        print(f'ГРЕШКА: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    raise SystemExit(main())
