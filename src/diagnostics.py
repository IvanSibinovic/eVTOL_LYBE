from __future__ import annotations

from collections import Counter, defaultdict
import math

CAUSE_LABELS = {
    'battery_before_repositioning': 'Батерија пре позиционог лета',
    'battery_after_repositioning': 'Батерија после позиционог лета',
    'battery_at_origin': 'Батерија летелице на полазишту',
    'aircraft_busy': 'Летелица завршава претходни задатак',
    'repositioning_flight': 'Долазак летелице позиционим летом',
    'cleaning': 'Чишћење', 'cleaning_crew': 'Чекање екипе за чишћење',
    'destination_stand': 'Одредишна паркинг позиција',
    'origin_tlof': 'Полазни TLOF/FATO', 'destination_tlof': 'Одредишни TLOF/FATO',
    'group_formation': 'Правило формирања групе',
    'priority_other_group': 'Преддодела другој групи по редоследу диспечера',
    'reserved_for_other_group': 'Летелица већ везана за другу групу',
    'pinned_aircraft_policy': 'Правило задржавања додељене летелице',
    'departure_preparation': 'Обавезна завршна припрема',
    'energy_unreachable': 'Потребна енергија није достижна',
    'resource_conflict': 'Неизводљива резервација ресурса',
    'unexplained_wait': 'Чекање без потпуног дијагностичког објашњења',
}


def merge_interval(rows, last, key, start, end, **fields):
    if end <= start + 1e-9:
        return
    old = last.get(key)
    if old is not None and old['end_s'] == start and all(old.get(k) == v for k, v in fields.items()):
        old['end_s'] = end
    else:
        row = dict(start_s=start, end_s=end, **fields)
        rows.append(row)
        last[key] = row


class Diagnostics:
    def __init__(self, retain_details=True):
        self.retain_details = retain_details
        self.waits, self.activities, self.queues, self.candidates = [], [], [], []
        self._last_wait, self._last_activity, self._last_queue = {}, {}, {}
        self.preposition_history = defaultdict(list)
        self.reposition_for_group = defaultdict(list)
        self.snapshot_id = 0
        self.totals = defaultdict(float)
        self.cause_seconds = defaultdict(float)
        self.cause_exposed = defaultdict(set)
        self.activity_seconds = defaultdict(float)
        self.port_queues = defaultdict(lambda: dict(peak=0, area_s=0., nonempty_s=0.))

    def _remember_preposition(self, pid, start, end, aircraft_id, codes):
        rows = self.preposition_history[pid]
        codes = tuple(codes)
        if rows and rows[-1][1] == start and rows[-1][2:] == (aircraft_id, codes):
            a, _, aid, old_codes = rows[-1]
            rows[-1] = (a, end, aid, old_codes)
        else:
            rows.append((start, end, aircraft_id, codes))

    def _preposition_metrics(self, sim):
        codes = tuple(
            c for c in CAUSE_LABELS
            if c not in ('repositioning_flight', 'group_formation')
        )
        codes += ('assignment_unresolved', 'unobserved')
        seconds = defaultdict(float)
        exposed = defaultdict(set)
        total = overlap = before_request = 0.0
        passengers = set()

        for p in sim.passengers.values():
            if p.waiting_time_s is None:
                continue
            if p.waiting_time_s <= sim.dispatcher.wait_limit_s + sim.eps:
                continue

            boarded = sim.executions.get(p.flight_id)
            if boarded is None:
                continue

            candidates = []
            for fid in self.reposition_for_group.get(p.group_id, ()):
                ex = sim.executions[fid]
                f = ex.flight
                if (
                    f.aircraft_id == boarded.flight.aircraft_id
                    and f.landing_end_s is not None
                    and f.landing_end_s <= p.boarding_start_s + sim.eps
                ):
                    candidates.append(ex)

            if not candidates:
                continue

            ex = max(candidates, key=lambda x: x.flight.landing_end_s)
            f = ex.flight
            takeoff = f.takeoff_start_s
            if takeoff is None:
                continue

            # Неопходно опслуживање после празног лета.
            # Чекање слободне екипе није део минималног трајања.
            ground = ex.disembarking_end_s - ex.plan.slot.landing_end_s
            if ex.cleaning:
                ground += ex.cleaning.end_s - ex.cleaning.start_s

            deadline = p.request_time_s + sim.dispatcher.wait_limit_s
            latest = deadline - (f.landing_end_s - takeoff) - ground

            if takeoff <= latest + sim.eps:
                continue

            passengers.add(p.id)
            total += takeoff - latest

            # Ако би полазак морао претходити захтеву,
            # тај део не приписујемо посматраним ограничењима.
            before_request += max(
                0.0, min(takeoff, p.request_time_s) - latest
            )
            left = max(latest, p.request_time_s)
            covered = 0.0

            for a, b, aid, observed in self.preposition_history.get(p.id, ()):
                dt = min(b, takeoff) - max(a, left)
                if dt <= 0:
                    continue

                covered += dt

                if aid != f.aircraft_id:
                    # Ранија одлука није била о стварно коришћеној
                    # летелици: не преносимо њена ограничења.
                    active = {'assignment_unresolved'}
                else:
                    active = {
                        'aircraft_busy' if c == 'repositioning_flight' else c
                        for c in observed if c != 'group_formation'
                    }
                    active = {
                        c if c in codes else 'unobserved' for c in active
                    }
                    active = active or {'unobserved'}

                if len(active) > 1:
                    overlap += dt

                for c in active:
                    seconds[c] += dt
                    exposed[c].add(p.id)

            missing = max(0.0, takeoff - left - covered)
            if missing > sim.eps:
                seconds['unobserved'] += missing
                exposed['unobserved'].add(p.id)

        result = {
            'preposition_late_departure_passengers': len(passengers),
            'preposition_late_departure_passenger_min': total / 60,
            'preposition_before_request_passenger_min': before_request / 60,
            'preposition_overlap_passenger_min': overlap / 60,
        }

        for c in codes:
            result[f'preposition_{c}_passenger_min'] = seconds[c] / 60
            result[f'preposition_{c}_passengers'] = len(exposed[c])

        return result

    def audit(self, rows):
        self.snapshot_id += 1
        if not self.retain_details:
            return
        for row in rows:
            self.candidates.append(dict(snapshot_id=self.snapshot_id, **row))

    @staticmethod
    def _cleaning(windows, rt, now):
        booking = rt.ground_cleaning or (rt.active.cleaning if rt.active else None)
        if booking:
            available = rt.active.disembarking_end_s if rt.active else now
            windows.extend([(available, booking.start_s, 'cleaning_crew', booking.resource_id),
                            (booking.start_s, booking.end_s, 'cleaning', booking.resource_id)])

    @staticmethod
    def _leg_windows(windows, plan, now, prep, battery_code, energy=True):
        if energy and plan.charging_required and plan.energy_ready_s is not None:
            windows.append((now, plan.energy_ready_s, battery_code, plan.estimate.origin))
        windows.extend((t.start_s, t.end_s, t.kind, t.resource_id) for t in plan.resource_delays)
        windows.append((plan.slot.takeoff_s-prep, plan.slot.takeoff_s,
                        'departure_preparation', plan.estimate.origin))

    def _waiting_windows(self, sim, group, decision, start):
        windows = [(start, sim.dispatcher.trigger_time(group), 'group_formation', group.origin)]
        scope, aircraft_id, detail = 'selected_plan', '', ''
        if decision and decision.reason == 'parking_capacity_relief':
            windows.append((start, decision.legs[0].slot.takeoff_s, 'destination_stand', group.destination))
            return windows, 'capacity_relief', decision.aircraft_id, 'parking_capacity_relief'
        # A committed repositioning remains attached to its group even when a
        # new tentative decision cannot yet be made (e.g. full destination).
        rt = sim.runtime.get(decision.aircraft_id) if decision else next(
            (r for r in sim.runtime.values() if r.pinned_group_id == group.id), None)
        if rt is None:
            scope = 'candidate_set'
            rows = sim.last_candidate_rows.get(group.id, [])
            causes = {(r['reason'] or 'unexplained_wait', r.get('constraint_vertiport', ''))
                      for r in rows if r['status'] != 'feasible'}
            for code, port in causes:
                windows.append((start, math.inf, code, port))
            if not causes:
                windows.append((start, math.inf, 'unexplained_wait', group.origin))
            return windows, scope, aircraft_id, detail
        aircraft_id = rt.aircraft.id
        detail = rt.aircraft.status.value
        self._cleaning(windows, rt, start)
        execution = rt.active
        if execution:
            if rt.pinned_group_id == group.id and not execution.flight.passenger_ids:
                self._leg_windows(windows, execution.plan, start, sim.dispatcher.preparation_s,
                                  'battery_before_repositioning')
                windows.append((execution.plan.slot.takeoff_s, execution.plan.slot.landing_end_s,
                                'repositioning_flight', execution.plan.estimate.destination))
            else:
                windows.append((start, execution.disembarking_end_s, 'aircraft_busy', execution.flight.id))
        elif decision:
            plan = decision.legs[0]
            if plan.estimate.operation.value == 'repositioning':
                self._leg_windows(windows, plan, start, sim.dispatcher.preparation_s,
                                  'battery_before_repositioning')
                # Ground-cleaning forecasts already include charging up to ready_s.
                # Observe remaining energy at the current time as well, to retain
                # overlap with the cleaning which makes this aircraft unavailable.
                charge = rt.charge
                required = plan.estimate.required_departure_energy_kwh
                if charge and rt.aircraft.energy_kwh < required - 1e-8:
                    model = sim.dispatcher.charging[charge.vertiport]
                    ready = charge.start_s + model.time_to_energy(charge.initial_energy_kwh, required)
                    windows.append((start, ready, 'battery_before_repositioning', charge.vertiport))
            for task in plan.ground_tasks:
                if task.kind == 'cleaning':
                    windows.append((task.start_s, task.end_s, 'cleaning', task.resource_id))
        return windows, scope, aircraft_id, detail

    def _passenger_intervals(self, sim, p, start, end, windows, stage, scope, aircraft_id, detail):
        boundaries = {start, end}
        for a, b, _, _ in windows:
            if start < a < end: boundaries.add(a)
            if start < b < end: boundaries.add(b)
        # Splitting at the passenger deadline makes overdue accounting exact.
        deadline = p.request_time_s + sim.dispatcher.wait_limit_s
        if start < deadline < end: boundaries.add(deadline)
        times = sorted(boundaries)
        for a, b in zip(times, times[1:]):
            mid = (a+b)/2
            active = {(code, resource) for x, y, code, resource in windows if x <= mid < y}
            codes = sorted({c for c, _ in active}) or ['unexplained_wait']
            if stage == 'before_boarding':
                self._remember_preposition(p.id, a, b, aircraft_id, codes)
            resources = '|'.join(sorted({f'{c}@{r}' for c, r in active if r}))
            actual_stage = ('departure_preparation' if stage == 'after_boarding'
                            and codes == ['departure_preparation'] else stage)
            seconds = b-a
            overdue = max(0., b-max(a,deadline)) if actual_stage == 'before_boarding' else 0.
            self.totals[actual_stage] += seconds
            self.totals['overdue'] += overdue
            if len(codes)>1:
                self.totals['overlap_overdue'] += overdue
                if actual_stage == 'after_boarding': self.totals['overlap_after_boarding'] += seconds
            if 'unexplained_wait' in codes: self.totals['unexplained'] += seconds
            for code in codes:
                if actual_stage == 'before_boarding':
                    self.cause_exposed[code].add(p.id)
                    self.cause_seconds[code,'overdue'] += overdue
                elif actual_stage == 'after_boarding':
                    self.cause_seconds[code,'after_boarding'] += seconds
            if not self.retain_details:
                continue
            merge_interval(self.waits, self._last_wait, (p.id, actual_stage), a, b,
                           passenger_id=p.id, group_id=p.group_id, aircraft_id=aircraft_id,
                           origin=p.origin, destination=p.destination, stage=actual_stage,
                           scope=scope, causes='|'.join(codes), constraint_locations=resources,
                           aircraft_activity=detail)

    def observe(self, sim, start, end):
        if end <= start:
            return
        waiting = []
        for group in sim.waiting_groups:
            decision = sim.pending_decisions.get(group.id)
            windows, scope, aircraft_id, detail = self._waiting_windows(sim, group, decision, start)
            for pid in group.passenger_ids:
                p = sim.passengers[pid]
                if p.boarding_start_s is None and p.request_time_s <= start:
                    waiting.append(p)
                    self._passenger_intervals(sim, p, start, end, windows, 'before_boarding',
                                              scope, aircraft_id, detail)
        counts = Counter(p.origin for p in waiting)
        for port in sim.dispatcher.charging:
            queue = self.port_queues[port]
            queue['peak'] = max(queue['peak'],counts[port])
            queue['area_s'] += counts[port]*(end-start)
            queue['nonempty_s'] += (end-start) if counts[port] else 0.
            if self.retain_details:
                merge_interval(self.queues, self._last_queue, port, start, end,
                               vertiport=port, waiting_passengers=counts[port])
        for rt in sim.runtime.values():
            a, ex = rt.aircraft, rt.active
            activity = a.status.value
            if activity == 'waiting_for_tlof':
                activity = 'waiting_for_departure'  # May still be waiting for energy.
            cleaning = rt.ground_cleaning or (ex.cleaning if ex else None)
            if cleaning and start < cleaning.start_s and (not ex or start >= ex.disembarking_end_s):
                activity = 'waiting_for_cleaning_crew'
            charge = rt.charge
            bounds = sorted({start, end, *([charge.start_s] if charge and start < charge.start_s < end else []),
                             *([charge.end_s] if charge and start < charge.end_s < end else [])})
            for left, right in zip(bounds, bounds[1:]):
                charging = bool(charge and charge.start_s <= left < charge.end_s)
                operation = ex.flight.estimate.operation.value if ex else ''
                self.activity_seconds[activity,operation] += right-left
                if charging: self.totals['charging_overlay'] += right-left
                if self.retain_details:
                    merge_interval(self.activities, self._last_activity, a.id, left, right,
                                   aircraft_id=a.id, activity=activity, charging=charging,
                                   location=a.location or '', flight_id=a.current_flight_id or '',
                                   operation=operation)
            if ex and ex.flight.passenger_ids and ex.plan.boarding_end_s <= start < ex.plan.slot.takeoff_s:
                windows = []
                self._leg_windows(windows, ex.plan, start, sim.dispatcher.preparation_s,
                                  sim.flight_energy_context[ex.flight.id])
                for pid in ex.flight.passenger_ids:
                    self._passenger_intervals(sim, sim.passengers[pid], start,
                                              min(end, ex.plan.slot.takeoff_s), windows,
                                              'after_boarding', 'committed_flight', a.id, activity)

    def compact_metrics(self, sim):
        late = {p.id for p in sim.passengers.values() if p.waiting_time_s is not None
                and p.waiting_time_s > sim.dispatcher.wait_limit_s+sim.eps}
        result = dict(diagnostic_before_boarding_min=self.totals['before_boarding']/60,
                      diagnostic_after_boarding_min=self.totals['after_boarding']/60,
                      diagnostic_overdue_min=self.totals['overdue']/60,
                      overlapping_overdue_passenger_min=self.totals['overlap_overdue']/60,
                      overlapping_after_boarding_passenger_min=self.totals['overlap_after_boarding']/60,
                      unexplained_wait_passenger_min=self.totals['unexplained']/60,
                      charging_aircraft_hours=self.totals['charging_overlay']/3600,
                      queue_peak=max((q['peak'] for q in self.port_queues.values()),default=0),
                      queue_passenger_minutes=sum(q['area_s'] for q in self.port_queues.values())/60)
        for code in CAUSE_LABELS:
            result[f'cause_{code}_late_passengers'] = len(self.cause_exposed[code] & late)
            result[f'cause_{code}_overdue_passenger_min'] = self.cause_seconds[code,'overdue']/60
            result[f'cause_{code}_after_boarding_passenger_min'] = self.cause_seconds[code,'after_boarding']/60
        activities = ('idle','boarding','waiting_for_departure','departure_finalization',
                      'takeoff','cruise','landing','disembarking','cleaning','waiting_for_cleaning_crew')
        for activity in activities:
            result[f'fleet_{activity}_hours'] = sum(v for (a,_),v in self.activity_seconds.items() if a==activity)/3600
        for operation in ('passenger','repositioning'):
            result[f'fleet_{operation}_flight_hours'] = sum(v for (a,o),v in self.activity_seconds.items()
                if o==operation and a in ('takeoff','cruise','landing'))/3600
        horizon = sim.now-sim.start_s
        for port in sim.dispatcher.charging:
            q = self.port_queues[port]
            result[f'{port}_queue_peak'] = q['peak']
            result[f'{port}_queue_nonempty_min'] = q['nonempty_s']/60
            result[f'{port}_queue_mean'] = q['area_s']/horizon if horizon else None
        result['fleet_idle_fraction'] = (result['fleet_idle_hours']*3600/(horizon*len(sim.runtime))
                                        if horizon else None)
        result.update(self._preposition_metrics(sim))
        return result


def resource_utilization(resources, ports, start, end):
    """Clip reservations to the actual replication window; include zero use."""
    rows = []
    horizon = end-start
    for port in ports:
        for kind in ('stand', 'movement', 'charger', 'cleaner'):
            units = resources.resource_ids(port, kind)
            changes = defaultdict(int)
            used = 0.0
            for unit in units:
                for r in resources.calendar(unit).reservations:
                    a, b = max(start, r.start_s), min(end, r.end_s)
                    if b > a:
                        used += b-a
                        changes[a] += 1
                        changes[b] -= 1
            occupied, previous, full, peak = 0, start, 0.0, 0
            for t, delta in sorted(changes.items()):
                if occupied == len(units): full += t-previous
                occupied += delta
                peak = max(peak, occupied)
                previous = t
            if occupied == len(units): full += end-previous
            rows.append(dict(vertiport=port, resource_kind=kind, capacity=len(units),
                             busy_unit_minutes=used/60, observed_minutes=horizon/60,
                             utilization=used/(len(units)*horizon) if horizon > 0 and units else None,
                             fully_occupied_minutes=full/60, peak_occupied_units=peak))
    return rows
