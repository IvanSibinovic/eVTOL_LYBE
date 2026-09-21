from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, IntEnum


class PassengerStatus(str, Enum):
    NOT_ARRIVED = 'not_arrived'
    WAITING = 'waiting'
    BOARDING = 'boarding'
    ON_BOARD = 'on_board'
    DISEMBARKING = 'disembarking'
    TERMINAL_TRANSFER = 'terminal_transfer'
    COMPLETED = 'completed'


class AircraftStatus(str, Enum):
    IDLE = 'idle'
    BOARDING = 'boarding'
    DEPARTURE_FINALIZATION = 'departure_finalization'
    WAITING_FOR_TLOF = 'waiting_for_tlof'
    TAKEOFF = 'takeoff'
    CRUISE = 'cruise'
    LANDING = 'landing'
    DISEMBARKING = 'disembarking'
    CLEANING = 'cleaning'


class FlightOperation(str, Enum):
    PASSENGER = 'passenger'
    REPOSITIONING = 'repositioning'


def _finite(value: float, label: str, minimum: float | None = None) -> None:
    if (type(value) not in (int, float) or not math.isfinite(value)
            or (minimum is not None and value < minimum)):
        raise ValueError(f'{label}: неисправна бројчана вредност {value!r}.')


@dataclass(slots=True)
class Passenger:
    id: str
    airline_flight_id: str
    origin: str
    destination: str
    request_time_s: float
    latest_desired_takeoff_s: float | None = None
    latest_terminal_arrival_s: float | None = None
    terminal_access_duration_s: float = 0.0
    status: PassengerStatus = PassengerStatus.NOT_ARRIVED
    group_id: str | None = None
    flight_id: str | None = None
    boarding_start_s: float | None = None
    takeoff_start_s: float | None = None
    landing_end_s: float | None = None
    disembarking_end_s: float | None = None
    terminal_arrival_s: float | None = None
    journey_end_s: float | None = None

    def __post_init__(self) -> None:
        if self.origin == self.destination:
            raise ValueError('Полазиште и одредиште путника морају бити различити.')
        _finite(self.request_time_s, 'request_time_s')
        _finite(self.terminal_access_duration_s, 'terminal_access_duration_s', 0)
        for value in (self.latest_desired_takeoff_s, self.latest_terminal_arrival_s):
            if value is not None:
                _finite(value, 'passenger deadline')

    @property
    def waiting_time_s(self) -> float | None:
        """Чекање до почетка укрцавања, без самог укрцавања."""
        if self.boarding_start_s is None:
            return None
        wait = self.boarding_start_s - self.request_time_s
        if wait < 0:
            raise ValueError('Укрцавање не може почети пре појаве путника.')
        return wait


@dataclass(slots=True)
class PassengerGroup:
    id: str
    origin: str
    destination: str
    capacity: int
    passenger_ids: tuple[str, ...] = ()
    oldest_request_time_s: float | None = None
    latest_request_time_s: float | None = None
    latest_desired_takeoff_s: float | None = None
    assigned_aircraft_id: str | None = None
    boarding_start_s: float | None = None

    def __post_init__(self) -> None:
        if type(self.capacity) is not int or self.capacity < 1:
            raise ValueError('Капацитет групе мора бити позитиван цео број.')
        if self.origin == self.destination:
            raise ValueError('Полазиште и одредиште групе морају бити различити.')
        self.passenger_ids = tuple(self.passenger_ids)
        if (len(self.passenger_ids) > self.capacity
                or len(set(self.passenger_ids)) != len(self.passenger_ids)):
            raise ValueError('Превише путника или поновљени путник у групи.')
        if self.passenger_ids and self.oldest_request_time_s is None:
            raise ValueError('За непразну групу потребно је време најстаријег захтева.')
        if self.passenger_ids and self.latest_request_time_s is None:
            raise ValueError('За непразну групу потребно је време последњег захтева.')
        for time in (self.oldest_request_time_s, self.latest_request_time_s,
                     self.latest_desired_takeoff_s):
            if time is not None:
                _finite(time, 'group time')
        if (self.oldest_request_time_s is not None and self.latest_request_time_s is not None
                and self.latest_request_time_s < self.oldest_request_time_s):
            raise ValueError('Неисправан редослед времена захтева групе.')

    @property
    def is_locked(self) -> bool:
        return self.boarding_start_s is not None

    def add_passenger(self, passenger: Passenger) -> None:
        """Додавање пре почетка B; додела летелице још не закључава групу."""
        if self.is_locked:
            raise ValueError('Група је закључана почетком укрцавања.')
        if (passenger.origin, passenger.destination) != (self.origin, self.destination):
            raise ValueError('Путник има другу руту.')
        if passenger.group_id not in (None, self.id):
            raise ValueError('Путник је већ у другој групи.')
        if len(self.passenger_ids) >= self.capacity or passenger.id in self.passenger_ids:
            raise ValueError('Група је пуна или већ садржи овог путника.')
        self.passenger_ids += (passenger.id,)
        passenger.group_id = self.id
        self.oldest_request_time_s = (passenger.request_time_s
            if self.oldest_request_time_s is None
            else min(self.oldest_request_time_s, passenger.request_time_s))
        self.latest_request_time_s = (passenger.request_time_s
            if self.latest_request_time_s is None
            else max(self.latest_request_time_s, passenger.request_time_s))
        if passenger.latest_desired_takeoff_s is not None:
            self.latest_desired_takeoff_s = (passenger.latest_desired_takeoff_s
                if self.latest_desired_takeoff_s is None
                else min(self.latest_desired_takeoff_s, passenger.latest_desired_takeoff_s))

    def lock(self, boarding_start_s: float) -> None:
        """Позвати на стварном почетку B, после провере свих путника."""
        _finite(boarding_start_s, 'boarding_start_s')
        if self.is_locked or not self.passenger_ids:
            raise ValueError('Празна или већ закључана група.')
        if self.latest_request_time_s is None or boarding_start_s < self.latest_request_time_s:
            raise ValueError('Укрцавање претходи појави неког путника у групи.')
        self.boarding_start_s = boarding_start_s


@dataclass(frozen=True, slots=True)
class AircraftSpec:
    """Једна заједничка, непроменљива спецификација за хомогену флоту."""
    id: str
    passenger_seats: int
    battery_capacity_kwh: float

    def __post_init__(self) -> None:
        if type(self.passenger_seats) is not int or self.passenger_seats < 1:
            raise ValueError('Број путничких места мора бити позитиван цео број.')
        _finite(self.battery_capacity_kwh, 'battery_capacity_kwh', 0)
        if self.battery_capacity_kwh == 0:
            raise ValueError('Капацитет батерије мора бити већи од нуле.')


@dataclass(slots=True)
class Aircraft:
    id: str
    spec: AircraftSpec
    location: str | None
    energy_kwh: float
    status: AircraftStatus = AircraftStatus.IDLE
    activity_end_s: float | None = None
    current_flight_id: str | None = None
    assigned_group_id: str | None = None
    next_group_id: str | None = None
    completed_legs_since_cleaning: int = 0
    total_completed_legs: int = 0
    # Пуњење је независно од активности: може да се преклапа са B/D/чишћењем.
    charging_since_s: float | None = None
    energy_updated_at_s: float = 0.0
    stand_id: str | None = None
    charger_id: str | None = None

    def __post_init__(self) -> None:
        self.set_energy(self.energy_kwh)
        for count in (self.completed_legs_since_cleaning, self.total_completed_legs):
            if type(count) is not int or count < 0:
                raise ValueError('Број завршених етапа мора бити ненегативан цео број.')

    @property
    def soc(self) -> float:
        return self.energy_kwh / self.spec.battery_capacity_kwh

    @property
    def is_charging(self) -> bool:
        return self.charging_since_s is not None

    def set_energy(self, energy_kwh: float) -> None:
        _finite(energy_kwh, 'energy_kwh', 0)
        if energy_kwh > self.spec.battery_capacity_kwh:
            raise ValueError('Енергија премашује капацитет батерије.')
        self.energy_kwh = energy_kwh


@dataclass(frozen=True, slots=True)
class FlightPhase:
    name: str
    duration_s: float
    energy_kwh: float


@dataclass(frozen=True, slots=True)
class FlightEstimate:
    route_id: str
    origin: str
    destination: str
    operation: FlightOperation
    total_distance_km: float
    cruise_distance_km: float
    node_path: tuple[str, ...]
    edge_path: tuple[str, ...]
    takeoff: FlightPhase
    cruise: FlightPhase
    landing: FlightPhase
    required_departure_energy_kwh: float

    @property
    def duration_s(self) -> float:
        return self.takeoff.duration_s + self.cruise.duration_s + self.landing.duration_s

    @property
    def energy_kwh(self) -> float:
        return self.takeoff.energy_kwh + self.cruise.energy_kwh + self.landing.energy_kwh


@dataclass(slots=True)
class Flight:
    """Једна етапа; estimate је план, остала поља бележе извршење.

    Енергија estimate обухвата само лет. На слетању се одузима номинална
    потрошња, не цео услов за полетање. Завршетак лета није крај путовања.
    """
    id: str
    aircraft_id: str
    estimate: FlightEstimate
    passenger_ids: tuple[str, ...] = ()
    group_id: str | None = None
    planned_takeoff_s: float | None = None
    takeoff_start_s: float | None = None
    landing_end_s: float | None = None
    departure_energy_kwh: float | None = None
    landing_energy_kwh: float | None = None

    def __post_init__(self) -> None:
        self.passenger_ids = tuple(self.passenger_ids)
        if len(set(self.passenger_ids)) != len(self.passenger_ids):
            raise ValueError('Поновљен путник на лету.')
        if self.estimate.operation == FlightOperation.REPOSITIONING and self.passenger_ids:
            raise ValueError('Позициони лет не може имати путнике.')
        if self.estimate.operation == FlightOperation.PASSENGER and not self.passenger_ids:
            raise ValueError('Путнички лет мора имати бар једног путника.')


class EventPriority(IntEnum):
    """Редослед истовремених догађаја из config.simulation."""
    COMPLETION = 0
    PASSENGER_REQUEST = 1
    GROUP_DEADLINE = 2
    DISPATCH = 3


@dataclass(order=True, frozen=True, slots=True)
class Event:
    """За heapq: време, приоритет, јединствени редни број у репликацији."""
    time_s: float
    priority: EventPriority
    sequence: int
    kind: str = field(compare=False)
    entity_id: str | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        _finite(self.time_s, 'event.time_s')
        EventPriority(self.priority)
        if type(self.sequence) is not int or self.sequence < 0:
            raise ValueError('Редни број догађаја мора бити ненегативан цео број.')
