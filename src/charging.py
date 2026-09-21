from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .input_data import InputData


def _number(value: float, label: str, minimum: float | None = 0) -> float:
    if (type(value) not in (int, float) or not math.isfinite(value)
            or (minimum is not None and value < minimum)):
        raise ValueError(f'{label}: неисправна бројчана вредност {value!r}.')
    return float(value)


@dataclass(frozen=True, slots=True)
class _Segment:
    lower_kwh: float
    upper_kwh: float
    lower_power_kw: float
    slope: float  # kW/kWh; P(E) = lower_power + slope*(E-lower)

    def power(self, energy_kwh: float) -> float:
        return self.lower_power_kw + self.slope * (energy_kwh - self.lower_kwh)

    def duration(self, start: float, end: float) -> float:
        power = self.power(start)
        if power <= 0:
            return math.inf
        if self.slope == 0:
            return 3600 * (end - start) / power
        return 3600 * math.log1p(self.slope * (end - start) / power) / self.slope

    def advance(self, start: float, seconds: float) -> float:
        power = self.power(start)
        if self.slope == 0:
            return start + power * seconds / 3600
        return start + power * math.expm1(self.slope * seconds / 3600) / self.slope


class ChargingModel:
    """Модел за капацитет батерије и један тип пуњача.

    Ефективна крива већ садржи губитке! Ефикасност множи само снагу
    пуњача: P = min(P_curve(SOC), eta * P_charger).
    На SOC=1 снага је нула; последња тачка криве је леви лимес снаге.
    Позитивне снаге у свим тачкама омогућавају достизање 100% за коначно
    време. Нулта снага пуњача/искључено пуњење дају недостижан виши циљ.
    """

    def __init__(self, capacity_kwh: float, charging: dict, charger_power_kw: float,
                 departure_finalization_s: float) -> None:
        self.capacity_kwh = _number(capacity_kwh, 'capacity_kwh')
        if self.capacity_kwh == 0:
            raise ValueError('Капацитет батерије мора бити позитиван.')
        expected = {
            'model': 'piecewise_linear_power_vs_soc',
            'power_basis': 'net_energy_stored_in_battery',
            'effective_power_rule': 'min(interpolated_curve_power_kw, charger_power_kw * charger_output_to_stored_energy_efficiency)',
            'integration_method': 'analytic_piecewise_integration_with_soc_boundaries',
            'interruption_lead_time_source': 'ground_handling.departure_finalization_min',
        }
        for key, value in expected.items():
            if charging[key] != value:
                raise ValueError(f'charging.{key}: неподржано правило.')
        if charging['apply_efficiency_again_to_curve'] is not False or charging['stop_at_full_soc'] is not True:
            raise ValueError('Потребна је нето крива, са заустављањем на SOC=1.')
        if type(charging['enabled']) is not bool:
            raise ValueError('charging.enabled мора бити логичка вредност.')
        self.enabled = charging['enabled']
        self.charger_power_kw = _number(charger_power_kw, 'charger_power_kw')
        self.efficiency = _number(charging['charger_output_to_stored_energy_efficiency'], 'efficiency')
        if not 0 < self.efficiency <= 1:
            raise ValueError('Ефикасност мора бити у опсегу (0, 1].')
        self.idle_target_soc = self._soc(charging['idle_target_soc'])
        self.start_delay_s = 60 * _number(charging['start_delay_after_on_stand_min'], 'start_delay')
        self.interruption_lead_s = _number(departure_finalization_s, 'departure_finalization_s')
        points = [(self._soc(p['soc']), _number(p['power_kw'], 'curve.power_kw'))
                  for p in charging['curve']]
        if len(points) < 2 or points[0][0] != 0 or points[-1][0] != 1:
            raise ValueError('Крива мора покривати SOC од 0 до 1.')
        if any(p <= 0 for _, p in points):
            raise ValueError('Снаге криве морају бити позитивне, укључујући леви лимес на SOC=1.')
        if any(s1 <= s0 or p1 > p0 for (s0, p0), (s1, p1) in zip(points, points[1:])):
            raise ValueError('SOC мора строго расти, а снага не сме расти.')
        cap = self.charger_power_kw * self.efficiency if self.enabled else 0.0
        segments = []
        for (s0, p0), (s1, p1) in zip(points, points[1:]):
            e0, e1 = s0 * self.capacity_kwh, s1 * self.capacity_kwh
            slope = (p1 - p0) / (e1 - e0)
            boundaries = [e0, e1]
            # Додатна граница где се крива укршта са ограничењем пуњача.
            if p1 < cap < p0:
                boundaries.insert(1, e0 + (cap - p0) / slope)
            for left, right in zip(boundaries, boundaries[1:]):
                midpoint_power = p0 + slope * ((left + right) / 2 - e0)
                if midpoint_power >= cap:
                    segments.append(_Segment(left, right, cap, 0.0))
                else:
                    segments.append(_Segment(left, right, p0 + slope * (left - e0), slope))
        self._segments = tuple(segments)

    @classmethod
    def from_inputs(cls, data: InputData, vertiport_id: str = 'LYBE') -> ChargingModel:
        try:
            infrastructure = next(v['charging'] for v in data.network['vertiports']
                                  if v['id'] == vertiport_id)
        except StopIteration as exc:
            raise ValueError(f'Непознат вертипорт: {vertiport_id}.') from exc
        power = infrastructure['power_per_charger_kw'] if infrastructure['charger_count'] > 0 else 0
        return cls(data.aircraft['battery']['modeled_usable_capacity_kwh'],
                   data.config['charging'], power,
                   60 * data.config['ground_handling']['departure_finalization_min'])

    @staticmethod
    def _soc(value: float) -> float:
        soc = _number(value, 'SOC')
        if soc > 1:
            raise ValueError('SOC мора бити у опсегу 0–1.')
        return soc

    def _energy(self, value: float) -> float:
        energy = _number(value, 'energy_kwh')
        if energy > self.capacity_kwh:
            raise ValueError('Енергија премашује капацитет батерије.')
        return energy

    def power_at_soc(self, soc: float) -> float:
        energy = self._soc(soc) * self.capacity_kwh
        if energy == self.capacity_kwh:
            return 0.0
        for segment in self._segments:
            if segment.lower_kwh <= energy < segment.upper_kwh:
                return segment.power(energy)
        raise ValueError('SOC није обухваћен кривом.')

    def time_to_energy(self, initial_energy_kwh: float, target_energy_kwh: float) -> float:
        """Активно време [s]; 0 ако је циљ испуњен, inf ако није достижан.

        Циљ изван физичког капацитета је грешка, не прећутно ограничење.
        Ова функција не додаје време прикључивања/искључивања.
        """
        start, target = self._energy(initial_energy_kwh), self._energy(target_energy_kwh)
        if target <= start:
            return 0.0
        return math.fsum(segment.duration(max(start, segment.lower_kwh), min(target, segment.upper_kwh))
                         for segment in self._segments
                         if max(start, segment.lower_kwh) < min(target, segment.upper_kwh))

    def energy_after(self, initial_energy_kwh: float, duration_s: float,
                     target_soc: float | None = None) -> float:
        """Енергија после активног пуњења; подразумевани циљ је idle_target_soc.

        Ако је енергија већ изнад циља, остаје непромењена. За додељен лет
        проследити target_soc = required_departure_energy_kwh / capacity_kwh.
        За наставак пуњења проследити последњу енергију и ново трајање.
        """
        current = self._energy(initial_energy_kwh)
        remaining = _number(duration_s, 'duration_s')
        target = self.capacity_kwh * (self.idle_target_soc if target_soc is None else self._soc(target_soc))
        if current >= target or remaining == 0:
            return current
        for segment in self._segments:
            end = min(target, segment.upper_kwh)
            if current >= end:
                continue
            seconds = segment.duration(current, end)
            if math.isinf(seconds):
                return current
            if remaining >= seconds:
                remaining -= seconds
                current = end
            else:
                return max(current, min(end, segment.advance(current, remaining)))
            if current >= target:
                break
        return current

    def available_charging_seconds(self, interval_start_s: float, interval_end_s: float,
                                   on_stand_s: float, takeoff_s: float | None = None) -> float:
        """Пресек интервала са [долазак+прикључивање, полетање-припрема].

        Не проверава резервације. За подељене интервале користити исти стварни
        on_stand_s: минут прикључивања се тада не одузима више пута.
        Времена могу бити негативна (пре поноћи). Без полетања нема горње
        границе осим краја посматраног интервала.
        """
        start = _number(interval_start_s, 'interval_start_s', None)
        end = _number(interval_end_s, 'interval_end_s', None)
        arrival = _number(on_stand_s, 'on_stand_s', None)
        if end < start:
            raise ValueError('Крај интервала претходи почетку.')
        if takeoff_s is not None:
            takeoff = _number(takeoff_s, 'takeoff_s', None)
            if takeoff < arrival:
                raise ValueError('Полетање претходи доласку на позицију.')
            end = min(end, takeoff - self.interruption_lead_s)
        if not self.enabled:
            return 0.0
        return max(0.0, end - max(start, arrival + self.start_delay_s))


def main(argv: list[str] | None = None) -> int:
    from .input_data import InputDataError, load_inputs

    parser = argparse.ArgumentParser(description='Провера модела пуњења, без симулације.')
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'config.json')
    parser.add_argument('--vertiport', default='LYBE')
    args = parser.parse_args(argv)
    try:
        model = ChargingModel.from_inputs(load_inputs(args.config), args.vertiport)
        print(f'Вертипорт: {args.vertiport}; батерија: {model.capacity_kwh:g} kWh; '
              f'пуњач: {model.charger_power_kw:g} kW.')
        print('SOC почетак → циљ     Активно пуњење [min]')
        for start, end in ((0.2, 0.5), (0.5, 0.8), (0.8, 0.9), (0.8, 1.0), (0.2, 1.0)):
            seconds = model.time_to_energy(start * model.capacity_kwh, end * model.capacity_kwh)
            print(f'{start:11.0%} → {end:4.0%} {seconds / 60:23.3f}')
        print('SOC после активног пуњења од 30% (циљ 100%):')
        for minutes in (5, 10, 15):
            energy = model.energy_after(0.3 * model.capacity_kwh, minutes * 60, target_soc=1)
            print(f'  {minutes:2d} min → {energy / model.capacity_kwh:.2%}; {energy:.3f} kWh')
        print(f'После доласка на позицију: +{model.start_delay_s:g} s до пуњења; '
              f'пре полетања: {model.interruption_lead_s:g} s без пуњења.')
        print('Табеле приказују само активно пуњење; симулација још није покренута.')
    except (InputDataError, ValueError, KeyError) as exc:
        print(f'ГРЕШКА: {exc}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    raise SystemExit(main())
