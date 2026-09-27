from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from terminalweave.core.clock import ensure_utc
from terminalweave.core.errors import ValidationError, ConflictError, CapacityError, NotFoundError
from terminalweave.rail.blocks import RailContainer, Wagon, RailBlockPlanner
from terminalweave.rail.cutoff import TrainCutoff

class ConnectionStatus(str, Enum):
    PLANNED = 'planned'
    REBOOKED = 'rebooked'
    STRANDED = 'stranded'
    LOADED = 'loaded'

@dataclass(frozen=True)
class DischargeForecast:
    container_id: str
    berthed_at: datetime
    moves_ahead: int
    crane_moves_per_hour: float
    discharged_at: datetime | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, 'berthed_at', ensure_utc(self.berthed_at))
        if self.discharged_at is not None:
            object.__setattr__(self, 'discharged_at', ensure_utc(self.discharged_at))
        if self.moves_ahead < 0 or self.crane_moves_per_hour <= 0:
            raise ValidationError('invalid discharge forecast')

    @property
    def quay_ready_at(self) -> datetime:
        if self.discharged_at is not None:
            return self.discharged_at
        return self.berthed_at + timedelta(hours=(self.moves_ahead + 1) / self.crane_moves_per_hour)

@dataclass(frozen=True)
class YardPath:
    blockers: int
    travel_m: float

    def __post_init__(self) -> None:
        if self.blockers < 0 or self.travel_m < 0:
            raise ValidationError('invalid yard path')

@dataclass(frozen=True)
class RailBox:
    container: RailContainer
    booked_train_id: str
    forecast: DischargeForecast
    path: YardPath
    docs_ready_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, 'docs_ready_at', ensure_utc(self.docs_ready_at))
        if self.forecast.container_id != self.container.container_id:
            raise ValidationError('forecast does not belong to container')

    @property
    def container_id(self) -> str:
        return self.container.container_id

    @property
    def destination(self) -> str:
        return self.container.destination

@dataclass(frozen=True)
class RailDeparture:
    train_id: str
    destination: str
    cutoff: TrainCutoff
    wagons: tuple[Wagon, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, 'wagons', tuple(self.wagons))
        if self.cutoff.train_id != self.train_id:
            raise ValidationError('cutoff does not belong to train')
        if not self.wagons:
            raise ValidationError('train must carry at least one wagon')
        if len({w.wagon_id for w in self.wagons}) != len(self.wagons):
            raise ValidationError('duplicate wagon id')
        if any((w.destination is not None and w.destination != self.destination for w in self.wagons)):
            raise ValidationError('wagon destination does not match train')

@dataclass(frozen=True)
class RailConnectionVerdict:
    container_id: str
    status: ConnectionStatus
    booked_train_id: str
    train_id: str | None
    quay_ready_at: datetime
    rail_ready_at: datetime
    physical_slack_seconds: float | None
    documentation_slack_seconds: float | None
    reason: str | None

@dataclass(frozen=True)
class RailConnectionPlan:
    verdicts: tuple[RailConnectionVerdict, ...]

    def for_container(self, container_id: str) -> RailConnectionVerdict:
        for verdict in self.verdicts:
            if verdict.container_id == container_id:
                return verdict
        raise NotFoundError(f'unknown container: {container_id}')

    def by_status(self, status: ConnectionStatus) -> tuple[RailConnectionVerdict, ...]:
        return tuple((v for v in self.verdicts if v.status == status))

class RailConnectionPlanner:
    # The safety buffer is applied in full to every cutoff check. No planning
    # path may shorten it to force a container onto a train.

    def __init__(self, *, safety_buffer: timedelta, seconds_per_reshuffle: float=180.0, travel_speed_m_per_min: float=500.0):
        if safety_buffer.total_seconds() < 0:
            raise ValidationError('safety buffer cannot be negative')
        if seconds_per_reshuffle < 0:
            raise ValidationError('reshuffle time cannot be negative')
        if travel_speed_m_per_min <= 0:
            raise ValidationError('travel speed must be positive')
        self._safety_buffer = safety_buffer
        self._seconds_per_reshuffle = seconds_per_reshuffle
        self._travel_speed_m_per_min = travel_speed_m_per_min
        self._loaded: dict[str, RailBox] = {}
        self._assignments: dict[str, str] = {}
        self._current_boxes: dict[str, RailBox] = {}
        self._trains: dict[str, RailDeparture] = {}
        self._capacity: dict[str, RailBlockPlanner] = {}

    @property
    def safety_buffer(self) -> timedelta:
        return self._safety_buffer

    def extraction_time(self, path: YardPath) -> timedelta:
        seconds = path.blockers * self._seconds_per_reshuffle + path.travel_m / self._travel_speed_m_per_min * 60.0
        return timedelta(seconds=seconds)

    def plan(self, boxes: list[RailBox], trains: list[RailDeparture]) -> RailConnectionPlan:
        self._loaded = {}
        return self._build(boxes, trains)

    def replan(self, boxes: list[RailBox], trains: list[RailDeparture]) -> RailConnectionPlan:
        frozen = {b.container_id for b in boxes} & set(self._loaded)
        if frozen:
            raise ValidationError(f'loaded containers cannot be replanned: {sorted(frozen)}')
        return self._build(boxes, trains)

    def mark_loaded(self, container_id: str) -> None:
        if container_id not in self._assignments:
            raise NotFoundError(f'container has no assigned train: {container_id}')
        self._loaded[container_id] = self._current_boxes[container_id]

    def _build(self, boxes: list[RailBox], trains: list[RailDeparture]) -> RailConnectionPlan:
        if len({b.container_id for b in boxes}) != len(boxes):
            raise ValidationError('duplicate container id')
        if len({t.train_id for t in trains}) != len(trains):
            raise ValidationError('duplicate train id')
        self._trains = {t.train_id: t for t in trains}
        self._capacity = {t.train_id: RailBlockPlanner(list(t.wagons)) for t in trains}
        verdicts: list[RailConnectionVerdict] = []
        for container_id in sorted(self._loaded):
            verdicts.append(self._keep_loaded(self._loaded[container_id]))
        for box in sorted(boxes, key=lambda b: (self._rail_ready_at(b)[1], b.container_id)):
            verdicts.append(self._plan_one(box))
        self._current_boxes = {**self._loaded, **{b.container_id: b for b in boxes}}
        self._assignments = {v.container_id: v.train_id for v in verdicts if v.train_id is not None}
        return RailConnectionPlan(tuple(sorted(verdicts, key=lambda v: v.container_id)))

    def _keep_loaded(self, box: RailBox) -> RailConnectionVerdict:
        train_id = self._assignments[box.container_id]
        train = self._trains.get(train_id)
        if train is None:
            raise ConflictError(f'loaded container {box.container_id} is on unknown train {train_id}')
        try:
            self._book(box, train_id)
        except CapacityError as exc:
            raise ConflictError(f'loaded container {box.container_id} no longer fits train {train_id}') from exc
        return self._verdict(box, ConnectionStatus.LOADED, train, None)

    def _plan_one(self, box: RailBox) -> RailConnectionVerdict:
        _, rail_ready = self._rail_ready_at(box)
        booked = self._trains.get(box.booked_train_id)
        if booked is not None and self._fits(box, booked, rail_ready):
            self._book(box, booked.train_id)
            return self._verdict(box, ConnectionStatus.PLANNED, booked, None)
        reason = self._booked_failure_reason(box, booked, rail_ready)
        alternative = self._earliest_alternative(box, rail_ready)
        if alternative is None:
            return self._verdict(box, ConnectionStatus.STRANDED, None, reason + '; no reachable departure with capacity')
        self._book(box, alternative.train_id)
        return self._verdict(box, ConnectionStatus.REBOOKED, alternative, reason)

    def _rail_ready_at(self, box: RailBox) -> tuple[datetime, datetime]:
        quay_ready = box.forecast.quay_ready_at
        return (quay_ready, quay_ready + self.extraction_time(box.path))

    def _cutoff_ok(self, box: RailBox, train: RailDeparture, rail_ready_at: datetime) -> bool:
        if box.docs_ready_at > train.cutoff.documentation_deadline:
            return False
        return rail_ready_at + self._safety_buffer <= train.cutoff.physical_deadline

    def _capacity_ok(self, box: RailBox, train_id: str) -> bool:
        return any((load.can_add(box.container) for load in self._capacity[train_id].loads))

    def _fits(self, box: RailBox, train: RailDeparture, rail_ready_at: datetime) -> bool:
        if box.destination != train.destination:
            return False
        if not self._cutoff_ok(box, train, rail_ready_at):
            return False
        return self._capacity_ok(box, train.train_id)

    def _book(self, box: RailBox, train_id: str) -> None:
        self._capacity[train_id].assign([box.container])

    def _booked_failure_reason(self, box: RailBox, booked: RailDeparture | None, rail_ready_at: datetime) -> str:
        if booked is None:
            return 'booked train not found'
        if box.destination != booked.destination:
            return 'booked train serves a different destination'
        if not self._cutoff_ok(box, booked, rail_ready_at):
            return 'booked train cutoff missed'
        return 'booked train has no wagon capacity'

    def _earliest_alternative(self, box: RailBox, rail_ready_at: datetime) -> RailDeparture | None:
        feasible = [t for t in self._trains.values() if t.train_id != box.booked_train_id and self._fits(box, t, rail_ready_at)]
        if not feasible:
            return None
        return min(feasible, key=lambda t: (t.cutoff.departure, t.train_id))

    def _verdict(self, box: RailBox, status: ConnectionStatus, train: RailDeparture | None, reason: str | None) -> RailConnectionVerdict:
        quay_ready, rail_ready = self._rail_ready_at(box)
        if train is None:
            return RailConnectionVerdict(box.container_id, status, box.booked_train_id, None, quay_ready, rail_ready, None, None, reason)
        physical_slack = (train.cutoff.physical_deadline - (rail_ready + self._safety_buffer)).total_seconds()
        doc_slack = (train.cutoff.documentation_deadline - box.docs_ready_at).total_seconds()
        return RailConnectionVerdict(box.container_id, status, box.booked_train_id, train.train_id, quay_ready, rail_ready, physical_slack, doc_slack, reason)
