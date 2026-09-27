from __future__ import annotations
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from enum import Enum
from terminalweave.core.clock import ensure_utc
from terminalweave.core.errors import ValidationError, ConflictError, NotFoundError, SafetyConstraintError
from terminalweave.rail.cutoff import TrainCutoff

class BoxStage(str, Enum):
    ON_VESSEL = 'on_vessel'
    IN_YARD = 'in_yard'
    LOADED = 'loaded'

class ConnectionStatus(str, Enum):
    KEPT = 'kept'
    RESCHEDULED = 'rescheduled'
    NO_TRAIN = 'no_train'
    LOADED = 'loaded'

@dataclass(frozen=True)
class DischargeProgress:
    work_start: datetime
    moves_per_hour: float
    quay_to_yard: timedelta = timedelta()

    def __post_init__(self) -> None:
        object.__setattr__(self, 'work_start', ensure_utc(self.work_start))
        if self.moves_per_hour <= 0 or self.quay_to_yard.total_seconds() < 0:
            raise ValidationError('invalid discharge progress')

@dataclass(frozen=True)
class YardPath:
    blockers: int
    reshuffle_each: timedelta
    travel_m: float
    speed_m_per_hour: float

    def __post_init__(self) -> None:
        if self.blockers < 0 or self.reshuffle_each.total_seconds() < 0 or self.travel_m < 0 or self.speed_m_per_hour <= 0:
            raise ValidationError('invalid yard path')

    def retrieval_time(self) -> timedelta:
        return self.blockers * self.reshuffle_each + timedelta(hours=self.travel_m / self.speed_m_per_hour)

@dataclass(frozen=True)
class RailBox:
    container_id: str
    destination: str
    booked_train: str
    teu: int
    stage: BoxStage
    moves_ahead: int
    yard_path: YardPath
    documentation_ready_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, 'documentation_ready_at', ensure_utc(self.documentation_ready_at))
        if self.teu < 1 or self.moves_ahead < 0:
            raise ValidationError('invalid rail box')

@dataclass(frozen=True)
class TrainOption:
    train_id: str
    destination: str
    cutoff: TrainCutoff
    slot_capacity_teu: int
    booked_teu: int = 0

    def __post_init__(self) -> None:
        if self.slot_capacity_teu <= 0 or not 0 <= self.booked_teu <= self.slot_capacity_teu:
            raise ValidationError('invalid train capacity')

    @property
    def departure(self) -> datetime:
        return self.cutoff.departure

    @property
    def remaining_teu(self) -> int:
        return self.slot_capacity_teu - self.booked_teu

@dataclass(frozen=True)
class BoxConnection:
    container_id: str
    booked_train: str
    train_id: str
    status: ConnectionStatus
    discharge_ready_at: datetime
    yard_ready_at: datetime
    required_ready_at: datetime
    missed_reasons: tuple[str, ...] = ()

    @property
    def slack_seconds(self) -> float:
        return (self.required_ready_at - self.yard_ready_at).total_seconds()

@dataclass(frozen=True)
class RailConnectionPlan:
    connections: tuple[BoxConnection, ...]
    safety_margin: timedelta
    as_of: datetime

    def connection(self, container_id: str) -> BoxConnection:
        for conn in self.connections:
            if conn.container_id == container_id:
                return conn
        raise NotFoundError(f'unknown container: {container_id}')

    def _select(self, status: ConnectionStatus) -> tuple[BoxConnection, ...]:
        return tuple((c for c in self.connections if c.status == status))

    def kept(self) -> tuple[BoxConnection, ...]:
        return self._select(ConnectionStatus.KEPT)

    def rescheduled(self) -> tuple[BoxConnection, ...]:
        return self._select(ConnectionStatus.RESCHEDULED)

    def unfulfilled(self) -> tuple[BoxConnection, ...]:
        return self._select(ConnectionStatus.NO_TRAIN)

    def loaded(self) -> tuple[BoxConnection, ...]:
        return self._select(ConnectionStatus.LOADED)

class ShipToRailPlanner:

    def __init__(self, safety_margin: timedelta):
        if safety_margin.total_seconds() < 0:
            raise ValidationError('safety margin cannot be negative')
        self.safety_margin = safety_margin

    def with_margin(self, margin: timedelta) -> 'ShipToRailPlanner':
        if margin.total_seconds() < self.safety_margin.total_seconds():
            raise SafetyConstraintError('safety margin cannot be shortened')
        return ShipToRailPlanner(margin)

    def ready_times(self, progress: DischargeProgress, box: RailBox, as_of: datetime) -> tuple[datetime, datetime]:
        as_of = ensure_utc(as_of)
        if box.stage == BoxStage.ON_VESSEL:
            discharge = progress.work_start + timedelta(hours=(box.moves_ahead + 1) / progress.moves_per_hour)
            available = discharge + progress.quay_to_yard
        else:
            discharge = as_of
            available = as_of
        return discharge, max(available, as_of) + box.yard_path.retrieval_time()

    def missed_reasons(self, box: RailBox, train: TrainOption, yard_ready_at: datetime, remaining_teu: int) -> tuple[str, ...]:
        reasons = []
        if yard_ready_at > train.cutoff.physical_deadline - self.safety_margin:
            reasons.append('physical_cutoff')
        if box.documentation_ready_at > train.cutoff.documentation_deadline:
            reasons.append('documentation_cutoff')
        if remaining_teu < box.teu:
            reasons.append('slot_capacity')
        return tuple(reasons)

    def earliest_alternative(self, box: RailBox, yard_ready_at: datetime, trains: list[TrainOption], remaining_teu: dict[str, int]) -> TrainOption | None:
        options = [t for t in trains if t.train_id != box.booked_train and t.destination == box.destination and (not self.missed_reasons(box, t, yard_ready_at, remaining_teu[t.train_id]))]
        return min(options, key=lambda t: (t.departure, t.train_id), default=None)

    def plan(self, progress: DischargeProgress, boxes: list[RailBox], trains: list[TrainOption], as_of: datetime) -> RailConnectionPlan:
        return self._build(progress, boxes, trains, as_of, None)

    def replan(self, previous: RailConnectionPlan, progress: DischargeProgress, boxes: list[RailBox], trains: list[TrainOption], as_of: datetime) -> RailConnectionPlan:
        return self._build(progress, boxes, trains, as_of, previous)

    def _build(self, progress: DischargeProgress, boxes: list[RailBox], trains: list[TrainOption], as_of: datetime, previous: RailConnectionPlan | None) -> RailConnectionPlan:
        as_of = ensure_utc(as_of)
        if len({b.container_id for b in boxes}) != len(boxes):
            raise ValidationError('duplicate container id')
        if len({t.train_id for t in trains}) != len(trains):
            raise ValidationError('duplicate train id')
        train_map = {t.train_id: t for t in trains}
        remaining = {t.train_id: t.remaining_teu for t in trains}
        connections: list[BoxConnection] = []
        timed = []
        for box in boxes:
            if box.stage == BoxStage.LOADED:
                conn = self._loaded_connection(box, train_map, as_of, previous)
                connections.append(conn)
                if conn.train_id in remaining:
                    remaining[conn.train_id] -= box.teu
                continue
            if box.booked_train not in train_map:
                raise ValidationError(f'unknown booked train: {box.booked_train}')
            discharge, yard_ready = self.ready_times(progress, box, as_of)
            timed.append((box, discharge, yard_ready))
        timed.sort(key=lambda x: (x[2], x[0].container_id))
        missed = []
        # booked trains are claimed first so late boxes cannot take slots from punctual ones
        for box, discharge, yard_ready in timed:
            train = train_map[box.booked_train]
            reasons = self.missed_reasons(box, train, yard_ready, remaining[train.train_id])
            if reasons:
                missed.append((box, discharge, yard_ready, reasons))
                continue
            remaining[train.train_id] -= box.teu
            connections.append(self._connection(box, train, ConnectionStatus.KEPT, discharge, yard_ready, ()))
        for box, discharge, yard_ready, reasons in missed:
            alternative = self.earliest_alternative(box, yard_ready, trains, remaining)
            if alternative is None:
                connections.append(self._connection(box, train_map[box.booked_train], ConnectionStatus.NO_TRAIN, discharge, yard_ready, reasons))
            else:
                remaining[alternative.train_id] -= box.teu
                connections.append(self._connection(box, alternative, ConnectionStatus.RESCHEDULED, discharge, yard_ready, reasons))
        return RailConnectionPlan(tuple(sorted(connections, key=lambda c: c.container_id)), self.safety_margin, as_of)

    def _connection(self, box: RailBox, train: TrainOption, status: ConnectionStatus, discharge: datetime, yard_ready: datetime, reasons: tuple[str, ...]) -> BoxConnection:
        return BoxConnection(box.container_id, box.booked_train, train.train_id, status, discharge, yard_ready, train.cutoff.physical_deadline - self.safety_margin, reasons)

    def _loaded_connection(self, box: RailBox, train_map: dict[str, TrainOption], as_of: datetime, previous: RailConnectionPlan | None) -> BoxConnection:
        if previous is not None:
            conn = previous.connection(box.container_id)
            if conn.status == ConnectionStatus.NO_TRAIN:
                raise ConflictError(f'loaded box had no fulfilled connection: {box.container_id}')
            return replace(conn, status=ConnectionStatus.LOADED)
        train = train_map.get(box.booked_train)
        required = train.cutoff.physical_deadline - self.safety_margin if train else as_of
        return BoxConnection(box.container_id, box.booked_train, box.booked_train, ConnectionStatus.LOADED, as_of, as_of, required, ())
