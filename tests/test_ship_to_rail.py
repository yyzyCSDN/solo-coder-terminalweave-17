from datetime import datetime, timedelta, timezone
import pytest

from terminalweave.core.errors import ConflictError, NotFoundError, SafetyConstraintError, ValidationError
from terminalweave.rail.cutoff import TrainCutoff
from terminalweave.rail.connection_plan import BoxStage, ConnectionStatus, DischargeProgress, RailBox, ShipToRailPlanner, TrainOption, YardPath

UTC = timezone.utc
D = datetime(2026, 1, 1, tzinfo=UTC)
PATH = YardPath(2, timedelta(minutes=5), 1200, 1200)
PROGRESS = DischargeProgress(D + timedelta(hours=2), 30, timedelta(minutes=18))

def train(train_id, departure, capacity=10, booked=0, destination='NORTH'):
    return TrainOption(train_id, destination, TrainCutoff(train_id, departure, timedelta(hours=2), timedelta(hours=1)), capacity, booked)

def box(container_id, booked='T1', stage=BoxStage.ON_VESSEL, moves=5, path=PATH, doc=None, teu=1):
    return RailBox(container_id, 'NORTH', booked, teu, stage, moves, path, doc or D + timedelta(hours=1))

def planner():
    return ShipToRailPlanner(timedelta(minutes=30))

def test_box_ready_before_cutoff_keeps_booked_train():
    plan = planner().plan(PROGRESS, [box('C1')], [train('T1', D + timedelta(hours=10))], D)
    conn = plan.connection('C1')
    assert conn.status == ConnectionStatus.KEPT and conn.train_id == 'T1'
    assert conn.yard_ready_at == D + timedelta(hours=3, minutes=40)
    assert conn.slack_seconds == pytest.approx(timedelta(hours=4, minutes=50).total_seconds())
    assert plan.unfulfilled() == ()

def test_missed_box_gets_earliest_alternative_train():
    delayed = DischargeProgress(D + timedelta(hours=9), 30, timedelta(minutes=18))
    trains = [train('T1', D + timedelta(hours=10)), train('T2', D + timedelta(hours=20)), train('T3', D + timedelta(hours=16))]
    plan = planner().plan(delayed, [box('C1')], trains, D)
    conn = plan.connection('C1')
    assert conn.status == ConnectionStatus.RESCHEDULED
    assert conn.booked_train == 'T1' and conn.train_id == 'T3'
    assert conn.missed_reasons == ('physical_cutoff',)

def test_safety_margin_is_never_shortened_to_release_a_box():
    p = planner()
    tight = DischargeProgress(D + timedelta(hours=7), 30, timedelta(minutes=18))
    plan = p.plan(tight, [box('C1')], [train('T1', D + timedelta(hours=10))], D)
    conn = plan.connection('C1')
    assert conn.yard_ready_at == D + timedelta(hours=8, minutes=40)
    assert conn.status == ConnectionStatus.NO_TRAIN and conn.missed_reasons == ('physical_cutoff',)
    with pytest.raises(SafetyConstraintError):
        p.with_margin(timedelta(minutes=15))
    assert p.with_margin(timedelta(minutes=45)).safety_margin == timedelta(minutes=45)
    with pytest.raises(ValidationError):
        ShipToRailPlanner(timedelta(minutes=-1))

def test_slot_capacity_is_consumed_in_ready_order():
    direct = YardPath(0, timedelta(), 0, 1.0)
    boxes = [box('C1', stage=BoxStage.IN_YARD, moves=0, path=direct), box('C2', stage=BoxStage.IN_YARD, moves=0, path=direct)]
    trains = [train('T1', D + timedelta(hours=10), capacity=1), train('T2', D + timedelta(hours=20))]
    plan = planner().plan(DischargeProgress(D, 30), boxes, trains, D)
    assert plan.connection('C1').status == ConnectionStatus.KEPT
    conn = plan.connection('C2')
    assert conn.status == ConnectionStatus.RESCHEDULED and conn.train_id == 'T2'
    assert conn.missed_reasons == ('slot_capacity',)

def test_documentation_cutoff_blocks_even_when_physical_path_fits():
    plan = planner().plan(PROGRESS, [box('C1', doc=D + timedelta(hours=9))], [train('T1', D + timedelta(hours=10))], D)
    conn = plan.connection('C1')
    assert conn.status == ConnectionStatus.NO_TRAIN and conn.missed_reasons == ('documentation_cutoff',)

def test_replan_reschedules_only_boxes_not_yet_loaded():
    trains = [train('T1', D + timedelta(hours=10), capacity=2), train('T2', D + timedelta(hours=20))]
    first = planner().plan(PROGRESS, [box('C1'), box('C2')], trains, D)
    assert first.connection('C2').status == ConnectionStatus.KEPT
    delayed = DischargeProgress(D + timedelta(hours=9), 30, timedelta(minutes=18))
    updated = [box('C1'), box('C2', stage=BoxStage.LOADED, moves=0)]
    second = planner().replan(first, delayed, updated, trains, D + timedelta(hours=3))
    loaded = second.connection('C2')
    assert loaded.status == ConnectionStatus.LOADED and loaded.train_id == 'T1'
    assert loaded.yard_ready_at == first.connection('C2').yard_ready_at
    moved = second.connection('C1')
    assert moved.status == ConnectionStatus.RESCHEDULED and moved.train_id == 'T2'
    assert len(second.loaded()) == 1 and len(second.rescheduled()) == 1

def test_replan_rejects_inconsistent_loaded_boxes():
    trains = [train('T1', D + timedelta(hours=10))]
    tight = DischargeProgress(D + timedelta(hours=7), 30, timedelta(minutes=18))
    first = planner().plan(tight, [box('C1')], trains, D)
    assert first.connection('C1').status == ConnectionStatus.NO_TRAIN
    with pytest.raises(ConflictError):
        planner().replan(first, tight, [box('C1', stage=BoxStage.LOADED, moves=0)], trains, D + timedelta(hours=3))
    with pytest.raises(NotFoundError):
        planner().replan(first, tight, [box('C9', stage=BoxStage.LOADED, moves=0)], trains, D + timedelta(hours=3))
