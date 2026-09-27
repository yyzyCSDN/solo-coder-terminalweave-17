from datetime import datetime, timedelta, timezone
import pytest

from terminalweave.core.errors import ConflictError, NotFoundError, ValidationError
from terminalweave.rail.blocks import RailContainer, Wagon
from terminalweave.rail.cutoff import TrainCutoff
from terminalweave.transshipment.rail_connection import (
    ConnectionStatus,
    DischargeForecast,
    RailBox,
    RailConnectionPlanner,
    RailDeparture,
    YardPath,
)

UTC = timezone.utc
D = datetime(2026, 1, 1, tzinfo=UTC)

def planner():
    return RailConnectionPlanner(safety_buffer=timedelta(minutes=30))

def train(tid, departure, dest="X", wagons=None, doc_before=timedelta(hours=2), phys_before=timedelta(hours=1)):
    wagons = wagons if wagons is not None else [Wagon(f"{tid}-W1", 60000, 40, dest)]
    return RailDeparture(tid, dest, TrainCutoff(tid, departure, doc_before, phys_before), tuple(wagons))

def box(cid, booked, berthed_at=D, moves_ahead=0, rate=30.0, blockers=0, travel_m=0.0, docs=None, weight=20000, length=20, dest="X", discharged_at=None):
    return RailBox(
        RailContainer(cid, dest, weight, length),
        booked,
        DischargeForecast(cid, berthed_at, moves_ahead, rate, discharged_at),
        YardPath(blockers, travel_m),
        docs if docs is not None else berthed_at,
    )

def test_box_catches_booked_train():
    t1 = train("T1", D + timedelta(hours=10))
    plan = planner().plan([box("A", "T1", moves_ahead=5)], [t1])
    v = plan.for_container("A")
    assert v.status == ConnectionStatus.PLANNED and v.train_id == "T1"
    assert v.quay_ready_at == D + timedelta(minutes=12)
    assert v.physical_slack_seconds == pytest.approx(8 * 3600 + 18 * 60)

def test_discharge_delay_rebooks_to_earliest_alternative():
    trains = [train("T1", D + timedelta(hours=4)), train("T2", D + timedelta(hours=8)), train("T3", D + timedelta(hours=12))]
    plan = planner().plan([box("A", "T1", moves_ahead=119)], trains)
    v = plan.for_container("A")
    assert v.status == ConnectionStatus.REBOOKED and v.train_id == "T2"
    assert v.reason == "booked train cutoff missed"

def test_yard_extraction_path_counts_against_cutoff():
    trains = [train("T1", D + timedelta(hours=3)), train("T2", D + timedelta(hours=6))]
    p = planner()
    assert p.extraction_time(YardPath(4, 15000)) == timedelta(minutes=42)
    plan = p.plan([box("A", "T1", moves_ahead=29, blockers=4, travel_m=15000)], trains)
    v = plan.for_container("A")
    assert v.rail_ready_at == D + timedelta(hours=1, minutes=42)
    assert v.status == ConnectionStatus.REBOOKED and v.train_id == "T2"

def test_wagon_capacity_spills_to_next_train():
    t1 = train("T1", D + timedelta(hours=4), wagons=[Wagon("W1", 60000, 40, "X")])
    t2 = train("T2", D + timedelta(hours=8))
    a = box("A", "T1", weight=25000, length=40)
    b = box("B", "T1", weight=25000, length=40)
    plan = planner().plan([a, b], [t1, t2])
    assert plan.for_container("A").train_id == "T1"
    v = plan.for_container("B")
    assert v.status == ConnectionStatus.REBOOKED and v.train_id == "T2"
    assert v.reason == "booked train has no wagon capacity"

def test_documentation_cutoff_is_enforced():
    trains = [train("T1", D + timedelta(hours=4)), train("T2", D + timedelta(hours=8))]
    late_docs = box("A", "T1", moves_ahead=29, docs=D + timedelta(hours=2, minutes=30))
    plan = planner().plan([late_docs], trains)
    v = plan.for_container("A")
    assert v.status == ConnectionStatus.REBOOKED and v.train_id == "T2"
    assert v.documentation_slack_seconds == pytest.approx(3.5 * 3600)

def test_safety_buffer_is_never_shortened():
    trains = [train("T1", D + timedelta(hours=2)), train("T2", D + timedelta(hours=6))]
    p = planner()
    ready_without_buffer = box("A", "T1", discharged_at=D + timedelta(minutes=45))
    plan = p.plan([ready_without_buffer], trains)
    v = plan.for_container("A")
    assert v.train_id == "T2" and v.status == ConnectionStatus.REBOOKED
    assert p.safety_buffer == timedelta(minutes=30)
    with pytest.raises(ValidationError):
        RailConnectionPlanner(safety_buffer=timedelta(seconds=-1))

def test_replan_only_moves_unloaded_boxes():
    t1 = train("T1", D + timedelta(hours=6))
    t2 = train("T2", D + timedelta(hours=10))
    p = planner()
    first = p.plan([box("A", "T1"), box("B", "T1")], [t1, t2])
    assert {v.status for v in first.verdicts} == {ConnectionStatus.PLANNED}
    p.mark_loaded("A")
    delayed_b = box("B", "T1", berthed_at=D + timedelta(hours=4, minutes=30), moves_ahead=29)
    second = p.replan([delayed_b], [t1, t2])
    assert second.for_container("A").status == ConnectionStatus.LOADED
    assert second.for_container("A").train_id == "T1"
    vb = second.for_container("B")
    assert vb.status == ConnectionStatus.REBOOKED and vb.train_id == "T2"
    with pytest.raises(ValidationError):
        p.replan([box("A", "T1")], [t1, t2])

def test_loaded_container_keeps_wagon_capacity_on_replan():
    t1 = train("T1", D + timedelta(hours=6), wagons=[Wagon("W1", 60000, 40, "X")])
    t2 = train("T2", D + timedelta(hours=10))
    p = planner()
    p.plan([box("A", "T1", length=40)], [t1, t2])
    p.mark_loaded("A")
    second = p.replan([box("B", "T1", length=40)], [t1, t2])
    assert second.for_container("B").train_id == "T2"

def test_replan_fails_closed_when_loaded_train_disappears():
    t1 = train("T1", D + timedelta(hours=6))
    t2 = train("T2", D + timedelta(hours=10))
    p = planner()
    p.plan([box("A", "T1")], [t1, t2])
    p.mark_loaded("A")
    with pytest.raises(ConflictError):
        p.replan([], [t2])

def test_replan_fails_closed_when_loaded_train_shrinks():
    t1 = train("T1", D + timedelta(hours=6), wagons=[Wagon("W1", 60000, 40, "X")])
    p = planner()
    p.plan([box("A", "T1", length=40)], [t1])
    p.mark_loaded("A")
    shrunk = train("T1", D + timedelta(hours=6), wagons=[Wagon("W1", 60000, 20, "X")])
    with pytest.raises(ConflictError):
        p.replan([], [shrunk])

def test_mark_loaded_requires_assignment():
    p = planner()
    with pytest.raises(NotFoundError):
        p.mark_loaded("A")
    p.plan([box("A", "T1", discharged_at=D + timedelta(hours=4))], [train("T1", D + timedelta(hours=2))])
    with pytest.raises(NotFoundError):
        p.mark_loaded("A")

def test_stranded_when_no_reachable_departure():
    plan = planner().plan([box("A", "T1", discharged_at=D + timedelta(hours=4))], [train("T1", D + timedelta(hours=2))])
    v = plan.for_container("A")
    assert v.status == ConnectionStatus.STRANDED and v.train_id is None
    assert v.physical_slack_seconds is None
    assert "no reachable departure with capacity" in v.reason

def test_unknown_booked_train_falls_back_to_alternative():
    plan = planner().plan([box("A", "MISSING")], [train("T1", D + timedelta(hours=6))])
    v = plan.for_container("A")
    assert v.status == ConnectionStatus.REBOOKED and v.train_id == "T1"
    assert v.reason == "booked train not found"

def test_destination_mismatch_never_matches():
    t1 = train("T1", D + timedelta(hours=6), dest="Y")
    plan = planner().plan([box("A", "T1", dest="X")], [t1])
    v = plan.for_container("A")
    assert v.status == ConnectionStatus.STRANDED
    assert "different destination" in v.reason

def test_forecast_and_model_validation():
    f = DischargeForecast("C", D, 9, 40.0)
    assert f.quay_ready_at == D + timedelta(minutes=15)
    assert DischargeForecast("C", D, 0, 40.0, D + timedelta(hours=1)).quay_ready_at == D + timedelta(hours=1)
    with pytest.raises(ValidationError):
        YardPath(-1, 0)
    with pytest.raises(ValidationError):
        DischargeForecast("C", D, -1, 30.0)
    with pytest.raises(ValidationError):
        RailDeparture("T1", "X", TrainCutoff("T2", D, timedelta(hours=1), timedelta(hours=1)), (Wagon("W", 60000, 40),))
    with pytest.raises(ValidationError):
        RailDeparture("T1", "X", TrainCutoff("T1", D, timedelta(hours=1), timedelta(hours=1)), (Wagon("W", 60000, 40, "Y"),))
    with pytest.raises(ValidationError):
        box("A", "T1").forecast.container_id and RailBox(
            RailContainer("A", "X", 20000, 20), "T1",
            DischargeForecast("B", D, 0, 30.0), YardPath(0, 0), D,
        )
    with pytest.raises(ValidationError):
        planner().plan([box("A", "T1"), box("A", "T1")], [train("T1", D + timedelta(hours=6))])
