"""
Tests for the CONTRACTS coverage gap.

The bug: CONTRACTS holds only the four instruments with a micro counterpart
(ES/MES, NQ/MNQ, CL/MCL, 6E/M6E), and two accessors over that one table disagreed
about whether a missing entry is fatal.

    round_trip_commission_usd  ->  .get() chain + DEFAULT_RT_COMMISSION_USD  (safe)
    comm_ticks                 ->  CONTRACTS[sym][kind]                     (KeyError)

So comm_ticks, and cost_ticks through it, raised KeyError for 6 of the 10
instruments -- RTY, GC, ZN, BTC, ETH, 6J -- while the sweep priced all ten without
complaint, because trade_sim._round_cost_r happens to call the safe accessor.

Why it went unnoticed: the two existing consumers both avoid the gap. reprice_corrected
iterates `sorted(CONTRACTS.keys())`, which is correct since it exists to compare
micro sizing; equity_corrected's SEVEN setup list is hardcoded to ES and CL, so it is
safe by coincidence rather than design and a setup naming GC would crash it. The
first genuine consumer -- HYP-01's viability frontier, which needs comm_ticks for all
ten to compute required_gross_r -- crashed immediately.

The distinction the fix preserves: "micro" must keep raising for those six, because
they genuinely have no micro contract and a silent default would invent one. "full"
must not, because every instrument has a full-size contract.
"""
from __future__ import annotations

import pytest

from src.config import INSTRUMENTS
from src.contracts import (
    CONTRACTS, DEFAULT_RT_COMMISSION_USD, comm_ticks, cost_ticks,
    round_trip_commission_usd,
)

MICRO_LESS = sorted(set(INSTRUMENTS) - set(CONTRACTS))


def test_the_gap_is_real_and_this_test_file_is_not_vacuous():
    """If CONTRACTS ever covers all ten, the micro-raises tests below stop testing
    anything and should be revisited rather than silently passing."""
    assert MICRO_LESS, "CONTRACTS now covers every instrument; revisit this file"
    assert set(MICRO_LESS) == {"RTY", "GC", "ZN", "BTC", "ETH", "6J"}


@pytest.mark.parametrize("sym", sorted(INSTRUMENTS))
def test_comm_ticks_full_resolves_for_every_instrument(sym):
    assert comm_ticks(sym, "full") > 0


@pytest.mark.parametrize("sym", sorted(INSTRUMENTS))
def test_comm_ticks_full_matches_what_the_sweep_actually_charges(sym):
    """The fallback must reproduce trade_sim._round_cost_r's own expression, not
    merely avoid raising. A plausible-looking wrong number here would silently
    disagree with every cost figure in the repo."""
    expected = round_trip_commission_usd(sym) / INSTRUMENTS[sym]["tick_value_usd"]
    assert comm_ticks(sym, "full") == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("sym", sorted(CONTRACTS))
def test_instruments_already_in_contracts_are_unchanged(sym):
    """Regression guard: the fix must not move the four that already worked."""
    spec = CONTRACTS[sym]["full"]
    assert comm_ticks(sym, "full") == pytest.approx(
        spec["rt_commission_usd"] / spec["tick_value_usd"], rel=1e-12)


@pytest.mark.parametrize("sym", sorted(CONTRACTS))
def test_contracts_and_instruments_agree_on_tick_value(sym):
    """The premise that makes the fallback safe. If these ever diverge, comm_ticks
    would return a different number depending on which table answered, and the
    fallback would introduce an inconsistency instead of removing one."""
    assert CONTRACTS[sym]["full"]["tick_value_usd"] == pytest.approx(
        INSTRUMENTS[sym]["tick_value_usd"], rel=1e-12)


@pytest.mark.parametrize("sym", MICRO_LESS)
def test_micro_still_raises_where_no_micro_contract_exists(sym):
    """A missing micro is a real absence. Defaulting it would invent a contract that
    cannot be traded and would understate cost per R -- micros are MORE expensive per
    R, not less (MES 1.120 ticks vs ES 0.248)."""
    with pytest.raises(KeyError, match="no 'micro' contract"):
        comm_ticks(sym, "micro")


@pytest.mark.parametrize("sym", sorted(CONTRACTS))
def test_micro_resolves_where_a_micro_contract_exists(sym):
    assert comm_ticks(sym, "micro") > comm_ticks(sym, "full"), (
        "micro commission per R must exceed full-size; see the contracts.py header")


def test_micro_error_names_the_instruments_that_do_have_one():
    with pytest.raises(KeyError) as ei:
        comm_ticks("ZN", "micro")
    msg = str(ei.value)
    for sym in CONTRACTS:
        assert sym in msg


def test_unknown_instrument_raises_rather_than_defaulting():
    with pytest.raises(KeyError, match="unknown instrument"):
        comm_ticks("XYZ", "full")


@pytest.mark.parametrize("sym", sorted(INSTRUMENTS))
def test_cost_ticks_resolves_for_every_instrument(sym):
    """cost_ticks calls comm_ticks, so it inherited the same failure."""
    assert cost_ticks(sym, "full", "NY") > 0


def test_cost_ticks_equals_commission_plus_slippage():
    from src.trade_sim import slippage_ticks_for
    for sym in INSTRUMENTS:
        for sess in INSTRUMENTS[sym].get("sessions", ["NY"]):
            assert cost_ticks(sym, "full", sess) == pytest.approx(
                comm_ticks(sym, "full") + slippage_ticks_for(sym, sess), rel=1e-12)


def test_comm_ticks_kind_defaults_to_full():
    """HYP-01 calls comm_ticks(sym) positionally in places; the default must be the
    kind that always exists."""
    for sym in sorted(INSTRUMENTS):
        assert comm_ticks(sym) == pytest.approx(comm_ticks(sym, "full"), rel=1e-12)


def test_fallback_uses_the_documented_default_commission():
    for sym in MICRO_LESS:
        assert comm_ticks(sym, "full") == pytest.approx(
            DEFAULT_RT_COMMISSION_USD / INSTRUMENTS[sym]["tick_value_usd"], rel=1e-12)


def test_eth_friction_is_the_outlier_the_cost_model_claims():
    """Not a coverage test -- a sanity anchor for HYP-01/02. config records ETH TOK
    slippage at 53.14 ticks; with commission that is ~54 ticks of round-trip cost,
    two orders of magnitude above ES/NY. Any future change that quietly flattens this
    would invalidate the friction conclusions rather than improve them."""
    eth = cost_ticks("ETH", "full", "TOK")
    es = cost_ticks("ES", "full", "NY")
    assert eth > 50.0
    assert eth / es > 40.0
