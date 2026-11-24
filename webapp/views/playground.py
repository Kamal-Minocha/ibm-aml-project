"""Feature Playground: build a hypothetical transaction, watch the score."""

import math

import streamlit as st

import scoring
import ui

bundle = ui.get_bundle()

PRESETS = {
    "Typical transaction": {},
    "Fan-out with rapid forwarding": {
        "pg_payment_format": "ACH",
        "pg_amount_paid": 9_500.0,
        "pg_sender_incoming_count_24h": 3,
        "pg_sender_unique_senders_24h": 3,
        "pg_sender_outgoing_count_24h": 40,
        "pg_sender_unique_recipients_24h": 40,
        "pg_sender_tx_count_1h": 25,
        "pg_sender_minutes_since": 1.0,
    },
}
MAX_AMOUNT = 1e13  # above every amount in the data; keeps 24h totals finite
COUNT_FIELDS = ("outgoing_count_24h", "unique_recipients_24h",
                "incoming_count_24h", "unique_senders_24h", "tx_count_1h")
SAVED_KEY = "pg_saved"  # survives page switches, unlike widget state


def playground_defaults() -> dict:
    d = bundle.defaults
    return {
        "pg_amount_paid": float(d["amount_paid"]),
        "pg_amount_received": float(d["amount_received"]),
        "pg_payment_format": d["payment_format"],
        "pg_payment_currency": d["payment_currency"],
        "pg_receiving_currency": d["receiving_currency"],
        "pg_sender_entity_type": d["sender_entity_type"],
        "pg_receiver_entity_type": d["receiver_entity_type"],
        "pg_same_bank": bool(d["same_bank"]),
        "pg_same_account": bool(d["same_account"]),
        "pg_hour": int(d["transaction_hour"]),
        **{
            f"pg_{side}_{name}": int(d[f"{side}_{name}"])
            for side in ("sender", "receiver")
            for name in COUNT_FIELDS
        },
        "pg_sender_minutes_since": float(d["sender_seconds_since_activity"]) / 60,
        "pg_receiver_minutes_since": float(d["receiver_seconds_since_activity"]) / 60,
    }


def apply_preset(name: str) -> None:
    values = {**playground_defaults(), **PRESETS[name]}
    st.session_state.update(values)
    st.session_state["pg_received_kept"] = values["pg_amount_received"]


def minutes_bounds(side: str) -> tuple[float, float]:
    """Allowed 'minutes since previous activity' for the current counts."""
    ss = st.session_state
    total = ss[f"pg_{side}_outgoing_count_24h"] + ss[f"pg_{side}_incoming_count_24h"]
    low, high = scoring.seconds_since_range(ss[f"pg_{side}_tx_count_1h"], total)
    return float(math.ceil(low / 60)), (high / 60 if high is not None else 1e6)


def sync_playground_state() -> None:
    """Restore values lost on page switches, then make every input
    consistent before any widget is drawn (Streamlit only allows changing
    a widget's value before it is created in a run)."""
    ss = st.session_state
    saved = ss.get(SAVED_KEY, {})
    for key, value in playground_defaults().items():
        if key not in ss:
            ss[key] = saved.get(key, value)

    if ss.pg_same_account:  # one account: the receiver *is* the sender
        ss.pg_same_bank = True
        ss.pg_receiver_entity_type = ss.pg_sender_entity_type
        for name in (*COUNT_FIELDS, "minutes_since"):
            ss[f"pg_receiver_{name}"] = ss[f"pg_sender_{name}"]

    for side in ("sender", "receiver"):
        p = f"pg_{side}_"
        ss[p + "unique_recipients_24h"] = min(ss[p + "unique_recipients_24h"],
                                              ss[p + "outgoing_count_24h"])
        ss[p + "unique_senders_24h"] = min(ss[p + "unique_senders_24h"],
                                           ss[p + "incoming_count_24h"])
        ss[p + "tx_count_1h"] = min(ss[p + "tx_count_1h"],
                                    ss[p + "outgoing_count_24h"] + ss[p + "incoming_count_24h"])
        low, high = minutes_bounds(side)
        ss[p + "minutes_since"] = float(min(max(ss[p + "minutes_since"], low), high))


def history_inputs(side: str, disabled: bool = False) -> None:
    ss = st.session_state
    p = f"pg_{side}_"
    if disabled:
        st.caption("Self-transfer: same account as the sender, so same history.")
    out, inc = ss[p + "outgoing_count_24h"], ss[p + "incoming_count_24h"]
    low, high = minutes_bounds(side)
    a, b = st.columns(2)
    a.number_input("Payments made", 0, 5_000, step=1, key=p + "outgoing_count_24h",
                   disabled=disabled)
    b.number_input("...to distinct accounts", 0, out, step=1,
                   key=p + "unique_recipients_24h", disabled=disabled)
    a.number_input("Payments received", 0, 5_000, step=1, key=p + "incoming_count_24h",
                   disabled=disabled)
    b.number_input("...from distinct accounts", 0, inc, step=1,
                   key=p + "unique_senders_24h", disabled=disabled)
    a.number_input("Transactions in the last hour", 0, out + inc, step=1,
                   key=p + "tx_count_1h", disabled=disabled)
    b.number_input("Minutes since previous activity", low, high, step=1.0,
                   key=p + "minutes_since", disabled=disabled,
                   help="Limited by the counts: activity in the last hour means at "
                        "most 60 minutes; none in 24h means over 1,440.")


ui.page_header(
    "Feature Playground",
    "Build a hypothetical transaction and watch the score respond. Inputs start "
    "at training-data medians. Fields the pipeline derives (rapid forwarding, "
    "fan-in/out, 24h amounts, weekday) are computed for you, and inputs are kept "
    "within the relationships that always hold in the real feature data.",
)
sync_playground_state()
ss = st.session_state

with st.container(horizontal=True, vertical_alignment="center"):
    st.markdown("**Start from:**", width="content")
    for name, icon in zip(PRESETS, (":material/restart_alt:", ":material/hub:"), strict=True):
        st.button(name, on_click=apply_preset, args=(name,), icon=icon)

categories = {
    c: [v for v in bundle.mappings[c] if v != scoring.MISSING_CATEGORY]
    for c in bundle.categorical_features
}
self_transfer = ss.pg_same_account
inputs, result = st.columns(2, gap="large")

with inputs:
    tab_tx, tab_sender, tab_receiver = st.tabs(
        [":material/receipt_long: This transaction",
         ":material/north_east: Sender, previous 24 hours",
         ":material/south_west: Receiver, previous 24 hours"]
    )
    with tab_tx:
        a, b = st.columns(2)
        a.number_input("Amount paid", 0.0, MAX_AMOUNT, step=100.0, key="pg_amount_paid")
        b.selectbox("Payment format", categories["payment_format"], key="pg_payment_format")
        a.selectbox("Payment currency", categories["payment_currency"], key="pg_payment_currency")
        b.selectbox("Receiving currency", categories["receiving_currency"],
                    key="pg_receiving_currency")
        currencies_differ = ss.pg_payment_currency != ss.pg_receiving_currency
        if currencies_differ:
            # Shown only when currencies differ. Streamlit forgets the value of
            # a hidden widget, so keep a copy and restore it on reappearance.
            if not ss.get("pg_received_shown", False):
                ss.pg_amount_received = ss.get("pg_received_kept", ss.pg_amount_received)
            st.number_input("Amount received (in receiving currency)", 0.0, MAX_AMOUNT,
                            step=100.0, key="pg_amount_received")
            ss["pg_received_kept"] = ss.pg_amount_received
        ss["pg_received_shown"] = currencies_differ
        st.slider("Hour of day", 0, 23, key="pg_hour")
        a, b = st.columns(2)
        a.checkbox("Self-transfer (same account)", key="pg_same_account")
        b.checkbox("Same bank", key="pg_same_bank", disabled=self_transfer)
        a.selectbox("Sender entity type", categories["sender_entity_type"],
                    key="pg_sender_entity_type")
        b.selectbox("Receiver entity type", categories["receiver_entity_type"],
                    key="pg_receiver_entity_type", disabled=self_transfer)
    with tab_sender:
        history_inputs("sender")
    with tab_receiver:
        history_inputs("receiver", disabled=self_transfer)

ss[SAVED_KEY] = {key: ss[key] for key in playground_defaults()}

overrides = {
    "amount_paid": ss.pg_amount_paid,
    "amount_received": ss.pg_amount_received,
    "payment_format": ss.pg_payment_format,
    "payment_currency": ss.pg_payment_currency,
    "receiving_currency": ss.pg_receiving_currency,
    "sender_entity_type": ss.pg_sender_entity_type,
    "receiver_entity_type": ss.pg_receiver_entity_type,
    "same_account": int(ss.pg_same_account),
    "same_bank": int(ss.pg_same_bank),
    "transaction_hour": ss.pg_hour,
}
for side in ("sender", "receiver"):
    for name in COUNT_FIELDS:
        overrides[f"{side}_{name}"] = ss[f"pg_{side}_{name}"]
    overrides[f"{side}_seconds_since_activity"] = ss[f"pg_{side}_minutes_since"] * 60

row = scoring.build_playground_row(bundle, overrides)
prepared = scoring.prepare_batch(row, bundle)
score_value = float(scoring.score(bundle, prepared.matrix)[0])
derived = row.iloc[0]

with result:
    with st.container(border=True):
        st.markdown("**The model's verdict**")
        ui.score_panel(score_value)
        st.markdown(
            f"Derived by the pipeline rules: rapid forwarding = "
            f"**{'yes' if derived['rapid_forwarding_indicator'] else 'no'}**, "
            f"sender received money in 24h = "
            f"**{'yes' if derived['sender_received_money_24h'] else 'no'}**, "
            f"same currency = **{'yes' if derived['same_currency'] else 'no'}**."
        )
        st.caption(
            "A what-if sandbox. Features not shown (day of month, account metadata) "
            "stay at training medians, and 24h amounts assume earlier transactions "
            "were the same size as this one."
        )

ui.explanation_chart(derived, prepared.matrix[0])
