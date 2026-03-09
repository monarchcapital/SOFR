import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression 
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, date, timedelta

from io import BytesIO
from matplotlib.backends.backend_pdf import PdfPages
import re
import threading
import time
import queue
import traceback

# =============================================================================
# LIVE FEED MODULE  —  Lightstreamer → price_df row injector
# =============================================================================
#
# Architecture:
#   • A single background thread runs the Lightstreamer client.
#   • Updates arrive in onItemUpdate() and are pushed onto a thread-safe
#     queue.Queue.
#   • On every Streamlit rerun, the main thread drains the queue and writes
#     the latest prices into st.session_state["live_prices"]
#     (dict: contract_code → price float).
#   • inject_live_row(price_df) appends (or overwrites) today's row in the
#     historical price DataFrame so the rest of the app sees live data.
#
# Contract mapping:
#   The Lightstreamer item name is "TT-<InstrumentId>".
#   We resolve InstrumentId → SOFR contract code (Z25, H26 …) via a
#   user-supplied mapping table in the sidebar.
# =============================================================================

_LS_SERVER   = "https://ls-md.corp.hertshtengroup.com/"
_LS_ADAPTER  = "TTsdkLSAdapter"
_LS_DATA_ADT = "HGL1_Adapter"

_LS_FIELDS = [
    "command", "Exchange", "Contract", "Product", "InstrumentId",
    "ClientRecvTime", "ExchangeRecvTime", "ServerRecvTime",
    "Open", "High", "Low", "Close", "Volume",
    "Last", "LastQty", "SeriesStatus",
    "Settle", "PrevSettle",
    "BestAsk", "BestAskQty", "BestBid", "BestBidQty",
    "IndSettle", "Price", "AdminPrice", "Admin", "Direction"
]

# Price field preference order — VWAP first, then fallbacks
_PRICE_FIELD_PRIORITY = ["Settle", "Last", "IndSettle", "Price", "Close"]

# TT prices arrive as integer ticks: 9633 = 96.330, 9675.5 = 96.755
# CME 3-Month SOFR: divide raw tick value by 100 to get xx.xxx
_TT_PRICE_SCALE = 100.0

# Session-state keys
_SS_LIVE_PRICES   = "live_prices"       # dict: contract -> price float (xx.xxx)
_SS_VWAP_STATE    = "vwap_state"        # dict: contract -> {"pv": float, "vol": float}
_SS_VWAP_PRICES   = "vwap_prices"       # dict: contract -> {"vwap": float, "bid": float, "ask": float, "ts": str}
_SS_LS_THREAD     = "ls_thread"
_SS_LS_CLIENT     = "ls_client"
_SS_LS_SUB        = "ls_subscription"
_SS_LS_QUEUE      = "ls_queue"
_SS_LS_CONNECTED  = "ls_connected"
_SS_LS_STATUS_MSG = "ls_status_msg"
_SS_ID_MAP        = "ls_id_map"
_SS_LS_LOG        = "ls_log"
_SS_LS_LOG_Q      = "ls_log_q"          # thread-safe queue for log messages from background thread


def _init_live_session_state():
    """Initialise all live-feed keys in session state (idempotent)."""
    defaults = {
        _SS_LIVE_PRICES:   {},
        _SS_VWAP_STATE:    {},    # VWAP accumulators per contract
        _SS_VWAP_PRICES:   {},    # latest VWAP snapshot with timestamp
        _SS_LS_THREAD:     None,
        _SS_LS_CLIENT:     None,
        _SS_LS_SUB:        None,
        _SS_LS_QUEUE:      queue.Queue(),
        _SS_LS_CONNECTED:  False,
        _SS_LS_STATUS_MSG: "Disconnected",
        _SS_ID_MAP:        dict(_DEFAULT_ID_MAP),
        _SS_LS_LOG:        [],
        _SS_LS_LOG_Q:      queue.Queue(),  # log messages from background thread
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _ls_log(msg: str):
    """
    Append a timestamped log message.
    Safe to call from main thread — drains log_q first then appends.
    Background thread logs are pushed onto _SS_LS_LOG_Q instead.
    """
    ts = datetime.now().strftime("%H:%M:%S")
    log = st.session_state.get(_SS_LS_LOG, [])
    log.append(f"[{ts}] {msg}")
    st.session_state[_SS_LS_LOG] = log[-50:]


def _drain_log_queue():
    """Move messages from the background thread's log queue into the session log list."""
    log_q = st.session_state.get(_SS_LS_LOG_Q)
    if not log_q:
        return
    log = st.session_state.get(_SS_LS_LOG, [])
    while not log_q.empty():
        try:
            log.append(log_q.get_nowait())
        except queue.Empty:
            break
    st.session_state[_SS_LS_LOG] = log[-50:]


def _raw_to_float(raw) -> float:
    """Parse a raw tick string/number to float, return None on failure."""
    if raw is None:
        return None
    try:
        return float(raw)
    except (ValueError, TypeError):
        return None


def _scale_price(raw_price: float) -> float:
    """
    Convert TT raw tick to proper SOFR price (xx.xxx).
    TT sends e.g. 9633 or 9675.5 — divide by 100 → 96.330 / 96.755.
    """
    if raw_price is None:
        return None
    return round(raw_price / _TT_PRICE_SCALE, 5)


def _pick_raw_price(update, fields) -> float:
    """
    Return the best available RAW tick price from a Lightstreamer update.
    VWAP is computed separately; this is used as the fallback non-VWAP price.
    """
    for field in _PRICE_FIELD_PRIORITY:
        if field == "VWAP":
            continue   # VWAP computed internally, not a field in the update
        if field in fields:
            raw = update.getValue(field)
            v = _raw_to_float(raw)
            if v is not None:
                return v
    return None


def _compute_vwap_payload(acc: dict,
                          ask_raw: float, ask_qty: float,
                          bid_raw: float, bid_qty: float,
                          last_raw: float, last_qty: float):
    """
    Pure function — NO session_state access. Safe to call from any thread.

    Computes the VWAP contribution from this update tick using:
        VWAP = (bid_price × ask_qty + ask_price × bid_qty) / (ask_qty + bid_qty)

    'ask_qty' and 'bid_qty' are the OPPOSITE side's quantities, so a large ask
    queue means competition on the buy side — the bid price is weighted higher,
    and vice versa.  Confirmed trades (Last × LastQty) also contribute.

    Returns (new_acc, snapshot_dict) where snapshot_dict contains all display fields.
    Returns (acc, None) if no usable data in this tick.
    """
    has_quote = (ask_raw is not None and ask_qty is not None and ask_qty > 0
                 and bid_raw is not None and bid_qty is not None and bid_qty > 0)
    has_trade = (last_raw is not None and last_qty is not None and last_qty > 0)

    if not has_quote and not has_trade:
        return acc, None

    new_acc = {"pv": acc["pv"], "vol": acc["vol"]}   # copy

    if has_quote:
        new_acc["pv"]  += bid_raw * ask_qty + ask_raw * bid_qty
        new_acc["vol"] += ask_qty + bid_qty

    if has_trade:
        new_acc["pv"]  += last_raw * last_qty
        new_acc["vol"] += last_qty

    if new_acc["vol"] <= 0:
        return new_acc, None

    vwap_scaled = _scale_price(new_acc["pv"] / new_acc["vol"])
    snapshot = {
        "vwap":     vwap_scaled,
        "bid":      _scale_price(bid_raw)  if bid_raw  is not None else None,
        "ask":      _scale_price(ask_raw)  if ask_raw  is not None else None,
        "bid_qty":  bid_qty,
        "ask_qty":  ask_qty,
        "last":     _scale_price(last_raw) if last_raw is not None else None,
        "last_qty": last_qty,
        "ts":       datetime.now().strftime("%H:%M:%S"),
        "acc_vol":  new_acc["vol"],
    }
    return new_acc, snapshot


def _tt_contract_to_sofr_code(tt_code: str) -> str:
    """
    Convert TT-style CME SOFR contract codes to the app contract codes.
    Handles prefixes: SR, SR3 (3-month SOFR), GE, 3SR
    Examples:  SR3Z5 → Z25,  SR3H6 → H26,  SR3M6 → M26,  SRH6 → H26
    Only returns a code for valid SOFR months: Z, H, M, U.
    """
    import re as _re
    # Strip known prefixes including SR3 (3-month SOFR futures)
    m = _re.match(r'^(?:SR3|SR|GE|3SR)?([FGHJKMNQUVXZ])(\d{1,2})$',
                  tt_code.strip().upper())
    if not m:
        return None
    month_letter = m.group(1)
    year_digits  = m.group(2)
    if month_letter not in {'Z', 'H', 'M', 'U'}:
        return None
    # Single digit year: '5' → '25', '6' → '26', etc.
    year_2d = f"2{year_digits}" if len(year_digits) == 1 else year_digits[-2:]
    return f"{month_letter}{year_2d}"


# ── Known instrument ID map (pre-populated, user can extend in sidebar) ──────
# SR3 = CME 3-Month SOFR Futures
_DEFAULT_ID_MAP = {
    "17739299224602468339": "Z25",
    "2264814074172926158":  "H26",
    "2518875037886751798":  "M26",
    "10056698436755136015": "U26",
    "3761391845186607269":  "Z26",
    "8786029629332899618":  "H27",
    "6064266935547558467":  "M27",
    "10582686653072545408": "U27",
    "17925935412019565973": "Z27",
    "3822227243959035490":  "H28",
    "12741923103719175711": "M28",
    "16673841811510079166": "U28",
    "7359239446017790966":  "Z28",
    "4211986922965728750":  "H29",
    "11432813735419224277": "M29",
    "9909662404632894188":  "U29",
    "12350987571259131621": "Z29",
}


def _resolve_contract(update, id_map: dict) -> str:
    """
    Map a Lightstreamer update to a SOFR contract code (e.g. 'H26').
    Tries (in order):
      1. InstrumentId field → id_map lookup
      2. Item name  "TT-<id>" → id_map lookup
      3. Contract field value  (TT CME style 'SRH6') → convert
    """
    inst_id = (update.getValue("InstrumentId") or "").strip()
    if inst_id and inst_id in id_map:
        return id_map[inst_id]

    item_name = update.getItemName()
    raw_id = item_name.replace("TT-", "").strip()
    if raw_id in id_map:
        return id_map[raw_id]

    contract_raw = (update.getValue("Contract") or "").strip()
    if contract_raw:
        return _tt_contract_to_sofr_code(contract_raw)

    return None


class _SubListener:
    """
    Lightstreamer subscription listener — runs on the BACKGROUND THREAD.
    IMPORTANT: Does NOT read or write st.session_state (not thread-safe).
    All raw field values are packed into the queue as a dict.
    The main thread's _drain_queue_to_session() does all accumulation + state writes.
    """

    def __init__(self, q: queue.Queue, id_map_ref: dict, log_q: queue.Queue):
        self._q      = q        # price update queue
        self._id_map = id_map_ref
        self._log_q  = log_q    # separate log queue (avoids session_state writes)

    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        try:
            self._log_q.put_nowait(f"[{ts}] {msg}")
        except Exception:
            pass

    def onItemUpdate(self, update):
        contract = _resolve_contract(update, self._id_map)
        if not contract:
            self._log(f"SKIP {update.getItemName()} — contract not resolved")
            return

        # Pack all raw values — accumulation happens in main thread
        payload = {
            "contract": contract,
            "ask_raw":  _raw_to_float(update.getValue("BestAsk")),
            "ask_qty":  _raw_to_float(update.getValue("BestAskQty")),
            "bid_raw":  _raw_to_float(update.getValue("BestBid")),
            "bid_qty":  _raw_to_float(update.getValue("BestBidQty")),
            "last_raw": _raw_to_float(update.getValue("Last")),
            "last_qty": _raw_to_float(update.getValue("LastQty")),
            # Fallback price fields (scaled when used)
            "settle":   _raw_to_float(update.getValue("Settle")),
            "last_p":   _raw_to_float(update.getValue("Last")),
        }
        self._q.put(payload)

        b = f"{_scale_price(payload['bid_raw']):.3f}" if payload['bid_raw'] else "—"
        a = f"{_scale_price(payload['ask_raw']):.3f}" if payload['ask_raw'] else "—"
        self._log(f"TICK {contract}  bid={b}  ask={a}")

    def onSubscriptionError(self, code, message):
        self._log(f"SUB ERROR {code}: {message}")

    def onEndOfSnapshot(self, item_name, item_pos):
        _ls_log(f"Snapshot complete: {item_name}")


def _run_ls_client(server, adapter, data_adapter, items, fields,
                   id_map_ref, q: queue.Queue):
    """
    Background thread: connects to Lightstreamer and listens for updates.
    Runs until _SS_LS_CONNECTED is set to False (via Disconnect button).
    """
    try:
        from lightstreamer.client import LightstreamerClient, Subscription as LSSub

        st.session_state[_SS_LS_STATUS_MSG] = "Connecting…"
        _ls_log(f"Connecting to {server}")

        ls = LightstreamerClient(server, adapter)
        ls.connect()

        st.session_state[_SS_LS_CLIENT]     = ls
        st.session_state[_SS_LS_CONNECTED]  = True
        st.session_state[_SS_LS_STATUS_MSG] = "Connected ✅"
        _ls_log("Connected")

        sub = LSSub("MERGE", ["TT-" + i for i in items], fields)
        sub.setDataAdapter(data_adapter)
        sub.setRequestedMaxFrequency("0.5")
        sub.setRequestedSnapshot("yes")

        log_q    = st.session_state.get(_SS_LS_LOG_Q, queue.Queue())
        listener = _SubListener(q, id_map_ref, log_q)
        sub.addListener(listener)
        ls.subscribe(sub)

        st.session_state[_SS_LS_SUB]        = sub
        st.session_state[_SS_LS_STATUS_MSG] = "Streaming ✅"
        _ls_log(f"Subscribed — {len(items)} instruments")

        while st.session_state.get(_SS_LS_CONNECTED, False):
            time.sleep(1)

        # Clean disconnect
        try:
            ls.unsubscribe(sub)
            ls.disconnect()
        except Exception:
            pass
        _ls_log("Disconnected cleanly")

    except ImportError:
        msg = ("lightstreamer-client-lib not installed. "
               "Run:  pip install lightstreamer-client-lib")
        _ls_log(f"ERROR: {msg}")
        st.session_state[_SS_LS_STATUS_MSG] = f"⚠️ {msg}"
        st.session_state[_SS_LS_CONNECTED]  = False

    except Exception as e:
        _ls_log(f"ERROR: {e}\n{traceback.format_exc()}")
        st.session_state[_SS_LS_STATUS_MSG] = f"Error: {e}"
        st.session_state[_SS_LS_CONNECTED]  = False


def _drain_queue_to_session(q: queue.Queue) -> bool:
    """
    Main-thread only. Drains all raw update payloads from the queue,
    accumulates VWAP, and writes results to session_state.

    Each payload is a dict with raw field values from onItemUpdate.
    All session_state writes happen here — never in the background thread.
    """
    _drain_log_queue()   # pull log messages from background thread first

    updated = False
    vwap_state  = st.session_state.get(_SS_VWAP_STATE,  {})
    vwap_prices = st.session_state.get(_SS_VWAP_PRICES, {})
    live_prices = st.session_state.get(_SS_LIVE_PRICES, {})
    src_map     = st.session_state.get("live_prices_source", {})

    while not q.empty():
        try:
            payload = q.get_nowait()
        except queue.Empty:
            break

        contract = payload.get("contract")
        if not contract:
            continue

        # Retrieve existing accumulator for this contract
        acc = vwap_state.get(contract, {"pv": 0.0, "vol": 0.0})

        # Compute VWAP (pure function — no session_state access)
        new_acc, snapshot = _compute_vwap_payload(
            acc,
            payload.get("ask_raw"), payload.get("ask_qty"),
            payload.get("bid_raw"), payload.get("bid_qty"),
            payload.get("last_raw"), payload.get("last_qty"),
        )
        vwap_state[contract] = new_acc

        if snapshot is not None:
            # VWAP available — store snapshot and use as the live price
            vwap_prices[contract] = snapshot
            live_prices[contract] = snapshot["vwap"]
            src_map[contract]     = "VWAP"
        else:
            # Fallback: use Settle then Last (already scaled by sender)
            if src_map.get(contract) == "VWAP":
                pass   # keep existing VWAP price — don't overwrite with fallback
            else:
                for raw_field in ["settle", "last_p"]:
                    raw = payload.get(raw_field)
                    if raw is not None:
                        live_prices[contract] = _scale_price(raw)
                        src_map[contract]     = "Fallback"
                        break

        updated = True

    # Flush all writes to session_state in one pass
    st.session_state[_SS_VWAP_STATE]        = vwap_state
    st.session_state[_SS_VWAP_PRICES]       = vwap_prices
    st.session_state[_SS_LIVE_PRICES]       = live_prices
    st.session_state["live_prices_source"]  = src_map

    return updated


def inject_live_row(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Inject today's live prices as the most-recent row in price_df.
    If a row for today already exists it is overwritten.
    Only contracts that appear as columns in price_df are updated —
    no new columns are added.
    """
    live = st.session_state.get(_SS_LIVE_PRICES, {})
    if not live:
        return price_df

    today = pd.Timestamp(date.today())
    known_contracts = price_df.columns.tolist()

    row_data = {c: live.get(c, np.nan) for c in known_contracts}
    new_row  = pd.Series(row_data, name=today)

    if new_row.notna().any():
        price_df = price_df.copy()
        if today in price_df.index:
            price_df.loc[today] = new_row
        else:
            price_df = pd.concat([price_df, new_row.to_frame().T])
        price_df.index = pd.to_datetime(price_df.index)
        price_df = price_df.sort_index()

    return price_df


def render_live_feed_sidebar(expiry_df: pd.DataFrame = None) -> bool:
    """
    Render the Live Feed section in the Streamlit sidebar.
    Call this after loading the expiry file.
    Returns True if at least one live price is available.
    """
    _init_live_session_state()

    st.sidebar.markdown("---")
    st.sidebar.header("🔴 Live Feed (Lightstreamer)")

    # ── Resolve connection state via thread.is_alive() ────────────────────
    thread = st.session_state.get(_SS_LS_THREAD)
    thread_alive = thread is not None and thread.is_alive()
    if thread_alive and not st.session_state.get(_SS_LS_CONNECTED):
        st.session_state[_SS_LS_CONNECTED] = True
    elif not thread_alive and st.session_state.get(_SS_LS_CONNECTED):
        st.session_state[_SS_LS_CONNECTED] = False
        st.session_state[_SS_LS_STATUS_MSG] = "Disconnected"

    is_connected = thread_alive
    _drain_log_queue()   # pull log messages even before price queue is drained
    live_prices  = st.session_state.get(_SS_LIVE_PRICES, {})
    n_live = sum(1 for v in live_prices.values()
                 if v is not None and not (isinstance(v, float) and np.isnan(v)))

    # ── Status banner ─────────────────────────────────────────────────────
    if is_connected:
        src_map = st.session_state.get("live_prices_source", {})
        n_vwap  = sum(1 for c,s in src_map.items() if s == "VWAP" and c in live_prices)
        st.sidebar.success(f"● Streaming — {n_live} contracts  ({n_vwap} VWAP)")
    else:
        st.sidebar.info(st.session_state.get(_SS_LS_STATUS_MSG, "Disconnected"))

    # ── Instrument ID mapping ─────────────────────────────────────────────
    st.sidebar.markdown("**Instrument ID → Contract mapping**")
    st.sidebar.caption("Pre-populated with all 17 SR3 SOFR contracts.")
    current_map  = st.session_state.get(_SS_ID_MAP, dict(_DEFAULT_ID_MAP)) or dict(_DEFAULT_ID_MAP)
    default_text = "\n".join(f"{k},{v}" for k, v in sorted(current_map.items(), key=lambda x: x[1]))
    id_map_text  = st.sidebar.text_area("ID Mapping (InstrumentId,Code)",
                                        value=default_text, height=150,
                                        key="ls_id_map_input")
    parsed_map = {}
    for line in id_map_text.strip().splitlines():
        if "," in line:
            parts = line.split(",", 1)
            if len(parts) == 2:
                iid, code = parts[0].strip(), parts[1].strip().upper()
                if iid and code:
                    parsed_map[iid] = code
    if not parsed_map:
        parsed_map = dict(_DEFAULT_ID_MAP)
    st.session_state[_SS_ID_MAP] = parsed_map

    all_mapped_ids   = list(parsed_map.keys())
    default_ids_text = "\n".join(all_mapped_ids)
    with st.sidebar.expander("Edit subscribed IDs (advanced)", expanded=False):
        st.caption("All mapped IDs are subscribed by default.")
        subscribed_text = st.text_area("Subscribed IDs", value=default_ids_text,
                                       height=200, key="ls_subscribed_ids")
    subscribed_ids = [s.strip() for s in
                      st.session_state.get("ls_subscribed_ids", default_ids_text).splitlines()
                      if s.strip()]

    # ── Fallback price field ──────────────────────────────────────────────
    price_field_choice = st.sidebar.selectbox(
        "Fallback price field (when no VWAP yet)",
        options=["Settle", "Last", "IndSettle", "Price", "Close"],
        index=0, key="ls_price_field",
        help="VWAP is computed automatically from bid/ask. This is the fallback.")
    _PRICE_FIELD_PRIORITY.clear()
    _PRICE_FIELD_PRIORITY.extend(
        [price_field_choice] + [f for f in ["Settle","Last","IndSettle","Price","Close"]
                                if f != price_field_choice])

    # ── Connect / Disconnect / Refresh row ───────────────────────────────
    c1, c2, c3 = st.sidebar.columns(3)
    with c1:
        if st.button("▶ Connect", key="ls_connect_btn", disabled=is_connected):
            if not subscribed_ids:
                st.sidebar.warning("Add at least one Instrument ID first.")
            else:
                st.session_state[_SS_LS_CONNECTED]  = True
                st.session_state[_SS_LS_STATUS_MSG] = "Connecting…"
                t = threading.Thread(
                    target=_run_ls_client,
                    args=(_LS_SERVER, _LS_ADAPTER, _LS_DATA_ADT,
                          subscribed_ids, _LS_FIELDS,
                          st.session_state[_SS_ID_MAP],
                          st.session_state[_SS_LS_QUEUE]),
                    daemon=True, name="LightstreamerFeedThread")
                st.session_state[_SS_LS_THREAD] = t
                t.start()
                st.rerun()
    with c2:
        if st.button("⏹ Stop", key="ls_disconnect_btn", disabled=not is_connected):
            st.session_state[_SS_LS_CONNECTED]  = False
            st.session_state[_SS_LS_STATUS_MSG] = "Disconnected"
            st.rerun()
    with c3:
        if st.button("🔃 Refresh", key="ls_refresh_btn",
                     help="Drain queue and refresh prices now"):
            q = st.session_state.get(_SS_LS_QUEUE)
            if q:
                _drain_queue_to_session(q)
            st.rerun()

    # ── Auto-refresh interval slider ──────────────────────────────────────
    auto_refresh = st.sidebar.checkbox("Auto-refresh", value=False, key="ls_auto_refresh")
    refresh_secs = st.sidebar.slider("Refresh interval (seconds)", min_value=5,
                                     max_value=300, value=15, step=5,
                                     key="ls_refresh_interval",
                                     disabled=not auto_refresh)

    # ── VWAP reset ────────────────────────────────────────────────────────
    if st.sidebar.button("🔄 Reset VWAP", key="ls_vwap_reset",
                         help="Clear intraday VWAP — use at session start"):
        st.session_state[_SS_VWAP_STATE]       = {}
        st.session_state[_SS_VWAP_PRICES]      = {}
        st.session_state["live_prices_source"] = {}
        st.session_state[_SS_LIVE_PRICES]      = {}
        _ls_log("VWAP reset by user")
        st.rerun()

    # ── Drain queue ───────────────────────────────────────────────────────
    q = st.session_state.get(_SS_LS_QUEUE)
    if q:
        _drain_queue_to_session(q)
    # Re-read after drain
    live_prices = st.session_state.get(_SS_LIVE_PRICES, {})
    n_live = sum(1 for v in live_prices.values()
                 if v is not None and not (isinstance(v, float) and np.isnan(v)))

    # ── VWAP live table (sidebar) ─────────────────────────────────────────
    if n_live > 0:
        src_map    = st.session_state.get("live_prices_source", {})
        vwap_snap  = st.session_state.get(_SS_VWAP_PRICES, {})
        with st.sidebar.expander("📊 Live VWAP prices", expanded=True):
            rows = []
            for contract in sorted(live_prices.keys()):
                price  = live_prices[contract]
                source = src_map.get(contract, "Fallback")
                snap   = vwap_snap.get(contract, {})
                rows.append({
                    "Contract": contract,
                    "Price":    f"{price:.3f}" if price is not None else "—",
                    "Source":   "VWAP" if source == "VWAP" else "Fallback",
                    "Bid":      f"{snap['bid']:.3f}" if snap.get("bid") else "—",
                    "Ask":      f"{snap['ask']:.3f}" if snap.get("ask") else "—",
                    "Updated":  snap.get("ts", "—"),
                })
            bbg_table(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.sidebar.caption("No live prices yet — connect and wait for updates.")

    # ── Feed log ──────────────────────────────────────────────────────────
    with st.sidebar.expander("Feed log", expanded=False):
        log = st.session_state.get(_SS_LS_LOG, [])
        st.text("\n".join(reversed(log)) if log else "No log entries yet.")

    # ── Auto-refresh trigger ──────────────────────────────────────────────
    if auto_refresh and is_connected:
        time.sleep(refresh_secs)
        st.rerun()

    return n_live > 0


# =============================================================================
# END LIVE FEED MODULE
# =============================================================================

# --- Configuration --- (FIXED: must be the very first Streamlit call)
st.set_page_config(layout="wide", page_title="SOFR Futures PCA Analyzer")

# ── Bloomberg Terminal UI Theme ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600;700&display=swap');

/* ═══════════════════════════════════════════════════════
   BLOOMBERG TERMINAL THEME — Streamlit
   KEY RULE: Never use catch-all selectors (* or div/span)
   that would break canvas-based dataframe rendering.
   ═══════════════════════════════════════════════════════ */

/* ── Backgrounds ── */
html, body,
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
.block-container {
    background-color: #0a0a0a !important;
}

[data-testid="stSidebar"],
[data-testid="stSidebarContent"],
section[data-testid="stSidebar"] > div {
    background-color: #060606 !important;
    border-right: 2px solid #ff6600 !important;
}

/* ── Font — only on known safe containers, NOT * ── */
.stApp,
.stMarkdown,
.stButton,
.stSelectbox,
.stMultiSelect,
.stTextInput,
.stNumberInput,
.stTextArea,
.stSlider,
.stCheckbox,
.stRadio,
.stFileUploader,
[data-testid="stMetric"],
[data-testid="stAlert"],
[data-testid="stExpander"],
[data-testid="stSidebar"] {
    font-family: 'IBM Plex Mono', 'Courier New', monospace !important;
}

/* ── Headings ── */
h1 {
    font-family: 'IBM Plex Mono', monospace !important;
    color: #ff6600 !important;
    font-size: 1.0rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    border-bottom: 2px solid #ff6600 !important;
    padding-bottom: 8px !important;
    margin-bottom: 1.2rem !important;
    background: transparent !important;
}
h2 {
    font-family: 'IBM Plex Mono', monospace !important;
    color: #ffaa00 !important;
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    background: linear-gradient(90deg, #1c1000 0%, #0a0a0a 80%) !important;
    padding: 5px 10px !important;
    border-left: 3px solid #ff6600 !important;
    margin-top: 1.4rem !important;
    margin-bottom: 0.5rem !important;
}
h3 {
    font-family: 'IBM Plex Mono', monospace !important;
    color: #00cccc !important;
    font-size: 0.68rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid #1e1e1e !important;
    padding-bottom: 3px !important;
    background: transparent !important;
}

/* ── Body text — scoped to Streamlit markdown only ── */
.stMarkdown p,
.stMarkdown li,
.stMarkdown a {
    color: #cccccc !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.77rem !important;
}
.stMarkdown strong, .stMarkdown b { color: #ffaa00 !important; }
.stMarkdown code {
    background-color: #111111 !important;
    color: #00cccc !important;
    border-radius: 0 !important;
    font-size: 0.72rem !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background-color: #111111 !important;
    border: 1px solid #222222 !important;
    border-top: 2px solid #ff6600 !important;
    padding: 10px 14px !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'IBM Plex Mono', monospace !important;
}
[data-testid="stMetricLabel"] p {
    color: #666666 !important;
    font-size: 0.57rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
}
[data-testid="stMetricValue"] div {
    color: #ffaa00 !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

/* ── Buttons ── */
.stButton > button {
    background-color: #111111 !important;
    color: #ff6600 !important;
    border: 1px solid #ff6600 !important;
    border-radius: 0 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.65rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 5px 14px !important;
    transition: background 0.12s, color 0.12s !important;
}
.stButton > button:hover {
    background-color: #ff6600 !important;
    color: #000000 !important;
}

/* ── Inputs ── */
div[data-baseweb="input"] > div,
div[data-baseweb="base-input"] > div {
    background-color: #111111 !important;
    border-color: #2a2a2a !important;
    border-radius: 0 !important;
}
div[data-baseweb="input"] input,
div[data-baseweb="base-input"] input {
    background-color: #111111 !important;
    color: #e0e0e0 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.74rem !important;
    caret-color: #ff6600 !important;
}
div[data-baseweb="textarea"] textarea {
    background-color: #111111 !important;
    color: #e0e0e0 !important;
    border-color: #2a2a2a !important;
    border-radius: 0 !important;
}

/* ── Selects ── */
div[data-baseweb="select"] > div {
    background-color: #111111 !important;
    border-color: #2a2a2a !important;
    border-radius: 0 !important;
    color: #e0e0e0 !important;
}
div[data-baseweb="select"] span {
    color: #e0e0e0 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.74rem !important;
}
div[data-baseweb="popover"] {
    background-color: #111111 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 0 !important;
}
div[data-baseweb="menu"] ul li {
    background-color: #111111 !important;
    color: #e0e0e0 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.73rem !important;
}
div[data-baseweb="menu"] ul li:hover {
    background-color: #1c1c1c !important;
}

/* ── Slider ── */
[data-testid="stSlider"] [role="slider"] {
    background-color: #ff6600 !important;
    border-color: #ff6600 !important;
}
[data-testid="stSlider"] [data-testid="stTickBar"] {
    color: #555555 !important;
}

/* ── Checkbox / Radio ── */
input[type="checkbox"], input[type="radio"] {
    accent-color: #ff6600 !important;
}
[data-testid="stCheckbox"] label p,
[data-testid="stRadio"] label p {
    color: #cccccc !important;
    font-size: 0.73rem !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background-color: #0f0f0f !important;
    border: 1px solid #1e1e1e !important;
    border-radius: 0 !important;
}
[data-testid="stExpander"] summary {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.67rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: #ffaa00 !important;
    background-color: #0f0f0f !important;
}
[data-testid="stExpander"] summary:hover { color: #ff6600 !important; }
[data-testid="stExpander"][open] { border-top: 2px solid #ff6600 !important; }

/* ── Alerts ── */
[data-testid="stAlert"] {
    border-radius: 0 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.7rem !important;
}
[data-testid="stSuccessAlert"],
div[data-testid="stAlert"].st-success {
    background-color: #001a08 !important;
    border-left: 3px solid #00cc44 !important;
    color: #00cc44 !important;
}
[data-testid="stWarningAlert"],
div[data-testid="stAlert"].st-warning {
    background-color: #1a0f00 !important;
    border-left: 3px solid #ffaa00 !important;
    color: #ffaa00 !important;
}
[data-testid="stErrorAlert"],
div[data-testid="stAlert"].st-error {
    background-color: #1a0000 !important;
    border-left: 3px solid #ff3333 !important;
    color: #ff3333 !important;
}
[data-testid="stInfoAlert"],
div[data-testid="stAlert"].st-info {
    background-color: #001020 !important;
    border-left: 3px solid #3399ff !important;
    color: #3399ff !important;
}

/* ── Captions ── */
[data-testid="stCaptionContainer"] p {
    color: #555555 !important;
    font-size: 0.62rem !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

/* ── Dataframe wrapper only — do NOT style internals ── */
[data-testid="stDataFrame"] {
    border: 1px solid #1e1e1e !important;
    border-radius: 0 !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background-color: #060606 !important;
    border-bottom: 1px solid #ff6600 !important;
    gap: 0 !important;
}
[data-testid="stTabs"] button[role="tab"] {
    background-color: #060606 !important;
    color: #555555 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.63rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    border-radius: 0 !important;
    padding: 6px 16px !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    background-color: #ff6600 !important;
    color: #000000 !important;
    font-weight: 700 !important;
    border-bottom: 2px solid #ff6600 !important;
}

/* ── File uploader ── */
[data-testid="stFileUploaderDropzone"] {
    background-color: #111111 !important;
    border: 1px dashed #2a2a2a !important;
    border-radius: 0 !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #ff6600 !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] p {
    color: #555555 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.7rem !important;
}

/* ── Sidebar internals ── */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #ff6600 !important;
    font-size: 0.65rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    border-bottom: 1px solid #1e1e1e !important;
    background: transparent !important;
    padding-bottom: 3px !important;
    margin-top: 0.8rem !important;
}
[data-testid="stSidebar"] .stMarkdown p {
    color: #aaaaaa !important;
    font-size: 0.7rem !important;
}
[data-testid="stSidebar"] hr { border-color: #1a1a1a !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #060606; }
::-webkit-scrollbar-thumb { background: #ff6600; border-radius: 0; }

/* ── Horizontal rule ── */
hr { border-color: #1e1e1e !important; }
</style>
""", unsafe_allow_html=True)



# ── Bloomberg Table Helpers ───────────────────────────────────────────────────
_BBG_TABLE_CSS = """
<style>
.bbg-table-wrap {
    width: 100%;
    overflow-x: auto;
    margin-bottom: 0.8rem;
}
.bbg-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'IBM Plex Mono', 'Courier New', monospace;
    font-size: 0.72rem;
    background-color: #0a0a0a;
    border: 1px solid #ff6600;
}
.bbg-table thead tr th {
    background-color: #1a0800;
    color: #ff6600;
    font-size: 0.63rem;
    font-weight: 700;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    border-bottom: 2px solid #ff6600;
    border-right: 1px solid #2a2a2a;
    padding: 6px 12px;
    text-align: right;
    white-space: nowrap;
}
.bbg-table thead tr th:first-child {
    text-align: left;
    color: #ffaa00;
}
.bbg-table tbody tr td {
    color: #e0e0e0;
    border-bottom: 1px solid #1a1a1a;
    border-right: 1px solid #1a1a1a;
    padding: 5px 12px;
    text-align: right;
    background-color: #0a0a0a;
    white-space: nowrap;
}
.bbg-table tbody tr td:first-child {
    color: #ffaa00;
    font-weight: 600;
    text-align: left;
    background-color: #0d0d0d;
    border-right: 1px solid #2a2a2a;
}
.bbg-table tbody tr:nth-child(even) td {
    background-color: #0f0f0f;
}
.bbg-table tbody tr:nth-child(even) td:first-child {
    background-color: #111111;
}
.bbg-table tbody tr:hover td {
    background-color: #1a0800 !important;
    color: #ffaa00 !important;
}
/* colour helpers applied inline */
.bbg-pos { color: #00cc44 !important; font-weight: 600; }
.bbg-neg { color: #ff3333 !important; font-weight: 600; }
.bbg-hi  { color: #ffaa00 !important; font-weight: 600; }
</style>
"""
_bbg_css_done = False


def _inject_bbg_table_css():
    global _bbg_css_done
    if not _bbg_css_done:
        st.markdown(_BBG_TABLE_CSS, unsafe_allow_html=True)
        _bbg_css_done = True


def _cell_class(val):
    """Auto-colour numeric cells: green if positive, red if negative."""
    try:
        v = float(str(val).replace(",", "").replace("%", "").strip())
        if v > 0:
            return 'bbg-pos'
        if v < 0:
            return 'bbg-neg'
    except (ValueError, TypeError):
        pass
    return ""


def _df_to_bbg_html(df, fmt=None, num_cols=None, colour_values=True, hide_index=False):
    """
    Convert a DataFrame (or Styler) to a Bloomberg-styled HTML table string.

    Args:
        df          : pd.DataFrame or pd.io.formats.style.Styler
        fmt         : dict {col: format_string} e.g. {'Price': '{:.4f}'}
        num_cols    : list of columns to right-align (auto-detected if None)
        colour_values: auto-colour positive/negative numbers
        hide_index  : omit the index column
    """
    import pandas as pd

    # Unwrap Styler → get the underlying DataFrame + any format dict
    if hasattr(df, 'data'):          # it's a Styler
        raw_df = df.data.copy()
    else:
        raw_df = df.copy()

    # Apply explicit format dict
    display_df = raw_df.copy().astype(object)
    if fmt:
        for col, f in fmt.items():
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda x: f.format(x) if pd.notna(x) else "—"
                )

    # Auto-format remaining float columns to 4 dp
    for col in display_df.columns:
        if col not in (fmt or {}):
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:.4f}" if isinstance(x, float) else (str(x) if pd.notna(x) else "—")
            )

    rows_html = []

    # Header
    index_th = "" if hide_index else f'<th>{raw_df.index.name or ""}</th>'
    cols_html = "".join(f"<th>{c}</th>" for c in display_df.columns)
    rows_html.append(f"<thead><tr>{index_th}{cols_html}</tr></thead>")

    # Body
    body_rows = []
    for idx, row in display_df.iterrows():
        idx_td = "" if hide_index else f'<td>{idx}</td>'
        cells = []
        for col, val in row.items():
            cls = _cell_class(val) if colour_values else ""
            cls_attr = f' class="{cls}"' if cls else ""
            cells.append(f"<td{cls_attr}>{val}</td>")
        body_rows.append(f"<tr>{idx_td}{''.join(cells)}</tr>")
    rows_html.append(f"<tbody>{''.join(body_rows)}</tbody>")

    return f'<div class="bbg-table-wrap"><table class="bbg-table">{"".join(rows_html)}</table></div>'


def bbg_table(df_or_styler, fmt=None, colour_values=True,
              hide_index=False, use_container_width=True, **kwargs):
    """
    Bloomberg HTML table — replaces st.dataframe().
    Extra st.dataframe kwargs (height, column_config, etc.) are silently ignored
    because we render via st.markdown instead of the canvas grid.
    """
    _inject_bbg_table_css()
    import pandas as pd
    # Extract format dict from Styler if present
    if hasattr(df_or_styler, '_display_funcs'):
        try:
            extracted = {
                col: df_or_styler._display_funcs.get((0, i), None)
                for i, col in enumerate(df_or_styler.data.columns)
            }
            if fmt is None:
                fmt = {}
        except Exception:
            pass
    html = _df_to_bbg_html(df_or_styler, fmt=fmt,
                            colour_values=colour_values,
                            hide_index=hide_index)
    st.markdown(html, unsafe_allow_html=True)


def bbg_st_table(df, **kwargs):
    """Bloomberg HTML table — replaces st.table()."""
    _inject_bbg_table_css()
    html = _df_to_bbg_html(df, colour_values=True)
    st.markdown(html, unsafe_allow_html=True)
# ── End Bloomberg Table Helpers ───────────────────────────────────────────────


# ── Bloomberg matplotlib/seaborn theme ────────────────────────────────────────
import matplotlib as _mpl
_BBG_BLACK  = "#000000"
_BBG_BG     = "#0d0d0d"
_BBG_PANEL  = "#111111"
_BBG_ORANGE = "#ff6600"
_BBG_AMBER  = "#ffaa00"
_BBG_WHITE  = "#e8e8e8"
_BBG_GRAY   = "#aaaaaa"
_BBG_GREEN  = "#00cc44"
_BBG_RED    = "#ff3333"
_BBG_BLUE   = "#3399ff"
_BBG_CYAN   = "#00cccc"
_BBG_GRID   = "#1e1e1e"
_BBG_CYCLE  = [_BBG_ORANGE, _BBG_CYAN, _BBG_GREEN, _BBG_AMBER, "#cc44ff", _BBG_BLUE, "#ff66aa", _BBG_RED]

_mpl.rcParams.update({
    "figure.facecolor":      _BBG_BG,
    "axes.facecolor":        _BBG_PANEL,
    "axes.edgecolor":        "#2a2a2a",
    "axes.labelcolor":       "#bbbbbb",
    "axes.titlecolor":       _BBG_AMBER,
    "axes.grid":             True,
    "axes.prop_cycle":       _mpl.cycler(color=_BBG_CYCLE),
    "axes.spines.top":       False,
    "axes.spines.right":     False,
    "grid.color":            _BBG_GRID,
    "grid.linewidth":        0.5,
    "grid.linestyle":        ":",
    "text.color":            _BBG_WHITE,
    "xtick.color":           "#bbbbbb",
    "ytick.color":           "#bbbbbb",
    "xtick.labelsize":       7,
    "ytick.labelsize":       7,
    "axes.labelsize":        8,
    "axes.titlesize":        9,
    "axes.titleweight":      "bold",
    "legend.facecolor":      "#0d0d0d",
    "legend.edgecolor":      "#2a2a2a",
    "legend.labelcolor":     _BBG_WHITE,
    "legend.fontsize":       7,
    "font.family":           "monospace",
    "figure.autolayout":     True,
    "savefig.facecolor":     _BBG_BG,
    "savefig.edgecolor":     _BBG_BG,
    "lines.linewidth":       1.4,
    "patch.edgecolor":       _BBG_BG,
})

def _bbg_fig(ax=None, fig=None):
    """Apply Bloomberg finishing touches to a fig/ax after drawing."""
    if fig is None and ax is not None:
        fig = ax.get_figure()
    if fig is None:
        return
    for _ax in fig.axes:
        _ax.set_facecolor(_BBG_PANEL)
        _ax.tick_params(colors="#cccccc", labelsize=7, which="both")
        for spine in _ax.spines.values():
            spine.set_edgecolor("#2a2a2a")
        if _ax.get_legend() is not None:
            _ax.get_legend().get_frame().set_facecolor("#0d0d0d")
            _ax.get_legend().get_frame().set_edgecolor("#2a2a2a")
    fig.patch.set_facecolor(_BBG_BG)
    # Force bright tick labels
    for _ax in fig.axes:
        for lbl in _ax.get_xticklabels() + _ax.get_yticklabels():
            lbl.set_color("#cccccc")
            lbl.set_fontsize(7)
        _ax.xaxis.label.set_color("#bbbbbb")
        _ax.yaxis.label.set_color("#bbbbbb")
        _ax.title.set_color(_BBG_AMBER)
# ── End Bloomberg Theme ───────────────────────────────────────────────────────

# --- PDF figure collections ---
if "SECTION5_FIGURES" not in st.session_state:
    st.session_state.SECTION5_FIGURES = []

if "SECTION9_FIGURES" not in st.session_state:
    st.session_state.SECTION9_FIGURES = []
if "SNAPSHOT_READY" not in st.session_state:
    st.session_state.SNAPSHOT_READY = False

# --- Helper Functions for Data Processing ---

# =============================================================================
# HARDCODED SR3 SOFR EXPIRY DATES (from sofr_expiry.csv — CME official calendar)
# =============================================================================
_SR3_EXPIRY_MAP = {
    # CME SR3 Last Trading Day = Tuesday before 3rd Wednesday of the NEXT quarterly month
    # Z(Dec) -> expires Mar+1yr | H(Mar) -> expires Jun | M(Jun) -> expires Sep | U(Sep) -> expires Dec
    # Z25 started Dec 2025, expires Mar 17 2026 (still active as of Mar 2026)
    "Z20": "2021-03-16",  # Tue Mar 16 2021
    "H21": "2021-06-15",  # Tue Jun 15 2021
    "M21": "2021-09-14",  # Tue Sep 14 2021
    "U21": "2021-12-14",  # Tue Dec 14 2021
    "Z21": "2022-03-15",  # Tue Mar 15 2022
    "H22": "2022-06-14",  # Tue Jun 14 2022
    "M22": "2022-09-20",  # Tue Sep 20 2022
    "U22": "2022-12-20",  # Tue Dec 20 2022
    "Z22": "2023-03-14",  # Tue Mar 14 2023
    "H23": "2023-06-20",  # Tue Jun 20 2023
    "M23": "2023-09-19",  # Tue Sep 19 2023
    "U23": "2023-12-19",  # Tue Dec 19 2023
    "Z23": "2024-03-19",  # Tue Mar 19 2024
    "H24": "2024-06-18",  # Tue Jun 18 2024
    "M24": "2024-09-17",  # Tue Sep 17 2024
    "U24": "2024-12-17",  # Tue Dec 17 2024
    "Z24": "2025-03-18",  # Tue Mar 18 2025
    "H25": "2025-06-17",  # Tue Jun 17 2025
    "M25": "2025-09-16",  # Tue Sep 16 2025
    "U25": "2025-12-16",  # Tue Dec 16 2025
    "Z25": "2026-03-17",  # Tue Mar 17 2026  ← STILL ACTIVE TODAY
    "H26": "2026-06-16",  # Tue Jun 16 2026
    "M26": "2026-09-15",  # Tue Sep 15 2026
    "U26": "2026-12-15",  # Tue Dec 15 2026
    "Z26": "2027-03-16",  # Tue Mar 16 2027
    "H27": "2027-06-15",  # Tue Jun 15 2027
    "M27": "2027-09-14",  # Tue Sep 14 2027
    "U27": "2027-12-14",  # Tue Dec 14 2027
    "Z27": "2028-03-14",  # Tue Mar 14 2028
    "H28": "2028-06-20",  # Tue Jun 20 2028
    "M28": "2028-09-19",  # Tue Sep 19 2028
    "U28": "2028-12-19",  # Tue Dec 19 2028
    "Z28": "2029-03-20",  # Tue Mar 20 2029
    "H29": "2029-06-19",  # Tue Jun 19 2029
    "M29": "2029-09-18",  # Tue Sep 18 2029
    "U29": "2029-12-18",  # Tue Dec 18 2029
    "Z29": "2030-03-19",  # Tue Mar 19 2030
}

def get_hardcoded_expiry_df() -> pd.DataFrame:
    """Returns the SR3 SOFR expiry DataFrame (hardcoded — no file upload needed)."""
    records = [{"Contract": code, "ExpiryDate": pd.to_datetime(dt)}
               for code, dt in _SR3_EXPIRY_MAP.items()]
    df = pd.DataFrame(records).set_index("Contract")
    df.index.name = "Contract"
    return df
# =============================================================================

# Use st.cache_data for performance as file loading is idempotent
@st.cache_data
def load_data(uploaded_file):
    """Loads CSV data into a DataFrame, adapting to price or expiry file formats."""
    if uploaded_file is None:
        return None
        
    try:
        # Read the uploaded file content to inspect the header for format identification
        uploaded_file.seek(0)
        file_content = uploaded_file.getvalue().decode("utf-8")
        uploaded_file.seek(0)
            
        # --- Case 1: Expiry File (MATURITY, DATE) ---
        if 'MATURITY,DATE' in file_content.split('\n')[0].upper():
            df = pd.read_csv(uploaded_file, sep=',')
            df = df.rename(columns={'MATURITY': 'Contract', 'DATE': 'ExpiryDate'})
            df = df.set_index('Contract')
            df['ExpiryDate'] = pd.to_datetime(df['ExpiryDate'])
            df.index.name = 'Contract'
            return df

        # --- Case 2: Price File (Date as index) ---
        df = pd.read_csv(
            uploaded_file, 
            index_col=0, 
            parse_dates=True,
            sep=',', 
            header=0 
        )
        
        df.index.name = 'Date'
        df = df.dropna(axis=1, how='all')
        
        for col in df.columns:
            # Ensure price columns are numeric
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        df = df.dropna(how='all')
        df = df[df.index.notna()]

        if df.empty or df.shape[1] == 0:
             raise ValueError("DataFrame is empty after processing or has no data columns.")
             
        return df
        
    except Exception as e:
        st.error(f"Error loading and processing data from {uploaded_file.name}: {e}")
        return None


@st.cache_data
def get_analysis_contracts(expiry_df, analysis_date):
    """Filters contract codes that expire on or after the analysis date."""
    if expiry_df is None:
        return pd.DataFrame()
    # FIXED: ensure analysis_date is a datetime for consistent comparison with ExpiryDate (which is datetime)
    if isinstance(analysis_date, date) and not isinstance(analysis_date, datetime):
        analysis_date = datetime.combine(analysis_date, datetime.min.time())
    future_expiries = expiry_df[expiry_df['ExpiryDate'] >= analysis_date].copy()
    future_expiries = future_expiries.sort_values(by='ExpiryDate')
    
    if future_expiries.empty:
        st.warning(f"No contracts found expiring on or after {analysis_date.strftime('%Y-%m-%d')}.")
    
    return future_expiries

@st.cache_data
def transform_to_analysis_curve(price_df, future_expiries_df):
    """Selects and orders historical prices for relevant contracts.
    Contracts with no historical data are excluded from PCA to avoid wiping training history.
    Their live prices remain visible in the price source log.
    """
    if price_df is None or future_expiries_df.empty:
        return pd.DataFrame(), []
    contract_order = future_expiries_df.index.tolist()
    valid_contracts = [c for c in contract_order if c in price_df.columns]
    if not valid_contracts:
        st.warning("No matching contract columns found in price data for the selected analysis date range.")
        return pd.DataFrame(), []
    today = pd.Timestamp(date.today())
    historical_mask = price_df.index < today
    has_history = [c for c in valid_contracts if price_df.loc[historical_mask, c].notna().any()]
    excluded = [c for c in valid_contracts if c not in has_history]
    if excluded:
        st.info(f"ℹ️ **{', '.join(excluded)}** excluded from PCA (no historical data). Live prices shown in price log.")
    if not has_history:
        st.warning("No contracts with historical data found.")
        return pd.DataFrame(), []
    return price_df[has_history], has_history


# --- GENERALIZED DERIVATIVE CALCULATION FUNCTIONS (k-step) ---

@st.cache_data
def calculate_k_step_spreads(analysis_curve_df, k):
    """
    Calculates spreads between contracts separated by 'k' steps (e.g., k=1 for 3M, k=2 for 6M, k=4 for 12M).
    CME Basis: C_i - C_{i+k}
    """
    if analysis_curve_df.empty or analysis_curve_df.shape[1] < k + 1:
        return pd.DataFrame()

    num_contracts = analysis_curve_df.shape[1]
    spreads_data = {}
    
    for i in range(num_contracts - k):
        short_maturity = analysis_curve_df.columns[i]
        long_maturity = analysis_curve_df.columns[i+k]
        
        spread_label = f"{short_maturity}-{long_maturity}"
        # Spread = C_i - C_{i+k}
        spreads_data[spread_label] = analysis_curve_df.iloc[:, i] - analysis_curve_df.iloc[:, i+k]
        
    return pd.DataFrame(spreads_data)

@st.cache_data
def calculate_k_step_butterflies(analysis_curve_df, k):
    """
    Calculates butterflies using contracts separated by 'k' steps (e.g., k=1 for 3M fly, k=2 for 6M fly, k=4 for 12M fly).
    Formula: C_i - 2 * C_{i+k} + C_{i+2k}
    Label Format: C_i-2xC_{i+k}+C_{i+2k}
    """
    if analysis_curve_df.empty or analysis_curve_df.shape[1] < 2 * k + 1:
        return pd.DataFrame()

    num_contracts = analysis_curve_df.shape[1]
    flies_data = {}

    for i in range(num_contracts - 2 * k):
        short_maturity = analysis_curve_df.columns[i]      # C_i
        center_maturity = analysis_curve_df.columns[i+k]   # C_{i+k}
        long_maturity = analysis_curve_df.columns[i+2*k]   # C_{i+2k}

        # Fly = C_i - 2*C_{i+k} + C_{i+2k}
        fly_label = f"{short_maturity}-2x{center_maturity}+{long_maturity}"

        flies_data[fly_label] = analysis_curve_df.iloc[:, i] - 2 * analysis_curve_df.iloc[:, i+k] + analysis_curve_df.iloc[:, i+2*k]

    return pd.DataFrame(flies_data)

# --- Double Butterfly Calculation Function ---
@st.cache_data
def calculate_k_step_double_butterflies(analysis_curve_df, k):
    """
    Calculates double butterflies using contracts separated by 'k' steps (e.g., k=1 for 3M DBF).
    Formula: C_i - 3 * C_{i+k} + 3 * C_{i+2k} - C_{i+3k}
    Label Format: C_i-3xC_{i+k}+3xC_{i+2k}-C_{i+3k}
    """
    # Need 4 contracts: C_i, C_{i+k}, C_{i+2k}, C_{i+3k}
    if analysis_curve_df.empty or analysis_curve_df.shape[1] < 3 * k + 1:
        return pd.DataFrame()

    num_contracts = analysis_curve_df.shape[1]
    dbflies_data = {}

    for i in range(num_contracts - 3 * k):
        c1_maturity = analysis_curve_df.columns[i]          # C_i
        c2_maturity = analysis_curve_df.columns[i+k]        # C_{i+k}
        c3_maturity = analysis_curve_df.columns[i+2*k]      # C_{i+2k}
        c4_maturity = analysis_curve_df.columns[i+3*k]      # C_{i+3k}

        # DBF = C_i - 3*C_{i+k} + 3*C_{i+2k} - C_{i+3k}
        dbfly_label = f"{c1_maturity}-3x{c2_maturity}+3x{c3_maturity}-{c4_maturity}"

        dbflies_data[dbfly_label] = (
            analysis_curve_df.iloc[:, i] 
            - 3 * analysis_curve_df.iloc[:, i+k] 
            + 3 * analysis_curve_df.iloc[:, i+2*k] 
            - analysis_curve_df.iloc[:, i+3*k]
        )

    return pd.DataFrame(dbflies_data)


def compute_all_derivatives_from_outrights_row(contract_labels, outrights_row):
    """Given a single outright curve (Series indexed by contract labels),
    compute all 3M/6M/12M spreads, flies, and double flies for that snapshot.

    This is used in the PCA shock engine to rebuild **all** curve derivatives
    from a shocked outright curve in a consistent way.
    """
    contracts = list(contract_labels)
    n = len(contracts)

    def _compute_for_k(k):
        spreads = {}
        flies = {}
        dbflies = {}

        # Spreads: C_i - C_{i+k}
        for i in range(n - k):
            c1 = contracts[i]
            c2 = contracts[i + k]
            spreads[f"{c1}-{c2}"] = outrights_row[c1] - outrights_row[c2]

        # Flies: C_i - 2*C_{i+k} + C_{i+2k}
        for i in range(n - 2 * k):
            c1 = contracts[i]
            c2 = contracts[i + k]
            c3 = contracts[i + 2 * k]
            flies[f"{c1}-2x{c2}+{c3}"] = (
                outrights_row[c1]
                - 2 * outrights_row[c2]
                + outrights_row[c3]
            )

        # Double Flies: C_i - 3*C_{i+k} + 3*C_{i+2k} - C_{i+3k}
        for i in range(n - 3 * k):
            c1 = contracts[i]
            c2 = contracts[i + k]
            c3 = contracts[i + 2 * k]
            c4 = contracts[i + 3 * k]
            dbflies[f"{c1}-3x{c2}+3x{c3}-{c4}"] = (
                outrights_row[c1]
                - 3 * outrights_row[c2]
                + 3 * outrights_row[c3]
                - outrights_row[c4]
            )

        return (
            pd.Series(spreads) if spreads else pd.Series(dtype=float),
            pd.Series(flies) if flies else pd.Series(dtype=float),
            pd.Series(dbflies) if dbflies else pd.Series(dtype=float),
        )

    # 3M (k=1), 6M (k=2), 12M (k=4)
    spreads_3M, flies_3M, dbf_3M = _compute_for_k(1)
    spreads_6M, flies_6M, dbf_6M = _compute_for_k(2)
    spreads_12M, flies_12M, dbf_12M = _compute_for_k(4)

    return {
        "3M_spreads": spreads_3M,
        "3M_flies": flies_3M,
        "3M_dbf": dbf_3M,
        "6M_spreads": spreads_6M,
        "6M_flies": flies_6M,
        "6M_dbf": dbf_6M,
        "12M_spreads": spreads_12M,
        "12M_flies": flies_12M,
        "12M_dbf": dbf_12M,
    }


# --- END GENERALIZED DERIVATIVE CALCULATION FUNCTIONS ---


def perform_pca(data_df):
    """Performs PCA on the input DataFrame (expected to be spreads for Fair Curve)."""
    data_df_clean = data_df.dropna()
    
    if data_df_clean.empty or data_df_clean.shape[0] < data_df_clean.shape[1]:
        return None, None, None, None, None

    # Standardize the data (PCA on Correlation Matrix - preferred for spread PCA)
    data_mean = data_df_clean.mean()
    data_std = data_df_clean.std()
    # FIXED: replace zero std with 1 to prevent division-by-zero (constant columns)
    data_std = data_std.replace(0, 1)
    data_scaled = (data_df_clean - data_mean) / data_std
    
    n_components = min(data_scaled.shape)

    pca = PCA(n_components=n_components)
    pca.fit(data_scaled)
    
    # Loadings (Eigenvectors on Correlation Matrix)
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=data_df_clean.columns
    )
    # Get Eigenvalues (Variance of the principal components)
    eigenvalues = pca.explained_variance_
    
    explained_variance_ratio = pca.explained_variance_ratio_
    
    scores = pd.DataFrame(
        pca.transform(data_scaled),
        index=data_df_clean.index,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    
    return loadings, explained_variance_ratio, eigenvalues, scores, data_df_clean

# --- PCA ON PRICES (FOR NON-UNIFORM PC1 VISUALIZATION) ---
def perform_pca_on_prices(price_df):
    """
    Performs PCA directly on Outright Price Levels using the COVARIANCE MATRIX 
    (unstandardized data), which results in a NON-UNIFORM PC1.
    """
    data_df_clean = price_df.dropna()
    
    if data_df_clean.empty or data_df_clean.shape[0] < data_df_clean.shape[1]:
        return None, None
        
    # Center the data, but DO NOT scale/standardize it (PCA on Covariance Matrix)
    # FIXED: drop constant columns before centering to avoid degenerate covariance matrix
    data_df_clean = data_df_clean.loc[:, data_df_clean.std() > 0]
    if data_df_clean.empty:
        return None, None
    data_centered = data_df_clean - data_df_clean.mean()
    
    n_components = min(data_centered.shape)

    pca = PCA(n_components=n_components)
    pca.fit(data_centered)
    
    # Loadings (Eigenvectors - the raw sensitivities)
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=data_df_clean.columns
    )
    
    explained_variance = pca.explained_variance_ratio_
    
    return loadings, explained_variance

# --- RECONSTRUCTION LOGIC ---

def _reconstruct_derivative(original_df, reconstructed_prices, derivative_type='spread'):
    """
    Helper to reconstruct a derivative from the reconstructed price curve.
    """
    if original_df.empty:
        return pd.DataFrame()

    # Align the original data index with the reconstructed prices index
    valid_indices = reconstructed_prices.index.intersection(original_df.index)
    original_df_aligned = original_df.loc[valid_indices]
    reconstructed_prices_aligned = reconstructed_prices.loc[valid_indices]
    
    reconstructed_data = {}
    
    for label in original_df_aligned.columns:
        
        try:
            if derivative_type == 'spread':
                # Spread: C_i - C_{i+k}. Label is X Spread: C_i-C_{i+k} (e.g., 3M Spread: Z25-M26)
                if ':' in label:
                    core_label = label.split(': ')[1] 
                else:
                    core_label = label
                    
                c1, c_long = core_label.split('-', 1)
                
                reconstructed_data[label + ' (PCA)'] = (
                    reconstructed_prices_aligned[c1 + ' (PCA)'] - reconstructed_prices_aligned[c_long + ' (PCA)']
                )
            
            elif derivative_type == 'fly':
                # Fly: C_i - 2 * C_{i+k} + C_{i+2k}. Label format: X Fly: C_i-2xC_{i+k}+C_{i+2k}
                if ':' in label:
                    core_label = label.split(': ')[1] 
                else:
                    core_label = label
                    
                parts = core_label.split('-', 1) 
                c1 = parts[0] 
                sub_parts = parts[1].split('+')
                c2_label = sub_parts[0].split('x')[1] 
                c3_label = sub_parts[1] 
                
                # Reconstruct the derivative
                reconstructed_data[label + ' (PCA)'] = (
                    reconstructed_prices_aligned[c1 + ' (PCA)'] - 
                    2 * reconstructed_prices_aligned[c2_label + ' (PCA)'] + 
                    reconstructed_prices_aligned[c3_label + ' (PCA)']
                )
            
            elif derivative_type == 'dbfly':
                # Double Fly: C_i - 3 * C_{i+k} + 3 * C_{i+2k} - C_{i+3k}. Label format: X Double Fly: C_i-3xC_{i+k}+3xC_{i+2k}-C_{i+3k}
                if ':' in label:
                    core_label = label.split(': ')[1] 
                else:
                    core_label = label
                    
                parts = core_label.split('-', 1) 
                c1 = parts[0] # C_i
                
                sub_parts_1 = parts[1].split('+')
                
                c2_label = sub_parts_1[0].split('x')[1] # C_{i+k} from '3xC_{i+k}'
                
                sub_parts_2 = sub_parts_1[1].split('-')
                
                c3_label = sub_parts_2[0].split('x')[1] # C_{i+2k} from '3xC_{i+2k}'
                c4_label = sub_parts_2[1] # C_{i+3k}
                
                # Reconstruct the derivative
                reconstructed_data[label + ' (PCA)'] = (
                    reconstructed_prices_aligned[c1 + ' (PCA)'] - 
                    3 * reconstructed_prices_aligned[c2_label + ' (PCA)'] + 
                    3 * reconstructed_prices_aligned[c3_label + ' (PCA)'] -
                    reconstructed_prices_aligned[c4_label + ' (PCA)']
                )
            
        except Exception as e:
             # Skip if reconstruction fails due to malformed label or missing price
             continue 
    
    reconstructed_df = pd.DataFrame(reconstructed_data, index=reconstructed_prices_aligned.index)
    
    original_rename = {col: col + ' (Original)' for col in original_df_aligned.columns}
    original_df_renamed = original_df_aligned.rename(columns=original_rename)
    
    return pd.merge(original_df_renamed, reconstructed_df, left_index=True, right_index=True)


def reconstruct_prices_and_derivatives(analysis_curve_df, reconstructed_spreads_3M_df, spreads_3M_df, spreads_6M_df, butterflies_3M_df, butterflies_6M_df, spreads_12M_df, butterflies_12M_df, double_butterflies_3M_df, double_butterflies_6M_df, double_butterflies_12M_df):
    """
    Reconstructs Outright Prices and all derivative types based on the
    reconstructed 3M spreads (PCA result) and the original nearest contract price anchor.

    SPREAD SIGN CONVENTION (critical — all derivative calcs depend on this):
        Spread(i, i+1) = Price(i) - Price(i+1)
        => Price(i+1) = Price(i) - Spread(i, i+1)

    For SOFR futures, prices decrease along the curve (front contract is highest),
    so spreads are positive in a normal (non-inverted) environment.
    This convention is set in calculate_k_step_spreads() and must match here.

    A runtime check below verifies the sign is consistent on the analysis date.
    If it fires, check whether calculate_k_step_spreads has been changed.
    """
    # Filter the analysis_curve_df to match the index of the reconstructed 3M spreads
    analysis_curve_df_aligned = analysis_curve_df.loc[reconstructed_spreads_3M_df.index]

    # --- SIGN CONVENTION CHECK ---
    # Verify that for the most recent row, Spread = Price(i) - Price(i+1) is consistent
    # with the column order in analysis_curve_df_aligned (ascending expiry = descending price).
    _check_cols = analysis_curve_df_aligned.columns
    if len(_check_cols) >= 2:
        _last_row = analysis_curve_df_aligned.iloc[-1]
        _c0, _c1 = _check_cols[0], _check_cols[1]
        _spread_key = f"{_c0}-{_c1}"
        if _spread_key in reconstructed_spreads_3M_df.columns:
            _spread_direct = _last_row[_c0] - _last_row[_c1]
            _spread_stored = spreads_3M_df[_spread_key].iloc[-1] if _spread_key in spreads_3M_df.columns else None
            if _spread_stored is not None and not np.isnan(_spread_stored) and not np.isnan(_spread_direct):
                # Signs must agree (both positive or both negative)
                if np.sign(_spread_direct) != np.sign(_spread_stored) and abs(_spread_direct) > 1e-6:
                    import warnings
                    warnings.warn(
                        f"SPREAD SIGN MISMATCH on {_c0}-{_c1}: "
                        f"direct={_spread_direct:.4f}, stored={_spread_stored:.4f}. "
                        "Price reconstruction will be inverted. "
                        "Check calculate_k_step_spreads sign convention.",
                        RuntimeWarning, stacklevel=2
                    )

    # --- FOMC-AWARE ANCHOR SELECTION ---
    #
    # WHY FOMC-BASED, NOT CALENDAR-BASED:
    # Each SR3 contract settles to the average SOFR rate over its reference quarter.
    # The most policy-meaningful anchor is the first contract whose settlement window
    # CONTAINS the next FOMC meeting — because that contract directly prices what the
    # Fed is about to decide.  Using it as the anchor means the PCA spread structure is
    # rebuilt around the single most liquid, most informative policy-rate pivot point.
    #
    # The old 30-day calendar rule was heuristic: it avoided expiry noise, but could
    # land on a contract that doesn't span any FOMC (e.g. a mid-quarter contract with
    # no meeting in its window), which is economically less meaningful.
    #
    # ALGORITHM:
    #   1. Find the next FOMC date on or after today.
    #   2. Walk through active contracts (ascending expiry).
    #   3. The anchor = first contract whose settlement start ≤ next FOMC ≤ expiry.
    #   4. Settlement start is approximated as expiry − 3 months (SR3 convention).
    #   5. If no contract spans the next FOMC (e.g. very deferred calendar), fall back
    #      to the first contract with >= 14 days to expiry.
    #
    # FOMC dates (hardcoded through 2027 — update annually):
    _FOMC_DATES = [
        '2024-01-31','2024-03-20','2024-05-01','2024-06-12','2024-07-31',
        '2024-09-18','2024-11-07','2024-12-18',
        '2025-01-29','2025-03-19','2025-05-07','2025-06-18','2025-07-30',
        '2025-09-17','2025-11-05','2025-12-17',
        '2026-01-28','2026-03-18','2026-05-06','2026-06-17','2026-07-29',
        '2026-09-16','2026-11-04','2026-12-16',
        '2027-01-27','2027-03-17','2027-05-05','2027-06-16',
    ]
    _fomc_ts_list = [pd.Timestamp(d) for d in _FOMC_DATES]
    _today_ts = pd.Timestamp(date.today())

    # Next upcoming FOMC
    _upcoming_fomcs = [f for f in _fomc_ts_list if f >= _today_ts]
    _next_fomc = min(_upcoming_fomcs) if _upcoming_fomcs else None

    _anchor_idx = 0
    _anchor_method = "front (fallback)"

    if _next_fomc is not None:
        for _ci, _c_label in enumerate(analysis_curve_df_aligned.columns):
            _exp_str = _SR3_EXPIRY_MAP.get(_c_label)
            if _exp_str is None:
                continue
            _exp_ts = pd.Timestamp(_exp_str)
            # SR3 settlement reference period: approximately expiry-3months to expiry
            _settle_start = _exp_ts - pd.DateOffset(months=3)
            if _settle_start <= _next_fomc <= _exp_ts:
                _anchor_idx = _ci
                _anchor_method = f"FOMC-anchored (next FOMC {_next_fomc.date()} in settlement window)"
                break
        else:
            # No contract spans the FOMC — fall back to first contract with >= 14 days remaining
            for _ci, _c_label in enumerate(analysis_curve_df_aligned.columns):
                _exp_str = _SR3_EXPIRY_MAP.get(_c_label)
                if _exp_str is None:
                    continue
                if (pd.Timestamp(_exp_str) - _today_ts).days >= 14:
                    _anchor_idx = _ci
                    _anchor_method = "14-day fallback (no contract spans next FOMC)"
                    break

    _anchor_label = analysis_curve_df_aligned.columns[_anchor_idx]
    _anchor_exp_str = _SR3_EXPIRY_MAP.get(_anchor_label)
    _anchor_days = (pd.Timestamp(_anchor_exp_str) - _today_ts).days if _anchor_exp_str else None

    if _anchor_idx > 0:
        st.info(
            f"ℹ️ **Anchor**: **{_anchor_label}** ({_anchor_method}"
            + (f", {_anchor_days}d to expiry" if _anchor_days else "")
            + f"). Front contract **{analysis_curve_df_aligned.columns[0]}** skipped."
        )

    nearest_contract_original = analysis_curve_df_aligned.iloc[:, _anchor_idx]
    nearest_contract_label = _anchor_label

    reconstructed_prices_df = pd.DataFrame(index=analysis_curve_df_aligned.index)
    reconstructed_prices_df[nearest_contract_label + ' (PCA)'] = nearest_contract_original  # Anchor

    spreads_3M_df_no_prefix = spreads_3M_df.copy()

    # Build forward (anchor → last contract)
    for i in range(_anchor_idx + 1, len(analysis_curve_df_aligned.columns)):
        prev_maturity = analysis_curve_df_aligned.columns[i - 1]
        current_maturity = analysis_curve_df_aligned.columns[i]
        spread_label_no_prefix = f"{prev_maturity}-{current_maturity}"
        if spread_label_no_prefix in reconstructed_spreads_3M_df.columns:
            reconstructed_prices_df[current_maturity + ' (PCA)'] = (
                reconstructed_prices_df[prev_maturity + ' (PCA)'] - reconstructed_spreads_3M_df[spread_label_no_prefix]
            )
        else:
            reconstructed_prices_df[current_maturity + ' (PCA)'] = reconstructed_prices_df[prev_maturity + ' (PCA)']

    # Build backward (anchor → front contract, if anchor is not col 0)
    for i in range(_anchor_idx - 1, -1, -1):
        next_maturity = analysis_curve_df_aligned.columns[i + 1]
        current_maturity = analysis_curve_df_aligned.columns[i]
        spread_label_no_prefix = f"{current_maturity}-{next_maturity}"
        if spread_label_no_prefix in reconstructed_spreads_3M_df.columns:
            reconstructed_prices_df[current_maturity + ' (PCA)'] = (
                reconstructed_prices_df[next_maturity + ' (PCA)'] + reconstructed_spreads_3M_df[spread_label_no_prefix]
            )
        else:
            reconstructed_prices_df[current_maturity + ' (PCA)'] = reconstructed_prices_df[next_maturity + ' (PCA)']

    original_price_rename = {col: col + ' (Original)' for col in analysis_curve_df_aligned.columns}
    original_prices_df = analysis_curve_df_aligned.rename(columns=original_price_rename)
    historical_outrights = pd.merge(original_prices_df, reconstructed_prices_df, left_index=True, right_index=True)


    # --- 2. Reconstruct Derivatives from Reconstructed Prices ---
    
    # Prepare derivative DFs with prefixes for _reconstruct_derivative to correctly rename columns
    spreads_3M_df_prefixed = spreads_3M_df_no_prefix.rename(columns=lambda x: f"3M Spread: {x}")
    butterflies_3M_df_prefixed = butterflies_3M_df.rename(columns=lambda x: f"3M Fly: {x}")
    spreads_6M_df_prefixed = spreads_6M_df.rename(columns=lambda x: f"6M Spread: {x}")
    butterflies_6M_df_prefixed = butterflies_6M_df.rename(columns=lambda x: f"6M Fly: {x}")
    spreads_12M_df_prefixed = spreads_12M_df.rename(columns=lambda x: f"12M Spread: {x}")
    butterflies_12M_df_prefixed = butterflies_12M_df.rename(columns=lambda x: f"12M Fly: {x}")
    
    # New Double Butterfly DFs
    double_butterflies_3M_df_prefixed = double_butterflies_3M_df.rename(columns=lambda x: f"3M Double Fly: {x}")
    double_butterflies_6M_df_prefixed = double_butterflies_6M_df.rename(columns=lambda x: f"6M Double Fly: {x}")
    double_butterflies_12M_df_prefixed = double_butterflies_12M_df.rename(columns=lambda x: f"12M Double Fly: {x}")

    historical_spreads_3M = _reconstruct_derivative(spreads_3M_df_prefixed, reconstructed_prices_df, derivative_type='spread')
    historical_butterflies_3M = _reconstruct_derivative(butterflies_3M_df_prefixed, reconstructed_prices_df, derivative_type='fly')
    
    historical_spreads_6M = _reconstruct_derivative(spreads_6M_df_prefixed, reconstructed_prices_df, derivative_type='spread')
    historical_butterflies_6M = _reconstruct_derivative(butterflies_6M_df_prefixed, reconstructed_prices_df, derivative_type='fly')
    
    historical_spreads_12M = _reconstruct_derivative(spreads_12M_df_prefixed, reconstructed_prices_df, derivative_type='spread')
    historical_butterflies_12M = _reconstruct_derivative(butterflies_12M_df_prefixed, reconstructed_prices_df, derivative_type='fly')
    
    # New Double Butterfly reconstructions
    historical_double_butterflies_3M = _reconstruct_derivative(double_butterflies_3M_df_prefixed, reconstructed_prices_df, derivative_type='dbfly')
    historical_double_butterflies_6M = _reconstruct_derivative(double_butterflies_6M_df_prefixed, reconstructed_prices_df, derivative_type='dbfly')
    historical_double_butterflies_12M = _reconstruct_derivative(double_butterflies_12M_df_prefixed, reconstructed_prices_df, derivative_type='dbfly')

    # MODIFIED: Return the new historical double butterfly DFs
    return historical_outrights, historical_spreads_3M, historical_butterflies_3M, historical_spreads_6M, historical_butterflies_6M, historical_spreads_12M, historical_butterflies_12M, historical_double_butterflies_3M, historical_double_butterflies_6M, historical_double_butterflies_12M, spreads_3M_df_no_prefix


# --- ORIGINAL HEDGING LOGIC (Section 6) ---

def calculate_reconstructed_covariance(loadings_df, eigenvalues, spread_std_dev, pc_count):
    """
    Calculates the covariance matrix of the STANDARDIZED spreads 
    reconstructed using the first 'pc_count' PCs: Sigma_scaled = L_p Lambda_p L_p^T
    Then scales back to original spread space: Sigma = (diag(sigma)) * Sigma_scaled * (diag(sigma))
    """
    # 1. Select the loadings and eigenvalues for the used PCs
    L_p = loadings_df.iloc[:, :pc_count].values # Loadings (Eigenvectors on Correlation Matrix)
    lambda_p = eigenvalues[:pc_count]           # Eigenvalues (Variance of standardized scores)
    
    # 2. Reconstruct the Covariance Matrix of the Standardized Data
    # Sigma_scaled = L_p * Lambda_p * L_p^T
    Sigma_scaled = L_p @ np.diag(lambda_p) @ L_p.T
    
    # 3. Scale back to the original spread data covariance matrix
    # Cov(X) = diag(sigma) * Cov(Z) * diag(sigma)
    Sigma = Sigma_scaled * np.outer(spread_std_dev.values, spread_std_dev.values)
    
    Sigma_df = pd.DataFrame(Sigma, index=loadings_df.index, columns=loadings_df.index)
    
    return Sigma_df

def calculate_best_and_worst_hedge_3M(trade_label, loadings_df, eigenvalues, pc_count, spreads_3M_df_clean):
    """
    Calculates the best (min residual risk) and worst (max residual risk) 
    hedge for a given 3M spread trade using the reconstructed covariance matrix, 
    and returns the full results DataFrame as well. (Section 6 - 3M Spreads only)
    """
    if trade_label not in loadings_df.index:
        return None, None, None
        
    spread_std_dev = spreads_3M_df_clean.std()
    
    # Reconstruct covariance matrix using selected PCs
    Sigma_reconstructed = calculate_reconstructed_covariance(
        loadings_df, eigenvalues, spread_std_dev, pc_count
    )
    
    trade_spread = trade_label
    
    results = []
    
    # Iterate through all other 3M spreads as potential hedges
    potential_hedges = [col for col in Sigma_reconstructed.columns if col != trade_spread]
    
    for hedge_spread in potential_hedges:
        
        # Terms from the reconstructed covariance matrix (Sigma)
        Var_Trade = Sigma_reconstructed.loc[trade_spread, trade_spread] # Var(T)
        Var_Hedge = Sigma_reconstructed.loc[hedge_spread, hedge_spread] # Var(H)
        Cov_TH = Sigma_reconstructed.loc[trade_spread, hedge_spread]    # Cov(T, H)
        
        # 1. Minimum Variance Hedge Ratio (k*)
        # FIXED: use near-zero threshold instead of exact == 0 (floating-point safety)
        if Var_Hedge <= 1e-9:
            k_star = 0
        else:
            k_star = Cov_TH / Var_Hedge
            
        # 2. Residual Variance at the minimum-variance hedge ratio k*:
        #    Var(T - k*H) = Var(T) - Cov(T,H)²/Var(H)  ≡  Var(T) - k* · Cov(T,H)
        #    This simplified form is ONLY valid at the optimal k* (not for arbitrary k).
        Residual_Variance = Var_Trade - (k_star * Cov_TH)
        Residual_Variance = max(0, Residual_Variance) 
        
        # 3. Residual Volatility in Rate % (1 price point = 100 bps = 1% Rate)
        Residual_Volatility_Rate_Pct = np.sqrt(Residual_Variance) * 100
        
        results.append({
            'Hedge Spread': hedge_spread,
            'Hedge Ratio (k*)': k_star,
            'Residual Volatility (Rate %)': Residual_Volatility_Rate_Pct
        })

    if not results:
        return None, None, None
        
    results_df = pd.DataFrame(results)
    
    # Best hedge minimizes Residual Volatility
    best_hedge = results_df.sort_values(by='Residual Volatility (Rate %)', ascending=True).iloc[0]
    
    # Worst hedge maximizes Residual Volatility
    worst_hedge = results_df.sort_values(by='Residual Volatility (Rate %)', ascending=False).iloc[0]
    
    # Return the individual best/worst series AND the full DataFrame
    return best_hedge, worst_hedge, results_df


# --- GENERALIZED HEDGING LOGIC (Section 7) ---

def calculate_derivatives_covariance_generalized(all_derivatives_df, scores_df, eigenvalues, pc_count):
    """
    Calculates the Raw Covariance Matrix for ALL derivatives (Spreads, Flies, Double Flies)
    by projecting their standardized time series onto the standardized 3M Spread PC scores.

    The covariance is built as:
        Sigma_Std = L_D · diag(lambda_p) · L_D^T  +  diag(residual_var)

    The residual_var term captures the variance of each derivative NOT explained by the
    selected PCs (i.e. 1 - R² per instrument in standardized space).  Omitting it
    (the previous implementation) silently dropped idiosyncratic risk and biased hedge
    ratios toward zero — especially for double-flies whose R² against 3M spread PCs is low.

    Returns the Raw Covariance Matrix, the aligned derivatives data, and the standardized loadings (L_D).
    """
    # 1. Align and clean data
    aligned_index = all_derivatives_df.index.intersection(scores_df.index)
    derivatives_aligned = all_derivatives_df.loc[aligned_index].dropna(axis=1)
    scores_aligned = scores_df.loc[aligned_index]

    if derivatives_aligned.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # 2. Standardize all derivatives
    derivatives_mean = derivatives_aligned.mean()
    derivatives_std = derivatives_aligned.std()
    derivatives_std = derivatives_std.replace(0, 1)   # guard constant columns
    derivatives_scaled = (derivatives_aligned - derivatives_mean) / derivatives_std

    # 3. OLS loadings of each standardized derivative onto the first pc_count PC scores
    loadings_data = {}
    residual_var = {}
    X = scores_aligned.iloc[:, :pc_count].values   # shape (T, pc_count)

    for col in derivatives_scaled.columns:
        y = derivatives_scaled[col].values
        reg = LinearRegression(fit_intercept=False)
        reg.fit(X, y)
        loadings_data[col] = reg.coef_

        # Residual variance in standardised space: Var(y) - Var(ŷ) = 1 - R²
        # (Var(y) = 1 by construction since y is standardised)
        y_hat = reg.predict(X)
        resid_var_i = float(np.var(y - y_hat, ddof=1))
        residual_var[col] = max(resid_var_i, 0.0)   # floor at zero for numerical safety

    # L_D: shape (n_instruments, pc_count)
    loadings_df = pd.DataFrame(
        loadings_data,
        index=[f'PC{i+1}' for i in range(pc_count)]
    ).T

    # 4. Covariance in standardised space = systematic + idiosyncratic
    L_D = loadings_df.values
    lambda_p = eigenvalues[:pc_count]
    Sigma_Std_systematic = L_D @ np.diag(lambda_p) @ L_D.T
    Sigma_Std_idiosyncratic = np.diag([residual_var[c] for c in loadings_df.index])
    Sigma_Std = Sigma_Std_systematic + Sigma_Std_idiosyncratic

    # 5. Scale back to raw (price-point) space
    sigma_vec = derivatives_std[loadings_df.index].values
    Sigma_Raw = Sigma_Std * np.outer(sigma_vec, sigma_vec)

    Sigma_Raw_df = pd.DataFrame(Sigma_Raw, index=loadings_df.index, columns=loadings_df.index)

    return Sigma_Raw_df, derivatives_aligned, loadings_df

def calculate_best_and_worst_hedge_generalized(trade_label, Sigma_Raw_df):
    """
    Calculates the best/worst hedge using the generalized Raw Covariance Matrix (Sigma_Raw_df).
    (Section 7 - All Derivatives)
    """
    
    if trade_label not in Sigma_Raw_df.index:
        return None, None, None
        
    results = []
    
    # Iterate through all other derivatives as potential hedges
    potential_hedges = [col for col in Sigma_Raw_df.columns if col != trade_label]
    
    for hedge_instrument in potential_hedges:
        
        # Terms from the reconstructed covariance matrix (Sigma)
        Var_Trade = Sigma_Raw_df.loc[trade_label, trade_label] # Var(T)
        Var_Hedge = Sigma_Raw_df.loc[hedge_instrument, hedge_instrument] # Var(H)
        Cov_TH = Sigma_Raw_df.loc[trade_label, hedge_instrument]    # Cov(T, H)
        
        # 1. Minimum Variance Hedge Ratio (k*)
        if Var_Hedge <= 1e-9: # Check for near-zero variance
            k_star = 0
        else:
            k_star = Cov_TH / Var_Hedge
            
        # 2. Residual Variance at the minimum-variance hedge ratio k*:
        #    Var(T - k*H) = Var(T) - Cov(T,H)²/Var(H)  ≡  Var(T) - k* · Cov(T,H)
        Residual_Variance = Var_Trade - (k_star * Cov_TH)
        Residual_Variance = max(0, Residual_Variance) 
        
        # 3. Residual Volatility in Rate % (1 price point = 100 bps = 1% Rate)
        Residual_Volatility_Rate_Pct = np.sqrt(Residual_Variance) * 100
        
        results.append({
            'Hedge Instrument': hedge_instrument,
            'Hedge Ratio (k*)': k_star,
            'Residual Volatility (Rate %)': Residual_Volatility_Rate_Pct
        })

    if not results:
        return None, None, None
        
    results_df = pd.DataFrame(results)
    
    # Best hedge minimizes Residual Volatility
    best_hedge = results_df.sort_values(by='Residual Volatility (Rate %)', ascending=True).iloc[0]
    
    # Worst hedge maximizes Residual Volatility
    worst_hedge = results_df.sort_values(by='Residual Volatility (Rate %)', ascending=False).iloc[0]
    
    # Return the individual best/worst series AND the full DataFrame
    return best_hedge, worst_hedge, results_df

# --- FACTOR-BASED HEDGING LOGIC (Section 8) ---

def calculate_factor_sensitivities(loadings_df_gen, pc_count):
    """
    Calculates the Standardized Sensitivity (Beta) of every derivative to the first three 
    principal components (Level, Slope, Curvature).
    """
    if loadings_df_gen.empty:
        return pd.DataFrame()

    # Define the factor mapping based on the first 3 PCs
    pc_map = {
        'PC1': 'Level (Whole Curve Shift)', 
        'PC2': 'Slope (Steepening/Flattening)', 
        'PC3': 'Curvature (Fly Risk)'
    }
    
    # Only use up to the number of available PCs, or 3, whichever is smaller
    available_pcs = loadings_df_gen.columns.intersection(list(pc_map.keys()))
    
    # Filter the generalized loadings L_D for the relevant PCs
    factor_sensitivities = loadings_df_gen.filter(items=available_pcs.tolist(), axis=1).copy()
    
    # Rename columns for clarity in the output
    factor_sensitivities.columns = [pc_map[col] for col in available_pcs]
    
    return factor_sensitivities

# --- NEW FUNCTION FOR TRIPLE FACTOR NEUTRALIZATION CHECK ---
def find_perfect_factor_hedge(trade_label, factor_sensitivities_df, mispricing_series, pc_count, tolerance=0.05):
    """
    Identifies a single hedge instrument that can simultaneously neutralize the first
    three principal components (Level, Slope, and Curvature) for a given trade.

    A "perfect" single-instrument hedge requires that the hedge ratios implied by each
    factor are all equal:  k_PC1 = k_PC2 = k_PC3  where  k_PCi = β_T_i / β_H_i.

    Tolerance is now RELATIVE: max pairwise difference divided by |mean(k)| must be
    below `tolerance` (default 5%).  The previous absolute tolerance of 1e-4 was
    scale-dependent — it virtually never triggered for k >> 1 and false-triggered for
    near-zero ratios.  A relative tolerance of 5% means the three implied hedge ratios
    agree to within 5% of their average magnitude.

    Returns a dictionary of results or None if no perfect hedge is found.
    """
    if trade_label not in factor_sensitivities_df.index:
        return {'error': f"Trade instrument '{trade_label}' not found in sensitivities.", 'result': None}

    available_factors = factor_sensitivities_df.columns.intersection(
        ['Level (Whole Curve Shift)', 'Slope (Steepening/Flattening)', 'Curvature (Fly Risk)']
    )

    if len(available_factors) < 3:
        return {'error': f"Need at least 3 PCs (Level, Slope, Curvature) for triple neutralization check. Only {len(available_factors)} available.", 'result': None}

    T_sens = factor_sensitivities_df.loc[trade_label, available_factors]

    if T_sens.abs().sum() < 1e-9:
        return {'error': "The trade itself has near-zero sensitivity to the first three factors, thus no hedging is needed for these factors.", 'result': None}

    potential_hedges = [col for col in factor_sensitivities_df.index if col != trade_label]

    best_match_result = None

    for hedge_instrument in potential_hedges:
        H_sens = factor_sensitivities_df.loc[hedge_instrument, available_factors]

        if (H_sens.abs() < 1e-9).any():
            continue

        k_ratios = T_sens / H_sens
        k1, k2, k3 = k_ratios.values

        avg_k = k_ratios.mean()
        abs_avg_k = abs(avg_k)

        # Relative tolerance: differences are expressed as fraction of |avg_k|.
        # Guard against near-zero avg_k (would make everything look "perfect").
        if abs_avg_k < 1e-6:
            continue

        diff1 = abs(k1 - k2) / abs_avg_k
        diff2 = abs(k1 - k3) / abs_avg_k
        diff3 = abs(k2 - k3) / abs_avg_k
        max_rel_diff = max(diff1, diff2, diff3)

        if max_rel_diff < tolerance:
            hedge_action = 'Short' if avg_k > 0 else 'Long'
            hedge_mispricing = mispricing_series.get(hedge_instrument, np.nan)

            result = {
                'Hedge Instrument': hedge_instrument,
                'Trade PC1 Sensitivity': T_sens.iloc[0],
                'Trade PC2 Sensitivity': T_sens.iloc[1],
                'Trade PC3 Sensitivity': T_sens.iloc[2],
                'Hedge PC1 Sensitivity': H_sens.iloc[0],
                'Hedge PC2 Sensitivity': H_sens.iloc[1],
                'Hedge PC3 Sensitivity': H_sens.iloc[2],
                'Hedge Ratio (|k|)': abs(avg_k),
                'Hedge Action': hedge_action,
                'Hedge Mispricing (Rate %)': hedge_mispricing,
                'Max Relative K Spread': max_rel_diff,   # renamed from Max K Difference
            }

            if best_match_result is None or max_rel_diff < best_match_result.get('Max Relative K Spread', tolerance):
                best_match_result = result

    if best_match_result:
        return {'error': None, 'result': best_match_result}
    else:
        return {'error': f"No single hedge instrument found to neutralize Level, Slope, and Curvature simultaneously within {tolerance*100:.0f}% relative tolerance.", 'result': None}


def calculate_all_factor_hedges(trade_label, factor_name, factor_sensitivities_df, Sigma_Raw_df):
    """
    Calculates the Factor Hedge Ratio and the resulting Residual Volatility for all potential 
    hedge instruments, for a specified factor.
    """
    if trade_label not in factor_sensitivities_df.index:
        return pd.DataFrame(), f"Trade instrument '{trade_label}' not found in sensitivities."
    if factor_name not in factor_sensitivities_df.columns:
        return pd.DataFrame(), f"Factor '{factor_name}' not found."
    if trade_label not in Sigma_Raw_df.index:
        return pd.DataFrame(), f"Trade instrument '{trade_label}' not found in covariance matrix."

    results = []
    
    Trade_Exposure = factor_sensitivities_df.loc[trade_label, factor_name]
    Var_Trade = Sigma_Raw_df.loc[trade_label, trade_label] # Var(T)
    
    # Iterate through all other derivatives as potential hedges
    potential_hedges = [col for col in Sigma_Raw_df.columns if col != trade_label]

    for hedge_instrument in potential_hedges:
        try:
            Hedge_Exposure = factor_sensitivities_df.loc[hedge_instrument, factor_name]
            Var_Hedge = Sigma_Raw_df.loc[hedge_instrument, hedge_instrument] # Var(H)
            Cov_TH = Sigma_Raw_df.loc[trade_label, hedge_instrument]        # Cov(T, H)

            # 1. Calculate Factor Hedge Ratio (k_factor)
            if abs(Hedge_Exposure) < 1e-9:
                k_factor = 0.0
                Residual_Volatility_Rate_Pct = np.nan # Cannot neutralize factor with zero-exposure hedge
            else:
                # k_factor neutralises the target factor: k = Beta_T / Beta_H
                k_factor = Trade_Exposure / Hedge_Exposure
                
                # Residual Variance: Var(T - k*H) = Var(T) + k²Var(H) - 2k·Cov(T,H)
                # Full formula used here because k_factor ≠ MVHR k*
                Residual_Variance = Var_Trade + (k_factor**2 * Var_Hedge) - (2 * k_factor * Cov_TH)
                Residual_Variance = max(0, Residual_Variance) 
                
                # Residual Volatility in Rate % (1 price point = 100 bps = 1% Rate)
                Residual_Volatility_Rate_Pct = np.sqrt(Residual_Variance) * 100
                
            results.append({
                'Hedge Instrument': hedge_instrument,
                'Trade Sensitivity': Trade_Exposure,
                'Hedge Sensitivity': Hedge_Exposure,
                f'Factor Hedge Ratio (k_factor)': k_factor,
                'Residual Volatility (Rate %)': Residual_Volatility_Rate_Pct
            })
            
        except Exception as e:
            continue

    if not results:
        return pd.DataFrame(), "No valid hedge candidates found."
        
    results_df = pd.DataFrame(results)
    
    # Sort by Residual Volatility (Rate %) to show the most effective hedges first
    results_df = results_df.sort_values(by='Residual Volatility (Rate %)', ascending=True, na_position='last')
    
    return results_df, None

# --- NEW HELPER FUNCTION for Mispricing ---
def calculate_derivative_mispricings(historical_derivatives_list, analysis_dt):
    """
    Calculates the mispricing (Original - PCA Fair) in Rate % for all derivatives 
    on the analysis date. (Was BPS, now divided by 100)
    
    Args:
        historical_derivatives_list (list[pd.DataFrame]): List of all historical derivative DFs 
                                                         (containing 'Original' and 'PCA' columns).
        analysis_dt (datetime.datetime): The single analysis date for the snapshot.

    Returns:
        pd.Series: Series indexed by derivative label (without suffix), with mispricing in Rate % as values.
    """
    mispricing_data = {}
    
    # Ensure analysis_dt is aligned to the dataframe index (usually date component or string format)
    analysis_date_key = analysis_dt.strftime('%Y-%m-%d')
    
    for df in historical_derivatives_list:
        if df.empty:
            continue
        # FIXED: try both datetime key and string key to handle mixed index types
        try:
            row = df.loc[analysis_dt]
        except KeyError:
            try:
                row = df.loc[analysis_date_key]
            except KeyError:
                continue
        
        # Iterate through all derivative columns that contain the original value
        for original_col in [col for col in df.columns if ' (Original)' in col]:
            pca_col = original_col.replace(' (Original)', ' (PCA)')
            
            if pca_col in row and not pd.isna(row[original_col]) and not pd.isna(row[pca_col]):
                # Remove the suffix to get the clean derivative label (e.g., '3M Spread: Z25-H26')
                derivative_label = original_col.replace(' (Original)', '')
                
                # Calculate mispricing in Rate %: (Original - PCA Fair) * 100 
                # MODIFIED: * 10000 -> * 100
                mispricing = (row[original_col] - row[pca_col]) * 100
                mispricing_data[derivative_label] = mispricing
                
    return pd.Series(mispricing_data, name='Hedge Mispricing (Rate %)') # MODIFIED: Column name update
# --- END NEW HELPER FUNCTION ---


# --- NEW FUNCTION FOR SECTION 8.3 ---
def create_instrument_universe_table(factor_sensitivities_df, Sigma_Raw_df, mispricing_series):
    """
    Creates a comprehensive table of all derivative instruments with their key attributes:
    Sensitivities, Total Volatility, and Mispricing.
    """
    if Sigma_Raw_df.empty or factor_sensitivities_df.empty:
        return pd.DataFrame()

    data = []
    
    # Calculate Total Volatility (Standard Deviation * 100)
    # Total Volatility is sqrt(Variance) * 100
    total_volatility = np.sqrt(np.diag(Sigma_Raw_df)) * 100
    total_vol_series = pd.Series(total_volatility, index=Sigma_Raw_df.index)

    for instrument in Sigma_Raw_df.index:
        
        # Determine Derivative Group (Spread, Fly, Double Fly)
        if 'Spread' in instrument and 'Double' not in instrument:
            instr_group = 'Spread'
        elif 'Double Fly' in instrument:
            instr_group = 'Double Fly'
        elif 'Fly' in instrument:
            instr_group = 'Fly'
        else:
            instr_group = 'Other'
            
        # Determine Maturity
        if '3M' in instrument:
            maturity = '3M'
        elif '6M' in instrument:
            maturity = '6M'
        elif '12M' in instrument:
            maturity = '12M'
        else:
            maturity = ''
            
        # Full Type
        full_type = f"{maturity} {instr_group}" if maturity else instr_group
        
        # Sensitivities (Handle missing factors if pc_count < 3)
        if instrument in factor_sensitivities_df.index:
            sensitivities = factor_sensitivities_df.loc[instrument]
            level_sens = sensitivities.get('Level (Whole Curve Shift)', np.nan)
            slope_sens = sensitivities.get('Slope (Steepening/Flattening)', np.nan)
            curve_sens = sensitivities.get('Curvature (Fly Risk)', np.nan)
        else:
            level_sens, slope_sens, curve_sens = np.nan, np.nan, np.nan
        
        # Mispricing (Rate %)
        mispricing = mispricing_series.get(instrument, np.nan)

        data.append({
            'Instrument': instrument,
            'Type': full_type,
            'Derivative Group': instr_group, # Column for filtering
            'Level Sensitivity': level_sens,
            'Slope Sensitivity': slope_sens,
            'Curvature Sensitivity': curve_sens,
            'Total Volatility (Rate %)': total_vol_series.loc[instrument],
            'Mispricing (Rate %)': mispricing
        })

    df = pd.DataFrame(data)
    return df
# --- END NEW FUNCTION ---


# --- Streamlit Application Layout ---

st.title("SOFR Futures PCA Analyzer")

# ── REGIME BANNER (placeholder container — filled after PCA runs) ──────────────
_regime_banner_slot = st.empty()


st.sidebar.header("1. Data Upload")
price_file = st.sidebar.file_uploader(
    "Upload Historical Price Data (e.g., 'SOFR rates.csv')", 
    type=['csv'], 
    key='price_upload'
)

# Initialize dataframes — expiry is hardcoded, no upload needed
price_df = load_data(price_file)
expiry_df = get_hardcoded_expiry_df()

# ── Live Feed ─────────────────────────────────────────────────────────────────
# Render the live feed sidebar section (connect/disconnect/config).
# This also drains the update queue on every rerun.
_live_active = render_live_feed_sidebar(expiry_df)

# If live prices are available, inject today's row into price_df so the rest
# of the app treats live data exactly like historical data.
if _live_active and price_df is not None:
    price_df = inject_live_row(price_df)

# ── End Live Feed ─────────────────────────────────────────────────────────────

# ── Price Source Log Panel (main area, visible when live feed is active) ─────
def render_price_source_panel(price_df_today, contracts_in_use, analysis_date):
    """
    Renders a collapsible panel in the main area showing the price source
    for every outright, spread, fly, and double-fly that will be used on
    the analysis date.  Sources: 'Live VWAP', 'Live Fallback', or 'CSV'.
    """
    live_prices = st.session_state.get("live_prices", {})
    src_map     = st.session_state.get("live_prices_source", {})
    vwap_snap   = st.session_state.get("vwap_prices", {})
    today       = date.today()
    is_today    = (analysis_date == today)

    rows = []
    for contract in contracts_in_use:
        price = price_df_today.get(contract, None)
        if price is None or (isinstance(price, float) and np.isnan(price)):
            price_str = "—"
            source    = "Missing"
            bid_str   = "—"
            ask_str   = "—"
            ts_str    = "—"
        elif is_today and contract in live_prices:
            raw_source = src_map.get(contract, "Fallback")
            source     = "Live VWAP" if raw_source == "VWAP" else "Live Fallback"
            price_str  = f"{price:.3f}"
            snap       = vwap_snap.get(contract, {})
            bid_str    = f"{snap['bid']:.3f}" if snap.get("bid") else "—"
            ask_str    = f"{snap['ask']:.3f}" if snap.get("ask") else "—"
            ts_str     = snap.get("ts", "—")
        else:
            source    = "CSV"
            price_str = f"{price:.3f}"
            bid_str   = "—"
            ask_str   = "—"
            ts_str    = str(analysis_date)

        rows.append({
            "Contract": contract,
            "Price":    price_str,
            "Source":   source,
            "Bid":      bid_str,
            "Ask":      ask_str,
            "As of":    ts_str,
        })

    if not rows:
        return

    df = pd.DataFrame(rows)

    # Colour-code by source
    def colour_source(val):
        if val == "Live VWAP":
            return "background-color: #002200; color: #00cc44; font-family: monospace"
        elif val == "Live Fallback":
            return "background-color: #1a1000; color: #ffaa00; font-family: monospace"
        elif val == "Missing":
            return "background-color: #1a0000; color: #ff3333; font-family: monospace"
        return ""

    with st.expander("📋 Price Source Log — inputs used for today's calculations", expanded=False):
        col_l, col_r = st.columns([3, 1])
        with col_l:
            st.caption(
                f"Analysis date: **{analysis_date}** | "
                f"Live contracts: **{sum(1 for r in rows if 'Live' in r['Source'])}** | "
                f"CSV contracts: **{sum(1 for r in rows if r['Source'] == 'CSV')}** | "
                f"Missing: **{sum(1 for r in rows if r['Source'] == 'Missing')}**"
            )
        with col_r:
            if st.button("🔃 Refresh prices", key="price_log_refresh"):
                q = st.session_state.get("ls_queue")
                if q:
                    # drain inline
                    while not q.empty():
                        try:
                            item = q.get_nowait()
                            contract2, price2 = item[0], item[1]
                            source2 = item[2] if len(item) > 2 else "price"
                            existing_src = st.session_state.get("live_prices_source", {}).get(contract2, "price")
                            if source2 == "VWAP" or existing_src != "VWAP":
                                st.session_state["live_prices"][contract2] = price2
                                sm = st.session_state.get("live_prices_source", {})
                                sm[contract2] = source2
                                st.session_state["live_prices_source"] = sm
                        except Exception:
                            break
                st.rerun()

        bbg_table(
            df.style.map(colour_source, subset=["Source"]),
            use_container_width=True,
            hide_index=True,
            height=min(400, 35 * len(df) + 40),
        )
        st.caption("🟢 Live VWAP = (bid×ask_qty + ask×bid_qty)/(ask_qty+bid_qty)  "
                   "🟡 Live Fallback = Settle/Last from feed  "
                   "⬜ CSV = historical file")

# Store reference for use after analysis_date is known
st.session_state["_price_source_panel_fn"] = render_price_source_panel

# Placeholder for L_D Loadings and Sigma_Raw_df, calculated in Section 7 and used in Section 8
loadings_df_gen = pd.DataFrame()
Sigma_Raw_df = pd.DataFrame()
spreads_3M_df_no_prefix = pd.DataFrame() # Also need this for Section 6 if price_df is not None

if price_df is not None:
    # --- Date Range Filter ---
    st.sidebar.header("2. Historical Date Range")
    min_date = price_df.index.min().date()
    max_date = price_df.index.max().date()

    # Default start: 5 years back from max_date (or min_date if data is shorter)
    _five_years_ago = max_date - timedelta(days=5 * 365)
    _default_start  = max(_five_years_ago, min_date)

    start_date, end_date = st.sidebar.date_input(
        "Select Historical Data Range for PCA Calibration", 
        value=[_default_start, max_date],
        min_value=min_date,
        max_value=max_date
    )
    
    price_df_filtered = price_df[(price_df.index.date >= start_date) & (price_df.index.date <= end_date)]
    
    # --- Analysis Date Selector (Maturity Roll) ---
    st.sidebar.header("3. Curve Analysis Date")

    # If live data was injected, default the analysis date to today
    today_date = date.today()
    if _live_active and today_date >= min_date and today_date <= max_date:
        default_analysis_date = today_date
    else:
        default_analysis_date = end_date
    if default_analysis_date < min_date:
        default_analysis_date = min_date

    analysis_date = st.sidebar.date_input(
        "Select **Single Date** for Curve Snapshot",
        value=default_analysis_date,
        min_value=min_date,
        max_value=max_date,
        key='analysis_date'
    )
    
    analysis_dt = datetime.combine(analysis_date, datetime.min.time())

else:
    st.info("Please upload the Historical Price Data CSV file to begin the analysis.")
    st.stop()



# ─── TAB LAYOUT ──────────────────────────────────────────────────────────────
_tab_pca, _tab_snap, _tab_hedge, _tab_macro, _tab_trade, _tab_export = st.tabs([
    "📊 PCA & Data",
    "📈 Curve Snapshots",
    "🛡️ Hedging",
    "🔭 Macro & Shocks",
    "🧩 Trade Ideas",
    "📥 Export",
])
# ──────────────────────────────────────────────────────────────────────────────

# --- Core Processing Logic ---
if not price_df_filtered.empty:
    
    # 1. Get the list of relevant contracts
    future_expiries_df = get_analysis_contracts(expiry_df, analysis_dt)
    
    if future_expiries_df.empty:
        st.warning("Could not establish a relevant contract curve. Please check your date filters.")
        st.stop()

    # 2. Transform historical prices to the required maturity curve
    analysis_curve_df, contract_labels = transform_to_analysis_curve(price_df_filtered, future_expiries_df)
    
    if analysis_curve_df.empty:
        st.warning("Data transformation failed. Check if contracts in the price file match contracts in the expiry file.")
        st.stop()
        
    # 3. Calculate Derivatives
    with _tab_pca:
        st.header("1. Data Derivatives Check (Contracts relevant to selected Analysis Date)")
    
        # 3M (k=1) - Used for PCA input
        spreads_3M_df_raw = calculate_k_step_spreads(analysis_curve_df, 1) # No prefix here
        butterflies_3M_df = calculate_k_step_butterflies(analysis_curve_df, 1)
        double_butterflies_3M_df = calculate_k_step_double_butterflies(analysis_curve_df, 1) 
    
        # 6M (k=2)
        spreads_6M_df = calculate_k_step_spreads(analysis_curve_df, 2)
        butterflies_6M_df = calculate_k_step_butterflies(analysis_curve_df, 2)
        double_butterflies_6M_df = calculate_k_step_double_butterflies(analysis_curve_df, 2) 
    
        # 12M (k=4)
        spreads_12M_df = calculate_k_step_spreads(analysis_curve_df, 4)
        butterflies_12M_df = calculate_k_step_butterflies(analysis_curve_df, 4)
        double_butterflies_12M_df = calculate_k_step_double_butterflies(analysis_curve_df, 4) 
    
    
        # Display the number of contracts and derivatives
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Contracts", len(contract_labels))
        col2.metric("3M Spreads", spreads_3M_df_raw.shape[1])
        col3.metric("3M Flies", butterflies_3M_df.shape[1])
        col4.metric("3M Double Flies", double_butterflies_3M_df.shape[1])
        col5.metric("Date Range Days", price_df_filtered.shape[0])

        # ── Price Source Log Panel ────────────────────────────────────────────
        _all_active_contracts = future_expiries_df.index.tolist()
        _price_row = {}
        _ts = analysis_dt if analysis_dt in price_df_filtered.index else pd.Timestamp(analysis_date)
        if _ts in price_df_filtered.index:
            _price_row = {c: price_df_filtered.loc[_ts, c] for c in _all_active_contracts if c in price_df_filtered.columns}

        _panel_fn = st.session_state.get("_price_source_panel_fn")
        if _panel_fn is not None:
            _panel_fn(_price_row, _all_active_contracts, analysis_date)
        # ── End Price Source Log ──────────────────────────────────────────────


        # 4. Perform PCA on 3M Spreads (Fair Curve)
        st.header("2. Principal Component Analysis (PCA) on 3M Spreads")
    
        loadings_spread, explained_variance_ratio, eigenvalues, scores, spreads_3M_df_clean = perform_pca(spreads_3M_df_raw)
    
        if loadings_spread is not None:

            # ── LIVE REGIME DETECTION BANNER ──────────────────────────────────────
            # Uses 60-day rolling z-score of PC1, PC2, PC3 scores and 5/20-day momentum.
            # All thresholds are derived from the data distribution, not hardcoded:
            #   - z-score threshold = ±1.5 (1.5 std from 60-day mean — robust to fat tails)
            #   - momentum flip = sign change in 5d vs 20d diff of PC score
            # No arbitrary bps numbers are shown — only regime labels from score statistics.
            def _compute_regime_banner(scores_df, analysis_dt):
                """
                Returns a list of (label, colour, detail) tuples for each active regime signal.
                Derived entirely from PC score z-scores and momentum — no hardcoded rate levels.
                """
                _signals = []
                _pc_labels = {0: ('Level (PC1)', 'level'), 1: ('Slope (PC2)', 'slope'), 2: ('Curvature (PC3)', 'curv')}
                _n_pcs = min(3, scores_df.shape[1])

                # Align to analysis date or nearest prior date
                _valid_idx = scores_df.index[scores_df.index <= analysis_dt]
                if len(_valid_idx) < 25:
                    return []
                _s = scores_df.loc[_valid_idx]

                for _pi in range(_n_pcs):
                    _col = scores_df.columns[_pi]
                    _series = _s[_col].dropna()
                    if len(_series) < 25:
                        continue

                    # Rolling 60-day z-score (or full history if shorter)
                    _w = min(60, len(_series) - 5)
                    _roll_mean = _series.rolling(_w).mean()
                    _roll_std  = _series.rolling(_w).std().replace(0, np.nan)
                    _z_now     = float((_series.iloc[-1] - _roll_mean.iloc[-1]) / _roll_std.iloc[-1]) \
                                 if not np.isnan(_roll_std.iloc[-1]) else 0.0
                    _z_prev    = float((_series.iloc[-5] - _roll_mean.iloc[-5]) / _roll_std.iloc[-5]) \
                                 if len(_series) > 6 and not np.isnan(_roll_std.iloc[-5]) else _z_now

                    # 5-day and 20-day momentum (raw score change)
                    _mom5  = float(_series.diff(5).iloc[-1])  if len(_series) > 5  else 0.0
                    _mom20 = float(_series.diff(20).iloc[-1]) if len(_series) > 20 else 0.0

                    _pc_name = _pc_labels[_pi][0]

                    # Level (PC1): drives overall curve level / dovish-hawkish axis
                    if _pi == 0:
                        if _z_now < -1.5:
                            _signals.append(("🟢 DOVISH", "#1a6e2e",
                                f"{_pc_name} z={_z_now:+.2f} — curve pricing significant cuts / rally mode"))
                        elif _z_now > 1.5:
                            _signals.append(("🔴 HAWKISH", "#8b0000",
                                f"{_pc_name} z={_z_now:+.2f} — curve pricing hikes / sell-off mode"))
                        else:
                            _signals.append(("⚪ LEVEL NEUTRAL", "#555555",
                                f"{_pc_name} z={_z_now:+.2f} — level factor within normal range"))
                        # Momentum flip detection: was negative, now turning positive (or vice versa)
                        if _z_prev < -0.5 and _z_now > -0.2:
                            _signals.append(("🔄 LEVEL FLIP ↑", "#cc7700",
                                f"{_pc_name} momentum turning hawkish — possible regime transition. "
                                f"5d mom={_mom5:+.3f}, 20d mom={_mom20:+.3f}"))
                        elif _z_prev > 0.5 and _z_now < 0.2:
                            _signals.append(("🔄 LEVEL FLIP ↓", "#cc7700",
                                f"{_pc_name} momentum turning dovish — possible regime transition. "
                                f"5d mom={_mom5:+.3f}, 20d mom={_mom20:+.3f}"))

                    # Slope (PC2): drives steepening/flattening/inversion
                    elif _pi == 1:
                        if _z_now < -1.5:
                            _signals.append(("🔴 EXTREME FLAT/INVERSION", "#8b0000",
                                f"{_pc_name} z={_z_now:+.2f} — curve at historical inversion extreme. "
                                f"Spread tighteners face strong headwind."))
                        elif _z_now < -0.75:
                            _signals.append(("🟠 FLAT REGIME", "#cc5500",
                                f"{_pc_name} z={_z_now:+.2f} — curve flatter than normal. "
                                f"Front spreads may be structurally compressed."))
                        elif _z_now > 1.5:
                            _signals.append(("🟢 STEEP REGIME", "#1a6e2e",
                                f"{_pc_name} z={_z_now:+.2f} — curve steeper than normal. "
                                f"Front spread wideners have slope wind at back."))
                        else:
                            _signals.append(("⚪ SLOPE NEUTRAL", "#555555",
                                f"{_pc_name} z={_z_now:+.2f} — slope within normal range"))
                        # Momentum: flattening vs steepening in progress
                        if _mom5 < 0 and _mom20 < 0:
                            _signals.append(("📉 FLATTENING TREND", "#cc5500",
                                f"{_pc_name} 5d={_mom5:+.3f}, 20d={_mom20:+.3f} — "
                                f"active flattening/inversion building. Short front spreads at risk."))
                        elif _mom5 > 0 and _mom20 > 0:
                            _signals.append(("📈 STEEPENING TREND", "#1a6e2e",
                                f"{_pc_name} 5d={_mom5:+.3f}, 20d={_mom20:+.3f} — "
                                f"active steepening. Long front spreads have momentum support."))

                    # Curvature (PC3): hump/belly risk
                    elif _pi == 2:
                        if abs(_z_now) > 1.5:
                            _dir = "HUMPED" if _z_now > 0 else "INVERTED HUMP"
                            _signals.append((f"🟡 CURVATURE: {_dir}", "#8b7000",
                                f"{_pc_name} z={_z_now:+.2f} — belly at extremes. "
                                f"Fly positions have elevated curvature risk."))

                return _signals

            _banner_signals = _compute_regime_banner(scores, analysis_dt)

            with _regime_banner_slot.container():
                st.markdown("### 📡 Live Regime Detection")
                if _banner_signals:
                    _bcols = st.columns(min(len(_banner_signals), 3))
                    for _bi, (_blabel, _bcolor, _bdetail) in enumerate(_banner_signals[:6]):
                        with _bcols[_bi % len(_bcols)]:
                            st.markdown(
                                f"<div style='background:{_bcolor}22; border-left:4px solid {_bcolor}; "
                                f"padding:8px 12px; border-radius:4px; margin-bottom:6px;'>"
                                f"<b style='color:{_bcolor}'>{_blabel}</b><br>"
                                f"<small style='color:#aaa'>{_bdetail}</small></div>",
                                unsafe_allow_html=True
                            )
                else:
                    st.info("Not enough data to compute regime signals (need ≥ 25 observations).")
            # ── END REGIME BANNER ──────────────────────────────────────────────────


            variance_df = pd.DataFrame({
                'PC': [f'PC{i+1}' for i in range(len(explained_variance_ratio))],
                'Explained Variance (%)': explained_variance_ratio * 100
            })
            variance_df['Cumulative Variance (%)'] = variance_df['Explained Variance (%)'].cumsum()
        
            col_var, col_pca_select = st.columns([1, 1])
        
            with col_var:
                bbg_table(variance_df, use_container_width=True)
            
            default_pc_count = min(3, len(explained_variance_ratio))
        
            with col_pca_select:
                st.subheader("Fair Curve & Hedging Setup")
                pc_count = st.slider(
                    "Select number of Principal Components (PCs) for Fair Curve & Hedging:", 
                    min_value=1, 
                    max_value=len(explained_variance_ratio), 
                    value=default_pc_count,
                    key='pc_slider'
                )
                total_explained = variance_df['Cumulative Variance (%)'].iloc[pc_count - 1]
                st.info(f"The selected **{pc_count} PCs** explain **{total_explained:.2f}%** of the total variance in the spreads. This is the risk model used.")


            # --- Component Loadings Heatmaps (Section 3) ---
            st.header("3. PC Loadings")
        
            # --- 3.1 Spread Loadings (Standard Method) ---
            st.subheader("3.1 PC Loadings Heatmap (PC vs. 3M Spreads)")
            st.markdown("""
            This heatmap shows the **Loadings (Eigenvectors)** of the first few PCs on each **3-Month Spread**. These weights are derived from **Standardized PCA** and represent how each spread contributes to the overall risk factors (Level, Slope, Curvature).
            * **Interpretation of Loadings (Weights):** The value of the loading (weight) indicates the **sensitivity** of that specific spread to the respective Principal Component. A high absolute value means the spread has historically been highly correlated with the movement of that PC factor.
            """)
        
            # Bloomberg theme set globally via rcParams
            fig_spread_loading, ax_spread_loading = plt.subplots(figsize=(12, 6))
        
            # Only plot the first `default_pc_count` PCs in the heatmap
            loadings_spread_plot = loadings_spread.iloc[:, :default_pc_count]
        
            sns.heatmap(
                loadings_spread_plot, 
                annot=True, 
                cmap='coolwarm', 
                fmt=".2f", 
                linewidths=0.5, 
                linecolor='#333333', 
                cbar_kws={'label': 'Loading Weight'}
            )
            ax_spread_loading.set_title(f'3.1 Component Loadings for First {default_pc_count} Principal Components (on Spreads)', fontsize=16)
            ax_spread_loading.set_xlabel('Principal Component')
            ax_spread_loading.set_ylabel('Spread Contract')
            _bbg_fig(fig=fig_spread_loading)
            st.pyplot(fig_spread_loading)

        
            # --- 3.2 Outright Loadings (User Requested Non-Uniform PC1) ---
            st.subheader("3.2 Outright Price Loadings (Non-Uniform PC1)")
            st.markdown("""
            This heatmap is derived from **PCA on Outright Prices (Covariance Matrix)**, not the 3M spreads.
            The purpose is to show the raw, unstandardized **price sensitivity** of each contract to the first few PCs. This often results in a **Non-Uniform Level (PC1)** factor, which can be useful for visualizing the raw change in the entire curve.
            """)
        
            loadings_prices, explained_variance_prices = perform_pca_on_prices(analysis_curve_df)
        
            if loadings_prices is not None:
            
                fig_price_loading, ax_price_loading = plt.subplots(figsize=(12, 6))
            
                loadings_price_plot = loadings_prices.iloc[:, :default_pc_count]
            
                sns.heatmap(
                    loadings_price_plot, 
                    annot=True, 
                    cmap='coolwarm', 
                    fmt=".2f", 
                    linewidths=0.5, 
                    linecolor='#333333', 
                    cbar_kws={'label': 'Loading Weight (Price Sensitivity)'}
                )
                ax_price_loading.set_title(f'3.2 Component Loadings for First {default_pc_count} Principal Components (on Outright Prices - Non-Uniform PC1)', fontsize=16)
                ax_price_loading.set_xlabel('Principal Component')
                ax_price_loading.set_ylabel('Contract')
                _bbg_fig(fig=fig_price_loading)
                st.pyplot(fig_price_loading)
            else:
                st.warning("Outright Price PCA failed. Not enough contracts or data available.")
            
            
            # --- PC Factor Scores Time Series (Section 4) ---
            def plot_pc_scores(scores_df, explained_variance_ratio):
                """Plots the time series of the first 3 PC scores."""
                pc_labels = ['Level (PC1)', 'Slope (PC2)', 'Curvature (PC3)']
                num_pcs = min(3, scores_df.shape[1])
            
                if num_pcs == 0:
                    return None
                
                fig, axes = plt.subplots(nrows=num_pcs, ncols=1, figsize=(15, 4 * num_pcs), sharex=True)
                if num_pcs == 1:
                    axes = [axes]
                
                plt.suptitle("Time Series of Principal Component Scores (Risk Factors)", fontsize=16, y=1.02)
            
                for i in range(num_pcs):
                    ax = axes[i]
                    pc_label = pc_labels[i]
                    variance_pct = explained_variance_ratio[i] * 100
                
                    ax.plot(scores_df.index, scores_df.iloc[:, i], label=f'{pc_label} ({variance_pct:.2f}% Var.)', linewidth=1.5, color=_BBG_CYCLE[i % len(_BBG_CYCLE)])
                    ax.axhline(0, color=_BBG_RED, linestyle='--', linewidth=0.8)
                    ax.set_title(f'{pc_label} Factor Score (Explaining {variance_pct:.2f}% of Spread Variance)', fontsize=14)
                    ax.grid(True, linestyle=':', alpha=0.6)
                    ax.set_ylabel('Score Value')
                    ax.legend(loc='upper left')

                plt.xlabel('Date')
                plt.tight_layout(rect=[0, 0.03, 1, 0.98])
                return fig

            st.header("4. PC Factor Scores Time Series")
            st.markdown("This plot shows the historical movement of the **latent risk factors** (Level, Slope, and Curvature) over the chosen historical range. The scores are derived from the **Spread PCA (3.1)**.")

            fig_scores = plot_pc_scores(scores, explained_variance_ratio)
            if fig_scores:
                _bbg_fig(fig=fig_scores)
                st.pyplot(fig_scores)

            # --- 4B. Rolling PCA Loading Stability ---
            st.subheader("4B. Rolling PCA Loading Stability")
            st.markdown("""
            Checks whether the PCA **eigenvector structure is stable over time**.
            Each point is the PC1 loading re-estimated on a trailing `window` of data.
            Wide swings indicate regime changes (e.g. hiking → easing) where the model
            may need re-calibration or a shorter lookback for PCA fitting.
            """)

            _roll_window_pca = st.slider(
                "Rolling window for loading stability (days):",
                min_value=60, max_value=max(60, len(spreads_3M_df_clean) - 10),
                value=min(252, max(60, len(spreads_3M_df_clean) // 2)),
                key="roll_pca_stability_window"
            )

            _n_roll = len(spreads_3M_df_clean)
            _roll_spreads = spreads_3M_df_clean.values
            _roll_dates = spreads_3M_df_clean.index
            _n_spreads = _roll_spreads.shape[1]
            _pc_names = [f'PC{i+1}' for i in range(min(3, _n_spreads))]

            # Collect rolling loadings for first 3 PCs
            _rolling_loadings = {pc: [] for pc in _pc_names}
            _rolling_dates_out = []

            for _t in range(_roll_window_pca, _n_roll):
                _window_data = _roll_spreads[_t - _roll_window_pca: _t]
                _w_mean = _window_data.mean(axis=0)
                _w_std = _window_data.std(axis=0)
                _w_std[_w_std < 1e-10] = 1.0
                _w_scaled = (_window_data - _w_mean) / _w_std
                try:
                    _pca_r = PCA(n_components=min(3, _n_spreads))
                    _pca_r.fit(_w_scaled)
                    for _pi, _pc in enumerate(_pc_names):
                        # Sign-stabilise: align rolling loading to full-sample loading direction
                        _full_loading = loadings_spread.iloc[:, _pi].values
                        _roll_loading = _pca_r.components_[_pi]
                        if np.dot(_full_loading, _roll_loading) < 0:
                            _roll_loading = -_roll_loading
                        _rolling_loadings[_pc].append(_roll_loading)
                    _rolling_dates_out.append(_roll_dates[_t])
                except Exception:
                    continue

            if _rolling_dates_out:
                _n_cols_stab = min(3, _n_spreads)
                _fig_stab, _axes_stab = plt.subplots(
                    nrows=_n_cols_stab, ncols=1,
                    figsize=(15, 3.5 * _n_cols_stab), sharex=True
                )
                if _n_cols_stab == 1:
                    _axes_stab = [_axes_stab]

                _pc_display = ['Level (PC1)', 'Slope (PC2)', 'Curvature (PC3)']
                _spread_labels = spreads_3M_df_clean.columns.tolist()

                for _pi, _pc in enumerate(_pc_names):
                    _ax = _axes_stab[_pi]
                    _load_matrix = np.array(_rolling_loadings[_pc])   # shape (T_roll, n_spreads)
                    for _si, _slabel in enumerate(_spread_labels):
                        _ax.plot(
                            _rolling_dates_out, _load_matrix[:, _si],
                            linewidth=0.9, alpha=0.7,
                            label=_slabel if _pi == 0 else "_nolegend_"
                        )
                    _ax.axhline(0, color=_BBG_RED, linestyle='--', linewidth=0.7)
                    _ax.set_title(f"{_pc_display[_pi]} — Rolling Loadings ({_roll_window_pca}d window)", fontsize=11)
                    _ax.set_ylabel("Loading")
                    _ax.grid(True, alpha=0.15)

                _axes_stab[0].legend(
                    loc='upper left', bbox_to_anchor=(1.01, 1),
                    fontsize=6, title="Spread", title_fontsize=7
                )
                plt.xlabel("Date")
                plt.tight_layout()
                _bbg_fig(fig=_fig_stab)
                st.pyplot(_fig_stab)

                # Instability metric: std of each loading over time
                _instab_rows = []
                for _pi, _pc in enumerate(_pc_names):
                    _load_matrix = np.array(_rolling_loadings[_pc])
                    for _si, _slabel in enumerate(_spread_labels):
                        _instab_rows.append({
                            "PC": _pc_display[_pi],
                            "Spread": _slabel,
                            "Loading Std (instability)": round(float(np.std(_load_matrix[:, _si])), 4)
                        })
                _instab_df = pd.DataFrame(_instab_rows)
                _most_unstable = _instab_df.sort_values("Loading Std (instability)", ascending=False).head(10)
                with st.expander("Most unstable loadings (top 10)", expanded=False):
                    bbg_table(_most_unstable, use_container_width=True)
                    st.caption("High instability (> 0.10) suggests this spread's factor exposure changes between regimes — treat PCA signals for it with caution.")
            else:
                st.info(f"Not enough data for rolling PCA (need > {_roll_window_pca} rows).")


            # --- Historical Reconstruction (Based on Spread PCA) ---

            # 1. Reconstruct 3M Spreads using only selected PCs
            data_mean = spreads_3M_df_clean.mean()
            data_std = spreads_3M_df_clean.std()
        
            scores_used = scores.values[:, :pc_count]
            loadings_used = loadings_spread.values[:, :pc_count]
        
            # Inverse transform (Scores @ Loadings^T) * StdDev + Mean
            reconstructed_scaled = scores_used @ loadings_used.T
        
            reconstructed_spreads_3M = pd.DataFrame(
                reconstructed_scaled * data_std.values + data_mean.values,
                index=spreads_3M_df_clean.index,
                columns=spreads_3M_df_clean.columns
            )

            # 2. Reconstruct Outright Prices and ALL Derivatives (3M, 6M, 12M)
            historical_outrights_df, historical_spreads_3M_df, historical_butterflies_3M_df, historical_spreads_6M_df, historical_butterflies_6M_df, historical_spreads_12M_df, historical_butterflies_12M_df, historical_double_butterflies_3M_df, historical_double_butterflies_6M_df, historical_double_butterflies_12M_df, spreads_3M_df_no_prefix = reconstruct_prices_and_derivatives(
                analysis_curve_df, 
                reconstructed_spreads_3M, 
                spreads_3M_df_raw, 
                spreads_6M_df, 
                butterflies_3M_df, 
                butterflies_6M_df, 
                spreads_12M_df, 
                butterflies_12M_df,
                double_butterflies_3M_df, 
                double_butterflies_6M_df, 
                double_butterflies_12M_df
            )

            # --------------------------- Mispricing Calculation for Section 8 ---------------------------
            # Combine all historical derivative DFs (those containing Original and PCA columns)
            all_historical_derivatives_list = [
                historical_spreads_3M_df, historical_butterflies_3M_df, historical_double_butterflies_3M_df,
                historical_spreads_6M_df, historical_butterflies_6M_df, historical_double_butterflies_6M_df,
                historical_spreads_12M_df, historical_butterflies_12M_df, historical_double_butterflies_12M_df,
            ]
        
            mispricing_series = calculate_derivative_mispricings(all_historical_derivatives_list, analysis_dt)
            # --------------------------------------------------------------------------------------------------


            # --- Curve Snapshot (Section 5) ---

    with _tab_snap:
            st.header("5. Curve Snapshot (Original vs. PCA Fair Value)")
        
            # FIXED: clear figure lists before populating to prevent duplicates on re-run
            st.session_state.SECTION5_FIGURES = []

            def get_previous_date(df, current_date):
                """Return the last available previous date in df before current_date."""
                try:
                    prev_dates = df.index[df.index < current_date]
                    if len(prev_dates) == 0:
                        return None
                    return prev_dates.max()
                except Exception:
                    return None


            def plot_snapshot(historical_df, derivative_type, current_date, pc_count, collect_for_pdf=True):
                """Plots the market vs PCA fair value snapshot (today vs previous day)."""

                try:
                    # 1. Today's snapshot
                    market_values = historical_df.loc[current_date].filter(like='(Original)')
                    pca_fair_values = historical_df.loc[current_date].filter(like='(PCA)')

                    # 2. Align and merge for plotting (today)
                    comparison = pd.DataFrame({
                        'Original': market_values.values,
                        'PCA Fair': pca_fair_values.values
                    }, index=[col.replace(f' (Original)', '').replace(f'{derivative_type}: ', '') for col in market_values.index])

                    if comparison.empty:
                        st.info(f"No {derivative_type} data available for the selected analysis date {analysis_date.strftime('%Y-%m-%d')} after combining Original and PCA Fair values.")
                        return

                    # 3. Previous-day snapshot
                    prev_date = get_previous_date(historical_df, current_date)
                    prev_series = None
                    if prev_date is not None:
                        try:
                            prev_market = historical_df.loc[prev_date].filter(like='(Original)')
                            prev_series = pd.Series(
                                prev_market.values,
                                index=[col.replace(f' (Original)', '').replace(f'{derivative_type}: ', '') for col in prev_market.index],
                                name='Prev Day'
                            )
                        except KeyError:
                            prev_series = None

                    # --- Plot the Derivative ---
                    fig, ax = plt.subplots(figsize=(15, 7))

                    ax.plot(
                        comparison.index,
                        comparison['Original'],
                        label=f'vwap',
                        marker='o',
                        linestyle='-',
                        linewidth=2.5,
                        color=_BBG_BLUE
                    )
                    ax.plot(
                        comparison.index,
                        comparison['PCA Fair'],
                        label=f'PCA',
                        marker='x',
                        linestyle='--',
                        linewidth=2.5,
                        color=_BBG_RED
                    )

                    # Previous-day original curve, if available
                    if prev_series is not None:
                        ax.plot(
                            prev_series.index,
                            prev_series.values,
                            label=f'settle',
                            marker='s',
                            linestyle='-.',
                            linewidth=2.0,
                            color=_BBG_GREEN
                        )

                    mispricing = comparison['Original'] - comparison['PCA Fair']
                    ax.axhline(0, color='#333333', linestyle='-', linewidth=0.5, alpha=0.7)

                    # Annotate the derivative with the largest absolute mispricing (today)
                    max_abs_mispricing = mispricing.abs().max()
                    if max_abs_mispricing > 0:
                        mispricing_contract = mispricing.abs().idxmax()
                        mispricing_value = mispricing.loc[mispricing_contract] * 100  # Rate %

                        ax.annotate(
                            f"Mispricing: {mispricing_value:.4f} Rate %",
                            (mispricing_contract, comparison.loc[mispricing_contract]['Original']), 
                            textcoords="offset points", 
                            xytext=(0, 10), 
                            ha='center', 
                            fontsize=10, 
                            bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5)
                        )

                    ax.set_title(f'Market {derivative_type} vs. PCA Fair {derivative_type} (Today vs Prev Day)', fontsize=16)
                    ax.set_xlabel(f'{derivative_type} Contract')
                    ax.set_ylabel(f'{derivative_type} Value (Price Difference)')
                    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
                    ax.grid(True, linestyle=':', alpha=0.6)

                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    _bbg_fig(fig=fig)
                    st.pyplot(fig)

                    # Collect Section 5 figure for PDF download
                    if collect_for_pdf:
                        st.session_state.SECTION5_FIGURES.append((fig, f"Section 5 – {derivative_type}"))

                    # --- Detailed Table ---
                    st.markdown(f"###### {derivative_type} Mispricing (Today vs PCA, with Prev Day if available)")
                    detailed_comparison = comparison.copy()
                    detailed_comparison.index.name = f'{derivative_type} Contract'
                    detailed_comparison['Mispricing (Rate %)'] = mispricing * 100 
                    detailed_comparison = detailed_comparison.rename(
                        columns={'Original': f'Original {derivative_type}', 'PCA Fair': f'PCA Fair {derivative_type}'}
                    )

                    # Add previous-day original column if exists
                    if prev_series is not None:
                        prev_align = prev_series.reindex(detailed_comparison.index)
                        detailed_comparison[f'Prev Day Original {derivative_type} ({prev_date.strftime("%Y-%m-%d")})'] = prev_align.values

                    bbg_table(
                        detailed_comparison.style.format({
                            f'Original {derivative_type}': "{:.4f}",
                            f'PCA Fair {derivative_type}': "{:.4f}",
                            'Mispricing (Rate %)': "{:.4f}"
                        }), 
                        use_container_width=True
                    )

                except KeyError:
                     st.error(f"The selected analysis date **{analysis_date.strftime('%Y-%m-%d')}** is not present in the filtered price data for {derivative_type}. Please choose a different date within the historical range.")
            def plot_shock_derivative_snapshot(historical_df, derivative_type, shocked_series, current_date, pc_count, title_suffix=""):
                """
                Plots Original vs PCA Fair vs Shock Scenario for a given derivative family
                on the selected analysis date, using the same x-axis ordering as Section 5.
                """
                try:
                    row = historical_df.loc[current_date]
                except KeyError:
                    st.info(f"No {derivative_type} data available for the selected analysis date in shock snapshot.")
                    return

                market_values = row.filter(like='(Original)')
                pca_fair_values = row.filter(like='(PCA)')

                if market_values.empty or pca_fair_values.empty:
                    st.info(f"{derivative_type}: Missing Original or PCA Fair values for shock snapshot.")
                    return

                # Build a clean instrument index WITHOUT tenor prefixes (e.g. '3M Spread: ')
                base_index = []
                for col in market_values.index:
                    core = col.replace(' (Original)', '')
                    if ': ' in core:
                        core = core.split(': ', 1)[1]
                    base_index.append(core)

                comparison = pd.DataFrame(
                    {
                        'Original': market_values.values,
                        'PCA Fair': pca_fair_values.values,
                    },
                    index=base_index,
                )

                if shocked_series is None or len(shocked_series) == 0:
                    st.info(f"No shocked series supplied for {derivative_type} in shock snapshot.")
                    return

                shocked_aligned = shocked_series.reindex(comparison.index)
                if shocked_aligned.isna().all():
                    st.info(f"Shocked series for {derivative_type} could not be aligned to instruments.")
                    return

                comparison['Shock Scenario'] = shocked_aligned.values

                fig, ax = plt.subplots(figsize=(15, 7))
                ax.plot(comparison.index, comparison['Original'], label=f'{derivative_type} Original', marker='o')
                ax.plot(comparison.index, comparison['PCA Fair'], label=f'{derivative_type} PCA Fair ({pc_count} PCs)', marker='x', linestyle='--')
                ax.plot(comparison.index, comparison['Shock Scenario'], label=f'{derivative_type} Shock {title_suffix}', marker='s', linestyle='-.')

                ax.set_title(f'{derivative_type} Snapshot under Shock {title_suffix}')
                ax.set_xlabel('Instrument')
                ax.set_ylabel('Value (Price Points)')
                ax.grid(True, linestyle=':', alpha=0.6)
                ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                _bbg_fig(fig=fig)
                st.pyplot(fig)

                # Collect Section 9 shock figure for PDF download
                if not st.session_state.SNAPSHOT_READY:
                 st.session_state.SECTION9_FIGURES.append((fig, f"Section 9 – {derivative_type} {title_suffix}".strip()))


            # --- 5.1 Outright Price/Rate Curve Snapshot ---

            st.subheader("5.1 Outright Price/Rate Curve Snapshot")
            try:
                # 1. Get the snapshot for the selected date
                market_prices = historical_outrights_df.loc[analysis_dt].filter(like='(Original)')
                pca_fair_prices = historical_outrights_df.loc[analysis_dt].filter(like='(PCA)')

                # 2. Align and merge for plotting
                curve_comparison = pd.DataFrame({
                    'Original': market_prices.values,
                    'PCA Fair': pca_fair_prices.values
                }, index=[col.replace(' (Original)', '') for col in market_prices.index])

                # --- Plot the Curve (Today vs Previous Day) ---
                fig_curve, ax_curve = plt.subplots(figsize=(15, 7))

                # Today
                ax_curve.plot(
                    curve_comparison.index,
                    curve_comparison['Original'],
                    label=f'Today Original Price ({analysis_dt.strftime("%Y-%m-%d")})',
                    marker='o',
                    linestyle='-',
                    linewidth=2.5,
                    color=_BBG_BLUE
                )
                ax_curve.plot(
                    curve_comparison.index,
                    curve_comparison['PCA Fair'],
                    label=f'Today PCA Fair Price ({pc_count} PCs)',
                    marker='x',
                    linestyle='--',
                    linewidth=2.5,
                    color=_BBG_RED
                )

                # Previous day
                prev_dt = get_previous_date(historical_outrights_df, analysis_dt)
                if prev_dt is not None:
                    try:
                        prev_prices = historical_outrights_df.loc[prev_dt].filter(like='(Original)')
                        prev_cmp = pd.Series(
                            prev_prices.values,
                            index=[col.replace(' (Original)', '') for col in prev_prices.index]
                        )
                        ax_curve.plot(
                            prev_cmp.index,
                            prev_cmp.values,
                            label=f'Prev Day Original Price ({prev_dt.strftime("%Y-%m-%d")})',
                            marker='s',
                            linestyle='-.',
                            linewidth=2.0,
                            color=_BBG_GREEN
                        )
                    except KeyError:
                        pass

                ax_curve.set_title('Market Price Curve vs. PCA Fair Value Curve (Price = 100 - Rate, Today vs Prev Day)', fontsize=16)
                ax_curve.set_xlabel('Contract Maturity')
                ax_curve.set_ylabel('Price (100 - Rate)')
                ax_curve.legend(loc='upper right')
                ax_curve.grid(True, linestyle=':', alpha=0.6)

                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                _bbg_fig(fig=fig_curve)
                st.pyplot(fig_curve)

                # Collect Outright curve figure for Section 5 PDF
                st.session_state.SECTION5_FIGURES.append((fig_curve, "Section 5 – Outright Curve"))

                # --- Detailed Contract Price/Rate Table (Outright) ---
                st.markdown("###### Outright Price and Rate Mispricing")
                detailed_comparison = curve_comparison.copy()
                detailed_comparison.index.name = 'Contract'
                detailed_comparison['Original Rate (%)'] = 100.0 - detailed_comparison['Original']
                detailed_comparison['PCA Fair Rate (%)'] = 100.0 - detailed_comparison['PCA Fair']
                detailed_comparison['Mispricing (Rate %)'] = (detailed_comparison['Original'] - detailed_comparison['PCA Fair']) * 100

                detailed_comparison = detailed_comparison.rename(
                    columns={'Original': 'Original Price', 'PCA Fair': 'PCA Fair Price'}
                )
                detailed_comparison = detailed_comparison[[
                    'Original Price', 'Original Rate (%)', 'PCA Fair Price', 'PCA Fair Rate (%)', 'Mispricing (Rate %)'
                ]]

                bbg_table(
                    detailed_comparison.style.format({
                        'Original Price': "{:.4f}",
                        'PCA Fair Price': "{:.4f}",
                        'Original Rate (%)': "{:.4f}",
                        'PCA Fair Rate (%)': "{:.4f}",
                        'Mispricing (Rate %)': "{:.4f}"
                    }), 
                    use_container_width=True
                )
            except KeyError:
                st.error(f"The selected analysis date **{analysis_date.strftime('%Y-%m-%d')}** is not present in the filtered price data for Outright Prices. Please choose a different date within the historical range.")
            # --------------------------- 3-Month (k=1) Derivatives ---------------------------
            # --- 5.2 Spread Snapshot (3M) ---
            st.subheader("5.2 3M Spread Snapshot (k=1, e.g., Z25-H26)")
            plot_snapshot(historical_spreads_3M_df, "3M Spread", analysis_dt, pc_count)

            # --- 5.3 Butterfly (Fly) Snapshot (3M) ---
            if not historical_butterflies_3M_df.empty:
                st.subheader("5.3 3M Butterfly (Fly) Snapshot (k=1, e.g., Z25-2xH26+M26)")
                plot_snapshot(historical_butterflies_3M_df, "3M Butterfly", analysis_dt, pc_count)
            else:
                st.info("Not enough contracts (need 3 or more) to calculate and plot 3M butterfly snapshot.")
            
            # --- 5.4 Double Butterfly (DBF) Snapshot (3M) --- 
            if not historical_double_butterflies_3M_df.empty:
                st.subheader(r"5.4 3M Double Butterfly (DBF) Snapshot ($k=1$, e.g., $Z25-3 \cdot H26+3 \cdot M26-U26$)")
                plot_snapshot(historical_double_butterflies_3M_df, "3M Double Butterfly", analysis_dt, pc_count)
            else:
                st.info("Not enough contracts (need 4 or more) to calculate and plot 3M double butterfly snapshot.")
            
            # --------------------------- 6-Month (k=2) Derivatives ---------------------------
            # --- 5.5 Spread Snapshot (6M) ---
            st.subheader("5.5 6M Spread Snapshot (k=2, e.g., Z25-M26)")
            plot_snapshot(historical_spreads_6M_df, "6M Spread", analysis_dt, pc_count)
        
            # --- 5.6 Butterfly (Fly) Snapshot (6M) ---
            if not historical_butterflies_6M_df.empty:
                st.subheader("5.6 6M Butterfly (Fly) Snapshot (k=2, e.g., Z25-2xM26+Z26)")
                plot_snapshot(historical_butterflies_6M_df, "6M Butterfly", analysis_dt, pc_count)
            else:
                st.info("Not enough contracts (need 5 or more) to calculate and plot 6M butterfly snapshot.")
            
            # --- 5.7 Double Butterfly (DBF) Snapshot (6M) --- 
            if not historical_double_butterflies_6M_df.empty:
                st.subheader(r"5.7 6M Double Butterfly (DBF) Snapshot ($k=2$, e.g., $Z25-3 \cdot M26+3 \cdot Z26-M27$)")
                plot_snapshot(historical_double_butterflies_6M_df, "6M Double Butterfly", analysis_dt, pc_count)
            else:
                st.info("Not enough contracts (need 7 or more) to calculate and plot 6M double butterfly snapshot.")

            # --------------------------- 12-Month (k=4) Derivatives ---------------------------
            # --- 5.8 Spread Snapshot (12M) ---
            st.subheader("5.8 12M Spread Snapshot (k=4, e.g., Z25-Z26)")
            plot_snapshot(historical_spreads_12M_df, "12M Spread", analysis_dt, pc_count)

            # --- 5.9 Butterfly (Fly) Snapshot (12M) ---
            if not historical_butterflies_12M_df.empty:
                st.subheader("5.9 12M Butterfly (Fly) Snapshot (k=4, e.g., Z25-2xZ26+Z27)")
                plot_snapshot(historical_butterflies_12M_df, "12M Butterfly", analysis_dt, pc_count)
            else:
                st.info("Not enough contracts (need 9 or more) to calculate and plot 12M butterfly snapshot.")

            # --- 5.10 Double Butterfly (DBF) Snapshot (12M) --- 
            if not historical_double_butterflies_12M_df.empty:
                st.subheader(r"5.10 12M Double Butterfly (DBF) Snapshot ($k=4$, e.g., $Z25-3 \cdot Z26+3 \cdot Z27-Z28$)")
                plot_snapshot(historical_double_butterflies_12M_df, "12M Double Butterfly", analysis_dt, pc_count)
            else:
                st.info("Not enough contracts (need 13 or more) to calculate and plot 12M double butterfly snapshot.")

        
            # --------------------------- Download all Section 5 snapshots as PDF ---------------------------
            st.subheader("Download All Section 5 Snapshots as PDF")
            SECTIONS_FIGURES = st.session_state.SECTION5_FIGURES

            if not st.session_state.SECTION5_FIGURES:
                st.info("Generate the Section 5 charts above to enable PDF download.")
            else:
                pdf_buffer_5 = BytesIO()
                with PdfPages(pdf_buffer_5) as pdf:
                    for fig, title in st.session_state.SECTION5_FIGURES:
                        if title:
                            fig.suptitle(title)
                        pdf.savefig(fig, bbox_inches="tight")

                pdf_buffer_5.seek(0)

                st.download_button(
                    label="📥 Download Section 5 Snapshots as PDF",
                    data=pdf_buffer_5,
                    file_name="SOFR.pdf",
                    mime="application/pdf",
                )

            # --------------------------- 6. PCA-Based Hedging Strategy (3M Spreads ONLY - Original Section) ---------------------------

    with _tab_hedge:
            st.header("6. PCA-Based Hedging Strategy (3M Spreads ONLY - Original Section)")
            # FIX: The following text must be wrapped in st.markdown() to prevent NameError
            st.markdown(f"""
            This section calculates the **Minimum Variance Hedge Ratio ($k^*$ )** for a chosen **3M spread** trade, using *another 3M spread* as the hedge. The calculation uses the **Covariance Matrix** of the **3M spreads**, which is **reconstructed using the selected {pc_count} Principal Components**.
            * **Trade:** Long 1 unit of the selected 3M spread.
            * **Hedge:** Short $k^*$ units of the hedging 3M spread.
            * **Volatility:** Expressed as **Rate %** ($1\\% = 100 \\text{{ BPS}}$).
            """)
        
            if spreads_3M_df_clean.shape[1] < 2:
                st.warning("Not enough 3M spreads available to calculate a hedge.")
            else:
                # Drop the prefixes for this section since the function is designed for 3M spreads without prefixes
                spread_labels_3m = spreads_3M_df_no_prefix.columns.tolist()
                trade_selection_3m = st.selectbox(
                    "Select 3M Spread Trade Instrument (T)", 
                    options=spread_labels_3m,
                    key='trade_3m_select'
                )
            
                # Run the hedging analysis
                best_hedge_data_3m, worst_hedge_data_3m, all_results_df_full_3m = calculate_best_and_worst_hedge_3M(
                    trade_selection_3m, loadings_spread, eigenvalues, pc_count, spreads_3M_df_clean
                )
            
                if best_hedge_data_3m is not None:
                    st.subheader(f"Trade: Long 1 unit of **{trade_selection_3m}**")
                
                    # --- Best Hedge ---
                    st.markdown("#### Best Hedge (Minimum Residual Risk)")
                    st.markdown(f"""
                    - **Hedge Instrument (H):** **{best_hedge_data_3m['Hedge Spread']}**
                    - **Hedge Action:** Short **{best_hedge_data_3m['Hedge Ratio (k*)']:.4f}** units.
                    - **Residual Volatility (Rate %):** **{best_hedge_data_3m['Residual Volatility (Rate %)']:.4f} Rate %** (Lowest Risk) # MODIFIED: Name and format update
                    """)
                
                    # --- Worst Hedge ---
                    st.markdown("#### Worst Hedge (Maximum Residual Risk)")
                    st.markdown(f"""
                    - **Hedge Instrument (H):** **{worst_hedge_data_3m['Hedge Spread']}**
                    - **Hedge Action:** Short **{worst_hedge_data_3m['Hedge Ratio (k*)']:.4f}** units.
                    - **Residual Volatility (Rate %):** **{worst_hedge_data_3m['Residual Volatility (Rate %)']:.4f} Rate %** (Highest Risk) # MODIFIED: Name and format update
                    """)
                
                    st.markdown("---")
                    st.markdown("###### Detailed Hedging Results (All 3M Spreads as Hedge Candidates - Sorted by Minimum Variance)")
                    # Use the full results DataFrame directly and sort it for display
                    all_results_df_full_3m = all_results_df_full_3m.sort_values(by='Residual Volatility (Rate %)', ascending=True) # MODIFIED: Sort column update
                
                    bbg_table(
                        all_results_df_full_3m.style.format({
                            'Hedge Ratio (k*)': "{:.4f}",
                            'Residual Volatility (Rate %)': "{:.4f}" # MODIFIED: Name and format update
                        }), 
                        use_container_width=True
                    )
                else:
                    st.warning("3M Hedging calculation failed. Check if enough historical data is available after filtering.")


            # --------------------------- 7. PCA-Based Generalized Hedging Strategy (Minimum Variance) ---------------------------
            st.header("7. PCA-Based Generalized Hedging Strategy (Minimum Variance)")
            st.markdown(f"""
            This section calculates the **Minimum Variance Hedge Ratio ($k^*$ )** for *any* derivative trade, using *any* other derivative as a hedge. The calculation is based on the **full covariance matrix** of all derivatives, which is **reconstructed using the selected {pc_count} Principal Components** derived from the 3M Spreads.
            * **Trade:** Long 1 unit of the selected instrument.
            * **Hedge:** Short $k^*$ units of the hedging instrument.
            * **Volatility:** Expressed as **Rate %** ($1\\% = 100 \\text{{ BPS}}$).
            """) # Note on Rate % update

            # --- HEDGING DATA PREPARATION (FOR SECTIONS 7 & 8) ---
        
            # 1. Combine all historical derivative time series into one DataFrame
            # **CRITICAL: Ensure all derivatives have unique, explicit prefixes**
            all_derivatives_list = [
                spreads_3M_df_raw.rename(columns=lambda x: f"3M Spread: {x}"), # Uses raw spread DF (no prefix)
                butterflies_3M_df.rename(columns=lambda x: f"3M Fly: {x}"),
                double_butterflies_3M_df.rename(columns=lambda x: f"3M Double Fly: {x}"), 
            
                spreads_6M_df.rename(columns=lambda x: f"6M Spread: {x}"),
                butterflies_6M_df.rename(columns=lambda x: f"6M Fly: {x}"),
                double_butterflies_6M_df.rename(columns=lambda x: f"6M Double Fly: {x}"), 
            
                spreads_12M_df.rename(columns=lambda x: f"12M Spread: {x}"),
                butterflies_12M_df.rename(columns=lambda x: f"12M Fly: {x}"),
                double_butterflies_12M_df.rename(columns=lambda x: f"12M Double Fly: {x}") 
            ]
        
            # Only keep non-empty dataframes
            all_derivatives_df_raw = pd.concat([df for df in all_derivatives_list if not df.empty], axis=1)

            # 2. Calculate Generalized Covariance Matrix (Sigma_Raw_df) and Loadings (loadings_df_gen)
            Sigma_Raw_df, all_derivatives_df_aligned, loadings_df_gen = calculate_derivatives_covariance_generalized(
                all_derivatives_df_raw, scores, eigenvalues, pc_count
            )
        
        
            if not Sigma_Raw_df.empty and Sigma_Raw_df.shape[1] > 1:
            
                # Get the list of all available derivative instruments
                all_derivatives_labels = Sigma_Raw_df.columns.tolist()
            
                trade_selection_gen = st.selectbox(
                    "Select Trade Instrument (T)", 
                    options=all_derivatives_labels,
                    key='trade_gen_select'
                )
            
                # Run the generalized hedging analysis
                best_hedge_data_gen, worst_hedge_data_gen, all_results_df_full_gen = calculate_best_and_worst_hedge_generalized(
                    trade_selection_gen, Sigma_Raw_df
                )
            
                if best_hedge_data_gen is not None:
                    st.subheader(f"Trade: Long 1 unit of **{trade_selection_gen}**")
                
                    # --- Best Hedge ---
                    st.markdown("#### Best Hedge (Minimum Residual Risk)")
                    st.markdown(f"""
                    - **Hedge Instrument (H):** **{best_hedge_data_gen['Hedge Instrument']}**
                    - **Hedge Action:** Short **{best_hedge_data_gen['Hedge Ratio (k*)']:.4f}** units.
                    - **Residual Volatility (Rate %):** **{best_hedge_data_gen['Residual Volatility (Rate %)']:.4f} Rate %** (Lowest Risk) # MODIFIED: Name and format update
                    """)
                
                    # --- Worst Hedge ---
                    st.markdown("#### Worst Hedge (Maximum Residual Risk)")
                    st.markdown(f"""
                    - **Hedge Instrument (H):** **{worst_hedge_data_gen['Hedge Instrument']}**
                    - **Hedge Action:** Short **{worst_hedge_data_gen['Hedge Ratio (k*)']:.4f}** units.
                    - **Residual Volatility (Rate %):** **{worst_hedge_data_gen['Residual Volatility (Rate %)']:.4f} Rate %** (Highest Risk) # MODIFIED: Name and format update
                    """)
                
                    st.markdown("---")
                    st.markdown("###### Detailed Hedging Results (All Derivatives as Hedge Candidates - Sorted by Minimum Variance)")
                    # Use the full results DataFrame directly and sort it for display
                    all_results_df_full_gen = all_results_df_full_gen.sort_values(by='Residual Volatility (Rate %)', ascending=True) # MODIFIED: Sort column update
                
                    bbg_table(
                        all_results_df_full_gen.style.format({
                            'Hedge Ratio (k*)': "{:.4f}",
                            'Residual Volatility (Rate %)': "{:.4f}" # MODIFIED: Name and format update
                        }), 
                        use_container_width=True
                    )
                else:
                    st.warning("Generalized Minimum Variance Hedging calculation failed for the selected trade. Check if enough historical data is available after filtering.")

            # ── 7B. ROLLING CORRELATION STABILITY ──────────────────────────────────
            # Motivation: the hedge ratios in Section 7 come from Sigma_Raw_df, which
            # is estimated over the FULL calibration window.  But correlations between
            # derivatives shift with regime (see Section 10B).  This panel shows how
            # the pairwise correlations between your selected trade and its top hedges
            # have evolved over time — making it immediately visible when a hedge ratio
            # calibrated in one regime will not hold in the current regime.
            #
            # Methodology:
            #   - Take the top-5 instruments by |correlation| with the user-selected trade
            #     from Sigma_Raw_df (= the instruments the Section 7 engine would rank highest).
            #   - Compute rolling 60-day Pearson correlation between each pair and the trade
            #     using the raw (not standardised) derivative time series.
            #   - Plot as time series. A flat line = stable hedge. A drifting/volatile line
            #     = hedge ratio is regime-dependent and should be used with caution.
            #
            if not Sigma_Raw_df.empty and not all_derivatives_df_aligned.empty:
                st.subheader("7B. Rolling Correlation Stability")
                st.markdown("""
                How stable are the correlations between your trade instrument and its best hedges?
                A **flat line** = hedge ratio is robust across regimes.
                A **drifting or volatile line** = the Section 7 ratio was calibrated on a different
                regime and may not hold today. Cross-reference with Section 4B loading stability.
                """)

                _corr_roll_window = st.slider(
                    "Rolling window for correlation (days):",
                    min_value=20, max_value=min(252, len(all_derivatives_df_aligned) - 5),
                    value=min(60, len(all_derivatives_df_aligned) // 3),
                    key="corr_stability_window"
                )

                # Top-5 instruments by absolute full-sample correlation with the trade
                if trade_selection_gen in Sigma_Raw_df.index:
                    _trade_var = Sigma_Raw_df.loc[trade_selection_gen, trade_selection_gen]
                    _corr_series = {}
                    for _hinst in Sigma_Raw_df.columns:
                        if _hinst == trade_selection_gen:
                            continue
                        _h_var = Sigma_Raw_df.loc[_hinst, _hinst]
                        _cov   = Sigma_Raw_df.loc[trade_selection_gen, _hinst]
                        _denom = np.sqrt(_trade_var * _h_var)
                        if _denom > 1e-12:
                            _corr_series[_hinst] = abs(_cov / _denom)

                    _top5 = sorted(_corr_series, key=_corr_series.get, reverse=True)[:5]

                    # Get raw derivative time series for rolling correlation
                    _avail = [c for c in [trade_selection_gen] + _top5
                              if c in all_derivatives_df_aligned.columns]

                    if len(_avail) >= 2:
                        _raw_sub = all_derivatives_df_aligned[_avail].dropna()

                        if len(_raw_sub) > _corr_roll_window + 5:
                            _fig_corr, _ax_corr = plt.subplots(figsize=(14, 5))

                            for _hinst in _avail[1:]:   # skip the trade itself
                                _roll_corr = _raw_sub[trade_selection_gen].rolling(
                                    _corr_roll_window, min_periods=max(10, _corr_roll_window // 3)
                                ).corr(_raw_sub[_hinst])

                                # Full-sample correlation as reference line
                                _full_corr_val = _corr_series.get(_hinst, np.nan)
                                _sign = np.sign(
                                    Sigma_Raw_df.loc[trade_selection_gen, _hinst]
                                ) if _hinst in Sigma_Raw_df.columns else 1
                                _roll_corr_signed = _roll_corr  # corr() returns signed value

                                _ax_corr.plot(
                                    _roll_corr_signed.index, _roll_corr_signed,
                                    linewidth=1.3, alpha=0.85,
                                    label=f"{_hinst} (full={_sign*_full_corr_val:.2f})"
                                )

                            _ax_corr.axhline(0,  color=_BBG_WHITE, linewidth=0.7, linestyle='--', alpha=0.4)
                            _ax_corr.axhline( 0.8, color=_BBG_GREEN, linewidth=0.5, linestyle=':', alpha=0.5)
                            _ax_corr.axhline(-0.8, color=_BBG_RED,   linewidth=0.5, linestyle=':', alpha=0.5)
                            _ax_corr.axvline(pd.Timestamp(analysis_dt), color=_BBG_AMBER,
                                             linewidth=1.2, linestyle='--', alpha=0.7, label="Analysis date")

                            _ax_corr.set_ylim(-1.05, 1.05)
                            _ax_corr.set_title(
                                f"Rolling {_corr_roll_window}d Correlation: {trade_selection_gen} vs Top-5 Hedges",
                                fontsize=11
                            )
                            _ax_corr.set_ylabel("Pearson Correlation")
                            _ax_corr.legend(loc='lower left', fontsize=7, ncol=2)
                            _ax_corr.grid(True, alpha=0.12)
                            _bbg_fig(fig=_fig_corr)
                            st.pyplot(_fig_corr)

                            # Instability summary: std of rolling corr over last 252 days
                            _instab_summary = []
                            for _hinst in _avail[1:]:
                                _rc = _raw_sub[trade_selection_gen].rolling(
                                    _corr_roll_window, min_periods=max(10, _corr_roll_window // 3)
                                ).corr(_raw_sub[_hinst])
                                _recent_rc = _rc.last('252D').dropna()
                                _full_c    = Sigma_Raw_df.loc[trade_selection_gen, _hinst] / \
                                             max(np.sqrt(Sigma_Raw_df.loc[trade_selection_gen, trade_selection_gen] *
                                                         Sigma_Raw_df.loc[_hinst, _hinst]), 1e-12)
                                _instab_summary.append({
                                    "Hedge": _hinst,
                                    "Full-sample ρ": round(_full_c, 3),
                                    "Current ρ": round(float(_rc.iloc[-1]), 3) if len(_rc) > 0 else np.nan,
                                    "1Y Corr Std (instability)": round(float(_recent_rc.std()), 3)
                                        if len(_recent_rc) > 5 else np.nan,
                                    "Regime Risk": "⚠️ HIGH" if len(_recent_rc) > 5 and _recent_rc.std() > 0.15
                                        else ("🟡 MODERATE" if len(_recent_rc) > 5 and _recent_rc.std() > 0.08
                                        else "✅ LOW")
                                })
                            _instab_df_corr = pd.DataFrame(_instab_summary)
                            bbg_table(_instab_df_corr, use_container_width=True)
                            st.caption(
                                "**Corr Std > 0.15** = correlation is highly regime-dependent. "
                                "The Section 7 hedge ratio for that instrument should be treated as an estimate, "
                                "not a stable number. Consider shortening the PCA calibration window "
                                "or using Section 8 factor hedging instead."
                            )
                        else:
                            st.info(f"Need more data than {_corr_roll_window}d window for rolling correlation.")
                    else:
                        st.info("Insufficient instruments available for correlation analysis.")
            # ── END 7B ─────────────────────────────────────────────────────────────


            # --------------------------- 8. PCA-Based Factor Hedging Strategy (Sensitivity Hedging - MODIFIED) ---------------------------
            st.header("8. PCA-Based Factor Hedging Strategy (Sensitivity Hedging)")
            st.markdown(f"""
            This strategy uses the Level, Slope, and Curvature factors (PC1, PC2, PC3) to identify hedges that neutralize specific factor exposures.
            * **Factor Exposures:** Standardized sensitivities (Beta) to the principal components.
            * **Volatility/Mispricing:** Expressed as **Rate %** ($1\\% = 100 \\text{{ BPS}}$).
            """) 
        
            # 1. Calculate Factor Sensitivities (L_D columns renamed)
            factor_sensitivities_df = calculate_factor_sensitivities(loadings_df_gen, pc_count)
        
            if not factor_sensitivities_df.empty and not Sigma_Raw_df.empty:
            
                # --- User Selections ---
                all_derivatives_labels_factor = factor_sensitivities_df.index.tolist()
                factor_names = factor_sensitivities_df.columns.tolist()
            
                col_trade_select, col_factor_select = st.columns(2)
            
                with col_trade_select:
                    trade_selection_factor = st.selectbox(
                        "Select Trade Instrument (T)", 
                        options=all_derivatives_labels_factor,
                        key='trade_factor_select'
                    )
                
                with col_factor_select:
                    st.info("Results will display the best hedge for all factors.")

                st.markdown("---")

                # --- 8.1 NEW: Triple Factor Neutralization Check ---
                st.subheader(f"8.1 **Triple Factor Neutralization** Check (Trade: {trade_selection_factor})")
                st.markdown(r"""
                This checks if any *single* hedge instrument **($H$)** can simultaneously neutralize the trade's **Level, Slope, and Curvature** exposure. This requires the ratio of factor sensitivities ($\frac{E_{PCi}(T)}{E_{PCi}(H)}$) to be nearly identical for all three factors, resulting in a single hedge ratio ($k$):
                $$\frac{E_{Level}(T)}{E_{Level}(H)} \approx \frac{E_{Slope}(T)}{E_{Slope}(H)} \approx \frac{E_{Curvature}(T)}{E_{Curvature}(H)} = k$$
                """)
            
                # Check for Triple Factor Hedge
                triple_hedge_check_result = find_perfect_factor_hedge(
                    trade_selection_factor, 
                    factor_sensitivities_df, 
                    mispricing_series, 
                    pc_count
                )
            
                if triple_hedge_check_result['result'] is not None:
                    res = triple_hedge_check_result['result']
                
                    # --- Display the results in a clear table ---
                    triple_data = {
                        'Metric': [
                            'Trade Instrument', 
                            'Hedge Instrument (H)', 
                            'Hedge Action',
                            'Hedge Ratio (|k|)',
                            'Trade PC1 (Level) Sensitivity', 
                            'Hedge PC1 (Level) Sensitivity', 
                            'Trade PC2 (Slope) Sensitivity',
                            'Hedge PC2 (Slope) Sensitivity',
                            'Trade PC3 (Curvature) Sensitivity',
                            'Hedge PC3 (Curvature) Sensitivity',
                            'Hedge Mispricing (Rate %)',
                            'Max Relative K Spread (Tolerance Check)'
                        ],
                        'Value': [
                            trade_selection_factor,
                            res['Hedge Instrument'],
                            f"{res['Hedge Action']} {res['Hedge Ratio (|k|)']:.4f} units",
                            f"{res['Hedge Ratio (|k|)']:.4f}",
                            f"{res['Trade PC1 Sensitivity']:.4f}",
                            f"{res['Hedge PC1 Sensitivity']:.4f}",
                            f"{res['Trade PC2 Sensitivity']:.4f}",
                            f"{res['Hedge PC2 Sensitivity']:.4f}",
                            f"{res['Trade PC3 Sensitivity']:.4f}",
                            f"{res['Hedge PC3 Sensitivity']:.4f}",
                            f"{res['Hedge Mispricing (Rate %)']:.4f}" if not np.isnan(res['Hedge Mispricing (Rate %)']) else 'N/A',
                            f"{res['Max Relative K Spread']:.2%}"
                        ]
                    }
                
                    st.success(f"**PERFECT FACTOR HEDGE FOUND!** The instrument **{res['Hedge Instrument']}** can neutralize the first three factors simultaneously.")
                    bbg_st_table(pd.DataFrame(triple_data).set_index('Metric'))
                
                else:
                    st.info(triple_hedge_check_result['error'])

                st.markdown("---") 

                # --- 8.2 Single Factor Neutralization Results ---
                st.subheader(f"8.2 **Single Factor Neutralization** Results (Trade: {trade_selection_factor})")
                st.markdown(f"The best hedge for each single factor minimizes the total remaining (residual) risk after neutralizing that specific factor's exposure.")
            
                summary_results = []
            
                # --- Run Hedging Analysis for All Factors ---
                for target_factor in factor_names:
                    factor_results_df, error_msg = calculate_all_factor_hedges(
                        trade_selection_factor, target_factor, factor_sensitivities_df, Sigma_Raw_df
                    )
                
                    if error_msg:
                        continue
                
                    # Filter out hedges with near-zero factor sensitivity (Ratio is meaningless/too large)
                    factor_results_df_clean = factor_results_df.dropna(subset=['Residual Volatility (Rate %)']) # MODIFIED: Column name update
                
                    if not factor_results_df_clean.empty:
                        # Find the SINGLE best hedge (minimum residual volatility) for the current factor
                        best_hedge_row = factor_results_df_clean.iloc[0]
                    
                        # --- FETCH HEDGE MISPRICING ---
                        best_hedge_instrument = best_hedge_row['Hedge Instrument']
                        # Use .get() to safely retrieve mispricing, defaulting to NaN if not found
                        hedge_mispricing = mispricing_series.get(best_hedge_instrument, np.nan) 
                        # ----------------------------
                    
                        # Determine the Hedge Action (Short/Long) based on the Hedge Ratio
                        k_factor_value = best_hedge_row[f'Factor Hedge Ratio (k_factor)']
                        if k_factor_value > 0:
                            hedge_action = 'Short'
                        elif k_factor_value < 0:
                            hedge_action = 'Long'
                        else:
                            hedge_action = 'N/A' # Should be rare if k_factor is non-zero
                        
                        summary_results.append({
                            'Factor to Neutralize': target_factor,
                            'Hedge Instrument': best_hedge_row['Hedge Instrument'],
                            'Hedge Action': hedge_action,
                            'Hedge Ratio (|k|)': abs(k_factor_value),
                            'Residual Volatility (Rate %)': best_hedge_row['Residual Volatility (Rate %)'], # MODIFIED: Column name update
                            'Hedge Mispricing (Rate %)': hedge_mispricing, # MODIFIED: Column name update
                            'Trade Sensitivity': best_hedge_row['Trade Sensitivity'],
                            'Hedge Sensitivity': best_hedge_row['Hedge Sensitivity']
                        })

                # --- Display Summary Table of Best Factor Hedges ---
                if summary_results:
                    summary_df = pd.DataFrame(summary_results).sort_values(by='Residual Volatility (Rate %)', ascending=True) # MODIFIED: Sort column update
                
                    # MODIFICATION: Insert 'Hedge Mispricing (BPS)' into the displayed columns
                    bbg_table(
                        summary_df[[
                            'Factor to Neutralize', 
                            'Hedge Instrument', 
                            'Hedge Action', 
                            'Hedge Ratio (|k|)', 
                            'Residual Volatility (Rate %)', # MODIFIED: Column name update
                            'Hedge Mispricing (Rate %)', # MODIFIED: Column name update
                            'Trade Sensitivity', 
                            'Hedge Sensitivity'
                        ]].style.format({
                            'Trade Sensitivity': "{:.4f}",
                            'Hedge Sensitivity': "{:.4f}",
                            'Hedge Ratio (|k|)': "{:.4f}",
                            'Residual Volatility (Rate %)': "{:.4f}", # MODIFIED: Format to 4 decimals for clarity
                            'Hedge Mispricing (Rate %)': "{:.4f}", # MODIFIED: Format to 4 decimals for clarity
                        }),
                        use_container_width=True
                    )
                
                    # --- NEW EXPLANATION OF THE TABLE ---
                    st.markdown("---")
                    st.markdown("### 💡 Explanation of Single Factor Hedging Results")
                    st.markdown(r"""
                    The table in **Section 8.2** shows the **ideal hedge instrument** to neutralize the risk from a *single, specific market factor* (Level, Slope, or Curvature).

                    A hedge is considered 'better' in this context because it **minimizes the Residual Volatility** for that specific factor's risk:
                
                    1.  **Factor Neutralization:** The `Factor Hedge Ratio (|k|)` is calculated as the ratio of the Trade's sensitivity to the Hedge's sensitivity for the target factor ($\frac{E_{Factor}(T)}{E_{Factor}(H)}$). When you enter the trade and the hedge at this ratio, the total portfolio exposure to that factor becomes zero.
                
                    2.  **Minimum Residual Volatility:** While the factor risk is zeroed out, residual risk from **all other factors** remains. The instrument displayed is the one that achieves that **factor neutrality** while simultaneously resulting in the **lowest overall residual risk** (as measured by `Residual Volatility (Rate %)`). This is determined using the full covariance matrix (Section 7's $\Sigma_{Raw}$) to precisely calculate the remaining, unhedged volatility.

                    3.  **Hedge Mispricing (Rate %):** This column provides the key trading signal. It shows the difference between the market price of the hedge instrument and its PCA Fair Value (`Original Price - PCA Fair Value`).
                        * **A high absolute mispricing** combined with a **low residual volatility** suggests a potentially **high-quality, high-alpha trade**. You are using an attractively mispriced instrument to neutralize a major risk factor, leaving only minimal idiosyncratic (unexplained) risk.
                    """)
                    # --- END NEW EXPLANATION ---
            
                else:
                     st.info(f"No valid factor hedge candidates found for trade **{trade_selection_factor}** across Level, Slope, or Curvature.")


                st.markdown("---") 

                # --- 8.3 Filtered Universe of Potential Hedges ---
                st.header("8.3 Filtered Universe of Potential Hedges")
                st.markdown("""
                This table provides a comprehensive view of all available derivative instruments, categorized by type (Spread, Fly, Double Fly). It presents the instrument's **risk attributes** (Sensitivities, Total Volatility) and its **trading signal** (Mispricing) to help identify high-quality hedging instruments.
            
                * **Note:** The hedging model is based on PCA of **Spreads/Derivatives**. Outright contracts are excluded here as they do not have the same standardized Level/Slope/Curvature factor exposures.
                """)
            
                # 1. Create the universe table
                instrument_universe_df = create_instrument_universe_table(factor_sensitivities_df, Sigma_Raw_df, mispricing_series)
            
                if not instrument_universe_df.empty:
                
                    # 2. Add Filter
                    derivative_options = ['All Derivatives'] + sorted(instrument_universe_df['Derivative Group'].unique().tolist())
                
                    # Exclude 'Other' if it's the only option or empty
                    if len(derivative_options) > 2 and 'Other' in derivative_options:
                        derivative_options.remove('Other')
                    
                    selected_group = st.radio(
                        "Select Derivative Group to View:", 
                        options=derivative_options,
                        index=0,
                        key='derivative_filter_83',
                        horizontal=True
                    )
                
                    # 3. Filter the table
                    if selected_group != 'All Derivatives':
                        filtered_df = instrument_universe_df[instrument_universe_df['Derivative Group'] == selected_group]
                    else:
                        filtered_df = instrument_universe_df.copy()
                
                    # 4. Prepare for display and sort
                    display_df = filtered_df.drop(columns=['Derivative Group']).sort_values(
                        by='Total Volatility (Rate %)', 
                        ascending=False
                    )
                
                    # 5. Display the table
                    st.markdown(f"###### Attributes for: **{selected_group}** (Total Instruments: {len(display_df)})")
                    bbg_table(
                        display_df.style.format({
                            'Level Sensitivity': "{:.4f}",
                            'Slope Sensitivity': "{:.4f}",
                            'Curvature Sensitivity': "{:.4f}",
                            'Total Volatility (Rate %)': "{:.4f}",
                            'Mispricing (Rate %)': "{:.4f}"
                        }).background_gradient(
                            subset=['Mispricing (Rate %)'], 
                            cmap='coolwarm', 
                            vmax=display_df['Mispricing (Rate %)'].abs().max() * 0.5 if not display_df['Mispricing (Rate %)'].abs().empty else 0.5,
                            vmin=-display_df['Mispricing (Rate %)'].abs().max() * 0.5 if not display_df['Mispricing (Rate %)'].abs().empty else -0.5 # Gradient strength
                        ),
                        use_container_width=True
                    )
                
                    st.markdown("""
                    ### 🎯 How to use this table for hedging:
                    * **Identify Mispriced Hedges (Signal):** Look for instruments with a high absolute **Mispricing (Rate %)** (deep red or deep blue in the background gradient). This is your potential *alpha* source.
                    * **Assess Factor Exposure (Risk Match):** Check the **Level, Slope, and Curvature Sensitivity**. If your main trade is exposed to the Slope factor, you'll need a hedge with a strong, opposite Slope Sensitivity.
                    * **Evaluate Hedge Impact (Risk):** The **Total Volatility (Rate %)** is the inherent risk of the hedge instrument itself. Using a high volatility hedge (top of the list) will require a more precise hedge ratio to avoid adding more risk than you remove.
                    """)

                else:
                    st.info("Instrument universe table could not be created. Ensure enough historical data is available.")

             
                # Display full sensitivities table as before for reference
                st.markdown("---")
                st.subheader(f"Factor Sensitivities (Standardized Beta) Table for Reference")
                st.markdown("This shows the raw input exposures used for the ratio calculation. Note: Outright prices are not included here as factor hedging applies to the derivatives used in the PCA structure.")
            
                bbg_table(
                    factor_sensitivities_df.style.format("{:.4f}"),
                    use_container_width=True
                )

        

    # --------------------------- 8.4 Historical Backtest of Trade + Hedge Pair ---------------------------
            st.markdown("---")
            st.subheader("8.4 Historical Backtest: Trade + Hedge Portfolio")

            st.markdown(r"""
            This section lets you **simulate the historical behaviour** of a **Trade + Hedge** combination:

            * You pick:
              - A **trade instrument** and direction/size.
              - A **hedge instrument** and hedge ratio $k$ (portfolio is $P = T - kH$).
            * The tool then:
              - Builds daily **P&L time series** for Trade, Hedge, and the combined portfolio.
              - Computes **volatility before vs after hedging**.
              - Shows the **cumulative P&L** evolution through time.

            This is exactly how a bank desk sanity-checks hedges before putting risk on.
            """)

            # --- Helper: safely retrieve historical price series for a derivative label ---
            def _get_price_series_for_label(derivative_label: str):
                """
                Safely retrieve the historical price series for a derivative label like:
                "3M Spread: Z25-Z26", "6M Fly: Z25-Z27", etc.

                This version:
                  ✔ uses the *_df naming convention consistently
                  ✔ checks globals() before accessing
                  ✔ gracefully returns None if data is missing
                """
                if ":" not in derivative_label:
                    return None

                prefix, rest = derivative_label.split(": ", 1)
                type_key = prefix.strip()

                # Map instrument family to the standard *_df historical dataframes
                hist_map_names = {
                    "3M Spread": "historical_spreads_3M_df",
                    "3M Fly": "historical_butterflies_3M_df",
                    "3M Double Fly": "historical_double_butterflies_3M_df",

                    "6M Spread": "historical_spreads_6M_df",
                    "6M Fly": "historical_butterflies_6M_df",
                    "6M Double Fly": "historical_double_butterflies_6M_df",

                    "12M Spread": "historical_spreads_12M_df",
                    "12M Fly": "historical_butterflies_12M_df",
                    "12M Double Fly": "historical_double_butterflies_12M_df",
                }

                if type_key not in hist_map_names:
                    return None

                dataset_name = hist_map_names[type_key]

                # Ensure the dataframe actually exists in the global namespace
                if dataset_name not in globals():
                    return None

                df = globals()[dataset_name]
                if df is None or df.empty:
                    return None

                col_name = f"{derivative_label} (Original)"
                if col_name not in df.columns:
                    return None

                return df[col_name].dropna()

            def _compute_hedged_pnl_series(
                trade_label: str,
                hedge_label: str,
                trade_direction: str,
                trade_units: float,
                hedge_ratio_k: float
            ):
                """
                Build daily P&L for Trade, Hedge and Portfolio:
                    P_T = sign_T * N_T * ΔT
                    P_H = -k * N_T * ΔH
                Portfolio PnL = P_T + P_H

                This uses daily *differences* in the instrument prices (already spreads/flies).
                """

                trade_series = _get_price_series_for_label(trade_label)
                hedge_series = _get_price_series_for_label(hedge_label)

                if trade_series is None:
                    st.error(f"Historical data not found for trade instrument: {trade_label}")
                    return None
                if hedge_series is None:
                    st.error(f"Historical data not found for hedge instrument: {hedge_label}")
                    return None

                df_prices = pd.concat(
                    [trade_series.rename("Trade"), hedge_series.rename("Hedge")],
                    axis=1
                ).dropna()

                if df_prices.empty:
                    st.error("No overlapping history between trade and hedge instruments.")
                    return None

                dTrade = df_prices["Trade"].diff().dropna()
                dHedge = df_prices["Hedge"].diff().dropna()

                pnl_df = pd.concat(
                    [dTrade.rename("dTrade"), dHedge.rename("dHedge")],
                    axis=1
                ).dropna()

                sign_T = 1 if trade_direction == "Long" else -1

                pnl_df["Trade PnL"] = sign_T * trade_units * pnl_df["dTrade"]
                pnl_df["Hedge PnL"] = -hedge_ratio_k * trade_units * pnl_df["dHedge"]
                pnl_df["Portfolio PnL"] = pnl_df["Trade PnL"] + pnl_df["Hedge PnL"]

                return pnl_df

            # --- UI for backtest ---
            if not Sigma_Raw_df.empty and Sigma_Raw_df.shape[1] > 1:

                backtest_labels = Sigma_Raw_df.columns.tolist()

                col_bt1, col_bt2 = st.columns(2)
                with col_bt1:
                    backtest_trade = st.selectbox(
                        "Backtest Trade Instrument",
                        options=backtest_labels,
                        key="backtest_trade"
                    )
                with col_bt2:
                    backtest_hedge = st.selectbox(
                        "Backtest Hedge Instrument",
                        options=[x for x in backtest_labels if x != backtest_trade],
                        key="backtest_hedge"
                    )

                col_bt3, col_bt4, col_bt5 = st.columns(3)
                with col_bt3:
                    backtest_trade_dir = st.selectbox(
                        "Trade Direction (for backtest)",
                        ["Long", "Short"],
                        key="backtest_trade_dir"
                    )
                with col_bt4:
                    backtest_trade_units = st.number_input(
                        "Trade Size (units)",
                        min_value=0.1,
                        value=1.0,
                        step=0.5,
                        key="backtest_trade_units"
                    )
                with col_bt5:
                    # Default k* from covariance for convenience
                    Var_H = Sigma_Raw_df.loc[backtest_hedge, backtest_hedge]
                    Cov_TH = Sigma_Raw_df.loc[backtest_trade, backtest_hedge]
                    default_k = float(Cov_TH / Var_H) if Var_H > 1e-9 else 0.0

                    backtest_k = st.number_input(
                        "Hedge Ratio k (portfolio = T - kH)",
                        value=default_k,
                        step=0.1,
                        format="%.4f",
                        key="backtest_k"
                    )

                if st.button("Run Historical Backtest", key="run_backtest"):
                    pnl_df = _compute_hedged_pnl_series(
                        trade_label=backtest_trade,
                        hedge_label=backtest_hedge,
                        trade_direction=backtest_trade_dir,
                        trade_units=backtest_trade_units,
                        hedge_ratio_k=backtest_k
                    )

                    if pnl_df is not None and not pnl_df.empty:

                        trade_vol = pnl_df["Trade PnL"].std() * 100
                        port_vol = pnl_df["Portfolio PnL"].std() * 100
                        vol_red_pct = (1 - port_vol / trade_vol) * 100 if trade_vol > 0 else float("nan")

                        st.markdown("### Volatility Before vs After Hedging")
                        st.markdown(f"""
                        - **Trade-only Volatility:** `{trade_vol:.4f}` Rate %  
                        - **Hedged Portfolio Vol:** `{port_vol:.4f}` Rate %  
                        - **Volatility Reduction:** `{vol_red_pct:.2f}%`
                        """)

                        cumulative = pnl_df.cumsum()

                        fig_bt, ax_bt = plt.subplots(figsize=(12, 5))
                        ax_bt.plot(cumulative.index, cumulative["Trade PnL"], label="Trade P&L")
                        ax_bt.plot(cumulative.index, cumulative["Hedge PnL"], label="Hedge P&L")
                        ax_bt.plot(cumulative.index, cumulative["Portfolio PnL"], label="Portfolio P&L", linewidth=2)

                        ax_bt.axhline(0, color=_BBG_GRAY, linewidth=0.8, linestyle="--")
                        ax_bt.set_title(
                            f"Cumulative P&L Backtest — {backtest_trade_dir} {backtest_trade_units} {backtest_trade} "
                            f"vs Hedge (k={backtest_k:.4f} × {backtest_hedge})"
                        )
                        ax_bt.set_ylabel("Cumulative P&L")
                        ax_bt.grid(True, linestyle=":", alpha=0.5)
                        ax_bt.legend()

                        _bbg_fig(fig=fig_bt)
                        st.pyplot(fig_bt)
            else:
                st.info("Backtest unavailable: covariance matrix for derivatives is empty.")

    # ------------------- Section 9: PCA Factor Shocks & Whole-Instrument Anchoring -------------------

with _tab_macro:
    st.header("9. PCA Factor Shocks & Whole-Instrument Anchoring")

    # =============================================================================
    # Helper functions
    # =============================================================================

    def parse_derivative(label):
        """
        Examples:
        '3M Spread: H26-M26'        -> [('H26', 1), ('M26', -1)]
        '3M Fly: H26-2xM26+U26'     -> [('H26', 1), ('M26', -2), ('U26', 1)]
        '3M Double Fly: H26-3xM26+3xU26-Z26'
                                     -> [('H26',1),('M26',-3),('U26',3),('Z26',-1)]
        """
        expr = label.split(":")[-1].replace(" ", "")
        tokens = re.findall(r'([+-]?)(\d*)x?([A-Z]\d{2})', expr)

        legs = []
        for sign, mult, c in tokens:
            s = -1 if sign == '-' else 1
            m = int(mult) if mult else 1
            legs.append((c, s * m))
        return legs


    def eval_derivative(label, curve):
        return sum(w * curve[c] for c, w in parse_derivative(label))


    # =============================================================================
    # 1. BUILD DERIVATIVE UNIVERSE
    # =============================================================================

    all_deriv_list = []

    if 'spreads_3M_df_raw' in locals():
        all_deriv_list.append(spreads_3M_df_raw.rename(columns=lambda x: f"3M Spread: {x}"))

    if 'butterflies_3M_df' in locals():
        all_deriv_list.append(butterflies_3M_df.rename(columns=lambda x: f"3M Fly: {x}"))

    if 'double_butterflies_3M_df' in locals():
        all_deriv_list.append(double_butterflies_3M_df.rename(columns=lambda x: f"3M Double Fly: {x}"))

    if 'spreads_6M_df' in locals():
        all_deriv_list.append(spreads_6M_df.rename(columns=lambda x: f"6M Spread: {x}"))

    if 'butterflies_6M_df' in locals():
        all_deriv_list.append(butterflies_6M_df.rename(columns=lambda x: f"6M Fly: {x}"))

    if not all_deriv_list:
        st.error("Derivative data not found. Run earlier sections first.")
        st.stop()

    all_deriv_df = pd.concat(all_deriv_list, axis=1)

    Sigma_raw, deriv_aligned, loadings_gen = calculate_derivatives_covariance_generalized(
        all_deriv_df, scores, eigenvalues, pc_count
    )

    # =============================================================================
    # 2. SELECT ANCHOR
    # =============================================================================

    anchor_label = st.selectbox(
        "Select Instrument to Anchor:",
        sorted(all_deriv_df.columns),
        key="section9_anchor_final"
    )

    # =============================================================================
    # 3. RUN ANCHORED PCA SHOCK
    # =============================================================================

    if st.button("Run Whole-Instrument Anchor Shock"):
        try:
            # STEP A: PCA FACTOR SHIFT (minimum-norm)
            # ----------------------------------------------------------------
            # mkt_val: market price of the anchor on analysis date
            # pca_fair: PCA model's fair value = (score · L) * sigma + mean
            # Z_target: gap in units of the instrument's own std dev (dimensionless)
            # delta_PC: minimum-norm PC shift to close that gap
            # ----------------------------------------------------------------
            mkt_val = all_deriv_df.loc[analysis_dt, anchor_label]

            L = loadings_gen.loc[anchor_label].iloc[:pc_count].values
            sigma = all_deriv_df[anchor_label].std()
            if sigma < 1e-9:
                sigma = 1.0
            mean = all_deriv_df[anchor_label].mean()

            pca_fair = (scores.loc[analysis_dt].iloc[:pc_count].values @ L) * sigma + mean
            Z_target = (mkt_val - pca_fair) / sigma

            Lm = L.reshape(1, -1)
            delta_PC = (Lm.T @ np.linalg.inv(Lm @ Lm.T) * Z_target).flatten()

            # STEP B: PROPAGATE TO 3M SPREAD DNA
            # ---------------------------------------------------------------------
            std3 = spreads_3M_df_clean.std()
            L_sp = loadings_spread.values[:, :pc_count]

            delta_spreads_raw = (L_sp @ delta_PC) * std3.values
            # Map deltas back to labeled series so alignment is by spread label, not position.
            # If any spreads were dropped by dropna() during PCA, the position-based approach
            # misapplies deltas to the wrong spreads.  Label-based alignment is safe.
            delta_spreads_series = pd.Series(delta_spreads_raw, index=spreads_3M_df_clean.columns)

            base_spreads = historical_spreads_3M_df.loc[analysis_dt].filter(like="(PCA)")
            base_spreads.index = [
                c.replace(" (PCA)", "").replace("3M Spread: ", "") for c in base_spreads.index
            ]

            shocked_spreads = base_spreads.copy()
            # Apply deltas only where label matches — safe against dropped/reordered spreads
            common_labels = base_spreads.index.intersection(delta_spreads_series.index)
            shocked_spreads.loc[common_labels] = (
                base_spreads.loc[common_labels] + delta_spreads_series.loc[common_labels]
            )

            # STEP C: REBUILD OUTRIGHT CURVE
            # Use the same FOMC anchor that Section 4 uses for reconstruction.
            # Previously this used contract_labels[len//2] (mid-curve) which is
            # inconsistent with the economically-motivated FOMC anchor in Section 4.
            # We re-derive the anchor index from the same FOMC logic here.
            # ---------------------------------------------------------------------
            outr_mkt = analysis_curve_df.loc[analysis_dt]

            # Re-derive FOMC anchor index (mirrors reconstruct_prices_and_derivatives logic)
            _fomc_anchor_idx_s9 = 0
            _today_s9 = pd.Timestamp(date.today())
            _upcoming_s9 = [f for f in _fomc_ts_list if f >= _today_s9] if '_fomc_ts_list' in globals() else []
            _next_fomc_s9 = min(_upcoming_s9) if _upcoming_s9 else None
            if _next_fomc_s9 is not None:
                for _ci9, _cl9 in enumerate(contract_labels):
                    _exp9 = _SR3_EXPIRY_MAP.get(_cl9)
                    if _exp9 is None:
                        continue
                    _exp9_ts = pd.Timestamp(_exp9)
                    _settle9 = _exp9_ts - pd.DateOffset(months=3)
                    if _settle9 <= _next_fomc_s9 <= _exp9_ts:
                        _fomc_anchor_idx_s9 = _ci9
                        break
                else:
                    for _ci9, _cl9 in enumerate(contract_labels):
                        _exp9 = _SR3_EXPIRY_MAP.get(_cl9)
                        if _exp9 and (pd.Timestamp(_exp9) - _today_s9).days >= 14:
                            _fomc_anchor_idx_s9 = _ci9
                            break

            pivot = contract_labels[_fomc_anchor_idx_s9]
            shocked_out = pd.Series(index=contract_labels, dtype=float)
            shocked_out[pivot] = outr_mkt[pivot]

            # forward
            p_idx = contract_labels.index(pivot)
            for i in range(p_idx + 1, len(contract_labels)):
                p, c = contract_labels[i - 1], contract_labels[i]
                shocked_out[c] = shocked_out[p] - shocked_spreads[f"{p}-{c}"]

            # backward
            for i in range(p_idx - 1, -1, -1):
                c, n = contract_labels[i], contract_labels[i + 1]
                shocked_out[c] = shocked_out[n] + shocked_spreads[f"{c}-{n}"]

            # ---------------------------------------------------------------------
            # STEP D: ENFORCE WHOLE-INSTRUMENT CONSTRAINT (CORRECT WAY)
            # ---------------------------------------------------------------------
            legs = parse_derivative(anchor_label)

            inst_val = sum(w * shocked_out[c] for c, w in legs)
            residual = mkt_val - inst_val

            norm = sum(w * w for _, w in legs)
            for c, w in legs:
                shocked_out[c] += (w / norm) * residual

            # ---------------------------------------------------------------------
            # STEP E: PLOT OUTRIGHT CURVE
            # ---------------------------------------------------------------------
            st.success(f"Successfully anchored: {anchor_label}")

            fig, ax = plt.subplots(figsize=(14, 5))
            ax.plot(contract_labels, outr_mkt, label="Market", marker='o', alpha=0.35)
            ax.plot(contract_labels, shocked_out, label="Shocked (Anchored)", marker='x', linestyle='--')

            for c, _ in legs:
                ax.scatter(c, outr_mkt[c], s=160, color=_BBG_RED, zorder=5)

            ax.set_title(f"Whole-Instrument Convergence: {anchor_label}")
            ax.legend()
            _bbg_fig(fig=fig)
            st.pyplot(fig)

            # ---------------------------------------------------------------------
            # STEP F: FAMILY SNAPSHOTS
            # ---------------------------------------------------------------------
            st.subheader("Family Impact Analysis")

            shocked_derivs = compute_all_derivatives_from_outrights_row(
                contract_labels, shocked_out
            )

            families = [
                ("3M Spread", "historical_spreads_3M_df", "3M_spreads"),
                ("3M Butterfly", "historical_butterflies_3M_df", "3M_flies"),
                ("3M Double Fly", "historical_double_butterflies_3M_df", "3M_dbf"),
                ("6M Spread", "historical_spreads_6M_df", "6M_spreads"),
                ("6M Butterfly", "historical_butterflies_6M_df", "6M_flies"),
            ]

            for label, hist_name, key in families:
                # FIXED: use globals() not locals() - these vars are in module scope, not local function scope
                hist_df = globals().get(hist_name)
                if hist_df is not None and not hist_df.empty:
                    with st.expander(f"View {label} Impact"):
                        plot_shock_derivative_snapshot(
                            hist_df,
                            label,
                            shocked_derivs[key],
                            analysis_dt,
                            pc_count,
                            f"(Anchored to {anchor_label})"
                        )

        except Exception as e:
            st.error(f"Convergence failed: {e}")


    # ------------------- Section 10: Precision Adaptive Envelopes & Stats -------------------

with _tab_snap:
    st.header("10. Precision Adaptive Envelopes (1σ & 2σ)")

    # 1. Determine total historical length for the default slider value
    sample_df = locals().get("historical_spreads_3M_df")
    total_hist_days = len(sample_df) if sample_df is not None else 252

    # --- Explanation of Resid and Z-Score ---
    st.info("""
    **Definitions for Precision Trading:**
    * **Resid (bps):** The raw gap between Market and Model. (+ is Rich, - is Cheap).
    * **Max Sigma:** The peak volatility 'benchmark' for that specific instrument.
    * **Z-Score:** The 'Severity' of the mispricing. Z > 2.0 means the Resid is more than twice the peak historical noise.
    """)

    # 2. Slider: Defaulting to the MAX historical range
    lookback_selection = st.slider(
        "Volatility Lookback Window (Days):", 
        min_value=10, 
        max_value=total_hist_days, 
        value=total_hist_days, 
        key="p10_precision_vfinal",
        help="Default is the full history. This finds the 'peak' noise level reached by each instrument."
    )

    def plot_with_stats_table(df, label, analysis_dt, window, curve_df=None):
        """
        Computes adaptive volatility bands, plots 1/2 sigma bands,
        and generates a detailed statistical table.

        New columns added (2025-03):
          Roll/qtr (bps)   — Roll-down for a spread/fly position over one quarter.
                             Derived from the live outright curve at analysis_dt:
                               Roll(spread i,i+1) = −fly(i,i+1,i+2)
                             Only computed for 3M Spread family; N/A for flies/double-flies
                             (their roll formulas are higher-order and not simply expressed
                              in terms of adjacent contracts without the full outright chain).
          BE days          — Breakeven holding period in trading days:
                               BE = |Resid (bps)| / |Roll per trading day|
                             = how many days before roll-down fully offsets the PCA signal.
                             Only meaningful when roll opposes the trade direction.
                             Uses 63 trading days per quarter.
        """
        # Identify instruments
        instruments = [c.replace(" (Original)", "") for c in df.columns if " (Original)" in c]
        if not instruments: return

        # Pre-compute roll-down map from outright curve at analysis_dt
        # Roll for spread[i,i+1] = −fly[i,i+1,i+2] (in bps, quarterly)
        # This is exact under the assumption of parallel curve shape shift.
        _roll_map = {}   # instrument_label → roll_bps_per_quarter
        _is_spread_family = "Spread" in label
        if _is_spread_family and curve_df is not None:
            try:
                _curve_row = curve_df.loc[analysis_dt] if analysis_dt in curve_df.index else \
                             curve_df.loc[curve_df.index[curve_df.index <= analysis_dt].max()]
                _curve_row = _curve_row.dropna()
                _cc = _curve_row.index.tolist()
                _pp = _curve_row.values
                # Build a label→(i,j) map for 3M spreads: label = "Ci-Cj"
                for _si in range(len(_cc) - 2):
                    _spread_lbl = f"{_cc[_si]}-{_cc[_si+1]}"
                    _fly_bps    = (_pp[_si] - 2*_pp[_si+1] + _pp[_si+2]) * 100
                    _roll_qtr   = -_fly_bps          # bps gained per quarter
                    _roll_map[_spread_lbl] = _roll_qtr
            except Exception:
                pass   # silently skip if curve not available for that date

        data_list = []
        for inst in instruments:
            orig_col, pca_col = f"{inst} (Original)", f"{inst} (PCA)"

            if orig_col in df.columns and pca_col in df.columns:
                full_res_bps = (df[orig_col] - df[pca_col]) * 100

                rolling_sigma = full_res_bps.rolling(window=window, min_periods=min(10, window)).std()
                peak_sigma_bps = rolling_sigma.quantile(0.95)
                if np.isnan(peak_sigma_bps) or peak_sigma_bps <= 0:
                    peak_sigma_bps = full_res_bps.std()
                if np.isnan(peak_sigma_bps) or peak_sigma_bps <= 0:
                    peak_sigma_bps = 1.0

                curr_mkt = df.loc[analysis_dt, orig_col]
                curr_pca = df.loc[analysis_dt, pca_col]
                curr_res = (curr_mkt - curr_pca) * 100

                z_score = curr_res / peak_sigma_bps

                if z_score > 2: signal = "EXTREME RICH"
                elif z_score > 1: signal = "RICH"
                elif z_score < -2: signal = "EXTREME CHEAP"
                elif z_score < -1: signal = "CHEAP"
                else: signal = "FAIR"

                # ── Roll-down (3M Spreads only) ─────────────────────────────────
                # Strip the family prefix to get the raw contract label used in _roll_map
                _raw_lbl = inst.replace("3M Spread: ", "").replace("6M Spread: ", "") \
                               .replace("3M Fly: ", "").replace("6M Fly: ", "") \
                               .replace("3M Double Fly: ", "").replace("6M Double Fly: ", "") \
                               .replace("12M Spread: ", "").replace("12M Fly: ", "") \
                               .replace("12M Double Fly: ", "")
                _roll_qtr  = _roll_map.get(_raw_lbl, np.nan)
                _roll_day  = _roll_qtr / 63 if not np.isnan(_roll_qtr) else np.nan

                # Breakeven: days until roll-down neutralises the current PCA signal.
                # Only meaningful when roll direction opposes the signal direction
                # (e.g. residual says CHEAP → you buy → negative roll erodes the gain).
                # When roll helps the trade, BE is shown as positive (infinite benefit).
                if not np.isnan(_roll_day) and abs(_roll_day) > 1e-6:
                    _be_days = abs(curr_res) / abs(_roll_day)
                    # Flag if roll opposes trade: residual > 0 (RICH → sell) but roll > 0 (helps)
                    # residual < 0 (CHEAP → buy) but roll < 0 (hurts)
                    _roll_opposes = (curr_res > 0 and _roll_day < 0) or (curr_res < 0 and _roll_day > 0)
                    if not _roll_opposes:
                        _be_days = np.nan  # roll is helping — no breakeven concern
                else:
                    _be_days = np.nan

                data_list.append({
                    "Instrument": inst,
                    "Market": curr_mkt, "Fair": curr_pca, "Resid (bps)": curr_res,
                    "Peak Sigma 95p (bps)": peak_sigma_bps,
                    "Z-Score": z_score,
                    "Signal": signal,
                    "Roll/qtr (bps)": _roll_qtr,
                    "BE days": _be_days,
                    "U2": curr_pca + (2 * peak_sigma_bps / 100),
                    "U1": curr_pca + (1 * peak_sigma_bps / 100),
                    "L1": curr_pca - (1 * peak_sigma_bps / 100),
                    "L2": curr_pca - (2 * peak_sigma_bps / 100),
                })

        if not data_list: return
        plot_df = pd.DataFrame(data_list)
        x = range(len(plot_df))

        # --- CHART ---
        fig, ax = plt.subplots(figsize=(15, 7))
        ax.plot(x, plot_df["U2"], color=_BBG_RED, linestyle=':', alpha=0.7, label="2σ Band (95p peak)")
        ax.plot(x, plot_df["L2"], color=_BBG_RED, linestyle=':', alpha=0.7)
        ax.plot(x, plot_df["U1"], color=_BBG_AMBER, linestyle='--', alpha=0.5, label="1σ Band (95p peak)")
        ax.plot(x, plot_df["L1"], color=_BBG_AMBER, linestyle='--', alpha=0.5)
        ax.plot(x, plot_df["Fair"], color=_BBG_GRAY, label="PCA Fair", linewidth=1.2, alpha=0.4)
        ax.plot(x, plot_df["Market"], color=_BBG_BLUE, marker='o', linewidth=2.5, label="Market", markersize=8)

        for i, row in plot_df.iterrows():
            text_color = _BBG_RED if abs(row['Z-Score']) > 1 else _BBG_CYAN
            ax.annotate(f"{row['Resid (bps)']:.1f}", (i, row['Market']),
                        xytext=(0, 12), textcoords="offset points",
                        ha='center', fontsize=9, fontweight='bold', color=text_color)

        ax.set_xticks(x); ax.set_xticklabels(plot_df["Instrument"], rotation=45, ha='right')
        ax.set_title(f"{label} Curve: Statistical Boundaries (95p-Rolling σ, {window}d window)", fontsize=14)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1)); ax.grid(True, alpha=0.15)
        _bbg_fig(fig=fig)
        st.pyplot(fig)

        st.session_state.SECTION9_FIGURES.append(
            (fig, f"Section 10 – {label}")
        )

        # --- TABLE ---
        st.write(f"**{label} Precision Statistics**")

        plot_df["Dist to 2σ (bps)"] = np.sign(plot_df["Resid (bps)"]) * (
            plot_df["Resid (bps)"].abs() - 2 * plot_df["Peak Sigma 95p (bps)"]
        )

        # Determine which columns are populated (roll only for 3M Spread family)
        _has_roll = _is_spread_family and plot_df["Roll/qtr (bps)"].notna().any()
        _display_cols = ["Instrument", "Resid (bps)", "Peak Sigma 95p (bps)", "Z-Score", "Signal", "Dist to 2σ (bps)"]
        _fmt = {"Resid (bps)": "{:.2f}", "Peak Sigma 95p (bps)": "{:.2f}",
                "Z-Score": "{:.2f}", "Dist to 2σ (bps)": "{:.2f}"}

        if _has_roll:
            _display_cols += ["Roll/qtr (bps)", "BE days"]
            _fmt["Roll/qtr (bps)"] = "{:.2f}"
            _fmt["BE days"] = lambda v: f"{v:.0f}d" if not np.isnan(v) else "—"
            # Annotate roll direction inline
            plot_df["Roll/qtr (bps)"] = plot_df["Roll/qtr (bps)"].round(2)

        _styled = plot_df[_display_cols].style \
            .background_gradient(subset=['Z-Score'], cmap='RdYlGn_r', vmin=-3, vmax=3) \
            .applymap(lambda v: 'color: red; font-weight: bold' if "EXTREME" in str(v) else '',
                      subset=['Signal']) \
            .format(_fmt, na_rep="—")

        if _has_roll:
            # Colour roll: green = helps trade, red = hurts trade (roll opposes residual)
            def _roll_colour(row):
                styles = [''] * len(row)
                _ri = list(row.index).index("Roll/qtr (bps)") if "Roll/qtr (bps)" in row.index else None
                if _ri is None: return styles
                _roll = row["Roll/qtr (bps)"]
                _resid = row["Resid (bps)"]
                if np.isnan(_roll): return styles
                _hurts = (_resid > 0 and _roll < 0) or (_resid < 0 and _roll > 0)
                styles[_ri] = 'color: #ff6666' if _hurts else 'color: #66cc66'
                return styles
            _styled = _styled.apply(_roll_colour, axis=1)

        bbg_table(_styled, use_container_width=True)

        if _has_roll:
            st.caption(
                "**Roll/qtr** = change in spread value from pure time passage over one quarter "
                "(= −fly of adjacent contracts). "
                "🔴 red = roll works against the PCA signal. "
                "🟢 green = roll supports it.  "
                "**BE days** = trading days before roll-down fully offsets the current PCA mispricing "
                "(only shown when roll opposes the signal)."
            )
        st.divider()

    # --- EXECUTE FOR ALL FAMILIES ---
    families = [
        ("3M Spread", "historical_spreads_3M_df"),
        ("6M Spread", "historical_spreads_6M_df"),
        ("3M Butterfly", "historical_butterflies_3M_df"),
        ("6M Butterfly", "historical_butterflies_6M_df"),
        ("3M Double Fly", "historical_double_butterflies_3M_df"),
        ("6M Double Fly", "historical_double_butterflies_6M_df")
    ]

    st.session_state.SECTION9_FIGURES = []

    for label, var in families:
        df_found = globals().get(var)
        if df_found is not None and not df_found.empty:
            plot_with_stats_table(df_found, label, analysis_dt, lookback_selection,
                                  curve_df=analysis_curve_df)   # pass outright curve for roll-down

    # ============================
    # SECTION 10B — MACRO REGIME PLAYBOOK
    # ============================

with _tab_macro:
    st.header("10B. Macro Regime Playbook")

    st.markdown("""
    Using 5 years of SOFR futures data (Nov 2020 – present), this section maps **how spreads and
    flies historically moved during each macro shock regime**.  All moves are in **bps**, using
    fixed curve-position notation:

    > **[0]** = front active contract · **[1]** = next · **[2],[3]...** = further out  
    > **Spread [i]–[i+1]** = Price[i] − Price[i+1]  *(positive = normal/upward-sloping in rate space)*  
    > **Fly [i]–[i+1]–[i+2]** = Price[i] − 2×Price[i+1] + Price[i+2]

    Signals are **directional tendencies observed in the data**, not guarantees.
    """)

    # ── REGIME DEFINITIONS ────────────────────────────────────────────────────────
    _REGIME_DEFS = {
        "🔴 Inflation Shock\n(CPI surge, hawkish repricing)": {
            "dates": ("2021-11-01", "2022-03-15"),
            "description": "Market suddenly prices aggressive hike path. Front contracts sell off hardest as near-term rate expectations reprice up.",
            "signals": {
                "Level": "SELL (prices fall sharply, -50 to -100 bps avg)",
                "Front 3M spreads [0]–[1],[1]–[2]": "WIDEN aggressively (+40 to +50 bps). Market prices rapid rate rises into the front.",
                "Back 3M spreads [6]–[7],[7]–[8]": "TIGHTEN or flip negative (-10 bps). Terminal rate expectations anchored.",
                "6M spreads [0]–[2]": "WIDEN sharply (+70 to +85 bps). Most sensitive indicator.",
                "Front flies [1]–[2]–[3]": "HIGHER (+20 bps). Hump builds as market prices peak-then-hold.",
                "Back flies [5]–[6]–[7]": "HIGHER (+7 bps). Curvature extends into mid-curve.",
            },
            "key_trade": "Long front spreads (Z-spread steepener). Short back spreads. Long mid-curve flies.",
        },
        "⚔️ Geopolitical Shock / War\n(Ukraine-type: oil spike + flight to quality)": {
            "dates": ("2022-02-24", "2022-04-30"),
            "description": "Dual shock: oil surge pushes inflation up (hawkish), but fear bids bonds at the back end. Front end sells, back anchors.",
            "signals": {
                "Level": "SELL (hawkish repricing, -100 bps avg)",
                "Front 3M spread [0]–[1]": "WIDEN dramatically (+70 bps). Front contracts price imminent hikes.",
                "Mid-back 3M spreads [3]–[4],[4]–[5]": "TIGHTEN (-20 to -30 bps). Inversion builds as terminal rate priced in.",
                "6M spread [0]–[2]": "WIDEN (+79 bps). Biggest bang in the front.",
                "6M spread [2]–[4]": "TIGHTEN (-52 bps). Back half inverts.",
                "Front fly [0]–[1]–[2]": "HIGHER (+60 bps). Extremely large — regime-defining signal.",
                "Back fly [3]–[4]–[5]": "LOWER (-11 bps). Back of curve flattens out.",
            },
            "key_trade": "Long front fly (huge move). Long front spread vs short mid spread. Steepener on the front 6M.",
        },
        "🏦 Banking / Credit Crisis\n(SVB/Signature type: flight to safety, cut pricing)": {
            "dates": ("2023-03-08", "2023-04-30"),
            "description": "Systemic fear triggers aggressive cut pricing in the front end. Near-term contracts rally hard, back end more stable.",
            "signals": {
                "Level": "BUY (all contracts rally, +55 bps avg)",
                "Front 3M spreads [0]–[1],[1]–[2]": "TIGHTEN sharply (-29 to -32 bps). Front rallies faster than back.",
                "Back 3M spreads [5]–[6],[6]–[7]": "WIDEN slightly (+11 to +14 bps). Back lags, curve starts to un-invert.",
                "6M spreads [0]–[2],[1]–[3]": "TIGHTEN hard (-52 to -61 bps). Most powerful signal in a banking crisis.",
                "All flies": "LOWER (-3 to -17 bps). Curvature collapses as inversion unwinds uniformly.",
            },
            "key_trade": "Short front spread (tightener). Long 6M spread tightener. Sell flies across the curve.",
        },
        "📉 Weak Jobs / Recession Fear\n(soft data miss, emergency cut pricing)": {
            "dates": ("2024-07-05", "2024-08-15"),
            "description": "Unexpected payroll miss triggers front-end panic rally. Market prices emergency cuts at the very front.",
            "signals": {
                "Level": "BUY (rally, +35 bps avg)",
                "Front 3M spreads [0]–[1],[1]–[2]": "TIGHTEN hard (-18 to -26 bps). Front two contracts rally most.",
                "Back 3M spreads [4]–[5],[5]–[6],[6]–[7]": "WIDEN slightly (+2 to +4 bps). Back anchors, starts to dis-invert.",
                "6M spread [0]–[2]": "TIGHTEN sharply (-44 bps). Strongest signal.",
                "6M spread [3]–[5]": "STABLE / slight widen (+1 bps). Back half unaffected.",
                "Front fly [0]–[1]–[2]": "HIGHER (+8 bps). Front kink builds.",
                "Mid flies [1]–[2]–[3],[2]–[3]–[4]": "LOWER (-11 to -13 bps). Mid-curve flattens as cuts priced uniformly.",
            },
            "key_trade": "Short front 6M spread (tightener). Long front fly. Sell mid flies.",
        },
        "🛢️ Oil Shock / Middle East\n(contained: stagflation risk but no Fed pivot)": {
            "dates": ("2023-10-07", "2023-11-15"),
            "description": "Risk-off but contained. No Fed pivot priced — market treats as stagflationary noise. Very small moves across the curve.",
            "signals": {
                "Level": "SMALL BUY (+12 bps avg). Flight to quality marginal.",
                "All spreads": "FLAT to ±3 bps. No strong signal — regime is ambiguous.",
                "All flies": "FLAT to ±4 bps. No directional conviction.",
                "Key observation": "When the oil shock is isolated WITHOUT imminent Fed action, SOFR curve barely moves. The market is pricing a 'wait and see' Fed.",
            },
            "key_trade": "No strong SOFR curve trade. Monitor for escalation into full inflation shock regime.",
        },
        "📈 Fed Cut Cycle\n(actual cutting in progress)": {
            "dates": ("2024-09-01", "2024-12-31"),
            "description": "Fed actively cutting. Front end reprices lower. Inversion unwinds from the front — spreads widen as front catches up to back.",
            "signals": {
                "Level": "SELL front (falls -78 bps avg). Curve dis-inverts.",
                "ALL 3M spreads": "WIDEN uniformly (+1 to +34 bps). Most pronounced at the front.",
                "Front 3M spread [0]–[1]": "WIDEN +34 bps. Largest move.",
                "6M spreads all": "WIDEN uniformly (+12 to +52 bps).",
                "All flies": "HIGHER (+1 to +15 bps). Curvature builds as front spreads widen unevenly.",
            },
            "key_trade": "Long ALL spreads (especially front). Long all flies. This is the cleanest, most uniform regime for spread trades.",
        },
        "🔒 Peak Rates / Terminal Plateau\n(hiking done, waiting)": {
            "dates": ("2023-08-01", "2023-10-31"),
            "description": "Fed on hold at terminal rate. Market gradually reprices risk premium. Front end stays pinned. Back end starts to sell off (term premium).",
            "signals": {
                "Level": "SELL slightly (-62 bps). Back end selling on term premium.",
                "Front spreads [0]–[1]": "WIDEN slightly (+12 bps).",
                "Back spreads [4]–[5],[5]–[6]": "WIDEN (+11 bps). Term premium pushes back prices lower.",
                "All flies": "FLAT to slight LOWER (-2 bps). Inversion is stable.",
            },
            "key_trade": "No strong trade. Monitor for catalyst. Curve is compressed — range-bound spreads.",
        },
        "🕊️ Regime Change: Pivot Expectations\n(market prices Fed turning dovish)": {
            "dates": ("2023-11-01", "2024-03-31"),
            "description": "Market gets ahead of Fed on cuts. Front end rallies on expectation, back end anchors. Curve starts to un-invert.",
            "signals": {
                "Level": "BUY (+44 bps). Pricing in cuts.",
                "Front spread [0]–[1]": "WIDEN (+5 bps). Small — front still inverted.",
                "Mid spreads [3]–[4],[4]–[5]": "TIGHTEN (-8 to -12 bps). Market extends cut path into mid-curve.",
                "6M spreads [2]–[4],[3]–[5]": "TIGHTEN (-12 to -21 bps). Inversion deepens momentarily before un-inversion.",
                "Front fly": "HIGHER (+3 bps).",
            },
            "key_trade": "Complex — inversion both extends and then reverses. Watch for transition: when front spread starts widening fast → regime has shifted to cut cycle.",
        },
    }

    # ── UI: Regime selector and display ──────────────────────────────────────────
    _regime_names = list(_REGIME_DEFS.keys())
    _selected_regime = st.selectbox(
        "Select Macro Regime:",
        _regime_names,
        key="macro_regime_selector"
    )
    _reg = _REGIME_DEFS[_selected_regime]

    # Show empirical data for this regime using uploaded CSV
    try:
        _price_df_raw = pd.read_csv('/dev/null')  # placeholder
    except Exception:
        pass

    # Use the already-loaded price_df, but restrict to columns that appear in
    # analysis_curve_df (which is guaranteed sorted by expiry date).
    # price_df.columns order is CSV-upload order which may NOT be expiry-ascending.
    # Using wrong column order makes positions [0],[1]... incorrect.
    _r_start, _r_end = _reg["dates"]
    _ac_cols = analysis_curve_df.columns.tolist() if 'analysis_curve_df' in globals() else []
    if price_df is not None and _ac_cols:
        # Subset price_df to expiry-ordered columns present in both dataframes
        _ac_in_price = [c for c in _ac_cols if c in price_df.columns]
        _r_sub = price_df.loc[_r_start:_r_end, _ac_in_price].dropna(axis=1, how='all') if _ac_in_price else pd.DataFrame()
    else:
        _r_sub = price_df.loc[_r_start:_r_end].dropna(axis=1, how='all') if price_df is not None else pd.DataFrame()

    col_reg1, col_reg2 = st.columns([1, 1])

    with col_reg1:
        st.markdown(f"**Description:** {_reg['description']}")
        st.markdown(f"**Empirical window:** `{_r_start}` → `{_r_end}`")
        st.markdown("---")
        st.markdown("**📊 Signal Map:**")
        for signal_name, direction in _reg["signals"].items():
            colour = "🟢" if any(w in direction.upper() for w in ["BUY", "WIDEN", "HIGHER"]) else (
                     "🔴" if any(w in direction.upper() for w in ["SELL", "TIGHTEN", "LOWER"]) else "⚪")
            st.markdown(f"- {colour} **{signal_name}**: {direction}")
        st.markdown("---")
        st.markdown(f"**💡 Key Trade:** {_reg['key_trade']}")

    with col_reg2:
        # Plot empirical curve moves for this regime
        if not _r_sub.empty and _r_sub.shape[1] >= 4:
            _r_valid = _r_sub.dropna(axis=1, how='any')
            if _r_valid.shape[1] >= 4:
                _r_s = _r_valid.iloc[0]
                _r_e = _r_valid.iloc[-1]
                n_contracts = min(10, _r_valid.shape[1])

                # Compute spread changes
                _spread_labels = [f"[{i}]–[{i+1}]\n{_r_valid.columns[i]}-{_r_valid.columns[i+1]}"
                                  for i in range(n_contracts - 1)]
                _spread_deltas = [((_r_e.iloc[i] - _r_e.iloc[i+1]) - (_r_s.iloc[i] - _r_s.iloc[i+1])) * 100
                                  for i in range(n_contracts - 1)]

                _fig_reg, (_ax_sp, _ax_fly) = plt.subplots(2, 1, figsize=(10, 7))

                # Spread changes
                _colors_sp = [_BBG_GREEN if d > 0 else _BBG_RED for d in _spread_deltas]
                _ax_sp.bar(range(len(_spread_deltas)), _spread_deltas, color=_colors_sp, alpha=0.85)
                _ax_sp.axhline(0, color=_BBG_WHITE, linewidth=0.8, linestyle='--')
                _ax_sp.set_xticks(range(len(_spread_labels)))
                _ax_sp.set_xticklabels(_spread_labels, rotation=45, ha='right', fontsize=6)
                _ax_sp.set_title(f"3M Spread Δ (bps) — {_selected_regime.split(chr(10))[0]}", fontsize=9)
                _ax_sp.set_ylabel("Δ bps")

                # Fly changes
                n_flies = min(8, _r_valid.shape[1] - 2)
                _fly_labels = [f"[{i}]–[{i+1}]–[{i+2}]\n{_r_valid.columns[i]}-{_r_valid.columns[i+1]}-{_r_valid.columns[i+2]}"
                               for i in range(n_flies)]
                _fly_deltas = [((_r_e.iloc[i] - 2*_r_e.iloc[i+1] + _r_e.iloc[i+2]) -
                                (_r_s.iloc[i] - 2*_r_s.iloc[i+1] + _r_s.iloc[i+2])) * 100
                               for i in range(n_flies)]
                _colors_fly = [_BBG_GREEN if d > 0 else _BBG_RED for d in _fly_deltas]
                _ax_fly.bar(range(len(_fly_deltas)), _fly_deltas, color=_colors_fly, alpha=0.85)
                _ax_fly.axhline(0, color=_BBG_WHITE, linewidth=0.8, linestyle='--')
                _ax_fly.set_xticks(range(len(_fly_labels)))
                _ax_fly.set_xticklabels(_fly_labels, rotation=45, ha='right', fontsize=6)
                _ax_fly.set_title("3M Fly Δ (bps)", fontsize=9)
                _ax_fly.set_ylabel("Δ bps")

                plt.tight_layout()
                _bbg_fig(fig=_fig_reg)
                st.pyplot(_fig_reg)
            else:
                st.info("Insufficient contract overlap in CSV for this regime window.")
        else:
            st.info("Upload CSV data to see empirical chart for this regime.")

    # Cross-regime summary heatmap
    st.markdown("---")
    st.subheader("Cross-Regime Spread Heatmap")
    st.markdown("Average 3M spread change (bps) at fixed curve positions across all regimes. Green = widened, Red = tightened.")

    _heatmap_data = {}
    _regime_short_labels = {
        "🔴 Inflation Shock\n(CPI surge, hawkish repricing)": "Inflation\nShock",
        "⚔️ Geopolitical Shock / War\n(Ukraine-type: oil spike + flight to quality)": "War/\nGeo",
        "🏦 Banking / Credit Crisis\n(SVB/Signature type: flight to safety, cut pricing)": "Banking\nCrisis",
        "📉 Weak Jobs / Recession Fear\n(soft data miss, emergency cut pricing)": "Weak\nJobs",
        "🛢️ Oil Shock / Middle East\n(contained: stagflation risk but no Fed pivot)": "Oil\nShock",
        "📈 Fed Cut Cycle\n(actual cutting in progress)": "Cut\nCycle",
        "🔒 Peak Rates / Terminal Plateau\n(hiking done, waiting)": "Peak\nRates",
        "🕊️ Regime Change: Pivot Expectations\n(market prices Fed turning dovish)": "Pivot\nExpect",
    }

    if price_df is not None:
        _ac_cols_hm = analysis_curve_df.columns.tolist() if 'analysis_curve_df' in globals() else []
        for _rname, _rdef in _REGIME_DEFS.items():
            _rs, _re = _rdef["dates"]
            # Restrict to expiry-ordered columns to ensure [0],[1]... positions are front-to-back
            if _ac_cols_hm:
                _ac_in_p = [c for c in _ac_cols_hm if c in price_df.columns]
                _rsub = price_df.loc[_rs:_re, _ac_in_p].dropna(axis=1, how='any') if _ac_in_p else pd.DataFrame()
            else:
                _rsub = price_df.loc[_rs:_re].dropna(axis=1, how='any')
            if _rsub.empty or _rsub.shape[1] < 5:
                continue
            _row = {}
            for _pi in range(min(7, _rsub.shape[1] - 1)):
                _key = f"[{_pi}]–[{_pi+1}]"
                _dsp = ((_rsub.iloc[-1, _pi] - _rsub.iloc[-1, _pi+1]) -
                        (_rsub.iloc[0, _pi] - _rsub.iloc[0, _pi+1])) * 100
                _row[_key] = round(_dsp, 1)
            _heatmap_data[_regime_short_labels.get(_rname, _rname[:15])] = _row

        if _heatmap_data:
            _hm_df = pd.DataFrame(_heatmap_data).T.fillna(0)
            _fig_hm, _ax_hm = plt.subplots(figsize=(12, 5))
            _vmax = max(abs(_hm_df.values.max()), abs(_hm_df.values.min()), 10)
            import matplotlib.colors as _mcolors
            _cmap = plt.cm.RdYlGn
            sns.heatmap(
                _hm_df, annot=True, fmt=".0f", cmap="RdYlGn",
                center=0, vmin=-_vmax, vmax=_vmax,
                linewidths=0.5, linecolor='#333',
                ax=_ax_hm,
                annot_kws={"size": 8, "family": "monospace"}
            )
            _ax_hm.set_title("3M Spread Δ (bps) by Regime and Curve Position", fontsize=11)
            _ax_hm.set_xlabel("Curve Position (fixed, front to back)")
            _ax_hm.set_ylabel("Macro Regime")
            _bbg_fig(fig=_fig_hm)
            st.pyplot(_fig_hm)

            st.caption("""
    **Reading the heatmap**: Each cell = how many bps the spread at that curve position moved during that regime.  
    Green = spread widened (front outperformed back in rate space).  Red = spread tightened / inverted further.  
    Positions are fixed: [0]–[1] = front two active contracts at the START of each regime window.
            """)

    # ============================
    # SECTION 11 — KALMAN FILTERED PCA FAIR CURVE
    # ============================


with _tab_snap:
    st.header("11. Kalman-Filtered PCA Fair Curve")

    st.markdown("""
    This section builds a **dynamic fair value curve** by applying a **Kalman filter**
    to PCA factor scores.  
    The **output snapshot is identical to Section 5**, but uses **noise-filtered factors**.
    """)

    # ----------------------------
    # Helper functions (LOCAL)
    # ----------------------------

    def _estimate_phi_ar1(series, clip=(0.70, 0.995)):
        x = np.asarray(series, dtype=float)
        if len(x) < 10:
            return 0.95
        # Demean before AR(1) OLS to get unbiased estimate
        x = x - x.mean()
        x_lag = x[:-1]
        x_now = x[1:]
        denom = np.dot(x_lag, x_lag)
        if denom < 1e-8:
            return 0.95
        phi = np.dot(x_now, x_lag) / denom
        return float(np.clip(phi, clip[0], clip[1]))


    def _kalman_filter_1d(observed, phi, q, r, P0=None):
        """
        Scalar Kalman filter for an AR(1) state-space model:
            x_t = phi * x_{t-1} + w_t,  w_t ~ N(0, q)
            y_t = x_t + v_t,             v_t ~ N(0, r)

        P0: initial state variance. If None, uses the steady-state approximation
        via a few fixed-point iterations of the Riccati equation.
        """
        n = len(observed)
        x_hat = np.zeros(n)
        P = np.zeros(n)
        x_hat[0] = observed[0]

        if P0 is not None:
            P[0] = P0
        else:
            # Steady-state Riccati approximation (20 iterations)
            _P = np.var(observed)
            for _ in range(20):
                _K = (phi**2 * _P + q) / (phi**2 * _P + q + r)
                _P = (1 - _K) * (phi**2 * _P + q)
            P[0] = _P

        for t in range(1, n):
            x_pred = phi * x_hat[t - 1]
            P_pred = phi**2 * P[t - 1] + q
            K = P_pred / (P_pred + r)
            x_hat[t] = x_pred + K * (observed[t] - x_pred)
            P[t] = (1 - K) * P_pred

        return x_hat


    # ----------------------------
    # UI toggle
    # ----------------------------

    use_kalman = st.checkbox(
        "Use Kalman-Filtered PCA Factors",
        value=True,
        key="use_kalman_section11"
    )

    # ----------------------------
    # Apply Kalman to PCA scores
    # ----------------------------

    kalman_scores = scores.copy() if scores is not None else pd.DataFrame()
    phi_rows = []

    if use_kalman:
        # ── Kalman noise tuning ─────────────────────────────────────────────────
        # q = process noise variance: how much the true factor moves per step.
        # r = measurement noise variance: how noisy the PCA score is as an observation.
        #
        # q/r = signal-to-noise ratio (SNR):
        #   Low  SNR (q/r << 1) → heavy smoothing, filter trusts model over observation.
        #   High SNR (q/r >> 1) → light smoothing, filter tracks observations closely.
        #
        # A defensible default for liquid futures PC scores:
        #   Theoretical q for AR(1): q = var * (1 - phi^2)  (innovation variance of the AR process)
        #   r = observation noise. For PCA scores there is no separate measurement noise,
        #   so r should be small. We use r = 0.05 * var (5% of total variance is noise).
        #   This gives q/r ≈ (1-phi^2)/0.05. For phi=0.95: q/r ≈ 0.1/0.05 = 2.0.
        #   Previous hardcoded r = 1.0 * var gave q/r = 0.01 — far too conservative.
        #
        # Exposed as a sidebar slider so practitioners can tune to their preference.
        _snr_pct = st.sidebar.slider(
            "Kalman SNR — process/measurement noise ratio (%)",
            min_value=1, max_value=500, value=200,
            key="kalman_snr_slider",
            help="Higher = less smoothing (filter tracks market more closely). Default 200% ≈ AR(1) innovation noise / 5% measurement noise."
        )
        _snr = _snr_pct / 100.0   # e.g. 200% → 2.0

        for i in range(pc_count):
            pc_name = kalman_scores.columns[i]
            raw_series = kalman_scores[pc_name].values

            phi = _estimate_phi_ar1(raw_series)
            var = np.var(raw_series)

            # q = AR(1) innovation variance (how much the process moves per step)
            q = var * (1 - phi**2)
            # r = measurement noise = q / SNR
            r = q / _snr if _snr > 1e-6 else q

            # P[0]: steady-state Kalman variance under both q and r
            # Solve discrete algebraic Riccati: P = phi^2*P + q - (phi^2*P)^2 / (phi^2*P + r)
            # Approximate with a few fixed-point iterations from a reasonable start
            _P = var
            for _ in range(20):
                _K = (phi**2 * _P + q) / (phi**2 * _P + q + r)
                _P = (1 - _K) * (phi**2 * _P + q)

            kalman_scores[pc_name] = _kalman_filter_1d(
                raw_series, phi=phi, q=q, r=r, P0=_P
            )

            phi_rows.append({
                "PC": pc_name,
                "Estimated φ": round(phi, 4),
                "q (process noise)": f"{q:.4e}",
                "r (meas. noise)": f"{r:.4e}",
                "q/r (SNR)": f"{q/r:.2f}" if r > 1e-12 else "∞"
            })

        st.subheader("Estimated PCA Factor Persistence")
        bbg_table(pd.DataFrame(phi_rows), use_container_width=True)

    else:
        st.info("Raw PCA scores are used (Kalman disabled).")

    # ----------------------------
    # Reconstruct 3M spreads
    # ----------------------------

    data_mean = spreads_3M_df_clean.mean()
    data_std = spreads_3M_df_clean.std()

    scores_used = (
        kalman_scores.values[:, :pc_count]
        if use_kalman
        else scores.values[:, :pc_count]
    )

    loadings_used = loadings_spread.values[:, :pc_count]

    reconstructed_scaled = scores_used @ loadings_used.T

    reconstructed_spreads_3M_kf = pd.DataFrame(
        reconstructed_scaled * data_std.values + data_mean.values,
        index=spreads_3M_df_clean.index,
        columns=spreads_3M_df_clean.columns
    )

    # ----------------------------
    # Reconstruct outrights & derivatives
    # ----------------------------

    (
        historical_outrights_kf,
        historical_spreads_3M_kf,
        historical_butterflies_3M_kf,
        historical_spreads_6M_kf,
        historical_butterflies_6M_kf,
        historical_spreads_12M_kf,
        historical_butterflies_12M_kf,
        historical_double_butterflies_3M_kf,
        historical_double_butterflies_6M_kf,
        historical_double_butterflies_12M_kf,
        _
    ) = reconstruct_prices_and_derivatives(
        analysis_curve_df,
        reconstructed_spreads_3M_kf,
        spreads_3M_df_raw,
        spreads_6M_df,
        butterflies_3M_df,
        butterflies_6M_df,
        spreads_12M_df,
        butterflies_12M_df,
        double_butterflies_3M_df,
        double_butterflies_6M_df,
        double_butterflies_12M_df
    )

    # ----------------------------
    # Snapshot output (Section-5 style)
    # ----------------------------

    st.subheader("11.1 Kalman Fair Curve Snapshot — 3M Spreads")
    plot_snapshot(
        historical_spreads_3M_kf,
        derivative_type="3M Spread",
        current_date=analysis_dt,
        pc_count=pc_count,
        collect_for_pdf=False
    )

    st.subheader("11.2 Kalman Fair Curve Snapshot — 3M Flies")
    plot_snapshot(
        historical_butterflies_3M_kf,
        derivative_type="3M Fly",
        current_date=analysis_dt,
        pc_count=pc_count,
        collect_for_pdf=False
    )

    st.subheader("11.3 Kalman Fair Curve Snapshot — 3M Double Flies")
    plot_snapshot(
        historical_double_butterflies_3M_kf,
        derivative_type="3M Double Fly",
        current_date=analysis_dt,
        pc_count=pc_count,
        collect_for_pdf=False
    )

    # ============================
    # END SECTION 11
    # ============================================================
    # SECTION 12: TRADE STRUCTURING & PCA MISPRICING CAPTURE
    # ============================================================

    # -------------------------------------------------------------------
    # 12.0 EXPRESSION QUALITY OF THE SELECTED INSTRUMENT
    # -------------------------------------------------------------------

    def compute_expression_quality(instrument, factor_sensitivities_df, Sigma_Raw_df, mispricing_series):
        """
        Absolute quality of ONE instrument as a trading vehicle
        """

        betas = factor_sensitivities_df.loc[instrument]
        mispricing = abs(mispricing_series.get(instrument, np.nan))

        # Factor purity: single-factor vs mixed exposure
        # FIXED: guard against zero total sensitivity
        total_abs = betas.abs().sum()
        factor_purity = betas.abs().max() / total_abs if total_abs > 1e-9 else 0.0

        # Avg absolute correlation vs entire universe — derived from covariance matrix.
        # FIX: exclude diagonal (self-correlation = 1.0) before taking the mean.
        # Including it biases the result upward by 1/N_instruments.
        diag = np.sqrt(np.diag(Sigma_Raw_df.values))
        diag_safe = np.where(diag > 1e-9, diag, 1.0)
        corr_matrix = Sigma_Raw_df.values / np.outer(diag_safe, diag_safe)
        corr_df = pd.DataFrame(corr_matrix, index=Sigma_Raw_df.index, columns=Sigma_Raw_df.columns)
        # Mask diagonal (self = 1.0) before averaging
        np.fill_diagonal(corr_matrix, np.nan)
        corr_df_nodiag = pd.DataFrame(corr_matrix, index=Sigma_Raw_df.index, columns=Sigma_Raw_df.columns)
        avg_abs_corr = corr_df_nodiag.abs().mean().get(instrument, np.nan)

        expression_quality = mispricing * factor_purity / (1 + avg_abs_corr)

        return {
            "Mispricing (Rate %)": mispricing,
            "Dominant Factor": betas.abs().idxmax(),
            "Factor Purity": factor_purity,
            "Avg Abs Correlation": avg_abs_corr,
            "Expression Quality Score": expression_quality
        }


    # -------------------------------------------------------------------
    # 12.1 ALTERNATIVE EXPRESSIONS OF THE SAME DISTORTION
    # -------------------------------------------------------------------

    def find_alternative_expressions(
        selected_instrument,
        instrument_universe_df,
        factor_sensitivities_df,
        Sigma_Raw_df,
        mispricing_series,
        top_n=5
    ):
        T = selected_instrument
        T_betas = factor_sensitivities_df.loc[T]
        T_mis = abs(mispricing_series.get(T, np.nan))

        maturity_tag = (
            "3M" if "3M" in T else
            "6M" if "6M" in T else
            "12M" if "12M" in T else ""
        )

        local_universe = instrument_universe_df[
            instrument_universe_df["Instrument"].str.contains(maturity_tag)
        ]

        rows = []

        for C in local_universe["Instrument"]:
            if C == T or C not in factor_sensitivities_df.index:
                continue

            C_betas = factor_sensitivities_df.loc[C]

            # Factor alignment (cosine similarity)
            # FIXED: guard against zero-norm vectors
            norm_T = np.linalg.norm(T_betas)
            norm_C = np.linalg.norm(C_betas)
            if norm_T < 1e-9 or norm_C < 1e-9:
                alignment = 0.0
            else:
                alignment = np.dot(T_betas, C_betas) / (norm_T * norm_C)

            # Pairwise correlation vs selected instrument (from covariance matrix)
            var_T_loc = Sigma_Raw_df.loc[T, T]
            var_C_loc = Sigma_Raw_df.loc[C, C]
            denom_corr = np.sqrt(var_T_loc * var_C_loc)
            corr_vs_selected = Sigma_Raw_df.loc[T, C] / denom_corr if denom_corr > 1e-9 else 0.0

            relative_score = T_mis * abs(alignment) / (1 + abs(corr_vs_selected))

            rows.append({
                "Alternative Instrument": C,
                "Factor Alignment": alignment,
                "Correlation vs Selected": corr_vs_selected,
                "Relative Expression Score": relative_score
            })

        df = pd.DataFrame(rows)
        return df.sort_values("Relative Expression Score", ascending=False).head(top_n)


    # -------------------------------------------------------------------
    # 12.2 FACTOR-ISOLATED COMBO TRADE
    # -------------------------------------------------------------------

    def build_factor_isolated_combo(
        primary_instr,
        hedge_instr,
        factor_sensitivities_df,
        Sigma_Raw_df,
        mispricing_series
    ):
        T_betas = factor_sensitivities_df.loc[primary_instr]
        H_betas = factor_sensitivities_df.loc[hedge_instr]

        dominant_factor = T_betas.abs().idxmax()

        # FIXED: guard against zero hedge sensitivity for the dominant factor
        h_dominant = H_betas[dominant_factor]
        if abs(h_dominant) < 1e-9:
            k = 0.0
        else:
            k = T_betas[dominant_factor] / h_dominant

        residuals = T_betas - k * H_betas

        var_T = Sigma_Raw_df.loc[primary_instr, primary_instr]
        var_H = Sigma_Raw_df.loc[hedge_instr, hedge_instr]
        cov_TH = Sigma_Raw_df.loc[primary_instr, hedge_instr]

        residual_var = var_T + k**2 * var_H - 2 * k * cov_TH
        residual_vol = np.sqrt(max(residual_var, 0)) * 100

        direction = (
            "Sell / Receive" if mispricing_series.get(primary_instr, 0) > 0
            else "Buy / Pay"
        )

        return {
            "Primary Instrument": primary_instr,
            "Hedge Instrument": hedge_instr,
            "Trade Direction": direction,
            "Target Factor": dominant_factor,
            "Hedge Ratio (k)": k,
            "Residual Level": residuals.get("Level (Whole Curve Shift)", np.nan),
            "Residual Slope": residuals.get("Slope (Steepening/Flattening)", np.nan),
            "Residual Curvature": residuals.get("Curvature (Fly Risk)", np.nan),
            "Residual Risk (Rate %)": residual_vol
        }


    # -------------------------------------------------------------------
    # 12.3 PCA MISPRICING CAPTURE (NOT $ PnL)
    # -------------------------------------------------------------------

    def backtest_pca_mispricing_capture(
        primary_instr,
        hedge_instr,
        k,
        historical_derivatives_list,
        holding_days=5
    ):
        mis_ts = {}

        for df in historical_derivatives_list:
            for col in df.columns:
                if col.endswith("(Original)"):
                    base = col.replace(" (Original)", "")
                    pca_col = col.replace("(Original)", "(PCA)")
                    if pca_col in df.columns:
                        mis_ts[base] = (df[col] - df[pca_col]) * 100

        mis_df = pd.DataFrame(mis_ts).dropna()

        if primary_instr not in mis_df or hedge_instr not in mis_df:
            return None

        combo_mis = mis_df[primary_instr] - k * mis_df[hedge_instr]

        # Entry signal: sign of the combined mispricing at entry time.
        # A mean-reversion trade profits when mispricing moves toward zero regardless of sign:
        #   RICH  (combo > 0): sell -> profit = combo_entry - combo_exit  (positive if it converges)
        #   CHEAP (combo < 0): buy  -> profit = -(combo_entry - combo_exit) = combo_exit - combo_entry
        # Without applying the signal direction, CHEAP trades score negative Sharpe even when
        # profitable, biasing hit-rate and Sharpe downward.
        # Fix: effective_capture = sign(combo_entry) * (combo_entry - combo_exit)
        entry_sign = np.sign(combo_mis.shift(-0))   # sign at entry date t
        raw_capture = combo_mis - combo_mis.shift(-holding_days)   # positive = converged
        capture = (entry_sign * raw_capture).dropna()
        cum_capture = capture.cumsum()

        capture_std = capture.std()
        # FIX: capture is an N-day return, NOT a 1-day return.
        # Annualising N-day returns: multiply by sqrt(252/N), not sqrt(252).
        # Using sqrt(252) overstates the Sharpe by sqrt(N) — e.g. 2.24x for N=5.
        sharpe = (capture.mean() / capture_std * np.sqrt(252 / holding_days)
                  if capture_std > 1e-9 else np.nan)

        return {
            "Total Mispricing Captured (Rate %)": cum_capture.iloc[-1],
            "Mean-Reversion Sharpe (annualised)": sharpe,
            "Hit Rate": (capture > 0).mean(),
            "Max Drawdown (Rate %)": (cum_capture - cum_capture.cummax()).min()
        }


    # -------------------------------------------------------------------
    # 12.4 STREAMLIT UI + EXPLANATIONS
    # -------------------------------------------------------------------

    st.header("12. Trade Structuring & PCA Mispricing Capture")

with _tab_trade:

    with st.expander("ℹ️ How to read Section 12 (definitions & formulas)", expanded=False):
        st.markdown(r"""
    ### Mispricing (Rate %)
    \[
    (\text{Market} - \text{PCA Fair}) \times 100
    \]

    ### Factor Purity
    \[
    \frac{\max(|\beta_L|,|\beta_S|,|\beta_C|)}
    {|\beta_L|+|\beta_S|+|\beta_C|}
    \]

    ### Avg Abs Correlation
    \[
    \frac{1}{N}\sum_{j\neq i} |\rho(i,j)|
    \]
    High = proxy / crowded (BAD)

    ### Expression Quality Score
    \[
    \frac{|\text{Mispricing}|\times \text{Factor Purity}}
    {1+\text{Avg Abs Corr}}
    \]

    ### Factor Alignment
    Cosine similarity of factor vectors (≈1 means same idea)

    ### Correlation vs Selected
    \[
    \rho(i,j)
    \]
    High = GOOD (same regional distortion)

    ### PCA Mispricing Capture (NOT $ PnL)
    \[
    (\text{Mis}_T - k\text{Mis}_H)_t -
    (\text{Mis}_T - k\text{Mis}_H)_{t+N}
    \]
    Units are **Rate %**, not dollars.
    """)

    if 'instrument_universe_df' not in globals() or instrument_universe_df is None or instrument_universe_df.empty:
        st.info("Section 12 requires Section 8 to run first (instrument universe not yet built).")
    else:
     selected_instr = st.selectbox(
        "1️⃣ Select instrument where you see distortion",
        instrument_universe_df["Instrument"].values
     )

     quality = compute_expression_quality(
        selected_instr, factor_sensitivities_df, Sigma_Raw_df, mispricing_series
     )

     st.subheader("A. Instrument quality")
     bbg_st_table(pd.DataFrame(quality, index=["Value"]).T)

     alt_df = find_alternative_expressions(
        selected_instr,
        instrument_universe_df,
        factor_sensitivities_df,
        Sigma_Raw_df,
        mispricing_series
     )

     st.subheader("B. Alternative expressions")
     bbg_table(alt_df, use_container_width=True)

     # -- FIX: all downstream code that uses alt_df/selected_instr/trade_instr/combo
     # must remain inside this else block to prevent NameError when Section 8 hasn't run.
     if alt_df.empty:
        st.info("No alternative expressions found for the selected instrument.")
     else:
        trade_instr = st.selectbox(
            "2️⃣ Choose instrument to trade",
            alt_df["Alternative Instrument"].values
        )

        combo = build_factor_isolated_combo(
            selected_instr,
            trade_instr,
            factor_sensitivities_df,
            Sigma_Raw_df,
            mispricing_series
        )

        st.subheader("C. Structured trade")
        bbg_st_table(pd.DataFrame(combo, index=["Value"]).T)

        holding_days = st.slider("Holding period (days)", 1, 20, 5)

        stats = backtest_pca_mispricing_capture(
            selected_instr,
            trade_instr,
            combo["Hedge Ratio (k)"],
            all_historical_derivatives_list,
            holding_days
        )

        if stats:
            st.subheader("D. PCA mispricing capture (NOT $ PnL)")
            bbg_st_table(pd.DataFrame(stats, index=["Value"]).T)

        # ---------------------------------------------------------------------
        # Instrument Level Curves (Separate Views, Actual Levels)
        # ---------------------------------------------------------------------
        st.subheader("Instrument Level Curves (Separate Views, Actual Levels)")

        historical_levels_df = pd.concat(all_historical_derivatives_list, axis=1)

        primary_col = f"{selected_instr} (Original)"
        hedge_col   = f"{trade_instr} (Original)"
        k_star      = combo["Hedge Ratio (k)"]

        if primary_col not in historical_levels_df.columns or hedge_col not in historical_levels_df.columns:
            st.warning("Original level series not available for selected instruments.")
        else:
            primary_series = historical_levels_df[primary_col].dropna()
            hedge_series   = historical_levels_df[hedge_col].dropna()

            common_idx = primary_series.index.intersection(hedge_series.index)

            if len(common_idx) < 10:
                st.warning("Not enough overlapping history for level curves.")
            else:
                primary_series = primary_series.loc[common_idx]
                hedge_series   = hedge_series.loc[common_idx]

                hedged_series = primary_series - k_star * hedge_series

                fig1, ax1 = plt.subplots(figsize=(15, 4))
                ax1.plot(primary_series.index, primary_series.values, linewidth=2.5)
                ax1.set_title(f"Primary Instrument Level: {selected_instr}", fontsize=14)
                ax1.set_ylabel("Instrument Level")
                ax1.grid(True, linestyle=":", alpha=0.6)
                _bbg_fig(fig=fig1)
                st.pyplot(fig1)

                fig2, ax2 = plt.subplots(figsize=(15, 4))
                ax2.plot(hedge_series.index, hedge_series.values, linewidth=2.5, linestyle="--")
                ax2.set_title(f"Hedge Instrument Level: {trade_instr}", fontsize=14)
                ax2.set_ylabel("Instrument Level")
                ax2.grid(True, linestyle=":", alpha=0.6)
                _bbg_fig(fig=fig2)
                st.pyplot(fig2)

                fig3, ax3 = plt.subplots(figsize=(15, 4))
                ax3.plot(hedged_series.index, hedged_series.values, linewidth=2.8)
                ax3.set_title(
                    f"Hedged Synthetic Instrument Level (Primary − {k_star:.3f} × Hedge)",
                    fontsize=14
                )
                ax3.set_xlabel("Date")
                ax3.set_ylabel("Instrument Level")
                ax3.grid(True, linestyle=":", alpha=0.6)
                _bbg_fig(fig=fig3)
                st.pyplot(fig3)



    # ======================
    # END SECTION 12
    # ==========================================================
    # 5.d — FILTERED MISPRICING TABLE + FAMILY FILTER + HEDGES
    # ==========================================================

    st.subheader("5.d Mispricing Filter + Family Selection + Hedge Suggestions")

    # Requires:
    # mispricing_series  -> from calculate_derivative_mispricings()
    # Sigma_Raw_df       -> PCA reconstructed covariance from Section 7

    if mispricing_series is not None and len(mispricing_series) > 0:

        # ------------------------------------------------------
        # BUILD MISPRICING DATAFRAME
        # ------------------------------------------------------
        mispricing_df = mispricing_series.reset_index()
        mispricing_df.columns = ["Instrument", "Mispricing (Rate %)"]

        # --- classify derivative family ---
        def classify_family(name):

            if "3M" in name:
                tenor = "3M"
            elif "6M" in name:
                tenor = "6M"
            elif "12M" in name:
                tenor = "12M"
            else:
                tenor = "Other"

            if "Double Fly" in name:
                typ = "Double Fly"
            elif "Fly" in name:
                typ = "Fly"
            elif "Spread" in name:
                typ = "Spread"
            else:
                typ = "Other"

            return f"{tenor} {typ}".strip()

        mispricing_df["Family"] = mispricing_df["Instrument"].apply(classify_family)
        mispricing_df["Abs Mispricing"] = mispricing_df["Mispricing (Rate %)"].abs()

        # ------------------------------------------------------
        # FILTER CONTROLS
        # ------------------------------------------------------
        col1, col2 = st.columns(2)

        # --- Threshold slider ---
        with col1:
            max_range = float(np.nanmax(np.abs(mispricing_series.values))) if len(mispricing_series) > 0 else 5.0

            threshold_rate = st.slider(
                "Minimum Absolute Mispricing Threshold (Rate %)",
                min_value=0.0,
                max_value=max_range if max_range > 0 else 5.0,
                value=min(0.10, max_range) if max_range > 0 else 0.10,
                step=0.01
            )

        # --- Family filter selector ---
        with col2:
            available_families = sorted(mispricing_df["Family"].unique().tolist())

            selected_families = st.multiselect(
                "Select Derivative Families",
                options=available_families,
                default=available_families,
                help="Filter by tenor and derivative type"
            )

        # ------------------------------------------------------
        # APPLY FILTERS
        # ------------------------------------------------------
        filtered_df = mispricing_df[
            (mispricing_df["Abs Mispricing"] >= threshold_rate) &
            (mispricing_df["Family"].isin(selected_families))
        ].sort_values("Abs Mispricing", ascending=False)

        # ------------------------------------------------------
        # MINIMUM VARIANCE HEDGE ENGINE (PCA COVARIANCE BASED)
        # ------------------------------------------------------
        def find_best_hedge(trade_label, Sigma):
            """
            Minimum Variance Hedge:
                k* = Cov(T,H) / Var(H)
                Residual Var = Var(T) - k*Cov(T,H)
            """
            if Sigma is None or Sigma.empty:
                return None, None, None, None

            if trade_label not in Sigma.index:
                return None, None, None, None

            Var_T = Sigma.loc[trade_label, trade_label]
            best_residual = np.inf
            best_hedge = None
            best_k = None

            for hedge in Sigma.columns:
                if hedge == trade_label:
                    continue

                Var_H = Sigma.loc[hedge, hedge]
                Cov_TH = Sigma.loc[trade_label, hedge]

                if Var_H <= 1e-9:
                    continue

                k = Cov_TH / Var_H
                residual_var = Var_T - k * Cov_TH
                residual_var = max(residual_var, 0)

                if residual_var < best_residual:
                    best_residual = residual_var
                    best_hedge = hedge
                    best_k = k

            if best_hedge is None:
                return None, None, None, None

            residual_vol = np.sqrt(best_residual) * 100
            action = "Short Hedge" if best_k > 0 else "Long Hedge"

            return best_hedge, abs(best_k), residual_vol, action

        # ------------------------------------------------------
        # COMPUTE HEDGE SUGGESTIONS
        # ------------------------------------------------------
        hedge_list = []
        hedge_ratio_list = []
        residual_list = []
        action_list = []

        for instr in filtered_df["Instrument"]:

            if 'Sigma_Raw_df' in globals() and not Sigma_Raw_df.empty:
                hedge, k, resid, action = find_best_hedge(instr, Sigma_Raw_df)
            else:
                hedge, k, resid, action = None, None, None, None

            hedge_list.append(hedge if hedge else "N/A")
            hedge_ratio_list.append(k if k else np.nan)
            residual_list.append(resid if resid else np.nan)
            action_list.append(action if action else "N/A")

        filtered_df["Suggested Hedge"] = hedge_list
        filtered_df["Hedge Ratio |k*|"] = hedge_ratio_list
        filtered_df["Hedge Action"] = action_list
        filtered_df["Residual Risk After Hedge (Rate %)"] = residual_list

        # ------------------------------------------------------
        # DISPLAY OUTPUT
        # ------------------------------------------------------
        if filtered_df.empty:
            st.info("No instruments match selected filters.")
        else:
            st.metric("Filtered Instruments", len(filtered_df))

            st.caption("""
    Hedge Basis: **Minimum Variance Hedge using PCA Risk Model**

    • PCA reconstructed covariance matrix  
    • Hedge ratio: k* = Cov(trade, hedge) / Var(hedge)  
    • Hedge selected to minimize residual volatility  
    • Residual risk shows remaining exposure after hedge
    """)

            bbg_table(
                filtered_df.drop(columns=["Abs Mispricing"]).style.format({
                    "Mispricing (Rate %)": "{:.4f}",
                    "Hedge Ratio |k*|": "{:.4f}",
                    "Residual Risk After Hedge (Rate %)": "{:.4f}"
                }),
                use_container_width=True
            )

    else:
        st.info("Mispricing data not available.")
        # ==========================================================
    # ==========================================================
    # ==========================================================
    # ==========================================================
    # 5.e — TRADE RELATIONSHIP EXPLORER
    # (Rolling Lookback + Correlation + Lead/Lag + Granger)
    # ==========================================================

    st.subheader("5.e Trade Relationship Explorer (Correlation + Lead/Lag + Granger Causality)")

    from statsmodels.tsa.stattools import grangercausalitytests  # FIXED: moved statsmodels import here (not at top to keep optional)

    try:

        # ------------------------------------------------------
        # BUILD MASTER DERIVATIVE TIMESERIES MATRIX
        # ------------------------------------------------------
        def extract_original_columns(df):
            if df is None or df.empty:
                return pd.DataFrame()

            cols = [c for c in df.columns if "(Original)" in c]
            if not cols:
                return pd.DataFrame()

            clean = df[cols].copy()
            clean.columns = [c.replace(" (Original)", "") for c in cols]
            return clean

        all_derivatives_list = [
            extract_original_columns(historical_spreads_3M_df),
            extract_original_columns(historical_butterflies_3M_df),
            extract_original_columns(historical_double_butterflies_3M_df),
            extract_original_columns(historical_spreads_6M_df),
            extract_original_columns(historical_butterflies_6M_df),
            extract_original_columns(historical_double_butterflies_6M_df),
            extract_original_columns(historical_spreads_12M_df),
            extract_original_columns(historical_butterflies_12M_df),
            extract_original_columns(historical_double_butterflies_12M_df),
        ]

        derivatives_ts = pd.concat(all_derivatives_list, axis=1).dropna(axis=1, how="all")

        if derivatives_ts.empty:
            st.info("No derivative time series available.")
            st.stop()

        # ------------------------------------------------------
        # ROLLING LOOKBACK WINDOW (FIXED ROLLING BACK)
        # ------------------------------------------------------
        total_days_available = len(derivatives_ts)

        if total_days_available < 30:
            st.warning("Not enough data for rolling analysis (minimum 30 days required).")
            st.stop()

        lookback_days = st.slider(
            "Rolling Lookback Window (Days Used for Analysis)",
            min_value=30,
            max_value=total_days_available,
            value=min(250, total_days_available)
        )

        # use most recent N days
        derivatives_ts = derivatives_ts.tail(lookback_days)

        # ------------------------------------------------------
        # TRADE SELECTION
        # ------------------------------------------------------
        trade_selected = st.selectbox(
            "Select Trade",
            sorted(derivatives_ts.columns.tolist())
        )

        # ------------------------------------------------------
        # FILTER CONTROLS
        # ------------------------------------------------------
        col1, col2, col3 = st.columns(3)

        with col1:
            corr_threshold = st.slider(
                "Min |Correlation|",
                0.0, 1.0, 0.50, 0.05
            )

        with col2:
            max_lag_days = st.slider(
                "Max Lead/Lag Days",
                1, 20, 5
            )

        with col3:
            granger_p_threshold = st.slider(
                "Max Granger p-value",
                0.01, 1.0, 0.05, 0.01
            )

        # ------------------------------------------------------
        # FFT LEAD/LAG DETECTION
        # ------------------------------------------------------
        def compute_lead_lag_fft(trade_series, other_series, max_lag):

            df = pd.concat([trade_series, other_series], axis=1).dropna()
            if len(df) < 50:
                return None, 0

            x = (df.iloc[:,0] - df.iloc[:,0].mean()) / df.iloc[:,0].std()
            y = (df.iloc[:,1] - df.iloc[:,1].mean()) / df.iloc[:,1].std()

            corr = np.correlate(x, y, mode="full")
            lags = np.arange(-len(x)+1, len(x))

            mask = (lags >= -max_lag) & (lags <= max_lag)
            corr = corr[mask]
            lags = lags[mask]

            idx = np.argmax(np.abs(corr))
            return corr[idx] / len(x), lags[idx]

        # ------------------------------------------------------
        # GRANGER CAUSALITY TEST
        # ------------------------------------------------------
        def granger_test(trade_series, other_series, max_lag=5):

            df = pd.concat([trade_series, other_series], axis=1).dropna()
            if len(df) < 100:
                return None

            try:
                result = grangercausalitytests(df, maxlag=max_lag, verbose=False)
                pvals = [result[i+1][0]["ssr_ftest"][1] for i in range(max_lag)]
                return min(pvals)
            except Exception:  # FIXED: bare except replaced with except Exception
                return None

        # ------------------------------------------------------
        # COMPUTE RELATIONSHIPS
        # ------------------------------------------------------
        trade_series_full = derivatives_ts[trade_selected].dropna()
        aligned_df = derivatives_ts.loc[trade_series_full.index]

        results = []

        for col in aligned_df.columns:

            if col == trade_selected:
                continue

            other_series = aligned_df[col].dropna()
            common_idx = trade_series_full.index.intersection(other_series.index)

            if len(common_idx) < 100:
                continue

            trade_series = trade_series_full.loc[common_idx]
            other_series = other_series.loc[common_idx]

            # correlation
            corr = trade_series.corr(other_series)

            # lag detection
            best_corr, best_lag = compute_lead_lag_fft(
                trade_series, other_series, max_lag_days
            )

            # granger test
            p_val = granger_test(trade_series, other_series, max_lag_days)

            if best_lag > 0:
                relation = "Trade Leads"
            elif best_lag < 0:
                relation = "Trade Follows"
            else:
                relation = "Simultaneous"

            if p_val is not None:
                if p_val < 0.01:
                    predict_strength = "Very Strong"
                elif p_val < 0.05:
                    predict_strength = "Predictive"
                else:
                    predict_strength = "Weak"
            else:
                predict_strength = "N/A"

            results.append({
                "Trade": trade_selected,
                "Instrument": col,
                "Correlation": corr,
                "Lag (Days)": best_lag,
                "Relationship": relation,
                "Granger p-value": p_val,
                "Predictive Strength": predict_strength,
                "Abs Correlation": abs(corr) if corr is not None else 0
            })

        if not results:
            st.info("No relationships found.")
            st.stop()

        df_results = pd.DataFrame(results)

        # ------------------------------------------------------
        # APPLY FILTERS
        # ------------------------------------------------------
        filtered = df_results[
            (df_results["Abs Correlation"] >= corr_threshold) &
            ((df_results["Granger p-value"].isna()) |
             (df_results["Granger p-value"] <= granger_p_threshold))
        ].sort_values("Abs Correlation", ascending=False)

        # ------------------------------------------------------
        # DISPLAY
        # ------------------------------------------------------
        if filtered.empty:
            st.info("No instruments match filters.")
        else:
            st.metric("Filtered Relationships", len(filtered))

            st.caption("""
    • Uses rolling lookback window from selected date range  
    • Correlation → co-movement strength  
    • Lag → who moves first  
    • Granger p-value → predictive power (lower = stronger)
    """)

            bbg_table(
                filtered.drop(columns=["Abs Correlation"]).style.format({
                    "Correlation": "{:.3f}",
                    "Granger p-value": "{:.4f}"
                }),
                use_container_width=True
            )

    except Exception as e:
        st.warning(f"Relationship analysis unavailable: {e}")

    # ============================================================
    # FINAL BLOCK — COMBINED SECTION 5 + SECTION 10 EXPORT
    # ============================================================

    # re is already imported in Section 9; BytesIO and PdfPages are imported at the top

    # ---------- Helper: normalize derivative names for pairing ----------
    def _normalize_derivative_name(title: str) -> str:
        """
        Extracts a canonical key used to pair Section 5 and Section 10 charts.
        Examples:
          'Section 5 – 3M Spread'          → '3M Spread'
          'Section 10 – 3M Spread (1σ…)'   → '3M Spread'
          'Section 5 – 3M Butterfly'        → '3M Butterfly'
          'Section 10 – 3M Double Fly'      → '3M Double Fly'
        """
        # Strip "Section N –" prefix
        if "–" in title:
            title = title.split("–", 1)[1].strip()
        # Strip envelope suffix
        for suffix in ["(1σ & 2σ)", "σ"]:
            title = title.replace(suffix, "")
        return title.strip()


    # ---------- Build paired ordering ----------
    def _build_combined_figure_order(section5_figs, section9_figs):
        """
        Interleaves Section 5 and Section 10 charts by derivative type.
        Order: 3M Spread S5, 3M Spread S10, 3M Butterfly S5, 3M Butterfly S10, ...
        Sort: tenor (3M < 6M < 12M) then type (Spread < Butterfly < Double Fly < Outright).
        """
        TYPE_ORDER = {
            "Outright Curve": 0,
            "Spread": 1, "Butterfly": 2, "Double Fly": 3, "Double Butterfly": 3,
        }
        TENOR_ORDER = {"3M": 0, "6M": 1, "12M": 2}

        def _sort_key(k: str):
            tenor = next((v for t, v in TENOR_ORDER.items() if t in k), 99)
            typ   = next((v for t, v in TYPE_ORDER.items() if t in k), 99)
            return (tenor, typ)

        grouped = {}
        for fig, name in section5_figs:
            key = _normalize_derivative_name(name)
            grouped.setdefault(key, {})["sec5"] = (fig, name)
        for fig, name in section9_figs:
            key = _normalize_derivative_name(name)
            grouped.setdefault(key, {})["sec10"] = (fig, name)

        ordered_keys = sorted(grouped.keys(), key=_sort_key)

        ordered = []
        for k in ordered_keys:
            block = grouped[k]
            if "sec5"  in block: ordered.append(block["sec5"])
            if "sec10" in block: ordered.append(block["sec10"])
        return ordered


    # ---------- Export PDF ----------
    def _export_full_curve_pdf(analysis_date_str: str):
        sec5  = st.session_state.get("SECTION5_FIGURES", [])
        sec10 = st.session_state.get("SECTION9_FIGURES", [])

        if not sec5 and not sec10:
            return None

        ordered = _build_combined_figure_order(sec5, sec10)

        buffer = BytesIO()
        with PdfPages(buffer) as pdf:
            d = pdf.infodict()
            d['Title']   = f"SOFR Futures PCA — Full Curve Diagnostics ({analysis_date_str})"
            d['Subject'] = "PCA Curve Snapshots & Precision Adaptive Envelopes"

            for fig, title in ordered:
                try:
                    display_title = title if title else "Chart"
                    # Use a COPY of the figure layout so we don't mutate the live fig
                    import copy as _copy
                    fig_copy = _copy.deepcopy(fig)
                    fig_copy.suptitle(
                        f"{display_title}\nAnalysis Date: {analysis_date_str}",
                        fontsize=10, y=1.01
                    )
                    pdf.savefig(fig_copy, bbox_inches="tight")
                    plt.close(fig_copy)
                except Exception:
                    # Fallback: save original without title mutation
                    try:
                        pdf.savefig(fig, bbox_inches="tight")
                    except Exception:
                        pass

        buffer.seek(0)
        return buffer

    # ---------- UI ----------
    st.markdown("---")
    st.header("📥 Full Curve Diagnostics Export")

with _tab_export:

    st.write(
    """
    Downloads ONE combined PDF containing:

    • Section 5 — Market vs PCA snapshots (all derivative families)
    • Section 10 — Precision Adaptive Envelopes (1σ & 2σ)

    Charts are paired by derivative type: Section 5 chart followed immediately by its Section 10 counterpart.
    Order: Outright → 3M Spread → 3M Butterfly → 3M Double Fly → 6M Spread → ...
    """
    )

    col_pdf1, col_pdf2 = st.columns(2)
    with col_pdf1:
        st.metric("Section 5 charts ready", len(st.session_state.get("SECTION5_FIGURES", [])))
    with col_pdf2:
        st.metric("Section 10 charts ready", len(st.session_state.get("SECTION9_FIGURES", [])))

    generate_pdf = st.button("Prepare Combined PDF", use_container_width=True)

    if generate_pdf:
        st.session_state.SNAPSHOT_READY = True
        full_pdf = _export_full_curve_pdf(str(analysis_date))
    else:
        full_pdf = None

    if full_pdf is not None:
        # Filename: SOFR_PCA_Full_Diagnostics_<analysis_date>.pdf
        safe_date = str(analysis_date).replace("/", "-").replace(" ", "_")
        pdf_filename = f"SOFR_{safe_date}.pdf"
        st.download_button(
            label=f"⬇️ Download Full Curve Diagnostics PDF  ({safe_date})",
            data=full_pdf,
            file_name=pdf_filename,
            mime="application/pdf",
            use_container_width=True
        )
    else:
        st.info("Generate Section 5 and Section 10 charts first, then click Prepare Combined PDF.")
