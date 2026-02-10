import streamlit as st
import pandas as pd
import numpy as np
from datetime import date

st.set_page_config(layout="wide")
st.title("ZQ → FOMC Meetings → Exact Compounded SR3")

# ======================================================
# 1. FOMC MEETING CALENDAR
# ======================================================
fomc_dates = pd.to_datetime([
    "2026-03-18", "2026-05-06", "2026-06-17", "2026-07-29",
    "2026-09-16", "2026-11-05", "2026-12-16",
    "2027-01-27", "2027-03-17", "2027-04-28",
    "2027-06-16", "2027-07-28"
]).sort_values()

# ======================================================
# 2. FIND FIRST FUTURE MEETING → ANCHOR MONTH
# ======================================================
today = pd.Timestamp.today().normalize()
future_meetings = fomc_dates[fomc_dates >= today]

if future_meetings.empty:
    st.error("No future FOMC meetings found.")
    st.stop()

first_meeting = future_meetings.min()

anchor_month = (
    first_meeting.to_period("M").to_timestamp()
    - pd.DateOffset(months=1)
)

months = pd.date_range(anchor_month, periods=13, freq="MS")

# ======================================================
# 3. ZQ INPUT TABLE
# ======================================================
st.subheader("ZQ Prices (Anchor + Next 12 Months)")

zq_df = pd.DataFrame({
    "Month": months.strftime("%b-%Y"),
    "ZQ Price": [96.50] * 13
})

zq_df = st.data_editor(
    zq_df,
    num_rows="fixed",
    use_container_width=True
)

zq_df["Month Rate"] = 100 - zq_df["ZQ Price"]

# ======================================================
# 4. BUILD MONTH / MEETING STRUCTURE
# ======================================================
rows = []

for i, m in enumerate(months):
    month_end = m + pd.offsets.MonthEnd(1)
    days = (month_end - m).days + 1

    meeting_idx = fomc_dates[
        (fomc_dates.month == m.month) &
        (fomc_dates.year == m.year)
    ]
    meeting_date = meeting_idx[0] if len(meeting_idx) else pd.NaT

    if pd.notna(meeting_date):
        pre = (meeting_date - m).days
        post = days - pre
    else:
        pre, post = days, 0

    rows.append({
        "Month": m.strftime("%b-%Y"),
        "Meeting Date": meeting_date,
        "Days": days,
        "Pre": pre,
        "Post": post,
        "Month Rate": zq_df.iloc[i]["Month Rate"],
    })

month_df = pd.DataFrame(rows)

# ======================================================
# 5. ZQ → MEETING PREMIA
# ======================================================
anchor_rate = month_df.iloc[0]["Month Rate"]
anchor_label = month_df.iloc[0]["Month"]

work_df = month_df.iloc[1:].reset_index(drop=True)

prev_rate = anchor_rate
premia = []

for _, r in work_df.iterrows():
    if r["Post"] == 0:
        premia.append(0.0)
        continue

    w_pre = r["Pre"] / r["Days"]
    w_post = r["Post"] / r["Days"]

    new_rate = (r["Month Rate"] - w_pre * prev_rate) / w_post
    premia.append((new_rate - prev_rate) * 100)
    prev_rate = new_rate

work_df["Meeting Premium (bps)"] = premia
work_df["Policy Rate After Meeting"] = (
    anchor_rate + np.cumsum(work_df["Meeting Premium (bps)"]) / 100
)

st.subheader("Implied FOMC Meeting Premia (from ZQ)")
st.dataframe(
    work_df[["Month", "Meeting Date", "Meeting Premium (bps)", "Policy Rate After Meeting"]],
    use_container_width=True
)

# ======================================================
# 6. BUILD EXACT POLICY STEP FUNCTION
# ======================================================
policy_steps = []
current_rate = anchor_rate

for _, r in work_df.iterrows():
    meet = r["Meeting Date"]
    if pd.isna(meet):
        continue

    policy_steps.append((meet.normalize(), current_rate))
    current_rate += r["Meeting Premium (bps)"] / 100

policy_steps.append((pd.Timestamp.max, current_rate))

# ======================================================
# 7. EXACT DAILY COMPOUNDING FUNCTION
# ======================================================
def compounded_sr3_rate(start, end, steps):
    acc = 1.0
    D = (end - start).days

    for i in range(len(steps) - 1):
        seg_start = max(start, steps[i][0])
        seg_end = min(end, steps[i + 1][0])

        if seg_start >= seg_end:
            continue

        rate = steps[i][1]
        days = (seg_end - seg_start).days
        acc *= (1 + rate / 360) ** days

    return 360 * (acc - 1) / D

# ======================================================
# 8. SR3 CONTRACTS
# ======================================================
sr3_quarters = pd.DataFrame([
    ("SR3H6", "2026-03-18", "2026-06-17"),
    ("SR3M6", "2026-06-17", "2026-09-16"),
    ("SR3U6", "2026-09-16", "2026-12-16"),
    ("SR3Z6", "2026-12-16", "2027-03-17"),
], columns=["SR3", "Start", "End"])

sr3_quarters["Start"] = pd.to_datetime(sr3_quarters["Start"])
sr3_quarters["End"] = pd.to_datetime(sr3_quarters["End"])
sr3_quarters = sr3_quarters.sort_values("Start")

# ======================================================
# 9. EXACT SR3 PRICING
# ======================================================
sr3_rows = []

for _, q in sr3_quarters.iterrows():
    sr3_rate = compounded_sr3_rate(q["Start"], q["End"], policy_steps)

    sr3_rows.append({
        "SR3": q["SR3"],
        "Start": q["Start"].date(),
        "End": q["End"].date(),
        "Implied SR3 Rate (%)": round(sr3_rate, 5),
        "Implied SR3 Price": round(100 - sr3_rate, 5),
    })

sr3_df = pd.DataFrame(sr3_rows)

st.subheader("SR3 – Exact Compounded Pricing")
st.dataframe(sr3_df, use_container_width=True)

# ======================================================
# 10. SR3 PRICE SPREADS (FRONT - NEXT)
# ======================================================
sr3_spreads_df = pd.DataFrame([
    {
        "Spread": f"{sr3_df.iloc[i]['SR3']} - {sr3_df.iloc[i+1]['SR3']}",
        "SR3 Price Spread": round(
            sr3_df.iloc[i]["Implied SR3 Price"]
            - sr3_df.iloc[i+1]["Implied SR3 Price"],
            5
        )
    }
    for i in range(len(sr3_df) - 1)
])

st.subheader("SR3 Calendar Spreads (Price)")
st.dataframe(sr3_spreads_df, use_container_width=True)

# ======================================================
# 11. FOOTER
# ======================================================
st.caption(
    f"As of {date.today()} | "
    f"Anchor month: {anchor_label} | "
    f"Anchor policy rate: {anchor_rate:.3f}%"
)
