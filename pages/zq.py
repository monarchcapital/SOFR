import streamlit as st
import pandas as pd
import numpy as np
from datetime import date

st.set_page_config(layout="wide")
st.title("ZQ → FOMC Meeting Premia → SR3 Pricing & Spreads")

# ======================================================
# 1. FOMC MEETING CALENDAR
# ======================================================
fomc_dates = pd.to_datetime([
    "2026-03-18", "2026-05-06", "2026-06-17", "2026-07-29",
    "2026-09-16", "2026-11-05", "2026-12-16",
    "2027-01-27", "2027-03-17", "2027-04-28",
    "2027-06-16", "2027-07-28"
])

# ======================================================
# 2. FIND FIRST FUTURE MEETING → ANCHOR MONTH
# ======================================================
today = pd.Timestamp.today()
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
# 3. ZQ INPUT TABLE (ANCHOR + 12)
# ======================================================
st.subheader("ZQ Prices (Anchor + Next 12 Months)")

zq_df = pd.DataFrame({
    "Month": months.strftime("%b-%Y"),
    "ZQ Price": [94.50] * 13
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
        "Month Rate": zq_df.iloc[i]["Month Rate"]
    })

month_df = pd.DataFrame(rows)

# ======================================================
# 5. ZQ → MEETING PREMIA
# ======================================================
anchor_rate = month_df.iloc[0]["Month Rate"]
anchor_label = month_df.iloc[0]["Month"]

work_df = month_df.iloc[1:].reset_index(drop=True)

prev = anchor_rate
premia = []

for _, r in work_df.iterrows():
    if r["Post"] == 0:
        premia.append(0.0)
        continue

    w_pre = r["Pre"] / r["Days"]
    w_post = r["Post"] / r["Days"]

    new_rate = (r["Month Rate"] - w_pre * prev) / w_post
    premia.append((new_rate - prev) * 100)
    prev = new_rate

work_df["Meeting Premium (bps)"] = premia
work_df["Policy Path"] = anchor_rate + np.cumsum(work_df["Meeting Premium (bps)"]) / 100

st.subheader("Implied FOMC Meeting Premia (from ZQ)")
st.dataframe(
    work_df[["Month", "Meeting Date", "Meeting Premium (bps)", "Policy Path"]],
    use_container_width=True
)

# ======================================================
# 6. SR3 QUARTERS
# ======================================================
sr3_quarters = pd.DataFrame([
    ("SR3H6", "2026-03-18", "2026-06-17"),
    ("SR3M6", "2026-06-17", "2026-09-16"),
    ("SR3U6", "2026-09-16", "2026-12-16"),
    ("SR3Z6", "2026-12-16", "2027-03-17"),
], columns=["SR3", "Start", "End"])

sr3_quarters["Start"] = pd.to_datetime(sr3_quarters["Start"])
sr3_quarters["End"] = pd.to_datetime(sr3_quarters["End"])

# ======================================================
# 7. MAP MEETING PREMIA → SR3
# ======================================================
sr3_detail_rows = []
sr3_summary_rows = []

for _, q in sr3_quarters.iterrows():
    total_days = (q["End"] - q["Start"]).days
    total_contribution_bps = 0.0

    for _, m in work_df.iterrows():
        meet = m["Meeting Date"]
        if pd.isna(meet) or meet < q["Start"] or meet >= q["End"]:
            continue

        days_after = (q["End"] - meet).days
        weight = days_after / total_days
        contrib_bps = weight * m["Meeting Premium (bps)"]
        total_contribution_bps += contrib_bps

        sr3_detail_rows.append({
            "SR3": q["SR3"],
            "Meeting Date": meet.date(),
            "Meeting Premium (bps)": round(m["Meeting Premium (bps)"], 2),
            "Weight": round(weight, 4),
            "Contribution (bps)": round(contrib_bps, 2),
        })

    implied_rate = anchor_rate + total_contribution_bps / 100
    implied_price = 100 - implied_rate

    sr3_summary_rows.append({
        "SR3": q["SR3"],
        "Anchor Rate (%)": round(anchor_rate, 3),
        "Total Meeting Impact (bps)": round(total_contribution_bps, 2),
        "Implied SR3 Rate (%)": round(implied_rate, 3),
        "Implied SR3 Price": round(implied_price, 3),
    })

sr3_summary_df = pd.DataFrame(sr3_summary_rows)

st.subheader("SR3 – Implied Rates & Prices")
st.dataframe(sr3_summary_df, use_container_width=True)

# ======================================================
# 8. ZQ PRICE SPREADS (ZQ1 - ZQ2)
# ======================================================
zq_spread_rows = []

for i in range(len(zq_df) - 1):
    zq_spread_rows.append({
        "Spread": f"{zq_df.iloc[i]['Month']} - {zq_df.iloc[i+1]['Month']}",
        "ZQ Price Spread": round(
            zq_df.iloc[i]["ZQ Price"] - zq_df.iloc[i+1]["ZQ Price"],
            4
        )
    })

zq_spreads_df = pd.DataFrame(zq_spread_rows)

st.subheader("ZQ Calendar Spreads (Price)")
st.dataframe(zq_spreads_df, use_container_width=True)

# ======================================================
# 9. ZQ PRICE FLIES (ZQ1 - 2*ZQ2 + ZQ3)
# ======================================================
zq_fly_rows = []

for i in range(len(zq_df) - 2):
    zq_fly_rows.append({
        "Fly": f"{zq_df.iloc[i]['Month']} - {zq_df.iloc[i+1]['Month']} - {zq_df.iloc[i+2]['Month']}",
        "ZQ Fly (Price)": round(
            zq_df.iloc[i]["ZQ Price"]
            - 2 * zq_df.iloc[i+1]["ZQ Price"]
            + zq_df.iloc[i+2]["ZQ Price"],
            4
        )
    })

zq_flies_df = pd.DataFrame(zq_fly_rows)

st.subheader("ZQ Flies (Price)")
st.dataframe(zq_flies_df, use_container_width=True)

# ======================================================
# 10. SR3 PRICE SPREADS (SR3H - SR3M)
# ======================================================
sr3_spread_rows = []

for i in range(len(sr3_summary_df) - 1):
    sr3_spread_rows.append({
        "Spread": f"{sr3_summary_df.iloc[i]['SR3']} - {sr3_summary_df.iloc[i+1]['SR3']}",
        "SR3 Price Spread": round(
            sr3_summary_df.iloc[i]["Implied SR3 Price"]
            - sr3_summary_df.iloc[i+1]["Implied SR3 Price"],
            4
        )
    })

sr3_spreads_df = pd.DataFrame(sr3_spread_rows)

st.subheader("SR3 Calendar Spreads (Price)")
st.dataframe(sr3_spreads_df, use_container_width=True)

# ======================================================
# 11. FOOTER
# ======================================================
st.caption(
    f"As of {date.today()} | "
    f"Anchor month: {anchor_label} | "
    f"Anchor rate: {anchor_rate:.2f}%"
)
