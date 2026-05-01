import io
import zipfile

import pandas as pd
import streamlit as st

st.set_page_config(page_title="GSC AI Overview Estimator", layout="wide")

INTENT_PATTERNS = {
    "informational": [
        "what is", "how to", "how do", "why", "when", "where", "who", "can ",
        "does ", "should ", "guide", "tips", "examples", "template", "meaning",
    ],
    "comparative": [
        "best", "top", "vs", "versus", "compare", "comparison", "alternatives", "software",
    ],
    "navigational": [
        "login", "sign in", "homepage", "contact", "address", "phone", "hours", "near me",
    ],
}


def normalize_page(value: str) -> str:
    if pd.isna(value):
        return ""
    value = str(value).strip()
    if value.startswith("http://") or value.startswith("https://"):
        from urllib.parse import urlparse
        parsed = urlparse(value)
        return parsed.path or "/"
    return value


def expected_ctr_for_position(position: float) -> float:
    if pd.isna(position):
        return 0.0
    if position <= 1:
        return 0.28
    if position <= 2:
        return 0.16
    if position <= 3:
        return 0.10
    if position <= 5:
        return 0.05
    if position <= 8:
        return 0.025
    if position <= 10:
        return 0.015
    return 0.008


def detect_intent(query: str) -> str:
    q = str(query or "").strip().lower()
    for intent, patterns in INTENT_PATTERNS.items():
        if any(p in q for p in patterns):
            return intent
    return "other"


def is_branded(query: str, brand_terms: list[str]) -> bool:
    q = str(query or "").lower()
    return any(term.lower() in q for term in brand_terms if term.strip())


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def prepare_gsc(df: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "page", "query", "clicks", "impressions", "ctr", "position"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    data = df.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date"])
    data["date"] = data["date"].dt.date.astype(str)
    data["page"] = data["page"].map(normalize_page)

    for col in ["clicks", "impressions", "ctr", "position"]:
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)

    if data["ctr"].max() > 1:
        data["ctr"] = data["ctr"] / 100.0

    return data


def score_row(row: pd.Series, brand_terms: list[str]) -> pd.Series:
    score = 0.0
    reasons = []

    query = str(row.get("query", ""))
    position = float(row.get("position", 0) or 0)
    clicks = float(row.get("clicks", 0) or 0)
    impressions = float(row.get("impressions", 0) or 0)
    ctr = float(row.get("ctr", 0) or 0)

    intent = detect_intent(query)
    branded = is_branded(query, brand_terms)
    expected_ctr = expected_ctr_for_position(position)

    if position > 5:
        score += 0.20
        reasons.append("Weak average rank but still receiving clicks")
    if position > 8:
        score += 0.15
    if clicks >= 3:
        score += 0.10
    if impressions >= 50:
        score += 0.10
    if ctr > expected_ctr:
        score += 0.20
        reasons.append("CTR is above expected for this position")

    if intent == "informational":
        score += 0.15
        reasons.append("Informational query")
    elif intent == "comparative":
        score += 0.15
        reasons.append("Comparative query")
    elif intent == "navigational":
        score -= 0.30
        reasons.append("Navigational query")

    if branded:
        score -= 0.25
        reasons.append("Branded query")

    score = clamp01(score)

    confidence = 0.30
    if impressions >= 100:
        confidence += 0.20
    if clicks >= 5:
        confidence += 0.15
    if position > 5 and ctr > expected_ctr:
        confidence += 0.15
    if intent in {"informational", "comparative"}:
        confidence += 0.10
    if impressions < 20:
        confidence -= 0.15
    confidence = clamp01(confidence)

    return pd.Series({
        "intent": intent,
        "branded": branded,
        "expected_ctr": round(expected_ctr, 4),
        "aio_likelihood": round(score, 4),
        "confidence": round(confidence, 4),
        "why": "; ".join(reasons[:3]) if reasons else "No strong signals",
    })


def bucket_label(score: float) -> str:
    if score >= 0.75:
        return "Very likely"
    if score >= 0.50:
        return "Likely"
    if score >= 0.25:
        return "Possible"
    return "Unlikely"


st.title("GSC AI Overview Estimator")
st.caption("Upload a Google Search Console CSV and estimate which queries are most likely benefiting from AI Overviews.")

with st.sidebar:
    st.header("Settings")
    brand_input = st.text_input("Brand terms (comma-separated)", value="")
    min_impressions = st.slider("Minimum impressions", 0, 1000, 50, 10)
    min_clicks = st.slider("Minimum clicks", 0, 100, 0, 1)
    min_score = st.slider("Minimum AIO likelihood", 0.0, 1.0, 0.25, 0.05)
    brand_terms = [x.strip() for x in brand_input.split(",") if x.strip()]

uploaded_file = st.file_uploader("Upload Search Console CSV or ZIP", type=["csv", "zip"])

with st.expander("Required columns"):
    st.code("date, page, query, clicks, impressions, ctr, position")

if uploaded_file is None:
    st.info("Upload a Search Console CSV export to begin.")
    st.stop()

try:
    if uploaded_file.name.lower().endswith(".zip"):
        z = zipfile.ZipFile(uploaded_file)
        csv_files = [name for name in z.namelist() if name.lower().endswith(".csv") and not name.endswith("/")]

        if not csv_files:
            st.error("No CSV file found inside the ZIP archive.")
            st.stop()

        if len(csv_files) > 1:
            selected_csv = st.selectbox("Select CSV file from ZIP", csv_files)
        else:
            selected_csv = csv_files[0]
            st.info(f"Using file from ZIP: {selected_csv}")

        with z.open(selected_csv) as f:
            raw = pd.read_csv(f)
    else:
        raw = pd.read_csv(uploaded_file)

    gsc = prepare_gsc(raw)
except Exception as exc:
    st.error(f"Could not read file: {exc}")
    st.stop()

scored = pd.concat([gsc, gsc.apply(score_row, axis=1, args=(brand_terms,))], axis=1)
scored["likelihood_bucket"] = scored["aio_likelihood"].apply(bucket_label)

filtered = scored[
    (scored["impressions"] >= min_impressions)
    & (scored["clicks"] >= min_clicks)
    & (scored["aio_likelihood"] >= min_score)
].copy()

query_summary = (
    filtered.groupby(["query", "page"], as_index=False)
    .agg({
        "clicks": "sum",
        "impressions": "sum",
        "ctr": "mean",
        "position": "mean",
        "aio_likelihood": "mean",
        "confidence": "mean",
        "why": "first",
        "intent": "first",
        "likelihood_bucket": "first",
    })
    .sort_values(["aio_likelihood", "confidence", "clicks"], ascending=False)
)

page_summary = (
    filtered.groupby("page", as_index=False)
    .agg({
        "clicks": "sum",
        "impressions": "sum",
        "aio_likelihood": "mean",
        "confidence": "mean",
        "query": "count",
    })
    .rename(columns={"query": "flagged_queries"})
    .sort_values(["aio_likelihood", "clicks"], ascending=False)
)

trend = (
    filtered.groupby("date", as_index=False)
    .agg({
        "clicks": "sum",
        "impressions": "sum",
        "aio_likelihood": "mean",
    })
    .sort_values("date")
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Queries analyzed", f"{len(scored):,}")
c2.metric("Queries flagged", f"{len(filtered):,}")
c3.metric("Avg AIO likelihood", f"{filtered['aio_likelihood'].mean():.1%}" if len(filtered) else "0.0%")
c4.metric("Avg confidence", f"{filtered['confidence'].mean():.1%}" if len(filtered) else "0.0%")

left, right = st.columns([1.2, 1])
with left:
    st.subheader("Estimated AI Overview Trend")
    if len(trend):
        st.line_chart(trend.set_index("date")[["aio_likelihood", "clicks", "impressions"]])
    else:
        st.write("No rows match your filters.")

with right:
    st.subheader("Top Pages")
    st.dataframe(page_summary.head(15), use_container_width=True)

st.subheader("Top Queries Likely Getting AI Overviews")
st.dataframe(
    query_summary[[
        "query", "page", "position", "clicks", "impressions", "ctr",
        "intent", "aio_likelihood", "confidence", "likelihood_bucket", "why"
    ]],
    use_container_width=True,
)

st.subheader("Raw Scored Data")
st.dataframe(scored.sort_values(["aio_likelihood", "confidence"], ascending=False), use_container_width=True)

st.download_button(
    "Download scored queries CSV",
    data=scored.to_csv(index=False).encode("utf-8"),
    file_name="gsc_aio_scored_queries.csv",
    mime="text/csv",
)

st.download_button(
    "Download filtered summary CSV",
    data=query_summary.to_csv(index=False).encode("utf-8"),
    file_name="gsc_aio_query_summary.csv",
    mime="text/csv",
)

with st.expander("How this works"):
    st.markdown(
        """
        This prototype estimates which queries may be benefiting from AI Overviews using only Search Console data.

        It boosts scores when a query:
        - ranks outside the top positions but still gets clicks
        - has a CTR above what we would roughly expect for that position
        - looks informational or comparative

        It lowers scores for:
        - branded queries
        - navigational queries

        This is an estimate, not exact AI Overview attribution.
        """
    )
