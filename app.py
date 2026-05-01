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


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {}
    for col in df.columns:
        clean = str(col).strip().lower()
        if clean == "top queries":
            col_map[col] = "query"
        elif clean == "top pages":
            col_map[col] = "page"
        elif clean == "clicks":
            col_map[col] = "clicks"
        elif clean == "impressions":
            col_map[col] = "impressions"
        elif clean == "ctr":
            col_map[col] = "ctr"
        elif clean == "position":
            col_map[col] = "position"
    return df.rename(columns=col_map)


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


def is_branded(text_value: str, brand_terms: list[str]) -> bool:
    q = str(text_value or "").lower()
    return any(term.lower() in q for term in brand_terms if term.strip())


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def prepare_query_export(df: pd.DataFrame) -> pd.DataFrame:
    data = normalize_columns(df.copy())
    required = {"query", "clicks", "impressions", "ctr", "position"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {', '.join(sorted(missing))}. "
            "Expected headers: Top queries, Clicks, Impressions, CTR, Position"
        )

    data = data[list(required)].copy()
    data["query"] = data["query"].astype(str).str.strip()

    for col in ["clicks", "impressions", "ctr", "position"]:
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)

    if data["ctr"].max() > 1:
        data["ctr"] = data["ctr"] / 100.0

    data = data[data["query"] != ""].copy()
    return data


def prepare_page_export(df: pd.DataFrame) -> pd.DataFrame:
    data = normalize_columns(df.copy())
    required = {"page", "clicks", "impressions", "ctr", "position"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {', '.join(sorted(missing))}. "
            "Expected headers: Top pages, Clicks, Impressions, CTR, Position"
        )

    data = data[list(required)].copy()
    data["page"] = data["page"].map(normalize_page)

    for col in ["clicks", "impressions", "ctr", "position"]:
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)

    if data["ctr"].max() > 1:
        data["ctr"] = data["ctr"] / 100.0

    data = data[data["page"] != ""].copy()
    return data


def score_query_row(row: pd.Series, brand_terms: list[str]) -> pd.Series:
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
        reasons.append("Ranks outside top 5 but still gets clicks")
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


def score_page_row(row: pd.Series, brand_terms: list[str]) -> pd.Series:
    score = 0.0
    reasons = []

    page = str(row.get("page", ""))
    position = float(row.get("position", 0) or 0)
    clicks = float(row.get("clicks", 0) or 0)
    impressions = float(row.get("impressions", 0) or 0)
    ctr = float(row.get("ctr", 0) or 0)

    branded = is_branded(page, brand_terms)
    expected_ctr = expected_ctr_for_position(position)

    if position > 5:
        score += 0.20
        reasons.append("Page ranks outside top 5 but still gets clicks")
    if position > 8:
        score += 0.15
    if clicks >= 3:
        score += 0.10
    if impressions >= 50:
        score += 0.10
    if ctr > expected_ctr:
        score += 0.20
        reasons.append("CTR is above expected for this position")
    if "/blog/" in page or "/guide/" in page or "/resources/" in page:
        score += 0.10
        reasons.append("Looks like informational content")
    if branded:
        score -= 0.15
        reasons.append("Likely branded page")

    score = clamp01(score)

    confidence = 0.30
    if impressions >= 100:
        confidence += 0.20
    if clicks >= 5:
        confidence += 0.15
    if position > 5 and ctr > expected_ctr:
        confidence += 0.15
    if impressions < 20:
        confidence -= 0.15
    confidence = clamp01(confidence)

    return pd.Series({
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
st.caption("Upload either a Search Console query export or page export and estimate what is most likely benefiting from AI Overviews.")

with st.sidebar:
    st.header("Settings")
    mode = st.radio("Upload type", ["Query export", "Page export"])
    brand_input = st.text_input("Brand terms (comma-separated)", value="")
    min_impressions = st.slider("Minimum impressions", 0, 10000, 50, 10)
    min_clicks = st.slider("Minimum clicks", 0, 1000, 0, 1)
    min_score = st.slider("Minimum AIO likelihood", 0.0, 1.0, 0.25, 0.05)
    brand_terms = [x.strip() for x in brand_input.split(",") if x.strip()]

uploaded_file = st.file_uploader("Upload Search Console CSV", type=["csv"])

with st.expander("Expected CSV headers"):
    st.code("Query export: Top queries, Clicks, Impressions, CTR, Position")
    st.code("Page export: Top pages, Clicks, Impressions, CTR, Position")

if uploaded_file is None:
    st.info("Upload your Search Console CSV to begin.")
    st.stop()

try:
    raw = pd.read_csv(uploaded_file)

    if mode == "Query export":
        data = prepare_query_export(raw)
        scored = pd.concat([data, data.apply(score_query_row, axis=1, args=(brand_terms,))], axis=1)
        primary_col = "query"
        table_cols = [
            "query", "clicks", "impressions", "ctr", "position",
            "intent", "aio_likelihood", "confidence", "likelihood_bucket", "why"
        ]
    else:
        data = prepare_page_export(raw)
        scored = pd.concat([data, data.apply(score_page_row, axis=1, args=(brand_terms,))], axis=1)
        primary_col = "page"
        table_cols = [
            "page", "clicks", "impressions", "ctr", "position",
            "aio_likelihood", "confidence", "likelihood_bucket", "why"
        ]

except Exception as exc:
    st.error(f"Could not read file: {exc}")
    st.stop()

scored["likelihood_bucket"] = scored["aio_likelihood"].apply(bucket_label)

filtered = scored[
    (scored["impressions"] >= min_impressions)
    & (scored["clicks"] >= min_clicks)
    & (scored["aio_likelihood"] >= min_score)
].copy()

filtered = filtered.sort_values(
    ["aio_likelihood", "confidence", "clicks", "impressions"],
    ascending=False,
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows analyzed", f"{len(scored):,}")
c2.metric("Rows flagged", f"{len(filtered):,}")
c3.metric("Avg AIO likelihood", f"{filtered['aio_likelihood'].mean():.1%}" if len(filtered) else "0.0%")
c4.metric("Avg confidence", f"{filtered['confidence'].mean():.1%}" if len(filtered) else "0.0%")

st.subheader(f"Top {primary_col.title()}s Likely Getting AI Overviews")
st.dataframe(
    filtered[table_cols],
    use_container_width=True,
)

st.subheader("Raw Scored Data")
st.dataframe(
    scored.sort_values(["aio_likelihood", "confidence"], ascending=False),
    use_container_width=True,
)

st.download_button(
    "Download scored CSV",
    data=scored.to_csv(index=False).encode("utf-8"),
    file_name="gsc_aio_scores.csv",
    mime="text/csv",
)

with st.expander("How this works"):
    st.markdown(
        """
        This prototype estimates which queries or pages may be benefiting from AI Overviews using Search Console export data.

        Query mode boosts:
        - weaker rank but still getting clicks
        - CTR above expected
        - informational or comparative terms

        Page mode boosts:
        - weaker rank but still getting clicks
        - CTR above expected
        - URLs that look like informational content

        This is an estimate, not exact AI Overview attribution.
        """
    )
