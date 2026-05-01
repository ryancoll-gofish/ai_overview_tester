import pandas as pd
import streamlit as st
from urllib.parse import urlparse

st.set_page_config(page_title="GSC AI Overview Estimator", layout="wide")

INTENT_PATTERNS = {
    "navigational": [
        "login", "sign in", "homepage", "contact", "address", "phone", "hours",
    ],
    "commercial": [
        "best", "top", "vs", "versus", "compare", "comparison",
        "software", "tool", "tools", "platform", "service", "services",
        "provider", "solution", "solutions", "company", "companies",
        "product", "products",
    ],
    "informational": [
        "what is", "how to", "how do", "why", "when", "where", "who",
        "guide", "tips", "examples", "template", "meaning",
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
        parsed = urlparse(value)
        return parsed.path or "/"
    return value


def parse_ctr(value) -> float:
    if pd.isna(value):
        return 0.0
    value = str(value).strip().replace("%", "").replace(",", "")
    if value == "":
        return 0.0
    try:
        num = float(value)
        return num / 100.0 if num > 1 else num
    except ValueError:
        return 0.0


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


def detect_intent(query: str, brand_terms: list[str]) -> str:
    q = str(query or "").strip().lower()

    if any(term.lower() in q for term in brand_terms if term.strip()):
        return "branded"

    if any(p in q for p in INTENT_PATTERNS["navigational"]):
        return "navigational"

    if any(p in q for p in INTENT_PATTERNS["commercial"]):
        return "commercial"

    if any(p in q for p in INTENT_PATTERNS["informational"]):
        return "informational"

    return "other"


def is_branded(text_value: str, brand_terms: list[str]) -> bool:
    t = str(text_value or "").lower()
    return any(term.lower() in t for term in brand_terms if term.strip())


def looks_informational(page: str) -> bool:
    p = str(page or "").lower()
    patterns = [
        "/blog/",
        "/guide/",
        "/guides/",
        "/resources/",
        "/learn/",
        "/insights/",
        "/articles/",
        "/news/",
        "/compare/",
        "/comparison/",
        "/best-",
        "/what-is-",
        "/how-to-",
    ]
    return any(pattern in p for pattern in patterns)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def bucket_label(score: float) -> str:
    if score >= 0.75:
        return "Very likely"
    if score >= 0.50:
        return "Likely"
    if score >= 0.25:
        return "Possible"
    return "Unlikely"


def prepare_query_export(df: pd.DataFrame) -> pd.DataFrame:
    data = normalize_columns(df.copy())
    required = {"query", "clicks", "impressions", "ctr", "position"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {', '.join(sorted(missing))}. "
            "Expected: Top queries, Clicks, Impressions, CTR, Position"
        )

    data = data[["query", "clicks", "impressions", "ctr", "position"]].copy()
    data["query"] = data["query"].astype(str).str.strip()
    data["clicks"] = pd.to_numeric(data["clicks"], errors="coerce").fillna(0)
    data["impressions"] = pd.to_numeric(data["impressions"], errors="coerce").fillna(0)
    data["position"] = pd.to_numeric(data["position"], errors="coerce").fillna(0)
    data["ctr"] = data["ctr"].apply(parse_ctr)
    data = data[data["query"] != ""].copy()
    return data


def prepare_page_export(df: pd.DataFrame) -> pd.DataFrame:
    data = normalize_columns(df.copy())
    required = {"page", "clicks", "impressions", "ctr", "position"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {', '.join(sorted(missing))}. "
            "Expected: Top pages, Clicks, Impressions, CTR, Position"
        )

    data = data[["page", "clicks", "impressions", "ctr", "position"]].copy()
    data["page"] = data["page"].astype(str).str.strip().map(normalize_page)
    data["clicks"] = pd.to_numeric(data["clicks"], errors="coerce").fillna(0)
    data["impressions"] = pd.to_numeric(data["impressions"], errors="coerce").fillna(0)
    data["position"] = pd.to_numeric(data["position"], errors="coerce").fillna(0)
    data["ctr"] = data["ctr"].apply(parse_ctr)
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

    intent = detect_intent(query, brand_terms)
    branded = intent == "branded"
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
    elif intent == "commercial":
        score += 0.20
        reasons.append("Commercial / comparison query")
    elif intent == "navigational":
        score -= 0.30
        reasons.append("Navigational query")
    elif intent == "branded":
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
    if intent in {"informational", "commercial"}:
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
    informational = looks_informational(page)
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
    if informational:
        score += 0.15
        reasons.append("URL looks informational")
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
    if informational:
        confidence += 0.10
    if impressions < 20:
        confidence -= 0.15

    confidence = clamp01(confidence)

    return pd.Series({
        "informational_url": informational,
        "branded": branded,
        "expected_ctr": round(expected_ctr, 4),
        "aio_likelihood": round(score, 4),
        "confidence": round(confidence, 4),
        "why": "; ".join(reasons[:3]) if reasons else "No strong signals",
    })


st.title("GSC AI Overview Estimator")
st.caption("Upload a Search Console query export, pages export, or both.")

with st.sidebar:
    st.header("Settings")
    brand_input = st.text_input("Brand terms (comma-separated)", value="")
    min_impressions = st.slider("Minimum impressions", 0, 1000000, 50, 50)
    min_clicks = st.slider("Minimum clicks", 0, 100000, 0, 1)
    min_score = st.slider("Minimum AIO likelihood", 0.0, 1.0, 0.25, 0.05)
    brand_terms = [x.strip() for x in brand_input.split(",") if x.strip()]

query_file = st.file_uploader("Upload Query CSV", type=["csv"], key="query_csv")
page_file = st.file_uploader("Upload Pages CSV", type=["csv"], key="page_csv")

with st.expander("Expected CSV headers"):
    st.code("Query CSV: Top queries, Clicks, Impressions, CTR, Position")
    st.code("Pages CSV: Top pages, Clicks, Impressions, CTR, Position")

if query_file is None and page_file is None:
    st.info("Upload a query CSV, a pages CSV, or both.")
    st.stop()

query_scored = None
page_scored = None
query_all = None
page_all = None

if query_file is not None:
    try:
        query_raw = pd.read_csv(query_file)
        query_data = prepare_query_export(query_raw)
        query_all = pd.concat(
            [query_data, query_data.apply(score_query_row, axis=1, args=(brand_terms,))],
            axis=1,
        )
        query_all["likelihood_bucket"] = query_all["aio_likelihood"].apply(bucket_label)

        query_scored = query_all[
            (query_all["impressions"] >= min_impressions)
            & (query_all["clicks"] >= min_clicks)
            & (query_all["aio_likelihood"] >= min_score)
        ].sort_values(
            ["aio_likelihood", "confidence", "clicks", "impressions"],
            ascending=False,
        )
    except Exception as exc:
        st.error(f"Could not process query CSV: {exc}")

if page_file is not None:
    try:
        page_raw = pd.read_csv(page_file)
        page_data = prepare_page_export(page_raw)
        page_all = pd.concat(
            [page_data, page_data.apply(score_page_row, axis=1, args=(brand_terms,))],
            axis=1,
        )
        page_all["likelihood_bucket"] = page_all["aio_likelihood"].apply(bucket_label)

        page_scored = page_all[
            (page_all["impressions"] >= min_impressions)
            & (page_all["clicks"] >= min_clicks)
            & (page_all["aio_likelihood"] >= min_score)
        ].sort_values(
            ["aio_likelihood", "confidence", "clicks", "impressions"],
            ascending=False,
        )
    except Exception as exc:
        st.error(f"Could not process pages CSV: {exc}")

if query_all is None and page_all is None:
    st.stop()

summary_cols = st.columns(4)

query_rows = len(query_scored) if query_scored is not None else 0
page_rows = len(page_scored) if page_scored is not None else 0
query_avg = query_scored["aio_likelihood"].mean() if query_scored is not None and len(query_scored) else 0
page_avg = page_scored["aio_likelihood"].mean() if page_scored is not None and len(page_scored) else 0

summary_cols[0].metric("Flagged queries", f"{query_rows:,}")
summary_cols[1].metric("Flagged pages", f"{page_rows:,}")
summary_cols[2].metric("Avg query score", f"{query_avg:.1%}")
summary_cols[3].metric("Avg page score", f"{page_avg:.1%}")

tab_names = []
if query_scored is not None:
    tab_names.append("Queries")
if page_scored is not None:
    tab_names.append("Pages")

tabs = st.tabs(tab_names)

tab_idx = 0

if query_scored is not None:
    with tabs[tab_idx]:
        st.subheader("Top Queries Likely Getting AI Overviews")
        st.dataframe(
            query_scored[
                [
                    "query",
                    "clicks",
                    "impressions",
                    "ctr",
                    "position",
                    "intent",
                    "aio_likelihood",
                    "confidence",
                    "likelihood_bucket",
                    "why",
                ]
            ],
            use_container_width=True,
        )

        st.download_button(
            "Download scored query CSV",
            data=query_all.to_csv(index=False).encode("utf-8"),
            file_name="gsc_query_aio_scores.csv",
            mime="text/csv",
            key="download_queries",
        )
    tab_idx += 1

if page_scored is not None:
    with tabs[tab_idx]:
        st.subheader("Top Pages Likely Getting AI Overviews")
        st.dataframe(
            page_scored[
                [
                    "page",
                    "clicks",
                    "impressions",
                    "ctr",
                    "position",
                    "aio_likelihood",
                    "confidence",
                    "likelihood_bucket",
                    "why",
                ]
            ],
            use_container_width=True,
        )

        st.download_button(
            "Download scored pages CSV",
            data=page_all.to_csv(index=False).encode("utf-8"),
            file_name="gsc_pages_aio_scores.csv",
            mime="text/csv",
            key="download_pages",
        )

with st.expander("How this works"):
    st.markdown(
        """
        Query uploads are stronger because they include actual search intent.

        Query intent is classified as:
        - informational
        - commercial
        - navigational
        - branded

        Page uploads are more approximate and use:
        - weaker ranking but still getting clicks
        - CTR above expected
        - whether the URL looks informational

        This is an estimate, not exact AI Overview attribution.
        """
    )
