import pandas as pd
import streamlit as st
from urllib.parse import urlparse

st.set_page_config(page_title="GSC Pages AI Overview Estimator", layout="wide")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {}
    for col in df.columns:
        clean = str(col).strip().lower()
        if clean == "top pages":
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
    value = str(value).strip().replace("%", "")
    if value == "":
        return 0.0
    try:
        num = float(value)
        if num > 1:
            return num / 100.0
        return num
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


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def is_branded_page(page: str, brand_terms: list[str]) -> bool:
    p = str(page or "").lower()
    return any(term.lower() in p for term in brand_terms if term.strip())


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


def prepare_gsc_pages_export(df: pd.DataFrame) -> pd.DataFrame:
    data = normalize_columns(df.copy())

    required = {"page", "clicks", "impressions", "ctr", "position"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {', '.join(sorted(missing))}. "
            "Expected headers: Top pages, Clicks, Impressions, CTR, Position"
        )

    data = data[["page", "clicks", "impressions", "ctr", "position"]].copy()
    data["page"] = data["page"].astype(str).str.strip().map(normalize_page)
    data["clicks"] = pd.to_numeric(data["clicks"], errors="coerce").fillna(0)
    data["impressions"] = pd.to_numeric(data["impressions"], errors="coerce").fillna(0)
    data["position"] = pd.to_numeric(data["position"], errors="coerce").fillna(0)
    data["ctr"] = data["ctr"].apply(parse_ctr)

    data = data[data["page"] != ""].copy()
    return data


def score_page_row(row: pd.Series, brand_terms: list[str]) -> pd.Series:
    score = 0.0
    reasons = []

    page = str(row.get("page", ""))
    position = float(row.get("position", 0) or 0)
    clicks = float(row.get("clicks", 0) or 0)
    impressions = float(row.get("impressions", 0) or 0)
    ctr = float(row.get("ctr", 0) or 0)

    branded = is_branded_page(page, brand_terms)
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
        "expected_ctr": round(expected_ctr, 4),
        "aio_likelihood": round(score, 4),
        "confidence": round(confidence, 4),
        "informational_url": informational,
        "branded": branded,
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


st.title("GSC Pages AI Overview Estimator")
st.caption("Upload a Search Console pages export and estimate which pages are most likely benefiting from AI Overviews.")

with st.sidebar:
    st.header("Settings")
    brand_input = st.text_input("Brand terms (comma-separated)", value="bandwidth")
    min_impressions = st.slider("Minimum impressions", 0, 1000000, 50, 50)
    min_clicks = st.slider("Minimum clicks", 0, 100000, 0, 1)
    min_score = st.slider("Minimum AIO likelihood", 0.0, 1.0, 0.25, 0.05)
    brand_terms = [x.strip() for x in brand_input.split(",") if x.strip()]

uploaded_file = st.file_uploader("Upload Search Console Pages CSV", type=["csv"])

with st.expander("Expected CSV headers"):
    st.code("Top pages, Clicks, Impressions, CTR, Position")

if uploaded_file is None:
    st.info("Upload your Search Console pages CSV to begin.")
    st.stop()

try:
    raw = pd.read_csv(uploaded_file)
    pages = prepare_gsc_pages_export(raw)
except Exception as exc:
    st.error(f"Could not read file: {exc}")
    st.stop()

scored = pd.concat([pages, pages.apply(score_page_row, axis=1, args=(brand_terms,))], axis=1)
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
c1.metric("Pages analyzed", f"{len(scored):,}")
c2.metric("Pages flagged", f"{len(filtered):,}")
c3.metric("Avg AIO likelihood", f"{filtered['aio_likelihood'].mean():.1%}" if len(filtered) else "0.0%")
c4.metric("Avg confidence", f"{filtered['confidence'].mean():.1%}" if len(filtered) else "0.0%")

st.subheader("Top Pages Likely Getting AI Overviews")
st.dataframe(
    filtered[[
        "page",
        "clicks",
        "impressions",
        "ctr",
        "position",
        "aio_likelihood",
        "confidence",
        "likelihood_bucket",
        "why",
    ]],
    use_container_width=True,
)

st.subheader("Raw Scored Data")
st.dataframe(
    scored.sort_values(["aio_likelihood", "confidence"], ascending=False),
    use_container_width=True,
)

st.download_button(
    "Download scored pages CSV",
    data=scored.to_csv(index=False).encode("utf-8"),
    file_name="gsc_pages_aio_scores.csv",
    mime="text/csv",
)

with st.expander("How this works"):
    st.markdown(
        """
        This prototype estimates which pages may be benefiting from AI Overviews using only Search Console page export data.

        It boosts scores when a page:
        - ranks outside the top positions but still gets clicks
        - has a CTR above expected for its position
        - looks like informational content based on the URL

        It lowers scores for pages that look heavily branded.

        This is an estimate, not exact AI Overview attribution.
        """
    )
