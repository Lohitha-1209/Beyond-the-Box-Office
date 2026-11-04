from __future__ import annotations

import os
from pathlib import Path
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "movie_project_output" / "movies_cleaned.csv"
RAW_DATA_PATH = BASE_DIR / "movies_raw.csv"
MIN_VALID_AMOUNT = 1_000
ACCENT = "#2E5E8C"
ACCENT2 = "#E07A3F"
GREY = "#8A8F98"
TEMPLATE = "plotly_white"

LANGUAGE_NAMES = {
    "ar": "Arabic", "bn": "Bengali", "cn": "Cantonese", "da": "Danish",
    "de": "German", "en": "English", "es": "Spanish", "fr": "French",
    "hi": "Hindi", "id": "Indonesian", "it": "Italian", "ja": "Japanese",
    "kn": "Kannada", "ko": "Korean", "ml": "Malayalam", "nl": "Dutch",
    "pl": "Polish", "pt": "Portuguese", "ru": "Russian", "sv": "Swedish",
    "ta": "Tamil", "te": "Telugu", "th": "Thai", "tr": "Turkish",
    "zh": "Chinese",
}


def load_movies() -> pd.DataFrame:
    if not RAW_DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Raw movie data was not found at {RAW_DATA_PATH}. "
            "Keep movies_raw.csv beside app.py in the project."
        )

    raw = pd.read_csv(RAW_DATA_PATH, low_memory=False)
    if DATA_PATH.is_file():
        movies = pd.read_csv(DATA_PATH, low_memory=False)
    else:
        movies = raw.copy()
        release_dates = pd.to_datetime(movies["release_date"], errors="coerce")
        movies["release_year"] = release_dates.dt.year
        movies["release_month"] = release_dates.dt.month
        movies["budget_analysis"] = movies["budget"]
        movies["revenue_analysis"] = movies["revenue"]

    required_columns = {
        "movie_id", "title", "release_year", "genres", "budget_analysis",
        "revenue_analysis", "runtime", "vote_average", "vote_count",
        "popularity", "original_language", "release_month",
    }
    missing = sorted(required_columns - set(movies.columns))
    if missing:
        raise ValueError(
            "The cleaned movie file is missing required columns: "
            + ", ".join(missing)
        )
    raw_required = {
        "movie_id", "genres", "release_date", "budget", "revenue", "runtime",
        "title", "vote_average", "vote_count", "popularity",
        "original_language",
    }
    raw_missing = sorted(raw_required - set(raw.columns))
    if raw_missing:
        raise ValueError(
            "The raw movie file is missing required columns: "
            + ", ".join(raw_missing)
        )
    if DATA_PATH.is_file() and (
        len(raw) != len(movies)
        or not raw["movie_id"].equals(movies["movie_id"])
    ):
        raise ValueError(
            "movies_raw.csv and movies_cleaned.csv must contain the same "
            "movies in the same order."
        )

    for column in (
        "release_year", "release_month", "budget_analysis", "revenue_analysis",
        "vote_average", "vote_count", "popularity",
    ):
        movies[column] = pd.to_numeric(movies[column], errors="coerce")

    movies["budget_eda"] = movies["budget_analysis"].where(
        movies["budget_analysis"] >= MIN_VALID_AMOUNT
    )
    movies["revenue_eda"] = movies["revenue_analysis"].where(
        movies["revenue_analysis"] >= MIN_VALID_AMOUNT
    )
    movies["runtime_eda"] = pd.to_numeric(
        movies.get("runtime"), errors="coerce"
    )
    movies.loc[movies["runtime_eda"] <= 0, "runtime_eda"] = np.nan
    observed_genres = movies["genres"].where(~raw["genres"].isna())
    movies["genre_list"] = observed_genres.apply(split_genres)
    movies["year"] = movies["release_year"].round().astype("Int64")
    movies["language"] = movies["original_language"].fillna("Unknown").astype(str)
    movies["language_name"] = movies["language"].map(LANGUAGE_NAMES).fillna(
        movies["language"]
    )
    if movies["year"].notna().sum() == 0:
        raise ValueError("The cleaned movie file has no usable release years.")
    return movies


def split_genres(value: object) -> list[str]:
    if pd.isna(value):
        return []
    parts = re.split(r"\s*[|,]\s*", str(value).strip())
    return list(dict.fromkeys(part for part in parts if part and part.lower() != "nan"))


MOVIES = load_movies()
YEAR_MIN = int(MOVIES["year"].min())
YEAR_MAX = int(MOVIES["year"].max())
YEAR_MARKS = {
    year: str(year)
    for year in range((YEAR_MIN // 5) * 5, YEAR_MAX + 1, 5)
}
if YEAR_MAX not in YEAR_MARKS:
    YEAR_MARKS[YEAR_MAX] = str(YEAR_MAX)


def empty_figure(title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text="No records match the selected filters.",
        x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
    )
    fig.update_layout(template=TEMPLATE, title=title, height=430)
    return fig


def style_figure(fig: go.Figure, title: str, height: int = 430) -> go.Figure:
    fig.update_layout(
        template=TEMPLATE,
        title=dict(text=title, x=0.02, xanchor="left"),
        height=height,
        margin=dict(t=70, l=55, r=25, b=55),
        legend_title_text="",
        hovermode="closest",
    )
    return fig


def filtered_movies(
    year_range: list[int] | None,
    genres: list[str] | None,
    languages: list[str] | None,
) -> pd.DataFrame:
    data = MOVIES
    if year_range and len(year_range) == 2:
        data = data[data["year"].between(year_range[0], year_range[1])]
    if genres:
        selected = set(genres)
        data = data[data["genre_list"].map(lambda values: bool(selected.intersection(values)))]
    if languages:
        data = data[data["language"].isin(languages)]
    return data.copy()


def make_figures(data: pd.DataFrame) -> list[go.Figure]:
    if data.empty:
        return [empty_figure(f"Plot {index}: EDA") for index in range(1, 13)]

    # Plot 1: Movie release trends
    years = data.dropna(subset=["year"]).groupby("year", as_index=False).size()
    fig1 = px.line(
        years, x="year", y="size", markers=True,
        labels={"year": "Release year", "size": "Movies"},
        title="1. Movie release trends",
    )
    fig1.update_traces(line_color=ACCENT)

    # Plot 2: Genre distribution, counting each movie once per genre.
    genre_rows = data[["genre_list"]].explode("genre_list").dropna()
    genre_counts = (
        genre_rows["genre_list"].value_counts().rename_axis("genre")
        .reset_index(name="movies").sort_values("movies")
    )
    fig2 = px.bar(
        genre_counts, x="movies", y="genre", orientation="h",
        labels={"movies": "Movies", "genre": "Genre"},
        title="2. Genre distribution",
        color_discrete_sequence=[ACCENT],
    )

    # Plot 3: Budget versus revenue for records with reported, plausible values.
    financial = data.dropna(subset=["budget_eda", "revenue_eda"]).copy()
    if financial.empty:
        fig3 = empty_figure("3. Budget vs revenue")
    else:
        financial["outcome"] = np.where(
            financial["revenue_eda"] >= financial["budget_eda"],
            "Revenue ≥ budget", "Revenue < budget",
        )
        fig3 = px.scatter(
            financial, x="budget_eda", y="revenue_eda", color="outcome",
            hover_name="title", log_x=True, log_y=True, opacity=0.55,
            labels={
                "budget_eda": "Budget (USD, log scale)",
                "revenue_eda": "Revenue (USD, log scale)",
                "outcome": "Outcome",
            },
            color_discrete_map={
                "Revenue ≥ budget": ACCENT, "Revenue < budget": ACCENT2,
            },
            title=f"3. Budget vs revenue ({len(financial):,} valid records)",
        )
        lower = min(financial["budget_eda"].min(), financial["revenue_eda"].min())
        upper = max(financial["budget_eda"].max(), financial["revenue_eda"].max())
        fig3.add_trace(go.Scatter(
            x=[lower, upper], y=[lower, upper], mode="lines",
            name="Break-even", line=dict(color=GREY, dash="dash"),
            hoverinfo="skip",
        ))

    # Plot 4: Ratings distribution.
    ratings = data.dropna(subset=["vote_average"])
    fig4 = px.histogram(
        ratings, x="vote_average", nbins=30,
        labels={"vote_average": "Average rating (0–10)", "count": "Movies"},
        title="4. Ratings distribution",
        color_discrete_sequence=[ACCENT],
    )

    # Plot 5: Revenue distributions by genre.
    by_genre = (
        financial[["title", "revenue_eda", "genre_list"]]
        .explode("genre_list").dropna(subset=["genre_list"])
    )
    counts = by_genre["genre_list"].value_counts()
    by_genre = by_genre[by_genre["genre_list"].isin(counts[counts >= 30].index)]
    if by_genre.empty:
        fig5 = empty_figure("5. Revenue by genre")
    else:
        fig5 = px.box(
            by_genre, x="revenue_eda", y="genre_list", points="outliers",
            log_x=True, labels={
                "revenue_eda": "Revenue (USD, log scale)",
                "genre_list": "Genre",
            },
            title="5. Revenue distribution by genre (valid financial records)",
            color_discrete_sequence=[ACCENT],
        )

    # Plot 6: Pairwise-complete Spearman correlations.
    corr_columns = {
        "budget_eda": "Budget", "revenue_eda": "Revenue",
        "runtime_eda": "Runtime", "vote_average": "Rating",
        "vote_count": "Vote count", "popularity": "Popularity",
    }
    corr_data = data[list(corr_columns)].rename(columns=corr_columns)
    corr = corr_data.corr(method="spearman")
    fig6 = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1, aspect="auto",
        title="6. Correlation heatmap (Spearman)",
    )

    # Plot 7: Ratings versus popularity with median ratings by popularity decile.
    popularity = data.dropna(subset=["popularity", "vote_average"]).copy()
    popularity = popularity[popularity["popularity"] > 0]
    if popularity.empty:
        fig7 = empty_figure("7. Ratings vs popularity")
    else:
        fig7 = px.scatter(
            popularity, x="popularity", y="vote_average",
            hover_name="title", log_x=True, opacity=0.3,
            labels={
                "popularity": "Popularity (log scale)",
                "vote_average": "Average rating (0–10)",
            },
            title="7. Ratings vs popularity",
            color_discrete_sequence=[ACCENT],
        )
        if popularity["popularity"].nunique() < 2:
            deciles = popularity.groupby("popularity", as_index=False).agg(
                rating=("vote_average", "median")
            )
            deciles["popularity"] = popularity["popularity"].median()
        else:
            popularity["popularity_decile"] = pd.qcut(
                popularity["popularity"], 10, duplicates="drop"
            )
            deciles = popularity.groupby(
                "popularity_decile", observed=True
            ).agg(
                popularity=("popularity", "median"),
                rating=("vote_average", "median"),
            )
        fig7.add_trace(go.Scatter(
            x=deciles["popularity"], y=deciles["rating"],
            mode="lines+markers", name="Median rating by decile",
            line=dict(color=ACCENT2, width=3),
        ))

    # Plot 8: Original language treemap, with smaller languages grouped together.
    language_counts = data["language_name"].value_counts()
    if language_counts.empty:
        fig8 = empty_figure("8. Movies by original language")
    else:
        top_languages = language_counts.head(10)
        language_table = pd.DataFrame({
            "language": top_languages.index.tolist(),
            "movies": top_languages.values,
        })
        other_count = int(language_counts.iloc[10:].sum())
        if other_count:
            language_table.loc[len(language_table)] = ["All other languages", other_count]
        fig8 = px.treemap(
            language_table, path=["language"], values="movies",
            color="movies", color_continuous_scale="Blues",
            title="8. Movies by original language",
        )

    # Plot 9: Median ROI per genre for eligible financial records.
    roi = financial[["title", "budget_eda", "revenue_eda", "genre_list"]].explode(
        "genre_list"
    ).dropna(subset=["genre_list"]).copy()
    if roi.empty:
        fig9 = empty_figure("9. Median ROI by genre")
    else:
        roi["roi_percent"] = (
            roi["revenue_eda"] / roi["budget_eda"] - 1
        ) * 100
        roi_counts = roi["genre_list"].value_counts()
        roi = roi[roi["genre_list"].isin(roi_counts[roi_counts >= 30].index)]
        roi_summary = (
            roi.groupby("genre_list", as_index=False)
            .agg(median_roi=("roi_percent", "median"))
            .sort_values("median_roi")
        )
        fig9 = px.bar(
            roi_summary, x="median_roi", y="genre_list", orientation="h",
            color="median_roi", color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
            labels={
                "median_roi": "Median ROI (%)",
                "genre_list": "Genre",
            },
            title="9. Median return on investment by genre",
        )
        fig9.add_vline(x=0, line_color=GREY, line_dash="dash")

    # Plot 10: Percentage earning back budget by budget quartile.
    if financial.empty:
        fig10 = empty_figure("10. Share earning back budget by budget tier")
    else:
        tiers = financial[["budget_eda", "revenue_eda"]].copy()
        if tiers["budget_eda"].nunique() < 2:
            tiers["budget_tier"] = "All budget records"
        else:
            tiers["budget_tier"] = pd.qcut(
                tiers["budget_eda"],
                min(4, tiers["budget_eda"].nunique()),
                duplicates="drop",
            )
        tiers["earned_back"] = tiers["revenue_eda"] >= tiers["budget_eda"]
        tier_summary = (
            tiers.groupby("budget_tier", observed=True)
            .agg(movies=("earned_back", "size"),
                 pct_back=("earned_back", lambda values: 100 * values.mean()))
            .reset_index()
        )
        tier_summary["budget_tier"] = tier_summary["budget_tier"].astype(str)
        fig10 = px.bar(
            tier_summary, x="budget_tier", y="pct_back",
            hover_data={"movies": True, "pct_back": ":.1f"},
            labels={
                "budget_tier": "Budget tier (quartiles)",
                "pct_back": "Movies earning back budget (%)",
            },
            title="10. Share of movies earning back their budget, by budget tier",
            color_discrete_sequence=[ACCENT],
        )
        fig10.update_yaxes(range=[0, 100])

    # Plot 11: Release year by month heatmap.
    seasonal = data.dropna(subset=["year", "release_month"]).copy()
    seasonal = seasonal[seasonal["release_month"].between(1, 12)]
    if seasonal.empty:
        fig11 = empty_figure("11. Release seasonality")
    else:
        seasonal_counts = seasonal.pivot_table(
            index="year", columns="release_month", values="title",
            aggfunc="count", fill_value=0,
        ).reindex(columns=range(1, 13), fill_value=0)
        fig11 = px.imshow(
            seasonal_counts, aspect="auto", color_continuous_scale="Blues",
            labels={"x": "Release month", "y": "Release year", "color": "Movies"},
            title="11. Release seasonality (year × month)",
        )

    # Plot 12: Highest-grossing eligible movies, plotted as budget vs revenue.
    if financial.empty:
        fig12 = empty_figure("12. Top 10 highest-grossing movies")
    else:
        top_ten = financial.nlargest(10, "revenue_eda").copy()
        fig12 = px.scatter(
            top_ten, x="budget_eda", y="revenue_eda", text="title",
            hover_name="title", log_x=True, log_y=True,
            labels={
                "budget_eda": "Budget (USD, log scale)",
                "revenue_eda": "Revenue (USD, log scale)",
            },
            title="12. Top 10 highest-grossing movies (budget vs revenue)",
            color_discrete_sequence=[ACCENT2],
        )
        fig12.update_traces(textposition="top center")

    figures = [
        fig1, fig2, fig3, fig4, fig5, fig6,
        fig7, fig8, fig9, fig10, fig11, fig12,
    ]
    for index, fig in enumerate(figures, start=1):
        if index not in (3, 5, 6, 7, 8, 9, 10, 11, 12) or fig.data:
            style_figure(fig, fig.layout.title.text or f"Plot {index}")
    return figures


app = Dash(__name__, title="Movie Analysis | EDA Dashboard")
server = app.server

CARD_STYLE = {
    "backgroundColor": "white",
    "borderRadius": "12px",
    "padding": "14px",
    "boxShadow": "0 2px 12px rgba(25, 45, 70, 0.08)",
    "minWidth": "0",
}

app.layout = html.Div(
    [
        html.Div(
            [
                html.H1("Movie Analysis — Exploratory Data Analysis"),
                html.P(
                    "Explore the project’s 12 EDA plots. Filters update every "
                    "chart; financial plots use reported values of at least $1,000."
                ),
            ],
            style={"marginBottom": "20px"},
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Release year"),
                        dcc.RangeSlider(
                            id="year-filter",
                            min=YEAR_MIN,
                            max=YEAR_MAX,
                            value=[YEAR_MIN, YEAR_MAX],
                            marks=YEAR_MARKS,
                            step=1,
                            allowCross=False,
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                    ],
                    style={"gridColumn": "1 / -1"},
                ),
                html.Div(
                    [
                        html.Label("Genre"),
                        dcc.Dropdown(
                            id="genre-filter",
                            options=[
                                {"label": genre, "value": genre}
                                for genre in sorted({
                                    genre for genres in MOVIES["genre_list"]
                                    for genre in genres
                                })
                            ],
                            multi=True,
                            placeholder="All genres",
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Label("Original language"),
                        dcc.Dropdown(
                            id="language-filter",
                            options=[
                                {"label": name, "value": code}
                                for code, name in sorted(
                                    MOVIES[["language", "language_name"]]
                                    .drop_duplicates()
                                    .itertuples(index=False, name=None),
                                    key=lambda item: item[1].casefold(),
                                )
                            ],
                            multi=True,
                            placeholder="All languages",
                        ),
                    ]
                ),
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(2, minmax(220px, 1fr))",
                "gap": "20px",
                "padding": "18px",
                "backgroundColor": "white",
                "borderRadius": "12px",
                "boxShadow": "0 2px 12px rgba(25, 45, 70, 0.08)",
                "marginBottom": "14px",
            },
        ),
        html.Div(id="filter-summary", style={"margin": "8px 2px 18px"}),
        html.Div(
            [
                dcc.Graph(id=f"plot-{index}", config={"displaylogo": False})
                for index in range(1, 13)
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(auto-fit, minmax(420px, 1fr))",
                "gap": "14px",
            },
        ),
        html.Footer(
            "Source: movie_project_output/movies_cleaned.csv | "
            "Financial records below $1,000 are excluded from financial plots.",
            style={"color": "#586574", "fontSize": "13px", "marginTop": "18px"},
        ),
    ],
    style={
        "maxWidth": "1500px",
        "margin": "0 auto",
        "padding": "24px",
        "fontFamily": "Arial, sans-serif",
        "color": "#203040",
        "backgroundColor": "#F4F7FA",
        "minHeight": "100vh",
    },
)


@app.callback(
    [Output(f"plot-{index}", "figure") for index in range(1, 13)]
    + [Output("filter-summary", "children")],
    Input("year-filter", "value"),
    Input("genre-filter", "value"),
    Input("language-filter", "value"),
)
def update_dashboard(
    year_range: list[int] | None,
    genres: list[str] | None,
    languages: list[str] | None,
) -> tuple[go.Figure, ...]:
    data = filtered_movies(year_range, genres, languages)
    figures = make_figures(data)
    return (*figures, f"Showing {len(data):,} movies after filters.")


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8050")),
        debug=False,
    )
