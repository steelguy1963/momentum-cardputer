from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf


# ==========================================
# 1. Configuration
# ==========================================

N_TOP_DISPLAY = 4

HISTORY_YEARS = 3
MOMENTUM_LAG_MONTHS = 1

SEOUL = ZoneInfo("Asia/Seoul")

TICKERS = [
    "VOO",
    "QQQM",
    "VTWO",
    "102110.KS",
    "1306.T",
    "FEZ",
    "372330.KS",
    "INDA",
    "SCHP",
    "GLD",
]

# Cardputer에서 표시할 짧은 이름
DISPLAY_NAMES = {
    "VOO": "VOO",
    "QQQM": "QQQM",
    "VTWO": "R2000",
    "102110.KS": "K200",
    "1306.T": "TOPIX",
    "FEZ": "EU50",
    "372330.KS": "HSTECH",
    "INDA": "INDIA",
    "SCHP": "TIPS",
    "GLD": "GOLD",
}

OUTPUT_PATH = Path("data/latest.json")


# ==========================================
# 2. 평가 기준일
# ==========================================

def get_evaluation_date() -> pd.Timestamp:
    """
    GitHub Actions가 실행되는 시점의
    한국 날짜를 평가 기준일로 사용한다.
    """
    return pd.Timestamp(datetime.now(SEOUL).date())


# ==========================================
# 3. 시장 데이터 수집
# ==========================================

def fetch_market_data(
    tickers: list[str],
    years: int,
) -> pd.DataFrame:
    """
    Yahoo Finance에서 일간 Close 가격을 가져온다.

    기존 코드와 달리 yfinance.info()를 사용하지 않는다.
    회사명 조회는 필요 없으며 Cardputer 표시명은
    DISPLAY_NAMES에서 직접 관리한다.
    """

    evaluation_date = get_evaluation_date()

    start_date = (
        evaluation_date
        - pd.DateOffset(years=years)
    ).replace(day=1)

    # yfinance의 end 날짜는 exclusive이므로
    # 평가일 다음 날을 지정한다.
    end_date = evaluation_date + pd.Timedelta(days=1)

    print(
        f"[INFO] Data period: "
        f"{start_date.date()} ~ {evaluation_date.date()}"
    )

    print(
        f"[INFO] Loading {len(tickers)} tickers..."
    )

    raw = yf.download(
        tickers,
        start=start_date.date().isoformat(),
        end=end_date.date().isoformat(),
        progress=False,
        auto_adjust=False,
        threads=False,
    )

    if raw.empty:
        raise RuntimeError(
            "Yahoo Finance returned no price data."
        )

    # 여러 ticker를 조회하면 일반적으로 MultiIndex
    if isinstance(raw.columns, pd.MultiIndex):

        if "Close" not in raw.columns.get_level_values(0):
            raise RuntimeError(
                "Close prices are not available."
            )

        df_close = raw["Close"].copy()

    else:

        if "Close" not in raw.columns:
            raise RuntimeError(
                "Close prices are not available."
            )

        df_close = raw[["Close"]].copy()

        if len(tickers) == 1:
            df_close.columns = tickers

    # 요청한 ticker 중 실제로 받은 것만 사용
    available_tickers = [
        ticker
        for ticker in tickers
        if ticker in df_close.columns
    ]

    df_close = df_close[available_tickers]

    # timezone 제거
    df_close.index = (
        pd.to_datetime(df_close.index)
        .tz_localize(None)
    )

    df_close = (
        df_close
        .sort_index()
        .ffill()
    )

    print(
        f"[INFO] Loaded tickers: "
        f"{len(available_tickers)}/{len(tickers)}"
    )

    missing = [
        ticker
        for ticker in tickers
        if ticker not in available_tickers
    ]

    if missing:
        print(
            "[WARN] Missing tickers: "
            + ", ".join(missing)
        )

    if df_close.empty:
        raise RuntimeError(
            "No requested ticker data is available."
        )

    return df_close


# ==========================================
# 4. 주간 데이터
# ==========================================

def prepare_weekly_data(
    df_daily: pd.DataFrame,
) -> pd.DataFrame:
    """
    기존 코드와 동일하게 금요일 기준으로
    주간 마지막 가격을 사용한다.
    """

    return (
        df_daily
        .resample("W-FRI")
        .last()
        .dropna(how="all")
    )


# ==========================================
# 5. 기존 모멘텀 계산 로직
# ==========================================

def calc_metrics_for_date(
    df_weekly: pd.DataFrame,
    eval_date: pd.Timestamp,
) -> pd.DataFrame:

    """
    기존 코드의 핵심 계산을 그대로 유지한다.

    12개월:
        raw return / weekly std

    6개월:
        raw return / weekly std

    최종 score:
        (12M score + 6M score) / 2
    """

    d_end = (
        eval_date
        - pd.DateOffset(
            months=MOMENTUM_LAG_MONTHS
        )
    )

    d_12m = (
        d_end
        - pd.DateOffset(years=1)
    )

    d_6m = (
        d_end
        - pd.DateOffset(months=6)
    )

    df_12m = (
        df_weekly
        .loc[d_12m:d_end]
        .dropna(axis=1)
    )

    df_6m = (
        df_weekly
        .loc[d_6m:d_end]
        .dropna(axis=1)
    )

    if len(df_12m) < 4 or len(df_6m) < 4:
        return pd.DataFrame()

    # 두 기간 모두 존재하는 ticker만 사용
    common_tickers = (
        df_12m.columns
        .intersection(df_6m.columns)
    )

    if len(common_tickers) == 0:
        return pd.DataFrame()

    df_12m = df_12m[common_tickers]
    df_6m = df_6m[common_tickers]

    # --------------------------------------
    # 12개월 수익률
    # --------------------------------------

    mom_12 = (
        df_12m.iloc[-1]
        / df_12m.iloc[0]
    ) - 1

    # --------------------------------------
    # 6개월 수익률
    # --------------------------------------

    mom_6 = (
        df_6m.iloc[-1]
        / df_6m.iloc[0]
    ) - 1

    # --------------------------------------
    # 주간 변동성
    # --------------------------------------

    std_12 = (
        df_12m
        .pct_change()
        .std()
    )

    std_6 = (
        df_6m
        .pct_change()
        .std()
    )

    # 0으로 나누는 문제 방지
    std_12 = std_12.mask(std_12 == 0)
    std_6 = std_6.mask(std_6 == 0)

    # --------------------------------------
    # 변동성 조정 모멘텀
    # --------------------------------------

    vol_adj_mom_12 = (
        mom_12 / std_12
    )

    vol_adj_mom_6 = (
        mom_6 / std_6
    )

    # --------------------------------------
    # 최종 score
    # --------------------------------------

    final_score = (
        vol_adj_mom_12
        + vol_adj_mom_6
    ) / 2

    result = pd.DataFrame(
        {
            "ticker": final_score.index,
            "score": final_score.values,
        }
    )

    result = result.dropna(
        subset=["score"]
    )

    return result


# ==========================================
# 6. 특정 날짜의 전체 순위 계산
# ==========================================

def rank_for_date(
    df_weekly: pd.DataFrame,
    eval_date: pd.Timestamp,
) -> pd.DataFrame:

    metrics = calc_metrics_for_date(
        df_weekly,
        eval_date,
    )

    if metrics.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "score",
                "rank",
            ]
        )

    # score가 높은 순
    # score가 같으면 ticker 알파벳 순
    ranked = (
        metrics
        .sort_values(
            ["score", "ticker"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )

    ranked["rank"] = (
        ranked.index + 1
    )

    return ranked


# ==========================================
# 7. 현재 Rank + 지난 달 Rank 결합
# ==========================================

def build_latest_result(
    df_weekly: pd.DataFrame,
    evaluation_date: pd.Timestamp,
) -> dict:

    # --------------------------------------
    # 현재 평가
    # --------------------------------------

    current_rank = rank_for_date(
        df_weekly,
        evaluation_date,
    )

    # --------------------------------------
    # 지난 달 평가
    # --------------------------------------

    previous_evaluation_date = (
        evaluation_date
        - pd.DateOffset(months=1)
    )

    previous_rank = rank_for_date(
        df_weekly,
        previous_evaluation_date,
    )

    if current_rank.empty:
        raise RuntimeError(
            "Could not calculate current ranking."
        )

    # ticker -> 지난 달 rank
    previous_rank_map = dict(
        zip(
            previous_rank["ticker"],
            previous_rank["rank"],
        )
    )

    # --------------------------------------
    # 현재 Top 4만 추출
    # --------------------------------------

    top = current_rank.head(
        N_TOP_DISPLAY
    )

    items = []

    for _, row in top.iterrows():

        ticker = row["ticker"]

        previous_rank_value = (
            previous_rank_map.get(ticker)
        )

        items.append(
            {
                "rank": int(row["rank"]),
                "ticker": ticker,
                "name": DISPLAY_NAMES.get(
                    ticker,
                    ticker,
                ),
                "previous_rank": (
                    int(previous_rank_value)
                    if previous_rank_value is not None
                    else None
                ),
            }
        )

    print("[INFO] Top 4:")

    for item in items:

        print(
            f"  {item['rank']} "
            f"{item['ticker']} "
            f"previous="
            f"{item['previous_rank']}"
        )

    # --------------------------------------
    # Cardputer용 JSON
    # --------------------------------------

    return {
        "generated_at": (
            datetime.now(SEOUL)
            .isoformat(timespec="seconds")
        ),

        "evaluation_date": (
            evaluation_date
            .date()
            .isoformat()
        ),

        "previous_evaluation_date": (
            previous_evaluation_date
            .date()
            .isoformat()
        ),

        "items": items,
    }


# ==========================================
# 8. JSON 저장
# ==========================================

def save_result(
    result: dict,
    output_path: Path,
) -> None:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 임시 파일에 먼저 작성
    temp_path = output_path.with_suffix(
        ".tmp"
    )

    with temp_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2,
        )

        file.write("\n")

    # 기존 파일을 새로운 파일로 교체
    temp_path.replace(
        output_path
    )

    print(
        f"[INFO] Result saved: "
        f"{output_path}"
    )


# ==========================================
# 9. Main
# ==========================================

def main() -> None:

    evaluation_date = (
        get_evaluation_date()
    )

    df_daily = fetch_market_data(
        TICKERS,
        HISTORY_YEARS,
    )

    df_weekly = prepare_weekly_data(
        df_daily
    )

    result = build_latest_result(
        df_weekly,
        evaluation_date,
    )

    save_result(
        result,
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()