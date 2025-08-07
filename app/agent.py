
import re
import json
import base64
import matplotlib.pyplot as plt
import duckdb
import pandas as pd
from bs4 import BeautifulSoup
import requests
from io import BytesIO
import numpy as np


async def process_question_file(text: str, attachments: list):
    try:
        if "wikipedia.org" in text.lower():
            return await handle_wikipedia_task(text)
        elif "indian high court" in text.lower():
            return await handle_indian_court_task(text, attachments)
        else:
            return {"error": "Unknown task. Please mention 'Wikipedia' or 'Indian High Court' in the input."}
    except Exception as e:
        return {"error": f"Unexpected error occurred: {str(e)}"}


async def handle_wikipedia_task(text: str):
    match = re.search(r"https://en\.wikipedia\.org/[^\s]+", text)
    if not match:
        return {"error": "No Wikipedia URL found in input."}

    url = match.group(0)
    response = requests.get(url)
    if response.status_code != 200:
        return {"error": f"Failed to fetch URL: {url}"}

    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table", {"class": "wikitable"})
    if not table:
        return {"error": "No 'wikitable' found on page."}

    df = pd.read_html(str(table))[0]
    df.columns = [str(col).lower().strip() for col in df.columns]

    for col in ['worldwide gross', 'year', 'rank', 'peak']:
        if col not in df.columns:
            return {"error": f"Missing expected column: {col}"}

    df['worldwide gross'] = df['worldwide gross'].replace('[\$,]', '', regex=True).astype(float)
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['rank'] = pd.to_numeric(df['rank'], errors='coerce')
    df['peak'] = pd.to_numeric(df['peak'], errors='coerce')

    ans1 = int(len(df[(df['worldwide gross'] >= 2_000_000_000) & (df['year'] < 2000)]))
    ans2_row = df[df['worldwide gross'] > 1_500_000_000].sort_values("year").head(1)
    ans2 = ans2_row.iloc[0]["title"] if not ans2_row.empty else "No result"
    ans3 = df[['rank', 'peak']].dropna().corr().loc["rank", "peak"]

    fig, ax = plt.subplots()
    ax.scatter(df['rank'], df['peak'])
    m, b = np.polyfit(df['rank'].dropna(), df['peak'].dropna(), 1)
    ax.plot(df['rank'], m * df['rank'] + b, 'r--')
    ax.set_xlabel("Rank")
    ax.set_ylabel("Peak")
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img_data = base64.b64encode(buf.read()).decode("utf-8")
    img_uri = f"data:image/png;base64,{img_data}"

    return {
        "How many movies grossed over $2B before 2000?": ans1,
        "Which is the earliest movie that grossed over $1.5B?": ans2,
        "Correlation between Rank and Peak": round(ans3, 6),
        "Scatterplot (base64 PNG)": img_uri
    }


async def handle_indian_court_task(text: str, attachments: list):
    parquet_file = next((f for f in attachments if f.filename.endswith('.parquet')), None)
    if not parquet_file:
        return {"error": "Missing .parquet attachment"}

    df = pd.read_parquet(BytesIO(await parquet_file.read()))

    if not {'decision_date', 'date_of_registration', 'court'}.issubset(df.columns):
        return {"error": "Missing one or more required columns in Parquet file."}

    df['delay'] = pd.to_datetime(df['decision_date']) - pd.to_datetime(df['date_of_registration'])
    df['delay_days'] = df['delay'].dt.days
    df['year'] = pd.to_datetime(df['decision_date']).dt.year

    most_cases = df[df['year'].between(2019, 2022)].groupby("court").size().idxmax()
    slope = (
        df[df['court'] == "33_10"]
        .groupby("year")[["year", "delay_days"]]
        .mean()
        .reset_index()
        .pipe(lambda d: np.polyfit(d['year'], d['delay_days'], 1)[0])
    )

    fig, ax = plt.subplots()
    yearly = df[df['court'] == "33_10"].groupby("year")["delay_days"].mean().reset_index()
    ax.scatter(yearly["year"], yearly["delay_days"])
    m, b = np.polyfit(yearly["year"], yearly["delay_days"], 1)
    ax.plot(yearly["year"], m * yearly["year"] + b, 'r--')
    ax.set_xlabel("Year")
    ax.set_ylabel("Avg Delay (days)")
    buf = BytesIO()
    fig.savefig(buf, format="png")
    img_uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    return {
        "Which high court disposed the most cases from 2019 - 2022?": most_cases,
        "What's the regression slope of delay days by year in court=33_10?": round(slope, 4),
        "Delay trend plot (base64 PNG)": img_uri
    }
