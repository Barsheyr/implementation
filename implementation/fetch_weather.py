"""Fetch NASA POWER daily data for the reference zones and build the 16 seasonal profiles.

Run on any computer with internet access:
    pip install requests pandas numpy
    python fetch_weather.py --out weather_dir

Then upload the folder `weather_dir` (weather_profiles.csv and weather_meta.json) to Kaggle as a dataset
and pass --weather-dir to cassava_experiment.py. No data are invented: if a request fails, the script stops.
"""
import argparse
import datetime as dt
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import requests

API = "https://power.larc.nasa.gov/api/temporal/daily/point"
PARAMS = ["T2M", "RH2M", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN"]
ZONES = {  # one reference town per region of Uganda (latitude, longitude)
    "Northern": ("Gulu", 2.78, 32.30),
    "Eastern": ("Soroti", 1.71, 33.61),
    "Central": ("Masaka", -0.34, 31.73),
    "Western": ("Mbarara", -0.61, 30.66),
}
PERIOD = ("20180101", "20191231")
SEASONS = {"MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11), "DJF": (12, 1, 2)}


def fetch_zone(lat, lon, period, retries=4):
    q = {"parameters": ",".join(PARAMS), "community": "AG", "latitude": lat, "longitude": lon,
         "start": period[0], "end": period[1], "format": "JSON"}
    for k in range(retries):
        try:
            r = requests.get(API, params=q, timeout=120)
            if r.status_code == 200:
                return r.json(), r.url
            print(f"HTTP {r.status_code} for ({lat},{lon}); retry {k + 1}", file=sys.stderr)
        except requests.RequestException as e:
            print(f"request error {e}; retry {k + 1}", file=sys.stderr)
        time.sleep(5 * (k + 1))
    raise SystemExit(f"NASA POWER request failed for ({lat},{lon}). Not inventing data: stop.")


def to_frame(js):
    par = js["properties"]["parameter"]
    df = pd.DataFrame({p: pd.Series(par[p]) for p in PARAMS})
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    return df


def build_profiles(period=PERIOD, zones=ZONES):
    rows, meta = [], {"zones": {}, "period": list(period), "api": API, "parameters": PARAMS,
                      "fetch_time_utc": dt.datetime.utcnow().isoformat() + "Z", "missing_value_code": -999,
                      "seasons": {k: list(v) for k, v in SEASONS.items()}}
    for zname, (town, lat, lon) in zones.items():
        js, url = fetch_zone(lat, lon, period)
        df = to_frame(js)
        n_days = len(df)
        n_missing_days = int((df == -999).any(axis=1).sum())
        df = df.replace(-999, np.nan)
        meta["zones"][zname] = {"town": town, "latitude": lat, "longitude": lon, "request_url": url,
                                "n_days": n_days, "n_days_with_missing_value": n_missing_days}
        for sname, months in SEASONS.items():
            d = df[df.index.month.isin(months)]
            # a December belongs to the DJF season of the following year; only complete seasons are used
            sy = d.index.year + (d.index.month == 12).astype(int)
            full = d.groupby(sy).size()
            keep = full[full >= 0.9 * (92 if sname in ("JJA", "SON") else 90)].index
            d = d[np.isin(sy, keep)]
            sy = sy[np.isin(sy, keep)]
            if len(d) == 0:
                raise SystemExit(f"No complete {sname} season for {zname} in the period")
            rain_by_year = d["PRECTOTCORR"].groupby(sy).sum(min_count=1)
            rows.append({"profile_id": len(rows), "zone": zname, "season": sname,
                         "T2M": d["T2M"].mean(), "RH2M": d["RH2M"].mean(),
                         "PRECTOTCORR": rain_by_year.mean(),
                         "ALLSKY_SFC_SW_DWN": d["ALLSKY_SFC_SW_DWN"].mean(),
                         "n_seasons_used": int(len(keep))})
        time.sleep(2)
    return pd.DataFrame(rows), meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="weather_dir")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    prof, meta = build_profiles()
    prof.to_csv(os.path.join(a.out, "weather_profiles.csv"), index=False)
    json.dump(meta, open(os.path.join(a.out, "weather_meta.json"), "w"), indent=2)
    print(prof.round(2).to_string(index=False))
    print("saved to", a.out)
