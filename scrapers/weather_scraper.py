#!/usr/bin/env python3
"""
NRL Weather Data Scraper (Open-Meteo Version) - FINAL

This script scrapes historical weather data for NRL matches using the free
Open-Meteo API. It uses latitude and longitude columns from the input file.
"""

import pandas as pd
import numpy as np
import requests
import time
import os
from datetime import datetime, date
import json

# --- Configuration ---
INPUT_FILE = 'nrlBaselineData.csv'
OUTPUT_FILE = 'nrlBaselineDataWithWeather.csv'
LOG_FILE = 'scrapers/weather_scraper_log.txt'
CACHE_FILE = 'scrapers/weather_cache_openmeteo.json'

REQUESTS_PER_MINUTE = 200
DELAY_BETWEEN_REQUESTS = 60 / REQUESTS_PER_MINUTE

def setup_logging():
    if not os.path.exists('scrapers'):
        os.makedirs('scrapers')
    with open(LOG_FILE, 'w') as f:
        f.write(f"Weather Scraper Log - Started: {datetime.now()}\n")
        f.write("="*60 + "\n\n")

def log_message(message):
    print(message)
    with open(LOG_FILE, 'a') as f:
        f.write(f"{datetime.now().strftime('%H:%M:%S')} - {message}\n")

def load_weather_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            log_message(f"  Warning: Could not read cache file {CACHE_FILE}. Starting fresh.")
    return {}

def save_weather_cache(cache):
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f, indent=2)
        log_message(f" Weather cache saved ({len(cache)} entries)")
    except Exception as e:
        log_message(f"  Could not save weather cache: {e}")

def fetch_weather_data_openmeteo(date_str, lat, lon, cache):
    cache_key = f"{date_str}_{lat:.4f}_{lon:.4f}"
    if cache_key in cache:
        return cache[cache_key]

    BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
    
    params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': date_str,
        'end_date': date_str,
        # <<< FINAL FIX #1: Removed 'relativehumidity_2m_mean' as it's not available on the historical API
        'daily': [
            'temperature_2m_mean', 'precipitation_sum', 'windspeed_10m_max', 'uv_index_max'
        ],
        'timezone': 'auto'
    }
    
    try:
        response = requests.get(BASE_URL, params=params, timeout=20)
        
        if response.status_code == 200:
            data = response.json()['daily']
            weather_info = {
                'temperature_c': data['temperature_2m_mean'][0],
                'humidity': None, # Set to None as we cannot fetch it from this endpoint
                'wind_speed_kph': data['windspeed_10m_max'][0],
                'precipitation_mm': data['precipitation_sum'][0],
                'condition_text': 'N/A',
                'uv_index': data['uv_index_max'][0],
                'visibility_km': None,
            }
            cache[cache_key] = weather_info
            return weather_info
        else:
            log_message(f"    API Error {response.status_code} for {date_str} @ ({lat},{lon}). Details: {response.text}")
            return None
    except Exception as e:
        log_message(f"    Exception fetching {date_str} @ ({lat},{lon}): {str(e)}")
        return None

def scrape_weather_for_matches(df_to_scrape):
    log_message("  Starting weather data scraping...")
    
    cache = load_weather_cache()
    initial_cache_size = len(cache)
    
    unique_requests = df_to_scrape[['Date_str', 'latitude', 'longitude']].drop_duplicates().dropna()
    
    log_message(f" Found {len(unique_requests)} unique historical date/location combinations to process.")
    log_message(f" {initial_cache_size} entries already in cache")
    
    new_requests = unique_requests.apply(lambda row: f"{row.Date_str}_{row.latitude:.4f}_{row.longitude:.4f}" not in cache, axis=1)
    new_requests_needed = new_requests.sum()

    log_message(f" Need to fetch {new_requests_needed} new weather records")
    
    if new_requests_needed == 0:
        log_message(" All required weather data already cached!")
        return cache
    
    estimated_seconds = (new_requests_needed * DELAY_BETWEEN_REQUESTS)
    log_message(f"  Estimated time: {estimated_seconds / 60:.1f} minutes")
    
    successful_requests = 0
    failed_requests = 0
    
    for idx, row in unique_requests[new_requests].iterrows():
        date = row['Date_str']
        lat = row['latitude']
        lon = row['longitude']
        
        log_message(f"   Fetching {date} @ ({lat:.2f}, {lon:.2f})...")
        weather_data = fetch_weather_data_openmeteo(date, lat, lon, cache)
        
        if weather_data:
            successful_requests += 1
        else:
            failed_requests += 1
        
        if (successful_requests + failed_requests) > 0 and (successful_requests + failed_requests) % 25 == 0:
            log_message(f"   Progress: {successful_requests + failed_requests}/{new_requests_needed}")
        
        time.sleep(DELAY_BETWEEN_REQUESTS)
        
        if successful_requests > 0 and successful_requests % 50 == 0:
            save_weather_cache(cache)
    
    log_message(f" Weather scraping complete!")
    log_message(f"   Successful: {successful_requests}, Failed: {failed_requests}")
    save_weather_cache(cache)
    return cache

def enhance_dataframe_with_weather(df, weather_cache):
    log_message(" Adding weather features to match data...")
    
    weather_columns = [
        'temperature_c', 'humidity', 'wind_speed_kph', 'precipitation_mm',
        'condition_text', 'uv_index', 'visibility_km'
    ]
    for col in weather_columns:
        df[col] = np.nan
    df['condition_text'] = df['condition_text'].astype('object')
    
    def apply_weather(row):
        cache_key = f"{row['Date_str']}_{row['latitude']:.4f}_{row['longitude']:.4f}"
        weather_data = weather_cache.get(cache_key)
        if weather_data:
            for col, value in weather_data.items():
                row[col] = value
        return row

    df = df.apply(apply_weather, axis=1)
    log_message(f" Applied weather data to match records.")
    return df

def create_derived_features(df):
    log_message(" Creating derived weather features...")
    if 'precipitation_mm' in df.columns:
        df['is_rainy'] = (df['precipitation_mm'] > 2.0).astype(int)
    if 'wind_speed_kph' in df.columns:
        df['is_windy'] = (df['wind_speed_kph'] > 25.0).astype(int)
    if 'temperature_c' in df.columns:
        conditions = [
            df['temperature_c'] <= 10,
            (df['temperature_c'] > 10) & (df['temperature_c'] <= 20),
            (df['temperature_c'] > 20) & (df['temperature_c'] <= 30)
        ]
        choices = ['cold', 'mild', 'warm']
        df['temperature_category'] = np.select(conditions, choices, default='hot')
        df.loc[df['temperature_c'].isna(), 'temperature_category'] = 'unknown'
    return df

def main():
    setup_logging()
    log_message("NRL Weather Scraper (Open-Meteo) Started")
    log_message("="*50)
    
    try:
        log_message(f" Loading original data from: {INPUT_FILE}")
        df = pd.read_csv(INPUT_FILE)
        
        df['parsed_date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
        
        original_rows = len(df)
        df.dropna(subset=['parsed_date', 'latitude', 'longitude'], inplace=True)
        if len(df) < original_rows:
            log_message(f"  Dropped {original_rows - len(df)} rows with missing date or coordinates.")

        # <<< FINAL FIX #2: Correctly filter for historical dates BEFORE passing to scraper
        historical_df = df[df['parsed_date'].dt.date < date.today()].copy()
        
        future_matches_count = len(df) - len(historical_df)
        if future_matches_count > 0:
            log_message(f"  Ignoring {future_matches_count} future-dated matches.")
        
        if historical_df.empty:
            log_message(" ERROR: No historical matches found in the dataset to process.")
            return False
            
        historical_df['Date_str'] = historical_df['parsed_date'].dt.strftime('%Y-%m-%d')

    except FileNotFoundError:
        log_message(f" ERROR: Could not find {INPUT_FILE}")
        return False
    except Exception as e:
        log_message(f" ERROR: Failed to load or clean data: {e}")
        return False
    
    # Pass the correctly filtered dataframe to the functions
    weather_cache = scrape_weather_for_matches(historical_df)
    enhanced_df = enhance_dataframe_with_weather(historical_df, weather_cache)
    enhanced_df = create_derived_features(enhanced_df)
    
    try:
        log_message(f"\n Saving enhanced dataset to: {OUTPUT_FILE}")
        enhanced_df.drop(columns=['parsed_date', 'Date_str'], inplace=True, errors='ignore')
        enhanced_df.to_csv(OUTPUT_FILE, index=False)
        log_message(f" Enhanced dataset with {len(enhanced_df.columns)} columns saved successfully!")
    except Exception as e:
        log_message(f" ERROR: Failed to save enhanced dataset: {e}")
        return False
        
    log_message(f"\n SUCCESS: Weather scraping completed!")
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print(f"\nScraping finished successfully. Check {OUTPUT_FILE} and {LOG_FILE}.")
    else:
        print(f"\nScraping failed. Please check {LOG_FILE} for errors.")