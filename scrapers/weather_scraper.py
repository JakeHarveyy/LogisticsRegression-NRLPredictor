#!/usr/bin/env python3
"""
NRL Weather Data Scraper

This script scrapes historical weather data for NRL matches using WeatherAPI.com
and creates an enhanced baseline CSV with weather features included.

Usage:
    python weather_scraper.py

Requirements:
    1. WeatherAPI.com API key (free tier: 1M calls/month)
    2. Set WEATHER_API_KEY environment variable or edit API_KEY below
    3. Original nrlBaselineData.csv file

Output:
    - nrlBaselineDataWithWeather.csv (enhanced dataset)
    - weather_scraper_log.txt (detailed log file)

Author: NRL Betting Model Team
Date: July 2025
"""

import pandas as pd
import numpy as np
import requests
import time
import os
from datetime import datetime
import json

# Configuration
API_KEY = os.getenv('WEATHER_API_KEY')  # Set your API key as environment variable
# Or set directly here: API_KEY = "your_api_key_here"

BASE_URL = "http://api.weatherapi.com/v1/history.json"

# File paths (relative to parent directory since script is in scrapers/ folder)
INPUT_FILE = '../nrlBaselineData.csv'
OUTPUT_FILE = '../nrlBaselineDataWithWeather.csv'
LOG_FILE = '../weather_scraper_log.txt'
CACHE_FILE = '../weather_cache.json'

# Rate limiting (WeatherAPI Pro Plus plan: much higher limits)
# Pro Plus typically allows 500+ calls/minute vs 100 for free tier
REQUESTS_PER_MINUTE = 300  # Conservative rate for Pro Plus
DELAY_BETWEEN_REQUESTS = 60 / REQUESTS_PER_MINUTE  # ~0.2 seconds between requests

def setup_logging():
    """Initialize logging"""
    with open(LOG_FILE, 'w') as f:
        f.write(f"Weather Scraper Log - Started: {datetime.now()}\n")
        f.write("="*60 + "\n\n")

def log_message(message):
    """Log message to both console and file"""
    print(message)
    with open(LOG_FILE, 'a') as f:
        f.write(f"{datetime.now().strftime('%H:%M:%S')} - {message}\n")

def load_weather_cache():
    """Load cached weather data to avoid duplicate API calls"""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        except:
            log_message("⚠️  Could not load weather cache, starting fresh")
    return {}

def save_weather_cache(cache):
    """Save weather cache to file"""
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f, indent=2)
        log_message(f"✓ Weather cache saved ({len(cache)} entries)")
    except Exception as e:
        log_message(f"⚠️  Could not save weather cache: {e}")

def get_city_mapping():
    """Map NRL cities to WeatherAPI locations"""
    return {
        'Sydney': 'Sydney,Australia',
        'Melbourne': 'Melbourne,Australia',
        'Brisbane': 'Brisbane,Australia',
        'Gold Coast': 'Gold Coast,Australia',
        'Auckland': 'Auckland,New Zealand',
        'Canberra': 'Canberra,Australia',
        'Newcastle': 'Newcastle,Australia',
        'Townsville': 'Townsville,Australia',
        'Perth': 'Perth,Australia',
        'Wollongong': 'Wollongong,Australia',
        'Cronulla': 'Sydney,Australia',    # Part of Sydney
        'Manly': 'Sydney,Australia',       # Part of Sydney
        'Parramatta': 'Sydney,Australia',  # Part of Sydney
        'Penrith': 'Sydney,Australia',     # Part of Sydney
        'Canterbury': 'Sydney,Australia',  # Part of Sydney
        'Leichhardt': 'Sydney,Australia',  # Part of Sydney
        'Redcliffe': 'Brisbane,Australia', # Part of Brisbane
        'Kogarah': 'Sydney,Australia',     # Part of Sydney
        'Liverpool': 'Sydney,Australia',   # Part of Sydney
        'Campbelltown': 'Sydney,Australia' # Part of Sydney
    }

def fetch_weather_data(date, city, api_key, cache):
    """
    Fetch weather data for a specific date and city
    
    Args:
        date (str): Date in YYYY-MM-DD format
        city (str): City name
        api_key (str): WeatherAPI.com API key
        cache (dict): Cache dictionary to avoid duplicate calls
    
    Returns:
        dict: Weather data or None if failed
    """
    
    cache_key = f"{date}_{city}"
    
    # Check cache first
    if cache_key in cache:
        return cache[cache_key]
    
    city_mapping = get_city_mapping()
    api_location = city_mapping.get(city, f"{city},Australia")
    
    params = {
        'key': api_key,
        'q': api_location,
        'dt': date
    }
    
    try:
        response = requests.get(BASE_URL, params=params, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            day_data = data['forecast']['forecastday'][0]['day']
            
            weather_info = {
                'temperature_c': day_data.get('avgtemp_c'),
                'humidity': day_data.get('avghumidity'),
                'wind_speed_kph': day_data.get('maxwind_kph'),
                'precipitation_mm': day_data.get('totalprecip_mm'),
                'condition_text': day_data.get('condition', {}).get('text', 'Unknown'),
                'uv_index': day_data.get('uv'),
                'visibility_km': day_data.get('avgvis_km')
            }
            
            # Cache the result
            cache[cache_key] = weather_info
            
            return weather_info
            
        else:
            log_message(f"   ✗ API Error {response.status_code} for {date} {city}")
            return None
            
    except Exception as e:
        log_message(f"   ✗ Exception fetching {date} {city}: {str(e)}")
        return None

def create_weather_features(weather_data):
    """
    Create derived weather features from raw weather data
    
    Args:
        weather_data (dict): Raw weather data
    
    Returns:
        dict: Enhanced weather features
    """
    
    if not weather_data:
        return {}
    
    features = weather_data.copy()
    
    # Rain impact (boolean) - significant rain is >2mm
    features['is_rainy'] = 1 if (weather_data.get('precipitation_mm', 0) > 2.0) else 0
    
    # Wind impact (boolean) - strong wind is >25 kph
    features['is_windy'] = 1 if (weather_data.get('wind_speed_kph', 0) > 25) else 0
    
    # Temperature categories
    temp = weather_data.get('temperature_c')
    if temp is not None:
        if temp <= 10:
            features['temperature_category'] = 'cold'
        elif temp <= 20:
            features['temperature_category'] = 'mild'
        elif temp <= 30:
            features['temperature_category'] = 'warm'
        else:
            features['temperature_category'] = 'hot'
    else:
        features['temperature_category'] = 'unknown'
    
    # Weather quality score (0-1, higher = better conditions)
    # Optimal: 15-25°C, <5mm rain, <20kph wind
    if all(x is not None for x in [temp, weather_data.get('precipitation_mm'), weather_data.get('wind_speed_kph')]):
        temp_score = 1 - min(abs(temp - 20) / 30, 1)  # Optimal at 20°C
        rain_score = 1 - min(weather_data.get('precipitation_mm', 0) / 20, 1)
        wind_score = 1 - min(weather_data.get('wind_speed_kph', 0) / 40, 1)
        
        features['weather_score'] = (temp_score * 0.4 + rain_score * 0.4 + wind_score * 0.2)
    else:
        features['weather_score'] = None
    
    return features

def scrape_weather_for_matches(df, api_key):
    """
    Scrape weather data for all unique match dates/cities
    
    Args:
        df (pd.DataFrame): Match data DataFrame
        api_key (str): WeatherAPI.com API key
    
    Returns:
        dict: Weather data cache
    """
    
    log_message("🌤️  Starting weather data scraping...")
    
    # Load existing cache
    cache = load_weather_cache()
    initial_cache_size = len(cache)
    
    # Get unique date/city combinations
    df['Date_str'] = pd.to_datetime(df['Date'], format='%d/%m/%Y').dt.strftime('%Y-%m-%d')
    unique_matches = df[['Date_str', 'City']].drop_duplicates()
    
    log_message(f"✓ Found {len(unique_matches)} unique match date/city combinations")
    log_message(f"✓ {initial_cache_size} entries already in cache")
    
    # Count how many we need to fetch
    new_requests_needed = 0
    for _, row in unique_matches.iterrows():
        cache_key = f"{row['Date_str']}_{row['City']}"
        if cache_key not in cache:
            new_requests_needed += 1
    
    log_message(f"✓ Need to fetch {new_requests_needed} new weather records")
    
    if new_requests_needed == 0:
        log_message("✓ All weather data already cached!")
        return cache
    
    # Estimate time
    estimated_minutes = (new_requests_needed * DELAY_BETWEEN_REQUESTS) / 60
    log_message(f"⏱️  Estimated time: {estimated_minutes:.1f} minutes")
    
    # Fetch missing weather data
    successful_requests = 0
    failed_requests = 0
    
    for idx, row in unique_matches.iterrows():
        date = row['Date_str']
        city = row['City']
        cache_key = f"{date}_{city}"
        
        # Skip if already cached
        if cache_key in cache:
            continue
        
        log_message(f"   Fetching {date} {city}...")
        
        weather_data = fetch_weather_data(date, city, api_key, cache)
        
        if weather_data:
            successful_requests += 1
            temp = weather_data.get('temperature_c', 'N/A')
            rain = weather_data.get('precipitation_mm', 'N/A')
            log_message(f"   ✓ Success: {temp}°C, {rain}mm rain")
        else:
            failed_requests += 1
        
        # Progress update
        if (successful_requests + failed_requests) % 25 == 0:
            log_message(f"   Progress: {successful_requests + failed_requests}/{new_requests_needed}")
        
        # Rate limiting
        time.sleep(DELAY_BETWEEN_REQUESTS)
        
        # Save cache periodically
        if successful_requests % 50 == 0 and successful_requests > 0:
            save_weather_cache(cache)
    
    log_message(f"✓ Weather scraping complete!")
    log_message(f"   Successful: {successful_requests}")
    log_message(f"   Failed: {failed_requests}")
    log_message(f"   Success rate: {successful_requests/(successful_requests+failed_requests)*100:.1f}%")
    
    # Save final cache
    save_weather_cache(cache)
    
    return cache

def enhance_dataframe_with_weather(df, weather_cache):
    """
    Add weather features to the match DataFrame
    
    Args:
        df (pd.DataFrame): Original match DataFrame
        weather_cache (dict): Weather data cache
    
    Returns:
        pd.DataFrame: Enhanced DataFrame with weather features
    """
    
    log_message("🌟 Adding weather features to match data...")
    
    # Initialize weather columns
    weather_columns = [
        'temperature_c', 'humidity', 'wind_speed_kph', 'precipitation_mm',
        'condition_text', 'uv_index', 'visibility_km', 'is_rainy', 'is_windy',
        'temperature_category', 'weather_score'
    ]
    
    for col in weather_columns:
        df[col] = np.nan
    
    # Add string versions for categorical data
    df['condition_text'] = df['condition_text'].astype('object')
    df['temperature_category'] = df['temperature_category'].astype('object')
    
    # Apply weather data to each match
    weather_applied = 0
    
    for idx, row in df.iterrows():
        date_str = row['Date_str']
        city = row['City']
        cache_key = f"{date_str}_{city}"
        
        if cache_key in weather_cache:
            raw_weather = weather_cache[cache_key]
            enhanced_weather = create_weather_features(raw_weather)
            
            # Apply all weather features
            for col in weather_columns:
                if col in enhanced_weather:
                    df.loc[idx, col] = enhanced_weather[col]
            
            weather_applied += 1
    
    log_message(f"✓ Applied weather data to {weather_applied}/{len(df)} matches ({weather_applied/len(df)*100:.1f}%)")
    
    # Clean up temporary column
    df = df.drop('Date_str', axis=1)
    
    return df

def generate_weather_summary(df):
    """Generate summary statistics for weather features"""
    
    log_message("\n📊 WEATHER FEATURES SUMMARY:")
    
    weather_coverage = df['temperature_c'].notna().sum()
    total_matches = len(df)
    
    log_message(f"   Coverage: {weather_coverage}/{total_matches} matches ({weather_coverage/total_matches*100:.1f}%)")
    
    if weather_coverage > 0:
        log_message(f"   Temperature: {df['temperature_c'].min():.1f}°C to {df['temperature_c'].max():.1f}°C (avg: {df['temperature_c'].mean():.1f}°C)")
        log_message(f"   Precipitation: {df['precipitation_mm'].mean():.1f}mm average ({df['precipitation_mm'].max():.1f}mm max)")
        log_message(f"   Wind speed: {df['wind_speed_kph'].mean():.1f}kph average ({df['wind_speed_kph'].max():.1f}kph max)")
        
        rainy_matches = df['is_rainy'].sum()
        windy_matches = df['is_windy'].sum()
        
        log_message(f"   Rainy matches (>2mm): {rainy_matches} ({rainy_matches/total_matches*100:.1f}%)")
        log_message(f"   Windy matches (>25kph): {windy_matches} ({windy_matches/total_matches*100:.1f}%)")
        
        # Temperature distribution
        temp_dist = df['temperature_category'].value_counts()
        log_message(f"   Temperature distribution:")
        for category, count in temp_dist.items():
            log_message(f"     {category}: {count} ({count/weather_coverage*100:.1f}%)")
        
        # Weather score distribution
        avg_weather_score = df['weather_score'].mean()
        log_message(f"   Average weather quality score: {avg_weather_score:.3f}/1.000")

def main():
    """Main execution function"""
    
    setup_logging()
    log_message("🏉 NRL Weather Data Scraper Started")
    log_message("="*50)
    
    # Check API key
    if not API_KEY:
        log_message("❌ ERROR: No WeatherAPI.com API key found!")
        log_message("   Please set WEATHER_API_KEY environment variable or edit the script")
        log_message("   Get your free API key at: https://www.weatherapi.com/")
        return False
    
    log_message(f"✓ API key configured: {API_KEY[:8]}...")
    
    # Load original data
    try:
        log_message(f"📂 Loading original data from: {INPUT_FILE}")
        df = pd.read_csv(INPUT_FILE)
        log_message(f"✓ Loaded {len(df)} matches from {df['Date'].min()} to {df['Date'].max()}")
        
        # Show unique cities
        unique_cities = sorted(df['City'].unique())
        log_message(f"✓ Found matches in {len(unique_cities)} cities: {', '.join(unique_cities)}")
        
    except FileNotFoundError:
        log_message(f"❌ ERROR: Could not find {INPUT_FILE}")
        log_message("   Make sure the original baseline data file exists")
        return False
    except Exception as e:
        log_message(f"❌ ERROR: Failed to load data: {e}")
        return False
    
    # Scrape weather data
    try:
        weather_cache = scrape_weather_for_matches(df, API_KEY)
        
        if len(weather_cache) == 0:
            log_message("⚠️  WARNING: No weather data collected")
            return False
        
    except Exception as e:
        log_message(f"❌ ERROR: Weather scraping failed: {e}")
        return False
    
    # Enhance DataFrame with weather features
    try:
        enhanced_df = enhance_dataframe_with_weather(df, weather_cache)
        
    except Exception as e:
        log_message(f"❌ ERROR: Failed to enhance DataFrame: {e}")
        return False
    
    # Generate summary
    generate_weather_summary(enhanced_df)
    
    # Save enhanced dataset
    try:
        log_message(f"\n💾 Saving enhanced dataset to: {OUTPUT_FILE}")
        enhanced_df.to_csv(OUTPUT_FILE, index=False)
        log_message(f"✓ Enhanced dataset saved successfully!")
        log_message(f"   Original columns: {len(df.columns)}")
        log_message(f"   Enhanced columns: {len(enhanced_df.columns)}")
        log_message(f"   New weather features: {len(enhanced_df.columns) - len(df.columns)}")
        
    except Exception as e:
        log_message(f"❌ ERROR: Failed to save enhanced dataset: {e}")
        return False
    
    log_message(f"\n🎉 SUCCESS: Weather scraping completed!")
    log_message(f"   Enhanced dataset: {OUTPUT_FILE}")
    log_message(f"   Log file: {LOG_FILE}")
    log_message(f"   Weather cache: {CACHE_FILE}")
    
    return True

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n✅ Weather scraping completed successfully!")
        print(f"📄 Check {OUTPUT_FILE} for the enhanced dataset")
        print(f"📋 Check {LOG_FILE} for detailed logs")
    else:
        print("\n❌ Weather scraping failed!")
        print(f"📋 Check {LOG_FILE} for error details")
