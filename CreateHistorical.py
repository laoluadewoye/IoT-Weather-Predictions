from tomllib import load as toml_load
from os.path import exists
from os import mkdir
from json import load as json_load, dump
from datetime import datetime, timedelta
from time import time, sleep
import pandas as pd
from requests import get, Response
from requests.exceptions import ConnectTimeout

# Import configuration file
with open('./create_historical_config.toml', 'rb') as config_file:
    config_settings = toml_load(config_file)

# Set constants
API_KEY: str = config_settings['API_KEY']
API_CALL_COUNT_FOLDER: str = config_settings['API_CALL_COUNT_FOLDER']
API_CALL_COUNT_LIMIT: int = config_settings['API_CALL_COUNT_LIMIT']
BSU_LATITUDE: float = config_settings['BSU_LATITUDE']
BSU_LONGITUDE: float = config_settings['BSU_LONGITUDE']
UNIT_SYSTEM_TYPE: str = config_settings['UNIT_SYSTEM_TYPE']
HIST_DAYS: int = config_settings['HIST_DAYS']
HIST_WEEKS: int = config_settings['HIST_WEEKS']
HIST_LOG_FOLDER: str = config_settings['HIST_LOG_FOLDER']
ROUND_LENGTH: int = config_settings['ROUND_LENGTH']
ROUND_DELAY_SEC: int = config_settings['ROUND_DELAY_SEC']
HIST_DATA_FOLDER: str = config_settings['HIST_DATA_FOLDER']

# API Key Check
assert API_KEY != 'GetYourOwnOneAPIKey', 'You need to add your own key here.'

# Get the current day
cur_day: str = datetime.now().strftime('%Y-%m-%d').replace('-', '_')


def add_log_line(weather_timestamp: int, api_timestamp: str, elapsed: timedelta,
                 status_code: int, reason: str) -> None:
    # Check if the folder and file exists
    if not exists(HIST_LOG_FOLDER):
        mkdir(HIST_LOG_FOLDER)

    # Write to log file
    log_filename: str = f'{HIST_LOG_FOLDER}/{cur_day}_log.csv'
    write_columns: bool = not exists(log_filename)
    with open(log_filename, 'a') as log_file:
        # Create the columns if needed
        if write_columns:
            log_file.write(
                'api_call_timestamp, requested_data_timestamp, '
                'time_to_respond, status_code, status_message\n'
            )

        # Add the log
        log_file.write(
            f'{api_timestamp}, {weather_timestamp}, '
            f'{round(elapsed.total_seconds(), 4)}, {status_code}, {reason}\n'
        )


def add_to_dict(hist_data: dict[str, list], response_json: dict) -> dict:
    # Add data to main dictionary
    try:
        hist_data['timestamp'].append(response_json['data'][0]['dt'])
    except KeyError:
        print('No timestamp found.')
        hist_data['timestamp'].append(-1)

    try:
        hist_data['temperature'].append(response_json['data'][0]['temp'])
    except KeyError:
        print('No temperature found.')
        hist_data['temperature'].append(-1)

    try:
        hist_data['air_pressure'].append(response_json['data'][0]['pressure'])
    except KeyError:
        print('No air pressure found.')
        hist_data['air_pressure'].append(-1)

    try:
        hist_data['humidity'].append(response_json['data'][0]['humidity'])
    except KeyError:
        print('No humidity found.')
        hist_data['humidity'].append(-1)

    try:
        hist_data['dew_point'].append(response_json['data'][0]['dew_point'])
    except KeyError:
        print('No dew point found.')
        hist_data['dew_point'].append(-1)

    try:
        hist_data['clouds'].append(response_json['data'][0]['clouds'])
    except KeyError:
        print('No clouds found.')
        hist_data['clouds'].append(-1)

    try:
        hist_data['visibility'].append(response_json['data'][0]['visibility'])
    except KeyError:
        print('No visibility found.')
        hist_data['visibility'].append(-1)

    try:
        hist_data['wind_speed'].append(response_json['data'][0]['wind_speed'])
    except KeyError:
        print('No wind speed found.')
        hist_data['wind_speed'].append(-1)

    try:
        hist_data['wind_degrees'].append(response_json['data'][0]['wind_deg'])
    except KeyError:
        print('No wind degrees found.')
        hist_data['wind_degrees'].append(-1)

    try:
        hist_data['owm_weather_id'].append(
            response_json['data'][0]['weather'][0]['id'])
    except (KeyError, IndexError):
        print('No weather id found.')
        hist_data['owm_weather_id'].append(-1)

    try:
        hist_data['owm_weather_main'].append(
            response_json['data'][0]['weather'][0]['main'])
    except (KeyError, IndexError):
        print('No weather main found.')
        hist_data['owm_weather_main'].append('None')

    try:
        hist_data['owm_weather_description'].append(
            response_json['data'][0]['weather'][0]['description'])
    except (KeyError, IndexError):
        print('No weather description found.')
        hist_data['owm_weather_description'].append('None')

    try:
        hist_data['owm_weather_icon'].append(
            response_json['data'][0]['weather'][0]['icon'])
    except (KeyError, IndexError):
        print('No weather icon found.')
        hist_data['owm_weather_icon'].append('None')

    return hist_data


def create_historical() -> None:
    # Import API call counter
    if not exists(f'{API_CALL_COUNT_FOLDER}/'):
        mkdir(API_CALL_COUNT_FOLDER)

    counter_file: str = f'{API_CALL_COUNT_FOLDER}/{cur_day}_counter.json'

    if not exists(counter_file):
        with open(counter_file, 'w') as cur_file:
            dump({'count': 0}, cur_file, indent=4)
        count_tracker: dict = {'count': 0}
    else:
        with open(counter_file) as cur_file:
            count_tracker: dict = json_load(cur_file)

    # Calculate the edges of the interval
    hour_as_seconds: int = 60 * 60
    hist_seconds_total: int = (HIST_DAYS + HIST_WEEKS * 7) * 24 * 60 * 60
    hist_end: int = int(time())
    hist_current: int = hist_end - hist_seconds_total

    # Obtain data in metric by default
    historical_data: dict[str, list] = {
        'timestamp': [], 'temperature': [], 'air_pressure': [], 'humidity': [],
        'dew_point': [], 'clouds': [], 'visibility': [], 'wind_speed': [],
        'wind_degrees': [], 'owm_weather_id': [], 'owm_weather_main': [],
        'owm_weather_description': [], 'owm_weather_icon': [],
    }

    round_index: int = 0
    while hist_current < hist_end:
        # Check the counter first
        if count_tracker['count'] >= API_CALL_COUNT_LIMIT:
            print('Daily API call limit reached. Please wait until tomorrow.')
            break

        # Create link for openweathermap API call
        one_call_link: str = (
            f'https://api.openweathermap.org/data/3.0/onecall/timemachine?'
            f'lat={BSU_LATITUDE}&lon={BSU_LONGITUDE}&dt={hist_current}&'
            f'appid={API_KEY}&units={UNIT_SYSTEM_TYPE}'
        )
        print(f'Requesting data for {datetime.fromtimestamp(hist_current)}...')

        # Get data
        response_timestamp: str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        try:
            response: Response = get(one_call_link, timeout=3)
        except ConnectTimeout:
            print(f'Connection for request for timestamp '
                  f'{hist_current} timed out.')
            continue
        finally:
            # Update API call counter
            count_tracker['count'] += 1
            with open(counter_file, 'w') as cur_file:
                dump(count_tracker, cur_file, indent=4)

        # Save response metadata to log
        add_log_line(
            hist_current, response_timestamp, response.elapsed,
            response.status_code, response.reason
        )

        # Add data to main dictionary
        historical_data: dict[str, list] = add_to_dict(
            historical_data, response.json()
        )

        # Add hour time interval to current time
        hist_current += hour_as_seconds

        # Update round index
        round_index += 1

        # Check if we are around the API call limit
        if round_index % ROUND_LENGTH == 0:
            print(f'Sleeping for {ROUND_DELAY_SEC} seconds...')
            for i in range(ROUND_DELAY_SEC):
                print(f'{ROUND_DELAY_SEC - i} seconds left...')
                sleep(1)

    # Save data
    historical_data_df: pd.DataFrame = pd.DataFrame(historical_data)

    if not exists(HIST_DATA_FOLDER):
        mkdir(HIST_DATA_FOLDER)
    hist_csv_filename = f'historical_data_{time()}'.replace('.', '_')
    historical_data_df.to_csv(
        f'{HIST_DATA_FOLDER}/{hist_csv_filename}.csv', index=False
    )


if __name__ == '__main__':
    create_historical()
