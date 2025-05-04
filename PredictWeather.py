from tomllib import load as toml_load
from os.path import exists
from glob import glob
from hashlib import sha256
from joblib import load as joblib_load
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from typing import Union
from requests import get, Response
from json import load as json_load, dump as json_dump
from pandas import DataFrame, concat, Series
from datetime import datetime
from time import sleep


# Import configuration file
with open('./create_historical_config.toml', 'rb') as config_file:
    config_settings: dict = toml_load(config_file)

# Set constants
API_KEY: str = config_settings['API_KEY']
API_CALL_COUNT_FOLDER: str = config_settings['API_CALL_COUNT_FOLDER']
API_CALL_COUNT_LIMIT: int = config_settings['API_CALL_COUNT_LIMIT']
BSU_LATITUDE: float = config_settings['BSU_LATITUDE']
BSU_LONGITUDE: float = config_settings['BSU_LONGITUDE']
UNIT_SYSTEM_TYPE: str = config_settings['UNIT_SYSTEM_TYPE']
HIST_DATA_FOLDER: str = config_settings['HIST_DATA_FOLDER']
TRAINING_OUTPUT_FOLDER: str = config_settings['TRAINING_OUTPUT_FOLDER']
TRAINING_OUTPUT_CHOICE: str = config_settings['TRAINING_OUTPUT_CHOICE']


def set_target_folder() -> str:
    target_folder: str = f'{TRAINING_OUTPUT_FOLDER}/{TRAINING_OUTPUT_CHOICE}'

    # Check if the target folder exists
    if exists(target_folder) and not TRAINING_OUTPUT_CHOICE == '':
        target_exists: bool = True
    elif TRAINING_OUTPUT_CHOICE == '':
        csv_files: list[str] = glob(f'{HIST_DATA_FOLDER}/*')
        hash_input: str = ''.join(csv_files)
        hash_output: str = sha256(hash_input.encode()).hexdigest()[:16]
        target_folder = f'{TRAINING_OUTPUT_FOLDER}/{hash_output}'
        target_exists: bool = exists(target_folder)
    else:
        target_exists: bool = False

    # End the program if no saved models are found
    assert target_exists, 'No saved models found.'

    return target_folder


def retrieve_weather_data() -> tuple:
    # Retrieve the API call counter
    cur_day: str = datetime.now().strftime('%Y-%m-%d').replace('-', '_')
    counter_file: str = f'{API_CALL_COUNT_FOLDER}/{cur_day}_counter.json'

    # Check if the counter file exists
    if not exists(counter_file):
        with open(counter_file, 'w') as cur_file:
            json_dump({'count': 0}, cur_file, indent=4)
        count_tracker: dict = {'count': 0}
    else:
        with open(counter_file) as cur_file:
            count_tracker: dict = json_load(cur_file)

    # Check the counter first
    assert count_tracker['count'] < API_CALL_COUNT_LIMIT, (
        'Daily API call limit reached. Please wait until tomorrow.'
    )

    # Create current and forecast call urls
    current_call_url: str = (f'https://api.openweathermap.org/data/2.5/weather?'
                             f'lat={BSU_LATITUDE}&lon={BSU_LONGITUDE}&'
                             f'appid={API_KEY}&units={UNIT_SYSTEM_TYPE}')
    five_day_call_url: str = (f'https://api.openweathermap.org/data/2.5/'
                              f'forecast?lat={BSU_LATITUDE}&'
                              f'lon={BSU_LONGITUDE}&appid={API_KEY}&'
                              f'units={UNIT_SYSTEM_TYPE}')

    # Make current and forecast call if capable
    call_success: bool = True
    current_data: Union[dict, None] = None
    five_day_data: Union[dict, None] = None
    try:
        count_tracker['count'] += 1
        current_response: Response = get(current_call_url)
        current_response.raise_for_status()
        current_data = current_response.json()

        count_tracker['count'] += 1
        five_day_response: Response = get(five_day_call_url)
        five_day_response.raise_for_status()
        five_day_data = five_day_response.json()
    except Exception:
        call_success = False
    finally:
        with open(counter_file, 'w') as cur_file:
            json_dump(count_tracker, cur_file, indent=4)

    # Return current and forecast data with success metric
    return current_data, five_day_data, call_success


def create_df(current: dict, five_day: dict) -> DataFrame:
    # Populate data dictionary then convert to dataframe
    data_dict: dict[str, list] = {
        'timestamp': [current['dt']],
        'temperature': [current['main']['temp']],
        'air_pressure': [current['main']['pressure']],
        'humidity': [current['main']['humidity']],
        'clouds': [current['clouds']['all']],
        'visibility': [current['visibility']],
        'wind_speed': [current['wind']['speed']],
        'owm_weather_id': [current['weather'][0]['id']],
        'owm_weather_main': [current['weather'][0]['main']],
        'owm_weather_description': [current['weather'][0]['description']],
        'owm_weather_icon': [current['weather'][0]['icon']]
    }
    for three_hour in five_day['list']:
        data_dict['timestamp'].append(three_hour['dt'])
        data_dict['temperature'].append(three_hour['main']['temp'])
        data_dict['air_pressure'].append(three_hour['main']['pressure'])
        data_dict['humidity'].append(three_hour['main']['humidity'])
        data_dict['clouds'].append(three_hour['clouds']['all'])
        data_dict['visibility'].append(three_hour['visibility'])
        data_dict['wind_speed'].append(three_hour['wind']['speed'])
        data_dict['owm_weather_id'].append(three_hour['weather'][0]['id'])
        data_dict['owm_weather_main'].append(three_hour['weather'][0]['main'])
        data_dict['owm_weather_description'].append(
            three_hour['weather'][0]['description']
        )
        data_dict['owm_weather_icon'].append(three_hour['weather'][0]['icon'])
    data_df: DataFrame = DataFrame(data_dict)

    return data_df


def scale_data(scaler: MinMaxScaler, df: DataFrame) -> DataFrame:
    input_columns: list[str] = [
        'temperature', 'air_pressure', 'humidity', 'clouds', 'visibility',
        'wind_speed'
    ]

    # Use scaler to scale values
    scaled_input_df: DataFrame = DataFrame(
        scaler.transform(df[input_columns]),
        columns=input_columns
    )

    # Reconstruct dataframe
    scaled_data_df: DataFrame = concat(
        [
            df[['timestamp']],
            scaled_input_df,
            df[[
                'owm_weather_id', 'owm_weather_main',
                'owm_weather_description', 'owm_weather_icon'
            ]]
        ],
        axis=1
    )

    return scaled_data_df


def predict_weather(cur_row: Series, dt_models: dict, encodings: dict) -> tuple:
    cur_row_inputs: Series = cur_row.drop([
        'timestamp', 'owm_weather_id', 'owm_weather_main',
        'owm_weather_description', 'owm_weather_icon'
    ])
    cur_row_inputs_frame: DataFrame = cur_row_inputs.to_frame().T

    # Predict ID
    id_pred = dt_models['owm_weather_id_enc'].predict(cur_row_inputs_frame)
    id_origin = encodings['owm_weather_id'][int(id_pred[0])]

    # Predict Main
    main_pred = dt_models['owm_weather_main_enc'].predict(cur_row_inputs_frame)
    main_origin = encodings['owm_weather_main'][int(main_pred[0])]

    # Predict Description
    desc_pred = dt_models['owm_weather_description_enc'].predict(
        cur_row_inputs_frame
    )
    desc_origin = encodings['owm_weather_description'][int(desc_pred[0])]

    # Predict Icon
    icon_pred = dt_models['owm_weather_icon_enc'].predict(cur_row_inputs_frame)
    icon_origin = encodings['owm_weather_icon'][int(icon_pred[0])]

    return id_origin, main_origin, desc_origin, icon_origin


def predict_forecast(df: DataFrame, dt_models: dict, encodings: dict) -> dict:
    forecast_results: dict[str, list] = {
        'Outputs': ['ID', 'Main', 'Description', 'Icon'],
        'Correct': [0, 0, 0, 0],
        'Incorrect': [0, 0, 0, 0]
    }

    # Prediction loop
    for row_index, row in df.iterrows():
        # Get predictions
        id_origin, main_origin, desc_origin, icon_origin = predict_weather(
            row, dt_models, encodings
        )

        # Get formatted timestamp
        formatted_timestamp: datetime = datetime.fromtimestamp(row['timestamp'])

        # Check if predictions are correct
        id_is_correct: bool = id_origin == row['owm_weather_id']
        id_correct_string: str = 'Correct' if id_is_correct else 'Incorrect'
        if id_is_correct:
            forecast_results['Correct'][0] += 1
        else:
            forecast_results['Incorrect'][0] += 1

        main_is_correct: bool = main_origin == row['owm_weather_main']
        main_correct_string: str = 'Correct' if main_is_correct else 'Incorrect'
        if main_is_correct:
            forecast_results['Correct'][1] += 1
        else:
            forecast_results['Incorrect'][1] += 1

        desc_is_correct: bool = desc_origin == row['owm_weather_description']
        desc_correct_string: str = 'Correct' if desc_is_correct else 'Incorrect'
        if desc_is_correct:
            forecast_results['Correct'][2] += 1
        else:
            forecast_results['Incorrect'][2] += 1

        icon_is_correct: bool = icon_origin == row['owm_weather_icon']
        icon_correct_string: str = 'Correct' if icon_is_correct else 'Incorrect'
        if icon_is_correct:
            forecast_results['Correct'][3] += 1
        else:
            forecast_results['Incorrect'][3] += 1

        # Print predictions
        if row_index == 0:
            print('Current Weather:')

        print('-' * 80)
        print(f'\tPredicted weather at {formatted_timestamp}:')
        print(f'\tID: {id_origin} ({id_correct_string})')
        print(f'\tMain: {main_origin} ({main_correct_string})')
        print(f'\tDescription: {desc_origin} ({desc_correct_string})')
        print(f'\tIcon: {icon_origin} ({icon_correct_string})')
        print('-' * 80)

        if row_index == 0:
            sleep(2)

        sleep(0.25)

    return forecast_results


def make_predictions() -> None:
    # Establish target folder
    target_folder: str = set_target_folder()

    # Import desired models, data scaler, and encodings
    dt_models: dict[str, DecisionTreeClassifier] = joblib_load(
        f'{target_folder}/dt_model_dict.joblib'
    )
    dt_scaler: MinMaxScaler = joblib_load(
        f'{target_folder}/dt_minmax_scaler.joblib'
    )
    output_encodings: dict[str, dict] = joblib_load(
        f'{target_folder}/historical_data_encodings.joblib'
    )

    # Obtain real time current and forecast data
    current, five_day, call_success = retrieve_weather_data()
    assert call_success, 'Unable to retrieve weather data.'

    # Populate data dictionary then convert to dataframe
    data_df: DataFrame = create_df(current, five_day)

    # Adjust units of measurement to metric if necessary
    if UNIT_SYSTEM_TYPE == 'standard':
        # Kelvin to Celsius
        data_df['temperature'] -= 273.15
    elif UNIT_SYSTEM_TYPE == 'imperial':
        # Fahrenheit to Celsius
        data_df['temperature'] = (data_df['temperature'] - 32) / 1.8
        # Miles per hour to meters per second
        data_df['wind_speed'] *= 0.44704

    # Scale the metric data
    scaled_data_df: DataFrame = scale_data(dt_scaler, data_df)

    # Make predictions and store the accuracy results
    forecast_results: dict[str, list] = predict_forecast(
        scaled_data_df, dt_models, output_encodings
    )
    forecast_results_df: DataFrame = DataFrame(forecast_results)
    forecast_results_df.to_csv(
        f'{target_folder}/forecast_results.csv', index=False
    )


if __name__ == '__main__':
    make_predictions()
