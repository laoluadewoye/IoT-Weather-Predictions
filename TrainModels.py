from tomllib import load as toml_load
from glob import glob
from hashlib import sha256
from typing import Union
from numpy import ndarray
from pandas import read_csv, DataFrame, concat, Series
from os.path import exists
from os import makedirs
from copy import deepcopy
from joblib import dump as joblib_dump
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


# Set a default theme for later
sns.set_theme()

# Import configuration file
with open('./create_historical_config.toml', 'rb') as config_file:
    config_settings = toml_load(config_file)

# Set constants
HIST_DATA_FOLDER: str = config_settings['HIST_DATA_FOLDER']
TRAINING_OUTPUT_FOLDER: str = config_settings['TRAINING_OUTPUT_FOLDER']
TRAINING_OUTPUT_CHOICE: str = config_settings['TRAINING_OUTPUT_CHOICE']


def create_csv(output_folder: str, csv_fps: list[str]) -> DataFrame:
    # Combine all the csv files
    combined_df: Union[DataFrame, None] = None
    for csv_fp in csv_fps:
        cur_df: DataFrame = read_csv(csv_fp)
        if combined_df is None:
            combined_df = cur_df
        else:
            combined_df = concat([combined_df, cur_df], axis=0)

    # Organize the dataframe
    combined_df = combined_df.reset_index(drop=True)
    combined_df = combined_df.sort_values(by=['timestamp'])

    # Save the dataframe
    combined_fp: str = f'{output_folder}/historical_data_combined.csv'
    combined_df.to_csv(combined_fp, index=False)

    return combined_df


def create_encodings(df: DataFrame, output_folder: str) -> dict[str, dict]:
    forward_encodings: dict[str, dict] = {
        'owm_weather_id': {},
        'owm_weather_main': {},
        'owm_weather_description': {},
        'owm_weather_icon': {}
    }
    reverse_encodings: dict[str, dict] = {
        'owm_weather_id': {},
        'owm_weather_main': {},
        'owm_weather_description': {},
        'owm_weather_icon': {}
    }

    # Populate encodings
    for column_name in forward_encodings.keys():
        unique_values: ndarray = df[column_name].unique()
        for index_encode in range(len(unique_values)):
            if isinstance(unique_values[index_encode], str):
                plain_encoding = unique_values[index_encode]
            else:
                plain_encoding = unique_values[index_encode].item()
            forward_encodings[column_name][plain_encoding] = index_encode
            reverse_encodings[column_name][index_encode] = plain_encoding

    # Save reverse encodings for later
    joblib_dump(
        reverse_encodings, f'{output_folder}/historical_data_encodings.joblib'
    )

    return forward_encodings


def train_dt_models(scaled_df: DataFrame, input_columns: list[str],
                    output_columns: list[str]) -> tuple:
    # Create training inputs
    x: DataFrame = scaled_df[input_columns]

    # Create empty structures
    dt_models: dict[str, DecisionTreeClassifier] = {}
    dt_results: dict[str, list] = {'Category': [], 'Accuracy': [], 'Loss': []}

    # Create category mapping
    category_map: dict[str, str] = {
        'owm_weather_id_enc': 'Weather ID',
        'owm_weather_main_enc': 'Main Label',
        'owm_weather_description_enc': 'Description',
        'owm_weather_icon_enc': 'Icon Choice'
    }

    for output_column in output_columns:
        # Create training output
        y: Series = scaled_df[output_column]

        # Create training splits
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, train_size=0.8, random_state=42
        )

        # Train and test the model
        new_dt_model: DecisionTreeClassifier = DecisionTreeClassifier()
        new_dt_model.fit(x_train, y_train)

        y_pred: ndarray = new_dt_model.predict(x_test)
        acc_score: float = accuracy_score(y_test, y_pred) * 100

        # Save model results
        dt_results['Category'].append(category_map[output_column])
        dt_results['Accuracy'].append(acc_score)
        dt_results['Loss'].append(100 - acc_score)

        # Save model
        dt_models[output_column] = new_dt_model

    return dt_models, dt_results


def create_dt_models(df: DataFrame, output_folder: str) -> None:
    # Make a working copy of the dataframe and remove timestamps
    working_df: DataFrame = deepcopy(df)
    del working_df['timestamp']

    input_columns: list[str] = [
        'temperature', 'air_pressure', 'humidity', 'clouds', 'visibility',
        'wind_speed'
    ]
    output_columns: list[str] = [
        'owm_weather_id_enc', 'owm_weather_main_enc',
        'owm_weather_description_enc', 'owm_weather_icon_enc'
    ]

    # Scale the data column by column and save information
    scaler: MinMaxScaler = MinMaxScaler()
    scaled_input_df: DataFrame = DataFrame(
        scaler.fit_transform(working_df[input_columns]),
        columns=input_columns
    )
    scaled_df: DataFrame = concat(
        [scaled_input_df, working_df[output_columns]], axis=1
    )

    joblib_dump(scaler, f'{output_folder}/dt_minmax_scaler.joblib')
    scaled_df.to_csv(
        f'{output_folder}/historical_data_scaled.csv', index=False
    )

    # Create and save model structures
    dt_models, dt_results = train_dt_models(
        scaled_df, input_columns, output_columns
    )
    dt_results_df: DataFrame = DataFrame(dt_results)

    joblib_dump(dt_models, f'{output_folder}/dt_model_dict.joblib')
    dt_results_df.to_csv(
        f'{output_folder}/dt_model_results.csv', index=False
    )

    # Create a grouped bar graph
    df_plot = dt_results_df.set_index('Category')
    df_plot.plot(kind='bar', rot=0, figsize=(10, 5))

    plt.title(
        'Measure of Model Accuracy vs Loss by Category',
        size=18,
        weight='bold'
    )
    plt.legend(title='Metric')
    plt.xlabel('Prediction Category of Model', size=15, weight='bold')
    plt.ylabel('Percentage (%)', size=15, weight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_folder}/dt_model_results.png', dpi=300)
    plt.close()


def train_models() -> None:
    # Import all csv files in folder
    csv_files: list[str] = glob(f'{HIST_DATA_FOLDER}/*')

    # Select or generate the hash of the unique output destination
    target_folder: str = f'{TRAINING_OUTPUT_FOLDER}/{TRAINING_OUTPUT_CHOICE}'
    if not exists(target_folder) or TRAINING_OUTPUT_CHOICE == '':
        hash_input: str = ''.join(csv_files)
        hash_output: str = sha256(hash_input.encode()).hexdigest()[:16]
        target_folder: str = f'{TRAINING_OUTPUT_FOLDER}/{hash_output}'

    # Save the files used to generate the dataframe
    if not exists(target_folder):
        makedirs(target_folder)

    with open(f'{target_folder}/csv_file_list.txt', 'w') as csv_list_file:
        for csv_file in csv_files:
            csv_list_file.write(f'{csv_file}\n')

    # Get or create the dataframe
    combined_filepath: str = f'{target_folder}/historical_data_combined.csv'
    if exists(combined_filepath):
        combined_df: DataFrame = read_csv(combined_filepath)
    else:
        combined_df: DataFrame = create_csv(target_folder, csv_files)

    # Create a set of encodings to map to columns
    combined_df_encodings: dict[str, dict] = create_encodings(combined_df, target_folder)

    for column in combined_df_encodings.keys():
        column_map: dict = combined_df_encodings[column]
        combined_df[f'{column}_enc'] = combined_df[column].map(column_map)
        del combined_df[column]

    encoded_filepath: str = f'{target_folder}/historical_data_encoded.csv'
    combined_df.to_csv(encoded_filepath, index=False)

    # Train a set decision tree models to predict string categories
    create_dt_models(combined_df, target_folder)
