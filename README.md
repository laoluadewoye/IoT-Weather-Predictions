# Welcome

To run this code, you first need to have your own API key from OpenWeatherMap. 

CreateHistorical requires access to their One Call 3.0 API, which is
basically free as long as you make less than 1000 calls per day to the API. 

The calls to the freely available current and forecast routes are not counted 
towards this total but this project counts all calls used throughout the 
script regardless. This means you would get blocked code-side before you 
accidentally go over the One Call limit and have to pay money. 

At most, maybe a couple cents. Check the counter json for the current day in 
the "daliy_api_call_counts" folder to know where you are at. Golden rule is 
that you can run Main.py or CreateHistorical.py no more than two times a day, 
as each would consume around 340 API calls in a full run.

# Instructions

This code was built to run in Python 3.12. To run, first install necessary
packages using "pip install -r requirements.txt". Next, you have four options
to run the code.

Next, check the create_historical_config.toml file. This is the configuration
settings that the project uses to run.

Finally, you have four Python programs that can be run.

* Main.py runs everything from top to bottom.
* CreateHistorical.py runs the script to obtain historical weather data within
  the last (insert time frame you set up using the config toml file).
* TrainModels.py uses the obtained historical weather data to train decision
  tree models.
* PredictWeather.py uses the trained decision tree models to predict future
  weather data.
