from CreateHistorical import create_historical
from TrainModels import train_models
from PredictWeather import make_predictions

if __name__ == '__main__':
    create_historical()
    train_models()
    make_predictions()
