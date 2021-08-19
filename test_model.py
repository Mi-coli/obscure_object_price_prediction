from model import predict
import pandas as pd

def data():
    return {
        "material": "ABS-M30",
        "quantity": 2,
        "complexity_score": 8.951051,
        "surface_area": '{"units": "in^2/", "value": 30.86906387028813}',
        "bounding_box_volume": '{"units": "in^3/", "value": 21.753358840942383}',
        "volume": '{"units": "in^3/", "value": 2.392122115044577}',
        "max_x_length": '{"units": "in", "value": 7.431044101715088}',
        "max_y_length": '{"units": "in", "value": 2.342428207397461}',
        "max_z_length": '{"units": "in", "value": 1.2497128248214722}',
        "optimal_fit_on_hp_build_plate": 1046,
    }


print(predict(pd.json_normalize(data())))

