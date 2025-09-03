import numpy as np
from config import CarParams

def get_cost_matrices(car_params: CarParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns the cost matrices Q, R and Q_e for the given race car."""
    Q = np.diag(
        [
            car_params.L11.item() ** 2,
            car_params.L22.item() ** 2,
            car_params.L33.item() ** 2,
            car_params.L44.item() ** 2,
            car_params.L55.item() ** 2,
            car_params.L66.item() ** 2,
        ]
    )

    R = np.diag([car_params.L77.item() ** 2, car_params.L88.item() ** 2])

    Q_e = np.diag(
        [
            car_params.LN11.item() ** 2,
            car_params.LN22.item() ** 2,
            car_params.LN33.item() ** 2,
            car_params.LN44.item() ** 2,
            car_params.LN55.item() ** 2,
            car_params.LN66.item() ** 2,
        ]
    )

    return Q, R, Q_e


if __name__ == "__main__":
    from config import get_default_car_params
    car_params = get_default_car_params()
    Q, R, Q_e = get_cost_matrices(car_params)
    print("Cost matrices:")
    print("Q:", Q)
    print("R:", R)
    print("Q_e:", Q_e)