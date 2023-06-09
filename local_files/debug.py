import numpy as np
import math
import torch


def debug_math_orients():
    a = np.array([0.1, 0.2, 0.3, 0.4]).reshape(2, 2)
    b = np.array([0.1, 0.2, 0.3, 0.4]).reshape(2, 2)

    # orients = math.atan2(a[:, :], b[:, :]) / math.pi * 180
    orients = np.arctan2(a[:, :], b[:, :]) / np.pi * 180
    print(orients)

    a = np.array([0.1])
    b = np.array([0.2])

    print(math.atan2(a, b)/np.pi * 180)
    print(np.arctan2(a, b)/math.pi * 180)
    pi = torch.acos(torch.zeros(1)).item() * 2

    print(torch.arctan(torch.tensor(a/b))/pi * 180)


    #


if __name__ == "__main__":
    print("Start")
    debug_math_orients()
    print("Emnd")

