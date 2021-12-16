import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import __db as db


class end_error(Exception):
    pass


def histoogramm(data):
    plt.subplot(3, 1, 1)
    plt.plot(data["time_ms"], linewidth=2.0)
    plt.xlabel("Время")
    plt.ylabel("Число ядер")
    plt.grid(True)
    plt.title(data["name"])

    plt.subplot(3, 1, 3)
    plt.plot(data["speed"], linewidth=2.0)
    plt.xlabel("Скорость")
    plt.ylabel("Число ядер")
    plt.grid(True)

    plt.show()

if __name__ == "__main__":
    data1 = db.read("1.json", "graphics")
    histoogramm(data1)
