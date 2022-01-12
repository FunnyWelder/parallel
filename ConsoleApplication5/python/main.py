import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import __db as db


class end_error(Exception):
    pass


def histoogramm(data):
    plt.subplot(2, 1, 1)
    plt.plot(data["time_ms"], linewidth=2.0)
    plt.ylabel("Время")
    plt.grid(True)
    plt.title(data["name"])

    plt.subplot(2, 1, 2)
    plt.plot(data["speed"], linewidth=2.0)
    plt.xlabel("Число ядер")
    plt.ylabel("Ускорение")
    plt.grid(True)
    
    plt.subplots_adjust(hspace = .05)

    plt.show()

if __name__ == "__main__":
    data1 = db.read("integrate_aligned.json", "graphics")
    histoogramm(data1)
    data1 = db.read("integrate_cpp.json", "graphics")
    histoogramm(data1)
    data1 = db.read("integrate_cpp_mtx.json", "graphics")
    histoogramm(data1)
    data1 = db.read("integrate_cpp_reduction.json", "graphics")
    histoogramm(data1)
    data1 = db.read("integrate_crit.json", "graphics")
    histoogramm(data1)
    data1 = db.read("integrate_false_sharing.json", "graphics")
    histoogramm(data1)
    data1 = db.read("integrate_omp_for.json", "graphics")
    histoogramm(data1)
    data1 = db.read("integrate_reduce.json", "graphics")
    histoogramm(data1)
    data1 = db.read("integrate_reduction.json", "graphics")
    histoogramm(data1)