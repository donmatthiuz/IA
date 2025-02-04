import matplotlib.pyplot as plt
import numpy as np


def graph_smv(data_X, labels_y, model):
    def get_hyperplane_value(x, weights, bias, offset):
        return (-weights[0] * x + bias + offset) / weights[1]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # Graficar los puntos con colores correspondientes
    scatter = plt.scatter(data_X[:, 0], data_X[:, 1], marker="o", c=labels_y, cmap="bwr", edgecolors="k")
    
    # Crear la frontera de decisión
    x_min = np.amin(data_X[:, 0])
    x_max = np.amax(data_X[:, 0])

    y_min = get_hyperplane_value(x_min, model.w, model.b, 0)
    y_max = get_hyperplane_value(x_max, model.w, model.b, 0)

    y_min_m = get_hyperplane_value(x_min, model.w, model.b, -1)
    y_max_m = get_hyperplane_value(x_max, model.w, model.b, -1)

    y_min_p = get_hyperplane_value(x_min, model.w, model.b, 1)
    y_max_p = get_hyperplane_value(x_max, model.w, model.b, 1)

    ax.plot([x_min, x_max], [y_min, y_max], "y--", label="Hiperplano de decisión")
    ax.plot([x_min, x_max], [y_min_m, y_max_m], "k", label="Margen -1")
    ax.plot([x_min, x_max], [y_min_p, y_max_p], "k", label="Margen +1")

    y_range_min = np.amin(data_X[:, 1])
    y_range_max = np.amax(data_X[:, 1])
    ax.set_ylim([y_range_min - 3, y_range_max + 3])

    ax.set_xlabel("blueAvgLevel")
    ax.set_ylabel("blueKills")
    ax.set_title("SVM - Frontera de decisión")

    # Agregar leyenda para los colores de los puntos
    legend_labels = {1: "Blue Wins", -1: "Blue Loose"}
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor="red", markersize=8, label="Blue Wins"),
               plt.Line2D([0], [0], marker='o', color='w', markerfacecolor="blue", markersize=8, label="Blue Loose")]

    ax.legend(handles=handles, loc="upper right")

    plt.show()
