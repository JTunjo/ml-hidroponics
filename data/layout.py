import os
import json
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from binsolver import BinSolver

# ===============================
# CONFIGURATION
# ===============================

API_KEY = "bs_debf5a55fd7f40caa45823d9f2563eb13f1749e64ebb4e3980e666b2a5deffbb"  # Make sure this is set

CONTAINER = {
    "id": "container-1",
    "w": 40,
    "h": 30,
    "d": 35,
    "quantity": 1
}

BOX = {
    "id": "small-box",
    "w": 19,
    "h": 6,
    "d": 13,
    "quantity": 100  # Set high to maximize packing
}

# ===============================
# PACK REQUEST
# ===============================

def pack_boxes():
    client = BinSolver(api_key="bs_debf5a55fd7f40caa45823d9f2563eb13f1749e64ebb4e3980e666b2a5deffbb")

    request = {
        "allowUnplaced": True,
        "objective": "maxItems",
        "items": [BOX],
        "bins": [CONTAINER],
    }

    response = client.pack(request)
    return response


# ===============================
# PRINT LAYOUT
# ===============================

def print_layout(response):
    print("\n===== PACKING RESULT =====")

    for b in response.bins:
        print(f"\nBIN: {b.id}")
        for item in b.items:
            print(
                f"Item {item.id} "
                f"at ({item.x}, {item.y}, {item.z}) "
                f"size ({item.w}, {item.h}, {item.d})"
            )


# ===============================
# 3D VISUALIZATION
# ===============================

def plot_bin(bin_data, bin_dimensions):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim([0, bin_dimensions["w"]])
    ax.set_ylim([0, bin_dimensions["h"]])
    ax.set_zlim([0, bin_dimensions["d"]])

    ax.set_xlabel("Width (W)")
    ax.set_ylabel("Height (H)")
    ax.set_zlabel("Depth (D)")

    for item in bin_data.items:
        x, y, z = item.x, item.y, item.z
        w, h, d = item.w, item.h, item.d

        # Create cube corners
        corners = [
            [x, y, z],
            [x+w, y, z],
            [x+w, y+h, z],
            [x, y+h, z],
            [x, y, z+d],
            [x+w, y, z+d],
            [x+w, y+h, z+d],
            [x, y+h, z+d],
        ]

        faces = [
            [corners[j] for j in [0,1,2,3]],
            [corners[j] for j in [4,5,6,7]],
            [corners[j] for j in [0,1,5,4]],
            [corners[j] for j in [2,3,7,6]],
            [corners[j] for j in [1,2,6,5]],
            [corners[j] for j in [4,7,3,0]],
        ]

        color = (random.random(), random.random(), random.random())

        ax.add_collection3d(
            Poly3DCollection(faces, alpha=0.5, facecolor=color)
        )

    plt.title("3D Packing Layout")
    plt.tight_layout()
    plt.show()


# ===============================
# MAIN
# ===============================

if __name__ == "__main__":
    response = pack_boxes()

    print_layout(response)

    for b in response.bins:
        plot_bin(b, CONTAINER)