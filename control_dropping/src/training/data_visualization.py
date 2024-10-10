import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import DataLoader
from training.data import get_joint_dataset


# Constants
PATH_DYNAMIX = os.path.join(os.environ["CONTROL_DROP_DIR"], "Data_Collection/Time_Dependent_Samples_4/")
PATH_CRITIQ = os.path.join(os.environ["CONTROL_DROP_DIR"], "Data_Collection/Action_Pred_Time_Dependent_Samples_4/")
BATCH_SIZE = 32
NUM_BATCHES_TO_ANALYZE = 10

def extract_xyz(data, key):
    if key == 'obj_location':
        return data[key].reshape(data[key].size(0), -1, 6)[:, :, :3]
    else:
        return data[key].reshape(data[key].size(0), -1, 3)

def analyze_data_range(loader):
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    z_min, z_max = float('inf'), float('-inf')

    for i, batch in enumerate(loader):
        if i >= NUM_BATCHES_TO_ANALYZE:
            break

        dynamix_data, dynamix_target = batch["dynamix"]
        
        for key in ['finger_1_location', 'finger_2_location', 'finger_3_location', 'palm_location', 'obj_location']:
            xyz_data = extract_xyz(dynamix_target, key)
            x_min = min(x_min, xyz_data[:, :, 0].min().item())
            x_max = max(x_max, xyz_data[:, :, 0].max().item())
            y_min = min(y_min, xyz_data[:, :, 1].min().item())
            y_max = max(y_max, xyz_data[:, :, 1].max().item())
            z_min = min(z_min, xyz_data[:, :, 2].min().item())
            z_max = max(z_max, xyz_data[:, :, 2].max().item())

    return max(abs(x_min), abs(x_max), abs(y_min), abs(y_max), abs(z_min), abs(z_max))

def visualize_3d_data(dynamix_target, plot_range):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    colors = ['r', 'g', 'b', 'c', 'm']
    labels = ['Finger 1', 'Finger 2', 'Finger 3', 'Palm', 'Objects']

    for i, key in enumerate(['finger_1_location', 'finger_2_location', 'finger_3_location', 'palm_location', 'obj_location']):
        xyz_data = extract_xyz(dynamix_target, key)
        for batch_idx in range(xyz_data.size(0)):
            ax.scatter(xyz_data[batch_idx, :, 0], xyz_data[batch_idx, :, 1], xyz_data[batch_idx, :, 2], 
                       c=colors[i], label=labels[i] if batch_idx == 0 else "", s=20)

    ax.set_xlim(-plot_range, plot_range)
    ax.set_ylim(-plot_range, plot_range)
    ax.set_zlim(-plot_range, plot_range)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    plt.title('3D Visualization of Hand and Objects')
    plt.tight_layout()
    plt.show()

def main():
    # Load the dataset
    train_dataset, _ = get_joint_dataset(PATH_DYNAMIX, PATH_CRITIQ, train_test_split=0.98)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Analyze data range
    plot_range = analyze_data_range(train_loader)
    print(f"Determined plot range: {plot_range}")

    # Visualize a few samples
    for i, batch in enumerate(train_loader):
        if i >= 5:  # Visualize 5 samples
            break
        dynamix_data, dynamix_target = batch["dynamix"]
        visualize_3d_data(dynamix_target, plot_range)

if __name__ == "__main__":
    main()
