import matplotlib.pyplot as plt
import numpy as np

def plot_tower_content_with_height(tower_schedules, tower_id, time_step, I, R, G, Z):
    # Get the schedule for the specified tower
    schedule = tower_schedules[tower_id]

    # Get the content at the specified time step for all shelves
    content_at_time = [shelf[time_step - 1] if time_step <= len(shelf) and len(shelf) > 0 else None for shelf in schedule]

    # Create a list of crops
    crops = ["Crop {}".format(i) if i != 19 else "Maintenance" for i in range(1, I + 1)]

    # Plot the tower content
    fig, ax = plt.subplots(figsize=(6, 4))
    shelf_labels = ["Shelf {}".format(r + 1) for r in range(R)]
    shelf_heights = np.array([G[tower_id, r+1] for r in range(R)])  # Convert to numpy array
    shelf_cumulative_heights = np.cumsum(shelf_heights)

    # Calculate the bottom positions of the y-ticks
    y_ticks_bottom = np.concatenate(([0], shelf_cumulative_heights[:-1]))

    for r in range(R):
        crop_index = content_at_time[r]
        if crop_index is not None and crop_index >= 0:
            crop_height = Z[crop_index]
            if crop_index == 19:  # Crop index 19 (19 - 1 = 18) corresponds to "Maintenance"
                crop_name = "Maintenance"
                crop_height = shelf_heights[r]
            else:
                crop_name = crops[crop_index-1].split()[1]  # Extract the crop name from "Crop X"
            crop_bottom = shelf_cumulative_heights[r] - crop_height - shelf_heights[r]  # Position crop at the bottom of the shelf
            crop_color = plt.cm.get_cmap('tab20')(crop_index / I)
            ax.barh(crop_bottom, 1, align='center', height=crop_height, left=0, color=crop_color)
            ax.text(0.5, crop_bottom + crop_height / 2, crop_name, ha='center', va='center', color='white', fontweight='bold')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, sum(shelf_heights))
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['0', '1'])
    ax.set_yticks(shelf_cumulative_heights - shelf_heights)  # Set y-tick positions
    ax.set_yticklabels(shelf_labels)
    ax.set_title("Tower {} - Time Step {}".format(tower_id, time_step))
    ax.grid(True, axis='x')

    plt.show()