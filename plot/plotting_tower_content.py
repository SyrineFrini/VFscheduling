import numpy as np
import matplotlib.pyplot as plt


def generate_tower_content(solution, I, R, P, T, theta):
    # Create a list of crops
    crops = ["Crop {}".format(i) for i in range(1, I + 1)]

    # Create a dictionary to store the job schedule for each tower
    tower_schedules = {}

    for p in range(1, P + 1):
        # Initialize the schedule for the current tower
        tower_schedule = [[-1 if not any(solution.get("X_{}_{}_{}_{}".format(i, p, r + 1, t + 1), 0) == 1 for i in range(1, I + 1)) else [] for t in range(T)] for r in range(R)]

        # Iterate over the shelves, time periods, and crops
        for r in range(R):
            for t in range(T):
                for i in range(1, I + 1):
                    if solution.get("X_{}_{}_{}_{}".format(i, p, r + 1, t + 1), 0) == 1:
                        crop_length = theta[i]
                        crop_start = t % T  # Adjust start time if rotation is needed
                        if crop_start + crop_length <= T:
                            crop_end = crop_start + crop_length
                            # Store the crop in the schedule for the relevant time steps
                            for ts in range(crop_start, crop_end):
                                tower_schedule[r][ts] = i
                        else:
                            crop_end = (crop_start + crop_length) % T
                            for ts in range(crop_start, T):
                                tower_schedule[r][ts] = i
                            for ts in range(crop_end):
                                tower_schedule[r][ts] = i

        # Store the schedule for the current tower
        tower_schedules[p] = tower_schedule

    return tower_schedules

def plot_tower_content(tower_schedules, tower_id, time_step, I, R):
    # Get the schedule for the specified tower
    schedule = tower_schedules[tower_id]
    print(schedule)

    # Get the content at the specified time step for all shelves
    content_at_time = [shelf[time_step - 1] if time_step <= len(shelf) and len(shelf) > 0 else None for shelf in schedule]  # Adjusting for 0-based indexing
    print(content_at_time)
    # Create a list of crops
    crops = ["Crop {}".format(i) if i != 19 else "Maintenance" for i in range(1, I + 1)]

    # Plot the tower content
    fig, ax = plt.subplots(figsize=(6, 4))
    shelf_labels = ["Shelf {}".format(r + 1) for r in range(R)]
    shelf_heights = np.arange(R)

    for r in range(R):
        crop_index = content_at_time[r]
        print(crop_index)# Adjusting for 0-based indexing
        if crop_index >= 0:
            if crop_index == 19:  # Crop index 19 (19 - 1 = 18) corresponds to "Maintenance"
                crop_name = "Maintenance"
            else:
                crop_name = crops[crop_index-1].split()[1]  # Extract the crop name from "Crop X"
            crop_color = plt.cm.get_cmap('tab20')(crop_index / I)
            ax.barh(r, 1, align='center', height=0.5, color=crop_color)
            ax.text(0.5, r, crop_name, ha='center', va='center', color='white', fontweight='bold')

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, R - 0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['0', '1'])
    ax.set_yticks(shelf_heights)
    ax.set_yticklabels(shelf_labels)
    ax.set_title("Tower {} - Time Step {}".format(tower_id, time_step))
    ax.grid(True, axis='x')

    plt.show()
