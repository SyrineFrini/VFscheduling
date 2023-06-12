import numpy as np
import matplotlib.pyplot as plt

def plot_tower_content(solution, I, R, P, T, t):
    # Create a list of crops
    crops = ["Crop {}".format(i) for i in range(1, I + 1)]

    # Create a dictionary to store the content of each tower at the given time step
    tower_content = {}

    for p in range(1, P + 1):
        tower_content[p] = [[] for _ in range(R)]

        # Iterate over the shelves
        for r in range(1, R + 1):
            # Check if a crop is planted at the current position (p, r, t)
            for i in range(1, I + 1):
                if solution.get("X_{}_{}_{}_{}".format(i, p, r, t), 0) == 1:
                    tower_content[p][r - 1].append(crops[i - 1])

    # Plot the tower content
    for p, content in tower_content.items():
        fig, ax = plt.subplots(figsize=(6, 4))
        shelf_labels = ["Shelf {}".format(r + 1) for r in range(R)]
        shelf_heights = np.arange(R)

        for r in range(R):
            crops_planted = content[r]
            if crops_planted:
                crop_colors = [plt.cm.get_cmap('tab20')(crops.index(crop) / I) for crop in crops_planted]
                crop_names = [crop.split()[1] for crop in crops_planted]  # Extract the crop name from "Crop X"
                ax.barh(r, 1, align='center', height=0.5, color=crop_colors)
                for i, name in enumerate(crop_names):
                    ax.text(0.5, r, name, ha='center', va='center', color='white', fontweight='bold')

        ax.set_xticks([0, 1])  # Set x-axis ticks to 0 and 1 only
        ax.set_xticklabels(['0', '1'])  # Set x-axis tick labels to '0' and '1'
        ax.set_ylim(-0.5, R - 0.5)
        ax.set_yticks(shelf_heights)
        ax.set_yticklabels(shelf_labels)
        ax.set_title("Tower {} - Time Step {}".format(p, t))
        ax.grid(True, axis='x')

        plt.show()

def generate_gantt_chart(solution, I, R, P, T, theta):
    # Create a list of crops
    crops = ["Crop {}".format(i) for i in range(1, I + 1)]

    # Create a dictionary to store the job schedule for each tower
    tower_schedules = {}

    # Create a color map to assign colors to crops
    color_map = {crops[i]: plt.cm.get_cmap('tab20')(i / I) for i in range(I)}

    for p in range(1, P + 1):
        # Initialize the schedule for the current tower
        tower_schedule = [[] for _ in range(R)]

        # Iterate over the shelves and time periods
        for r in range(1, R + 1):
            for t in range(1, T + 1):
                # Check if a crop is planted at the current position (p, r, t)
                for i in range(1, I + 1):
                    if solution.get("X_{}_{}_{}_{}".format(i, p, r, t), 0) == 1:
                        crop_length = theta[i]
                        crop_start = ((t) % T)  # Adjust start time if rotation is needed
                        tower_schedule[r - 1].append((crop_start, crop_length, crops[i - 1]))

        # Store the schedule for the current tower
        tower_schedules[p] = tower_schedule

    # Plot the Gantt chart for each tower separately
    for p, tower_schedule in tower_schedules.items():
        fig, ax = plt.subplots(figsize=(10, 5))

        for r in range(R):
            for job in tower_schedule[r]:
                color = color_map[job[2]]
                ax.barh(r, job[1], align='center', height=0.5, left=job[0], color=color)
                ax.text(job[0] + job[1] / 2, r + 0.25, job[2], ha='center', va='center')

            ax.set_yticks(range(R))
            ax.set_yticklabels(["Shelf {}".format(r + 1) for r in range(R)])
            ax.set_xlabel("Time")
            ax.set_title("Tower {}".format(p))

        plt.show()

