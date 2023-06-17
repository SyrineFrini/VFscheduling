import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def animate_tower_content(tower_schedules, tower_id, I, R):
    # Get the schedule for the specified tower
    schedule = tower_schedules[tower_id]

    # Create a list of crops
    crops = ["Crop {}".format(i) if i != 19 else "Maintenance" for i in range(1, I + 1)]

    # Get the number of time steps
    T = len(schedule[0])

    fig, ax = plt.subplots(figsize=(6, 4))
    shelf_labels = ["Shelf {}".format(r + 1) for r in range(R)]
    shelf_heights = np.arange(R)

    def update_content(step):
        ax.clear()

        # Get the content at the specified time step for all shelves
        content_at_time = [shelf[step] for shelf in schedule]  # Adjusting for 0-based indexing

        for r in range(R):
            crop_index = content_at_time[r] - 1  # Adjusting for 0-based indexing
            if crop_index >= 0:
                if crop_index == 18:  # Crop index 19 (19 - 1 = 18) corresponds to "Maintenance"
                    crop_name = "Maintenance"
                else:
                    crop_name = crops[crop_index].split()[1]  # Extract the crop name from "Crop X"
                crop_color = plt.cm.get_cmap('tab20')(crop_index / I)
                ax.barh(r, 1, align='center', height=0.5, color=crop_color)
                ax.text(0.5, r, crop_name, ha='center', va='center', color='white', fontweight='bold')

        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, R - 0.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['0', '1'])
        ax.set_yticks(shelf_heights)
        ax.set_yticklabels(shelf_labels)
        ax.set_title("Tower {} - Time Step {}".format(tower_id, step + 1))
        ax.grid(True, axis='x')

    ani = animation.FuncAnimation(fig, update_content, frames=T, interval=1000, repeat=True)
    plt.show()
