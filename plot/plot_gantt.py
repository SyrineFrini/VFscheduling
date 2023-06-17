import matplotlib.pyplot as plt

def generate_gantt_chart(solution, I, R, P, T, theta):
    # Create a list of crops
    crops = ["Crop {}".format(i) if i != 19 else "Maint." for i in range(1, I + 1)]

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
                        crop_end = crop_start + crop_length

                        # Check if the crop exceeds the time horizon
                        if crop_end > T:
                            # First bar within the time horizon
                            bar1_length = T - crop_start
                            tower_schedule[r - 1].append((crop_start, bar1_length, crops[i - 1]))

                            # Second bar starting from the beginning
                            bar2_length = crop_length - bar1_length
                            tower_schedule[r - 1].append((0, bar2_length, crops[i - 1]))
                        else:
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

