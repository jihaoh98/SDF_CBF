import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Initial state and gradient field
robot_init_state = [0, 0]
gradientField = [2, 3]  # Direction of the arrow

# Desired arrow length and width
arrow_length = 2
arrow_width = 0.1

# Create a figure and axis
fig, ax = plt.subplots()

# Create a FancyArrow with adjusted length and width
robot_arrow = mpatches.FancyArrow(
    robot_init_state[0],
    robot_init_state[1],
    gradientField[0] * 0.1,
    gradientField[1] * 0.1,
    width=0.05,
    color='k',
)

# Add the arrow to the plot
ax.add_patch(robot_arrow)

# Set the limits of the plot
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)

# Display the plot
plt.show()
