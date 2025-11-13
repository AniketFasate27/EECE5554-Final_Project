# Generate system architecture diagram
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Hardware Layer
hw_box = FancyBboxPatch((0.5, 7), 2, 1.5, boxstyle="round,pad=0.1", 
                         edgecolor='blue', facecolor='lightblue', linewidth=2)
ax.add_patch(hw_box)
ax.text(1.5, 7.75, 'Hardware\nESP32 + MPU6050', ha='center', va='center', 
        fontsize=10, weight='bold')

# FreeRTOS Layer
rtos_box = FancyBboxPatch((3, 7), 2, 1.5, boxstyle="round,pad=0.1", 
                          edgecolor='green', facecolor='lightgreen', linewidth=2)
ax.add_patch(rtos_box)
ax.text(4, 7.75, 'FreeRTOS\nDual-Core\nProcessing', ha='center', va='center', 
        fontsize=10, weight='bold')

# Data Collection
data_box = FancyBboxPatch((5.5, 7), 2, 1.5, boxstyle="round,pad=0.1", 
                          edgecolor='orange', facecolor='lightyellow', linewidth=2)
ax.add_patch(data_box)
ax.text(6.5, 7.75, 'Data Collection\nCSV Logging\n100 Hz', ha='center', va='center', 
        fontsize=10, weight='bold')

# ML Layer
ml_box = FancyBboxPatch((8, 7), 2, 1.5, boxstyle="round,pad=0.1", 
                        edgecolor='red', facecolor='lightcoral', linewidth=2)
ax.add_patch(ml_box)
ax.text(9, 7.75, 'Machine Learning\nFault Detection\n95% Accuracy', ha='center', 
        va='center', fontsize=10, weight='bold')

# Arrows
arrow1 = FancyArrowPatch((2.5, 7.75), (3, 7.75), arrowstyle='->', 
                         mutation_scale=20, linewidth=2, color='black')
arrow2 = FancyArrowPatch((5, 7.75), (5.5, 7.75), arrowstyle='->', 
                         mutation_scale=20, linewidth=2, color='black')
arrow3 = FancyArrowPatch((7.5, 7.75), (8, 7.75), arrowstyle='->', 
                         mutation_scale=20, linewidth=2, color='black')
ax.add_patch(arrow1)
ax.add_patch(arrow2)
ax.add_patch(arrow3)

# Core 0 Tasks
core0_box = FancyBboxPatch((1, 4.5), 3.5, 1.8, boxstyle="round,pad=0.1", 
                           edgecolor='purple', facecolor='lavender', linewidth=2)
ax.add_patch(core0_box)
ax.text(2.75, 5.8, 'Core 0 (Real-Time)', ha='center', va='center', 
        fontsize=11, weight='bold')
ax.text(2.75, 5.2, 'Data Acquisition\n100 Hz Sampling\nPriority: 3', 
        ha='center', va='center', fontsize=9)

# Core 1 Tasks
core1_box = FancyBboxPatch((5.5, 4.5), 3.5, 1.8, boxstyle="round,pad=0.1", 
                           edgecolor='teal', facecolor='lightcyan', linewidth=2)
ax.add_patch(core1_box)
ax.text(7.25, 5.8, 'Core 1 (Processing)', ha='center', va='center', 
        fontsize=11, weight='bold')
ax.text(7.25, 5.2, 'Moving Average Filter\nData Output\nPriority: 1-2', 
        ha='center', va='center', fontsize=9)

# Fault Types
fault_box = FancyBboxPatch((2, 1.5), 6, 2, boxstyle="round,pad=0.1", 
                           edgecolor='darkred', facecolor='mistyrose', linewidth=2)
ax.add_patch(fault_box)
ax.text(5, 3, 'Detectable Faults', ha='center', va='center', 
        fontsize=12, weight='bold')
ax.text(5, 2.3, 'Imbalance | Misalignment | Bearing Defects | Looseness', 
        ha='center', va='center', fontsize=9)

plt.title('Smart Motor Health Diagnostics System Architecture', 
          fontsize=16, weight='bold', pad=20)
plt.tight_layout()
plt.savefig('presentation_architecture.png', dpi=300, bbox_inches='tight')
plt.show()