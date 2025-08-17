# 2D Physics Simulator

## Overview
This is a toy project designed for simulating 2D physics. This project utilizes quadtrees for efficient collision detection and allows for simulating mass-spring-damper systems.

## Key Features
A few examples are provided in ```examples.ipynb``` highlighting the key features of the project.

### Collision detection with Quadtrees

Quadtrees are used to efficiently identify potential neighbors for each object, reducing unnecessary collision checks. This improves the time complexity of collision detection from the naive $O(n^2)$ to $O(n\log{n})$ with respect to the number of objects.

 <table>
  <tr>
    <td> <img src="results/test1_q.gif" title="Quadtree collision detection example 1" style="width: 100%;"/></td>
    <td><img src="results/test1_q_2.gif" title="Quadtree collision detection example 2" style="width: 100%;"/></td>
  </tr>
</table> 

### Simulating mass-spring-damper systems

The simulator also supports springs and dampers. By varying spring and damping constants, one can model different physical behaviors. These can be combined to simulate "soft bodies", as shown in the examples below.

 <table>
  <tr>
    <td> <img src="results/test3.gif" title="Mass-spring-damper systems example 1" style="width: 100%;"/></td>
    <td><img src="results/test3_2.gif" title="Mass-spring-damper systems example 2" style="width: 100%;"/></td>
  </tr>
</table> 

### Physics based collisions

By adjusting the coefficient of restitution, collisions with walls and between objects can be simulated across a spectrum from perfectly elastic to inelastic collisons.

 <table>
  <tr>
    <td> <img src="results/test2_e.gif" title="Perfectly elastic collisions" style="width: 100%;"/></td>
    <td><img src="results/test2_i.gif" title="Inelastic collisions" style="width: 100%;"/></td>
  </tr>
</table> 

## Basic Usage

```python
import numpy as np
from physics_environment import Environment, Circle, Spring, Gravity

# Example of a 'force field' (acts independently on each object).
# Force fields are functions taking an object and modifying its acceleration.
def gravity(obj):
    obj.acc[1] += -10  # downward acceleration
    obj.acc[0] += 0    # no horizontal force

# Create 100 circles with random radii, positions, and velocities
objects = [
    Circle(
        radius=np.random.uniform(low=0.1, high=0.1),
        pos=np.random.uniform(low=-5, high=5, size=2),
        vel=np.random.uniform(low=-0.1, high=0.1, size=2)
    )
    for i in range(100)
]

# Global force fields (can include wind, magnetism, drag, etc.)
force_fields = [gravity]

# Interaction forces ('springs') between objects
# length = rest length, k = spring constant, c = damping
springs = [Spring(objects[0], objects[1], length=1, spring_constant=1, damping_constant=0)]

# Create the simulation environment
env = Environment(
    objects=objects,
    force_fields=force_fields,
    springs=springs,
    wall_restitution_coeff=0.5,
    circle_restitution_coeff=0.3,
    debug=False,        # visualize momentum and energy when True
    display_qtree=False # overlay quadtree visualization when True
)

# Run simulation and save animation
env.save_animation("results/example.gif",
                   time_steps=300, sub_steps=11, dt=0.1, fps=16)

```