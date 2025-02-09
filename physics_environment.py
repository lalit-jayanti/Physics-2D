import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
import time

from quadtree import quadTree, boundingBox, Node


def calc_time(func):

    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f'{func.__name__} \t: {end-start} s')
        return result

    return wrapper


class Spring:

    def __init__(self, obj1, obj2,
                 length=1,
                 spring_constant=1,
                 damping_constant=0):

        self.obj1 = obj1
        self.obj2 = obj2
        self.length = length
        self.spring_constant = spring_constant
        self.damping_constant = damping_constant

    def apply(self):

        direction = self.obj2.pos-self.obj1.pos
        distance = np.linalg.norm(direction)
        if (distance == 0):
            return

        direction /= distance

        elongation = distance - self.length

        self.obj1.acc += (
            + self.spring_constant*direction*elongation
            - self.damping_constant*direction
            * np.dot(self.obj1.vel-self.obj2.vel, direction)
        )/self.obj1.mass

        self.obj2.acc += (
            - self.spring_constant*direction*elongation
            - self.damping_constant*direction
            * np.dot(self.obj2.vel-self.obj1.vel, direction)
        )/self.obj2.mass


class Circle:

    def __init__(self, radius=1, pos=[0, 0], vel=[0, 0], acc=[0, 0]):

        self.radius = radius
        self.mass = np.pi*(self.radius**2)

        self.pos = np.array(pos, dtype=np.float64)
        self.vel = np.array(vel, dtype=np.float64)
        self.acc = np.array(acc, dtype=np.float64)


class Environment:

    def __init__(self, objects,
                 force_fields=[],
                 springs=[],
                 wall_restitution_coeff=0.8,
                 circle_restitution_coeff=0.99,
                 display_qtree=False,
                 debug=False):

        self.objects = objects

        self.max_obj_radius = max(self.objects, key=lambda x: x.radius).radius
        self.qtree = None

        self.debug = debug
        self.display_qtree = display_qtree
        self.cmap = cm.get_cmap("plasma")

        self.radius = 10
        self.force_fields = force_fields
        self.springs = springs

        self.wall_restitution_coeff = wall_restitution_coeff   # wall-circle collision
        self.circle_restitution_coeff = circle_restitution_coeff  # circle-circle collison

    def make_qtree(self):
        self.qtree = quadTree(boundary=boundingBox(0, 0, self.radius))
        for obj in self.objects:
            self.qtree.insert(Node(obj.pos[0], obj.pos[1], obj))

    def apply_forces(self):

        for field in self.force_fields:
            for obj in self.objects:
                field(obj)

        for spring in self.springs:
            spring.apply()

    def step(self, dt=0.01):

        for obj in self.objects:
            obj.vel += obj.acc*dt
            obj.pos += obj.vel*dt
            obj.acc[0], obj.acc[1] = 0, 0

    def collision_vel(self, m1, m2, v1, v2):
        return (self.circle_restitution_coeff*m2*(v2-v1)+m1*v1+m2*v2)/(m1+m2)

    def handle_circle_circle(self, obj1, obj2):

        rad_pos = (np.linalg.norm(obj2.pos-obj1.pos))

        diff = obj1.radius + obj2.radius - rad_pos

        if diff < 0:
            return

        center_line = (obj2.pos-obj1.pos)/rad_pos

        v1 = np.dot(obj1.vel, center_line)*center_line
        v2 = np.dot(obj2.vel, center_line)*center_line

        obj1.pos -= (
            diff*center_line * (obj2.mass)/(obj1.mass + obj2.mass)
        )
        obj2.pos += (
            diff*center_line * (obj1.mass)/(obj1.mass + obj2.mass)
        )

        if (np.dot((v2-v1), center_line) > 0):
            return

        obj1.vel -= v1
        obj2.vel -= v2

        obj1.vel += self.collision_vel(obj1.mass, obj2.mass, v1, v2)
        obj2.vel += self.collision_vel(obj2.mass, obj1.mass, v2, v1)

    def brute_handle_collisions(self):

        for obj in self.objects:
            rad_pos = (obj.pos[0]**2 + obj.pos[1]**2)**0.5 + obj.radius
            if rad_pos > self.radius:
                obj.pos *= self.radius/rad_pos

                wall_normal = -obj.pos/np.linalg.norm(obj.pos)
                normal_vel = wall_normal*np.dot(obj.vel, wall_normal)
                obj.vel -= normal_vel
                obj.vel += -self.wall_restitution_coeff*normal_vel

        for i in range(len(self.objects)):
            obj1 = self.objects[i]
            for j in range(i+1, len(self.objects)):
                obj2 = self.objects[j]
                self.handle_circle_circle(obj1, obj2)

    def quad_tree_collisions(self):

        epsilon = 1e-2  # tolerance for quadtree query

        self.make_qtree()

        for obj in self.objects:
            rad_pos = (obj.pos[0]**2 + obj.pos[1]**2)**0.5 + obj.radius
            if rad_pos > self.radius:
                obj.pos *= self.radius/rad_pos

                wall_normal = -obj.pos/np.linalg.norm(obj.pos)
                if (np.dot(obj.vel, wall_normal) >= 0):
                    return

                normal_vel = wall_normal*np.dot(obj.vel, wall_normal)
                obj.vel -= normal_vel
                obj.vel += -self.wall_restitution_coeff*normal_vel

        for i in range(len(self.objects)):
            obj1 = self.objects[i]
            points = self.qtree.queryRange(box=boundingBox(
                x_center=obj1.pos[0],
                y_center=obj1.pos[1],
                half_dimension=obj1.radius+self.max_obj_radius+epsilon
            ))

            for point in points:
                obj2 = point.data

                if obj2 is obj1:
                    continue

                self.handle_circle_circle(obj1, obj2)

    def handle_collisions(self):
        self.quad_tree_collisions()
        # self.brute_handle_collisions()

    def render(self, t):

        for i, obj in enumerate(self.objects):
            # self.patches[0] is reserved for the background
            self.patches[i+1].center = obj.pos[0], obj.pos[1]

        for i, spring in enumerate(self.springs):
            self.lines[i].set_data([spring.obj1.pos[0], spring.obj2.pos[0]],
                                   [spring.obj1.pos[1], spring.obj2.pos[1]])

        if self.debug:

            phy_qty = self.compute_physical_quantities()

            self.phy_ax[0].plot(t, phy_qty['momentum'][0], "ro")
            self.phy_ax[0].plot(t, phy_qty['momentum'][1], "go")
            self.phy_ax[0].plot(t, np.linalg.norm(phy_qty['momentum']), "bo")

            self.phy_ax[1].plot(t, phy_qty['kinetic_energy'], "bo")

        if self.display_qtree:

            if self.qtree is not None:
                for patch in self.sim_ax.patches:
                    if isinstance(patch, patches.Rectangle):
                        patch.remove()
                self.qtree.visualize(self.sim_ax)

    def setup_rendering(self):

        plt.clf()
        plt.cla()

        if self.debug:

            self.fig = plt.figure(figsize=(8, 4), constrained_layout=True)
            self.gs = self.fig.add_gridspec(2, 2)

            self.sim_ax = self.fig.add_subplot(self.gs[:, 0])

            self.phy_ax = [self.fig.add_subplot(self.gs[0, 1]),
                           self.fig.add_subplot(self.gs[1, 1])]

            self.phy_ax[0].set_xlabel('Time s')
            self.phy_ax[0].set_ylabel('Momentum kg m/s')

            self.phy_ax[1].set_xlabel('Time s')
            self.phy_ax[1].set_ylabel('Kinetic Energy J')

            for ax in self.phy_ax:
                ax.grid('on')

        else:
            self.fig, self.sim_ax = plt.subplots(figsize=(4, 4))
            self.sim_ax.set_axis_off()

        self.sim_ax.set_xlim([-self.radius, self.radius])
        self.sim_ax.set_ylim([-self.radius, self.radius])
        self.sim_ax.set_aspect('equal')

        self.patches = [plt.Circle((0, 0), self.radius, color='black')]
        self.lines = []

        for obj in self.objects:
            self.patches.append(plt.Circle(obj.pos, obj.radius,
                                           color=self.cmap(np.random.uniform())))

        for spring in self.springs:
            self.lines.append(plt.Line2D([spring.obj1.pos[0], spring.obj2.pos[0]],
                                         [spring.obj1.pos[1], spring.obj2.pos[1]],
                                         color=self.cmap(np.random.uniform()),
                                         linewidth=2))

        for patch in self.patches:
            self.sim_ax.add_patch(patch)

        for line in self.lines:
            self.sim_ax.add_line(line)

        plt.tight_layout()

    def simulate(self, time_steps=100, sub_steps=1, dt=0.01):

        self.setup_rendering()

        for t in range(time_steps):

            self.apply_forces()

            for _ in range(sub_steps):
                self.step(dt/sub_steps)
                self.handle_collisions()

            self.render(t)

            plt.pause(0.001)

    def save_animation(self, fname, time_steps=100, sub_steps=1, dt=0.01, fps=16):

        self.setup_rendering()

        def update(frame):
            self.apply_forces()

            for _ in range(sub_steps):
                self.step(dt/sub_steps)
                self.handle_collisions()

            self.render(frame)

        animation = FuncAnimation(self.fig, update, frames=time_steps)
        gif_writer = PillowWriter(fps=fps)
        animation.save(fname, writer=gif_writer)

    def compute_physical_quantities(self):

        kinetic_energy = 0
        total_momentum = 0

        for obj in self.objects:

            kinetic_energy += 0.5*obj.mass*np.linalg.norm(obj.vel)**2
            total_momentum += obj.mass*obj.vel

        return {"kinetic_energy": kinetic_energy, "momentum": total_momentum}
