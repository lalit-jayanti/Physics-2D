import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class Circle:

    def __init__(self, radius=1, pos=[0, 0], vel=[0, 0], acc=[0, 0]):

        self.radius = radius
        self.mass = np.pi*(self.radius**2)

        self.pos = np.array(pos, dtype=np.float64)
        self.vel = np.array(vel, dtype=np.float64)
        self.acc = np.array(acc, dtype=np.float64)


class Environment:

    def __init__(self):

        self.objects = [Circle(np.random.uniform(low=0.3, high=0.3),
                               np.random.uniform(low=-5, high=5, size=2),
                               np.random.uniform(low=-20, high=20, size=2))
                        for i in range(200)]

        self.radius = 10

    def apply_forces(self):

        for obj in self.objects:
            obj.acc[1] = -30

    def step(self, dt=0.01):

        for obj in self.objects:
            obj.vel += obj.acc*dt
            obj.pos += obj.vel*dt
            obj.acc[0], obj.acc[1] = 0, 0

    def collision_vel(self, m1, m2, v1, v2):

        return ((m1-m2)*v1 + 2*m2*v2)/(m1+m2)

    def handle_collisions(self):

        for obj in self.objects:
            rad_pos = (obj.pos[0]**2 + obj.pos[1]**2)**0.5 + obj.radius
            if rad_pos > self.radius:
                obj.pos *= self.radius/rad_pos
                obj.vel *= -1

        for i in range(len(self.objects)):
            obj1 = self.objects[i]
            for j in range(i+1, len(self.objects)):
                obj2 = self.objects[j]

                rad_pos = (np.linalg.norm(obj2.pos-obj1.pos))

                diff = obj1.radius + obj2.radius - rad_pos

                if diff < 0:
                    continue

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
                    continue

                obj1.vel -= v1
                obj2.vel -= v2

                obj1.vel += self.collision_vel(obj1.mass, obj2.mass, v1, v2)
                obj2.vel += self.collision_vel(obj2.mass, obj1.mass, v2, v1)

    def render(self):

        for i, obj in enumerate(self.objects):
            self.patches[i+1].center = obj.pos[0], obj.pos[1]

    def setup_rendering(self):

        self.fig, self.ax = plt.subplots(nrows=2, ncols=1,
                                         height_ratios=[2, 1])
        self.ax[0].set_xlim([-11, 11])
        self.ax[0].set_ylim([-11, 11])
        self.ax[0].set_aspect('equal')

        cmap = cm.get_cmap("hsv")

        self.patches = [plt.Circle((0, 0), self.radius, color='black')]
        for obj in self.objects:
            self.patches.append(plt.Circle(obj.pos, obj.radius,
                                           color=cmap(np.random.uniform())))

        for patch in self.patches:
            self.ax[0].add_patch(patch)

    def simulate(self, time_steps=100, dt=0.01):

        self.setup_rendering()

        for t in range(time_steps):

            self.apply_forces()
            self.step(dt)
            self.handle_collisions()

            self.render()

            phy_qty = self.compute_physical_quantities()
            # self.ax[1].plot(t, phy_qty['momentum'][0], "ro")
            # self.ax[1].plot(t, phy_qty['momentum'][1], "go")
            # self.ax[1].plot(t, np.linalg.norm(phy_qty['momentum']), "bo")
            self.ax[1].plot(t, phy_qty['kinetic_energy'], "bo")


            plt.pause(0.001)

    def compute_physical_quantities(self):

        kinetic_energy = 0
        total_momentum = 0

        for obj in self.objects:

            kinetic_energy += 0.5*obj.mass*np.linalg.norm(obj.vel)**2
            total_momentum += obj.mass*obj.vel

        return {"kinetic_energy": kinetic_energy, "momentum": total_momentum}


def main():

    env = Environment()
    env.simulate(time_steps=1000, dt=0.01)


if __name__ == '__main__':
    main()
