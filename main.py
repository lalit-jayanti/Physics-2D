import numpy as np

from physics_environment import Environment, Circle


def main():

    objects = [
        Circle(np.random.uniform(low=0.1, high=2),
               np.random.uniform(low=-5, high=5, size=2),
               np.random.uniform(low=0, high=0, size=2))
        for i in range(25)
    ]

    env = Environment(objects=objects, debug=False)
    env.simulate(time_steps=1000, dt=0.01)


if __name__ == '__main__':
    main()
