import numpy as np

from physics_environment import Environment, Circle, Spring


def gravity(obj):

    obj.acc[1] += -10
    obj.acc[0] += 0


def main():

    # Example 1

    # objects = [
    #     Circle(np.random.uniform(low=0.2, high=0.2),
    #            np.random.uniform(low=-5, high=5, size=2),
    #            np.random.uniform(low=0, high=0, size=2))
    #     for i in range(200)
    # ]

    # force_fields = [gravity]

    # env = Environment(objects=objects,
    #                   force_fields=force_fields,
    #                   debug=False)

    # env.simulate(time_steps=1000, sub_steps=3, dt=0.01)

    # Example 2

    objects = [Circle(2,  [5, 5],  [-1, 1]),
               Circle(2, [-5, -5], [1, -1])]

    springs = [Spring(objects[0], objects[1], 0.1, 100, 2)]

    env = Environment(objects=objects,
                      springs=springs,
                      debug=False)

    env.simulate(time_steps=3000, sub_steps=3, dt=0.01)

    # Example 3

    # objects = [Circle(1,  [2, 0],  [0, 0]),
    #            Circle(1,  [2*np.cos(2*np.pi/3), 2*np.sin(2*np.pi/3)],  [0, 0]),
    #            Circle(1, [2*np.cos(-2*np.pi/3), 2*np.sin(-2*np.pi/3)],  [0, 0]),
    #            Circle(2, [0,6])]

    # springs = [Spring(objects[0], objects[2], np.linalg.norm(objects[0].pos-objects[2].pos), 100000, 10),
    #            Spring(objects[1], objects[2], np.linalg.norm(objects[1].pos-objects[2].pos), 100000, 10),
    #            Spring(objects[0], objects[1], np.linalg.norm(objects[0].pos-objects[1].pos), 100000, 10)]

    # force_fields = [gravity]

    # env = Environment(objects=objects,
    #                   springs=springs,
    #                   force_fields = force_fields,
    #                   debug=False)

    # env.simulate(time_steps=3000, sub_steps=3, dt=0.01)


if __name__ == '__main__':
    main()
