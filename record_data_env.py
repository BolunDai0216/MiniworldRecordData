import math
import pickle
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pyglet
from pyglet.window import key


class RecordDataEnv:
    def __init__(self, env, no_time_limit, domain_rand, save_data_filename):
        self.env = env

        if no_time_limit:
            self.env.max_episode_steps = math.inf
        if domain_rand:
            self.env.domain_rand = True

        self.data = None
        self.filename = save_data_filename

    def run(self):
        print("============")
        print("Instructions")
        print("============")
        print("move: arrow keys\npickup: P\ndrop: D\ndone: ENTER\nquit: ESC")
        print("============")

        self.reset()

        # Create the display window
        self.env.render()

        env = self.env

        @env.unwrapped.window.event
        def on_key_press(symbol, modifiers):
            """
            This handler processes keyboard commands that
            control the simulation
            """

            if symbol == key.BACKSPACE or symbol == key.SLASH:
                print("RESET")
                self.env.reset()
                self.env.render()
                return

            if symbol == key.ESCAPE:
                self.env.close()

            if symbol == key.UP:
                self.step(self.env.actions.move_forward)
            elif symbol == key.DOWN:
                self.step(self.env.actions.move_back)
            elif symbol == key.LEFT:
                self.step(self.env.actions.turn_left)
            elif symbol == key.RIGHT:
                self.step(self.env.actions.turn_right)
            elif symbol == key.PAGEUP or symbol == key.P:
                self.step(self.env.actions.pickup)
            elif symbol == key.PAGEDOWN or symbol == key.D:
                self.step(self.env.actions.drop)
            elif symbol == key.ENTER:
                self.step(self.env.actions.done)

        @env.unwrapped.window.event
        def on_key_release(symbol, modifiers):
            pass

        @env.unwrapped.window.event
        def on_draw():
            self.env.render()

        @env.unwrapped.window.event
        def on_close():
            pyglet.app.exit()

        # get top-down view of the environment
        self.env_img, self.scale = self.env.render_top_view(
            self.env.vis_fb, render_agent=False, return_scale=True
        )

        # Enter main event loop
        pyglet.app.run()

        self.env.close()
        self.save_data()

    def reset(self):
        if self.data is not None:
            self.data.append(deepcopy(self.episode_data))
        else:
            self.data = []

        self.env.reset()
        self.episode_data = []
        self.episode_data.append(
            {
                "agent_pos": self.env.agent.pos,
                "agent_dir": self.env.agent.dir,
            }
        )

    def step(self, action):
        print(
            "step {}/{}: {}".format(
                self.env.step_count + 1,
                self.env.max_episode_steps,
                self.env.actions(action).name,
            )
        )

        obs, reward, termination, truncation, info = self.env.step(action)

        self.episode_data.append(
            {
                "agent_pos": self.env.agent.pos,
                "agent_dir": self.env.agent.dir,
            }
        )

        if reward > 0:
            print(f"reward={reward:.2f}")

        if termination or truncation:
            print("done!")
            self.reset()

        self.env.render()

    def save_data(self):
        with open(self.filename, "wb") as handle:
            pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.plot_data()

    def plot_data(self):
        fig, ax = plt.subplots(1, 1)
        ax.imshow(self.env_img)

        x_scale, z_scale = self.scale["x_scale"], self.scale["z_scale"]
        x_offset, z_offset = self.scale["x_offset"], self.scale["z_offset"]

        for i, data in enumerate(self.data):
            _pos = np.array([d["agent_pos"] for d in data if True])
            ax.plot(
                x_scale * _pos[:, 0] + x_offset,
                z_scale * _pos[:, 2] + z_offset,
                color="darkorange",
                alpha=(i + 1) / len(self.data),
                linewidth=3,
            )

        # Turn off x/y axis numbering/ticks
        ax.xaxis.set_ticks_position("none")
        ax.yaxis.set_ticks_position("none")
        _ = ax.set_xticklabels([])
        _ = ax.set_yticklabels([])

        plt.show()

        plt.savefig(
            "imgs/miniworld_record.png",
            dpi=200,
            transparent=False,
            bbox_inches="tight",
        )
