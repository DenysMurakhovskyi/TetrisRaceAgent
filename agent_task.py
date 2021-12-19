import gym
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import warnings

warnings.filterwarnings("ignore")
from QMemory import QState, QMemory, Track

team_name = 'lab_ml'  # TODO: change your teamname / username
folder = 'tetris_race_qlearning'
env_name = 'TetrisRace-v0'  # do not change this


# ===== UNMUTABLE CONTENT =====
class EpisodeHistory:
    def __init__(self, env,
                 learn_rate,
                 discount,
                 capacity,
                 plot_episode_count=200,
                 max_timesteps_per_episode=200,
                 goal_avg_episode_length=195,
                 goal_consecutive_episodes=100
                 ):
        self.lengths = np.zeros(capacity, dtype=int)
        self.plot_episode_count = plot_episode_count
        self.max_timesteps_per_episode = max_timesteps_per_episode
        self.goal_avg_episode_length = goal_avg_episode_length
        self.goal_consecutive_episodes = goal_consecutive_episodes

        self.lr = learn_rate
        self.df = discount

        self.lvl_step = env.unwrapped.walls_per_level
        self.lvl_num = env.unwrapped.levels
        self.difficulty = env.unwrapped.level_difficulty
        self.point_plot = None
        self.mean_plot = None
        self.level_plots = []
        self.fig = None
        self.ax = None

    def __getitem__(self, episode_index):
        return self.lengths[episode_index]

    def __setitem__(self, episode_index, episode_length):
        self.lengths[episode_index] = episode_length

    def create_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(14, 7), facecolor='w', edgecolor='k')
        self.fig.canvas.set_window_title("Episode Length History. Team {}".format(team_name))

        self.ax.set_xlim(0, self.plot_episode_count + 5)
        self.ax.set_ylim(0, self.max_timesteps_per_episode + 5)
        self.ax.yaxis.grid(True)
        self.ax.set_title("Episode Length History (lr {}, df {})".format(self.lr, self.df))
        self.ax.set_xlabel("Episode #")
        self.ax.set_ylabel("Length, timesteps")
        self.point_plot, = plt.plot([], [], linewidth=2.0, c="#1d619b")
        self.mean_plot, = plt.plot([], [], linewidth=3.0, c="#df3930")
        for i in range(0, self.lvl_num):
            self.level_plots.append(plt.plot([], [], linewidth=1.0, c="#207232", ls='--'))

    def update_plot(self, episode_index):
        plot_right_edge = episode_index
        plot_left_edge = max(0, plot_right_edge - self.plot_episode_count)

        # Update point plot.
        x = range(plot_left_edge, plot_right_edge)
        y = self.lengths[plot_left_edge:plot_right_edge]
        self.point_plot.set_xdata(x)
        self.point_plot.set_ydata(y)
        self.ax.set_xlim(plot_left_edge, plot_left_edge + self.plot_episode_count)

        # Update levels plots
        for i in range(1, self.lvl_num + 1):
            xl = range(plot_left_edge, plot_right_edge)
            yl = np.zeros(len(xl))
            yl[:] = i * self.lvl_step
            cur_lvl_curve = self.level_plots[i - 1][0]
            cur_lvl_curve.set_xdata(xl)
            cur_lvl_curve.set_ydata(yl)
            self.ax.set_xlim(plot_left_edge, plot_left_edge + self.plot_episode_count)

        # Update rolling mean plot.
        mean_kernel_size = 101
        rolling_mean_data = np.concatenate((np.zeros(mean_kernel_size), self.lengths[plot_left_edge:episode_index]))
        rolling_mean_data = pd.DataFrame(rolling_mean_data)
        rolling_means = rolling_mean_data.rolling(mean_kernel_size).mean()[mean_kernel_size:]
        self.mean_plot.set_xdata(range(plot_left_edge, plot_left_edge + len(rolling_means)))
        self.mean_plot.set_ydata(rolling_means)

        # Repaint the surface.
        plt.draw()
        plt.pause(0.0001)

    def is_goal_reached(self, episode_index):
        ''' DO NOT CHANGE THIS FUNCTION CODE.'''
        # From here agent will receive signal about end of learning
        arr = self.lengths[episode_index - self.goal_consecutive_episodes + 1:episode_index + 1]
        avg = np.average(arr)
        if self.difficulty == 'Easy':
            answer = avg >= self.goal_avg_episode_length + 0.5
        elif len(arr) > 0:
            density = 2 * np.max(arr) * np.min(arr) / (np.max(arr) + np.min(arr))
            answer = avg >= self.goal_avg_episode_length + 0.5 and density >= avg

        return answer


# ==== MUTABLE CONTENT =====
def indexEpisode():
    try:
        indexEpisode.a += 1
    except AttributeError:
        indexEpisode.a = 0
    return indexEpisode.a


class Controler:
    def __init__(self, episodes_num=10000, global_env=None):
        if global_env is None:
            global_env = []
        self.team_name = team_name
        self.exp_dir = folder + '/' + self.team_name
        random_state = 0
        self.agent_history = []
        self.history_f = True
        self.window = 50
        self.episode_index = 0

        env = gym.make(env_name)
        env.seed(random_state)
        np.random.seed(random_state)
        lr, df, exr, exrd = 10, 10, 10, 10

        self.env = gym.wrappers.Monitor(env, self.exp_dir + '/video', force=True, resume=False,
                                        video_callable=self.video_callable)

        episode_history, end_index = self.run_agent(lr, df, exr, exrd, self.env)

    def run_agent(self, rate, factor, exploration, exp_decay, env, verbose=False):
        max_episodes_to_run = env.unwrapped.total_episodes
        max_timesteps_per_episode = env.unwrapped.walls_num

        goal_avg_episode_length = env.unwrapped.walls_num
        wall_coef = 6 / env.unwrapped.walls_num
        goal_consecutive_episodes = int(wall_coef * self.window)  # how many times agent can consecutive run succesful

        plot_episode_count = 200
        plot_redraw_frequency = 10

        agent = TetrisRaceQLearningAgent(env,
                                         learning_rate=rate,
                                         discount_factor=factor,
                                         exploration_rate=exploration,
                                         exploration_decay_rate=exp_decay
                                         )

        episode_history = EpisodeHistory(env,
                                         learn_rate=rate,
                                         discount=factor,
                                         capacity=max_episodes_to_run,
                                         plot_episode_count=plot_episode_count,
                                         max_timesteps_per_episode=max_timesteps_per_episode,
                                         goal_avg_episode_length=goal_avg_episode_length,
                                         goal_consecutive_episodes=goal_consecutive_episodes)
        episode_history.create_plot()

        finish_freq = [0.5, True]  # desired percent finishes in window, flag to run subtask once
        for episode_index in range(0, max_episodes_to_run):
            timestep_index = 0
            observation = env.reset()

            while True:
                action = agent.choose_action(observation)
                observation_, reward, done, info = env.step(action)  # Perform the action and observe the new state.

                if verbose:
                    env.render()

                if done and timestep_index < max_timesteps_per_episode - 1:
                    reward = -max_episodes_to_run

                QDF = agent.act(observation, action, reward, observation_)
                observation = observation_

                if done:
                    self.episode_index += 1
                    # self.saveQmemoryInFile(agent.q_memory)
                    self.done_manager(self, episode_index, [], [], 'D')
                    if self.done_manager(self, episode_index, [], finish_freq, 'S') and finish_freq[1]:
                        foo = Classification()
                        finish_freq[1] = False
                    # foo = Regression(QDF)
                    episode_history[episode_index] = timestep_index + 1
                    if verbose or episode_index % plot_redraw_frequency == 0:
                        episode_history.update_plot(episode_index)

                    if episode_history.is_goal_reached(episode_index):
                        print("Goal reached after {} episodes!".format(episode_index + 1))
                        end_index = episode_index + 1
                        foo = Regression(QDF)
                        self.done_manager(self, [], plt, [], 'P')

                        return episode_history, end_index
                    break
                elif env.unwrapped.wall_iterator - timestep_index > 1:
                    timestep_index += 1
            print("Goal not reached after {} episodes.".format(max_episodes_to_run))
            end_index = max_episodes_to_run
            return episode_history, end_index

    def done_manager(self, episode_ind, plt, top, mode):
        # Call this function to handle episode end event and for storing some
        # result files, pictures etc

        if mode == 'D':  # work with history data
            refresh_each = 100
            self.agent_history.append(self.env.unwrapped.wall_iterator)
            if episode_ind % refresh_each == 0 and self.history_f:
                root = self.exp_dir.split('/')[0]
                base = '/_data'
                path = root + base
                if not os.path.exists(path):
                    os.makedirs(path)
                with open(path + '/' + self.team_name + '.pickle', 'wb') as f:
                    pickle.dump(self.agent_history, f)
        if mode == 'P':  # work with progress plot
            path = self.exp_dir + '/learn_curve'
            name = '/W ' + str(self.env.unwrapped.walls_num) + \
                   '_LR ' + str(self.learning_rate) + '_DF ' + str(self.discount_factor)
            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig(path + name + '.png')
        if mode == 'S':  # call subtasks when condition
            if episode_ind > self.window:
                arr = self.agent_history[episode_ind - self.window: episode_ind]
                mx = np.max(arr)
                ind = np.where(arr == mx)[0]
                count = ind.shape[0]
                prc = count / self.window if mx > self.env.unwrapped.walls_per_level * 2 else 0
                x = self.agent_history
                total_finishes = sum(map(lambda x: x > self.env.unwrapped.walls_per_level * 2, x))

                return prc >= top[0] and total_finishes > 100

    def video_callable(episode_id):
        # call agent draw each N episodes
        return episode_id % 300 == 0

    def log_timestep(self, index, action, reward, observation):
        # print parameters in console
        format_string = "   ".join(['Timestep:{}',
                                    'Action:{}',
                                    'Reward:{}',
                                    'Car pos:{}',
                                    'WallY pos:{}'])
        print('Timestep: format string ', format_string.format(index, action, reward,
                                                               observation[0], observation[1]))

    def save_history(self, history, experiment_dir):
        # Save the episode lengths to CSV.
        filename = os.path.join(experiment_dir, "episode_history.csv")
        dataframe = pd.DataFrame(history.lengths, columns=["length"])
        dataframe.to_csv(filename, header=True, index_label="episode")

    def saveQmemoryInFile(q_memory):
        file = open('./logQmemmory__.txt', 'a')
        file.write('File be update: ' + str(datetime.now()) + " -- \n")
        index = indexEpisode()
        if (index == 0):
            file.write("\n")
            file.write("+++++++++++++++++++++\n")
            file.write("\n")
        file.write('new Episode\n')
        i = 0
        while i > (-100000):
            arr = q_memory.getAllStateWhereY(i)
            if (not arr):
                break;
            file.write('_____ Y __ =>' + str(i) + ' ---count state---=>' + str(len(arr)) + '\n')
            for item in arr:
                file.write(item.toString() + ' episode= ' + str(index) + '\n')
            i -= 1
        file.close()


class Regression:
    def __init__(self):
        pass


class Classification:
    def __init__(self):
        pass


class TetrisRaceQLearningAgent:
    def __init__(self, env, learning_rate=0.5, discount_factor=0.5,
                 exploration_rate=0.5, exploration_decay_rate=0.5):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay_rate = exploration_decay_rate
        self.actions = env.unwrapped.actions
        self._num_actions = len(self.actions)
        self.state = None
        self.action = None

        self.q_memory = QMemory()
        self.track = Track()

        self.wall_iterator = env.unwrapped.wall_iterator  # passed walls counter

        self.q_table = None

    def choose_action(self, observation):
        action = np.random.choice(self.actions);
        qState = QState(observation, -1)
        isQStateInMemory = self.q_memory.find(qState)
        if isQStateInMemory:
            qState = isQStateInMemory
            if qState.getWeightActL() < qState.getWeightActR():
                action = 1
            else:
                if qState.getWeightActL() > qState.getWeightActR():
                    action = 0
        return action

    def act(self, state, action, reward, state_):
        # =============== TODO: Your code here ===============
        #  Here agent takes action('moves' somewhere), knowing
        #  the value of Q - table, corresponds current state.
        #  Also in each step agent should note that current
        #  'Q-value' can become 'better' or 'worsen'. So,
        #   an agent can update knowledge about env, updating Q-table.
        #   Remember that agent should choose max of Q-value in  each step
        if reward == 0:
            is_state_ = QState(state_, -1)
            q_state_ = self.q_memory.find(is_state_)
            if q_state_:
                if (q_state_.getWeightActL() < 0) and (q_state_.getWeightActR() < 0):
                    is_state = QState(state, action)
                    q_state = self.q_memory.find(is_state)
                    if q_state:
                        if action == 0:
                            q_state.setWeightActL(-1)
                        else:
                            q_state.setWeightActR(-1)
            else:
                self.q_memory.remember(QState(state, action, reward))
        else:
            is_state = QState(state, action, reward)
            q_state = self.q_memory.find(is_state)
            if q_state:
                print('is Fatall algoritm')
                pass
            else:
                q_state = is_state
                q_state.setWeightActR(-1)
                q_state.setWeightActL(-1)
                pr_state = self.q_memory.getParentRight(q_state)
                pl_state = self.q_memory.getParentLeft(q_state)
                pr_state.setWeightActL(-1)
                pr_state.setWeightActR(-1)

                pl_state.setWeightActR(-1)
                pl_state.setWeightActL(-1)
                if action == 1:
                    p_state = self.q_memory.getParentLeft(pr_state)
                    p_state.setWeightActR(-1)
                    print(p_state.toString())
                else:
                    p_state = self.q_memory.getParentRight(pl_state)
                    p_state.setWeightActL(-1)
                    print(p_state.toString())
                self.q_memory.remember(q_state)


def main(env):
    obj = Controler
    obj.__init__(obj(), global_env=env)


if __name__ == "__main__":
    main(env_name)
