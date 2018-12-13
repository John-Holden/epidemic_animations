"""
Created on Mon Jan 15 15:30:51 2018
@author: b7068818

This file is to be called by main.py in the parent directory.
"""
import numpy as np
import os, sys
import matplotlib.pyplot as plt

class Init(object):

    # Initialise the parameters and values for the simulation
    def __init__(self, param):

        if param['lattice'] == 'random':
            self.tree_dist = np.array(np.random.uniform(low=0, high=1, size=param['dim']) < param['rho'], dtype=int)
            self.dim = np.shape(self.tree_dist)
            self.sea_ind = np.zeros(self.dim)
            infected = np.zeros(self.dim)
            infected[param['epi_c']] = 2
        elif param['lattice'] == 'l_hill':
            data_name = param['data_set_name']
            name = os.getcwd() + '/input_data/' + data_name
            tree_dist = np.load(name)
            self.tree_dist = tree_dist
            self.dim = np.shape(self.tree_dist)
            sea_ind = np.zeros(self.dim)
            self.sea_ind = sea_ind
            sea_ind[np.where(tree_dist == -1)] = np.nan
            infected = np.zeros(self.dim)
            infected[800:810, 400:410] = 2
        self.beta_value = param['beta']
        self.run_time = param['run_time']
        self.life_time = param['life_time']
        self.removed = np.zeros(self.dim)
        self.rand = np.random.random(self.dim)
        self.beta_dist = self.beta_value * np.ones(self.dim)
        self.survival_times = np.ones(self.dim) * self.life_time
        self.infected = infected
        return

    def Cmap(self):
        import numpy as np
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        cmap = plt.cm.YlOrRd
        cmaplist = [cmap(i) for i in range(cmap.N)]
        # Get list of 13 colours yellow to
        # First colour as blue, represents Sea
        # Second, black for empty or R state
        # Third, green for susceptible S tree
        # Rest, infectious
        #todo : this could be generalised so that th colour map dynamically generates descrete cbar upto lifetime
        cmaplist = cmaplist[0:300:22]
        cmaplist[0] = (.0, .0, .0)
        cmaplist[1] = (.0, .4, .0)
        cmaplist[-1] = (.0, .1, 0.4)
        bounds = np.linspace(0, 12, 13)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
        return cmap, norm, bounds

    def plot_frame(self, i , pc, Arr , Cmap, Norm, Bounds, Name, sea_ind,mode):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        plt.gca().set_axis_off()
        plt.margins(0, 0)
        if mode == 'anim':
            #ax.set_title('Time: T = ' + str(i))
            im = ax.imshow(Arr, clim=[0, 11], cmap=Cmap)
            #plt.colorbar(im, boundaries=Bounds, norm=Norm)
        elif mode == 'static':
            from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
            from mpl_toolkits.axes_grid1.inset_locator import mark_inset
            Arr = np.load('dat.npy')
            Arr = np.flip(Arr, axis=0)
            im = ax.imshow(Arr, clim=[0, 11], cmap=Cmap, origin='lower')
            axins = zoomed_inset_axes(ax, 4, loc=1, bbox_to_anchor=(1, 1, 0.45, -0.01), bbox_transform=ax.transAxes)  # zoom = 6
            axins.imshow(Arr, interpolation="nearest", cmap=Cmap)
            x1, x2, y1, y2 = 297, 397, 220, 320
            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)
            plt.xticks(visible=False)
            plt.yticks(visible=False)
            mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
            pass

        plt.savefig("frames/frame_" + str(i) + "_.pdf", bbox_inches='tight', pad_inches=0)
        return

    def disease_algo(self, i, infected, tree_dist, removed, beta_dist, survival_times, param, rand):
        # Run Algorithm.........
        potential_infected = np.roll(infected, 1, axis=0) + np.roll(infected, -1, axis=0) + \
                           + np.roll(infected, 1, axis=1) + np.roll(infected, -1, axis=1)
        # All cells which could potentially be infected given beta
        potential_infected = tree_dist * potential_infected
        beta_dynamics = beta_dist > np.random.permutation(rand)
        # New infected cells, initialised with value 2
        new_infected = 2 * (potential_infected * beta_dynamics > 0)
        # Add to infection matrix, each infected site increases by 1 unit for time through infection
        infected = infected + (infected > 0) + new_infected
        # When infected reaches time limit they are killed off and put into removed category
        new_removed = np.array(infected == survival_times, dtype=int)
        removed = (removed + new_removed) > 0
        # Take away infected trees from the susceptible class
        tree_dist = tree_dist * (np.logical_not(infected > 1))
        # Take away removed trees from the infected class
        infected = infected * (np.logical_not(new_removed == 1))
        return infected, tree_dist, removed

def main(param, name):

    from matplotlib import animation as anim
    import matplotlib.pyplot as plt
    import sys

    pc = Init(param)
    removed, infected, tree_dist, rand, beta_dist, l_time_dist, l_time, run_time, sea_ind = [pc.removed, pc.infected,
                                                                                            pc.tree_dist, pc.rand,
                                                                                            pc.beta_dist, pc.survival_times,
                                                                                            pc.life_time, pc.run_time,
                                                                                            pc.sea_ind]
    cmap, norm, bounds = pc.Cmap()
    fig, ax = plt.subplots()
    im = ax.imshow(tree_dist + infected, clim=[0, 12], cmap=cmap)
    plt.colorbar(im, boundaries=bounds, norm=norm)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title('frame: 0')

    def animate(i, Infected, Tree_dist, Removed, Beta_dist, L_time_dist, Param, Rand, Plot_frame, Name, sea_ind):
        print(i)
        Infected_, Tree_dist_, Removed_ = pc.disease_algo(i, Infected, Tree_dist, Removed, Beta_dist,
                                                          L_time_dist, Param, Rand)

        Infected[:], Tree_dist[:], Removed[:] = Infected_, Tree_dist_, Removed_
        plot_arr = Infected + Tree_dist + sea_ind + (Removed * 12)
        im.set_array(plot_arr)
        ax.set_title('frame: ' + str(i))
        # Uncomment for single frame plots
        #if i == 100 or i == 400:
            #Plot_frame(i, pc, Arr=plot_arr, Cmap=cmap, Norm=norm, Bounds=bounds, Name=Name, sea_ind=sea_ind, mode='anim')
        #if i == 1:
            #np.save('dat', plot_arr)
            #Plot_frame(i, pc, Arr=plot_arr, Cmap=cmap, Norm=norm, Bounds=bounds, Name=Name, sea_ind=sea_ind, mode='static')
        if len(np.where(Infected > 0)[0]) == 0:
            print('no infected left')
            sys.exit()
        else:
            return im,

    anim = anim.FuncAnimation(fig, animate, fargs=(infected, tree_dist, removed, beta_dist, l_time_dist, param,
                                                   rand, pc.plot_frame, name, sea_ind), frames=run_time, interval=20, blit=True)
    anim.save('simulations/' + name + '.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
    return


if __name__ == "__main__":

    main(param, name)
