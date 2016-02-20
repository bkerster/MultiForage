from __future__ import division
import numpy as np
import pandas as pd
from numba import jit
from pandas import Series, DataFrame
from scipy.spatial import distance
import matplotlib.pyplot as plt
import os, random, re, csv
from multiprocessing import Pool

# Basic usage instructions:
# Before running the model you will need to generate resources
# this can be done with starBuilder.py
# You will also need to edit map_name_maker, currently the map locations are hardcoded in that method

#After that you simply need to run main(). Check the bottom of the file for an example. 



def map_reader(file_name, width=1280, height=1024, box_size=16):
    ''' Reads in a map file with the given file location, width, height, and box size
        The map is then returned as a numpy array that represents the map as grid with each location having a resource value'''
    world = pd.read_csv(file_name, delim_whitespace=True, header=None)
    world = np.array(world)
    world_map = np.zeros( (width, height) )
    for item in world:
        world_map[item[0],item[1]] = item[2]
    s_map = np.zeros( (width / box_size, height / box_size) )
    for i in range(s_map.shape[0]):
        for j in range(s_map.shape[1]):
            s_map[i,j] += world_map[i*box_size:(i+1)*box_size,
                                    j*box_size:(j+1)*box_size].sum() #sums all values in each 16x16 square
    return s_map

def map_name_maker(num_stars, clustering, map_num=None):
    '''Generates the file name for a given set up map conditions. Supplying a map_num allows one to pick a
        specific map, otherwise one is randomly chosen with the given resource values
        It is important to note that the location of the maps is currently hard coded and needs to be modified before use '''
    if map_num is None:
        map_num = random.randint(0,999)
    if len(str(num_stars)) < 4:
        num_stars = '0' + str(num_stars)
    if os.name == 'posix':  #location if on osx/linux
        name = os.path.join('/Users/abarr/Sites/simple/maps',
                            '{}stars{}-{}.txt'.format(num_stars, clustering, map_num))
    elif os.name == 'nt': #location if on windows
        name = os.path.join('C:\\Users\\Bryan\\Documents\\simpleForage\\maps',
                            '{}stars{}-{}.txt'.format(num_stars, clustering, map_num))
    return name

def file_name_parser(file_name):
    '''Generates a map name from a file name.
        This function is currently not used '''
    clust = re.search('(clust)([0-9])' ,file_name).groups()[1]
    res = re.search('(res)([0-9]{3,4})' ,file_name).groups()[1]
    map_num = re.search('(map)([0-9]{1,3})', file_name).groups()[1]
    bg = re.search('(wbg|nobg)', file_name).groups()[0]
    return map_reader(map_name_maker(res, clust, map_num))

def tuple_file_name_parser(file_name):
    '''Pulls information from a file name and returns it as a tuple
        This is currently unused '''
    clust = re.search('(clust)([0-9])' ,file_name).groups()[1]
    res = re.search('(res)([0-9]{3,4})' ,file_name).groups()[1]
    map_num = re.search('(map)([0-9]{1,3})', file_name).groups()[1]
    bg = re.search('(wbg|nobg)', file_name).groups()[0]
    return (res, clust, map_num)

@jit('f8(f8,f8,f8,f8)')
def dist( x1,y1,x2,y2 ):
    '''returns the distance between two points'''
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

#what objects do I want on the agent?
##the agent's representation of the world.
## map of visited locations
##the value it places on OTHER agents
## methods to parse location values
## methods to generate next location        
class Agent:
    def __init__(self, gamma, beta, prob_values, attraction_value=None, name=None):
        self.visited = [] # will be made up of tuples (x, y, value, name)
        self.prob_values = prob_values #this needs to have a location removed each time anything is visited
        self.xyvals = None # the list of x, y, and estimated values
        self.gamma = gamma
        self.beta = beta
        self.prev_loc = None
        self.name = 1 if name is None else name
        self.attraction_value = attraction_value #how much value the agent places on locations visited by other agents. Can be + or -
    
    def calc_map_values(self):  
        #this was a hacky place to use a dataframe, Just use an array instead
        #make sure unvisited is a copy and not a reference
        '''Calculates the value at each unvisited location
            Takes a DataFrame of unvisited locations and a numpy array of visited locations'''
        v = np.array(self.visited)
        x = np.apply_along_axis(self.get_value, 1, self.prob_values, v )
        
        self.xyvals = self.prob_values.copy()
        self.xyvals[:,2] = x
        return self.xyvals
    
    def get_value_small(self, curr_loc, visited):
        '''Returns the value for a given location. 
            This function is used only when very few locations have been explored'''
        val = 0
        for loc in visited:
            val += loc[2] / dist(curr_loc[0], curr_loc[1], loc[0], loc[1]) 
        return val
    
    def get_value(self, unvisited_loc, visited):
        # both curr_loc need to become local variables, visited needs to be an array
        '''Calculates the value for a given location given all other visited locations'''
        #take distance between curr_loc and all visited points
        #reduce to 1d vector
        #multiply each dist by its value
        #sum
        if visited.ndim < 2 or len(visited) < 2:
            return self.get_value_small(unvisited_loc, visited)
        val = np.sum(visited[:,2] / (distance.cdist( visited[:,0:2], [[unvisited_loc[0], unvisited_loc[1]]] )[:,0]) )
        return val   
        
    def calc_prob_distance(self):
        '''Returns the probability distribution
            Takes a DataFrame containing the world state, gamma, beta, and the newly visited location'''
        if self.prev_loc is None:
            self.prob_values[:,2] = np.exp( self.gamma * self.xyvals[:,2] )
            return self.prob_values[:,2]
        distances = np.apply_along_axis(self.get_distance_to_point, 1, self.xyvals )

        vals = self.gamma * (self.xyvals[:,2] + 1.0) / (self.beta * distances + 1.0)
        self.prob_values[:,2] = np.exp(vals)
        return self.prob_values[:,2]
        
    def get_distance_to_point(self, point):
        '''Helper function to calculate distances'''
        return dist(point[0], point[1], self.prev_loc[0], self.prev_loc[1])
        
    def select_location(self, world):
        index = self.weighted_choice(self.prob_values[:,2].copy())
        x, y, val = self.xyvals[index, :]
        points = world[x,y]
        self.visited.append( (x, y, self.filter_vals(x,y, points, val), self.name ) )
        self.prev_loc = (x, y)
        return index, x, y, val, points
        
    def add_visited_location(self, loc, index, source_name):
        '''Adds a location to visited and removes it as a possible target.
            Used to keep track of locations visited by other agents'''
        self.visited.append( (loc[0], loc[1], self.attraction_value, source_name) )
        self.remove_index(index)
        return
        
    
    def weighted_choice(self, weights):
        '''Picks a location based on the weighted probalities'''
        weights = pd.Series(weights)
        totals = np.cumsum(weights)
        norm = totals.iloc[-1]
        r = np.random.rand()
        throw = r*norm
        ind = np.searchsorted(np.array(totals), throw)
        return weights.index[ind]
        #return np.random.choice(len(weights), p=(weights/weights.sum()) )
        
    def filter_vals(self, x, y, points, val):
        '''Ensures that the log is not taken of 0'''
        if points == 0:
            if val < 0.00001:
                return -1 * np.log(0.00001)
            return -1 * np.log(val)
        else:
            return points
            
    def remove_index(self, index):
        self.prob_values = np.delete(self.prob_values, index, 0)
        return


def main(density, clustering, map_num, gamma, beta):
    '''This function will perform one run of the model with the given parameters and write the results to disk
        Note: the out_dir variable hard codes the folder that the results will be output to'''
    out_dir = 'multi_test' #modify this to change the output location of the model
    type_spec = 'ln'
    world = map_reader(map_name_maker(density, clustering, map_num))
    output = [] #used to produce an output to visualized later
    
    #data = {'x':[], 'y':[], 'val':[]}
    data = np.zeros( (world.shape[0]*world.shape[1], 3) )
    count = 0
    for i in range(world.shape[0]):
        for j in range(world.shape[1]):
            data[count, 0] = i #x
            data[count, 1] = j #y
            data[count, 2] = world[i,j] #value
            count += 1
    #df = DataFrame(data)
    
    agents = []
    agents.append( Agent(gamma, beta, data, -1, 1) ) # gamma, beta, data, attraction_value, name
    agents.append( Agent(gamma, beta, data, -1, 2) )
    for i in range(150):
        for agent in agents:
            #calculate, and then visit a location
            vals = agent.calc_map_values() #calculates xyvals
            vals = agent.calc_prob_distance() #calculates prob_values
            index, x, y, val, points = agent.select_location(world)
            output.append( (x, y, val, points, agent.name) )
            agent.remove_index(index)    
            
            #communicate with other agents
            for other_agent in agents:
                if other_agent.name != agent.name:
                    other_agent.add_visited_location( (x,y), index, agent.name )
                    

    clust = np.array(agent.visited)

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    #ind = get_score_indices(clust, world)
    fname = os.path.join(out_dir, 'gamma{}-beta{}-clust{}-res{}-map{}'.format(gamma, beta, clustering, density, map_num))
    
    out = np.array(output)
    np.savetxt(fname + '.txt', out)
    #import pdb; pdb.set_trace()
    p1 = out[out[:,4] == 1]
    p2 = out[out[:,4] == 2]
    
    fig = plt.figure(None, (12,6))
    ax = plt.subplot(1,2,1)
    plt.plot(p1[:,0], p1[:,1], '-o')
    plt.xlim(0,80)
    plt.ylim(0,64)
    ax = plt.subplot(1,2,2)
    plt.plot(p2[:,0], p2[:,1], '-o')
    plt.xlim(0,80)
    plt.ylim(0,64)    
    plt.suptitle('clust{}-res{}-lambda {}-{}'.format(clustering, density, gamma, type_spec))
    #plt.title('Score: {}'.format(clust[ind,2].sum()))
    plt.savefig(fname+'.png', dpi=200)
    plt.close("all")
    
    return [density, clustering, map_num, gamma, beta]

def callback_func(params):
    print params

    
# The lines in this if statement serve to run the model in bulk. Each run is performed by calling main().
# This example is set up to run across all resource conditions and a variety of a gamma and beta values
# editing the lists will modify what the model does.
if __name__ == '__main__':
    density = 600
    clustering = 3
    map_num = 101
    gamma = 5.5
    beta = 0.125
    main(density, clustering, map_num, gamma, beta)
    
    # pool = Pool(processes=5)
    # densitys = [100, 600, 1100]
    # clusterings = [1, 3, 5]
    # gammas = [4.5, 5.0, 5.5, 6.0, 6.5]
    # betas = [0.075, 0.1, 0.125, 0.15, 0.175]
    # for density in densitys:
        # for clustering in clusterings:
            # for gamma in gammas:
                # for beta in betas:
                    # for map_num in range(20):
                        # pool.apply_async(main, [density, clustering, map_num, gamma, beta], callback=callback_func)
    # pool.close()
    # pool.join()
    # print 'done'
