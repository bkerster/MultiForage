# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import Image, ImageDraw
import pandas as pd
import os, re, math, sys

# <codecell>
#os.chdir('/Users/bkerster/Sites/model/h_maps2')

# <codecell>

def map_reader(file_name):
    world = pd.read_csv(file_name, delim_whitespace=True, header=None)
    world = np.array(world)
    world_map = np.zeros( (1280, 1024) )
    for item in world:
        world_map[item[0],item[1]] = item[2]
    s_map = np.zeros( (1280 / 16, 1024 / 16) )
    for i in range(s_map.shape[0]):
        for j in range(s_map.shape[1]):
            s_map[i,j] += world_map[i*16:(i+1)*16, j*16:(j+1)*16].sum()
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
        name = os.path.join('/home/bkerster/site/simple/maps',
                            '{}stars{}-{}.txt'.format(num_stars, clustering, map_num))
    elif os.name == 'nt': #location if on windows
        name = os.path.join('C:\\Users\\Bryan\\Documents\\simpleForage\\maps',
                            '{}stars{}-{}.txt'.format(num_stars, clustering, map_num))
    return name

def convert_map_to_list(world):
    world_list = []
    for x in range(world.shape[0]):
        for y in range(world.shape[1]):
            if world[x,y] > 0:
                world_list.append( [x, y, world[x,y]] )
    return np.array(world_list)

# <codecell>

def file_name_parser(file_name):
    clust = re.search('(clust)([0-9])' ,file_name).groups()[1]
    res = re.search('(res)([0-9]{3,4})' ,file_name).groups()[1]
    map_num = re.search('(map)([0-9]{1,3})', file_name).groups()[1]
    #bg = re.search('(wbg|nobg)', file_name).groups()[0]
    return map_reader(map_name_maker(res, clust, map_num))

# <codecell>

def drawLine(x1, y1, x2, y2, draw, color="red"):
    draw.line([(x1,y1),(x2,y2)], fill=color, width=2)
    return draw
    
def drawDot(x, y, draw, size=5, color="yellow"):
    draw.rectangle((x,y, x+size-1, y+size-1), fill=color)
    return draw
    
def drawRectAround(x, y, draw, color = "pink"):
    xCorner = x - (1280 / 15 / 2)
    yCorner = y - (1024 / 15 / 2)
    draw.rectangle( (xCorner, yCorner, xCorner + (1280 / 15), yCorner + (1024 / 15) ), outline=color )
    return draw

def transformAbout(centerX, centerY, x, y):
    #var shipX:int = int(zoomX - (VIEW_XWINDOW / ZOOM_FACTOR / 2) + (ship.x / ZOOM_FACTOR))
    #a = int(centerX - (1280 / 15 / 2) + (x / 15))
    #b = int(centerY -  (1024 / 15 / 2) + (y / 15))
    xCorner = centerX - (1280 / 15 / 2) #corner
    yCorner = centerY - (1024 / 15 / 2)
    
    a = xCorner + (x / 15)
    b = yCorner + (y / 15)
    
    return (a, b)

def get_score_indices(visited_locs, world):
    ind = []
    for i in range(len(visited_locs)):
        if world[visited_locs[i,0], visited_locs[i,1]] > 0:
            ind.append(i)
    return ind

# def hit_color(score, max_score):
    # if max_score == 0:
        # max_score = 200
    # ratio = 200 / max_score
    # shade = score * ratio + 55
    # try:
        # int(shade)
    # except ValueError:
        # import pdb; pdb.set_trace()
    # return int(shade)
    
def hit_color(score, max_score):
    if max_score == 0:
        max_score = 175
    if score > max_score:
        return 255
    ratio = 175 / max_score
    shade = score * ratio + 80
    breaks = range(80, 260, 25)
    shade = breaks[np.searchsorted(breaks, shade)]
    try:
        int(shade)
    except ValueError:
        import pdb; pdb.set_trace()
    return int(shade)

def star_hit_color(score, max_score):
    score = int(score)
    breaks = range(130, 260, 25) #generates 7 different "levels" of shade #changed it to 5 levels
    if score <= len(breaks):
        shade = breaks[score - 1]
    elif score > len(breaks):
        shade = breaks[len(breaks) - 1]
    else:
        raise ValueError('Invalid score presented. Score must be greater than 0')
    return shade

def drawRectAround_new(x, y, draw, color = "pink"):
    xCorner = x - (16 / 2)
    yCorner = y - (16 / 2)
    draw.rectangle( (xCorner, yCorner, xCorner + 16, yCorner + 16 ), outline=color )
    return draw

    
if __name__ == '__main__':
    os.chdir(sys.argv[1])
    files = [file for file in os.listdir('.') if file[-4:] == '.txt']
            
    for file in files:
        print file
        visited_small = np.loadtxt(file)
        im = Image.new("RGBA", (1280, 1024), "black")
        draw = ImageDraw.Draw(im)
        
        #show resources
        world = file_name_parser(file)
        world_list = convert_map_to_list(world)  
        max_point = np.array(world_list)[:,2].max()
        
        
        
        for frame, row in enumerate(visited_small):
            

            # #draw the hits
            # score_indices = get_score_indices(visited_small, world)
            # #import pdb; pdb.set_trace()
            max_score = visited_small[:,3].max()
               
            if row[3] > 0:
                color = hit_color(row[3], max_score)
                draw.rectangle(
                                [row[0]*16,
                                 row[1]*16,
                                 row[0]*16+16,
                                 row[1]*16+16],
                                 fill = (0, color, 0, 255)
                                 )
                                 #fill = (255-color, 255, 255-color, 255)
                                #)
            else:
                #draw the misses
                max_val = np.abs(visited_small[:,2]).max()
                max_val = 0.00001 if max_val<=0 else max_val
                val = 0.00001 if row[2] <= 0 else row[2]

                
                
                
                # if placed value positive
                # if val > 0:
                    #color = (255, 255-hit_val, 255-hit_val, 255) #fade to white
                calced_val = -np.log(val)
                if calced_val < 0:
                    hit_val = hit_color(np.absolute(calced_val), np.absolute(-np.log(max_val)))
                    color = (hit_val, 0, 0, 255) #fade to black
                else:
                    hit_val = hit_color(calced_val, max_score)
                    color = (0, hit_val, 0, 255)
                # if placed value negative
                # elif val < 0:
                    # #color = (255-hit_val, 255-hit_val, 255, 255) #fade to white
                    # hit_val = hit_color(-np.log(np.abs(val)), -np.log(max_val))
                    # color = (0, 0, hit_val, 255) #fade to black
                # elif row[2] == 0:
                    # #color = (255, 255, 255, 255)
                    # color = "black"
                draw.rectangle([row[0]*16, row[1]*16,
                                       row[0]*16+16, row[1]*16+16],
                                       fill = color
                                      )
               
                        
            #draw the lines 
            if frame > 0:
                draw.line([visited_small[frame-1][0]*16+7, visited_small[frame-1][1]*16+7,
                         visited_small[frame][0]*16+7, visited_small[frame][1]*16+7], fill='grey', width=2)

            for item in world_list:
                color = (star_hit_color(item[2], max_point), star_hit_color(item[2], max_point), 0, 255)
                draw = drawDot(item[0]*16+5, item[1]*16+5, draw, size=8, color=color)
            
            out_folder = 'movie4'
            if not os.path.exists(out_folder):
                os.mkdir(out_folder)
            im.save(os.path.join(out_folder, '{}-{}.png'.format(file[:-4], str(frame).zfill(3))))

