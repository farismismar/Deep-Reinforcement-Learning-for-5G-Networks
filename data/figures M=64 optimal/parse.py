#!/usr/local/bin/python3
# Note, if any output has NAN in it, we drop the entire episode from the calculation.

import glob
import re
import numpy as np
import pandas as pd

measurement_files = glob.glob('measurements*.txt')
reward_files = glob.glob('optimal*.txt')
f1 = open('ue_1_sinr.txt', 'a')
f2 = open('ue_2_sinr.txt', 'a')
f3 = open('ue_1_power.txt', 'a')
f4 = open('ue_2_power.txt', 'a')

#episodes = []
#pattern = '_([0-9]+)_'
#for filename in measurement_files:
#	episode = re.findall(pattern, filename)
#	episodes.append(episode[0])

rewards = []
episodes = []
pattern = '([0-9\.]+)'
for filename in reward_files:
    f = open(filename, 'r')
    text = f.readline()
    f.close()
    reward = re.findall(pattern, text)
    episodes.append(reward[0])
    rewards.append(reward[2][:-1])

episodes = np.array(episodes).astype(int)
rewards = np.array(rewards).astype(float)

pd.DataFrame(data={'episode': episodes,
                   'reward': rewards}).to_csv("convergence.txt", index=False, header=False)

pattern = re.compile('[\[\]_ \':a-z]+') # get rid of [], colons, and words.

for file in measurement_files:
    f = open(file, 'r')
    lines = f.read()
    sinr1 = lines.split(':')[1]
    sinr2 = lines.split(':')[2]
    tx1 = lines.split(':')[3]
    tx2 = lines.split(':')[4]
       
    # Clean up sinr1, 2 by replacing pattern with ''
    f1.write('{},'.format(re.sub(pattern, '', sinr1)))
    f2.write('{},'.format(re.sub(pattern, '', sinr2)))
    f3.write('{},'.format(re.sub(pattern, '', tx1)))
    f4.write('{},'.format(re.sub(pattern, '', tx2)))

f1.close()
f2.close()
f3.close()
f4.close()

f.close()
