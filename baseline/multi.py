from multiprocessing import Process, Pipe

# FAST ENV

# this is a environment wrapper. it wraps the RunEnv and provide interface similar to it. The wrapper do a lot of pre and post processing (to make the RunEnv more trainable), so we don't have to do them in the main program.

from observation_processor import generate_observation as go
import numpy as np

'''
## Values in the observation vector
y, vx, vy, ax, ay, rz, vrz, arz of pelvis (8 values)
x, y, vx, vy, ax, ay, rz, vrz, arz of head, torso, toes_l, toes_r, talus_l, talus_r (9*6 = 54 values)
rz, vrz, arz of ankle_l, ankle_r, back, hip_l, hip_r, knee_l, knee_r (7*3 = 21 values)
activation, fiber_len, fiber_vel for all muscles (3*18)
x, y, vx, vy, ax, ay ofg center of mass (6)
8 + 9*6 + 7*3 + 3*18 + 6 = 143
'''
class fastenv:
    def __init__(self,e,skipcount):
        self.e = e
        self.stepcount = 0

        self.old_observation = None
        self.skipcount = skipcount # 4

    def obg(self,plain_obs):
        # observation generator
        # derivatives of observations extracted here.
        processed_observation, self.old_observation = go(plain_obs, self.old_observation, step=self.stepcount)
        return np.array(processed_observation)

    def step(self,action):
        action = [float(action[i]) for i in range(len(action))]

        import math
        for num in action:
            if math.isnan(num):
                print('NaN met',action)
                raise RuntimeError('this is bullshit')

        sr = 0
        sp = 0
        for j in range(self.skipcount):
            self.stepcount+=1
            oo,r,d,i = self.e.step(action)

            headx = oo["body_pos"]["head"][0]
            py = oo["body_pos"]["pelvis"][1]

            kneer = oo["joint_pos"]["knee_r"][0]
            kneel = oo["joint_pos"]["knee_l"][0]

            lean = min(0.3, max(0, headx - 0.15)) * 0.05
            joint = sum([max(0, k-0.1) for k in [kneer, kneel]]) * 0.03
            penalty = lean + joint            

            o = self.obg(oo)
            sr += r
            sp += penalty
            
            if d == True:
                break

        return o,sr,d,i,sp

    def reset(self):
        self.stepcount=0
        self.old_observation = None

        oo = self.e.reset()
        # o = self.e.reset(difficulty=2)
        self.lastx = oo["body_pos"]["pelvis"][0]
        o = self.obg(oo)
        return o
