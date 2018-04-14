import numpy as np

class fifo:
    def __init__(self,size):
        self.size = size
        self.buf = [None for i in range(size)]
        self.head = 0
        self.tail = 0

    def push(self,obj):
        self.buf[self.tail] = obj
        self.tail+=1
        self.tail%= self.size

    def pop(self):
        item = self.buf[self.head]
        self.head+=1
        self.head%= self.size
        return item

    def fromhead(self,index):
        return self.buf[(self.head+index)%self.size]

    def fromtail(self,index):
        return self.buf[(self.tail-index-1)%self.size]

    def dump(self,reason):
        # dump the content into file
        with open('fifodump.txt','a') as f:
            string = 'fifodump reason: {}\n'.format(reason)
            for i in self.buf:
                string+=str(i)+'\n'
            string+='head:{} tail:{}\n'.format(self.head,self.tail)
            f.write(string)

def project_observation(observation):
    state_desc = observation

    # Augmented environment from the L2R challenge
    res = []
    pelvis = None

    for body_part in ["pelvis", "head","torso","toes_l","toes_r","talus_l","talus_r"]:
        cur = []
        cur += state_desc["body_pos"][body_part][0:2]
        cur += state_desc["body_vel"][body_part][0:2]
        cur += state_desc["body_acc"][body_part][0:2]
        cur += state_desc["body_pos_rot"][body_part][2:]
        cur += state_desc["body_vel_rot"][body_part][2:]
        cur += state_desc["body_acc_rot"][body_part][2:]
        if body_part == "pelvis":
            pelvis = cur
            res += cur[1:]
        else:
            cur_upd = cur
            cur_upd[:2] = [cur[i] - pelvis[i] for i in range(2)]
            cur_upd[6:7] = [cur[i] - pelvis[i] for i in range(6,7)]
            res += cur

    for joint in ["ground_pelvis", "ankle_l","ankle_r","back","hip_l","hip_r","knee_l","knee_r"]:
        res += state_desc["joint_pos"][joint]
        res += state_desc["joint_vel"][joint]
        res += state_desc["joint_acc"][joint]

    for muscle in state_desc["muscles"].keys():
        res += [state_desc["muscles"][muscle]["activation"]]
        res += [state_desc["muscles"][muscle]["fiber_length"]]
        res += [state_desc["muscles"][muscle]["fiber_velocity"]]

    cm_pos = [state_desc["misc"]["mass_center_pos"][i] - pelvis[i] for i in range(2)]
    res += cm_pos + state_desc["misc"]["mass_center_vel"] + state_desc["misc"]["mass_center_acc"]

    res += state_desc["forces"]["foot_l"][0:6] + state_desc["forces"]["foot_l"][12:24]
    res += state_desc["forces"]["foot_r"][0:6] + state_desc["forces"]["foot_r"][12:24]

    return res

'''
## Values in the observation vector
y, vx, vy, ax, ay, rz, vrz, arz of pelvis (8 values)
x, y, vx, vy, ax, ay, rz, vrz, arz of head, torso, toes_l, toes_r, talus_l, talus_r (9*6 = 54 values)
rz, vrz, arz of ankle_l, ankle_r, back, hip_l, hip_r, knee_l, knee_r (7*3 = 21 values)
activation, fiber_len, fiber_vel for all muscles (3*18)
x, y, vx, vy, ax, ay ofg center of mass (6)
8 + 9*6 + 7*3 + 3*18 + 6 = 143
'''
# 41 dim to 48 dim
def process_observation(observation):
    observation = project_observation(observation)
    o = list(observation) # an array

    # o[38]= min(6,o[38])/7 # ball info are included later in the stage
    # o[38]=0
    # o[39]=0
    # o[40]=0
    # o[39]/=5
    # o[40]/=5

    o[0]-= 0.9 # minus py by 0.5

    o[8] /=4 # divide pvr by 4
    o[1] /=8 # divide pvx by 10

    return o

_stepsize = 0.01
flatten = lambda l: [item for sublist in l for item in sublist]

def final_processing(l):
    # normalize to prevent excessively large input
    for idx in range(len(l)):
        if l[idx] > 1: l[idx] = np.sqrt(l[idx])
        if l[idx] < -1: l[idx] = - np.sqrt(-l[idx])
    return l

# expand observation from 48 to 48*7 dims
processed_dims = 48 + 14*1 + 3*2 + 1*0 + 8
# processed_dims = 41*8
def generate_observation(new, old=None, step=None):
    global _stepsize
    if step is None:
        raise Exception('step should be a valid integer')

    # deal with old
    if old is None:
        if step!=0:
            raise Exception('step nonzero, old == None, how can you do such a thing?')

        old = {'dummy':None,'balls':[],'que':fifo(1200),'last':step-1}
        for i in range(6):
            old['que'].push(new)

    q = old['que']

    if old['last']+1 != step:
        raise Exception('step not monotonically increasing by one')
    else:
        old['last'] += 1

    return final_processing(process_observation(new)), old

    if step > 1: # bug in osim-rl
        if q.fromtail(0)[36] != new[36]:
            # if last obs and this obs have different psoas value
            print('@step {} Damned'.format(step))
            q.push(['compare(que, new):', q.fromtail(0)[36], new[36]])
            q.dump(reason='obsmixed')
            raise Exception('Observation mixed up, potential bug in parallel code.')

    # q.pop() # remove head
    q.push(new) # add to tail

    # process new
    def lp(n):return list(process_observation(n))
    new_processed = lp(new)

    def bodypart_velocities(at):
        return [(q.fromtail(0+at)[i]-q.fromtail(1+at)[i])/_stepsize for i in range(22,36)]

    def relative_bodypart_velocities(at):
        # velocities, but relative to pelvis.
        bv = bodypart_velocities(at)
        pv1,pv2 = bv[2],bv[3]
        for i in range(len(bv)):
            if i%2==0:
                bv[i] -= pv1
            else:
                bv[i] -= pv2
        return bv

    vels = [bodypart_velocities(k) for k in [0,1]] #[[14][14]]
    relvels = [relative_bodypart_velocities(k) for k in [0,]] #[[14]]
    accs = [
        [
            (vels[t][idx] - vels[t+1][idx])/_stepsize
            for idx in range(len(vels[0]))]
        for t in [0,]]
    # [[14]]

    fv = [(v/8 if (idx%2==0) else v/1) for idx,v in enumerate(flatten(vels))]
    frv = [(rv/8 if (idx%2==0) else rv/1) for idx,rv in enumerate(flatten(relvels))]
    fa = [a/10 for a in flatten(accs)]
    final_observation = new_processed + frv
    # final_observation = new_processed + fv + frv + fa
    # 48+14*4

    # final_observation += flatten(
    #     [lp(q.fromtail(idx))[38:41] for idx in reversed([4,8,16,32,64])]
    # )
    # # 4 * 5
    # # 48*4

    balls = old['balls']
#    ball_ahead = True
#    if new[38] == 100:
        # if no ball ahead
#        ball_ahead = False

    def addball_if_new():
#        nonlocal ball_ahead
        current_pelvis = new[1]
        current_ball_relative = new[38]
        current_ball_height = new[39]
        current_ball_radius = new[40]

        absolute_ball_pos = current_ball_relative + current_pelvis

        if current_ball_radius == 0: # no balls ahead
            return

        compare_result = [abs(b[0] - absolute_ball_pos) < 1e-9 for b in balls]
        # [False, False, False, False] if is different ball

        got_new = sum([(1 if r==True else 0)for r in compare_result]) == 0

        if got_new:
            # for every ball there is
            for b in balls:
                # if this new ball is smaller in x than any ball there is
                if absolute_ball_pos < (b[0] - 1e-9):
                    print(absolute_ball_pos,balls)
                    print('(@ step )'+str(step)+')Damn! new ball closer than existing balls.')
                    q.dump(reason='ballcloser')
                    raise Exception('new ball closer than the old ones.')

            if new[38] != 100:
                balls.append([
                    absolute_ball_pos,
                    current_ball_height,
                    current_ball_radius,
                ])
            if len(balls)>3:
                # edit: since num_of_balls became 10, this check is removed.
                pass
                # print(balls)
                # print('(@ step '+str(step)+')What the fuck you just did! Why num of balls became greater than 3!!!')
                # q.dump(reason='ballgt3')
                # raise Exception('ball number greater than 3.')
        else:
            pass # we already met this ball before.

    if step > 0:
        # initial observation is very wrong, due to implementation bug.
        addball_if_new()

    ball_vectors = []
    current_pelvis = new[1]

    # there should be at most 3 balls
    # edit: there could be as much as 10 balls
    for i in range(2):
        if i<len(balls):
            idx = len(balls)-1-i
            # one ball: [0th none none]
            # two balls: [1st 0th none]
            # 3 balls: [2nd 1st 0th]
            # 4 balls: [3rd 2nd 1st]

            rel = balls[idx][0] - current_pelvis
            falloff = 1
            ball_vectors.append([
                min(8,max(-3, rel))/7, # ball pos relative to current pos
                balls[idx][1] * 5 * falloff, # radius
                balls[idx][2] * 5 * falloff, # height
            ])
        else:
            ball_vectors.append([
                -3/7,
                0,
                0,
            ])

    if new[38] != 100:
        pass
    else:
        ball_vectors.append([
        8/7,
        0,
        0,
        ])
        ball_vectors = ball_vectors[1:]

    # 9-d
    final_observation += flatten(reversed(ball_vectors))

    # episode_end_indicator = max(0, (step/1000-0.6))/10 # lights up when near end-of-episode
    # final_observation[1] = episode_end_indicator
    #
    # final_observation += [episode_end_indicator]

    flat_ahead_indicator = np.clip((current_pelvis - 5.0)/2, 0.0, 1.0)
    # # 0 at 5m, 1 at 7m
    #
    # final_observation += [flat_ahead_indicator]

    foot_touch_indicators = []
    for i in [29,31,33,35]: # y of toes and taluses
        # touch_ind = 1 if new[i] < 0.05 else 0
        touch_ind = np.clip((0.0 - new[i]) * 5 + 0.5, 0., 1.)
        touch_ind2 = np.clip((0.1 - new[i]) * 5 + 0.5, 0., 1.)
        # touch_ind2 = 1 if new[i] < 0.1 else 0
        foot_touch_indicators.append(touch_ind)
        foot_touch_indicators.append(touch_ind2)
    final_observation+=foot_touch_indicators # 8dim

    # for i,n in enumerate(new_processed):
    #     print(i,n)

    final_processing(final_observation)

    return final_observation, old

if __name__=='__main__':
    ff = fifo(4)
    ff.push(1)
    ff.push(3)
    ff.push(5)
    ff.pop()
    ff.pop()
    ff.push(6)
    ff.push(7)

    print(ff.fromhead(0))
    print(ff.fromhead(1))
    print(ff.fromtail(0))
    print(ff.fromtail(1))
