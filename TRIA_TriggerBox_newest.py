#TRIA_generator_player
from psychopy import visual, core, event, gui
from psychopy.hardware import keyboard
import math
import csv
import numpy as np
import serial
import pyxid2
import random
import os

### COM 5 trigger box, COM 9 is cedrus 
## 

triggerbox = serial.Serial( port='COM5', 
baudrate=115200,
bytesize=serial.EIGHTBITS, 
stopbits=serial.STOPBITS_ONE, 
parity=serial.PARITY_NONE,
 timeout=0.1 )

#Cedrus for behavioral only
devices = pyxid2.get_xid_devices()
if not devices:
    raise RuntimeError("No Cedrus device found")
cedrus = devices[0]

#Subject Info 
dlg = gui.Dlg(title="Subject Info")
dlg.addField("Subject number:")
ok_data = dlg.show()
if dlg.OK:
    subject_number = ok_data[0]
else:
    core.quit()
filename = f"subj{subject_number}.csv"
    
#Trial setup 
interaction_types = ['generalization', 'alliance', 'displacement', 'defense']

trial_type_counts = {
    'no_reveal': 120,
    'incongruent': 120,
    'congruent': 120
}

trials = []

# Build trials
for trial_type, count in trial_type_counts.items():
    for _ in range(count):
        trials.append({
            'trial_type': trial_type,
            'is_violation': (trial_type == 'incongruent'),
            'interaction_type': random.choice(interaction_types)
        })

# Assign reverse_positions to exactly half
for i, trial in enumerate(trials):
    trial['reverse_positions'] = (i < len(trials) // 2)

# Shuffle after assigning reverse_positions
random.shuffle(trials)


def send_eeg_marker(trial_type, interaction_type, event_type):
    interaction_to_bit = {
        'generalization': 0,
        'alliance': 1,
        'displacement': 2,
        'defense': 3,
    }
    bit_idx = interaction_to_bit[interaction_type]

    # Bits 0-3 = trial start, Bits 4-7 = response
    if event_type == 'start':
        marker_code = 1 << bit_idx
    else:
        marker_code = 1 << (bit_idx + 4)

    print(f"{trial_type}/{interaction_type}/{event_type}  {marker_code}")

    # Send TTL pulse via TriggerBox
    triggerbox.write(bytes([marker_code])) 
    core.wait(0.005) 
    triggerbox.write(bytes([0]))

    #This code uses bits 0-3 to mark trial start and bits 4-7 for a response, leading to VMRK pairs of 1-16 (generalization), 2-32 (alliance), 4-64 (displacement) and 8-128 (defense)
    
#Helpers
def get_bump_target(start_pos, target_pos, stim_radius, target_radius):
    dx = target_pos[0] - start_pos[0]
    dy = target_pos[1] - start_pos[1]
    dist = math.hypot(dx, dy)
    if dist == 0:
        return start_pos
    ux = dx / dist
    uy = dy / dist
    stop_x = target_pos[0] - ux * (stim_radius + target_radius)
    stop_y = target_pos[1] - uy * (stim_radius + target_radius)
    return (stop_x, stop_y)

def update_eyes(agent, eyes, eye_separation=20, eye_y_offset=0):
    if len(eyes) < 4:
        return 
    left_eye_center = (agent.pos[0] - eye_separation/2, agent.pos[1] + eye_y_offset)
    right_eye_center = (agent.pos[0] + eye_separation/2, agent.pos[1] + eye_y_offset)
    eyes[0].pos = left_eye_center
    eyes[1].pos = left_eye_center
    eyes[2].pos = right_eye_center
    eyes[3].pos = right_eye_center
    
def bump_with_recoil(
    bump_agent, bump_eyes, bump_start, bump_end,
    target_agent, target_eyes, target_start, target_recoil_end,
    win, others=[], fps=60,
    forward_duration=0.35,
    recoil_duration=0.05,      
    return_duration=0.3       
):
    total_forward_frames = int(forward_duration * fps)
    for frame in range(total_forward_frames):
        t = frame / total_forward_frames
        bump_agent.pos = (lerp(bump_start[0], bump_end[0], t),
                          lerp(bump_start[1], bump_end[1], t))
        update_eyes(bump_agent, bump_eyes)
        for other in others: other.draw()
        bump_agent.draw()
        for eye in bump_eyes: eye.draw()
        target_agent.draw()
        for eye in target_eyes: eye.draw()
        win.flip(clearBuffer=True)
    total_recoil_frames = int(recoil_duration * fps)
    for frame in range(total_recoil_frames):
        t = frame / total_recoil_frames
        target_agent.pos = (lerp(target_start[0], target_recoil_end[0], t),
                            lerp(target_start[1], target_recoil_end[1], t))
        update_eyes(target_agent, target_eyes)
        for other in others: other.draw()
        bump_agent.draw()
        for eye in bump_eyes: eye.draw()
        target_agent.draw()
        for eye in target_eyes: eye.draw()
        win.flip(clearBuffer=True)
    total_return_frames = int(return_duration * fps)
    xs_bump = np.linspace(bump_end[0], bump_start[0], total_return_frames)
    ys_bump = np.linspace(bump_end[1], bump_start[1], total_return_frames)
    xs_target = np.linspace(target_recoil_end[0], target_start[0], total_return_frames)
    ys_target = np.linspace(target_recoil_end[1], target_start[1], total_return_frames)

    for frame in range(total_return_frames):
        bump_agent.pos = (xs_bump[frame], ys_bump[frame])
        update_eyes(bump_agent, bump_eyes)
        target_agent.pos = (xs_target[frame], ys_target[frame])
        update_eyes(target_agent, target_eyes)
        for other in others: other.draw()
        bump_agent.draw()
        for eye in bump_eyes: eye.draw()
        target_agent.draw()
        for eye in target_eyes: eye.draw()
        win.flip(clearBuffer=True)
    bump_agent.pos = bump_start
    target_agent.pos = target_start
    update_eyes(bump_agent, bump_eyes)
    update_eyes(target_agent, target_eyes)

def lerp(start, end, t):
    return start + (end - start) * t
    
fixed_blob_vertices = None
def make_blob_vertices(n_points=18, radius=40, jitter=8):
    global fixed_blob_vertices
    if fixed_blob_vertices is None:
        angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        radii = radius + np.random.uniform(-jitter, jitter, n_points)
        fixed_blob_vertices = [(r*np.cos(a), r*np.sin(a)) for r,a in zip(radii, angles)]
    return fixed_blob_vertices

class QuestionBlob:
    def __init__(self, win, blob, question):
        self.blob = blob
        self.question = question
        self.win = win
    def draw(self):
        self.blob.draw()

def create_googly_eyes(win, center, eye_separation=20, eye_y_offset=0, eye_radius=10, pupil_radius=4):
    left_eye_center = (center[0] - eye_separation/2, center[1] + eye_y_offset)
    right_eye_center = (center[0] + eye_separation/2, center[1] + eye_y_offset)
    eyes = []
    for eye_center in [left_eye_center, right_eye_center]:
        eyeball = visual.Circle(win, pos=eye_center, radius=eye_radius,
                                fillColor='white', lineColor='black', lineWidth=2)
        pupil = visual.Circle(win, pos=eye_center, radius=pupil_radius,
                              fillColor='black', lineColor='black')
        eyes.extend([eyeball, pupil])
    return eyes

def create_stimuli(win, reverse_positions=False):
    if reverse_positions:
        yellow_pos = (120, 120)
        blue_pos = (-120, 120)
    else:
        yellow_pos = (-120, 120)
        blue_pos = (120, 120)
    yellow_circle = visual.Circle(win, pos=yellow_pos, radius=32,
                             fillColor=[1,1,-1], lineColor=[1,1,-1])
    blue_square = visual.Rect(win, pos=blue_pos, width=60, height=60,
                          fillColor=[-1,-1,1], lineColor=[-1,-1,1])
    yellow_eyes = create_googly_eyes(win, center=yellow_circle.pos, eye_separation=20, eye_y_offset=0)
    blue_eyes = create_googly_eyes(win, center=blue_square.pos, eye_separation=20, eye_y_offset=0)
    blob_vertices = make_blob_vertices()
    blob = visual.ShapeStim(win, vertices=blob_vertices, closeShape=True, fillColor=[-0.5, -0.5, -0.5],
                            lineColor='black', lineWidth=3)
    blob.pos = (0, -120)
    question_circle = QuestionBlob(win, blob, None)
    return yellow_circle, blue_square, question_circle, yellow_eyes, blue_eyes
    
def run_trial(trial, win, cedrus, fps=60):
    reverse_positions = trial.get('reverse_positions', False)
    trial_type = trial.get('trial_type', None)
    itype = trial['interaction_type']
    violation = trial['is_violation']
    yellow_circle, blue_square, question_circle, yellow_eyes, blue_eyes = create_stimuli(win, reverse_positions=reverse_positions)
    
    cedrus.clear_response_queue()
    yellow_circle.draw()
    for eye in yellow_eyes: eye.draw()
    blue_square.draw()
    for eye in blue_eyes: eye.draw()
    question_circle.draw()
    win.callOnFlip(
        send_eeg_marker,
        trial['trial_type'],
        trial['interaction_type'],
        'start'
    )
    win.flip()
    core.wait(0.7)

    #1st bump
    yellow_to_blue = get_bump_target(yellow_circle.pos, blue_square.pos, yellow_circle.radius, blue_square.width / 2)
    dx = blue_square.pos[0] - yellow_circle.pos[0]
    dy = blue_square.pos[1] - yellow_circle.pos[1]
    dist = math.hypot(dx, dy)
    recoil_end = (blue_square.pos[0] + 25 * dx / dist, blue_square.pos[1] + 25 * dy / dist)

    bump_with_recoil(
        bump_agent=yellow_circle,
        bump_eyes=yellow_eyes,
        bump_start=yellow_circle.pos,
        bump_end=yellow_to_blue,
        target_agent=blue_square,
        target_eyes=blue_eyes,
        target_start=blue_square.pos,
        target_recoil_end=recoil_end,
        win=win,
        others=[question_circle],
        fps=fps,
        forward_duration=0.35,
        recoil_duration=0.05,
        return_duration=0.3
    )

    yellow_circle.draw()
    for eye in yellow_eyes: eye.draw()
    blue_square.draw()
    for eye in blue_eyes: eye.draw()
    question_circle.draw()
    core.wait(1)

    #2nd bump
    if itype == "generalization":
        yellow_to_question = get_bump_target(yellow_circle.pos, (0, -100), yellow_circle.radius, 40)
        recoil_end = (question_circle.blob.pos[0], question_circle.blob.pos[1] - 25)

        bump_with_recoil(
            bump_agent=yellow_circle,
            bump_eyes=yellow_eyes,
            bump_start=yellow_circle.pos,
            bump_end=yellow_to_question,
            target_agent=question_circle.blob,
            target_eyes=[],  
            target_start=question_circle.blob.pos,
            target_recoil_end=recoil_end,
            win=win,
            others=[blue_square] + blue_eyes + [question_circle],
            fps=fps,
            forward_duration=0.35,
            recoil_duration=0.05,
            return_duration=0.3
        )
        intended = "blue_square"

    elif itype == "alliance":
        question_to_blue = get_bump_target((0, -100), blue_square.pos, 40, blue_square.width / 2)
        dx = blue_square.pos[0] - question_circle.blob.pos[0]
        dy = blue_square.pos[1] - question_circle.blob.pos[1]
        dist = math.hypot(dx, dy)
        recoil_end = (blue_square.pos[0] + 25 * dx / dist, blue_square.pos[1] + 25 * dy / dist)

        bump_with_recoil(
            bump_agent=question_circle.blob,
            bump_eyes=[],
            bump_start=question_circle.blob.pos,
            bump_end=question_to_blue,
            target_agent=blue_square,
            target_eyes=blue_eyes,
            target_start=blue_square.pos,
            target_recoil_end=recoil_end,
            win=win,
            others=[yellow_circle] + yellow_eyes + [question_circle],
            fps=fps,
            forward_duration=0.35,
            recoil_duration=0.05,
            return_duration=0.3
        )
        intended = "yellow_circle"

    elif itype == "defense":
        question_to_yellow = get_bump_target((0, -100), yellow_circle.pos, 40, yellow_circle.radius)
        dx = yellow_circle.pos[0] - question_circle.blob.pos[0]
        dy = yellow_circle.pos[1] - question_circle.blob.pos[1]
        dist = math.hypot(dx, dy)
        recoil_end = (yellow_circle.pos[0] + 25 * dx / dist, yellow_circle.pos[1] + 25 * dy / dist)

        bump_with_recoil(
            bump_agent=question_circle.blob,
            bump_eyes=[],
            bump_start=question_circle.blob.pos,
            bump_end=question_to_yellow,
            target_agent=yellow_circle,
            target_eyes=yellow_eyes,
            target_start=yellow_circle.pos,
            target_recoil_end=recoil_end,
            win=win,
            others=[blue_square] + blue_eyes + [question_circle],
            fps=fps,
            forward_duration=0.35,
            recoil_duration=0.05,
            return_duration=0.3
        )
        intended = "blue_square"

    elif itype == "displacement":
        blue_to_question = get_bump_target(blue_square.pos, (0, -100), blue_square.width / 2, 40)
        recoil_end = (question_circle.blob.pos[0], question_circle.blob.pos[1] - 25)

        bump_with_recoil(
            bump_agent=blue_square,
            bump_eyes=blue_eyes,
            bump_start=blue_square.pos,
            bump_end=blue_to_question,
            target_agent=question_circle.blob,
            target_eyes=[],
            target_start=question_circle.blob.pos,
            target_recoil_end=recoil_end,
            win=win,
            others=[yellow_circle] + yellow_eyes + [question_circle],
            fps=fps,
            forward_duration=0.35,
            recoil_duration=0.05,
            return_duration=0.3
        )
        intended = "yellow_circle"
    question_circle.blob.pos = (0, -120)

    response = None
    reaction_time = None
    cedrus.clear_response_queue()
    win.callOnFlip(cedrus.reset_timer)  # Start timing
    yellow_circle.draw()
    for eye in yellow_eyes: eye.draw()
    blue_square.draw()
    for eye in blue_eyes: eye.draw()
    question_circle.draw()
    win.flip(clearBuffer=True)

    while response is None:
        if 'escape' in event.getKeys():
            win.close()
            core.quit()
            
        cedrus.poll_for_response()

        if cedrus.has_response():
            resp = cedrus.get_next_response()
            if resp['pressed'] and resp['key'] in (0, 6):  
                if resp['key'] == 0:
                    response = 'yellow'
                else:  
                    response = 'blue'
                
                reaction_time = resp['time'] / 1000.0
                send_eeg_marker(
                    trial['trial_type'],
                    trial['interaction_type'],
                    'response'
                )
                break
    
        core.wait(0.001) 

    #Rate correct
    if trial_type == 'no_reveal':
        correct = 1 if (response == 'yellow' and intended == "yellow_circle") or \
                      (response == 'blue' and intended == "blue_square") else 0
    else:  
        correct = 1 if (response == 'yellow' and intended == "yellow_circle") or \
                      (response == 'blue' and intended == "blue_square") else 0

    #Setup reveal 
    if trial_type == 'no_reveal':
        revealed = question_circle.blob
        revealed_eyes = []
    else:
        if violation:
            revealed_type = "yellow_circle" if intended == "blue_square" else "blue_square"
        else:
            revealed_type = intended

        if revealed_type == "yellow_circle":
            pos = (0, -120)
            revealed = visual.Circle(win, radius=32, fillColor=[1, 1, -1], lineColor=[1, 1, -1], pos=pos)
        else:
            pos = (0, -100)
            revealed = visual.Rect(win, width=60, height=60, fillColor=[-1, -1, 1], lineColor=[-1, -1, 1], pos=pos)
        revealed_eyes = create_googly_eyes(win, center=pos, eye_separation=20, eye_y_offset=0)
    yellow_circle.draw()
    for eye in yellow_eyes:
        eye.draw()
    blue_square.draw()
    for eye in blue_eyes:
        eye.draw()
    core.wait(0.17)

    #Draw reveal 
    if trial_type == 'no_reveal':
        question_circle.draw()
    else:
        revealed.draw()
        for eye in revealed_eyes:
            eye.draw()
    win.flip() 
    core.wait(1.5) #
    
    #Inter-trial fixation
    fix_bg = visual.Rect(win, width=600, height=600, fillColor='black', lineColor='black') 
    fix_cross = visual.ShapeStim(
        win,
        vertices=[(-10, 0), (10, 0), (0, 0), (0, -10), (0, 10)],
        lineWidth=4,
        closeShape=False,
        lineColor='white'
    )
    fix_bg.draw()
    fix_cross.draw()
    win.flip()
    core.wait(0.5)
    
    #save trial info
    return {
        "interaction_type": itype,
        "trial_type": trial.get("trial_type", None),
        "is_violation": violation,
        "response": response,
        "correct": correct,
        "reaction_time": reaction_time,
    }

#Start experiment 
COLOR_KW = dict(colorSpace='rgb')

win = visual.Window(fullscr=True, color='grey', units='pix', waitBlanking=True,
    useFBO=True,
    allowGUI=False, multiSample=False)
    
def wait_for_red_screen(win, stim_list):
    cedrus.clear_response_queue()
    pressed = False

    while True:
        for stim in stim_list:
            stim.draw()
        win.flip()

        cedrus.poll_for_response()
        while cedrus.has_response():
            resp = cedrus.get_next_response()

            # Only accept a fresh press event
            if resp['key'] == 3 and resp['pressed']:
                if not pressed:
                    pressed = True
                    cedrus.clear_response_queue()  # flush ghost events
                    return
                    
def hexagon_positions(center, radius):
    cx, cy = center
    return [(cx + radius * math.cos(math.pi / 3 * i),
             cy + radius * math.sin(math.pi / 3 * i)) for i in range(6)]
radius = 60
circle_positions = hexagon_positions(center=(-150, 0), radius=radius)
square_positions = hexagon_positions(center=(150, 0), radius=radius)
circle_color = [1, 1, -1]  
square_color = [-1, -1, 1]

def draw_occluded_scene(text):
    text.draw()
    blob.draw()
    win.flip()
def draw_reveal_scene(reveal_color, text):
    text.draw()
    if reveal_color == 'yellow':
        yellow_circle_hidden.draw()
        for e in yellow_eyes_hidden: e.draw()
    elif reveal_color == 'blue':
        blue_square_hidden.draw()
        for e in blue_eyes_hidden: e.draw()
    win.flip()

#Animation setup
circles = [visual.Circle(win, radius=25, fillColor=circle_color, lineColor=circle_color, pos=pos, **COLOR_KW)
           for pos in circle_positions]
circle_eyes = [create_googly_eyes(win, center=c.pos) for c in circles]
squares = [visual.Rect(win, width=50, height=50, fillColor=square_color, lineColor=square_color, pos=pos, **COLOR_KW)
           for pos in square_positions]
square_eyes = [create_googly_eyes(win, center=s.pos) for s in squares]
circle_start = (80, 0)
square_start = (-80, 0)
yellow_circle_intro = visual.Circle(win, radius=32, fillColor=circle_color, lineColor=circle_color, pos=circle_start, **COLOR_KW)
yellow_eyes_intro = create_googly_eyes(win, center=yellow_circle_intro.pos)
blue_square_intro = visual.Rect(win, width=60, height=60, fillColor=square_color, lineColor=square_color, pos=square_start, **COLOR_KW)
blue_eyes_intro = create_googly_eyes(win, center=blue_square_intro.pos)

#Triad and bump parameters
bump_end_yellow = get_bump_target(circle_start, square_start, stim_radius=32, target_radius=30)
dx = square_start[0] - circle_start[0]
dy = square_start[1] - circle_start[1]
dist = math.hypot(dx, dy)
ux, uy = dx / dist, dy / dist
recoil_end_yellow = (square_start[0] - 25 * ux, square_start[1] - 25 * uy)
bump_end_blue = get_bump_target(square_start, circle_start, stim_radius=30, target_radius=32)
dx2 = circle_start[0] - square_start[0]
dy2 = circle_start[1] - square_start[1]
dist2 = math.hypot(dx2, dy2)
ux2, uy2 = dx2 / dist2, dy2 / dist2
recoil_end_blue = (circle_start[0] + 25 * ux2, circle_start[1] + 25 * uy2)
blob_vertices = make_blob_vertices()
blob = visual.ShapeStim(win, vertices=blob_vertices, closeShape=True,
                        fillColor=[-0.5, -0.5, -0.5], lineColor='black', lineWidth=3, pos=(0,0), **COLOR_KW)
yellow_circle_hidden = visual.Circle(win, radius=32, fillColor=circle_color, lineColor=circle_color, pos=(0,0), **COLOR_KW)
yellow_eyes_hidden = create_googly_eyes(win, center=yellow_circle_hidden.pos)
blue_square_hidden = visual.Rect(win, width=60, height=60, fillColor=square_color, lineColor=square_color, pos=(0,0), **COLOR_KW)
blue_eyes_hidden = create_googly_eyes(win, center=blue_square_hidden.pos)
    
#Screen1
welcome_text = visual.TextStim(
    win,
    text=("Welcome! In this experiment, you will see interactions among cartoon characters. "
          "You will see an attack between two characters, and then a third character will become involved. "
          "Press RED to continue."),
    color=[1, 1, 1], height=30, wrapWidth=700, **COLOR_KW
)

wait_for_red_screen(win, [welcome_text])

#Screen 2
group_text = visual.TextStim(
    win,
    text=("There are two groups of characters: yellow circles and blue squares. "
          "Press RED to see examples."),
    color=[1, 1, 1], pos=(0, 250), height=24, wrapWidth=700, **COLOR_KW
)

all_stimuli = ([group_text] + 
               circles + 
               [eye for eyes in circle_eyes for eye in eyes] + 
               squares + 
               [eye for eyes in square_eyes for eye in eyes])

wait_for_red_screen(win, all_stimuli)

#Screen 3
attack_intro_text = visual.TextStim(
    win,
    text=("Here is an example of an attack. "
          "Press RED to play the animation, 2 times."),
    color='white', height=24, wrapWidth=700, pos=(0, 250)
)

attack_count = 0
static_stimuli = [attack_intro_text, yellow_circle_intro, blue_square_intro] + yellow_eyes_intro + blue_eyes_intro

while attack_count < 2:
    wait_for_red_screen(win, static_stimuli)
    if attack_count == 0:
        yellow_to_blue = get_bump_target(
            yellow_circle_intro.pos, blue_square_intro.pos,
            yellow_circle_intro.radius, blue_square_intro.width / 2
        )
        dx = blue_square_intro.pos[0] - yellow_circle_intro.pos[0]
        dy = blue_square_intro.pos[1] - yellow_circle_intro.pos[1]
        dist = math.hypot(dx, dy)
        recoil_end = (
            blue_square_intro.pos[0] + 25 * dx / dist,
            blue_square_intro.pos[1] + 25 * dy / dist
        )
        bump_with_recoil(
            bump_agent=yellow_circle_intro,
            bump_eyes=yellow_eyes_intro,
            bump_start=yellow_circle_intro.pos,
            bump_end=yellow_to_blue,
            target_agent=blue_square_intro,
            target_eyes=blue_eyes_intro,
            target_start=blue_square_intro.pos,
            target_recoil_end=recoil_end,
            win=win,
            others=[attack_intro_text],
            fps=60
        )
    else:
        # Blue bumps yellow
        blue_to_yellow = get_bump_target(
            blue_square_intro.pos, yellow_circle_intro.pos,
            blue_square_intro.width / 2, yellow_circle_intro.radius
        )
        dx = yellow_circle_intro.pos[0] - blue_square_intro.pos[0]
        dy = yellow_circle_intro.pos[1] - blue_square_intro.pos[1]
        dist = math.hypot(dx, dy)
        recoil_end = (
            yellow_circle_intro.pos[0] + 25 * dx / dist,
            yellow_circle_intro.pos[1] + 25 * dy / dist
        )
        bump_with_recoil(
            bump_agent=blue_square_intro,
            bump_eyes=blue_eyes_intro,
            bump_start=blue_square_intro.pos,
            bump_end=blue_to_yellow,
            target_agent=yellow_circle_intro,
            target_eyes=yellow_eyes_intro,
            target_start=yellow_circle_intro.pos,
            target_recoil_end=recoil_end,
            win=win,
            others=[attack_intro_text],
            fps=60
        )
    attack_count += 1
    core.wait(1)
        
#Screen 4
occluder_text = visual.TextStim(
    win,
    text=("A third character will become involved, but it will initially be hidden behind a grey screen. "
          "Press RED to see an example of the character being revealed. Once you have viewed the reveal 2 times, press RED to continue."),
    color='white', height=24, pos=(0, 250), wrapWidth=700
)

occluder_stimuli = [occluder_text, blob]
reveal_yellow_stimuli = [occluder_text, yellow_circle_hidden] + yellow_eyes_hidden
reveal_blue_stimuli = [occluder_text, blue_square_hidden] + blue_eyes_hidden
wait_for_red_screen(win, occluder_stimuli)
for _ in range(72):  # 1.2 seconds at 60fps
    for stim in reveal_yellow_stimuli:
        stim.draw()
    win.flip()  
wait_for_red_screen(win, occluder_stimuli)
for _ in range(72):
    for stim in reveal_blue_stimuli:
        stim.draw()
    win.flip()
wait_for_red_screen(win, occluder_stimuli)

#Screen 5
response_instructions = visual.TextStim(
    win,
    text=("Your task is to use your social intuition to guess who the hidden character is. "
          "Press the YELLOW button if you think it is yellow, or the BLUE button if you think it is blue. \n\n\n"
          "Sometimes the revealed character will match what you would normally expect; other times, it may turn out to be different from what you might predict. \n"
          "Sometimes, the character will stay hidden. \n\n\n"
          "Try to respond as quickly and intuitively as you can. Press RED when you're ready to begin. Good luck!"),
    color='white', height=24, wrapWidth=700
)
wait_for_red_screen(win, [response_instructions])

#Experiment start + saving trial info
with open(filename, 'w', newline='') as csvfile:
    fieldnames = ['trial_number','trial_type','interaction_type','reverse_positions','is_violation','response','correct','reaction_time']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    for idx, trial in enumerate(trials, start=1):

        if idx in (120, 240):
            cedrus.clear_response_queue()
            break_text = visual.TextStim(
                win,
                text=(
                    "Great job so far. Now you may take a break. One of our assistants will greet you shortly. "
                    "Press RED to continue to the next block"
                ),
                color='white', height=30, wrapWidth=1000
            )
            print("PARTICIPANT ON BREAK, CHECK IN ON THEM (and don't stop recording)")
            wait_for_red_screen(win, [break_text])
            continue

        trial_data = run_trial(trial, win, cedrus, fps=60)
        writer.writerow({
            'trial_number': idx,
            'trial_type': trial_data.get('trial_type', ''),
            'interaction_type': trial_data.get('interaction_type', ''),
            'reverse_positions': trial.get('reverse_positions', ''),
            'is_violation': trial_data.get('is_violation', ''),
            'response': trial_data.get('response', ''),
            'correct': trial_data.get('correct', ''),
            'reaction_time': trial_data.get('reaction_time', '')
        })
        csvfile.flush()
        os.fsync(csvfile.fileno())


#Experiment end
thanks = visual.TextStim(win, text="Thank you for participating!", color='white', height=30)
thanks.draw()
win.flip()
core.wait(5)
win.close(); core.quit()