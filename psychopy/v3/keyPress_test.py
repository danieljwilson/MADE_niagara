#http://forum.cogsci.nl/index.php?p=/discussion/206/solved-key-realease

from psychopy import visual, core
from pyglet.window import key

win = visual.Window([400, 400])

#key=pyglet.window.key
response = key.KeyStateHandler()
win.winHandle.push_handlers(response)


#self.experiment.window.winHandle.push_handlers(response)
#win = self.experiment.window
time = core.Clock()
press = visual.TextStim(win, pos=(0,0), text='Hold down the SPACEBAR...')
release = visual.TextStim(win, pos=(0,0), text='When you want, release the SPACEBAR')
# spacebar not pressed
i = 0
while not i:
    while not response[key.LEFT] and not response[key.RIGHT] and not response[key.UP] and not response[key.DOWN]:
        press.draw()
        win.flip()
    t1 = time.getTime() # spacebar pressed! (first timestamp)
    while response[key.LEFT] or response[key.RIGHT]:
        release.draw()
        win.flip()
    t2 = time.getTime() # spacebar released! (second timestamp)
    spacebar_time = t2-t1 # compute the key pressing time