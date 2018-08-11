import sys
import random
from itertools import cycle
import os

import numpy as np

import pygame
import pygame.surfarray as surfarray
from pygame.locals import *

FPS = 30
SCREEN_WIDTH = 288
SCREEN_HEIGHT = 512
PIPE_GAP_SIZE = 100

class FlappyBird:
    def __init__(self):
        pygame.init()
        self.fps_clock = pygame.time.Clock()

        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption('Flappy Bird')

        assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets/')
        self.images = self.load_images(assets_dir)
        self.sounds = self.load_sounds(assets_dir)
        self.hitmasks = self.set_hitmasks(self.images)

        self.bird_width = self.images['bird']['red'][0].get_width()
        self.bird_height = self.images['bird']['red'][0].get_height()
        self.pipe_width = self.images['pipes']['red'][0].get_width()
        self.pipe_height = self.images['pipes']['red'][0].get_height()
        self.background_width = self.images['background']['day'].get_width()

        self.bird_index_gen = cycle([0, 1, 2, 1])
        self.base_y = SCREEN_HEIGHT * 0.79
        self.base_shift = self.images['base'].get_width() - self.background_width
        self.pipe_vel_x = -4
        self.bird_vel_y = 0
        self.bird_max_vel_y = 10
        self.bird_min_vel_y = -8
        self.bird_acc_y = 1
        self.bird_flap_acc = -9

        self.score = 0
        self.bird_index = 0
        self.loop_iter = 0
        self.base_x = 0
        self.upper_pipes = []
        self.lower_pipes = []
        self.bird_flapped = False
        self.bird_x = int(SCREEN_WIDTH * 0.2)
        self.bird_y = int((SCREEN_HEIGHT - self.bird_height) // 2)
        self.bird_color = random.choice(tuple(self.images['bird']))
        self.pipe_color = random.choice(tuple(self.images['pipes']))
        self.background_color = random.choice(tuple(self.images['background']))

        self.reset()

    def reset(self):
        self.score = 0
        self.bird_index = 0
        self.loop_iter = 0
        self.base_x = 0
        self.bird_x = int(SCREEN_WIDTH * 0.2)
        self.bird_y = int((SCREEN_HEIGHT - self.bird_height) // 2)
        self.bird_color = random.choice(tuple(self.images['bird']))
        self.pipe_color = random.choice(tuple(self.images['pipes']))
        self.background_color = random.choice(tuple(self.images['background']))

        new_pipe1 = self.get_random_pipe()
        new_pipe2 = self.get_random_pipe()
        self.upper_pipes = []
        self.upper_pipes = [{'x': SCREEN_WIDTH, 'y': new_pipe1[0]['y']},
                            {'x': SCREEN_WIDTH + (SCREEN_WIDTH // 2), 'y': new_pipe2[0]['y']}]
        self.lower_pipes = []
        self.lower_pipes = [{'x': SCREEN_WIDTH, 'y': new_pipe1[1]['y']},
                            {'x': SCREEN_WIDTH + (SCREEN_WIDTH // 2), 'y': new_pipe2[1]['y']}]

        self.bird_flapped = False

    def frame_step(self, input_actions):
        pygame.event.pump()

        reward = 0.1
        terminal = False

        if sum(input_actions) != 1:
            raise ValueError('Multiple input actions!')

        # input_actions[0] == 1: do nothing
        # input_actions[1] == 1: flap the bird
        if input_actions[1] == 1:
            if self.bird_y > -2 * self.bird_height:
                self.bird_vel_y = self.bird_flap_acc
                self.bird_flapped = True
                self.sounds['wing'].play()

        # check for score
        bird_mid_pos = self.bird_x + self.bird_width // 2
        for pipe in self.upper_pipes:
            pipe_mid_pos = pipe['x'] + self.pipe_width // 2
            if pipe_mid_pos <= bird_mid_pos < pipe_mid_pos + 4:
                self.score += 1
                self.sounds['point'].play()
                reward = 1

        # bird index base x change
        if (self.loop_iter + 1) % 3 == 0:
            self.bird_index = next(self.bird_index_gen)
        self.loop_iter = (self.loop_iter + 1) % 30
        self.base_x = -((-self.base_x + 100) % self.base_shift)

        # bird's movement
        if self.bird_vel_y < self.bird_max_vel_y and not self.bird_flapped:
            self.bird_vel_y += self.bird_acc_y
        if self.bird_flapped:
            self.bird_flapped = False
        self.bird_y += min(self.bird_vel_y, self.base_y - self.bird_y - self.bird_height)
        if self.bird_y < 0:
            self.bird_y = 0

        # move pipes to left
        for u, l in zip(self.upper_pipes, self.lower_pipes):
            u['x'] += self.pipe_vel_x
            l['x'] += self.pipe_vel_x

        # add new pipe when first pipe is about to touch left of screen
        if 0 < self.upper_pipes[0]['x'] < 5:
            new_pipe = self.get_random_pipe()
            self.upper_pipes.append(new_pipe[0])
            self.lower_pipes.append(new_pipe[1])

        # remove first pipe if its out of the screen
        if self.upper_pipes[0]['x'] < -self.pipe_width:
            self.upper_pipes.pop(0)
            self.lower_pipes.pop(0)

        # check if crash here
        is_crash = self.check_crash({'x': self.bird_x, 'y': self.bird_y, 'index': self.bird_index},
                                    self.upper_pipes, self.lower_pipes)
        if is_crash:
            self.sounds['hit'].play()
            self.sounds['die'].play()
            self.reset()
            terminal = True
            reward = -1

        # draw sprites
        self.screen.blit(self.images['background'][self.background_color], (0, 0))

        for u, l in zip(self.upper_pipes, self.lower_pipes):
            self.screen.blit(self.images['pipes'][self.pipe_color][0], (u['x'], u['y']))
            self.screen.blit(self.images['pipes'][self.pipe_color][1], (l['x'], l['y']))

        self.screen.blit(self.images['base'], (self.base_x, self.base_y))
        # print score so player overlaps the score
        self.show_score(self.score)
        self.screen.blit(self.images['bird'][self.bird_color][self.bird_index], (self.bird_x, self.bird_y))

        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.update()
        self.fps_clock.tick(FPS)
        return image_data, reward, terminal

    def load_images(self, assets_dir):
        sprites_dir = os.path.join(assets_dir, 'sprites/')

        images = {}

        # load bird images
        images['bird'] = {}
        for c in ['red', 'blue', 'yellow']:
            paths = [os.path.join(sprites_dir, '%sbird-upflap.png' % c),
                     os.path.join(sprites_dir, '%sbird-midflap.png' % c),
                     os.path.join(sprites_dir, '%sbird-downflap.png' % c)]
            images['bird'][c] = [pygame.image.load(i).convert_alpha() for i in paths]

        # load pipe images
        images['pipes'] = {}
        for c in ['red', 'green']:
            path = os.path.join(sprites_dir, 'pipe-%s.png' % c)
            lower_pipe = pygame.image.load(path).convert_alpha()
            upper_pipe = pygame.transform.rotate(lower_pipe, 180)
            images['pipes'][c] = []
            images['pipes'][c].append(upper_pipe)
            images['pipes'][c].append(lower_pipe)

        # load number images
        images['numbers'] = {}
        for n in range(10):
            path = os.path.join(sprites_dir, '%s.png' % str(n))
            images['numbers'][n] = pygame.image.load(path).convert_alpha()

        # load background images
        images['background'] = {}
        for b in ['day', 'night']:
            shape = os.path.join(sprites_dir, 'background-%s.png' % b)
            images['background'][b] = pygame.image.load(shape).convert()

        # load base image
        images['base'] = pygame.image.load(os.path.join(sprites_dir, 'base.png')).convert()

        return images

    def load_sounds(self, assets_dir):
        sounds_dir = os.path.join(assets_dir, 'sounds/')

        # os windows
        if 'win' in sys.platform:
            sound_ext = '.wav'
        else:
            sound_ext = '.ogg'

        sounds = {}

        # load sounds
        for s in ['die', 'hit', 'point', 'swoosh', 'wing']:
            sounds[s] = pygame.mixer.Sound(sounds_dir + s + sound_ext)

        return sounds

    def set_hitmasks(self, images):
        hitmasks = {}
        hitmasks['bird'] = [self.get_hitmask(images['bird']['red'][0]),
                            self.get_hitmask(images['bird']['red'][1]),
                            self.get_hitmask(images['bird']['red'][2])]

        hitmasks['pipe'] = [self.get_hitmask(images['pipes']['red'][0]),
                            self.get_hitmask(images['pipes']['red'][1])]

        return hitmasks

    def get_hitmask(self, image):
        mask = []
        for x in range(image.get_width()):
            mask.append([])
            for y in range(image.get_height()):
                mask[x].append(bool(image.get_at((x, y))[3]))
        return mask

    def get_random_pipe(self):
        gap_ys = [20, 30, 40, 50, 60, 70, 80, 90]
        index = random.randint(0, len(gap_ys) - 1)
        gap_y = gap_ys[index]

        gap_y += int(self.base_y * 0.2)
        pipe_x = SCREEN_WIDTH + 10

        return [{'x': pipe_x, 'y': gap_y - self.pipe_height},
                {'x': pipe_x, 'y': gap_y + PIPE_GAP_SIZE}]

    def check_crash(self, bird, upper_pipes, lower_pipes):
        bi = bird['index']
        bird['w'] = self.images['bird']['red'][0].get_width()
        bird['h'] = self.images['bird']['red'][0].get_height()

        # if bird crashes into ground
        if bird['y'] + bird['h'] >= self.base_y - 1:
            return True
        else:
            bird_rect = pygame.Rect(bird['x'], bird['y'], bird['w'], bird['h'])

            for u, l in zip(upper_pipes, lower_pipes):
                # upper and lower pipe rects
                u_rect = pygame.Rect(u['x'], u['y'], self.pipe_width, self.pipe_height)
                l_rect = pygame.Rect(l['x'], l['y'], self.pipe_width, self.pipe_height)

                # bird and upper/lower pipe hitmasks
                b_hitmask = self.hitmasks['bird'][bi]
                u_hitmask = self.hitmasks['pipe'][0]
                l_hitmask = self.hitmasks['pipe'][1]

                # if bird collided with upipe or lpipe
                u_collide = self.pixel_collision(bird_rect, u_rect, b_hitmask, u_hitmask)
                l_collide = self.pixel_collision(bird_rect, l_rect, b_hitmask, l_hitmask)

                if u_collide or l_collide:
                    return True

        return False

    def pixel_collision(self, rect1, rect2, hitmask1, hitmask2):
        rect = rect1.clip(rect2)

        if rect.width == 0 or rect.height == 0:
            return False

        x1, y1 = rect.x - rect1.x, rect.y - rect1.y
        x2, y2 = rect.x - rect2.x, rect.y - rect2.y

        for x in range(rect.width):
            for y in range(rect.height):
                if hitmask1[x1 + x][y1 + y] and hitmask2[x2 + x][y2 + y]:
                    return True
        return False

    def show_score(self, score):
        score_digits = [int(x) for x in list(str(score))]
        total_width = 0

        for digit in score_digits:
            total_width += self.images['numbers'][digit].get_width()

        x_offset = (SCREEN_WIDTH - total_width) // 2

        for digit in score_digits:
            self.screen.blit(self.images['numbers'][digit], (x_offset, SCREEN_HEIGHT * 0.1))
            x_offset += self.images['numbers'][digit].get_width()
