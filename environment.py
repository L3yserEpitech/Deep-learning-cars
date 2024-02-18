import pygame
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math

BLACK = [0, 0, 0]
init_y_pos = 50
reward = 0

def draw_walls(self):
    wall_thickness = 10
    self.walls = []
    self.finish = []
    self.walls.append(pygame.Rect(300, -20, 150, wall_thickness))
    self.walls.append(pygame.Rect(300, 0, wall_thickness, 150))
    self.walls.append(pygame.Rect(450, 0, wall_thickness, 100))
    self.walls.append(pygame.Rect(200+100, 100, wall_thickness, 150))
    self.walls.append(pygame.Rect(800-200+100, 100, wall_thickness, 400))
    self.walls.append(pygame.Rect(350+100, 100, 250, wall_thickness))
    self.walls.append(pygame.Rect(200+100, 250, 250, wall_thickness))
    self.walls.append(pygame.Rect(450+100, 250, wall_thickness, 100))
    self.walls.append(pygame.Rect(0+100, 350, 460, wall_thickness))
    self.walls.append(pygame.Rect(0+100, 350, wall_thickness, 800-350))
    self.walls.append(pygame.Rect(150+100, 350+150, 460, wall_thickness))
    self.walls.append(pygame.Rect(150+100, 500, wall_thickness, 100))
    self.finish.append(pygame.Rect(10+100, 600-30, 140, 30))
    for wall in self.walls:
        pygame.draw.rect(self.screen, BLACK, wall)
    for finish in self.finish:
        pygame.draw.rect(self.screen, [255, 255, 0], finish)

class game_environnement:
    
    def __init__(self):
        pygame.init()
        self.screen_width = 800
        self.screen_height = 600
        self.walls = []
        self.screen = self.screen = pygame.display.set_mode([self.screen_width, self.screen_height])
        pygame.display.set_caption('DEEP LEARNING CARS')
        self.background_color = [255, 255, 255]
        self.screen.fill(self.background_color)
        draw_walls(self)
        self.img_car = pygame.image.load('car.png')
        self.img_car_scale = pygame.transform.scale(self.img_car, (250/9, 500/9))
        self.img_car_rotate = pygame.transform.rotate(self.img_car_scale, 180)

        self.width, self.height = self.img_car.get_size()
        self.init_x_pos = 365
        self.init_y_pos = 50
        self.x_pos = float(self.init_x_pos)
        self.y_pos = float(self.init_y_pos)
        self.screen.blit(self.img_car_rotate, (self.x_pos - (self.width/5) / 2, self.y_pos - (self.height/5) / 2))
        self.finish_center_x = self.finish[0].x + self.finish[0].width / 2
        self.finish_center_y = self.finish[0].y + self.finish[0].height / 2

        pygame.display.update()


    def test_collision(self, pos_x, pos_y):
        car_rect = pygame.Rect(pos_x - (self.width/5) / 2, pos_y - (self.height/5) / 2, self.width/9, self.height/9)
        for wall in self.walls:
            if car_rect.colliderect(wall):
                return True
        return False

    def finish_game(self, pos_x, pos_y):
        car_rect = pygame.Rect(pos_x - (self.width/5) / 2, pos_y - (self.height/5) / 2, self.width/9, self.height/9)
        if car_rect.colliderect(self.finish[0]):
            print("win")
            return True
        return False


    def moove_right(self):
        print("mes poses dans le right", self.x_pos, self.y_pos)
        self.x_pos += 0.25
        self.screen.fill(self.background_color)
        draw_walls(self)
        self.img_car_rotate = pygame.transform.rotate(self.img_car_scale, 90*3)
        self.width, self.height = self.img_car.get_size()
        self.screen.blit(self.img_car_rotate, (self.x_pos - (self.width/5) / 2, self.y_pos - (self.height/5) / 2))
        pygame.display.update()
        return self.x_pos, self.y_pos
    
    def moove_left(self):
        print("mes poses dans le left", self.x_pos, self.y_pos)
        self.x_pos -= 0.25
        self.screen.fill(self.background_color)
        draw_walls(self)
        self.img_car_rotate = pygame.transform.rotate(self.img_car_scale, 90*1)
        self.width, self.height = self.img_car.get_size()
        self.screen.blit(self.img_car_rotate, (self.x_pos - (self.width/5) / 2, self.y_pos - (self.height/5) / 2))
        pygame.display.update()
        return self.x_pos, self.y_pos
    
    def moove_up(self):
        # print()
        print("mes poses dans le up", self.x_pos, self.y_pos)
        self.y_pos -= 0.25
        self.screen.fill(self.background_color)
        draw_walls(self)
        self.img_car_rotate = pygame.transform.rotate(self.img_car_scale, 90*0)
        self.width, self.height = self.img_car.get_size()
        self.screen.blit(self.img_car_rotate, (self.x_pos - (self.width/5) / 2, self.y_pos - (self.height/5) / 2))
        pygame.display.update()
        return self.x_pos, self.y_pos
    
    def moove_down(self):
        print("mes poses dans le down", self.x_pos, self.y_pos)
        self.y_pos += 0.25
        self.screen.fill(self.background_color)
        draw_walls(self)
        self.img_car_rotate = pygame.transform.rotate(self.img_car_scale, 90*2)
        self.width, self.height = self.img_car.get_size()
        self.screen.blit(self.img_car_rotate, (self.x_pos - (self.width/5) / 2, self.y_pos - (self.height/5) / 2))
        pygame.display.update()
        return self.x_pos, self.y_pos
        
    def reset(self):
        self.x_pos = self.init_x_pos
        self.y_pos = self.init_y_pos
        print("mes poses dans le resest", self.x_pos, self.y_pos)
        self.img_car_rotate = pygame.transform.rotate(self.img_car_scale, 180)
        return (self.x_pos, self.y_pos)
    
    def step(self, action):
        reward = 0
        done = False
        if action in ACTIONS:
            # print("mon action : ", action)
            self.my_new_state = ACTIONS[action]()
            print ("Mon New state que je veux la nan", self.my_new_state)

        if self.test_collision(self.my_new_state[0], self.my_new_state[1]):
            # print("Ca touche")
            reward = -200
            done = True
            # self.reset()
        elif self.finish_game(self.my_new_state[0], self.my_new_state[1]):
            reward = 1000
            done = True
            # self.reset()
        else :
            self.finish_center_x = round(self.finish_center_x, 3)
            self.finish_center_y = round(self.finish_center_y, 3)
            new_x = round(self.my_new_state[0], 3)
            new_y = round(self.my_new_state[1], 3)

            dist_to_finish = math.sqrt((self.finish_center_x - new_x) ** 2 + (self.finish_center_y - new_y) ** 2)

            dist_to_finish = round(dist_to_finish, 3)
            # print("ma distance de l'arrivée est de : ", (dist_to_finish))
            reward = -(dist_to_finish / 100)
            done = False
        print("mon Done est à ", done )
        return self.my_new_state, reward, done
    
env = game_environnement()

ACTIONS = {
    0: env.moove_up,
    1: env.moove_left,
    2: env.moove_right,
    3: env.moove_down
}

# i = 0
# running = True
# while running:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             running = False
#             break

#     keys = pygame.key.get_pressed()
#     action = None 

#     if keys[pygame.K_d]:
#         print("right")
#         i = i + 1
#         print (i)
#         action = 2
#     elif keys[pygame.K_q]:
#         print("left")
#         action = 1
#     elif keys[pygame.K_z]:
#         print("up")
#         action = 0
#     elif keys[pygame.K_s]:
#         print("down")
#         action = 3

#     if action is not None:
#         wesh, salut, cava = env.step(action)
