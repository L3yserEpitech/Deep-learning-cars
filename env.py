import pygame
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math

BLACK = [0, 0, 0]
init_y_pos = 50

def draw_walls(self):
    wall_thickness = 10
    
    self.wall0 = [200+100, 100, wall_thickness, 150]
    pygame.draw.rect(self.screen, BLACK, pygame.Rect(self.wall0))
    self.wall1 = [800-200+100, 100, wall_thickness, 400]
    pygame.draw.rect(self.screen, BLACK, pygame.Rect(self.wall1))
    self.wall2 = [350+100, 100, 250, wall_thickness]
    pygame.draw.rect(self.screen, BLACK, pygame.Rect(self.wall2))
    self.wall3 = [200+100, 250, 250, wall_thickness]
    pygame.draw.rect(self.screen, BLACK, pygame.Rect(self.wall3))
    self.wall4 = [450+100, 250, wall_thickness, 100]
    pygame.draw.rect(self.screen, BLACK, pygame.Rect(self.wall4))
    self.wall5 = [0+100, 350, 460, wall_thickness]
    pygame.draw.rect(self.screen, BLACK, pygame.Rect(self.wall5))
    self.wall6 = [0+100, 350, wall_thickness, 800-350]
    pygame.draw.rect(self.screen, BLACK, pygame.Rect(self.wall6))
    self.wall7 = [150+100, 350+150, 460, wall_thickness]
    pygame.draw.rect(self.screen, BLACK, pygame.Rect(self.wall7))
    self.wall8 = [150+100, 500, wall_thickness, 100]
    pygame.draw.rect(self.screen, BLACK, pygame.Rect(self.wall8))
    self.wall9 = [10+100, 600-30, 140, 30]
    pygame.draw.rect(self.screen, [255, 255, 0], pygame.Rect(self.wall9))
    

class game_environnement:
    
    def __init__(self):
        pygame.init()
        self.screen_width = 800
        self.screen_height = 600
        self.screen = self.screen = pygame.display.set_mode([self.screen_width, self.screen_height])
        pygame.display.set_caption('DEEP LEARNING CARS')
        self.background_color = [255, 255, 255]
        self.screen.fill(self.background_color)
        draw_walls(self)
        self.img_car = pygame.image.load('car.png')
        self.img_car_scale = pygame.transform.scale(self.img_car, (250/5, 500/5))
        self.img_car_rotate = pygame.transform.rotate(self.img_car_scale, 180)

        self.width, self.height = self.img_car.get_size()
        self.init_x_pos = 365
        self.init_y_pos = 50
        self.x_pos = self.init_x_pos
        self.y_pos = self.init_y_pos
        self.screen.blit(self.img_car_rotate, (self.x_pos - (self.width/5) / 2, self.y_pos - (self.height/5) / 2))


        pygame.display.update()

    def moove_right(self):
        self.x_pos += 0.1
        self.screen.fill(self.background_color)
        draw_walls(self)
        self.img_car_rotate = pygame.transform.rotate(self.img_car_scale, 90*3)
        self.width, self.height = self.img_car.get_size()
        self.screen.blit(self.img_car_rotate, (self.x_pos - (self.width/5) / 2, self.y_pos - (self.height/5) / 2))
        pygame.display.update()
    
    def moove_left(self):
        self.x_pos -= 0.1
        self.screen.fill(self.background_color)
        draw_walls(self)
        self.img_car_rotate = pygame.transform.rotate(self.img_car_scale, 90*1)
        self.width, self.height = self.img_car.get_size()
        self.screen.blit(self.img_car_rotate, (self.x_pos - (self.width/5) / 2, self.y_pos - (self.height/5) / 2))
        pygame.display.update()
    
    def moove_up(self):
        self.y_pos -= 0.1
        self.screen.fill(self.background_color)
        draw_walls(self)
        self.img_car_rotate = pygame.transform.rotate(self.img_car_scale, 90*0)
        self.width, self.height = self.img_car.get_size()
        self.screen.blit(self.img_car_rotate, (self.x_pos - (self.width/5) / 2, self.y_pos - (self.height/5) / 2))
        pygame.display.update()
    
    def moove_down(self):
        self.y_pos += 0.1
        self.screen.fill(self.background_color)
        draw_walls(self)
        self.img_car_rotate = pygame.transform.rotate(self.img_car_scale, 90*2)
        self.width, self.height = self.img_car.get_size()
        self.screen.blit(self.img_car_rotate, (self.x_pos - (self.width/5) / 2, self.y_pos - (self.height/5) / 2))
        pygame.display.update()
        
    def reset(self):
        # Réinitialiser le jeu et retourner l'état initial
        pass
        # return initial_state
    
    def step(self, action):
        # Appliquer l'action et retourner le nouvel état, la récompense et si le jeu est terminé
        pass
        # return new_state, reward, done
    
    def render(self):
        # Afficher le jeu
        pygame.display.update()
    
    def close(self):
        # Fermer proprement le jeu
        pygame.quit()
    
class my_model(nn.Module):
    def __init__(self):
        super(my_model, self).__init__()
        # Définir l'architecture du modèle
        pass
    
    def forward(self, x):
        # Implémenter la propagation avant
        pass
    
env = game_environnement()
model = my_model()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            break
    state = env.reset()

    keys = pygame.key.get_pressed()
    if keys[pygame.K_d]:
        env.moove_right()
    elif keys[pygame.K_q]:
        env.moove_left()
    elif keys[pygame.K_z]:
        env.moove_up()
    elif keys[pygame.K_s]:
        env.moove_down()


    
    
