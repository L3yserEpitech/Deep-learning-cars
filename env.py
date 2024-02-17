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
    self.walls = []

    self.walls.append(pygame.Rect(200+100, 100, wall_thickness, 150))
    self.walls.append(pygame.Rect(800-200+100, 100, wall_thickness, 400))
    self.walls.append(pygame.Rect(350+100, 100, 250, wall_thickness))
    self.walls.append(pygame.Rect(200+100, 250, 250, wall_thickness))
    self.walls.append(pygame.Rect(450+100, 250, wall_thickness, 100))
    self.walls.append(pygame.Rect(0+100, 350, 460, wall_thickness))
    self.walls.append(pygame.Rect(0+100, 350, wall_thickness, 800-350))
    self.walls.append(pygame.Rect(150+100, 350+150, 460, wall_thickness))
    self.walls.append(pygame.Rect(150+100, 500, wall_thickness, 100))
    self.walls.append(pygame.Rect(10+100, 600-30, 140, 30))
    for wall in self.walls:
        pygame.draw.rect(self.screen, BLACK, wall)

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
        self.img_car_scale = pygame.transform.scale(self.img_car, (250/6, 500/6))
        self.img_car_rotate = pygame.transform.rotate(self.img_car_scale, 180)

        self.width, self.height = self.img_car.get_size()
        self.init_x_pos = 365
        self.init_y_pos = 50
        self.x_pos = self.init_x_pos
        self.y_pos = self.init_y_pos
        self.screen.blit(self.img_car_rotate, (self.x_pos - (self.width/5) / 2, self.y_pos - (self.height/5) / 2))

        pygame.display.update()


    def test_collision(self):
        car_rect = pygame.Rect(self.x_pos, self.y_pos, self.width/6, self.height/6)
        for wall in self.walls:
            if car_rect.colliderect(wall):
                return True
        return False

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

    if env.test_collision():
        env.x_pos = env.init_x_pos
        env.y_pos = env.init_y_pos
    keys = pygame.key.get_pressed()
    if keys[pygame.K_d]:
        env.moove_right()
    elif keys[pygame.K_q]:
        env.moove_left()
    elif keys[pygame.K_z]:
        env.moove_up()
    elif keys[pygame.K_s]:
        env.moove_down()
