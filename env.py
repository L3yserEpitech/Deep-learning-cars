import pygame
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math

BLACK = [0, 0, 0]

def draw_walls(self):
    wall_thickness = 10
    
    wall = [200+100, 100, wall_thickness, 150] 
    pygame.draw.rect(self.screen, BLACK, pygame.Rect(*wall))
    wall = [800-200+100, 100, wall_thickness, 400]
    pygame.draw.rect(self.screen, BLACK, pygame.Rect(*wall))
    wall = [350+100, 100, 250, wall_thickness]
    pygame.draw.rect(self.screen, BLACK, pygame.Rect(*wall))
    wall = [200+100, 250, 250, wall_thickness]
    pygame.draw.rect(self.screen, BLACK, pygame.Rect(*wall))
    wall = [450+100, 250, wall_thickness, 100]
    pygame.draw.rect(self.screen, BLACK, pygame.Rect(*wall))
    wall = [0+100, 350, 460, wall_thickness]
    pygame.draw.rect(self.screen, BLACK, pygame.Rect(*wall))
    wall = [0+100, 350, wall_thickness, 800-350]
    pygame.draw.rect(self.screen, BLACK, pygame.Rect(*wall))
    wall = [150+100, 350+150, 460, wall_thickness]
    pygame.draw.rect(self.screen, BLACK, pygame.Rect(*wall))
    wall = [150+100, 500, wall_thickness, 100]
    pygame.draw.rect(self.screen, BLACK, pygame.Rect(*wall))
    wall = [10+100, 600-30, 140, 30]
    pygame.draw.rect(self.screen, [255, 255, 0], pygame.Rect(*wall))
    
    

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
        self.img_car_scale = pygame.transform.scale(self.img_car, (150/4.5, 300/4.5))
        self.img_car_rotate = pygame.transform.rotate(self.img_car_scale, 180)
        self.screen.blit(self.img_car_rotate, (365, 50))

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


    
    
