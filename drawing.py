import os
import pygame as pg
from pygame.constants import FULLSCREEN
import numpy as np
import torch
from agent import Agent
class Game():
    def __init__(self):
        pg.init()
        pg.font.init()
        self.width, self.height = 1920, 1080
        self.fps = 300
        self.dt = 1/self.fps
        self.grid_height = self.height // (6/4)
        self.drawing_surface = pg.Surface((644, 644))
        self.downscaled_drawing = pg.Surface((28, 28))
        self.grid_height_offset = (self.height - self.grid_height) / 2
        self.grid_width_offset = (self.width - self.grid_height) / 4 - 100
        self.saved_drawings = 0
        self.data_points = []
        self.show_drawing = True
        self.show_samples = False
        self.sample_surface = pg.Surface((self.width, self.height))
        self.predicted_number = 0
        self.framerate = 0
        self.agent = Agent()
        self.outputs = []
        self.probability_array = np.empty(62, dtype=tuple)
        print(self.probability_array.dtype)
            
    def start(self):
        self.screen=pg.display.set_mode((0, 0), FULLSCREEN)
        pg.display.set_caption("Game!")
        self.fpsClock = pg.time.Clock()
        
    def quit(self):
        pg.quit()
        exit(0)

    def run(self):
        while True:
            self.update()
            self.draw()
            self.fpsClock.tick(self.fps)
    
    def update(self):
        self.framerate += 1
        for event in pg.event.get():
            if event.type == pg.quit:
                self.quit()
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    self.quit()
                if event.key == pg.K_c:
                    self.drawing_surface.fill((0, 0, 0))
                
                if event.key == pg.K_SPACE:
                    #self.save_drawing()
                    pass
                    
        if pg.mouse.get_pressed()[0]:
            self.draw_to_drawing()
            
        if pg.mouse.get_pressed()[2]:
            self.draw_to_drawing(erase=True)
        
        if self.framerate % 8 == 0:
            self.update_predicted_number()
        
    def update_predicted_number(self):
        array_drawing = pg.surfarray.array_red(self.downscaled_drawing)
        array_drawing = np.divide(array_drawing, 255)
        array_drawing = torch.tensor(array_drawing, dtype=torch.float32)
        array_drawing = array_drawing.unsqueeze(0).unsqueeze(0)
        array_drawing = array_drawing.to(self.agent.device)
        outputs = self.agent.model(array_drawing)
        self.outputs = torch.softmax(outputs, 1, dtype=torch.float32)
        list1 = []
        i = 0
        for probability in self.outputs[0]:
            probability_percent = probability.item() * 100
            list1.append([probability_percent, self.agent.dataset_classes[i]])
            i += 1
        list1.sort(reverse=True)
        i = 0
        for item in list1:
            self.probability_array[i] = tuple(item)
            i += 1
        _, predictions = torch.max(self.outputs, 1)
        self.predicted_number = self.agent.dataset_classes[predictions.item()]
        pass
    
    def draw(self):
        self.screen.fill((30, 30, 30))
        if self.show_drawing:
            self.draw_downscaled_drawing((self.grid_width_offset * 3 + 400, self.grid_height_offset))
            self.screen.blit(self.drawing_surface, (self.grid_width_offset, self.grid_height_offset))
            #draw text of the predicted number
            self.blit_text(str(self.predicted_number), (150, 150), self.screen, 150)
            #draw all outputs
            output_offset = np.array((0, 70))
            pos = np.array((self.width - 100, 100))
            i = 0
            for item in self.probability_array:
                if item:
                    offset = output_offset * i
                    if 0 <= pos[1] + offset[1] < self.height:
                        self.blit_text(f"{item[1]}: {item[0]:.2f}%", pos + offset, self.screen, 30)
                i += 1
        elif self.show_samples:
            self.screen.blit(self.sample_surface, (0, 0))
        pg.display.flip()
    
    def draw_to_drawing(self, size = 24, erase=False):
        mouse_pos = pg.mouse.get_pos()
        mouse_pos = np.array(mouse_pos)
        offset = np.array((self.grid_width_offset, self.grid_height_offset))
        if erase:
            pg.draw.circle(self.drawing_surface, (0, 0, 0), mouse_pos - offset, size)
        else:
            pg.draw.circle(self.drawing_surface, (255, 255, 255), mouse_pos - offset, size)
    
    def draw_downscaled_drawing(self, offset):
        self.downscaled_drawing = pg.transform.smoothscale(self.drawing_surface, (28, 28))
        surface = pg.transform.scale(self.downscaled_drawing, self.drawing_surface.get_size())
        self.screen.blit(surface, offset)
          
    def blit_text(self, text, position, surface, size, aa=True, font="helvetica", color=(255,255,255)):
        myFont = pg.font.SysFont(font, size)
        text_font = myFont.render(text, aa, (color))
        surface.blit(text_font, text_font.get_rect(center=position))
            
if __name__ == '__main__':
    game = Game()
    game.start()
    game.run()
    