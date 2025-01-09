import pygame as pg # Для отрисовки
import numpy as np # Для быстрых массивов
from numba import njit # Для лучшей оптимизации


# Константы
FPS = 60
LOGIC_UPDATE_TIME = 10
WIDTH = 1280
HEIGHT = 720
TILE_SIZE = 5


class Game:
    def __init__(self):
        # Размеры поля в клетках, +4 для создания иллюзии бесконечности поля
        self.field_width = WIDTH // TILE_SIZE + 4
        self.field_height = HEIGHT // TILE_SIZE + 4
        
        # Для ограничения частоты обработки логики
        self.update_timer = 0

        # Флаги
        self.running = True
        self.pause = True
        self.painting = False
        self.do_step = False

        # Инициализация pygame
        pg.init()
        self.win = pg.display.set_mode((WIDTH, HEIGHT))
        self.clock = pg.time.Clock()

        # Массивы полей
        self.current_field = np.zeros(shape=(self.field_height, self.field_width))
        self.next_step_field = np.zeros(shape=(self.field_height, self.field_width))

    # Метод обработки ивентов
    def process_event(self):
        for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.running = False
                elif event.type == pg.MOUSEBUTTONDOWN:
                    self.painting = True
                elif event.type == pg.MOUSEBUTTONUP:
                    self.painting = False
                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_p:
                        self.pause = not self.pause
                    elif event.key == pg.K_u:
                        self.next_step_field = np.array(np.random.random_integers(low=0, high=1, size=(self.field_height, self.field_width)))
                    elif event.key == pg.K_s:
                        self.do_step = True
                    elif event.key == pg.K_c:
                        self.next_step_field = np.zeros(shape=(self.field_height, self.field_width))

    # Метод отрисовки
    def print_field(self):
        # Рисование сетки
        for x in range(0, WIDTH, TILE_SIZE):
            pg.draw.line(self.win, pg.Color('dimgray'), (x, 0), (x, HEIGHT))
        for y in range(0, HEIGHT, TILE_SIZE):
            pg.draw.line(self.win, pg.Color('dimgray'), (0, y), (WIDTH, y))

        # Получаем живые клетки и рисуем их
        living_cells = np.flip(np.transpose(np.where(self.current_field == 1)))
        for cell in living_cells:
            x, y = cell
            pg.draw.rect(self.win, pg.Color('pink'), (x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE))
        # Обновляем текущее поле
        self.current_field = self.next_step_field.copy()

    # Метод обновления поля
    def update_field(self):
        # Просчитываем следующий ход
        self.next_step_field = _update_field(field=self.current_field, width=self.field_width, height=self.field_height)

    # Метод рисования клеток игроком
    def paint(self):
        # Переводим координаты мышки в локальные координаты поля
        x = pg.mouse.get_pos()[0] // TILE_SIZE
        y = pg.mouse.get_pos()[1] // TILE_SIZE

        # Если мышка внутри окна
        if 0 <= x <= self.field_width and 0 <= y <= self.field_height:
            # На ЛКМ закрышиваем
            if pg.mouse.get_pressed()[0]:
                self.next_step_field[y][x] = 1
            # На ПКМ очищаем
            elif pg.mouse.get_pressed()[2]:
                self.next_step_field[y][x] = 0

    # Основной метод
    def run(self):
        while self.running:
            # Считаем Delta time
            dt = self.clock.tick(FPS) / 1000

            # Обрабатываем ивенты
            self.process_event()

            # Если игрок начал рисовать, то обрабатываем это
            if self.painting:
                self.paint()

            # Если игра не на паузе или игрок хочет сделать шаг
            if not self.pause or self.do_step:
                # Считаем задуржку между обработкой логики
                self.update_timer += dt

                # Ограничение обработки логики для понимания происходящего на экране
                if self.update_timer >= (1.0 / LOGIC_UPDATE_TIME) or self.do_step:
                    # Шаг требовался один, так что выключаем его
                    self.do_step = False
                    self.update_timer = 0
                    self.update_field()

            # Отрисовка поля
            self.print_field()

            # Обновление экрана
            pg.display.flip()
            self.win.fill(pg.Color('black')) 

        pg.quit()

''' 
Это функиця обработки логики вынесена отдельно т.к.
Если игрок захочет повысить скорость обработки логики и уменьшить размер клеток/увеличть разрешение
Игра начинает сильно тормозить.
Для оптимизации я использовал модель numba.
Однако он работает только с функциями, и переписывать логику игры из ООП я не стал.
'''
@njit(fastmath=True)
def _update_field(field, width, height):
    # Создаём пустой список, в котором будет итог вычислений
    res = np.empty(shape=field.shape)
    for x in range(width):
        for y in range(height):
            # Счётчик клеток, окружающих текущюю
            count = 0
            
            # Здесь проходимся по окружающим клеткам.
            # Если координаты клетки - (x, y)
            # То координаты окружающих клеток лежат в [(x - 1, y - 1), (x + 1, y + 1)]
            for i in range(y - 1, y + 2):
                for j in range(x - 1, x + 2):
                    # Проверяем, что не вышли за пределы массива
                    if 0 <= i <= height and 0 <= j <= width:
                        # Если клетка жива
                        if field[i][j] == 1:
                            count += 1
            
            # Если текущая клетка была жива
            if field[y][x]:
                # Её учитывать не надо
                count -= 1
                # Она остаётся живой при 2 или 3 соседях и умирает в других случаях
                res[y][x] = 1 if count == 2 or count == 3 else 0
            # Если клетка была мертва
            else:
                # Она оживает при 3 соседях
                res[y][x] = 1 if count == 3 else 0

    return res