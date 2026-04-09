import numpy as np
import pygame as pg


class DoublePendulum:
    def __init__(self, m1, m2, L1, L2, theta1, theta2, omega1=0.0, omega2=0.0):
        # physical constants of the system
        self.m1 = m1
        self.m2 = m2
        self.L1 = L1
        self.L2 = L2
        self.g = 9.81

        # starting parameters
        self.theta1 = theta1
        self.theta2 = theta2
        self.omega1 = omega1
        self.omega2 = omega2

    def get_cartesian_coords(self):
        x1 = self.L1 * np.sin(self.theta1)
        y1 = self.L1 * np.cos(self.theta1)

        x2 = self.L1 * np.sin(self.theta1) + self.L2 * np.sin(self.theta2)
        y2 = self.L1 * np.cos(self.theta1) + self.L2 * np.cos(self.theta2)

        return x1, y1, x2, y2

    def get_accelerations(self, theta1, theta2, omega1, omega2):
        delta_theta = theta1 - theta2

        den = 2 * self.m1 + self.m2 - self.m2 * np.cos(2 * delta_theta)

        num_11 = self.g * (2 * self.m1 + self.m2) * np.sin(theta1)
        num_12 = self.m2 * self.g * np.sin(theta1 - 2 * theta2)
        num_13 = 2 * np.sin(delta_theta) * self.m2 * (
                omega2 ** 2 * self.L2 + omega1 ** 2 * self.L1 * np.cos(delta_theta))

        num1 = -(num_11 + num_12 + num_13)
        alpha1 = num1 / (self.L1 * den)

        num_21 = omega1 ** 2 * self.L1 * (self.m1 + self.m2)
        num_22 = self.g * (self.m1 + self.m2) * np.cos(theta1)
        num_23 = omega2 ** 2 * self.L2 * self.m2 * np.cos(delta_theta)
        num2 = 2 * np.sin(delta_theta) * (num_21 + num_22 + num_23)

        alpha2 = num2 / (self.L2 * den)

        return alpha1, alpha2

    def get_derivatives(self, state):
        theta1, theta2, omega1, omega2 = state

        alpha1, alpha2 = (self.get_accelerations(theta1, theta2, omega1, omega2))

        return np.array([omega1, omega2, alpha1, alpha2])

    def step(self, dt):
        current_state = np.array([self.theta1, self.theta2, self.omega1, self.omega2])
        k1 = self.get_derivatives(current_state)
        k2 = self.get_derivatives(current_state + 0.5 * dt * k1)
        k3 = self.get_derivatives(current_state + 0.5 * dt * k2)
        k4 = self.get_derivatives(current_state + dt * k3)

        new_state = current_state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        self.theta1, self.theta2, self.omega1, self.omega2 = new_state


# --- הגדרות המסך והמערכת ---
WIDTH, HEIGHT = 800, 600
ORIGIN = (WIDTH // 2, HEIGHT // 3)  # נקודת התלייה של המטוטלת

pg.init()
screen = pg.display.set_mode((WIDTH, HEIGHT))
pg.display.set_caption("Double Pendulum Simulation")
clock = pg.time.Clock()

# 1. יצירת האובייקט של המטוטלת (עם אורכים מותאמים לפיקסלים במסך)
# m1=10, m2=10, L1=150, L2=150, theta1=np.pi/2, theta2=np.pi/2
pendulum = DoublePendulum(20, 20, 150, 150, np.pi / 2, np.pi / 2)

running = True
dt = 0.05  # גודל צעד הזמן (אפשר לשחק עם זה אחר כך כדי להאיץ/להאט)

trail = []

while running:
    # --- טיפול באירועים ---
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False

    # --- עדכון הפיזיקה ---
    pendulum.step(dt)

    # --- ציור המסך ---
    screen.fill((255, 255, 255))  # ניקוי המסך לצבע לבן

    # המשימה שלך מתחילה כאן!
    x1, y1, x2, y2 = pendulum.get_cartesian_coords()
    x1 = x1 + ORIGIN[0]
    x2 = x2 + ORIGIN[0]
    y1 = y1 + ORIGIN[1]
    y2 = y2 + ORIGIN[1]

    if len(trail) > 150:
        trail.pop(0)
    trail.append((x2, y2))

    if len(trail) > 1:
        pg.draw.lines(screen, (255, 0, 0), False, trail, 2)

    pg.draw.line(screen, (0, 255, 0), ORIGIN, (x1, y1), 5)
    pg.draw.line(screen, (0, 255, 0), (x1, y1), (x2, y2), 5)
    pg.draw.circle(screen, (0, 255, 0), (x1, y1), 10)
    pg.draw.circle(screen, (0, 255, 0), (x2, y2), 10)

    pg.display.flip()  # עדכון התצוגה למסך
    clock.tick(60)  # הגבלת קצב הריצה ל-60 פריימים בשנייה

pg.quit()
