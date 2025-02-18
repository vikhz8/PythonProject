# #библиоткека Matplotlib:
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # Шаг 1: Считываем данные из Excel
# file_path = 'data.xlsx'  # Путь к файлу Excel
# df = pd.read_excel(file_path)
#
# # Проверяем структуру данных
# print(df.head())  # Вывод первых строк для проверки
#
# # Шаг 2: Построение графика
# plt.figure(figsize=(8, 6))  # Размер окна графика
#
# # Используем данные из DataFrame
# plt.plot(df['x'], df['y'], marker='o', label='Зависимость y от x')
#
# # Настройка графика
# plt.title('График из Excel')
# plt.xlabel('Ось X')
# plt.ylabel('Ось Y')
# plt.legend()
# plt.grid(True)  # Включить сетку
# # Шаг 3: Отображаем график
# plt.show()


#создание графика по таблице excel
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # Шаг 1: Считываем данные из Excel
# file_path = 'x,y.xlsx'  # Путь к файлу Excel
# df = pd.read_excel(file_path)
#
# # Шаг 2: Построение графика
# plt.figure(figsize=(7, 3))  # Размер окна графика
#
# # Используем данные из DataFrame
# plt.plot(df['x'], df['y'], marker='o', label='Зависимость y от x')
#
# # Настройка графика
# plt.title('График из Excel')
# plt.xlabel('Ось X')
# plt.ylabel('Ось Y')
# plt.legend()
# plt.grid(True)  # Включить сетку
#
# # Шаг 3: Отображаем график
# plt.show()

#
# import matplotlib.pyplot as plt
# import numpy as np
#
# plt.style.use('_mpl-gallery')   #mpl-gallery определнный график
#
# # make data:
# x = 2 + np.arange(8)
# y = [3.5, 5.7, 2.5, 4.9, 6.5, 6.6, 2.6, 3.0]
# # plot
# fig, ax = plt.subplots()
#
# ax.bar(x, y, width=1, edgecolor="white", linewidth=0.7)
#
# ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
#        ylim=(0, 8), yticks=np.arange(1, 8))
#
# plt.show()


#построение маятника
# from numpy import sin, cos
# import numpy as np
# import matplotlib.pyplot as plt
# import scipy.integrate as integrate
# import matplotlib.animation as animation
#
# G = 9.8  # acceleration due to gravity, in m/s^2
# L1 = 1.0  # length of pendulum 1 in m
# L2 = 1.0  # length of pendulum 2 in m
# M1 = 1.0  # mass of pendulum 1 in kg
# M2 = 1.0  # mass of pendulum 2 in kg
#
#
# def derivs(state, t):
#
#     dydx = np.zeros_like(state)
#     dydx[0] = state[1]
#
#     del_ = state[2] - state[0]
#     den1 = (M1 + M2)*L1 - M2*L1*cos(del_)*cos(del_)
#     dydx[1] = (M2*L1*state[1]*state[1]*sin(del_)*cos(del_) +
#                M2*G*sin(state[2])*cos(del_) +
#                M2*L2*state[3]*state[3]*sin(del_) -
#                (M1 + M2)*G*sin(state[0]))/den1
#
#     dydx[2] = state[3]
#
#     den2 = (L2/L1)*den1
#     dydx[3] = (-M2*L2*state[3]*state[3]*sin(del_)*cos(del_) +
#                (M1 + M2)*G*sin(state[0])*cos(del_) -
#                (M1 + M2)*L1*state[1]*state[1]*sin(del_) -
#                (M1 + M2)*G*sin(state[2]))/den2
#
#     return dydx
#
# # create a time array from 0..100 sampled at 0.05 second steps
# dt = 0.05
# t = np.arange(0.0, 20, dt)
#
# # th1 and th2 are the initial angles (degrees)
# # w10 and w20 are the initial angular velocities (degrees per second)
# th1 = 120.0
# w1 = 0.0
# th2 = -10.0
# w2 = 0.0
#
# # initial state
# state = np.radians([th1, w1, th2, w2])
#
# # integrate your ODE using scipy.integrate.
# y = integrate.odeint(derivs, state, t)
#
# x1 = L1*sin(y[:, 0])
# y1 = -L1*cos(y[:, 0])
#
# x2 = L2*sin(y[:, 2]) + x1
# y2 = -L2*cos(y[:, 2]) + y1
#
# fig = plt.figure()
# ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
# ax.grid()
#
# line, = ax.plot([], [], 'o-', lw=2)
# time_template = 'time = %.1fs'
# time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
#
#
# def init():
#     line.set_data([], [])
#     time_text.set_text('')
#     return line, time_text
#
#
# def animate(i):
#     thisx = [0, x1[i], x2[i]]
#     thisy = [0, y1[i], y2[i]]
#
#     line.set_data(thisx, thisy)
#     time_text.set_text(time_template % (i*dt))
#     return line, time_text
#
# ani = animation.FuncAnimation(fig, animate, np.arange(1, len(y)),
#                               interval=25, blit=True, init_func=init)
#
# # ani.save('double_pendulum.mp4', fps=15)
# plt.show()


# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
#
# # Загружаем данные
# df = pd.read_excel("x,y.xlsx")  # Укажи свой файл
# X = df["X"]
# Y = df["Y"]
# Z = df["Z"]
# # Построение 3D-точек
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X, Y, Z, c=Z, cmap='coolwarm')  # Точки окрашены по значению Z
#
# plt.show()


# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
#
#
# df = pd.read_excel("x,y.xlsx")  # Укажи свой файл
# # Преобразуем данные в сетку
# X = np.unique(df["X"])  # Уникальные значения X
# Y = np.unique(df["Y"])  # Уникальные значения Y
# X, Y = np.meshgrid(X, Y)  # Создаем сетку
#
# # Создаем матрицу Z (значения Z должны соответствовать (X, Y))
# Z = df.pivot(index="Y", columns="X", values="Z").values
#
# # Создаем 3D-график
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# # Рисуем поверхность
# ax.plot_surface(X, Y, Z, cmap="plasma")
#
# # Добавляем подписи
# ax.set_xlabel("X ось")
# ax.set_ylabel("Y ось")
# ax.set_zlabel("Z ось")
#
# plt.show()



# from skimage import data
# import matplotlib.pyplot as plt
#
# image = data.camera()
# plt.imshow(image, cmap='gray')
# plt.axis('off')
# plt.title("Camera")
# plt.show()

import ssl

ssl._create_default_https_context = ssl._create_stdlib_context

# from skimage import io
# import matplotlib.pyplot as plt
#
# # Загрузка изображения с URL или локального файла
# image = io.imread("https://www.sunny-cat.ru/datas/users/1-fransis028.jpg")
#
# # Отображение изображения
# plt.imshow(image)
# plt.axis('off')  # Убираем оси для красоты
# plt.title("Original Image")
# plt.show()

from skimage import io, color
import matplotlib.pyplot as plt

# Загрузка изображения
image = io.imread("https://www.sunny-cat.ru/datas/users/1-fransis028.jpg")

# Преобразование изображения в оттенки серого
gray_image = color.rgb2gray(image)

# Отображение изображения
plt.imshow(gray_image, cmap='gray')
plt.axis('off')
plt.title("Gray Cat")
plt.show()