# DIPLOMA project

# Задача
Реализация системы сегментации естественных типов подстилающих поверхностей на изображении с камеры робота-доставщика в городской среде.

# Датасет
Были самостоятельно собраны кадры различных подстилающих поверхностей в условиях зимней городской среды. 
В ходе предобработки и аннотации данных были определены 6 основных типов подстилающих поверхностей.

<img width="504" alt="image" src="https://github.com/grigoriiott/DIPLOMA_proj/assets/92350053/cbc16f2f-77df-47b6-9756-0032eb5eda8e">

Обработанный датасет был опубликован на kaggle: https://www.kaggle.com/datasets/grigoriiott/urban-underlying-surfaces-in-winter-dataset

# Модель
<img width="814" alt="image" src="https://github.com/grigoriiott/DIPLOMA_proj/assets/92350053/7a972d32-0c9c-425f-8891-f59c953ff6b5">

В качетсве фреймворка для работы был выбран Detectron 2. Для реализации сегментации была выбрана модель R50-FPN, которая являлась самой "легкой" из MASK R-CNN моделей, представленных в Detectron 2.

# Обучение и валидация 
<img width="1101" alt="image" src="https://github.com/grigoriiott/DIPLOMA_proj/assets/92350053/7d8de412-7b96-49c0-b874-636466750a3e">

В ходе обучения удалось достичь представленных ниже метрик: 
<img width="910" alt="image" src="https://github.com/grigoriiott/DIPLOMA_proj/assets/92350053/fcbdcde9-9712-46cd-8c6e-ed32bf3412df">

# Оптимизация 
Дополнительно были внедрены алгоритмы оптимизации получаемого на выходе изображения (Устранены коллизии, исчезновения сегментов между кадрами).
<img width="526" alt="image" src="https://github.com/grigoriiott/DIPLOMA_proj/assets/92350053/219bf1e8-7d5d-4cf4-af6d-af79c27a8645">

# Прмиер работы
https://user-images.githubusercontent.com/92350053/236374680-86ebc91f-2430-46aa-9291-eb36de6fb8b1.mov

