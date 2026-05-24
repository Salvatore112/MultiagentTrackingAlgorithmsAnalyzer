![CI: Linter](https://github.com/muffin003/Stochastic-Optimization-Benchmark/actions/workflows/lint.yml/badge.svg)
![CI: Tests](https://github.com/muffin003/Stochastic-Optimization-Benchmark/actions/workflows/test.yml/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Multiagent Tracking Algorithms Analyzer

Данная система предназначена для анализа и сравнения мультиагентных алгоритмов трекинга.

## Требования

- **Python 3.12+**

## Сборка
```bash
git clone https://github.com/Salvatore112/MultiagentTrackingAlgorithms.git
cd MultiagentTrackingAlgorithms
python -m venv .venv
. .venv/bin/activate
pip install -r req.txt
```

## Запуск
```bash
python manage.py migrate
python manage.py runserver
```
По умолчанию приложение будет запущено по адресу `http://127.0.0.1:8000/`.

## Usage
После запуска сервера вы можете перейти по адресу `http://127.0.0.1:8000/`. Откроется страница настройки, где можно выбрать алгоритмы и сконфигурировать симуляцию.
В данном примере 3 датчика наблюдают за 4 целями. Симуляция сгенерирует 200 секунд движения целей и расстояния от них до каждого датчика в каждый момент времени.
<img width="1170" height="1344" alt="setup" src="https://github.com/user-attachments/assets/3faf41a9-c75e-43e2-9dab-b141c3b3a978" />

После того как всё настроено, пользователь может запустить симуляцию. Выбранные алгоритмы будут выполнены, после чего отобразятся графики оценок каждого датчика, а также график изменения ошибки. Вы сможете увидеть, как каждый датчик наблюдал за целями и как менялась его ошибка.

<img width="1775" height="1181" alt="mut_left_part" src="https://github.com/user-attachments/assets/f7b22366-3bb1-4d7a-8ef5-891640b19fac" />
<img width="1777" height="1181" alt="mut_right_part" src="https://github.com/user-attachments/assets/c429c39f-bc67-4ec3-926e-2bb6ea169b59" />


## Связь с нами

Если вы хотите помочь проекту, вы можете открыть pull request. Вы можете связаться со [@мной](https://t.me/unacerveza1), чтобы задать любые вопросы или дать совет.
