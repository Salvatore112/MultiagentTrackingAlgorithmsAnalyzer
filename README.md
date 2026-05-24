![CI: Linter](https://github.com/muffin003/Stochastic-Optimization-Benchmark/actions/workflows/lint.yml/badge.svg)
![CI: Tests](https://github.com/muffin003/Stochastic-Optimization-Benchmark/actions/workflows/test.yml/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# System for analysis of multi-agent algorithms for objects tracking

This system is designed to analyze and compare different multi-agent tracking algorithms

## Requirements

- **Python 3.12+**

## Build
```bash
git clone https://github.com/Salvatore112/MultiagentTrackingAlgorithms.git
cd MultiagentTrackingAlgorithms
python -m venv .venv
. .venv/bin/activate
pip install -r req.txt
```

## Launch
```bash
python manage.py migrate
python manage.py runserver
```
The application will start at `http://127.0.0.1:8000/` by default.

## Usage
After you start the server, you can go to `http://127.0.0.1:8000/`. The setup page will open where you can choose algorithms and configure the simulation.
In this example there would be 3 sensors observing 4 targets. The simulation will generate 200 seconds of the target's movements and their distances to each sensor at each moment. 
The distances will be noise. The noise is uniformly genereated from the interval [-0.5, 0.5]
<img width="1170" height="1344" alt="setup" src="https://github.com/user-attachments/assets/3faf41a9-c75e-43e2-9dab-b141c3b3a978" />

After everything is configured, the user can start the simulation. The chosen algorithms will be ran and the plots of the estimates of each sensor will be shown as well as the plot of the error evolution. You can then see how each sensor observed targets and how its error changed.
<img width="1775" height="1181" alt="mut_left_part" src="https://github.com/user-attachments/assets/f7b22366-3bb1-4d7a-8ef5-891640b19fac" />
<img width="1777" height="1181" alt="mut_right_part" src="https://github.com/user-attachments/assets/c429c39f-bc67-4ec3-926e-2bb6ea169b59" />


## Contact us

If you want to help the project you may open a pull request. You can contact [@me](https://t.me/unacerveza1) to ask any questions or give an advice
