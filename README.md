# PC-ACO
Pairwise Comparisons-based Ant Colony Optimization

* Algorithm for solving multiple objective continuous optimization problems which allows taking into account user preferences in the form of pairwise comparisons.
* Based on the ant colony metaheuristic.
* The optimization is steered with user preferences towards more desired solutions.
* The preferences are expressed by pairwise comparisons which are translated into a value function preference model.
* 5 value function approaches are implemented and compared.
* The performance of the algorithm is tested on a set of benchmark problems.

## Requirements
Available in `requirements.txt`. Install all required packages by running:
```
pip install -r requirements.txt
```

## Running guide

### Run with default parameters:
```
python pcaco.py
```
### Show help about available parameters:
```
python pcaco.py -h
```
### Run tests:
```
python -m unittest discover -s test/
```

## Plot of an example run for WFG1
![MDVF plot for WFG1](https://github.com/adam-handke/pc-aco/blob/main/example.png?raw=true)
