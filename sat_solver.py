import random
import functools
import argparse
from collections import defaultdict
import json
import os
import timeit
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def parse_dimacs_cnf(file_path):
    def conv(clause):
        my_clause = []
        for literal in clause:
            if literal < 0:
                my_clause.append((abs(literal), -1))
            else:
                my_clause.append((abs(literal), 1))
        return my_clause


    clauses = []

    with open(file_path, 'r') as f:
        clause = []
        for line in f.readlines():
            if line[0] == 'c':
                continue
            elif line[0] == 'p':
                _, _, variables, num_clauses = line.split()
                variables, num_clauses= int(variables), int(num_clauses)                
                continue
            else:
                for literal in line.split():
                    if literal == '0':
                        clauses.append(clause)
                        clause = []
                    else:
                        clause.append(int(literal))
        if clause:
            clauses.append(clause)

    my_clauses = [conv(clause) for clause in clauses]

    return my_clauses, list(range(1, variables+1))

def to_dimacs(file_path, clauses):
    num_clauses = len(clauses)
    vars = set()
    for clause in clauses:
        for (prop, sign) in clause:
            vars.add((prop, abs(sign)))
    num_vars = len(vars)

    next_var_name = 1
    var_name_map = {}
    with open(file_path, 'w') as f:
        f.write(f'p cnf {num_vars} {num_clauses}\n')

        for clause in clauses:
            output_clause = []
            for (prop, sign) in clause:
                prop_norm = (prop, abs(sign))
                prop_name = None
                if prop_norm not in var_name_map:
                    var_name_map[prop_norm] = next_var_name
                    next_var_name += 1
                prop_name = var_name_map[prop_norm]
                output_clause.append(f"{sign * prop_name}")
            output_clause.append('0')

            f.write(' '.join(output_clause) + '\n')
    if next_var_name - 1 != num_vars:
        raise Exception('num_vars does not match number of variables in clauses')
    return var_name_map

# DPLL with Unit-preference rule and splitting rule TODO maybe try using sets for lookup
def solve_first_clause(clauses):
    # returns an updated version of clauses where
    # if a prop is true, clauses it's in are removed, if a prop is false, it is removed
    # from the clause and if a unit clauses becomes empty as a result, return None
    def simplify(prop, val, clauses):
        if prop[-1] < 0:
            raise Exception('prop should be positive')
        negated_prop = (prop[0], -prop[-1])
        true_prop = None
        false_prop = None
        if val:
            true_prop = prop
            false_prop = negated_prop
        else:
            true_prop = negated_prop
            false_prop = prop

        new_clauses = []
        for clause in clauses:
            if true_prop in clause:
                # remove clause, it shortcircuits to true
                continue
            elif false_prop in clause:
                # remove prop from clause
                new_clause = [prop for prop in clause if prop != false_prop]
                # if clause is empty, shortcircuits to false
                if not new_clause:
                    return None
                new_clauses.append(new_clause)
            else:
                new_clauses.append(clause[::]) # don't need to copy if we never mutate

        return new_clauses

    # use backtracking
    # TODO use bitmask for assigment and lookup instead of copying list
    # elems: (clauses, assignment)
    queue = [(clauses, {})]
    while queue:
        clauses, assignments = queue.pop()
        if not clauses:
            return assignments
        # unit-preference rule (TODO do all unit rules at once and have map that tracks size of clauses)
        unit_clauses = [clause for clause in clauses if len(clause) == 1]
        if unit_clauses:
            unit_clause = unit_clauses[0]
            unit_prop = unit_clause[0]
            unit_prop_norm = (unit_prop[0], abs(unit_prop[1]))
            unit_val_norm = True if unit_prop[1] == 1 else False
            new_clauses = simplify(unit_prop_norm, unit_val_norm, clauses)
            if new_clauses is None:
                continue
                
            if unit_prop in assignments:
                raise Exception('unit_prop already assigned')
            new_assigments = {**assignments, unit_prop_norm: unit_val_norm}

            queue.append((new_clauses, new_assigments))
        else:
            # splitting rule
            prop = clauses[0][0]
            prop_norm = (prop[0], abs(prop[1]))
            val_norm = True if prop[1] == 1 else False
            new_clauses = simplify(prop_norm, val_norm, clauses)
            if new_clauses is not None:
                queue.append((new_clauses, {**assignments, prop_norm: val_norm}))
            new_clauses = simplify(prop_norm, not val_norm, clauses)
            if new_clauses is not None:
                queue.append((new_clauses, {**assignments, prop_norm: not val_norm}))
    return None

def solve_random(clauses, num_dpll_calls_arr, assignments_ptr):
    start = time.time()
    num_dpll_calls_arr[0] = 0
    seed_value = 37
    random.seed(seed_value)
    # returns an updated version of clauses where
    # if a prop is true, clauses it's in are removed, if a prop is false, it is removed
    # from the clause and if a unit clauses becomes empty as a result, return None
    def simplify(prop, val, clauses, props_norm):
        if prop[-1] < 0:
            raise Exception('prop should be positive')
        negated_prop = (prop[0], -prop[-1])
        true_prop = None
        false_prop = None
        if val:
            true_prop = prop
            false_prop = negated_prop
        else:
            true_prop = negated_prop
            false_prop = prop

        new_clauses = []
        for clause in clauses:
            if true_prop in clause:
                # remove clause, it shortcircuits to true
                continue
            elif false_prop in clause:
                # remove prop from clause
                new_clause = [prop for prop in clause if prop != false_prop]
                # if clause is empty, shortcircuits to false
                if not new_clause:
                    return None
                new_clauses.append(new_clause)
            else:
                new_clauses.append(clause[::]) # don't need to copy if we never mutate

        return new_clauses, [p for p in props_norm if p != prop]

    props_norm_init = set()
    for clause in clauses:
        for prop in clause:
            props_norm_init.add((prop[0], abs(prop[1])))

    # use backtracking
    # TODO use bitmask for assigment and lookup instead of copying list
    # elems: (clauses, assignment, unit_props)
    queue = [(clauses, {}, list(props_norm_init))]
    while queue:
        num_dpll_calls_arr[0] += 1
        clauses, assignments, props_norm = queue.pop()
        if not clauses:
            assignments_ptr[0] = assignments
            return assignments
        if time.time() - start > 3:
            num_dpll_calls_arr[0] = -1
            return None
        # unit-preference rule (TODO do all unit rules at once and have map that tracks size of clauses)
        unit_clauses = [clause for clause in clauses if len(clause) == 1]
        if unit_clauses:
            unit_clause = unit_clauses[0]
            unit_prop = unit_clause[0]
            unit_prop_norm = (unit_prop[0], abs(unit_prop[1]))
            unit_val_norm = True if unit_prop[1] == 1 else False
            temp = simplify(unit_prop_norm, unit_val_norm, clauses, props_norm)
            if temp is None:
                continue
            new_clauses, new_props_norm = temp
                
            if unit_prop in assignments:
                raise Exception('unit_prop already assigned')
            new_assigments = {**assignments, unit_prop_norm: unit_val_norm}

            queue.append((new_clauses, new_assigments, new_props_norm))
        else:
            # splitting rule
            # uniformly randomly select a proposition from props_norm
            prop_norm = random.choice(props_norm)
            val_norm = random.choice([True, False])
            temp = simplify(prop_norm, val_norm, clauses, props_norm)
            if temp is not None:
                new_clauses, new_props_norm = temp 
                queue.append((new_clauses, {**assignments, prop_norm: val_norm}, new_props_norm))
            temp = simplify(prop_norm, not val_norm, clauses, props_norm)
            if temp is not None:
                new_clauses, new_props_norm = temp
                queue.append((new_clauses, {**assignments, prop_norm: not val_norm}, new_props_norm))
    assignments_ptr[0] = None
    return None

def solve_2_clause(clauses, num_dpll_calls_arr, assignments_ptr):
    start = time.time()
    num_dpll_calls_arr[0] = 0
    # print(clauses[0])
    seed_value = 37
    random.seed(seed_value)
    # returns an updated version of clauses where
    # if a prop is true, clauses it's in are removed, if a prop is false, it is removed
    # from the clause and if a unit clauses becomes empty as a result, return None
    def simplify(prop, val, clauses, two_clause_occurrences):
        if prop[-1] < 0:
            raise Exception('prop should be positive')
        negated_prop = (prop[0], -prop[-1])
        true_prop = None
        false_prop = None
        if val:
            true_prop = prop
            false_prop = negated_prop
        else:
            true_prop = negated_prop
            false_prop = prop

        new_clauses = []
        new_two_clause_occurrences = {**two_clause_occurrences}
        for clause in clauses:
            if true_prop in clause:
                # remove clause, it shortcircuits to true
                if len(clause) == 2:
                    for prop in clause:
                        prop_norm = (prop[0], abs(prop[1]))
                        new_two_clause_occurrences[prop_norm] -= 1
                continue
            elif false_prop in clause:
                # remove prop from clause
                new_clause = [prop for prop in clause if prop != false_prop]
                # if clause is empty, shortcircuits to false
                if not new_clause:
                    return None
                if len(clause) == 2:
                    for prop in clause:
                        prop_norm = (prop[0], abs(prop[1]))
                        new_two_clause_occurrences[prop_norm] -= 1
                elif len(new_clause) == 2:
                    for prop in new_clause:
                        prop_norm = (prop[0], abs(prop[1]))
                        new_two_clause_occurrences[prop_norm] += 1
                new_clauses.append(new_clause)
            else:
                new_clauses.append(clause[::]) # don't need to copy if we never mutate

        return new_clauses, new_two_clause_occurrences

    # props_norm_init = set()
    # for clause in clauses:
    #     for prop in clause:
    #         props_norm_init.add((prop[0], abs(prop[1])))

    two_clause_occurrences_init = {}
    for clause in clauses:
        if len(clause) == 2:
            for prop in clause:
                prop_norm = (prop[0], abs(prop[1]))
                if prop_norm not in two_clause_occurrences_init:
                    two_clause_occurrences_init[prop_norm] = 0
                two_clause_occurrences_init[prop_norm] += 1
    for clause in clauses:
        for prop in clause:
            prop_norm = (prop[0], abs(prop[1]))
            if prop_norm not in two_clause_occurrences_init:
                two_clause_occurrences_init[prop_norm] = 0
    # if len(two_clause_occurrences_init) != len(props_norm_init):
    #     raise Exception('two_clause_occurrences_init and props_norm_init should have same length')

    # use backtracking
    # TODO use bitmask for assigment and lookup instead of copying list
    # elems: (clauses, assignment, unit_props)
    queue = [(clauses, {}, two_clause_occurrences_init)]
    while queue:
        num_dpll_calls_arr[0] += 1
        clauses, assignments, two_clause_occurrences = queue.pop()
        if not clauses:
            assignments_ptr[0] = assignments
            return assignments
        if time.time() - start > 3:
            num_dpll_calls_arr[0] = -1
            return None
        # unit-preference rule (TODO do all unit rules at once and have map that tracks size of clauses)
        unit_clauses = [clause for clause in clauses if len(clause) == 1]
        if unit_clauses:
            unit_clause = unit_clauses[0]
            unit_prop = unit_clause[0]
            unit_prop_norm = (unit_prop[0], abs(unit_prop[1]))
            unit_val_norm = True if unit_prop[1] == 1 else False
            temp = simplify(unit_prop_norm, unit_val_norm, clauses, two_clause_occurrences)
            if temp is None:
                continue
            new_clauses, new_two_clause_occurrences = temp
                
            if unit_prop in assignments:
                raise Exception('unit_prop already assigned')
            new_assigments = {**assignments, unit_prop_norm: unit_val_norm}

            queue.append((new_clauses, new_assigments, new_two_clause_occurrences))
        else:
            # splitting rule
            # pick the one in the most 2-clauses
            max_occurences = functools.reduce(max, two_clause_occurrences.values())
            most_common = [i for i in two_clause_occurrences if two_clause_occurrences[i] == max_occurences]
            # print(len(most_common))
            prop_norm = random.choice(most_common)
            val_norm = True
            temp = simplify(prop_norm, val_norm, clauses, two_clause_occurrences)
            if temp is not None:
                new_clauses, new_two_clause_occurrences = temp
                queue.append((new_clauses, {**assignments, prop_norm: val_norm}, new_two_clause_occurrences))
            temp = simplify(prop_norm, not val_norm, clauses, two_clause_occurrences)
            if temp is not None:
                new_clauses, new_two_clause_occurrences = temp
                queue.append((new_clauses, {**assignments, prop_norm: not val_norm}, new_two_clause_occurrences))
    assignments_ptr[0] = None
    return None

# 2-clause but tiebreaks by choosing the proposition with the largest difference betweeen
# positive and negative instances, and assigns to satisfy the more common parity
def solve_2_clause_with_parity_prop_true(clauses):
    # print(clauses[0])
    seed_value = 37
    random.seed(seed_value)
    # returns an updated version of clauses where
    # if a prop is true, clauses it's in are removed, if a prop is false, it is removed
    # from the clause and if a unit clauses becomes empty as a result, return None
    def simplify(prop, val, clauses, two_clause_occurrences):
        if prop[-1] < 0:
            raise Exception('prop should be positive')
        negated_prop = (prop[0], -prop[-1])
        true_prop = None
        false_prop = None
        if val:
            true_prop = prop
            false_prop = negated_prop
        else:
            true_prop = negated_prop
            false_prop = prop

        new_clauses = []
        new_two_clause_occurrences = {**two_clause_occurrences}
        for clause in clauses:
            if true_prop in clause:
                # remove clause, it shortcircuits to true
                if len(clause) == 2:
                    for prop in clause:
                        prop_norm = (prop[0], abs(prop[1]))
                        new_two_clause_occurrences[prop_norm] -= 1
                continue
            elif false_prop in clause:
                # remove prop from clause
                new_clause = [prop for prop in clause if prop != false_prop]
                # if clause is empty, shortcircuits to false
                if not new_clause:
                    return None
                if len(clause) == 2:
                    for prop in clause:
                        prop_norm = (prop[0], abs(prop[1]))
                        new_two_clause_occurrences[prop_norm] -= 1
                elif len(new_clause) == 2:
                    for prop in new_clause:
                        prop_norm = (prop[0], abs(prop[1]))
                        new_two_clause_occurrences[prop_norm] += 1
                new_clauses.append(new_clause)
            else:
                new_clauses.append(clause[::]) # don't need to copy if we never mutate

        return new_clauses, new_two_clause_occurrences

    # props_norm_init = set()
    # for clause in clauses:
    #     for prop in clause:
    #         props_norm_init.add((prop[0], abs(prop[1])))

    two_clause_occurrences_init = {}
    for clause in clauses:
        if len(clause) == 2:
            for prop in clause:
                prop_norm = (prop[0], abs(prop[1]))
                if prop_norm not in two_clause_occurrences_init:
                    two_clause_occurrences_init[prop_norm] = 0
                two_clause_occurrences_init[prop_norm] += 1
    # print(clauses)
    for clause in clauses:
        # print(clause)
        for prop in clause:
            # print(prop)
            prop_norm = (prop[0], abs(prop[1]))
            if prop_norm not in two_clause_occurrences_init:
                two_clause_occurrences_init[prop_norm] = 0
    # if len(two_clause_occurrences_init) != len(props_norm_init):
    #     raise Exception('two_clause_occurrences_init and props_norm_init should have same length')

    # use backtracking
    # TODO use bitmask for assigment and lookup instead of copying list
    # elems: (clauses, assignment, unit_props)
    queue = [(clauses, {}, two_clause_occurrences_init)]
    while queue:
        clauses, assignments, two_clause_occurrences = queue.pop()
        if not clauses:
            return assignments
        # unit-preference rule (TODO do all unit rules at once and have map that tracks size of clauses)
        unit_clauses = [clause for clause in clauses if len(clause) == 1]
        if unit_clauses:
            unit_clause = unit_clauses[0]
            unit_prop = unit_clause[0]
            unit_prop_norm = (unit_prop[0], abs(unit_prop[1]))
            unit_val_norm = True if unit_prop[1] == 1 else False
            temp = simplify(unit_prop_norm, unit_val_norm, clauses, two_clause_occurrences)
            if temp is None:
                continue
            new_clauses, new_two_clause_occurrences = temp
                
            if unit_prop in assignments:
                raise Exception('unit_prop already assigned')
            new_assigments = {**assignments, unit_prop_norm: unit_val_norm}

            queue.append((new_clauses, new_assigments, new_two_clause_occurrences))
        else:
            # splitting rule
            # pick the one in the most 2-clauses
            max_occurences = functools.reduce(max, two_clause_occurrences.values())
            most_common = [i for i in two_clause_occurrences if two_clause_occurrences[i] == max_occurences]
            # break ties by choosing the proposition with the largest difference betweeen positive and negative instances
            occurs = defaultdict(int)
            for clause in clauses:
                for prop in clause:
                    occurs[prop] += 1
            def parity_diff(prop_norm):
                negated_prop = (prop_norm[0], -prop_norm[-1])
                return abs(occurs[prop_norm] - occurs[negated_prop])
            best_prop = None
            for i in most_common:
                v = parity_diff(i)
                if best_prop is None or v > best_prop[1]:
                    best_prop = (i, v)
            # most_common.append(best_prop) # useless
            prop_norm = best_prop[0]
            # prop_norm = random.choice(most_common[:-1])
            # prop_norm = random.choice(most_common)
            # assign the value that makes the more common parity True
            val_norm = True if occurs[prop_norm] >= occurs[(prop_norm[0], -prop_norm[-1])] else False
            # val_norm = True
            temp = simplify(prop_norm, val_norm, clauses, two_clause_occurrences)
            if temp is not None:
                new_clauses, new_two_clause_occurrences = temp
                queue.append((new_clauses, {**assignments, prop_norm: val_norm}, new_two_clause_occurrences))
            temp = simplify(prop_norm, not val_norm, clauses, two_clause_occurrences)
            if temp is not None:
                new_clauses, new_two_clause_occurrences = temp
                queue.append((new_clauses, {**assignments, prop_norm: not val_norm}, new_two_clause_occurrences))
    return None

def solve_2_clause_least_with_parity_prop_true(clauses):
    # print(clauses[0])
    seed_value = 37
    random.seed(seed_value)
    # returns an updated version of clauses where
    # if a prop is true, clauses it's in are removed, if a prop is false, it is removed
    # from the clause and if a unit clauses becomes empty as a result, return None
    def simplify(prop, val, clauses, two_clause_occurrences):
        if prop[-1] < 0:
            raise Exception('prop should be positive')
        negated_prop = (prop[0], -prop[-1])
        true_prop = None
        false_prop = None
        if val:
            true_prop = prop
            false_prop = negated_prop
        else:
            true_prop = negated_prop
            false_prop = prop

        new_clauses = []
        new_two_clause_occurrences = {**two_clause_occurrences}
        for clause in clauses:
            if true_prop in clause:
                # remove clause, it shortcircuits to true
                if len(clause) == 2:
                    for prop in clause:
                        prop_norm = (prop[0], abs(prop[1]))
                        new_two_clause_occurrences[prop_norm] -= 1
                continue
            elif false_prop in clause:
                # remove prop from clause
                new_clause = [prop for prop in clause if prop != false_prop]
                # if clause is empty, shortcircuits to false
                if not new_clause:
                    return None
                if len(clause) == 2:
                    for prop in clause:
                        prop_norm = (prop[0], abs(prop[1]))
                        new_two_clause_occurrences[prop_norm] -= 1
                elif len(new_clause) == 2:
                    for prop in new_clause:
                        prop_norm = (prop[0], abs(prop[1]))
                        new_two_clause_occurrences[prop_norm] += 1
                new_clauses.append(new_clause)
            else:
                new_clauses.append(clause[::]) # don't need to copy if we never mutate

        return new_clauses, new_two_clause_occurrences

    # props_norm_init = set()
    # for clause in clauses:
    #     for prop in clause:
    #         props_norm_init.add((prop[0], abs(prop[1])))

    two_clause_occurrences_init = {}
    for clause in clauses:
        for prop in clause:
            prop_norm = (prop[0], abs(prop[1]))
            if prop_norm not in two_clause_occurrences_init:
                two_clause_occurrences_init[prop_norm] = 0
            if len(clause) == 2:
                two_clause_occurrences_init[prop_norm] += 1
    # print(clauses)
    for clause in clauses:
        # print(clause)
        for prop in clause:
            # print(prop)
            prop_norm = (prop[0], abs(prop[1]))
            if prop_norm not in two_clause_occurrences_init:
                two_clause_occurrences_init[prop_norm] = 0
    # if len(two_clause_occurrences_init) != len(props_norm_init):
    #     raise Exception('two_clause_occurrences_init and props_norm_init should have same length')

    # use backtracking
    # TODO use bitmask for assigment and lookup instead of copying list
    # elems: (clauses, assignment, unit_props)
    queue = [(clauses, {}, two_clause_occurrences_init)]
    while queue:
        clauses, assignments, two_clause_occurrences = queue.pop()
        if not clauses:
            return assignments
        # unit-preference rule (TODO do all unit rules at once and have map that tracks size of clauses)
        unit_clauses = [clause for clause in clauses if len(clause) == 1]
        if unit_clauses:
            unit_clause = unit_clauses[0]
            unit_prop = unit_clause[0]
            unit_prop_norm = (unit_prop[0], abs(unit_prop[1]))
            unit_val_norm = True if unit_prop[1] == 1 else False
            temp = simplify(unit_prop_norm, unit_val_norm, clauses, two_clause_occurrences)
            if temp is None:
                continue
            new_clauses, new_two_clause_occurrences = temp
                
            if unit_prop in assignments:
                raise Exception('unit_prop already assigned')
            new_assigments = {**assignments, unit_prop_norm: unit_val_norm}

            queue.append((new_clauses, new_assigments, new_two_clause_occurrences))
        else:
            # splitting rule
            # pick the one in the most 2-clauses
            def min_f(x, y):
                if x == 0 and y == 0:
                    return x
                if x == 0:
                    return y
                if y == 0:
                    return x
                return min(x, y)
            min_occurences = functools.reduce(min_f, two_clause_occurrences.values())
            least_common = [i for i in two_clause_occurrences if two_clause_occurrences[i] == min_occurences]
            # break ties by choosing the proposition with the largest difference betweeen positive and negative instances
            occurs = defaultdict(int)
            for clause in clauses:
                for prop in clause:
                    occurs[prop] += 1
            def parity_diff(prop_norm):
                negated_prop = (prop_norm[0], -prop_norm[-1])
                return abs(occurs[prop_norm] - occurs[negated_prop])
            best_prop = None
            for i in least_common:
                v = parity_diff(i)
                if best_prop is None or v > best_prop[1]:
                    best_prop = (i, v)
            # most_common.append(best_prop) # useless
            prop_norm = best_prop[0]
            # prop_norm = random.choice(most_common[:-1])
            # prop_norm = random.choice(most_common)
            # assign the value that makes the more common parity True
            val_norm = True if occurs[prop_norm] >= occurs[(prop_norm[0], -prop_norm[-1])] else False
            # val_norm = True
            temp = simplify(prop_norm, val_norm, clauses, two_clause_occurrences)
            if temp is not None:
                new_clauses, new_two_clause_occurrences = temp
                queue.append((new_clauses, {**assignments, prop_norm: val_norm}, new_two_clause_occurrences))
            temp = simplify(prop_norm, not val_norm, clauses, two_clause_occurrences)
            if temp is not None:
                new_clauses, new_two_clause_occurrences = temp
                queue.append((new_clauses, {**assignments, prop_norm: not val_norm}, new_two_clause_occurrences))
    return None

def solve_2_clause_least_with_parity_prop_false(clauses):
    # print(clauses[0])
    seed_value = 37
    random.seed(seed_value)
    # returns an updated version of clauses where
    # if a prop is true, clauses it's in are removed, if a prop is false, it is removed
    # from the clause and if a unit clauses becomes empty as a result, return None
    def simplify(prop, val, clauses, two_clause_occurrences):
        if prop[-1] < 0:
            raise Exception('prop should be positive')
        negated_prop = (prop[0], -prop[-1])
        true_prop = None
        false_prop = None
        if val:
            true_prop = prop
            false_prop = negated_prop
        else:
            true_prop = negated_prop
            false_prop = prop

        new_clauses = []
        new_two_clause_occurrences = {**two_clause_occurrences}
        for clause in clauses:
            if true_prop in clause:
                # remove clause, it shortcircuits to true
                if len(clause) == 2:
                    for prop in clause:
                        prop_norm = (prop[0], abs(prop[1]))
                        new_two_clause_occurrences[prop_norm] -= 1
                continue
            elif false_prop in clause:
                # remove prop from clause
                new_clause = [prop for prop in clause if prop != false_prop]
                # if clause is empty, shortcircuits to false
                if not new_clause:
                    return None
                if len(clause) == 2:
                    for prop in clause:
                        prop_norm = (prop[0], abs(prop[1]))
                        new_two_clause_occurrences[prop_norm] -= 1
                elif len(new_clause) == 2:
                    for prop in new_clause:
                        prop_norm = (prop[0], abs(prop[1]))
                        new_two_clause_occurrences[prop_norm] += 1
                new_clauses.append(new_clause)
            else:
                new_clauses.append(clause[::]) # don't need to copy if we never mutate

        return new_clauses, new_two_clause_occurrences

    # props_norm_init = set()
    # for clause in clauses:
    #     for prop in clause:
    #         props_norm_init.add((prop[0], abs(prop[1])))

    two_clause_occurrences_init = {}
    for clause in clauses:
        for prop in clause:
            prop_norm = (prop[0], abs(prop[1]))
            if prop_norm not in two_clause_occurrences_init:
                two_clause_occurrences_init[prop_norm] = 0
            if len(clause) == 2:
                two_clause_occurrences_init[prop_norm] += 1
    # print(clauses)
    for clause in clauses:
        # print(clause)
        for prop in clause:
            # print(prop)
            prop_norm = (prop[0], abs(prop[1]))
            if prop_norm not in two_clause_occurrences_init:
                two_clause_occurrences_init[prop_norm] = 0
    # if len(two_clause_occurrences_init) != len(props_norm_init):
    #     raise Exception('two_clause_occurrences_init and props_norm_init should have same length')

    # use backtracking
    # TODO use bitmask for assigment and lookup instead of copying list
    # elems: (clauses, assignment, unit_props)
    queue = [(clauses, {}, two_clause_occurrences_init)]
    while queue:
        clauses, assignments, two_clause_occurrences = queue.pop()
        if not clauses:
            return assignments
        # unit-preference rule (TODO do all unit rules at once and have map that tracks size of clauses)
        unit_clauses = [clause for clause in clauses if len(clause) == 1]
        if unit_clauses:
            unit_clause = unit_clauses[0]
            unit_prop = unit_clause[0]
            unit_prop_norm = (unit_prop[0], abs(unit_prop[1]))
            unit_val_norm = True if unit_prop[1] == 1 else False
            temp = simplify(unit_prop_norm, unit_val_norm, clauses, two_clause_occurrences)
            if temp is None:
                continue
            new_clauses, new_two_clause_occurrences = temp
                
            if unit_prop in assignments:
                raise Exception('unit_prop already assigned')
            new_assigments = {**assignments, unit_prop_norm: unit_val_norm}

            queue.append((new_clauses, new_assigments, new_two_clause_occurrences))
        else:
            # splitting rule
            # pick the one in the most 2-clauses
            def min_f(x, y):
                if x == 0 and y == 0:
                    return x
                if x == 0:
                    return y
                if y == 0:
                    return x
                return min(x, y)
            min_occurences = functools.reduce(min_f, two_clause_occurrences.values())
            least_common = [i for i in two_clause_occurrences if two_clause_occurrences[i] == min_occurences]
            # break ties by choosing the proposition with the largest difference betweeen positive and negative instances
            occurs = defaultdict(int)
            for clause in clauses:
                for prop in clause:
                    occurs[prop] += 1
            def parity_diff(prop_norm):
                negated_prop = (prop_norm[0], -prop_norm[-1])
                return abs(occurs[prop_norm] - occurs[negated_prop])
            best_prop = None
            for i in least_common:
                v = parity_diff(i)
                if best_prop is None or v > best_prop[1]:
                    best_prop = (i, v)
            # most_common.append(best_prop) # useless
            prop_norm = best_prop[0]
            # prop_norm = random.choice(most_common[:-1])
            # prop_norm = random.choice(most_common)
            # assign the value that makes the more common parity True
            val_norm = False if occurs[prop_norm] >= occurs[(prop_norm[0], -prop_norm[-1])] else True
            # val_norm = True
            temp = simplify(prop_norm, val_norm, clauses, two_clause_occurrences)
            if temp is not None:
                new_clauses, new_two_clause_occurrences = temp
                queue.append((new_clauses, {**assignments, prop_norm: val_norm}, new_two_clause_occurrences))
            temp = simplify(prop_norm, not val_norm, clauses, two_clause_occurrences)
            if temp is not None:
                new_clauses, new_two_clause_occurrences = temp
                queue.append((new_clauses, {**assignments, prop_norm: not val_norm}, new_two_clause_occurrences))
    return None

def solve_2_clause_with_parity_propleast_true(clauses):
    # print(clauses[0])
    seed_value = 37
    random.seed(seed_value)
    # returns an updated version of clauses where
    # if a prop is true, clauses it's in are removed, if a prop is false, it is removed
    # from the clause and if a unit clauses becomes empty as a result, return None
    def simplify(prop, val, clauses, two_clause_occurrences):
        if prop[-1] < 0:
            raise Exception('prop should be positive')
        negated_prop = (prop[0], -prop[-1])
        true_prop = None
        false_prop = None
        if val:
            true_prop = prop
            false_prop = negated_prop
        else:
            true_prop = negated_prop
            false_prop = prop

        new_clauses = []
        new_two_clause_occurrences = {**two_clause_occurrences}
        for clause in clauses:
            if true_prop in clause:
                # remove clause, it shortcircuits to true
                if len(clause) == 2:
                    for prop in clause:
                        prop_norm = (prop[0], abs(prop[1]))
                        new_two_clause_occurrences[prop_norm] -= 1
                continue
            elif false_prop in clause:
                # remove prop from clause
                new_clause = [prop for prop in clause if prop != false_prop]
                # if clause is empty, shortcircuits to false
                if not new_clause:
                    return None
                if len(clause) == 2:
                    for prop in clause:
                        prop_norm = (prop[0], abs(prop[1]))
                        new_two_clause_occurrences[prop_norm] -= 1
                elif len(new_clause) == 2:
                    for prop in new_clause:
                        prop_norm = (prop[0], abs(prop[1]))
                        new_two_clause_occurrences[prop_norm] += 1
                new_clauses.append(new_clause)
            else:
                new_clauses.append(clause[::]) # don't need to copy if we never mutate

        return new_clauses, new_two_clause_occurrences

    # props_norm_init = set()
    # for clause in clauses:
    #     for prop in clause:
    #         props_norm_init.add((prop[0], abs(prop[1])))

    two_clause_occurrences_init = {}
    for clause in clauses:
        if len(clause) == 2:
            for prop in clause:
                prop_norm = (prop[0], abs(prop[1]))
                if prop_norm not in two_clause_occurrences_init:
                    two_clause_occurrences_init[prop_norm] = 0
                two_clause_occurrences_init[prop_norm] += 1
    # print(clauses)
    for clause in clauses:
        # print(clause)
        for prop in clause:
            # print(prop)
            prop_norm = (prop[0], abs(prop[1]))
            if prop_norm not in two_clause_occurrences_init:
                two_clause_occurrences_init[prop_norm] = 0
    # if len(two_clause_occurrences_init) != len(props_norm_init):
    #     raise Exception('two_clause_occurrences_init and props_norm_init should have same length')

    # use backtracking
    # TODO use bitmask for assigment and lookup instead of copying list
    # elems: (clauses, assignment, unit_props)
    queue = [(clauses, {}, two_clause_occurrences_init)]
    while queue:
        clauses, assignments, two_clause_occurrences = queue.pop()
        if not clauses:
            return assignments
        # unit-preference rule (TODO do all unit rules at once and have map that tracks size of clauses)
        unit_clauses = [clause for clause in clauses if len(clause) == 1]
        if unit_clauses:
            unit_clause = unit_clauses[0]
            unit_prop = unit_clause[0]
            unit_prop_norm = (unit_prop[0], abs(unit_prop[1]))
            unit_val_norm = True if unit_prop[1] == 1 else False
            temp = simplify(unit_prop_norm, unit_val_norm, clauses, two_clause_occurrences)
            if temp is None:
                continue
            new_clauses, new_two_clause_occurrences = temp
                
            if unit_prop in assignments:
                raise Exception('unit_prop already assigned')
            new_assigments = {**assignments, unit_prop_norm: unit_val_norm}

            queue.append((new_clauses, new_assigments, new_two_clause_occurrences))
        else:
            # splitting rule
            # pick the one in the most 2-clauses
            max_occurrences = functools.reduce(max, two_clause_occurrences.values())
            most_common = [i for i in two_clause_occurrences if two_clause_occurrences[i] == max_occurrences]
            # break ties by choosing the proposition with the largest difference betweeen positive and negative instances
            occurs = defaultdict(int)
            for clause in clauses:
                for prop in clause:
                    occurs[prop] += 1
            def parity_diff(prop_norm):
                negated_prop = (prop_norm[0], -prop_norm[-1])
                return abs(occurs[prop_norm] - occurs[negated_prop])
            best_prop = None
            for i in most_common:
                v = parity_diff(i)
                if best_prop is None or v < best_prop[1]:
                    best_prop = (i, v)
            # most_common.append(best_prop) # useless
            prop_norm = best_prop[0]
            # prop_norm = random.choice(most_common[:-1])
            # prop_norm = random.choice(most_common)
            # assign the value that makes the more common parity True
            val_norm = True if occurs[prop_norm] >= occurs[(prop_norm[0], -prop_norm[-1])] else False
            # val_norm = True
            temp = simplify(prop_norm, val_norm, clauses, two_clause_occurrences)
            if temp is not None:
                new_clauses, new_two_clause_occurrences = temp
                queue.append((new_clauses, {**assignments, prop_norm: val_norm}, new_two_clause_occurrences))
            temp = simplify(prop_norm, not val_norm, clauses, two_clause_occurrences)
            if temp is not None:
                new_clauses, new_two_clause_occurrences = temp
                queue.append((new_clauses, {**assignments, prop_norm: not val_norm}, new_two_clause_occurrences))
    return None

def solve_2_clause_with_parity_propleast_false(clauses):
    # print(clauses[0])
    seed_value = 37
    random.seed(seed_value)
    # returns an updated version of clauses where
    # if a prop is true, clauses it's in are removed, if a prop is false, it is removed
    # from the clause and if a unit clauses becomes empty as a result, return None
    def simplify(prop, val, clauses, two_clause_occurrences):
        if prop[-1] < 0:
            raise Exception('prop should be positive')
        negated_prop = (prop[0], -prop[-1])
        true_prop = None
        false_prop = None
        if val:
            true_prop = prop
            false_prop = negated_prop
        else:
            true_prop = negated_prop
            false_prop = prop

        new_clauses = []
        new_two_clause_occurrences = {**two_clause_occurrences}
        for clause in clauses:
            if true_prop in clause:
                # remove clause, it shortcircuits to true
                if len(clause) == 2:
                    for prop in clause:
                        prop_norm = (prop[0], abs(prop[1]))
                        new_two_clause_occurrences[prop_norm] -= 1
                continue
            elif false_prop in clause:
                # remove prop from clause
                new_clause = [prop for prop in clause if prop != false_prop]
                # if clause is empty, shortcircuits to false
                if not new_clause:
                    return None
                if len(clause) == 2:
                    for prop in clause:
                        prop_norm = (prop[0], abs(prop[1]))
                        new_two_clause_occurrences[prop_norm] -= 1
                elif len(new_clause) == 2:
                    for prop in new_clause:
                        prop_norm = (prop[0], abs(prop[1]))
                        new_two_clause_occurrences[prop_norm] += 1
                new_clauses.append(new_clause)
            else:
                new_clauses.append(clause[::]) # don't need to copy if we never mutate

        return new_clauses, new_two_clause_occurrences

    # props_norm_init = set()
    # for clause in clauses:
    #     for prop in clause:
    #         props_norm_init.add((prop[0], abs(prop[1])))

    two_clause_occurrences_init = {}
    for clause in clauses:
        if len(clause) == 2:
            for prop in clause:
                prop_norm = (prop[0], abs(prop[1]))
                if prop_norm not in two_clause_occurrences_init:
                    two_clause_occurrences_init[prop_norm] = 0
                two_clause_occurrences_init[prop_norm] += 1
    # print(clauses)
    for clause in clauses:
        # print(clause)
        for prop in clause:
            # print(prop)
            prop_norm = (prop[0], abs(prop[1]))
            if prop_norm not in two_clause_occurrences_init:
                two_clause_occurrences_init[prop_norm] = 0
    # if len(two_clause_occurrences_init) != len(props_norm_init):
    #     raise Exception('two_clause_occurrences_init and props_norm_init should have same length')

    # use backtracking
    # TODO use bitmask for assigment and lookup instead of copying list
    # elems: (clauses, assignment, unit_props)
    queue = [(clauses, {}, two_clause_occurrences_init)]
    while queue:
        clauses, assignments, two_clause_occurrences = queue.pop()
        if not clauses:
            return assignments
        # unit-preference rule (TODO do all unit rules at once and have map that tracks size of clauses)
        unit_clauses = [clause for clause in clauses if len(clause) == 1]
        if unit_clauses:
            unit_clause = unit_clauses[0]
            unit_prop = unit_clause[0]
            unit_prop_norm = (unit_prop[0], abs(unit_prop[1]))
            unit_val_norm = True if unit_prop[1] == 1 else False
            temp = simplify(unit_prop_norm, unit_val_norm, clauses, two_clause_occurrences)
            if temp is None:
                continue
            new_clauses, new_two_clause_occurrences = temp
                
            if unit_prop in assignments:
                raise Exception('unit_prop already assigned')
            new_assigments = {**assignments, unit_prop_norm: unit_val_norm}

            queue.append((new_clauses, new_assigments, new_two_clause_occurrences))
        else:
            # splitting rule
            # pick the one in the most 2-clauses
            max_occurrences = functools.reduce(max, two_clause_occurrences.values())
            most_common = [i for i in two_clause_occurrences if two_clause_occurrences[i] == max_occurrences]
            # break ties by choosing the proposition with the largest difference betweeen positive and negative instances
            occurs = defaultdict(int)
            for clause in clauses:
                for prop in clause:
                    occurs[prop] += 1
            def parity_diff(prop_norm):
                negated_prop = (prop_norm[0], -prop_norm[-1])
                return abs(occurs[prop_norm] - occurs[negated_prop])
            best_prop = None
            for i in most_common:
                v = parity_diff(i)
                if best_prop is None or v < best_prop[1]:
                    best_prop = (i, v)
            # most_common.append(best_prop) # useless
            prop_norm = best_prop[0]
            # prop_norm = random.choice(most_common[:-1])
            # prop_norm = random.choice(most_common)
            # assign the value that makes the more common parity True
            val_norm = False if occurs[prop_norm] >= occurs[(prop_norm[0], -prop_norm[-1])] else True
            # val_norm = True
            temp = simplify(prop_norm, val_norm, clauses, two_clause_occurrences)
            if temp is not None:
                new_clauses, new_two_clause_occurrences = temp
                queue.append((new_clauses, {**assignments, prop_norm: val_norm}, new_two_clause_occurrences))
            temp = simplify(prop_norm, not val_norm, clauses, two_clause_occurrences)
            if temp is not None:
                new_clauses, new_two_clause_occurrences = temp
                queue.append((new_clauses, {**assignments, prop_norm: not val_norm}, new_two_clause_occurrences))
    return None

def solve_2_clause_with_parity_prop_false(clauses, num_dpll_calls_arr, assignments_ptr):
    start = time.time()
    num_dpll_calls_arr[0] = 0
    # print(clauses[0])
    seed_value = 37
    random.seed(seed_value)
    # returns an updated version of clauses where
    # if a prop is true, clauses it's in are removed, if a prop is false, it is removed
    # from the clause and if a unit clauses becomes empty as a result, return None
    def simplify(prop, val, clauses, two_clause_occurrences):
        if prop[-1] < 0:
            raise Exception('prop should be positive')
        negated_prop = (prop[0], -prop[-1])
        true_prop = None
        false_prop = None
        if val:
            true_prop = prop
            false_prop = negated_prop
        else:
            true_prop = negated_prop
            false_prop = prop

        new_clauses = []
        new_two_clause_occurrences = {**two_clause_occurrences}
        for clause in clauses:
            if true_prop in clause:
                # remove clause, it shortcircuits to true
                if len(clause) == 2:
                    for prop in clause:
                        prop_norm = (prop[0], abs(prop[1]))
                        new_two_clause_occurrences[prop_norm] -= 1
                continue
            elif false_prop in clause:
                # remove prop from clause
                new_clause = [prop for prop in clause if prop != false_prop]
                # if clause is empty, shortcircuits to false
                if not new_clause:
                    return None
                if len(clause) == 2:
                    for prop in clause:
                        prop_norm = (prop[0], abs(prop[1]))
                        new_two_clause_occurrences[prop_norm] -= 1
                elif len(new_clause) == 2:
                    for prop in new_clause:
                        prop_norm = (prop[0], abs(prop[1]))
                        new_two_clause_occurrences[prop_norm] += 1
                new_clauses.append(new_clause)
            else:
                new_clauses.append(clause[::]) # don't need to copy if we never mutate

        return new_clauses, new_two_clause_occurrences

    # props_norm_init = set()
    # for clause in clauses:
    #     for prop in clause:
    #         props_norm_init.add((prop[0], abs(prop[1])))

    two_clause_occurrences_init = {}
    for clause in clauses:
        if len(clause) == 2:
            for prop in clause:
                prop_norm = (prop[0], abs(prop[1]))
                if prop_norm not in two_clause_occurrences_init:
                    two_clause_occurrences_init[prop_norm] = 0
                two_clause_occurrences_init[prop_norm] += 1
    # print(clauses)
    for clause in clauses:
        # print(clause)
        for prop in clause:
            # print(prop)
            prop_norm = (prop[0], abs(prop[1]))
            if prop_norm not in two_clause_occurrences_init:
                two_clause_occurrences_init[prop_norm] = 0
    # if len(two_clause_occurrences_init) != len(props_norm_init):
    #     raise Exception('two_clause_occurrences_init and props_norm_init should have same length')

    # use backtracking
    # TODO use bitmask for assigment and lookup instead of copying list
    # elems: (clauses, assignment, unit_props)
    queue = [(clauses, {}, two_clause_occurrences_init)]
    while queue:
        num_dpll_calls_arr[0] += 1
        clauses, assignments, two_clause_occurrences = queue.pop()
        if not clauses:
            assignments_ptr[0] = assignments
            return assignments
        if time.time() - start > 3:
            num_dpll_calls_arr[0] = -1
            return None
        # unit-preference rule (TODO do all unit rules at once and have map that tracks size of clauses)
        unit_clauses = [clause for clause in clauses if len(clause) == 1]
        if unit_clauses:
            unit_clause = unit_clauses[0]
            unit_prop = unit_clause[0]
            unit_prop_norm = (unit_prop[0], abs(unit_prop[1]))
            unit_val_norm = True if unit_prop[1] == 1 else False
            temp = simplify(unit_prop_norm, unit_val_norm, clauses, two_clause_occurrences)
            if temp is None:
                continue
            new_clauses, new_two_clause_occurrences = temp
                
            if unit_prop in assignments:
                raise Exception('unit_prop already assigned')
            new_assigments = {**assignments, unit_prop_norm: unit_val_norm}

            queue.append((new_clauses, new_assigments, new_two_clause_occurrences))
        else:
            # splitting rule
            # pick the one in the most 2-clauses
            max_occurences = functools.reduce(max, two_clause_occurrences.values())
            most_common = [i for i in two_clause_occurrences if two_clause_occurrences[i] == max_occurences]
            # break ties by choosing the proposition with the largest difference betweeen positive and negative instances
            occurs = defaultdict(int)
            for clause in clauses:
                for prop in clause:
                    occurs[prop] += 1
            def parity_diff(prop_norm):
                negated_prop = (prop_norm[0], -prop_norm[-1])
                return abs(occurs[prop_norm] - occurs[negated_prop])
            best_prop = None
            for i in most_common:
                v = parity_diff(i)
                if best_prop is None or v > best_prop[1]:
                    best_prop = (i, v)
            # most_common.append(best_prop) # useless
            prop_norm = best_prop[0]
            # prop_norm = random.choice(most_common[:-1])
            # prop_norm = random.choice(most_common)
            # assign the value that makes the more common parity True
            val_norm = False if occurs[prop_norm] >= occurs[(prop_norm[0], -prop_norm[-1])] else True
            # val_norm = True
            temp = simplify(prop_norm, val_norm, clauses, two_clause_occurrences)
            if temp is not None:
                new_clauses, new_two_clause_occurrences = temp
                queue.append((new_clauses, {**assignments, prop_norm: val_norm}, new_two_clause_occurrences))
            temp = simplify(prop_norm, not val_norm, clauses, two_clause_occurrences)
            if temp is not None:
                new_clauses, new_two_clause_occurrences = temp
                queue.append((new_clauses, {**assignments, prop_norm: not val_norm}, new_two_clause_occurrences))
    assignments_ptr[0] = None
    return None

def solve_2_clause_with_parity_prop_always_true(clauses):
    # print(clauses[0])
    seed_value = 37
    random.seed(seed_value)
    # returns an updated version of clauses where
    # if a prop is true, clauses it's in are removed, if a prop is false, it is removed
    # from the clause and if a unit clauses becomes empty as a result, return None
    def simplify(prop, val, clauses, two_clause_occurrences):
        if prop[-1] < 0:
            raise Exception('prop should be positive')
        negated_prop = (prop[0], -prop[-1])
        true_prop = None
        false_prop = None
        if val:
            true_prop = prop
            false_prop = negated_prop
        else:
            true_prop = negated_prop
            false_prop = prop

        new_clauses = []
        new_two_clause_occurrences = {**two_clause_occurrences}
        for clause in clauses:
            if true_prop in clause:
                # remove clause, it shortcircuits to true
                if len(clause) == 2:
                    for prop in clause:
                        prop_norm = (prop[0], abs(prop[1]))
                        new_two_clause_occurrences[prop_norm] -= 1
                continue
            elif false_prop in clause:
                # remove prop from clause
                new_clause = [prop for prop in clause if prop != false_prop]
                # if clause is empty, shortcircuits to false
                if not new_clause:
                    return None
                if len(clause) == 2:
                    for prop in clause:
                        prop_norm = (prop[0], abs(prop[1]))
                        new_two_clause_occurrences[prop_norm] -= 1
                elif len(new_clause) == 2:
                    for prop in new_clause:
                        prop_norm = (prop[0], abs(prop[1]))
                        new_two_clause_occurrences[prop_norm] += 1
                new_clauses.append(new_clause)
            else:
                new_clauses.append(clause[::]) # don't need to copy if we never mutate

        return new_clauses, new_two_clause_occurrences

    # props_norm_init = set()
    # for clause in clauses:
    #     for prop in clause:
    #         props_norm_init.add((prop[0], abs(prop[1])))

    two_clause_occurrences_init = {}
    for clause in clauses:
        if len(clause) == 2:
            for prop in clause:
                prop_norm = (prop[0], abs(prop[1]))
                if prop_norm not in two_clause_occurrences_init:
                    two_clause_occurrences_init[prop_norm] = 0
                two_clause_occurrences_init[prop_norm] += 1
    # print(clauses)
    for clause in clauses:
        # print(clause)
        for prop in clause:
            # print(prop)
            prop_norm = (prop[0], abs(prop[1]))
            if prop_norm not in two_clause_occurrences_init:
                two_clause_occurrences_init[prop_norm] = 0
    # if len(two_clause_occurrences_init) != len(props_norm_init):
    #     raise Exception('two_clause_occurrences_init and props_norm_init should have same length')

    # use backtracking
    # TODO use bitmask for assigment and lookup instead of copying list
    # elems: (clauses, assignment, unit_props)
    queue = [(clauses, {}, two_clause_occurrences_init)]
    while queue:
        clauses, assignments, two_clause_occurrences = queue.pop()
        if not clauses:
            return assignments
        # unit-preference rule (TODO do all unit rules at once and have map that tracks size of clauses)
        unit_clauses = [clause for clause in clauses if len(clause) == 1]
        if unit_clauses:
            unit_clause = unit_clauses[0]
            unit_prop = unit_clause[0]
            unit_prop_norm = (unit_prop[0], abs(unit_prop[1]))
            unit_val_norm = True if unit_prop[1] == 1 else False
            temp = simplify(unit_prop_norm, unit_val_norm, clauses, two_clause_occurrences)
            if temp is None:
                continue
            new_clauses, new_two_clause_occurrences = temp
                
            if unit_prop in assignments:
                raise Exception('unit_prop already assigned')
            new_assigments = {**assignments, unit_prop_norm: unit_val_norm}

            queue.append((new_clauses, new_assigments, new_two_clause_occurrences))
        else:
            # splitting rule
            # pick the one in the most 2-clauses
            max_occurences = functools.reduce(max, two_clause_occurrences.values())
            most_common = [i for i in two_clause_occurrences if two_clause_occurrences[i] == max_occurences]
            # break ties by choosing the proposition with the largest difference betweeen positive and negative instances
            occurs = defaultdict(int)
            for clause in clauses:
                for prop in clause:
                    occurs[prop] += 1
            def parity_diff(prop_norm):
                negated_prop = (prop_norm[0], -prop_norm[-1])
                return abs(occurs[prop_norm] - occurs[negated_prop])
            best_prop = None
            for i in most_common:
                v = parity_diff(i)
                if best_prop is None or v > best_prop[1]:
                    best_prop = (i, v)
            # most_common.append(best_prop) # useless
            prop_norm = best_prop[0]
            # prop_norm = random.choice(most_common[:-1])
            # prop_norm = random.choice(most_common)
            # assign the value that makes the more common parity True
            # val_norm = False if occurs[prop_norm] >= occurs[(prop_norm[0], -prop_norm[-1])] else True
            val_norm = True
            temp = simplify(prop_norm, val_norm, clauses, two_clause_occurrences)
            if temp is not None:
                new_clauses, new_two_clause_occurrences = temp
                queue.append((new_clauses, {**assignments, prop_norm: val_norm}, new_two_clause_occurrences))
            temp = simplify(prop_norm, not val_norm, clauses, two_clause_occurrences)
            if temp is not None:
                new_clauses, new_two_clause_occurrences = temp
                queue.append((new_clauses, {**assignments, prop_norm: not val_norm}, new_two_clause_occurrences))
    return None

def solve_2_clause_with_parity_prop_always_false(clauses):
    # print(clauses[0])
    seed_value = 37
    random.seed(seed_value)
    # returns an updated version of clauses where
    # if a prop is true, clauses it's in are removed, if a prop is false, it is removed
    # from the clause and if a unit clauses becomes empty as a result, return None
    def simplify(prop, val, clauses, two_clause_occurrences):
        if prop[-1] < 0:
            raise Exception('prop should be positive')
        negated_prop = (prop[0], -prop[-1])
        true_prop = None
        false_prop = None
        if val:
            true_prop = prop
            false_prop = negated_prop
        else:
            true_prop = negated_prop
            false_prop = prop

        new_clauses = []
        new_two_clause_occurrences = {**two_clause_occurrences}
        for clause in clauses:
            if true_prop in clause:
                # remove clause, it shortcircuits to true
                if len(clause) == 2:
                    for prop in clause:
                        prop_norm = (prop[0], abs(prop[1]))
                        new_two_clause_occurrences[prop_norm] -= 1
                continue
            elif false_prop in clause:
                # remove prop from clause
                new_clause = [prop for prop in clause if prop != false_prop]
                # if clause is empty, shortcircuits to false
                if not new_clause:
                    return None
                if len(clause) == 2:
                    for prop in clause:
                        prop_norm = (prop[0], abs(prop[1]))
                        new_two_clause_occurrences[prop_norm] -= 1
                elif len(new_clause) == 2:
                    for prop in new_clause:
                        prop_norm = (prop[0], abs(prop[1]))
                        new_two_clause_occurrences[prop_norm] += 1
                new_clauses.append(new_clause)
            else:
                new_clauses.append(clause[::]) # don't need to copy if we never mutate

        return new_clauses, new_two_clause_occurrences

    # props_norm_init = set()
    # for clause in clauses:
    #     for prop in clause:
    #         props_norm_init.add((prop[0], abs(prop[1])))

    two_clause_occurrences_init = {}
    for clause in clauses:
        if len(clause) == 2:
            for prop in clause:
                prop_norm = (prop[0], abs(prop[1]))
                if prop_norm not in two_clause_occurrences_init:
                    two_clause_occurrences_init[prop_norm] = 0
                two_clause_occurrences_init[prop_norm] += 1
    # print(clauses)
    for clause in clauses:
        # print(clause)
        for prop in clause:
            # print(prop)
            prop_norm = (prop[0], abs(prop[1]))
            if prop_norm not in two_clause_occurrences_init:
                two_clause_occurrences_init[prop_norm] = 0
    # if len(two_clause_occurrences_init) != len(props_norm_init):
    #     raise Exception('two_clause_occurrences_init and props_norm_init should have same length')

    # use backtracking
    # TODO use bitmask for assigment and lookup instead of copying list
    # elems: (clauses, assignment, unit_props)
    queue = [(clauses, {}, two_clause_occurrences_init)]
    while queue:
        clauses, assignments, two_clause_occurrences = queue.pop()
        if not clauses:
            return assignments
        # unit-preference rule (TODO do all unit rules at once and have map that tracks size of clauses)
        unit_clauses = [clause for clause in clauses if len(clause) == 1]
        if unit_clauses:
            unit_clause = unit_clauses[0]
            unit_prop = unit_clause[0]
            unit_prop_norm = (unit_prop[0], abs(unit_prop[1]))
            unit_val_norm = True if unit_prop[1] == 1 else False
            temp = simplify(unit_prop_norm, unit_val_norm, clauses, two_clause_occurrences)
            if temp is None:
                continue
            new_clauses, new_two_clause_occurrences = temp
                
            if unit_prop in assignments:
                raise Exception('unit_prop already assigned')
            new_assigments = {**assignments, unit_prop_norm: unit_val_norm}

            queue.append((new_clauses, new_assigments, new_two_clause_occurrences))
        else:
            # splitting rule
            # pick the one in the most 2-clauses
            max_occurences = functools.reduce(max, two_clause_occurrences.values())
            most_common = [i for i in two_clause_occurrences if two_clause_occurrences[i] == max_occurences]
            # break ties by choosing the proposition with the largest difference betweeen positive and negative instances
            occurs = defaultdict(int)
            for clause in clauses:
                for prop in clause:
                    occurs[prop] += 1
            def parity_diff(prop_norm):
                negated_prop = (prop_norm[0], -prop_norm[-1])
                return abs(occurs[prop_norm] - occurs[negated_prop])
            best_prop = None
            for i in most_common:
                v = parity_diff(i)
                if best_prop is None or v > best_prop[1]:
                    best_prop = (i, v)
            # most_common.append(best_prop) # useless
            prop_norm = best_prop[0]
            # prop_norm = random.choice(most_common[:-1])
            # prop_norm = random.choice(most_common)
            # assign the value that makes the more common parity True
            # val_norm = False if occurs[prop_norm] >= occurs[(prop_norm[0], -prop_norm[-1])] else True
            val_norm = False
            temp = simplify(prop_norm, val_norm, clauses, two_clause_occurrences)
            if temp is not None:
                new_clauses, new_two_clause_occurrences = temp
                queue.append((new_clauses, {**assignments, prop_norm: val_norm}, new_two_clause_occurrences))
            temp = simplify(prop_norm, not val_norm, clauses, two_clause_occurrences)
            if temp is not None:
                new_clauses, new_two_clause_occurrences = temp
                queue.append((new_clauses, {**assignments, prop_norm: not val_norm}, new_two_clause_occurrences))
    return None

def solve_2_clause_with_parity_true(clauses):
    # print(clauses[0])
    seed_value = 37
    random.seed(seed_value)
    # returns an updated version of clauses where
    # if a prop is true, clauses it's in are removed, if a prop is false, it is removed
    # from the clause and if a unit clauses becomes empty as a result, return None
    def simplify(prop, val, clauses, two_clause_occurrences):
        if prop[-1] < 0:
            raise Exception('prop should be positive')
        negated_prop = (prop[0], -prop[-1])
        true_prop = None
        false_prop = None
        if val:
            true_prop = prop
            false_prop = negated_prop
        else:
            true_prop = negated_prop
            false_prop = prop

        new_clauses = []
        new_two_clause_occurrences = {**two_clause_occurrences}
        for clause in clauses:
            if true_prop in clause:
                # remove clause, it shortcircuits to true
                if len(clause) == 2:
                    for prop in clause:
                        prop_norm = (prop[0], abs(prop[1]))
                        new_two_clause_occurrences[prop_norm] -= 1
                continue
            elif false_prop in clause:
                # remove prop from clause
                new_clause = [prop for prop in clause if prop != false_prop]
                # if clause is empty, shortcircuits to false
                if not new_clause:
                    return None
                if len(clause) == 2:
                    for prop in clause:
                        prop_norm = (prop[0], abs(prop[1]))
                        new_two_clause_occurrences[prop_norm] -= 1
                elif len(new_clause) == 2:
                    for prop in new_clause:
                        prop_norm = (prop[0], abs(prop[1]))
                        new_two_clause_occurrences[prop_norm] += 1
                new_clauses.append(new_clause)
            else:
                new_clauses.append(clause[::]) # don't need to copy if we never mutate

        return new_clauses, new_two_clause_occurrences

    # props_norm_init = set()
    # for clause in clauses:
    #     for prop in clause:
    #         props_norm_init.add((prop[0], abs(prop[1])))

    two_clause_occurrences_init = {}
    for clause in clauses:
        if len(clause) == 2:
            for prop in clause:
                prop_norm = (prop[0], abs(prop[1]))
                if prop_norm not in two_clause_occurrences_init:
                    two_clause_occurrences_init[prop_norm] = 0
                two_clause_occurrences_init[prop_norm] += 1
    # print(clauses)
    for clause in clauses:
        # print(clause)
        for prop in clause:
            # print(prop)
            prop_norm = (prop[0], abs(prop[1]))
            if prop_norm not in two_clause_occurrences_init:
                two_clause_occurrences_init[prop_norm] = 0
    # if len(two_clause_occurrences_init) != len(props_norm_init):
    #     raise Exception('two_clause_occurrences_init and props_norm_init should have same length')

    # use backtracking
    # TODO use bitmask for assigment and lookup instead of copying list
    # elems: (clauses, assignment, unit_props)
    queue = [(clauses, {}, two_clause_occurrences_init)]
    while queue:
        clauses, assignments, two_clause_occurrences = queue.pop()
        if not clauses:
            return assignments
        # unit-preference rule (TODO do all unit rules at once and have map that tracks size of clauses)
        unit_clauses = [clause for clause in clauses if len(clause) == 1]
        if unit_clauses:
            unit_clause = unit_clauses[0]
            unit_prop = unit_clause[0]
            unit_prop_norm = (unit_prop[0], abs(unit_prop[1]))
            unit_val_norm = True if unit_prop[1] == 1 else False
            temp = simplify(unit_prop_norm, unit_val_norm, clauses, two_clause_occurrences)
            if temp is None:
                continue
            new_clauses, new_two_clause_occurrences = temp
                
            if unit_prop in assignments:
                raise Exception('unit_prop already assigned')
            new_assigments = {**assignments, unit_prop_norm: unit_val_norm}

            queue.append((new_clauses, new_assigments, new_two_clause_occurrences))
        else:
            # splitting rule
            # pick the one in the most 2-clauses
            max_occurences = functools.reduce(max, two_clause_occurrences.values())
            most_common = [i for i in two_clause_occurrences if two_clause_occurrences[i] == max_occurences]
            # break ties by choosing the proposition with the largest difference betweeen positive and negative instances
            occurs = defaultdict(int)
            for clause in clauses:
                for prop in clause:
                    occurs[prop] += 1
            def parity_diff(prop_norm):
                negated_prop = (prop_norm[0], -prop_norm[-1])
                return abs(occurs[prop_norm] - occurs[negated_prop])
            best_prop = None
            for i in most_common:
                v = parity_diff(i)
                if best_prop is None or v > best_prop[1]:
                    best_prop = (i, v)
            # most_common.append(best_prop) # useless
            # prop_norm = best_prop[0]
            # prop_norm = random.choice(most_common[:-1])
            prop_norm = random.choice(most_common)
            # assign the value that makes the more common parity True
            val_norm = True if occurs[prop_norm] >= occurs[(prop_norm[0], -prop_norm[-1])] else False
            # val_norm = False
            temp = simplify(prop_norm, val_norm, clauses, two_clause_occurrences)
            if temp is not None:
                new_clauses, new_two_clause_occurrences = temp
                queue.append((new_clauses, {**assignments, prop_norm: val_norm}, new_two_clause_occurrences))
            temp = simplify(prop_norm, not val_norm, clauses, two_clause_occurrences)
            if temp is not None:
                new_clauses, new_two_clause_occurrences = temp
                queue.append((new_clauses, {**assignments, prop_norm: not val_norm}, new_two_clause_occurrences))
    return None

def solve_2_clause_with_parity_false(clauses):
    # print(clauses[0])
    seed_value = 37
    random.seed(seed_value)
    # returns an updated version of clauses where
    # if a prop is true, clauses it's in are removed, if a prop is false, it is removed
    # from the clause and if a unit clauses becomes empty as a result, return None
    def simplify(prop, val, clauses, two_clause_occurrences):
        if prop[-1] < 0:
            raise Exception('prop should be positive')
        negated_prop = (prop[0], -prop[-1])
        true_prop = None
        false_prop = None
        if val:
            true_prop = prop
            false_prop = negated_prop
        else:
            true_prop = negated_prop
            false_prop = prop

        new_clauses = []
        new_two_clause_occurrences = {**two_clause_occurrences}
        for clause in clauses:
            if true_prop in clause:
                # remove clause, it shortcircuits to true
                if len(clause) == 2:
                    for prop in clause:
                        prop_norm = (prop[0], abs(prop[1]))
                        new_two_clause_occurrences[prop_norm] -= 1
                continue
            elif false_prop in clause:
                # remove prop from clause
                new_clause = [prop for prop in clause if prop != false_prop]
                # if clause is empty, shortcircuits to false
                if not new_clause:
                    return None
                if len(clause) == 2:
                    for prop in clause:
                        prop_norm = (prop[0], abs(prop[1]))
                        new_two_clause_occurrences[prop_norm] -= 1
                elif len(new_clause) == 2:
                    for prop in new_clause:
                        prop_norm = (prop[0], abs(prop[1]))
                        new_two_clause_occurrences[prop_norm] += 1
                new_clauses.append(new_clause)
            else:
                new_clauses.append(clause[::]) # don't need to copy if we never mutate

        return new_clauses, new_two_clause_occurrences

    # props_norm_init = set()
    # for clause in clauses:
    #     for prop in clause:
    #         props_norm_init.add((prop[0], abs(prop[1])))

    two_clause_occurrences_init = {}
    for clause in clauses:
        if len(clause) == 2:
            for prop in clause:
                prop_norm = (prop[0], abs(prop[1]))
                if prop_norm not in two_clause_occurrences_init:
                    two_clause_occurrences_init[prop_norm] = 0
                two_clause_occurrences_init[prop_norm] += 1
    # print(clauses)
    for clause in clauses:
        # print(clause)
        for prop in clause:
            # print(prop)
            prop_norm = (prop[0], abs(prop[1]))
            if prop_norm not in two_clause_occurrences_init:
                two_clause_occurrences_init[prop_norm] = 0
    # if len(two_clause_occurrences_init) != len(props_norm_init):
    #     raise Exception('two_clause_occurrences_init and props_norm_init should have same length')

    # use backtracking
    # TODO use bitmask for assigment and lookup instead of copying list
    # elems: (clauses, assignment, unit_props)
    queue = [(clauses, {}, two_clause_occurrences_init)]
    while queue:
        clauses, assignments, two_clause_occurrences = queue.pop()
        if not clauses:
            return assignments
        # unit-preference rule (TODO do all unit rules at once and have map that tracks size of clauses)
        unit_clauses = [clause for clause in clauses if len(clause) == 1]
        if unit_clauses:
            unit_clause = unit_clauses[0]
            unit_prop = unit_clause[0]
            unit_prop_norm = (unit_prop[0], abs(unit_prop[1]))
            unit_val_norm = True if unit_prop[1] == 1 else False
            temp = simplify(unit_prop_norm, unit_val_norm, clauses, two_clause_occurrences)
            if temp is None:
                continue
            new_clauses, new_two_clause_occurrences = temp
                
            if unit_prop in assignments:
                raise Exception('unit_prop already assigned')
            new_assigments = {**assignments, unit_prop_norm: unit_val_norm}

            queue.append((new_clauses, new_assigments, new_two_clause_occurrences))
        else:
            # splitting rule
            # pick the one in the most 2-clauses
            max_occurences = functools.reduce(max, two_clause_occurrences.values())
            most_common = [i for i in two_clause_occurrences if two_clause_occurrences[i] == max_occurences]
            # break ties by choosing the proposition with the largest difference betweeen positive and negative instances
            occurs = defaultdict(int)
            for clause in clauses:
                for prop in clause:
                    occurs[prop] += 1
            def parity_diff(prop_norm):
                negated_prop = (prop_norm[0], -prop_norm[-1])
                return abs(occurs[prop_norm] - occurs[negated_prop])
            best_prop = None
            for i in most_common:
                v = parity_diff(i)
                if best_prop is None or v > best_prop[1]:
                    best_prop = (i, v)
            # most_common.append(best_prop) # useless
            # prop_norm = best_prop[0]
            # prop_norm = random.choice(most_common[:-1])
            prop_norm = random.choice(most_common)
            # assign the value that makes the more common parity True
            val_norm = False if occurs[prop_norm] >= occurs[(prop_norm[0], -prop_norm[-1])] else True
            # val_norm = False
            temp = simplify(prop_norm, val_norm, clauses, two_clause_occurrences)
            if temp is not None:
                new_clauses, new_two_clause_occurrences = temp
                queue.append((new_clauses, {**assignments, prop_norm: val_norm}, new_two_clause_occurrences))
            temp = simplify(prop_norm, not val_norm, clauses, two_clause_occurrences)
            if temp is not None:
                new_clauses, new_two_clause_occurrences = temp
                queue.append((new_clauses, {**assignments, prop_norm: not val_norm}, new_two_clause_occurrences))
    return None


def generate_riddle_cnf():
    # PROPS: NOT H_1,blu := ((1, 'blue'), -1)

    HOUSES = range(1, 5+1)
    COLORS = ['blue', 'green', 'red', 'white', 'yellow']
    NATIONALITIES = ['danish', 'british', 'german', 'norwegian', 'swedish']
    BEVERAGES = ['beer', 'coffee', 'milk', 'tea', 'water']
    CIGARS = ['blend', 'bluemaster', 'dunhill', 'pallmall', 'prince']
    PETS = ['bird', 'cat', 'dog', 'fish', 'horse']

    def all_assigned(property):
        clauses = []
        for value in property:
            clause = []
            for house in HOUSES:
                clause.append(((house, value), 1))
            clauses.append(clause)
        return clauses

    def at_most_one(property):
        clauses = []
        # at least 1 (could probably remove since we're requiring all properties to be assigned)
        # at most 1
        for house in HOUSES:
            for i in range(len(property)):
                for j in range(i+1, len(property)):
                    clauses.append([((house, property[i]), -1), ((house, property[j]), -1)])

        return clauses

    def disjunction_to_conjunction(clauses):
        # (a * b) + (c * d) + ... becomes (a + c + ...) * (a + d + ...) * ...
        if not clauses:
            return []
        rest = disjunction_to_conjunction(clauses[1:])
        new_clauses = []
        for literal in clauses[0]:
            if rest:
                new_clauses += [[literal] + i for i in rest]
            else:
                new_clauses += [[literal]]
        return new_clauses

    def one_of_has_both(property1, property2):
        clauses = []
        for house in HOUSES:
            clauses.append([((house, property1), 1), ((house, property2), 1)])
        return disjunction_to_conjunction(clauses)

    # output needs to be converted to conjunction
    def one_left_of_other_helper(property1, property2):
        disjunctive_clauses = []
        for i in range(1, len(HOUSES)-1 + 1):
            disjunctive_clauses.append([((i, property1), 1), ((i+1, property2), 1)])
        return disjunctive_clauses

    def one_left_of_other(property1, property2):
        return disjunction_to_conjunction(one_left_of_other_helper(property1, property2))

    def one_next_to_other(property1, property2):
        disjunctive_clauses = []
        disjunctive_clauses += one_left_of_other_helper(property1, property2)
        disjunctive_clauses += one_left_of_other_helper(property2, property1)
        return disjunction_to_conjunction(disjunctive_clauses)

    clauses = []

    # Constraints
    """
    every property is assigned to some house
    """
    clauses += all_assigned(COLORS)
    clauses += all_assigned(NATIONALITIES)
    clauses += all_assigned(BEVERAGES)
    clauses += all_assigned(CIGARS)
    clauses += all_assigned(PETS)
    """
    each house 1 color, each house 1 nationality, each house 1 beverage, 
    each house 1 cigar, each house 1 pet
    """
    clauses += at_most_one(COLORS)
    clauses += at_most_one(NATIONALITIES)
    clauses += at_most_one(BEVERAGES)
    clauses += at_most_one(CIGARS)
    clauses += at_most_one(PETS)

    # Hints
    clauses += one_of_has_both('british', 'red')
    clauses += one_of_has_both('swedish', 'dog')
    clauses += one_of_has_both('danish', 'tea')
    clauses += one_left_of_other('green', 'white')
    clauses += one_of_has_both('green', 'coffee')
    clauses += one_of_has_both('pallmall', 'bird')
    clauses += one_of_has_both('yellow', 'dunhill')
    clauses += [[((3, 'milk'), 1)]]
    clauses += [[((1, 'norwegian'), 1)]]
    clauses += one_next_to_other('blend', 'cat')
    clauses += one_next_to_other('horse', 'dunhill')
    clauses += one_of_has_both('bluemaster', 'beer')
    clauses += one_of_has_both('german', 'prince')
    clauses += one_next_to_other('norwegian', 'blue')
    clauses += one_next_to_other('blend', 'water')

    return clauses

def get_info_about_fish_owner(assignments):
    house = None
    for ((h, prop), _), val in assignments.items():
        if prop == 'fish' and val:
            house = h
    
    info = [house]
    for ((h, prop), _), val in assignments.items():
        if h == house and val:
            info.append(prop)
    return info

def get_only_true_assignments(assignments):
    return sorted([(h, prop) for (h, prop), val in assignments.items() if val], key=lambda x: x[0])

def gen_random_formula(L, N):
    # with 3 literals per clause
    clauses = []
    for _ in range(L):
        clause = []
        while len(clause) < 3:
            num = random.randint(1, N)
            if num not in clause:
                clause.append(num)
        real_clause = []
        for literal in clause:
            if random.randint(0, 1) == 0:
                real_clause.append((literal, 1))
            else:
                real_clause.append((literal, -1))
        clauses.append(real_clause)
    return clauses

def get_L_for_ratio(N, ratio):
    return int(N * ratio)

def print_stats(f, list_of_clauses):
    res = []
    for clauses in list_of_clauses:
        num_dpll_calls = [0]
        assignments_ptr = [None]
        # start = time.time()
        execution_time = timeit.timeit(stmt=lambda: f(clauses, num_dpll_calls, assignments_ptr), number=1)
        # assignments = f(clauses, num_dpll_calls)
        assignments = assignments_ptr[0]
        # end = time.time()
        num_dpll_calls = num_dpll_calls[0]
        if num_dpll_calls == -1:
            print('timed out')
            continue # skip this one it took too long
        res_i = {}
        res_i['num_dpll_calls'] = num_dpll_calls
        res_i['sat'] = assignments is not None
        # res_i['time'] = end - start
        res_i['time'] = execution_time
        res.append(res_i)
    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random', action='store_true')
    parser.add_argument('--two-clause', action='store_true')
    parser.add_argument('--custom-heuristic', action='store_true')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--read', action='store_true')
    parser.add_argument('--make-graphs', action='store_true')
    args = parser.parse_args()

    assignments = None
    if args.random:
        clauses = generate_riddle_cnf()
        assignments = solve_random(clauses)
    elif args.make_graphs:
        with open('experiment_out_full.json', 'r') as f:
            data = json.load(f)
            for N in [100, 150]:
                plt.close()
                # functions = list(data[str(N)]["3"]["results"].keys())
                ratios = [ratio for ratio in data[str(N)].keys()]
                median_times = []
                for ratio in ratios:
                    # print([ratio for ratio in data['100']])
                    issat = [trial["sat"] for trial in data[str(N)][ratio]["results"]["solve_2_clause"]]
                    # print(issat)
                    if len(issat) < 10:
                        # print(len(times))
                        # raise Exception("No times found")
                        median_times.append(np.nan)
                    else:
                        sats = 0
                        for i in issat:
                            if i == True:
                                sats += 1
                            elif i == False:
                                pass
                            else:
                                raise Exception()
                        median_times.append(sats / float(len(issat)))
                
                times = median_times
                # print(ratios)
                # print(times, len(times))
                to_graph_x = []
                to_graph_y = []
                for i in range(len(ratios)):
                    if not np.isnan(times[i]):
                        to_graph_x.append(ratios[i])
                        to_graph_y.append(times[i])
                # print(to_graph_x, to_graph_y)
                plt.plot(to_graph_x, to_graph_y, linestyle='-', marker='o')

                property = "percent satisfiable"
                plt.xlabel("Ratio")
                plt.ylabel(f"Median {property}")
                plt.title(f"Median {property} vs. Ratio for N={N}")
                plt.grid(True)
                # plt.legend()

                plt.savefig(f'plot_N={N}_{property}.png')
                # plt.show()
            for N in [100, 150]:
                for property in ["time", "num_dpll_calls"]:
                    plt.close()
                    functions = list(data[str(N)]["3"]["results"].keys())
                    ratios = [ratio for ratio in data[str(N)].keys()]

                    median_times = {func: [] for func in functions[::-1]}

                    for ratio in ratios:
                        for func in functions:
                            # print([ratio for ratio in data['100']])
                            times = [trial[property] for trial in data[str(N)][ratio]["results"][func]]
                            if len(times) < 3 or (ratio == '4.4' and func == 'solve_random'):
                                # print(len(times))
                                # raise Exception("No times found")
                                median_times[func].append(np.nan)
                            else:
                                median_time = sorted(times)[len(times) // 2]
                                median_times[func].append(median_time)
                    
                    for func, times in median_times.items():
                        # print(ratios)
                        # print(times, len(times))
                        to_graph_x = []
                        to_graph_y = []
                        for i in range(len(ratios)):
                            if not np.isnan(times[i]):
                                to_graph_x.append(ratios[i])
                                to_graph_y.append(times[i])
                        # print(to_graph_x, to_graph_y)
                        plt.plot(to_graph_x, to_graph_y, label=func, linestyle='-', marker='o')

                    plt.xlabel("Ratio")
                    plt.ylabel(f"Median {property}")
                    plt.title(f"Median {property} vs. Ratio for N={N}")
                    plt.grid(True)
                    plt.legend()

                    plt.savefig(f'plot_N={N}_{property}.png')
                    # plt.show()
    elif args.read:
        with open('experiment_out_full.json', 'r') as f:
            experiment_out = json.load(f)
            # print(experiment_out)
            # get the median time and num_dpll_calls for each of the 3 functions
            for N in [100, 150]:
                for ratio in [3, 3.2, 3.4, 3.6, 3.8, 4, 4.2, 4.4, 4.6, 4.8, 5, 5.2, 5.4, 5.6, 5.8, 6]:
                    L = get_L_for_ratio(N, ratio)
                    num_trials = 100
                    fs = []
                    if N  < 150:
                        fs.append('solve_random')
                    for f in fs + ['solve_2_clause', 'solve_2_clause_with_parity_prop_false']:
                        times = []
                        calls = []
                        for i in experiment_out[str(N)][str(ratio)]['results'][f]:
                            times.append(i['time'])
                            calls.append(i['num_dpll_calls'])
                        if not times:
                            print(f, N, L, 'no results')
                            continue
                        times.sort()
                        calls.sort()
                        med = times[len(times)//2]
                        med_calls = calls[len(calls)//2]
                        print(f, N, L, med, med_calls)
    elif args.two_clause:
        clauses = generate_riddle_cnf()
        assignments = solve_2_clause(clauses)
    elif args.custom_heuristic:
        clauses = generate_riddle_cnf()
        assignments = solve_2_clause_with_parity_prop_true(clauses)
    elif args.all:
        fileout = 'experiment_out_full.json'
        if fileout in os.listdir():
            raise Exception('experiment_out.json already exists')

        experiment_out = {}
        for N in [100, 150]:
            functions_to_test = []
            if N < 150:
                functions_to_test.append(solve_random)
            functions_to_test += [
                                # solve_random,
                                solve_2_clause, # use this one
                                #  solve_2_clause_with_parity_prop_true, # not good weirdly
                                solve_2_clause_with_parity_prop_false, # use this one
                                #  solve_2_clause_least_with_parity_prop_true, # never finishes running??
                                #  solve_2_clause_least_with_parity_prop_false, # always slower than solve_2_clause_with_parity_prop_false
                                #  solve_2_clause_with_parity_propleast_true, # same always worse
                                #  solve_2_clause_with_parity_propleast_false,  # doesnt seem to finish running
                                #  solve_2_clause_with_parity_prop_always_true, # bad
                                #  solve_2_clause_with_parity_prop_always_false, # bad
                                #  solve_2_clause_with_parity_true, # bad
                                #  solve_2_clause_with_parity_false # bad
                                ]
            print(f"Testing {', '.join([f.__name__ for f in functions_to_test])}")
            experiment_out[N] = {}
            for ratio in [3, 3.2, 3.4, 3.6, 3.8, 4, 4.2, 4.4, 4.6, 4.8, 5, 5.2, 5.4, 5.6, 5.8, 6]:
                L = get_L_for_ratio(N, ratio)
                num_trials = 100
                experiment_out[N][ratio] = {}
                experiment_out[N][ratio]['L'] = L
                experiment_out[N][ratio]['num_trials'] = num_trials

                print(f'N: {N}, L: {L}, num_trials: {num_trials}')
                list_of_clauses = []
                for _ in range(num_trials):
                    list_of_clauses.append(gen_random_formula(L, N))

                func_results = {}
                for f in functions_to_test:
                    res = print_stats(f, list_of_clauses)
                    func_results[f.__name__] = res
                    # print(f"Execution time for {f.__name__}: {end - start} seconds")
                experiment_out[N][ratio]['results'] = func_results
        with open(fileout, 'w') as f:
            json.dump(experiment_out, f)
