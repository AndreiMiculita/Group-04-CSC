# Copyright (C) 2014, 2015 University of Vienna
# All rights reserved.
# BSD license.
# Author: Ali Baharev <ali.baharev@gmail.com>
from __future__ import print_function
from contextlib import contextmanager
from copy import deepcopy
import errno
import imp
import itertools
import gzip
import sys, os
from time import strftime
from six import exec_ 
from namedlist import namedlist
from py3compat import izip, izip_longest, StringIO, cPickle_load, \
                      cPickle_dump, cPickle_HIGHEST_PROTOCOL
# From SDOPT

class StringBuffer:
    
    def __init__(self):
        self.buff = StringIO()

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.buff.close()
    
    def __call__(self, *args):
        self.buff.writelines(args)
        self.buff.write('\n')
        
    def getvalue(self):
        return self.buff.getvalue()
    
#-------------------------------------------------------------------------------

def print_timestamp():
    print(strftime("%H:%M:%S"))

#-------------------------------------------------------------------------------

DATADIR = 'data'+os.sep

def serialize(obj, filename):
    with gzip.open(filename, 'wb') as f:
        cPickle_dump(obj, f, cPickle_HIGHEST_PROTOCOL)

def deserialize(filename):
    with gzip.open(filename, 'rb') as f:
        return cPickle_load(f)

def marshal_dump(obj, filename):
    from marshal import dump
    with open(filename, 'wb') as f:
        dump(obj, f)

def marshal_load(filename):
    from marshal import load
    with open(filename, 'rb') as f:
        return load(f)

#-------------------------------------------------------------------------------

@contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

#-------------------------------------------------------------------------------

def remove_if_exists(filename):
    try:
        os.remove(filename)
    except OSError as e:
        if e.errno != errno.ENOENT:
            raise

def create_dir(path):
    try:
        os.mkdir(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def create_dir_w_parents(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

#-------------------------------------------------------------------------------

# http://code.activestate.com/recipes/82234-importing-a-dynamically-generated-module/
def import_code(code):
    module = imp.new_module('someFakeName')
    try:
        exec_(code, module.__dict__)
    except:
        print(code)
        raise
    return module

#-------------------------------------------------------------------------------

def pairwise(iterable):
    '''A generator object is returned.
    []  pairwise: []
    [1] pairwise: []
    [1,2,3] pairwise: [(1, 2), (2, 3)].'''
    a, b = itertools.tee(iterable)
    next(b, None)
    return izip(a, b)

def duplicates(iterable):
    seen = set()
    seen_add = seen.add
    return sorted(set(e for e in iterable if e in seen or seen_add(e)))

def contains_none(iterable):
    return any(e is None for e in iterable)

def argsort(lst):
    # http://stackoverflow.com/questions/3382352
    return sorted(range(len(lst)), key=lst.__getitem__)

#-------------------------------------------------------------------------------
# Simple cycle utilities

def edges_of_cycle(seq):
    a, b = itertools.tee(seq)
    next(b, None)
    return izip_longest(a, b, fillvalue=seq[0])

def to_cycle(simple_path_nodes):
    seq = rotate_min_to_first_pos(simple_path_nodes)
    a, b = itertools.tee(seq)
    next(b, None)
    return tuple( izip_longest(a, b, fillvalue=seq[0]) )

def rotate_min_to_first_pos(lst):
    assert lst, 'expecting a non-empty list'
    n = min(enumerate(lst), key=lambda t: t[1])[0] # index of min element
    return lst[n:] + lst[:n] # do the actual rotation 

def get_all_cycles(g, cutoff=500):
    from networkx import simple_cycles
    # Try to enumerate cutoff+1 simple cycles
    cycles = list(itertools.islice(simple_cycles(g), cutoff+1))
    # If that succeeds, we give up
    if len(cycles) > cutoff:
        print('More than', cutoff, 'simple cycles, giving up...')
        return False, None
    # Otherwise we have enumerated ALL cycles, we return the edges of each
    edges_per_cycle = [to_cycle(c) for c in cycles ]
    return True, edges_per_cycle

#-------------------------------------------------------------------------------
# DiGraph utilities

def info(g, list_of_simple_cycles=None):
    from networkx import simple_cycles, strongly_connected_components
    # Compare with other info and SDOPT
    print('Nodes:', g.number_of_nodes())
    print('Edges:', g.number_of_edges())
    if list_of_simple_cycles is None:
        list_of_simple_cycles = list(simple_cycles(g))
    print('Loops:', len(list_of_simple_cycles))
    sccs = list(strongly_connected_components(g))
    nontriv_sccs = [ sc for sc in sccs if len(sc) > 1 ]
    print('SCCs: ', len(nontriv_sccs))
    small_sccs = [ sc for sc in nontriv_sccs if len(sc) < 10 ]
    if small_sccs:
        print('Small nontrivial SCCs:')
        for scc in small_sccs:
            print(sorted(n for n in scc))
    print()

def info_short(g, log=print):
    from networkx import strongly_connected_components
    log('Nodes:', g.number_of_nodes())
    log('Edges:', g.number_of_edges())
    sccs = list(strongly_connected_components(g))
    nontriv_sccs = [ sc for sc in sccs if len(sc) > 1 ]
    log('SCCs: ', len(nontriv_sccs))
    log()

def double_check(g_orig, cost, elims, is_labeled=False, log=print):
    from networkx import is_directed_acyclic_graph
    # Cost check
    total_cost = _cost_of_elim(g_orig, elims) if is_labeled else len(elims)
    assert (total_cost == cost), (total_cost, cost)
    # Acyclicity check
    g = deepcopy(g_orig)
    for u,v in elims:
        g.remove_edge(u,v)
    assert is_directed_acyclic_graph(g)
    msg = 'Double-checked: cost = {}, and the graph is acyclic, OK!'
    log(msg.format(cost))

def _cost_of_elim(g_orig, elims):
    return sum( g_orig[u][v]['weight'] for u,v in elims  )

def split_to_nontrivial_sccs(g):
    from networkx import strongly_connected_components
    # Removes single node SCCs, and returns true SCCs
    sccs = list(strongly_connected_components(g))
    # Check deepcopy also in SDOPT
    return [ g.subgraph(sc).copy() for sc in sccs if len(sc) > 1 ]

#-------------------------------------------------------------------------------
# Gurobi utilities

class TimeLimit(BaseException):
    pass

def has_gurobi():
    try:
        import gurobipy as grb
        _ = grb.__name__ # just to make the annoying warning go away
    except ImportError:
        return False
    return True


# Counters: ILPs solved, nodes explored, simplex iterations performed.
# These will be mutated in-place if passed as the keyword argument stats.
Stats = namedlist('Stats', 'name  params  is_optimal cost  ILP  node  iter  time')

def solve_ilp(m, stats=None):
    'Raises TimeLimit if reached. Returns True on success, False otherwise.'
    from gurobipy import GRB
    m.optimize()
    if stats:
        stats.ILP += 1
        stats.node += int(m.nodeCount)
        stats.iter += int(m.iterCount)
    status = m.status
    if status == GRB.status.TIME_LIMIT:
        obj = int(round(m.getObjective().getValue()))
        print('Objective when time limit reached:', obj)
        raise TimeLimit()
    if status == GRB.status.INF_OR_UNBD or status == GRB.status.INFEASIBLE \
      or status == GRB.status.UNBOUNDED:
        print('Infeasible or unbounded model')
        return False
    if status != GRB.status.OPTIMAL:
        print('Optimization was stopped with status %d' % status)
        return False
    return True

#-------------------------------------------------------------------------------

def has_matplotlib():
    try:
        import matplotlib
        _ = matplotlib.__name__ # just to make the annoying warning go away
    except ImportError:
        return False
    return True
