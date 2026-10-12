#!/usr/bin/env python3
"""Replot either experiment from its saved JSON outputs only."""
import argparse
from reviewer_controls.plotting import plot_reviewer,plot_rq2

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('experiment',choices=['reviewer','rq2'])
    p.add_argument('output_dir')
    a=p.parse_args()
    (plot_reviewer if a.experiment=='reviewer' else plot_rq2)(a.output_dir)
