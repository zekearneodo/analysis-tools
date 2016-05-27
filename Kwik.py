# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 12:05:54 2014

@author: Josh Siegle

Loads .kwd files

"""

import h5py
import numpy as np


def load(filename, dataset=0):
    
    # loads raw data into an HDF5 dataset
    # NOT converted to microvolts --- need to multiply by 0.195 scaling factor
    # timestamps may need to be shifted by get_experiment_start_time() to align with events
        
    f = h5py.File(filename, 'r')
    
    data = {}
    
    data['info'] = f['recordings'][str(dataset)].attrs
    data['data'] = f['recordings'][str(dataset)]['data'] # not converted to microvolts!!!! need to multiply by 0.195
    data['timestamps'] = ((np.arange(0,data['data'].shape[0])
                         + data['info']['start_time'])       
                         / data['info']['sample_rate'])
                         
    return data


def convert(filename, filetype='dat', dataset=0):

    f = h5py.File(filename, 'r')
    fnameout = filename[:-3] + filetype

    if filetype == 'dat':    
        data = f['recordings'][str(dataset)]['data'][:,:]
        data.tofile(fnameout)
    
    
def write(filename, dataset=0, bit_depth=1.0, sample_rate=25000.0):
    
    f = h5py.File(filename, 'w-')
    f.attrs['kwik_version'] = 2
    
    grp = f.create_group("/recordings/0")
    
    dset = grp.create_dataset("data", dataset.shape, dtype='i16')
    dset[:,:] = dataset
    
    grp.attrs['start_time'] = 0.0
    grp.attrs['start_sample'] = 0
    grp.attrs['sample_rate'] = sample_rate
    grp.attrs['bit_depth'] = bit_depth
    
    f.close()
    

def get_sample_rate(f):
    return f['recordings']['0'].attrs['sample_rate'] 


def get_edge_times(f, TTLchan, rising=True, time_in_samples=False):
    
    events_for_chan = np.where(np.squeeze(f['event_types']['TTL']['events']['user_data']['event_channels']) == TTLchan)
    
    edges = np.where(np.squeeze(f['event_types']['TTL']['events']['user_data']['eventID']) == 1*rising) 
    
    edges_for_chan = np.intersect1d(events_for_chan, edges)
    
    edge_samples = np.squeeze(f['event_types']['TTL']['events']['time_samples'][:])[edges_for_chan]
    edges = edge_samples if time_in_samples else edge_samples / get_sample_rate(f)

    return edges


def get_rising_edge_times(filename, TTLchan, time_in_samples=False):
    
    f = h5py.File(filename, 'r')
    
    return get_edge_times(f, TTLchan, True, time_in_samples)
    

def get_falling_edge_times(filename, TTLchan, time_in_samples=False):
    
    f = h5py.File(filename, 'r')
    
    return get_edge_times(f, TTLchan, False, time_in_samples)


def get_experiment_start_time(filename, time_in_samples=False):
    
    f = h5py.File(filename, 'r')

    start_samples = f['event_types']['Messages']['events']['time_samples'][1]
    start = start_samples if time_in_samples else start_samples / get_sample_rate(f)
    return start

    
            
                         
                        