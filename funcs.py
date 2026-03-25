from ase.io import read, iread
import ase.units
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from aseMolec import pltProps as pp
from aseMolec import anaAtoms as aa
import pandas as pd
import numpy as np
import tol_colors as tc


def get_mol_histograms(xyz_dict):
    histogram_dict = {
        key:np.histogram(np.array([a.info['Nmols'] for a in xyz]), [1,2,3,4,5,6,7])[0]
        for key, xyz in xyz_dict.items()
    }
    return histogram_dict


def collect_comp(db):
    from collections import Counter
    buf = {}
    for at in db:
        if at.info['Nmols'] in buf:
            try:
                buf[at.info['Nmols']] += [at.info['Comp']]
            except:
                buf[at.info['Nmols']] += [at.info['config_type']]
        else:
            try:
                buf[at.info['Nmols']] = [at.info['Comp']]
            except:
                buf[at.info['Nmols']] = [at.info['config_type']]

    comp = {}
    for b in buf:
        comp[b] = dict(Counter(buf[b]))
    return comp


def flatten_comp(comp_dict):
    flattened_comp_dict = {}
    for csize, dist in comp_dict.items():
        tmp = {
            'EMC':0,
            'EC':0,
            'EC and EMC + other':0,
            'EC or EMC + other':0,
            'Other':0,
        }
        for comp, count in dist.items():
            part = comp.split(':')
            part = [p.split('(')[0] for p in part]
            part = [''.join([i for i in p if not i.isdigit()]) for p in part]
            if ('EC' not in part) and ('EMC' not in part):
                tmp['Other'] += count            
            elif ('EC' in part) and ('EMC' in part):
                tmp['EC and EMC + other'] += count
            elif ('EC' in part) and ('EMC' not in part):
                if len(set(part)) == 1:
                    tmp['EC'] += count
                else:
                    tmp['EC or EMC + other'] += count
            elif ('EC' not in part) and ('EMC' in part):
                if len(set(part)) == 1:
                    tmp['EMC'] += count
                else:
                    tmp['EC or EMC + other'] += count
            else:
                tmp['EC or EMC + other'] += count

        flattened_comp_dict[csize] = tmp
    df = pd.DataFrame(flattened_comp_dict).T
    return df.sort_index(inplace=False)


def hist_dict_to_df(hist_dict):
    hist_df = pd.DataFrame(hist_dict)
    hist_df.index.name = 'Cluster size, Nmols'
    hist_df.index += 1
    return hist_df


def get_expectation_values(hist_df, arr):
    column_totals = hist_df.aggregate('sum', axis=0)
    prob_df = hist_df/column_totals
    expectation = (prob_df*arr[:,None]).aggregate('sum', axis=0).round(decimals=2)
    return expectation


def get_dynamic_data(data, handle=''):
    """Collect time (fs) and MSD (Å/fs) vectors.
    
    This function can be modified to treat any data based on the handle-variable.
    """
    if handle in ('B97D3', 'PBED3', 'PBED2', 'Clusters-Large', 'Clusters-Medium', 'Clusters-Small'):
        time = data['Time']['data']
        msd = data['MSD']['data']
        density = data['Density']['data']
    return time, msd, density


def get_trajectory_results(thermo, threshold=5e-2, window=2000, handle=None):
    results = {
        'Data set':[],
        'Labels':[],
        'Seed ID':[],
        'Sample ID':[],
        'Composition':[],
        'Temperature /K':[],
        'Density /g*cm-3':[],
        'Density std. /g*cm-3':[],
        'Diff. coeff. /1e6 cm2*s-1':[],
        'Diff. fit slope error /1e6 cm2*s-1':[],
        'Diff. fit R2':[],
        'Start time /ns':[],
        'Start time R2':[],
    }
    print(f'Using slope error thresold of: {threshold}, and moving window of: {window} frames')
    for i, tag in enumerate(thermo):
        print()
        print(tag)
        data_set, labels, seed, dset_sample = tag.split('/')
        seed_id = int(seed[-1])
        sample_id = int(dset_sample[-1])
        for j, (traj_name, traj_data) in enumerate(thermo[tag].items()):
            print(traj_name)
            ensemble, comp, temp = traj_name.split('_')
            temp = int(temp[:-1])
            
            # Collect data
            ## If loading data from sources with a different structure from this, this must be controlled using
            ## the handle-variable. In such case, get_dynamic_data must also be modified to accomodate this.
            if handle == 'Labels':
                time, msd, density = get_dynamic_data(traj_data, handle=labels)
            elif handle == 'Data set':
                time, msd, density = get_dynamic_data(traj_data, handle=data_set)

            # Obtain optimal starting time from unit log-log slope by fitting slopes in a moving window
            start_index, start_time_R2 = get_start_index(time, msd, threshold=threshold, window=window, start_id=0)

            if start_index != None:
                # Obtain diffusion coefficient with fitting errors
                diffusion_coeff, slope_error, diffusion_coeff_R2 = diffusion_coefficient(time[start_index:],msd[start_index:])
                print(f'start index: {start_index}, time steps: {len(time)}, start time R2: {start_time_R2:.5f}, Diff. coeff. (m^2/s): {diffusion_coeff:.5e}, slope error: {slope_error:.5e}, Diff. coeff. R2: {diffusion_coeff_R2:.5f}')
                
                results['Data set'].append(data_set)
                results['Labels'].append(label_map[labels])
                results['Seed ID'].append(seed_id)
                results['Sample ID'].append(sample_id)
                results['Composition'].append(comp_map[comp])
                results['Temperature /K'].append(temp)
                results['Density /g*cm-3'].append(np.mean(density))
                results['Density std. /g*cm-3'].append(np.std(density))
                results['Diff. coeff. /1e6 cm2*s-1'].append(diffusion_coeff*1e10)
                results['Diff. fit slope error /1e6 cm2*s-1'].append(slope_error*1e10)
                results['Diff. fit R2'].append(diffusion_coeff_R2)
                results['Start time /ns'].append(time[start_index]*1e-6)
                results['Start time R2'].append(start_time_R2)
            else:
                results['Data set'].append(np.nan)
                results['Labels'].append(np.nan)
                results['Seed ID'].append(np.nan)
                results['Sample ID'].append(np.nan)
                results['Composition'].append(np.nan)
                results['Temperature /K'].append(np.nan)
                results['Density /g*cm-3'].append(np.nan)
                results['Density std. /g*cm-3'].append(np.nan)
                results['Diff. coeff. /1e6 cm2*s-1'].append(np.nan)
                results['Diff. fit slope error /1e6 cm2*s-1'].append(np.nan)
                results['Diff. fit R2'].append(np.nan)
                results['Start time /ns'].append(np.nan)
                results['Start time R2'].append(np.nan)
    df = pd.DataFrame(results)
    return df


def get_xyz(tag, handle='xyz'):
    import glob
    flist = [name for name in glob.glob(tag)]
    for f in flist:
        if handle in f:
            print(f)
            return read(f, ':')


def RMSE(x,y):
    rmse = np.sqrt(np.mean((x-y)**2))
    rrmse = rmse/np.sqrt(np.mean((x-np.mean(x))**2))
    return rmse, rrmse


def get_all_thermo(tag, handle='.thermo'):
    """This needs to be exhanged for Ioans version in order to be consistent with
    zenodo.
    """
    import glob
    import os
    from aseMolec import pltProps as pp
    
    thermo = {}
    flist = [name for name in glob.glob(tag) if handle in name]
    for f in flist:
        print(f)
        aux = os.path.basename(f)
        key = os.path.splitext(aux)[0]
        if 'xyz' in handle:
            thermo.update({key: read(f, ':')})
        else:
            thermo.update({key: pp.loadtxttag(f)})
    return thermo


def diffusion_coefficient(time, msd):
    """Determines the diffusion coefficient in three dimensions
    from the slope of the MSD-curve.
    """
    from scipy.stats import linregress

    res = linregress(time,msd)
    diffusion_coeff = res.slope/6 #final unit: A^2/fs
    diffusion_coeff *= 1e-5 #final unit: m^2/s
    diffusion_err = res.stderr*1e-5 #final unit: m^2/s

    # Two-sided t-test for error in slope
    from scipy.stats import t
    tinv = lambda p, df: abs(t.ppf(p/2, df))
    ts = tinv(0.05, len(time)-2)
    return diffusion_coeff, ts*diffusion_err, res.rvalue**2


def get_slope(time, msd):
    from scipy.stats import linregress
    res = linregress(time,msd)
    return res.slope, res.intercept, res.stderr, res.rvalue**2


def get_start_index(time, msd, threshold, window, start_id=0):
    window = int(window)
    start_id = int(start_id)
    ids = {}
    for i, t in enumerate(time[start_id:]):
        if i > window:
            
            slope, intercept, err, r2 = get_slope(np.log(time[i:i+window]),np.log(msd[i:i+window]))
            if abs(slope-1) < threshold:
                ids[start_id+i] = r2
    idx, r2 = max(ids.items(), key=lambda x: x[1])
    return idx, r2


def rvalue_formatting(x):
    if x < 0.8:
        return 'background-color: red'
    elif x < 0.9:
        return 'background-color: orange'
    elif x < 0.95:
        return 'background-color: yellow'
    else:
        return None

def rvalue_formatting_latex(x, trunc=6):
    if x < 0.8:
        return r'\color{RED}'+str(x)[:trunc]
    elif x < 0.9:
        return r'\color{ORANGE}'+str(x)[:trunc]
    elif x < 0.95:
        return r'\color{YELLOW}'+str(x)[:trunc]
    else:
        return str(x)[:trunc]

def get_colour(name):
    cset = tc.tol_cset('bright')
    colmap = {
        'EMC':'blue',
        'EC:EMC (3:7)':'green',
        'EC:EMC (7:3)':'yellow',
        'EC':'red',
    }
    return getattr(cset, colmap[name])


def get_mol_positions(atoms):
	positions = []
	mol_set = set(atoms.arrays['molID'])
	for ID in mol_set:
		mol = atoms[atoms.arrays['molID'] == ID]
		com = mol.get_center_of_mass()
		positions.append(com)
	positions = np.array(positions)
	return positions


def mol_rdf(atoms, rmax, nbins, return_num_mols=False):
	positions = get_mol_positions(atoms)
	nmols, dim = positions.shape
	box_length = atoms.get_volume()**(1/3)
	for i, p in enumerate(positions):
		# PBC
		delta = positions[i+1:]-positions[i]
		delta -= box_length*np.round(delta/box_length)
		
		# Get counts
		dists = np.linalg.norm(delta, axis=1)
		s = sorted(dists[dists <= rmax])[1:]
		if i == 0:
			counts, bins = np.histogram(s, nbins, (0,rmax))
		else:
			counts += np.histogram(s, nbins, (0,rmax))[0]

	if return_num_mols:
		return counts, bins, nmols
	else:
		return counts, bins
     

def compute_mol_rdf(traj, rmax=None, nbins=100, return_rho=False):
	for i, atoms in enumerate(traj):
		if i == 0:
			counts, bins, nmols = mol_rdf(atoms, rmax=rmax, nbins=nbins, return_num_mols=True)
			volume = atoms.get_volume()
			box_length = volume**(1/3)
			r = 0.5*(bins[1:]+bins[:-1])
		else:
			counts += mol_rdf(atoms, rmax=rmax, nbins=nbins)[0]
	
	num_snapshots = i+1
	nvalid = nmols/2
	n_k = counts/(nvalid*num_snapshots)	
	dr = rmax/nbins
	shell_volumes = 4/3*np.pi*((r+dr)**3-r**3)
	rho = nmols/volume
	
	g = n_k/(rho*shell_volumes)
	if return_rho:
		return g, r, rho
	else:
		return g, r


def get_mol_rdfs(path, tags, slice=':'):
    import glob
    from aseMolec import anaAtoms as aa

    rdfs = {}
    number_densities = {}
    for tag in tags:
        data_set, labels, seed, dset_sample = tag.split('/')
        flist = [name for name in glob.glob(f'{path}{tag}/dynamics/*') if '.xyz' in name]
        name = f'{data_set}_{labels}'
        print(name, dset_sample, seed)

        for file in flist:
            print(file)
            comp = file.split('/')[-1].split('.')[0][4:-5]
            comp = comp_map[comp]
            traj = read(file, slice)
            aa.find_molecs(traj, fct=1.0)
            g, r, rho = compute_mol_rdf(traj, rmax=11, return_rho=True)

            if name not in rdfs:
                rdfs[name] = {}
            if dset_sample not in rdfs[name]:
                rdfs[name][dset_sample] = {}
            if seed not in rdfs[name][dset_sample]:
                rdfs[name][dset_sample][seed] = {}
            
            if name not in number_densities:
                number_densities[name] = {}
            if dset_sample not in number_densities[name]:
                number_densities[name][dset_sample] = {}
            if seed not in number_densities[name][dset_sample]:
                number_densities[name][dset_sample][seed] = {}

            rdfs[name][dset_sample][seed][comp] = [g, r]
            number_densities[name][dset_sample][seed][comp] = rho
            del traj
    
    return rdfs, number_densities

def get_indices_of_sign_change(lst):
    arr = np.array(lst)
    sign_changes = np.where(np.diff(np.sign(arr)) != 0)[0]
    return sign_changes


def yaml_to_multidf(fname, index_list):
    import yaml
    import pandas as pd
    
    # Load YAML
    with open(fname, "r") as f:
        raw = yaml.safe_load(f)
    f.close()

    # Convert to DataFrame
    df = pd.DataFrame(raw['metadata'])

    # Set MultiIndex
    df = df.set_index(index_list)
    return df


def download_file(url, output_file):
	import requests
	from pathlib import Path
	print(f"Downloading {url} -> {output_file}")
	response = requests.get(url)
	response.raise_for_status()

	Path(output_file).parent.mkdir(parents=True, exist_ok=True)

	with open(output_file, "wb") as f:
		f.write(response.content)


def unzip_file(zip_path, extract_to):
	from pathlib import Path
	import zipfile
	print(zip_path)
	print(f"Unzipping {zip_path} -> {extract_to}")
	Path(extract_to).mkdir(parents=True, exist_ok=True)

	with zipfile.ZipFile(zip_path, "r") as z:
		z.extractall(extract_to)


def download_unpack(root, config_path, key):
	import yaml; import os.path
	with open(config_path, "r") as f:
		config = yaml.safe_load(f)

	for item in config[key]:
		url = item["url"]
		output_file = item["output_file"]
		extract_to = item["extract_to"]

		if not os.path.isfile(root+output_file):
			download_file(url, root+output_file)
		unzip_file(root+output_file, root+extract_to)


def flatten_results(df, prop_handle, merge_type):
    if merge_type == 'mean':
        return df[prop_handle].mean()
    elif merge_type == 'std':
        return df[prop_handle].std()


def get_committee_results(df, group_by, droplist):
    dfs = []
    for group_name, group in df.groupby(group_by):
        tmp = []
        for comp_name, comp in group.groupby('Composition'):
            t = comp.iloc[0].copy()
            t['Density /g*cm-3'] = flatten_results(comp, 'Density /g*cm-3', merge_type='mean')
            t['Density std. /g*cm-3'] = flatten_results(comp, 'Density /g*cm-3', merge_type='std')
            t['Diff. coeff. /1e6 cm2*s-1'] = flatten_results(comp, 'Diff. coeff. /1e6 cm2*s-1', merge_type='mean')
            t['Diff. fit slope error /1e6 cm2*s-1'] = flatten_results(comp, 'Diff. coeff. /1e6 cm2*s-1', merge_type='std')
            t = t.drop(droplist)
            tmp.append(t)
        dfs.append(pd.DataFrame(tmp)) 
    return pd.concat(dfs)


def self_interaction_mask(array):
    SI = [[0 if a==b else 1 for b in array] for a in array]
    return np.array(SI)


def get_pair_histogram(atoms, rmax, nbins, pbc, probability=False, inter_only=False, threshold=None, molIDs=None):
    if pbc:
        distances = atoms.get_all_distances(mic=True)
    else:
        distances = atoms.get_all_distances()
    
    if inter_only and (molIDs is not None):
        distances = distances*self_interaction_mask(molIDs)
        
        if threshold is not None:
            ids = np.where((distances>0) & (distances<threshold))[0]
            if len(ids) > 0:
                print(ids)

        iu = np.triu_indices(len(distances), k=1)
        dists = distances[iu]
        del iu
    else:
        dists = distances
    
    dr = rmax/nbins
    edges = np.arange(0, rmax+dr, dr)
    counts, bins = np.histogram(dists, bins=edges)
    counts[0] = 0
    r = 0.5*(bins[1:]+bins[:-1])
    n_r = counts/(len(atoms)-1)
    del atoms
    del dists

    if probability:
        p_r = n_r/np.cumsum(n_r)*dr
        return p_r, r
    else:
        return n_r, r


def compute_rdf(atoms, rmax=None, nbins=100, rho=None, return_rho=False, nbr_density=False, pbc=True, probability=False, inter_only=False, use_max=False, threshold=None):
    if inter_only and ('molID' not in atoms[0].arrays):
        aa.find_molecs(atoms, fct=1.0)
        
    for i, a in enumerate(atoms):
        if inter_only and ('molID' in a.arrays):
            molIDs = a.arrays['molID']
        else:
            molIDs = None

        
        if (rho is None) and (nbr_density is False):
            rho = len(a)/a.get_volume()
        elif nbr_density is True:
            rho = 1

        if i == 0:
            counts, r = get_pair_histogram(a, rmax=rmax, nbins=nbins, pbc=pbc, probability=probability, inter_only=inter_only, threshold=threshold, molIDs=molIDs)
            max_counts = counts
            max_rho = rho
        else:
            new_counts = get_pair_histogram(a, rmax=rmax, nbins=nbins, pbc=pbc, probability=probability, inter_only=inter_only, threshold=threshold, molIDs=molIDs)[0]
            max_counts = np.maximum(max_counts, new_counts)
            max_rho = max(max_rho, rho)
            counts += new_counts
        del a

    num_images = len(atoms)
    dr = rmax/nbins
    
    if use_max:
        counts = max_counts
        n_k = 2*counts
        rho = max_rho
    else:
        n_k = 2*counts/num_images

    n_ideal = 4/3*np.pi*((r+dr)**3-r**3)*rho
    
    if nbr_density:
        g = n_k
    else:
        g = n_k/n_ideal
    
    if return_rho:
        return g, r, rho
    else:
        return g, r


def evaluate_nbr_distributions(dict_item, **kwargs):
    label, atoms = dict_item
    g, r, rho = compute_rdf(atoms, nbr_density=False, return_rho=True, **kwargs)
    n, _ = compute_rdf(atoms, nbr_density=True, return_rho=False, **kwargs)
    result = (g, n, r, rho)
    return label, result


def parallelize_eval(partial_func, data):
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor() as executor:
        results = executor.map(partial_func, data)
    return results

def get_mol_rdfs(path, tags, slice=':'):
    import glob
    from aseMolec import anaAtoms as aa

    rdfs = {}
    number_densities = {}
    for tag in tags:
        data_set, labels, seed, dset_sample = tag.split('/')
        flist = [name for name in glob.glob(f'{path}{tag}/dynamics/*') if '.xyz' in name]
        name = f'{data_set}_{labels}'
        print(name, dset_sample, seed)

        for file in flist:
            print(file)
            comp = file.split('/')[-1].split('.')[0][4:-5]
            comp = comp_map[comp]
            traj = read(file, slice)
            aa.find_molecs(traj, fct=1.0)
            g, r, rho = compute_mol_rdf(traj, rmax=11, return_rho=True)

            if name not in rdfs:
                rdfs[name] = {}
            if dset_sample not in rdfs[name]:
                rdfs[name][dset_sample] = {}
            if seed not in rdfs[name][dset_sample]:
                rdfs[name][dset_sample][seed] = {}
            
            if name not in number_densities:
                number_densities[name] = {}
            if dset_sample not in number_densities[name]:
                number_densities[name][dset_sample] = {}
            if seed not in number_densities[name][dset_sample]:
                number_densities[name][dset_sample][seed] = {}

            rdfs[name][dset_sample][seed][comp] = [g, r]
            number_densities[name][dset_sample][seed][comp] = rho
            del traj
    print('finished!')
    return rdfs, number_densities
