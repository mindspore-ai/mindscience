# Copyright 2023 Huawei Technologies Co., Ltd & CPL YiQin GAO Research Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"assign"
import os
import stat
import time
import io
import itertools
import pickle
from Bio.PDB import PDBParser
import numpy as np

from commons.res_constants import EQUI_VARIANCE
from assign.init_assign import get_ur_list2

YES = "yes"
NO = "no"


def make_ensembles(pdb_list):
    '''make_ensembles'''
    ensembles = []
    for pdb_file in pdb_list:
        with open(pdb_file, "r") as f:
            pdb_str = f.read()
        pdb_fh = io.StringIO(pdb_str)
        parser = PDBParser()
        structure = parser.get_structure('none', pdb_fh)
        model = list(structure.get_models())[0]  # todo: multi models
        chain = list(model.get_chains())[0]  # todo: multimer

        ensemble = {}
        start_id = -1
        for _, res in enumerate(chain):
            res_id = int(res.id[1])

            if start_id == -1:
                start_id = res_id
            res_id = res_id - start_id + 1

            ensemble[res_id] = {"aatype": res.resname}

            for atom in res:
                ensemble.get(res_id)[atom.name] = atom.coord
        ensembles.append(ensemble)
    return ensembles


def calculatebounds(factor, peaks, settings, bound_corrected=None, ensemble=None):
    """
    calculate lower and upper bounds for every peak using
    the calibration 'factor'. values are stored.
    'bound_corrected': list of restraints which are classified
    as correct after bound-modification.
    """

    if settings.get("calibration").get('use_bounds') == YES:
        va_settings = settings.get("violation_analysis")

        new_lbound = va_settings.get('lower_bound_correction').get('value')
        new_ubound = va_settings.get('upper_bound_correction').get('value')
        new_d = (new_ubound - new_lbound) / 2.

        # [r['upper_bound'] = new_ubound for r in peaks]#dict
        # [r['lower_bond'] = new_lbound for r in peaks]
        # [r['distance'] = new_d for r in peaks]

        for r in peaks:
            r['upper_bound'] = new_ubound
            r['lower_bond'] = new_lbound
            r['distance'] = new_d

        return

    factor = np.power(factor, 1. / 6)

    peak_sizes = get_refpeak_sizes(peaks, settings.get("calibration").get("volume_or_intensity"))

    # Malliavin/Bardiaux rMat
    cs = settings.get("calibration")
    if cs.get('relaxation_matrix') == YES and ensemble is not None:

        ispa_peak_sizes = np.array([p.getIspa() for p in peaks])
        peak_theoric_vol = np.array([p.getTheoricVolume() for p in peaks])

        ratio = ispa_peak_sizes / peak_theoric_vol
        distances = factor * np.power(peak_sizes * ratio, -1. / 6)


    else:

        distances = factor * np.power(peak_sizes, -1. / 6)

    ## TODO: hard-coded 0.125

    if cs['error_estimator'] == 'intensity':
        errors = 0.125 * np.power((factor * np.power(peak_sizes, -1. / 6)), 2.)

    else:
        errors = 0.125 * np.power(distances, 2.)

    ## lower bounds are >= 0.

    lower_bounds = np.clip(distances - errors, 0., 1.e10)
    upper_bounds = distances + errors

    for i, _ in enumerate(peaks):
        peak = peaks[i]

        peak['distance'] = distances[i]
        peak['lower_bond'] = lower_bounds[i]
        peak['upper_bound'] = upper_bounds[i]

    ## Set new (fixed) bounds for bound-corrected restraints

    if bound_corrected:
        va_settings = settings.get("violation_analysis")

        if va_settings.get('lower_bound_correction')['enabled'] == YES:
            new_bound = va_settings.get('lower_bound_correction').get('value')
            [r.setLowerBound(new_bound) for r in bound_corrected]

        if va_settings.get('upper_bound_correction').get('enabled') == YES:
            new_bound = va_settings.get('upper_bound_correction').get('value')
            [r.setUpperBound(new_bound) for r in bound_corrected]


def getdistances(atom1, atom2, ensemble=None, number_of_best_structures=7):
    '''getdistances'''
    res1 = atom1.get('res')
    atype1 = atom1.get('name')
    restype1 = atom1.get("restype")
    res2 = atom2.get('res')
    atype2 = atom2.get('name')
    restype2 = atom2.get("restype")

    if ensemble is None:
        raise ValueError('No coordinates have been read in.')
    ## calculate distances
    d = [1000000.0] * len(ensemble)

    if res1 in ensemble[0].keys() and res2 in ensemble[0].keys():
        atype1_equi = set(list(ensemble[0][res1].keys())).intersection(
            EQUI_VARIANCE.get(restype1).get(atype1).get("equivariance"))

        atype2_equi = set(list(ensemble[0][res2].keys())).intersection(
            EQUI_VARIANCE.get(restype2).get(atype2).get("equivariance"))

        if atype1_equi and atype2_equi:
            x_all = []
            y_all = []
            for atype1, atype2 in itertools.product(atype1_equi, atype2_equi):

                x_perstruct = []
                y_perstruct = []
                for _, structure in enumerate(ensemble):
                    if atype1 in structure[res1].keys() and atype2 in structure[res2].keys():
                        x_perstruct.append(structure[res1][atype1])
                        y_perstruct.append(structure[res2][atype2])

                x = np.stack(x_perstruct)
                x_all.append(x)
                y = np.stack(y_perstruct)
                y_all.append(y)
            x_all = np.stack(x_all)
            y_all = np.stack(y_all)

            d = np.sqrt(np.sum(np.power(x_all - y_all, 2), -1))

            d = np.min(d, axis=0)

        else:
            print(res1, restype1, ensemble[0][res1].keys(), EQUI_VARIANCE.get(restype1).get(atype1).get("equivariance"))
            print(res2, restype2, ensemble[0][res2].keys(), EQUI_VARIANCE.get(restype2).get(atype2).get("equivariance"))
            print(atom1, atom2)

    n = number_of_best_structures
    if n != 'all':
        d = d[:n]
    return d


def effective_distances(contribution, ensemble):
    '''effective_distances'''
    d = [getdistances(sp.get('Atom1'), sp.get('Atom2'), ensemble) for sp in contribution.get('spin_pairs')]

    ## for each structure: calculate partial volume

    volumes = np.sum(np.power(d, -6.), axis=0)

    return np.power(volumes, -1. / 6)


def average(x, n=None, exponent=1., axis=0):  # y
    """
    Returns (n^{-1} sum_1^n x_i^exponent)^{1/exponent}.
    if 'n' is not None, it is used instead of len(x)
    sum is taken wrt to axis 'axis'
    """
    x = np.array(x, float)

    if n is None:
        n = np.shape(x)[axis]

    return (np.sum(np.power(x, exponent), axis) / n) ** (1. / exponent)


def analysepeak2(ensemble, peak, tol, va_settings, lower_correction=None,
                 upper_correction=None, sig_mode="fix"):
    '''analysepeak2'''

    ## for every structure: calculate effective contributon-distance
    ## d_avg is a [n_c x n_s] dim. array
    ## n_c: number of contributions
    ## n_s: number of structures in ensemble

    d_avg = [effective_distances(c, ensemble) \
             for c in peak.get('analysis').get('contributions')]
    d_avg = np.stack(d_avg)

    ## Effective lower/upper bounds

    if tol is None:
        tol = va_settings.get('violation_tolerance')

    if lower_correction is not None:
        lower = lower_correction
    else:
        lower = peak.get('lower_bond')

    if upper_correction is not None:
        upper = upper_correction
    else:
        upper = peak.get('upper_bound')

    dist = peak.get('distance')

    if sig_mode == "fix":
        violated_lower = np.less(d_avg, lower - tol)
        violated_upper = np.greater(d_avg, upper + tol)
    else:
        violated_lower = np.less(d_avg, dist - tol)
        violated_upper = np.greater(d_avg, dist + tol)

    violated = np.logical_or(violated_lower, violated_upper)

    violated = 1 - (violated.shape[0] > violated).sum(axis=0) > 0

    r_viol = float(sum(violated)) / float(len(violated))

    return r_viol


def analysepeak(ensemble, peak, tol, va_settings,
                lower_correction=None, upper_correction=None, sig_mode="fix"):
    '''analysepeak'''

    ## for every structure: calculate effective contributon-distance
    ## d_avg is a [n_c x n_s] dim. array
    ## n_c: number of contributions
    ## n_s: number of structures in ensemble

    d_avg = [effective_distances(c, ensemble) \
             for c in peak.get('analysis').get('contributions')]

    d_avg = np.power(np.sum(np.power(d_avg, -6), axis=0), -1. / 6)

    ## Effective lower/upper bounds

    if tol is None:
        tol = va_settings.get('violation_tolerance')

    if lower_correction is not None:
        lower = lower_correction
    else:
        lower = peak.get('lower_bond')

    if upper_correction is not None:
        upper = upper_correction
    else:
        upper = peak.get('upper_bound')

    dist = peak.get('distance')
    ## calculate fraction of violated distances
    ## 1: distance is violated, 0: distance lies within bounds

    if sig_mode == "fix":
        violated_lower = np.less(d_avg, lower - tol)
        violated_upper = np.greater(d_avg, upper + tol)
    else:
        violated_lower = np.less(d_avg, dist - tol)
        violated_upper = np.greater(d_avg, dist + tol)

    violated = np.logical_or(violated_lower, violated_upper)

    r_viol = float(sum(violated)) / float(len(violated))

    return r_viol


def tolerance(ensemble, peak):
    '''tolerance'''

    ## for every structure: calculate effective contributon-distance
    ## d_avg is a [n_c x n_s] dim. array
    ## n_c: number of contributions
    ## n_s: number of structures in ensemble

    d_eff = [effective_distances(c, ensemble) \
             for c in peak.getContributions()]

    ## Effective lower/upper bounds
    dist = peak.getDistance()

    d_eff = np.power(sum(np.power(d_eff, -6), axis=0), -1. / 6)
    for i, _ in enumerate(d_eff):
        d_eff[i] = abs(dist - d_eff[i])

    return d_eff


def doviolationanalysis(restraints, ensemble, va_settings):
    """
    'restraints': list of AriaPeaks. The bounds of every
    restraint in that list is checked against distances found
    in the 'ensemble'.
    'targets': list of AriaPeaks. The violationAnalyser will
    store all intermediate results in their analysis-section.
    Note: we assume, that peaks[i] corresponds to results[i]
    for all i !. If a restraint has been violated, the
    corresponding 'target'_restraint will be marked as violated.
    """

    violated = []
    non_violated = []

    ## get threshold for current iteration

    if va_settings.get('sigma_mode') == 'auto':
        ecars = []
        for restraint in restraints:
            temp_ecars = tolerance(ensemble, restraint)
            for ecar in temp_ecars:
                ecars.append(ecar)

        ecar_avg = sum(ecars) / len(ecars)
        tol = 0
        for n_ecar in ecars:
            tol = tol + np.power(n_ecar - ecar_avg, 2)
        print("TOLERANCE ", np.power(tol / len(ecars), 1. / 2))
        tol = np.power(tol / len(ecars), 1. / 2) * va_settings['violation_tolerance']
        print("AVG RESTRAINT ", tol)

    else:
        tol = None

    for restraint in restraints:
        r_viol = analysepeak(ensemble, restraint, tol, va_settings, sig_mode=va_settings['sigma_mode'])

        ##
        ## If a restraint has been violated in too many structures
        ## (according to 'threshold'), mark is a violated.
        ##
        threshold = va_settings.get('violation_threshold')
        if r_viol > threshold:
            restraint.get('analysis')['is_violated'] = 1
            violated.append(restraint)

        else:
            restraint.get('analysis')['is_violated'] = 0
            non_violated.append(restraint)

    ## For violated restraints: if bound-correction is enabled,
    ## repeat violation-analysis with modified bounds.

    if va_settings.get('lower_bound_correction').get('enabled') == YES:
        new_lower = va_settings.get('lower_bound_correction').get('value')
    else:
        new_lower = None

    if va_settings.get('upper_bound_correction').get('enabled') == YES:
        new_upper = va_settings.get('upper_bound_correction').get('value')
    else:
        new_upper = None

    if new_lower is not None or new_upper is not None:

        ## We forget 'store_analysis' here, since it has already
        ## been stored (if set).

        r_viol = [analysepeak(ensemble, r, tol, va_settings,
                              lower_correction=new_lower,
                              upper_correction=new_upper,
                              sig_mode=va_settings.get('sigma_mode')) for r in violated]

        ## List of restraint-indices which are no longer
        ## violated after bound modification.

        indices = np.flatnonzero(np.less(r_viol, threshold))
        new_non_violated = [violated[i] for i in indices]

        [r.analysis.isViolated(0) for r in new_non_violated]

    else:
        new_non_violated = None

    return violated, non_violated, new_non_violated


def get_refpeak_sizes(peaks, volume_or_intensity="volume"):
    '''get_refpeak_sizes'''
    ref_peaks = [p.get('ref_peak') for p in peaks]

    if volume_or_intensity == 'volume':
        peak_sizes = [p.get('volume')[0] for p in ref_peaks]
    else:
        peak_sizes = [p.get('intensity')[0] for p in ref_peaks]
    return peak_sizes


def calculatepeaksize(peak, ensemble):
    '''calculatepeaksize'''

    if not peak:
        raise ValueError('No contributions in xpk: %d' %
                         peak.getId())

    ## for each structure: calculate effective distance
    ## for contribution, i.e. distances between atoms
    ## of every spinpair are averaged according to the
    ## type of the given contribution.

    avg_distances = [effective_distances(c, ensemble) for c in peak.get('analysis').get('contributions')]

    ## for each contribution: calculate ensemble-average
    ## TODO: average -> _average, probably faster
    avg_distances = average(avg_distances, axis=1)

    ## calculate NOEs
    d = np.power(avg_distances, -6.)

    ## NOE is sum over partial NOEs

    if np.sum(d) == np.inf:
        print(d)

        for c in peak.get('analysis').get('contributions'):
            if np.sum(effective_distances(c, ensemble)) < 0.00001:
                print(c)
            print(effective_distances(c, ensemble))

        print(peak)
        raise ValueError
    return np.sum(d)


def dodumbocalibration(peaks, volume_or_intensity="volume"):
    '''dodumbocalibration'''
    peak_sizes = get_refpeak_sizes(peaks, volume_or_intensity)

    ## Assume that average distance of atoms
    ## causing an NOE is 3.0A

    d_calib = 3.0

    sum_noe_calc = len(peaks) * (d_calib ** -6)

    factor = sum(peak_sizes, dtype=np.float64) / sum_noe_calc

    return factor


def calculateestimator(peaks, ensemble, calibration_settings, use_cutoff=1):
    '''calculateestimator'''
    if not peaks:
        raise (ValueError, 'No peaks specified.')

    if calibration_settings['volume_or_intensity'] == 'volume':
        exp_peak_sizes = [p['ref_peak']['volume'][0] \
                          for p in peaks]  # dict
    else:
        exp_peak_sizes = [p['ref_peak']['intensity'][0] \
                          for p in peaks]  # dict

    model_peak_sizes = np.array([calculatepeaksize(p, ensemble) for p in peaks])

    ## larger than noe_cutoff.

    if use_cutoff:
        noe_cutoff = calibration_settings['distance_cutoff'] ** (-6.)
    else:
        noe_cutoff = 0.

    strong_noes = np.greater_equal(model_peak_sizes, noe_cutoff)

    sum_noe_model = np.sum(np.compress(strong_noes,
                                       model_peak_sizes))
    sum_noe_exp = np.sum(np.compress(strong_noes,
                                     exp_peak_sizes), dtype=np.float64)

    ## if there are no NOEs larger than noe_cutoff,

    if sum_noe_model <= 1.e-30:
        return None

    ## calculate estimator
    if calibration_settings['estimator'] == 'ratio_of_averages':
        factor = sum_noe_exp / sum_noe_model

    ## store calculated peak-size

    return factor


def docalibration(restraints, ensemble, all_settings):
    '''docalibration'''
    # BARDIAUX 2.2
    # ConstraintList can bypass calibration

    calibration_settings = all_settings["calibration"]
    if calibration_settings['use_bounds'] == YES:
        print('calibration disabled')
        return 1.

    if ensemble is None:
        factor = dodumbocalibration(restraints)
    else:
        factor = calculateestimator(restraints, ensemble, calibration_settings,
                                    use_cutoff=1.0)

        if factor is None:
            d_cutoff = calibration_settings['distance_cutoff']

            s = 'Could not perform 1st calibration, since ' + \
                'no distances less than %.1f A were found in the ' + \
                'ensemble. Omitting distance-cutoff and ' + \
                'calibrating again...'

            print(s % d_cutoff)

            factor = calculateestimator(restraints, ensemble, calibration_settings,
                                        use_cutoff=0.0)

    return factor


def filter_weights(weights, cutoff):
    """
    Let I be the index-list of weights whose
    sum is >= cutoff. The function returns indices
    in range(0, len(weights)) which are not in I
    """

    ## sort weights in descending order
    indices = np.argsort(weights)
    indices = np.take(indices, np.arange(len(indices) - 1, -1, -1))
    s_weights = np.take(weights, indices)

    x = np.add.accumulate(s_weights)

    try:
        index = np.flatnonzero(np.greater(x, cutoff))[1]
    except Exception as _:
        index = len(indices)

    ## we limit the number of contributing
    ## weights to max_n.

    ## BARDIAUX
    # test maxn remove peak

    ## Return set of large and small weights.

    return indices[:index], indices[index:]


def assign(restraint_list, ensemble, all_settings, filter_contributions=1):
    '''assign'''

    def average1(x):
        return np.sum(np.array(x), axis=0) / len(x)

    def variance(x, avg=None):
        if avg is None:
            avg = average1(x)

        return np.sum(np.power(np.array(x) - avg, 2), axis=0) / (len(x) - 1.)

    def standarddeviation(x, avg=None):
        return np.sqrt(variance(x, avg))

    all_contributions = []
    weights = []

    for restraint in restraint_list:

        distances = []

        contributions = restraint['analysis']['contributions']
        all_contributions.append(contributions)

        for contribution in contributions:

            ## for every structure: get effective distance
            ## for 'contribution'

            d = effective_distances(contribution, ensemble)

            d_avg = np.average(d)
            distances.append(d_avg)

            if len(d) > 1:
                sd = standarddeviation(d, avg=d_avg)
            else:
                sd = None

            contribution['average_distance'] = [d_avg, sd]

        ## calculate partial NOE wrt to ensemble-averaged
        ## distance. The partial NOE serves as weight which
        ## subsequently will be normalized to 1.

        w = np.power(distances, -6.)

        ## normalize weights and store weights

        w /= np.sum(w)

        weights.append(w)

    settings = all_settings["assign"]
    cutoff = settings['weight_cutoff']

    # if cutoff is not None:
    if cutoff is not None and filter_contributions:

        ## 1. disable all contributions according to
        ##    the partial-assignment scheme.
        ## 2. allow at most 'max_contributions' contributions

        for i, _ in enumerate(weights):

            w = weights[i]
            c = all_contributions[i]

            on, off = filter_weights(w, cutoff)

            for index in off:
                c[index]['weight'] = 0.
            for index in on:
                c[index]['weight'] = w[index]
    else:

        ## if cutoff is not set, enable all contribution.

        ## note: setting 'max_contributions' does not
        ## apply in that case since we have no rule
        ## how to select contributions which remain
        ## active.

        for i, _ in enumerate(weights):

            contributions = all_contributions[i]
            w = weights[i]

            for j, _ in enumerate(w):
                contributions[j]["weight"] = w[j]


def run_iteration(peaks, ensemble, all_settings):
    '''run_iteration'''
    if ensemble is None:
        ## BARDIAUX
        # test maxn remove peak
        maxn = all_settings["assign"]['max_contributions']
        for p in peaks:
            for c in p['analysis']['contributions']:
                if c is not None and isinstance(c.get('weight'), float) and c['weight'] > 0.:
                    activate = c
                    if len(activate) > maxn and not p['ref_peak']['reliable']:
                        p['active'] = 0

        return peaks

    else:

        ## Calculate initial calibraton factor.

        factor = docalibration(peaks, ensemble, all_settings)

        ## Calculate upper/lower bounds for restraints of current
        ## iteration.

        t = time.time()

        calculatebounds(factor, peaks, all_settings, ensemble=ensemble)

        s = '1st calibration and calculation of new ' + \
            'distance-bounds done (calibration factor: %e)'
        print(s % factor)
        print('Time: %ss' % str(time.time() - t))

        ##
        ## Violation Analysis
        ##
        ## Assess every restraint regarding its degree of violation.
        ##
        ## Violated restraints will be disabled for the
        ## current iteration and thus will not be used during
        ## structure calculation.
        ##

        t = time.time()

        violated, non_violated, new_non_violated = doviolationanalysis(peaks, ensemble, \
                                                   all_settings['violation_analysis'])

        ## Augment set of non-violated restraints

        if new_non_violated:
            non_violated += new_non_violated

        n = len(peaks)
        n_viol = len(violated)
        p_viol = n_viol * 100. / n

        s = 'Violation analysis done: %d / %d restraints ' + \
            '(%.1f %%) violated.'

        print(s % (n_viol, n, p_viol))

        if new_non_violated:
            s = 'Number of valid restraints has been increased ' + \
                'by %d (%.1f%%) after applying a bound-correction.'

            p_new = len(new_non_violated) * 100. / n

            print(s % (len(new_non_violated), p_new))

        print('Time: %ss' % str(time.time() - t))

        ##
        ## 2nd calibration - wrt to non-violated restraints.
        ## If no restraints have been violated, we use the
        ## 1st calibration factor.
        ## Again, we do not store results.
        ##

        if non_violated:
            factor = docalibration(non_violated, ensemble, all_settings)

        ##
        ## Activate restraints explicitly.
        ## We consider a restraint as active, if it has
        ## not been violated or if its reference cross-peak is
        ## 'reliable'.
        ##

        for r in peaks:
            if not r['analysis']['is_violated'] or r['ref_peak']['reliable']:
                r['active'] = 1  # ?
            else:
                r['active'] = 0

        ## Store final calibration factor for current iteration.

        ## Calculate upper/lower bounds for restraint-list
        ## used in the current iteration. I.e. these bounds
        ## will be used to calculated the structures.

        t = time.time()

        calculatebounds(factor, peaks, all_settings, new_non_violated, ensemble=ensemble)

        s = 'Final calibration and calculation of new distance-bounds' + \
            ' done (calibration factor: %e).' % factor
        print(s)
        print('Time: %ss' % str(time.time() - t))
        ##
        ## Partial assignment for restraint-list used in
        ## current iteration, i.e. for all restraints:
        ## Calculate weight for every contribution
        ## and (depends possibly on partial analyser
        ## settings) throw away 'unlikely' contributions.
        ##
        ## If we do not have an ensemble, all contributions
        ## are activated.
        ##

        t = time.time()

        assign(peaks, ensemble, all_settings, filter_contributions=True)
        print('Partial assignment done.')
        print('Time: %ss' % str(time.time() - t))
        return None


def assign_iteration(ur_tuple_path, ur_path, pdb_path, peak_list_path, all_settings, filter_names=None):
    '''assign_iteration'''
    for path in [ur_tuple_path, ur_path]:
        os.makedirs(path, exist_ok=True)

    names = os.listdir(pdb_path)
    names = [name for name in names if ".pdb" in name]
    pdb_list = [os.path.join(pdb_path, name) for name in names]

    prot_names = sorted(os.listdir(peak_list_path))

    if filter_names:
        prot_names = list(set(prot_names).intersection(set(list(filter_names))))

    for prot_name in prot_names:
        if prot_name == "2K0M":
            continue  # bad peak list
        prot_path = os.path.join(peak_list_path, prot_name)
        file_list = os.listdir(prot_path)

        cur_pdb_list = [name for name in pdb_list if prot_name in name]

        if not cur_pdb_list:
            continue
        ensembles = make_ensembles(cur_pdb_list)

        peak_file_list = []
        for file in file_list:
            if file.startswith("new_spectrum"):
                peak_file_list.append(file)

        all_peak_list = []
        for peak_file in peak_file_list:
            full_peak_filename = os.path.join(prot_path, peak_file)
            with open(full_peak_filename, "rb") as f:
                peak_list = pickle.load(f)

            print("\n\n", full_peak_filename, len(peak_list))
            if not peak_list:
                continue
            run_iteration(peak_list, ensembles, all_settings)

            all_peak_list.extend(peak_list)

        ur_list, ur_list_tuple = get_ur_list2(all_peak_list, long_distance_threshold=0)
        os_flags = os.O_RDWR | os.O_CREAT
        os_modes = stat.S_IRWXU
        with os.fdopen(os.open(ur_path + "/" + prot_name + ".pkl", os_flags, os_modes), "wb") as f:
            pickle.dump(ur_list, f)
        with os.fdopen(os.open(ur_tuple_path + "/" + prot_name + ".pkl", os_flags, os_modes), "wb") as f:
            pickle.dump(ur_list_tuple, f)
