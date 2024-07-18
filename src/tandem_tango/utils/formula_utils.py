#####################################################
# gk@reder.io
#####################################################
import sys
import numpy as np
import molmass
import cvxpy as cp
import more_itertools
import re

#####################################################

def charged_mass(mass, charge):
    if type(mass) == str:
        mass = molmass.Formula(mass).isotope.mass
    em = (-1 * int(charge)) * molmass.ELECTRON.mass
    res = mass + em
    return(res)

def uncharged_mass(mass, charge):
    if type(mass) == str:
        mass = molmass.Formula(mass).isotope.mass
    em = int(charge) * molmass.ELECTRON.mass
    res = mass + em
    return(res)

def adduct_mass(mass, adduct):
    adduct_amends, adduct_charge = parse_adduct(adduct)
    adduct_atoms = ''.join([''.join([x[0] for i in range(x[1])]) for x in adduct_amends])
    if adduct_atoms != '':
        adduct_mass = form_to_mz(adduct_atoms, adduct_charge)
    else:
        adduct_mass = 0.0
    res = charged_mass(mass, adduct_charge)
    res = res + adduct_mass
    return(res)


def parse_adduct(adduct):
    d_lookup = {
        # return [(Atom, amount)], charge)

        # Positive mode adducts
        "[M]+" : ([], 1),
        "[M+H]+" : ([('H', 1)], 1),
        "[M+Na]+" : ([('Na', 1)], 1),
        "[M+K]+" : ([('K', 1)], 1),
        "[M+NH4]+" : ([('N', 1), ("H", 4)], 1),
        "[M+CH3OH+H]+" : ([('C', 1), ('H', 5), ('O', 1)], 1),
        "[M+ACN+H]+" :([('C', 2), ('H', 4), ('N', 1)], 1),
        "[M+2H]2+" : ([('H', 2)], 2),
        # Negative mode adducts
        "[M]-" : ([], -1),
        "[M-H]-" : ([('H', -1)], -1),
        "[M-H2O-H]-" : ([('H', -3), ('O', -1)], 1),
        "[M+Na-2H]-" : ([('Na', 1), ('H', -2)], -1),
        "[M+K-2H]-" : ([('K', 1), ('H', -2)], -1),
        "[M+Cl]-" : ([('Cl', 1)], -1),
        "[M+FA-H]-" : ([('H', 1), ('C', 1), ('O', 2)], -1),
        "[M+HAc-H]-" : ([('C', 2), ("H", 3), ("O", 2)], -1),
        "[M-2H]2-" : ([('H', -2)], -2)
    }
    if adduct not in d_lookup.keys():
        sys.exit(f"Error - don't know how to parse adduct {(adduct)}")
    return(d_lookup[adduct])

def to_adduct(form, adduct):
    fd = {x[0] : x[1] for x in molmass.Formula(form).composition()}
    adduct_amends = parse_adduct(adduct)[0]
    for (e, add_e) in adduct_amends:
        if e not in fd.keys():
            fd[e] = 0
        fd[e] += add_e
    if np.any([x < 0 for x in list(fd.values())]):
        raise ValueError(f"Created an impossible formula with {form} and adduct {adduct}")
    res = molmass.Formula(''.join([f"{k}{v}" for k, v in fd.items() if v > 0])).formula
    return(res)

def from_adduct(form, adduct):
    fd = {x[0] : x[1] for x in molmass.Formula(form).composition()}
    adduct_amends = parse_adduct(adduct)[0]
    for (e, add_e) in adduct_amends:
        if e not in fd.keys():
            fd[e] = 0
        fd[e] -= add_e
    if np.any([x < 0 for x in list(fd.values())]):
        sys.exit(f"Error - created an impossible formula with {form} and adduct {adduct}")
    res = molmass.Formula(''.join([f"{k}{v}" for k, v in fd.items() if v > 0])).formula
    return(res)

def form_to_mz(form, charge, adduct = None):
    if adduct != None:
        adduct_amends, charge = parse_adduct(adduct)
        form = to_adduct(form, adduct)
    res = charged_mass(molmass.Formula(form).isotope.mass, charge = charge)
    res = abs(res / charge)
    return(res)

def get_refs(ref_formula = "CHNOPSNa"):
    comp = molmass.Formula(ref_formula).composition()
    ref_elements = [x[0] for x in comp]
    t_masses = np.array([molmass.Formula(x).isotope.mass for x in ref_elements])
    return((ref_elements, t_masses))

def form_to_vec(form, ref_formula = "CHNOPSNa"):
    (ref_elements, t_masses) = get_refs(ref_formula)
    fd = {x[0] : x[1] for x in molmass.Formula(form).composition()}
    for e in ref_elements:
        if e not in fd.keys():
            fd[e] = 0
    return(np.array([fd[k] for k in ref_elements]))

def vec_to_form(vec, ref_formula = "CHNOPSNa"):
    (ref_elements, t_masses) = get_refs(ref_formula)
    vec = [round(x, 2) for x in vec]
    res = ""
    res = molmass.Formula("".join([f"{e}{int(vec[i_e])}" if vec[i_e] >= 0.99 else "" for i_e, e in enumerate(ref_elements)])).formula
    return(res)

def find_best_form(mass, parent_form, tolerance_da = 0.005, charge = 0, verbose = False, cvxy_verbose = False, solver = 'SCIP', du_min = None):
    (ref_elements, t_masses) = get_refs(parent_form)
    if verbose:
        print(ref_elements)
        print(t_masses)
    if charge != 0:
        mass = uncharged_mass(mass, charge)
    parent_vec = form_to_vec(parent_form, ref_formula=parent_form)
    if verbose:
        print(parent_vec)
    F = cp.Variable(len(ref_elements), integer = True)
    objective = cp.Minimize(cp.sum_squares(F @ t_masses - mass))
    constraints = []
    for i in range(F.shape[0]):
        constraints.append(F[i] >= 0)
        constraints.append(F[i] <= parent_vec[i])
        if verbose:
            print(f"0 <= {ref_elements[i]} <= {parent_vec[i]}")
    prob = cp.Problem(objective, constraints)
    prob.solve(solver = solver, verbose = cvxy_verbose)
    if verbose:
        print("status:", prob.status)
        print("optimal value", prob.value)
        print("optimal var", F.value)
    best_form = vec_to_form(F.value, ref_formula = parent_form)
    th_mass = molmass.Formula(best_form).isotope.mass
    error = abs(th_mass - mass)
    if verbose:
        print(f"observed mass {mass}")
        print(f"predicted mass {th_mass}")
        print(f"Error {error}")
    if error > tolerance_da:
        return(None, None, None)
    elif du_min != None:
        val_table = {}
        for e in ["H", "F", "Cl", "Br", "I", "Li", "Na", "K", "Rb", "Cs"]:
            val_table[e] = 1
        for e in ["O", "S", "Se", "Be", "Mg", "Ca", "Sr", "Ba"]:
            val_table[e] = 2
        for e in ["N", "P", "B", "As", "Sb"]:
            val_table[e] = 3
        for e in ["C", "Si", "Ge", "Sn"]:
            val_table[e] = 4

        du_form = 0
        for E, nE, _, _ in molmass.Formula(best_form).composition():
            if E in val_table.keys():
                vE = val_table[E]
            else:
                vE = 2
            du_form += nE * (vE - 2)
        du_form = 1 + ( 0.5 * du_form )
        if du_form < du_min:
            return(None, None, None)
        else:
            return(best_form, th_mass, error)    
    else:
        return(best_form, th_mass, error)

 
def generate_all_forms(parent_form):
    l1 = [[atom for x in range(atom_num)] for atom, atom_num, _, _ in molmass.Formula(parent_form).composition().astuple()]
    l2 = list(np.concatenate(l1). flat)

    m_forms_tuples = (x for l in range(1, len(l2) + 1) for x in more_itertools.distinct_combinations(l2, l))
    m_forms_tuples_converted = (tuple((molmass.Formula(x).formula for x in t)) for t in m_forms_tuples)
    m_forms = (molmass.Formula("".join(t)) for t in m_forms_tuples_converted)
    
    out_list = [(m.formula, m.isotope.mass) for m in m_forms]
    return(out_list)


def find_best_forms(mass, all_forms, tolerance_da = 0.005, charge = 0, verbose = False, du_min = None):
    if charge != 0:
        mass = uncharged_mass(mass, charge)
    found_hits = [(x[0], x[1], abs(x[1] - mass)) for x in all_forms if abs(x[1] - mass) <= tolerance_da]
    output = found_hits
    if len(output) == 0:
        output = [(None, None, None)]
    else:
        output = sorted(output, key = lambda tup : tup[-1])

    if du_min != None:
        output_filtered = []
        val_table = {}
        for e in ["H", "F", "Cl", "Br", "I", "Li", "Na", "K", "Rb", "Cs"]:
            val_table[e] = 1
        for e in ["O", "S", "Se", "Be", "Mg", "Ca", "Sr", "Ba"]:
            val_table[e] = 2
        for e in ["N", "P", "B", "As", "Sb"]:
            val_table[e] = 3
        for e in ["C", "Si", "Ge", "Sn"]:
            val_table[e] = 4
        for form, th_mass, error in output:
            if form == None:
                continue
            du_form = 0
            for E, nE, _, _ in molmass.Formula(form).composition().astuple():
                E = re.sub(r'\d', '', E) # strip the atom of the isotope label if it has one
                if E in val_table.keys():
                    vE = val_table[E]
                else:
                    vE = 2
                du_form += nE * (vE - 2)
            du_form = 1 + ( 0.5 * du_form )
            if du_form >= du_min:
                output_filtered.append((form, th_mass, error))
        if len(output_filtered) == 0:
            output_filtered = [(None, None, None)]
        output = output_filtered
    best_forms = [x[0] for x in output]
    th_masses = [x[1] for x in output]
    errors = [x[2] for x in output]
    if charge != 0:
        th_masses = [charged_mass(m, charge) if m else None for m in th_masses]
    output = (best_forms, th_masses, errors)
    return(output)
