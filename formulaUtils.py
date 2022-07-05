import sys
import os
import numpy as np
import molmass
import matplotlib.pyplot as plt
import cvxpy as cp
# from pyscipopt.scip import Model 

def chargedMass(mass, charge):
    if type(mass) == str:
        mass = molmass.Formula(mass).isotope.mass
    em = (-1 * int(charge)) * molmass.ELECTRON.mass
    res = mass + em
    return(res)

def unchargedMass(mass, charge):
    if type(mass) == str:
        mass = molmass.Formula(mass).isotope.mass
    em = int(charge) * molmass.ELECTRON.mass
    res = mass + em
    return(res)

def adductMass(mass, adduct):
    adductAmends, adductCharge = parseAdduct(adduct)
    adductAtoms = ''.join([''.join([x[0] for i in range(x[1])]) for x in adductAmends])
    # adductCharge = np.sum([x[2] for x in adductAmends])
    # adductCharge = adductAmends[1]
    if adductAtoms != '':
        adductMass = formToMZ(adductAtoms, adductCharge)
    else:
        adductMass = 0.0
    res = chargedMass(mass, adductCharge)
    res = res + adductMass
    return(res)


def parseAdduct(adduct):
    # dLookup = {
    #     # return (Atom, amount, charge)
    #     '[M+H]+' : [('H', 1, 1)],
    #     '[M+Na]+' : [('Na', 1, 1)],
    #     '[M-H]-' : [('H', -1, -1)],
    # }
    dLookup = {
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
    if adduct not in dLookup.keys():
        sys.exit(f"Error - don't know how to parse adduct {(adduct)}")
    return(dLookup[adduct])

def toAdduct(form, adduct):
    fd = {x[0] : x[1] for x in molmass.Formula(form).composition()}
    adductAmends = parseAdduct(adduct)[0]
    for (e, addE) in adductAmends:
        if e not in fd.keys():
            fd[e] = 0
        fd[e] += addE
    if np.any([x < 0 for x in list(fd.values())]):
        # sys.exit(f"Error - created an impossible formula with {form} and adduct {adduct}")
        raise ValueError(f"Created an impossible formula with {form} and adduct {adduct}")
    res = molmass.Formula(''.join([f"{k}{v}" for k, v in fd.items() if v > 0])).formula
    return(res)

def fromAdduct(form, adduct):
    fd = {x[0] : x[1] for x in molmass.Formula(form).composition()}
    adductAmends = parseAdduct(adduct)[0]
    for (e, addE) in adductAmends:
        if e not in fd.keys():
            fd[e] = 0
        fd[e] -= addE
    if np.any([x < 0 for x in list(fd.values())]):
        sys.exit(f"Error - created an impossible formula with {form} and adduct {adduct}")
    res = molmass.Formula(''.join([f"{k}{v}" for k, v in fd.items() if v > 0])).formula
    return(res)

def formToMZ(form, charge, adduct = None):
    # if charge not in [-1, 1, 0]:
        # sys.exit(f'Error in formToMZ function call - charge ({charge}) must be either -1, 1, or 0')
    # if adduct not in [None, "[M+H]+", "[M-H]-", "[M+Na]+"]:
        # sys.exit(f'Error in formToMZ function call - adduct ({adduct}) must be in {list(dAdducts.keys())}')
    # if adduct != None:
        # form = toAdduct(form, adduct)
    if adduct != None:
        adductAmends, charge = parseAdduct(adduct)
        form = toAdduct(form, adduct)
    res = chargedMass(molmass.Formula(form).isotope.mass, charge = charge)
    res = abs(res / charge)
    return(res)

def getRefs(refFormula = "CHNOPSNa"):
    comp = molmass.Formula(refFormula).composition()
    refElements = [x[0] for x in comp]
    tMasses = np.array([molmass.Formula(x).isotope.mass for x in refElements])
    return((refElements, tMasses))

def formToVec(form, refFormula = "CHNOPSNa"):
    (refElements, tMasses) = getRefs(refFormula)
    fd = {x[0] : x[1] for x in molmass.Formula(form).composition()}
    for e in refElements:
        if e not in fd.keys():
            fd[e] = 0
    return(np.array([fd[k] for k in refElements]))

def vecToForm(vec, refFormula = "CHNOPSNa"):
    (refElements, tMasses) = getRefs(refFormula)
    vec = [round(x, 2) for x in vec]
    res = ""
    res = molmass.Formula("".join([f"{e}{int(vec[i_e])}" if vec[i_e] >= 0.99 else "" for i_e, e in enumerate(refElements)])).formula
    return(res)

def findBestForm(mass, parentForm, toleranceDa = 0.005, charge = 0, verbose = False, cvxy_verbose = False, solver = 'SCIP'):
    (refElements, tMasses) = getRefs(parentForm)
    if verbose:
        print(refElements)
        print(tMasses)
    if charge != 0:
        mass = unchargedMass(mass, charge)
    parentVec = formToVec(parentForm, refFormula=parentForm)
    if verbose:
        print(parentVec)
    F = cp.Variable(len(refElements), integer = True)
    objective = cp.Minimize(cp.sum_squares(F @ tMasses - mass))
    constraints = []
    for i in range(F.shape[0]):
        constraints.append(F[i] >= 0)
        constraints.append(F[i] <= parentVec[i])
        if verbose:
            print(f"0 <= {refElements[i]} <= {parentVec[i]}")
    prob = cp.Problem(objective, constraints)
    prob.solve(solver = solver, verbose = cvxy_verbose)
    if verbose:
        print("status:", prob.status)
        print("optimal value", prob.value)
        print("optimal var", F.value)
    bestForm = vecToForm(F.value, refFormula = parentForm)
    thMass = molmass.Formula(bestForm).isotope.mass
    error = abs(thMass - mass)
    if verbose:
        print(f"observed mass {mass}")
        print(f"predicted mass {thMass}")
        print(f"Error {error}")
    if error > toleranceDa:
        return(None, None, None)
    else:
        return(bestForm, thMass, error)

 