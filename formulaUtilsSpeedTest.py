import sys
import os
import numpy as np
import molmass
import matplotlib.pyplot as plt
import cvxpy as cp
import more_itertools
import re
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

def findBestForm(mass, parentForm, toleranceDa = 0.005, charge = 0, verbose = False, cvxy_verbose = False, solver = 'SCIP', DuMin = None):
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
    elif DuMin != None:
        valTable = {}
        for e in ["H", "F", "Cl", "Br", "I", "Li", "Na", "K", "Rb", "Cs"]:
            valTable[e] = 1
        for e in ["O", "S", "Se", "Be", "Mg", "Ca", "Sr", "Ba"]:
            valTable[e] = 2
        for e in ["N", "P", "B", "As", "Sb"]:
            valTable[e] = 3
        for e in ["C", "Si", "Ge", "Sn"]:
            valTable[e] = 4

        DuForm = 0
        for E, nE, _, _ in molmass.Formula(bestForm).composition():
            if E in valTable.keys():
                vE = valTable[E]
            else:
                vE = 2
            DuForm += nE * (vE - 2)
        DuForm = 1 + ( 0.5 * DuForm )
        if DuForm < DuMin:
            return(None, None, None)
        else:
            return(bestForm, thMass, error)    
    else:
        return(bestForm, thMass, error)

 
def generateAllForms(parentForm):
    l1 = [[atom for x in range(atomNum)] for atom, atomNum, _, _ in molmass.Formula(parentForm).composition().astuple()]
    l2 = list(np.concatenate(l1). flat)

    mForms_tuples = (x for l in range(1, len(l2) + 1) for x in more_itertools.distinct_combinations(l2, l))
    mForms_tuples_converted = (tuple((molmass.Formula(x).formula for x in t)) for t in mForms_tuples)
    mForms = (molmass.Formula("".join(t)) for t in mForms_tuples_converted)
    # mForms = (molmass.Formula("".join(x)) for l in range(1, len(l2) + 1) for x in more_itertools.distinct_combinations(l2, l))
    
    outList = [(m.formula, m.isotope.mass) for m in mForms]
    return(outList)


def findBestForms(mass, allForms, toleranceDa = 0.005, charge = 0, verbose = False, DuMin = None):
    if charge != 0:
        mass = unchargedMass(mass, charge)
    # allForms = generateAllForms(parentForm)
    foundHits = [(x[0], x[1], abs(x[1] - mass)) for x in allForms if abs(x[1] - mass) <= toleranceDa]
    output = foundHits
    # foundHits = (molmass.Formula("".join(x)) for l in range(1, len(l2)) for x in more_itertools.distinct_combinations(l2, l) if abs(molmass.Formula("".join(x)).isotope.mass - mass) <= toleranceDa)
    # output = []
    # for bestForm in foundHits:
        # bestForm = x.formula
        # thMass = x.isotope.mass
        # error = abs(thMass - mass)
        # output.append((bestForm, thMass, error))
    if len(output) == 0:
        output = [(None, None, None)]
    else:
        output = sorted(output, key = lambda tup : tup[-1])

    if DuMin != None:
        outputFiltered = []
        valTable = {}
        for e in ["H", "F", "Cl", "Br", "I", "Li", "Na", "K", "Rb", "Cs"]:
            valTable[e] = 1
        for e in ["O", "S", "Se", "Be", "Mg", "Ca", "Sr", "Ba"]:
            valTable[e] = 2
        for e in ["N", "P", "B", "As", "Sb"]:
            valTable[e] = 3
        for e in ["C", "Si", "Ge", "Sn"]:
            valTable[e] = 4
        for form, thMass, error in output:
            if form == None:
                continue
            DuForm = 0
            for E, nE, _, _ in molmass.Formula(form).composition().astuple():
                E = re.sub(r'\d', '', E) # strip the atom of the isotope label if it has one
                if E in valTable.keys():
                    vE = valTable[E]
                else:
                    vE = 2
                DuForm += nE * (vE - 2)
            DuForm = 1 + ( 0.5 * DuForm )
            if DuForm >= DuMin:
                outputFiltered.append((form, thMass, error))
        if len(outputFiltered) == 0:
            outputFiltered = [(None, None, None)]
        output = outputFiltered
        output = ([x[0] for x in output], [x[1] for x in output], [x[2] for x in output])
    return(output)


 

