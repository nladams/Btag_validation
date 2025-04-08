import os
import sys
import uproot
import numpy as np
import glob
from ROOT import TH1F, TFile, TLorentzVector, TCanvas, TVector3
import ROOT
from array import array
import vector
# import correctionlib
import correctionlib._core as core
#sys.path.append('/cms/rand/NanoAODMaker/CMSSW_15_0_0_pre2/src/Structures')
import nanoAOD_Scouting_Data_Structure as ds


def getCorrFileName(year, era):

    correctionFile = f"{ds.data_dict[year][era]['jec'][0]}"
    correctionName = f"{ds.data_dict[year][era]['jec'][1]}"
    correctionSet = core.CorrectionSet.from_file(correctionFile)

    return correctionSet, correctionName


def makeJetEnergyCorrection(corr_set, corr_name, Jet_pt, Jet_eta, Jet_phi, Jet_mass, Jet_area):
    
    jec, compoundLevel, algo = (corr_name, 'L1L2L3Res', 'AK4PFPuppi')
    # jec, compoundLevel, algo = (corr_name, 'L1L2L3Res', 'AK4PFchs')

    sf = corr_set.compound['{}_{}_{}'.format(jec, compoundLevel, algo)]

    num_jets = Jet_pt.size
    corrected_jet_pt = np.zeros(num_jets)
    corrected_jet_mass = np.zeros(num_jets)

    for jet_idx in range(num_jets):
        corr = sf.evaluate(float(Jet_area[jet_idx]), float(Jet_eta[jet_idx]), float(Jet_phi[jet_idx]), float(Jet_pt[jet_idx]), float(Jet_pt[jet_idx]/Jet_area[jet_idx]))
        corrjetpt = Jet_pt[jet_idx]*corr
        corrjetmass = Jet_mass[jet_idx]*corr

        corrected_jet_pt[jet_idx] = corrjetpt
        corrected_jet_mass[jet_idx] = corrjetmass
    
    corr_Jet_pt = corrected_jet_pt
    corr_Jet_mass = corrected_jet_mass

    return corr_Jet_pt, corr_Jet_mass


def makeType1METCorrection(Jet_pt, Jet_eta, Jet_phi, Corrected_Jet_pt, Met_pt, Met_phi):
    raw_jet_sum = vector.obj(pt = 0.0, eta = 0.0, phi = 0.0)
    for j_pt, j_eta, j_phi in zip(Jet_pt, Jet_eta, Jet_phi):
        inter_vec = vector.obj(pt = j_pt, eta = j_eta, phi = j_phi)
        raw_jet_sum += inter_vec

    corr_jet_sum = vector.obj(pt = 0.0, eta = 0.0, phi = 0.0)
    for corr_j_pt, j_eta, j_phi in zip(Corrected_Jet_pt, Jet_eta, Jet_phi):
        inter_vec = vector.obj(pt = corr_j_pt, eta = j_eta, phi = j_phi)
        corr_jet_sum += inter_vec

    type_1_corr = raw_jet_sum - corr_jet_sum

    raw_met = vector.obj(rho = Met_pt, eta = 0, phi = Met_phi)

    corrected_met = raw_met + type_1_corr

    corr_MET_pt = corrected_met.rho
    corr_MET_phi = corrected_met.phi
    

    return corr_MET_pt, corr_MET_phi



if __name__ in "__main__":
    corr_set, corr_name = getCorrFileName('2024','F')
    print(corr_set, corr_name)
