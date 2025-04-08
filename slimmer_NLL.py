# python3 slimmer.py 0 False
#!/usr/bin/env python 
import os
import sys
import uproot
import glob
import numpy as np
import vector
import numba as nb
from numba import jit
from numba.types import bool_, int_, float_, float64
from array import array
from ROOT import TH1F, TFile, TLorentzVector, TCanvas, TVector3
import ROOT
from pathlib import Path
import shutil
import slimmer_functions as sf
from JERC.makeJEC_improved import makeJetEnergyCorrection
from JERC.makeJEC_improved import makeType1METCorrection
from JERC.makeJEC_improved import getCorrFileName
import JERC.nanoAOD_Scouting_Data_Structure as ds
from neutrinoMomentum import *

from itertools import permutations

''' Slimmer for negative Log method for b tagging efficiency
    "Identification of heavy-flavour jets with the CMS
    detector in pp collisions at 13 TeV"
    https://iopscience.iop.org/article/10.1088/1748-0221/13/05/P05011/pdf
    page 70

    there will 24 possible combinations per event. Each event will have a D_nu,min and a -log(lambda) which is the quantity that will be minimized. 

    -log(lambda) = -log(lambda_m) - log(lambda_nu)
    
    lambda_m is a 2D PDF that uses gen matched reconstructed W and Top distribution. I use RooHistPDF from a mass plot in backup_finalout.root. Histogram is not made here
    lambda_nu is a 1D PDF is same thing

'''

num = int(sys.argv[1])
condor = str(sys.argv[2])
inputfile = str(sys.argv[3])
MC = str(sys.argv[4])
# print(condor)

with open(inputfile, "r") as data_files:
    files = data_files.readlines()
    file = files[num]
    file = file.replace('\n', '')

path = file.rsplit('/', 1)[0]
file_name = file.rsplit('/', 1)[1]
print('filename: ', file_name)
path_to_file = file
print(path_to_file)

out_dir = f'{path}/slimmed'
Path(out_dir).mkdir(parents=True, exist_ok=True)
temp_outfile = f'./{file_name}'
perm_outfile = f'{out_dir}/{file_name}'

if condor == 'False':
    outfile = TFile(perm_outfile, 'RECREATE')
else:
    outfile = TFile(temp_outfile, 'RECREATE')

tree = ROOT.TTree('Events', 'Events')

# branch descriptions farther down

Jet_nConstituents = ROOT.std.vector('int')()
Jet_nCH = ROOT.std.vector('int')()
Jet_nNH = ROOT.std.vector('int')()
Jet_nElectrons = ROOT.std.vector('int')()
Jet_nMuons = ROOT.std.vector('int')()
Jet_nPhotons = ROOT.std.vector('int')()
Jet_Area = ROOT.std.vector('float')()
Jet_Eta = ROOT.std.vector('float')()
Jet_Mass = ROOT.std.vector('float')()
Jet_Phi = ROOT.std.vector('float')()
Jet_Pt = ROOT.std.vector('float')()
Jet_Bscore = ROOT.std.vector('float')()
Jet_ChEmEFrac = ROOT.std.vector('float')()
Jet_ChHEFrac = ROOT.std.vector('float')()
Jet_MuEFrac = ROOT.std.vector('float')()
Jet_NeEmEFrac = ROOT.std.vector('float')()
Jet_NeHEFrac = ROOT.std.vector('float')()

Muon_Pt = ROOT.std.vector('float')()
Muon_Eta = ROOT.std.vector('float')()
Muon_Phi = ROOT.std.vector('float')()
Muon_Mass = ROOT.std.vector('float')()
Muon_ecalIso = ROOT.std.vector('float')()
Muon_hcalIso = ROOT.std.vector('float')()
Muon_trackIso = ROOT.std.vector('float')()

Algo_Selected_Configuration_Code = array('i',[0])

Algo_Selected_NLL_tot = array('f',[0.0])
Algo_Selected_NLL_nu = array('f',[0.0])
Algo_Selected_NLL_m = array('f',[0.0])

Algo_b_l_cand_Pt = array('f',[0.0])
Algo_b_l_cand_Eta = array('f',[0.0])
Algo_b_l_cand_Phi = array('f',[0.0])
Algo_b_l_cand_Mass = array('f',[0.0])
Algo_b_l_cand_score = array('f',[0.0])

Algo_b_h_cand_Pt = array('f',[0.0])
Algo_b_h_cand_Eta = array('f',[0.0])
Algo_b_h_cand_Phi = array('f',[0.0])
Algo_b_h_cand_Mass = array('f',[0.0])
Algo_b_h_cand_score = array('f',[0.0])

Algo_q1_cand_Pt = array('f',[0.0])
Algo_q1_cand_Eta = array('f',[0.0])
Algo_q1_cand_Phi = array('f',[0.0])
Algo_q1_cand_Mass = array('f',[0.0])
Algo_q1_cand_score = array('f',[0.0])

Algo_q2_cand_Pt = array('f',[0.0])
Algo_q2_cand_Eta = array('f',[0.0])
Algo_q2_cand_Phi = array('f',[0.0])
Algo_q2_cand_Mass = array('f',[0.0])
Algo_q2_cand_score = array('f',[0.0])

Algo_nu_cand_Pt = array('f',[0.0])
Algo_nu_cand_Eta = array('f',[0.0])
Algo_nu_cand_Phi = array('f',[0.0])
Algo_nu_cand_Mass = array('f',[0.0])
Algo_nu_cand_D = array('f',[0.0])

Algo_Selected_Top_Mass = array('f',[0.0])
Algo_Selected_W_Mass = array('f',[0.0])

if MC == 'True':
    Algo_is_completely_correct = array('i',[0]) # 0 (no) 1(yes) -1 (doesnt exist)
    Algo_is_hadronic_correct = array('i',[0])
    Algo_is_leptonic_correct = array('i',[0])
    Algo_b_h_cand_score_gen_matched = array('f',[0.0])
    Algo_b_l_cand_score_gen_matched = array('f',[0.0])

Jet_N = array('i', [0])
Muon_N = array('i', [0])
HT = array('f', [0.0])
L1_SingleMu11_SQ14_BMTF = array('i', [0])
Event = array('i', [0])
Run = array('i', [0])
LuminosityBlock = array('i', [0])
MET_Pt = array('f', [0.0])
MET_Phi = array('f', [0.0])

# Jet_Pt and Jet_Mass branches are corrected
tree.Branch('Jet_nConstituents', Jet_nConstituents)
tree.Branch('Jet_nCH', Jet_nCH)
tree.Branch('Jet_nNH', Jet_nNH)
tree.Branch('Jet_nElectrons', Jet_nElectrons)
tree.Branch('Jet_nMuons', Jet_nMuons)
tree.Branch('Jet_nPhotons', Jet_nPhotons)
tree.Branch('Jet_Area', Jet_Area)
tree.Branch('Jet_Eta', Jet_Eta)
tree.Branch('Jet_Mass', Jet_Mass)
tree.Branch('Jet_Phi', Jet_Phi)
tree.Branch('Jet_Pt', Jet_Pt)
tree.Branch('Jet_Bscore', Jet_Bscore)
tree.Branch('Jet_ChEmEFrac', Jet_ChEmEFrac)
tree.Branch('Jet_ChHEFrac', Jet_ChHEFrac)
tree.Branch('Jet_MuEFrac', Jet_MuEFrac)
tree.Branch('Jet_NeEmEFrac', Jet_NeEmEFrac)
tree.Branch('Jet_NeHEFrac', Jet_NeHEFrac)

tree.Branch('Muon_Pt', Muon_Pt)
tree.Branch('Muon_Eta', Muon_Eta)
tree.Branch('Muon_Phi', Muon_Phi)
tree.Branch('Muon_Mass', Muon_Mass)
tree.Branch('Muon_ecalIso', Muon_ecalIso)
tree.Branch('Muon_hcalIso', Muon_hcalIso)
tree.Branch('Muon_trackIso', Muon_trackIso)

# Alog_* branches have best configuration in the event
tree.Branch('Algo_Selected_Configuration_Code', Algo_Selected_Configuration_Code, 'Algo_Selected_Configuration_Code/I') # code is four integers. (index of b_l + 1, index of b_h + 1, index of q1 + 1, index of q2 + 1)

tree.Branch('Algo_Selected_NLL_tot', Algo_Selected_NLL_tot, 'Algo_Selected_NLL_tot/F')
tree.Branch('Algo_Selected_NLL_nu', Algo_Selected_NLL_nu, 'Algo_Selected_NLL_nu/F')
tree.Branch('Algo_Selected_NLL_m', Algo_Selected_NLL_m, 'Algo_Selected_NLL_m/F')

# bjet leptonic side
tree.Branch('Algo_b_l_cand_Pt', Algo_b_l_cand_Pt, 'Algo_b_l_cand_Pt/F') 
tree.Branch('Algo_b_l_cand_Eta', Algo_b_l_cand_Eta, 'Algo_b_l_cand_Eta/F')
tree.Branch('Algo_b_l_cand_Phi', Algo_b_l_cand_Phi, 'Algo_b_l_cand_Phi/F')
tree.Branch('Algo_b_l_cand_Mass', Algo_b_l_cand_Mass, 'Algo_b_l_cand_Mass/F')
tree.Branch('Algo_b_l_cand_score', Algo_b_l_cand_score, 'Algo_b_l_cand_score/F')

# bjet hadronic side
tree.Branch('Algo_b_h_cand_Pt', Algo_b_h_cand_Pt, 'Algo_b_h_cand_Pt/F') 
tree.Branch('Algo_b_h_cand_Eta', Algo_b_h_cand_Eta, 'Algo_b_h_cand_Eta/F')
tree.Branch('Algo_b_h_cand_Phi', Algo_b_h_cand_Phi, 'Algo_b_h_cand_Phi/F')
tree.Branch('Algo_b_h_cand_Mass', Algo_b_h_cand_Mass, 'Algo_b_h_cand_Mass/F')
tree.Branch('Algo_b_h_cand_score', Algo_b_h_cand_score, 'Algo_b_h_cand_score/F')

# quark jet from W   
tree.Branch('Algo_q1_cand_Pt', Algo_q1_cand_Pt, 'Algo_q1_cand_Pt/F')
tree.Branch('Algo_q1_cand_Eta', Algo_q1_cand_Eta, 'Algo_q1_cand_Eta/F')
tree.Branch('Algo_q1_cand_Phi', Algo_q1_cand_Phi, 'Algo_q1_cand_Phi/F')
tree.Branch('Algo_q1_cand_Mass', Algo_q1_cand_Mass, 'Algo_q1_cand_Mass/F')
tree.Branch('Algo_q1_cand_score', Algo_q1_cand_score, 'Algo_q1_cand_score/F')

# quark jet from W
tree.Branch('Algo_q2_cand_Pt', Algo_q2_cand_Pt, 'Algo_q2_cand_Pt/F')
tree.Branch('Algo_q2_cand_Eta' , Algo_q2_cand_Eta, 'Algo_q2_cand_Eta/F')
tree.Branch('Algo_q2_cand_Phi', Algo_q2_cand_Eta, 'Algo_q2_cand_Eta/F')
tree.Branch('Algo_q2_cand_Mass', Algo_q2_cand_Mass, 'Algo_q2_cand_Mass/F')
tree.Branch('Algo_q2_cand_score', Algo_q2_cand_score, 'Algo_q2_cand_score/F')

# neutrino
tree.Branch('Algo_nu_cand_Pt', Algo_nu_cand_Pt, 'Algo_nu_cand_Pt/F')
tree.Branch('Algo_nu_cand_Eta', Algo_nu_cand_Eta, 'Algo_nu_cand_Eta/F')
tree.Branch('Algo_nu_cand_Phi', Algo_nu_cand_Phi, 'Algo_nu_cand_Phi/F')
tree.Branch('Algo_nu_cand_Mass', Algo_nu_cand_Mass, 'Algo_nu_cand_Mass/F')
tree.Branch('Algo_nu_cand_D', Algo_nu_cand_D, 'Algo_nu_cand_D/F')

tree.Branch('Algo_Selected_Top_Mass', Algo_Selected_Top_Mass, 'Algo_Selected_Top_Mass/F')
tree.Branch('Algo_Selected_W_Mass', Algo_Selected_W_Mass, 'Algo_Selected_W_Mass/F')

# each HLT jet will have a entry in Jet_Gen_Match_Status
# # 5 = B1 = b from top
# -5 = B2 = bbar from tbar
# 1 = Q1 = quark from W 
# -1 = Q2 = antiquark from W
# 99 = no match
if MC == 'True':
    tree.Branch('Algo_is_completely_correct', Algo_is_completely_correct, 'Algo_is_completely_correct/I')
    tree.Branch('Algo_is_hadronic_correct', Algo_is_hadronic_correct, 'Algo_is_hadronic_correct/I')
    tree.Branch('Algo_is_leptonic_correct', Algo_is_leptonic_correct, 'Algo_is_leptonic_correct/I')
    tree.Branch('Algo_b_h_cand_score_gen_matched', Algo_b_h_cand_score_gen_matched, 'Algo_b_h_cand_score_gen_matched/F')
    tree.Branch('Algo_b_l_cand_score_gen_matched', Algo_b_l_cand_score_gen_matched, 'Algo_b_l_cand_score_gen_matched/F')

# event variables
tree.Branch('Jet_N', Jet_N, 'Jet_N/I')
tree.Branch('Muon_N', Muon_N, 'Muon_N/I')
tree.Branch('HT', HT, 'HT/F')
tree.Branch('L1_SingleMu11_SQ14_BMTF', L1_SingleMu11_SQ14_BMTF, 'L1_SingleMu11_SQ14_BMTF/I')
tree.Branch('Event', Event, 'Event/I')
tree.Branch('Run', Run, 'Run/I')
tree.Branch('LuminosityBlock', LuminosityBlock, 'LuminosityBlock/I')
tree.Branch('MET_Pt', MET_Pt, 'MET_Pt/F')
tree.Branch('MET_Phi', MET_Phi, 'MET_Phi/F')


with uproot.open(f'{path}/{file_name}')['Events'] as data:
    jet_nconstituents_raw = data['ScoutingPFJetRecluster_nConstituents'].array(library='np')
    jet_nch_raw = data['ScoutingPFJetRecluster_nCh'].array(library='np')
    jet_nnh_raw = data['ScoutingPFJetRecluster_nNh'].array(library='np')
    jet_nelectrons_raw = data['ScoutingPFJetRecluster_nElectrons'].array(library='np')
    jet_nmuons_raw = data['ScoutingPFJetRecluster_nMuons'].array(library='np')
    jet_nphotons_raw = data['ScoutingPFJetRecluster_nPhotons'].array(library='np')
    jet_area_raw = data['ScoutingPFJetRecluster_area'].array(library='np')
    jet_eta_raw = data['ScoutingPFJetRecluster_eta'].array(library='np')
    jet_mass_raw = data['ScoutingPFJetRecluster_mass'].array(library='np')
    jet_phi_raw = data['ScoutingPFJetRecluster_phi'].array(library='np')
    jet_pt_raw = data['ScoutingPFJetRecluster_pt'].array(library='np')
    jet_bscore_raw = data['ScoutingPFJetRecluster_ak4ScoutingRun3_btagged'].array(library='np')
    jet_chemefrac_raw = data['ScoutingPFJetRecluster_chEmEF'].array(library='np')
    jet_chhefrac_raw = data['ScoutingPFJetRecluster_chHEF'].array(library='np')
    jet_muefrac_raw = data['ScoutingPFJetRecluster_muEF'].array(library='np')
    jet_neemefrac_raw = data['ScoutingPFJetRecluster_neEmEF'].array(library='np')
    jet_nehefrac_raw = data['ScoutingPFJetRecluster_neHEF'].array(library='np')

    muon_pt_raw = data['ScoutingMuonVtx_pt'].array(library='np')
    muon_eta_raw = data['ScoutingMuonVtx_eta'].array(library='np')
    muon_phi_raw = data['ScoutingMuonVtx_phi'].array(library='np')
    muon_mass_raw = data['ScoutingMuonVtx_m'].array(library='np')
    muon_ecaliso_raw = data['ScoutingMuonVtx_ecalIso'].array(library='np')
    muon_hcaliso_raw = data['ScoutingMuonVtx_hcalIso'].array(library='np')
    muon_trackiso_raw = data['ScoutingMuonVtx_trackIso'].array(library='np')

    met_pt_raw = data['ScoutingMET_pt'].array(library='np')
    met_phi_raw = data['ScoutingMET_phi'].array(library='np')
    jet_n_raw = data['nScoutingPFJetRecluster'].array(library='np')
    muon_n_raw = data['nScoutingMuonVtx'].array(library='np')
    l1_singlemu11_sq14_bmtf_raw = data['L1_SingleMu11_SQ14_BMTF'].array(library='np')
    event_raw = data['event'].array(library='np')
    run_raw = data['run'].array(library='np')
    luminosityblock_raw = data['luminosityBlock'].array(library='np')

    if MC == 'True': 
        ngenpart_raw = data['nGenPart'].array(library='np')
        genPart_pdgId_raw = data['GenPart_pdgId'].array(library='np')
        genPart_status_raw = data['GenPart_status'].array(library='np')
        genPart_mother_idx_raw =data['GenPart_genPartIdxMother'].array(library='np')
        genpart_phi_raw = data['GenPart_phi'].array(library='np')
        genpart_eta_raw = data['GenPart_eta'].array(library='np')
        genpart_pt_raw = data['GenPart_pt'].array(library='np')

########################
#                      #
#   BEGIN EVENT LOOP   #
#                      #
########################

# generate combos
raw_idx_list = list(permutations([0, 1, 2, 3])) # all possible combinations of 4 where order matters (24 of them)
idx_list = [comb for comb in raw_idx_list if comb[-2] < comb[-1] ] # remove combos where last 2 indexes are the same but flipped (order of W jets dont matter)

nevents = event_raw.size

tot_count = 0

tot_correct_count = 0
incorrect_count = 0

had_correct_count = 0
lep_correct_count = 0


year = ''
era = ''

if MC == 'True':
    year = 'MC2024'
    era = 'TT'
if MC == 'False':
    year =  '2024'
    era = 'F'

corr_set, corr_name = getCorrFileName(year, era)

for evt in range(nevents):

    ########################
    # Trigger requirements #
    ########################
    # if l1_singlemu11_sq14_bmtf_raw[evt] < 1:
    #     continue
    
    #########################
    # Jet-level + Muon cuts #
    #########################
    # jet_number_cut = jet_n_raw[evt] > 2
    constituents_cut = jet_nconstituents_raw[evt] <= 50
    # jet_n_cuts = np.logical_and.reduce((jet_number_cut, constituents_cut))
    # print("jet_pt_raw[evt]: ", jet_pt_raw[evt])
    # print("jet_nconstituents_raw[evt]: ", jet_nconstituents_raw[evt])

    nehefrac_cut = jet_nehefrac_raw[evt] < 0.9
    chhefrac_cut = jet_chhefrac_raw[evt] > 0.01
    muefrac_cut = jet_muefrac_raw[evt] < 0.8
    chemefrac_cut = jet_chemefrac_raw[evt] < 0.9
    neemefrac_cut = jet_neemefrac_raw[evt] < 0.9
    quality_cuts = np.logical_and.reduce((nehefrac_cut, chhefrac_cut, muefrac_cut, chemefrac_cut, neemefrac_cut))

    muon_pt_cut = muon_pt_raw[evt] > 25
    muon_eta_cut = np.absolute(muon_eta_raw[evt]) < 2.4

    muon_cuts = np.logical_and.reduce((muon_pt_cut, muon_eta_cut))

    pt_cut = jet_pt_raw[evt] > 20
    eta_cut = np.absolute(jet_eta_raw[evt]) < 2.4
    kinematic_cuts = np.logical_and.reduce((pt_cut, eta_cut))

    total_cut = np.logical_and.reduce((constituents_cut, quality_cuts, kinematic_cuts))
    cut_for_HT = np.logical_and.reduce((quality_cuts, kinematic_cuts))

    jet_nconstituents = jet_nconstituents_raw[evt][total_cut]
    jet_nch = jet_nch_raw[evt][total_cut]
    jet_nnh = jet_nnh_raw[evt][total_cut]
    jet_nelectrons = jet_nelectrons_raw[evt][total_cut]
    jet_nmuons = jet_nmuons_raw[evt][total_cut]
    jet_nphotons = jet_nphotons_raw[evt][total_cut]
    jet_area = jet_area_raw[evt][total_cut]
    jet_eta = jet_eta_raw[evt][total_cut]
    jet_mass = jet_mass_raw[evt][total_cut]
    jet_phi = jet_phi_raw[evt][total_cut]
    jet_pt = jet_pt_raw[evt][total_cut]
    jet_bscore = jet_bscore_raw[evt][total_cut]
    jet_chemefrac = jet_chemefrac_raw[evt][total_cut]
    jet_chhefrac = jet_chhefrac_raw[evt][total_cut]
    jet_muefrac = jet_muefrac_raw[evt][total_cut]
    jet_neemefrac = jet_neemefrac_raw[evt][total_cut]
    jet_nehefrac = jet_nehefrac_raw[evt][total_cut]

    muon_pt = muon_pt_raw[evt][muon_cuts]
    muon_eta = muon_eta_raw[evt][muon_cuts]
    muon_phi = muon_phi_raw[evt][muon_cuts]
    muon_mass = muon_mass_raw[evt][muon_cuts]
    muon_ecaliso = muon_ecaliso_raw[evt][muon_cuts]
    muon_hcaliso = muon_hcaliso_raw[evt][muon_cuts]
    muon_trackiso = muon_trackiso_raw[evt][muon_cuts]

    ht = jet_pt_raw[evt][cut_for_HT].sum()
    # if ht < 360.:
    #     continue
    
    ####################
    # Event-level cuts #
    ####################

    njets = np.sum(total_cut)
    nmuons = np.sum(muon_cuts)

    if njets != 4 or nmuons != 1:
        Jet_nConstituents.clear()
        Jet_nCH.clear()
        Jet_nNH.clear()
        Jet_nElectrons.clear()
        Jet_nMuons.clear()
        Jet_nPhotons.clear()
        Jet_Area.clear()
        Jet_Eta.clear()
        Jet_Mass.clear()
        Jet_Phi.clear()
        Jet_Pt.clear()
        Jet_Bscore.clear()
        Jet_ChEmEFrac.clear()
        Jet_ChHEFrac.clear()
        Jet_MuEFrac.clear()
        Jet_NeEmEFrac.clear()
        Jet_NeHEFrac.clear()

        Muon_Pt.clear()
        Muon_Eta.clear()
        Muon_Phi.clear()
        Muon_Mass.clear()
        Muon_ecalIso.clear()
        Muon_hcalIso.clear()
        Muon_trackIso.clear()

        continue

    ###################
    # jet corrections #
    ###################
    
    jet_pt_corr, jet_mass_corr = makeJetEnergyCorrection(corr_set, corr_name, jet_pt, jet_phi, jet_eta, jet_mass, jet_area)
    #print("corrected pt, mass", jet_pt_corr, jet_mass_corr)
    MET_Pt_corr, MET_Phi_corr = makeType1METCorrection(jet_pt, jet_eta, jet_phi, jet_pt_corr, met_pt_raw[evt], met_phi_raw[evt])
    #print("corrected MET pt, phi", MET_Pt_corr, MET_Phi_corr)
    print("")

    #################
    # gen matching  #
    #################

    if MC == "True":
        # each HLT jet will have a entry in Jet_Gen_Match_Status
        # # 5 = B1 = b from top
        # -5 = B2 = bbar from tbar
        # 1 = Q1 = quark from W 
        # -1 = Q2 = antiquark from W
        # 99 = no match 
        Jet_Gen_Match_Status = np.zeros(njets)
        Muon_Is_Gen_Matched =  np.zeros(nmuons) # one entry per muon, 1 if gen matched to a muon, -1 if gen match to anti muon, 0 otherwise

        [B1_gen_vec, B2_gen_vec, Q1_gen_vec, Q2_gen_vec, LEP_gen_vec, gen_key] = sf.genMatcher(ngenpart_raw[evt], genPart_pdgId_raw[evt], genPart_status_raw[evt], genPart_mother_idx_raw[evt], genpart_phi_raw[evt], genpart_eta_raw[evt], genpart_pt_raw[evt])
        
        #print("gen 4 vectors eta phi:")
        #print("b", B1_vec.Eta(), B1_vec.Phi())
        #print("bbar", B2_vec.Eta(), B2_vec.Phi())
        #print("quark", Q1_vec.Eta(), Q1_vec.Phi())
        #print("antiquark", Q2_vec.Eta(), Q2_vec.Phi())
        #print("lepton", LEP_vec.Eta(), LEP_vec.Phi())

        for i in range(njets):
            temp = ROOT.TLorentzVector()
            temp.SetPtEtaPhiM(jet_pt_corr[i], jet_eta[i], jet_phi[i], jet_mass_corr[i])
            if sf.genMatchDeltaR(B1_gen_vec, temp, 0.1):
                #print("jet", i, "matched to B1")
                Jet_Gen_Match_Status[i] = 5
                continue
            if sf.genMatchDeltaR(B2_gen_vec, temp, 0.1):
                #print("jet", i, "matched to B2")
                Jet_Gen_Match_Status[i] = -5
                continue
            if sf.genMatchDeltaR(Q1_gen_vec, temp, 0.1):
                #print("jet", i, "matched to Q1")
                Jet_Gen_Match_Status[i] = 1
                continue
            if sf.genMatchDeltaR(Q2_gen_vec, temp, 0.1):
                #print("jet", i, "matched to Q2")
                Jet_Gen_Match_Status[i] = -1
                continue
            else:
                Jet_Gen_Match_Status[i] = 99

        for j in range(nmuons):
            temp = ROOT.TLorentzVector()
            temp.SetPtEtaPhiM(muon_pt[j], muon_eta[j], muon_phi[j], muon_mass[j])
            #print("is muon gen matched?",sf.genMatchDeltaR(LEP_gen_vec, temp, 0.1))
            #print("key:", gen_key)
            if gen_key == 1: 
                Muon_Is_Gen_Matched[j] = 1*sf.genMatchDeltaR(LEP_gen_vec, temp, 0.1)
                #print("pushed back", sf.genMatchDeltaR(LEP_gen_vec, temp, 0.1))
            if gen_key == 2: #anti muon
                Muon_Is_Gen_Matched[j] = -1 * sf.genMatchDeltaR(LEP_gen_vec, temp, 0.1)
                #print("pushed back", -1 * sf.genMatchDeltaR(LEP_gen_vec, temp, 0.1))
            if gen_key == 0:
                Muon_Is_Gen_Matched[j] = 0 # key = 0 means there is no gen muon
                #print("pushed back", 0)

        gen_had_W_mass, gen_had_Top_mass, correct_q1_idx, correct_q2_idx, correct_b_h_idx, correct_b_l_idx, muon_key = sf.GenMatchAllFour(jet_pt_corr, jet_eta, jet_phi, jet_mass_corr, Jet_Gen_Match_Status, Muon_Is_Gen_Matched)
        if gen_had_W_mass != 99:
            correct_configuration = [correct_b_l_idx, correct_b_h_idx, correct_q1_idx, correct_q2_idx]
            print("correct code:", correct_configuration)

    print("jet pt", jet_pt_corr)
    print("jet eta", jet_eta)
    print("jet phi", jet_phi)
    print("jet mass", jet_mass_corr)
    print("jet scores", jet_bscore)
    print("muon pt, eta, phi, mass", muon_pt, muon_eta, muon_phi, muon_mass)
    print("MET pt, phi", MET_Pt_corr, MET_Phi_corr)
    
    ####################################
    # fill standard jet + muon branches #
    ###################################
    for jet in range(njets):
        Jet_nConstituents.push_back(int(jet_nconstituents[jet]))
        Jet_nCH.push_back(int(jet_nch[jet]))
        Jet_nNH.push_back(int(jet_nnh[jet]))
        Jet_nElectrons.push_back(int(jet_nelectrons[jet]))
        Jet_nMuons.push_back(int(jet_nmuons[jet]))
        Jet_nPhotons.push_back(int(jet_nphotons[jet]))
        Jet_Area.push_back(jet_area[jet])
        Jet_Eta.push_back(jet_eta[jet])
        Jet_Mass.push_back(jet_mass_corr[jet])
        Jet_Phi.push_back(jet_phi[jet])
        Jet_Pt.push_back(jet_pt_corr[jet]) 
        Jet_Bscore.push_back(jet_bscore[jet])
        Jet_ChEmEFrac.push_back(jet_chemefrac[jet])
        Jet_ChHEFrac.push_back(jet_chhefrac[jet])
        Jet_MuEFrac.push_back(jet_muefrac[jet])
        Jet_NeEmEFrac.push_back(jet_neemefrac[jet])
        Jet_NeHEFrac.push_back(jet_nehefrac[jet])

    for muon in range(nmuons):
        Muon_Pt.push_back(muon_pt[muon])
        Muon_Eta.push_back(muon_eta[muon])
        Muon_Phi.push_back(muon_phi[muon])
        Muon_Mass.push_back(muon_mass[muon])
        Muon_ecalIso.push_back(muon_ecaliso[muon])
        Muon_hcalIso.push_back(muon_hcaliso[muon])
        Muon_trackIso.push_back(muon_trackiso[muon])

    ############################
    # fill standard event vars #
    ############################
    HT[0] = ht
    L1_SingleMu11_SQ14_BMTF[0] = int(l1_singlemu11_sq14_bmtf_raw[evt])
    Event[0] = int(event_raw[evt])
    Run[0] = int(run_raw[evt])
    LuminosityBlock[0] = int(luminosityblock_raw[evt])
    Jet_N[0] = njets
    Muon_N[0] = nmuons
    MET_Pt[0] = MET_Pt_corr
    MET_Phi[0] = MET_Phi_corr

    ###################
    # algorithm start #
    ###################

    Muon_vec = TLorentzVector()
    MET_vec = TLorentzVector()

    Muon_vec.SetPtEtaPhiM(muon_pt[0], muon_eta[0], muon_phi[0], muon_mass[0])
    MET_vec.SetPtEtaPhiM(MET_Pt_corr, 0, MET_Phi_corr, 0)

    #print("code format: (B_Lep, B_Had, Q1, Q2)")
    # 24 quadruplet combos, but order of Q1 and Q2 doesnt matter, so theres actually  12 combinations

    # empty arrays for storing variables of interest during the loop. lowest NLL_tot is chosen and corresponding entry in all below matrices are pushed to their branch
    code_arr = np.zeros(len(idx_list))

    NLL_tot_arr = np.zeros(len(idx_list))
    NLL_had_arr = np.zeros(len(idx_list))
    NLL_lep_arr = np.zeros(len(idx_list))

    W_mass_arr = np.zeros(len(idx_list))
    Top_mass_arr = np.zeros(len(idx_list))

    b_l_pt_arr = np.zeros(len(idx_list))
    b_l_eta_arr  = np.zeros(len(idx_list))
    b_l_phi_arr = np.zeros(len(idx_list))
    b_l_mass_arr = np.zeros(len(idx_list))
    b_l_score_arr = np.zeros(len(idx_list))

    b_h_pt_arr = np.zeros(len(idx_list))
    b_h_eta_arr = np.zeros(len(idx_list))
    b_h_phi_arr = np.zeros(len(idx_list))
    b_h_mass_arr = np.zeros(len(idx_list))
    b_h_score_arr = np.zeros(len(idx_list))

    q1_pt_arr  = np.zeros(len(idx_list))
    q1_eta_arr = np.zeros(len(idx_list))
    q1_phi_arr = np.zeros(len(idx_list))
    q1_mass_arr = np.zeros(len(idx_list))
    q1_score_arr = np.zeros(len(idx_list))

    q2_pt_arr  = np.zeros(len(idx_list))
    q2_eta_arr = np.zeros(len(idx_list))
    q2_phi_arr = np.zeros(len(idx_list))
    q2_mass_arr = np.zeros(len(idx_list))
    q2_score_arr = np.zeros(len(idx_list))

    nu_pt_arr = np.zeros(len(idx_list))
    nu_eta_arr = np.zeros(len(idx_list))
    nu_phi_arr = np.zeros(len(idx_list))
    nu_mass_arr = np.zeros(len(idx_list))
    D_nu_arr = np.zeros(len(idx_list))

    ####################
    # combination loop #
    ####################
    c = 0
    for comb in idx_list:
        code = int(str(comb[0] + 1) + str(comb[1] + 1) + str(comb[2] + 1) + str(comb[3]+ 1)) # when saving the a branch, need to add 1 because leading zeros get dropped
        print("comb", comb)
        B_L_vec = TLorentzVector() # leptonic b cand
        B_H_vec = TLorentzVector() # hadronic b cand
        Q1_vec = TLorentzVector() # quark cand
        Q2_vec = TLorentzVector() # antiquark cand
        nu_vec = TLorentzVector() # neutrino

        B_L_vec.SetPtEtaPhiM(jet_pt_corr[comb[0]], jet_eta[comb[0]], jet_phi[comb[0]], jet_mass_corr[comb[0]])
        B_H_vec.SetPtEtaPhiM(jet_pt_corr[comb[1]], jet_eta[comb[1]], jet_phi[comb[1]], jet_mass_corr[comb[1]])   
        Q1_vec.SetPtEtaPhiM(jet_pt_corr[comb[2]], jet_eta[comb[2]], jet_phi[comb[2]], jet_mass_corr[comb[2]]) 
        Q2_vec.SetPtEtaPhiM(jet_pt_corr[comb[3]], jet_eta[comb[3]], jet_phi[comb[3]], jet_mass_corr[comb[3]])   

        B_L_score = jet_bscore[comb[0]]
        B_H_score = jet_bscore[comb[1]]
        Q1_score = jet_bscore[comb[2]]
        Q2_score = jet_bscore[comb[3]]

        #print("--->BL", B_L_vec.Pt(), B_L_vec.Eta(), B_L_vec.Phi(), B_L_vec.M(), B_L_score)    
        #print("---->BH", B_H_vec.Pt(), B_H_vec.Eta(), B_H_vec.Phi(), B_H_vec.M(), B_H_score)    
        #print("---->Q1", Q1_vec.Pt(), Q1_vec.Eta(), Q1_vec.Phi(), Q1_vec.M(), Q1_score)    
        #print("---->Q2", Q2_vec.Pt(), Q2_vec.Eta(), Q2_vec.Phi(), Q2_vec.M(), Q2_score)    
    
        ##################
        # leptonic side #
        ##################
        try:
            sol = singleNeutrinoSolution(B_L_vec, Muon_vec, MET_vec.Px(), MET_vec.Py(), sigma2=[[0.001,0],[0,0.001]]) # sigma2 is uncertainty matrix, just made it small numbers
            #print("    neutrino solution", sol.nu, "chi2", sol.chi2)
            nu_px, nu_py, nu_pz = sol.nu[0], sol.nu[1], sol.nu[2]
            nu_E = np.sqrt(nu_px**2 + nu_py**2 + nu_pz**2)
            nu_vec.SetPxPyPzE(nu_px, nu_py, nu_pz, nu_E)
            D_nu = np.sqrt( (nu_vec.Px() - MET_vec.Px())**2 + (nu_vec.Py() - MET_vec.Py())**2 ) 
        
        except (IndexError, ZeroDivisionError): # no solution usually when muon + bjet mass > 172, so theres no way to make a top
            print("    no solution for neutrino")
            continue

        #print("    neutrino pt eta phi m ", nu_vec.Pt(), nu_vec.Eta(), nu_vec.Phi(), nu_vec.M())
        #print("    D_nu", D_nu)

        prob_D_nu = sf.getPdfVal1D('backup_finalout.root', 'Correct_D_nu_for_pdf', 0, 150, D_nu)
        NLL_lep = -np.log(prob_D_nu)
        #print('    prob d nu',prob_D_nu, 'NLL', NLL_lep)
    
       ##################
       # hadronic side  #
       ##################
        Had_Top = B_H_vec + Q1_vec + Q2_vec
        Had_W = Q1_vec + Q2_vec

        Had_Top_Mass = Had_Top.M()
        Had_W_Mass = Had_W.M()

        #print("    top mass",Had_Top_Mass, "W mass", Had_W_Mass)

        prob_Had = sf.getPdfVal2D('backup_finalout.root', 'Gen_Had_W_vs_Had_Top_for_pdf', 0, 500, 0, 500, Had_W_Mass, Had_Top_Mass)

        if prob_Had == 0:
            print("    zero probability for hadronic side")
            continue

        NLL_had = -np.log(prob_Had)

        #print('    prob_had', prob_Had, 'NLL', NLL_had)    

        NLL_tot = NLL_had + NLL_lep

        #print("    NLL total", NLL_tot)

        code_arr[c] = int(code)

        NLL_tot_arr[c] = NLL_tot
        NLL_had_arr[c] = NLL_had
        NLL_lep_arr[c] = NLL_lep

        W_mass_arr[c] = Had_W_Mass
        Top_mass_arr[c] = Had_Top_Mass

        b_l_pt_arr[c] = B_L_vec.Pt()
        b_l_eta_arr[c] = B_L_vec.Eta()
        b_l_phi_arr[c] = B_L_vec.Phi()
        b_l_mass_arr[c] = B_L_vec.M()
        b_l_score_arr[c] = B_L_score

        b_h_pt_arr[c] = B_H_vec.Pt()
        b_h_eta_arr[c] = B_H_vec.Eta()
        b_h_phi_arr[c] = B_H_vec.Phi()
        b_h_mass_arr[c] = B_H_vec.M()
        b_h_score_arr[c] = B_H_score

        q1_pt_arr[c] = Q1_vec.Pt()
        q1_eta_arr[c] = Q1_vec.Eta()
        q1_phi_arr[c] = Q1_vec.Phi()
        q1_mass_arr[c] = Q1_vec.M()
        q1_score_arr[c] = Q1_score

        q2_pt_arr[c] = Q2_vec.Pt()
        q2_eta_arr[c] = Q2_vec.Eta()
        q2_phi_arr[c] = Q2_vec.Phi()
        q2_mass_arr[c] = Q2_vec.M()
        q2_score_arr[c] = Q2_score

        nu_pt_arr[c] = nu_vec.Pt()
        nu_eta_arr[c] = nu_vec.Eta()
        nu_phi_arr[c] = nu_vec.Phi()
        nu_mass_arr[c] = nu_vec.M()
        D_nu_arr[c] = D_nu

        c+=1

    if not np.any(NLL_tot_arr[NLL_tot_arr != 0]):
        print("no valid configurations")
        continue

    #############
    # selection #
    #############
    if np.any(NLL_tot_arr):
        print("all (nonzero) lambdas", NLL_tot_arr[NLL_tot_arr != 0])
        golden_idx = np.argmin(NLL_tot_arr[NLL_tot_arr != 0])
        int_arr =  [int(digit) for digit in str(int(code_arr[golden_idx]))]
        selected_code = [integer - 1 for integer in int_arr]
        print("GOLDEN INDEX", golden_idx, "SELCECTED CODE", selected_code)
        arr_master = [code_arr, NLL_tot_arr, NLL_had_arr, NLL_lep_arr, W_mass_arr, Top_mass_arr, D_nu_arr, 
                  b_l_pt_arr, b_l_score_arr,
                  b_h_pt_arr, b_h_score_arr, 
                  q1_pt_arr, q1_score_arr, 
                q2_pt_arr, q2_score_arr]
        print("code_arr, NLL_tot_arr, NLL_had_arr, NLL_lep_arr, W_mass_arr, Top_mass_arr, D_nu arr, b_l_pt_arr, b_l_score_arr, b_h_pt_arr, b_h_score_arr, q1_pt_arr, q1_score_arr, q2_pt_arr, q2_score_arr")
        for i in range(len(code_arr)):
            print("")
            print([array[i] for array in arr_master])
   
        ###########################
        # fill algorithm branches #
        ###########################
    
        Algo_Selected_Configuration_Code[0] = int(code_arr[golden_idx]) 

        Algo_Selected_NLL_tot[0] = NLL_tot_arr[golden_idx]
        Algo_Selected_NLL_nu[0] = NLL_lep_arr[golden_idx]
        Algo_Selected_NLL_m[0] = NLL_had_arr[golden_idx]
    
        Algo_b_l_cand_Pt[0] = b_l_pt_arr[golden_idx]
        Algo_b_l_cand_Eta[0] = b_l_eta_arr[b_l_eta_arr != 0][golden_idx]
        Algo_b_l_cand_Phi[0] = b_l_phi_arr[golden_idx]
        Algo_b_l_cand_Mass[0] = b_l_mass_arr[golden_idx]
        Algo_b_l_cand_score[0] = b_l_score_arr[golden_idx]

        Algo_b_h_cand_Pt[0] = b_h_pt_arr[golden_idx]
        Algo_b_h_cand_Eta[0] = b_h_eta_arr[golden_idx]
        Algo_b_h_cand_Phi[0] = b_h_phi_arr[golden_idx]
        Algo_b_h_cand_Mass[0] = b_h_mass_arr[golden_idx]
        Algo_b_h_cand_score[0] = b_h_score_arr[golden_idx]

        Algo_q1_cand_Pt[0] = q1_pt_arr[golden_idx]
        Algo_q1_cand_Eta[0] = q1_eta_arr[golden_idx]
        Algo_q1_cand_Phi[0] = q1_phi_arr[golden_idx]
        Algo_q1_cand_Mass[0] = q1_mass_arr[golden_idx]
        Algo_q1_cand_score[0] = q1_score_arr[golden_idx]

        Algo_q2_cand_Pt[0] = q2_pt_arr[golden_idx]
        Algo_q2_cand_Eta[0] = q2_eta_arr[golden_idx]
        Algo_q2_cand_Phi[0] = q2_phi_arr[golden_idx]
        Algo_q2_cand_Mass[0] = q2_mass_arr[golden_idx]
        Algo_q2_cand_score[0] = q2_score_arr[golden_idx]

        Algo_nu_cand_Pt[0] = nu_pt_arr[golden_idx]
        Algo_nu_cand_Eta[0] = nu_eta_arr[golden_idx]
        Algo_nu_cand_Phi[0] = nu_phi_arr[golden_idx]
        Algo_nu_cand_Mass[0] = nu_mass_arr[golden_idx]
        Algo_nu_cand_D[0] = D_nu_arr[golden_idx]

        Algo_Selected_Top_Mass[0] = Top_mass_arr[golden_idx]
        Algo_Selected_W_Mass[0] = W_mass_arr[golden_idx]

        if MC == 'True' and gen_had_W_mass != 99:
            tot_count +=1

            if sf.equal_or_swapped(correct_configuration, selected_code): # all right
                tot_correct_count += 1
                print("completely correct")
                Algo_is_completely_correct[0] = 1
                Algo_is_hadronic_correct[0] = 1
                Algo_is_leptonic_correct[0] = 1
                Algo_b_h_cand_score_gen_matched[0] = b_l_score_arr[golden_idx]
                Algo_b_l_cand_score_gen_matched[0] = b_h_score_arr[golden_idx]

            elif correct_configuration[0] == selected_code[0]: # lep right, had wrong
                lep_correct_count += 1
                print("leptonic correct")
                Algo_is_leptonic_correct[0] = 1
                Algo_is_hadronic_correct[0] = 0
                Algo_is_completely_correct[0] = 0

            
            elif sf.equal_or_swapped(selected_code[-3:], correct_configuration[-3:]): # had right, lep wrong
                had_correct_count += 1
                print("hadronic correct")
                Algo_is_hadronic_correct[0] = 1
                Algo_is_leptonic_correct[0] = 0
                Algo_is_completely_correct[0] = 0

            else: # all wrong
                print("wrong")
                incorrect_count +=1
                Algo_is_completely_correct[0] = 0
                Algo_is_hadronic_correct[0] = 0
                Algo_is_leptonic_correct[0] = 0

        if MC == 'True' and gen_had_W_mass == 99:
            Algo_is_completely_correct[0] = -1
            Algo_is_hadronic_correct[0] = -1
            Algo_is_leptonic_correct[0] = -1

    tree.Fill()

    Jet_nConstituents.clear()
    Jet_nCH.clear()
    Jet_nNH.clear()
    Jet_nElectrons.clear()
    Jet_nMuons.clear()
    Jet_nPhotons.clear()
    Jet_Area.clear()
    Jet_Eta.clear()
    Jet_Mass.clear()
    Jet_Phi.clear()
    Jet_Pt.clear()
    Jet_Bscore.clear()
    Jet_ChEmEFrac.clear()
    Jet_ChHEFrac.clear()
    Jet_MuEFrac.clear()
    Jet_NeEmEFrac.clear()
    Jet_NeHEFrac.clear()

    Muon_Pt.clear()
    Muon_Eta.clear()
    Muon_Phi.clear()
    Muon_Mass.clear()
    Muon_ecalIso.clear()
    Muon_hcalIso.clear()
    Muon_trackIso.clear()

outfile.cd()
outfile.Write()
outfile.Close()

print(tot_count, "possible to be reconstructed ", tot_correct_count, "completely reconstructed","out of the", tot_count - tot_correct_count, "leftover,", had_correct_count, " correct had side", lep_correct_count, "correct lep side", incorrect_count, "incorrect (either lep or had)")
# Move temporary outfile to permanent outfile at end of script (saves i/o when run on condor)
if condor == 'False':
    print(f'Slimming complete. File location: {out_dir}/{file_name}')
else:
    shutil.move(temp_outfile, perm_outfile)
    print(f'Slimming complete. File location: {out_dir}/{file_name}')
