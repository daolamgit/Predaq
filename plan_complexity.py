import pydicom as dicom
import os, glob
import numpy as np
import csv
from argparse import ArgumentParser

class Plan(object):
    def __init__(self, file_path, fraction_dose):
        self.File_path          = file_path
        self.metadata               = dicom.read_file( file_path)
        self.fraction_dose          = fraction_dose
        self.total_pair = 60  # number of MLC Leaves

        self.min_leafgap    = .5 #less than this is equi. 0

        self.total_beams = len( self.metadata.BeamSequence)
        self.treatment_site = self.metadata.RTPlanLabel
        self.PatientID  = self.metadata.PatientID

        #treatID
        self.TreatID = self.metadata.BeamSequence[0].TreatmentMachineName

        ###move it up here for those used heavitly
        self.BeamSequence = self.metadata.BeamSequence
        self.FractionGroupSequence  = self.metadata.FractionGroupSequence

        self.MU_order = self.find_MU_order()

        self.BA = np.zeros( self.total_beams) #Average Area
        self.BI = np.zeros( self.total_beams) #Average Irregularity
        self.BM = np.zeros( self.total_beams) #B
        self.UAA = np.zeros(self.total_beams) #
        self.Beam_ID = []



        #composite feature for plan
        self.PA = 0
        self.PI = 0
        self.PM = 0
        self.PMU = 0
        self.PUAA = 0


        self.MU_beam = np.zeros(self.total_beams)

        ###matrices used for simplify the calculation function
        self.AA =  [] # list of np.zeros(total_segments)
        self.AP = [] #list of np.zeros(total_segments)
        self.AP_v = [] #np.zeros(total_segments)
        self.AP_h = [] #np.zeros(total_segments)
        self.nRegs = []  # np.zeros(total_segments)

        self.AI = [] #list of np.zeros(total_segments)
        self.MU_seg = []#list of np.zeros(total_segments)

        #3D array
        self.leftpositions = [] #list of np.empty(shape=(total_pair, total_segments))
        self.rightpositions = [] #list of np.empty(shape=(total_pair, total_segments))
        self.apertures = [] #list of np.empty(shape=(total_pair, total_segments))

        ###Some more feature from Valdes
        self.SAS2 = [] #list of list of portion of open leaf <2mm
        self.SAS5 = [] #list of list of portion of open leaf <5mm
        self.SAS10 = [] #list of list of portion of open leaf <10mm
        self.SAS20 = []  # list of list of portion of open leaf <10mm

        #new features
        self.MSAS2 = np.zeros( self.total_beams)
        self.MSAS5 = np.zeros( self.total_beams)
        self.MSAS10 = np.zeros( self.total_beams)
        self.MSAS20 = np.zeros(self.total_beams)
        self.MFA = np.zeros( self.total_beams) #mean field area
        self.MLO = np.zeros( self.total_beams) #mean leaf opening
        self.MAD = np.zeros( self.total_beams)# the aveage distance between midpoint and the central axis
        self.MUCP = np.zeros( self.total_beams)#average MU per cp
        self.AAJA = np.zeros( self.total_beams)#Average of AA/JA
        self.MaxJ = np.zeros( self.total_beams)#Maximum Jaw Size



        #new new features
        self.MaxSAS2 = np.zeros( self.total_beams)
        self.MaxSAS5 = np.zeros(self.total_beams)
        self.MaxSAS10 = np.zeros(self.total_beams)
        self.MaxSAS20 = np.zeros(self.total_beams)

        self.MLO = np.zeros(self.total_beams)  # mean leaf opening
        self.MLO2 = np.zeros(self.total_beams)  # mean leaf opening
        self.MLO3 = np.zeros(self.total_beams)  # mean leaf opening
        self.MLO4 = np.zeros(self.total_beams)  # mean leaf opening
        self.MLO5 = np.zeros(self.total_beams)  # mean leaf opening

        self.minAP_h = np.zeros(self.total_beams)  # stats of peripheral
        self.maxAP_h = np.zeros(self.total_beams)  # stats of peripheral
        self.minAP_v = np.zeros(self.total_beams)  # stats of peripheral
        self.maxAP_v = np.zeros(self.total_beams)  # stats of peripheral

        self.maxnRegs = np.zeros(self.total_beams)  # max Regions in a beam

        #new features : MCS, EM
        self.LSV    = [] # list of list of segment LSV
        self.AAV    = [] # same
        self.MCS    = np.zeros( self.total_beams) #MCS feature
        self.EM     = np.zeros( self.total_beams)
        self.AA_0   = np.zeros( self.total_beams)

        self.Beam_energy = np.zeros( self.total_beams)
        self.mlc_pattern()
        self.leaf_mat        = np.concatenate( ( np.ones( self.leaftop_idx.size) * self.leafwidth1,\
                                            np.ones( self.leafmid_idx.size) * self.leafwidth2,\
                                            np.ones( self.leafbottom_idx.size) * self.leafwidth1))
        self.jawx = [] #list of tuple (jawx1, jawx2)
        self.jawy = [] #list of tuple (jawy1, jawy2)
        #self.calculate_plan_complexity()
        self.read_beam()
        #self.calculate_plan_complexity( )

        self.calculate_plan_complexity_more_features()

    def find_MU_order(self):
        '''
        Beam sequence and MU sequence not match so
        try mapping here
        :return:
        '''
        res = list(range(  self.total_beams))

        for m in range( self.total_beams):
            beam_number = self.BeamSequence[m].BeamNumber
            for f in range( self.total_beams):
                reference_beam_number = self.FractionGroupSequence[0].ReferencedBeamSequence[f].ReferencedBeamNumber
                if beam_number == reference_beam_number:
                    res[m] = f
                    break
            else:
                print ("Error in matching beam and MU in plan ", self.File_path)

        if not res == range(  self.total_beams):
            print ("Beam and MU order swap! ", self.File_path, "\n")

        return res

    def calculate_plan_complexity_more_features(self):
        for m in range( self.total_beams):
            current_beam = self.BeamSequence[m]
            self.Beam_ID.append( current_beam.BeamName)
            self.Beam_energy[m] = current_beam.ControlPointSequence[0].NominalBeamEnergy

            if current_beam.NumberOfControlPoints <=1:
                continue
            total_segments = current_beam.NumberOfControlPoints

            # some initialization for each beam
            self.AA.append( np.zeros(total_segments))
            self.AP.append(np.zeros(total_segments))
            self.AI.append(np.zeros(total_segments))

            self.AP_h.append(np.zeros(total_segments))
            self.AP_v.append(np.zeros(total_segments))

            #for 3D array
            self.leftpositions.append( np.empty(shape=(self.total_pair, total_segments)))
            self.rightpositions.append(np.empty(shape=(self.total_pair, total_segments)))
            self.apertures.append( np.empty(shape=(self.total_pair, total_segments)))

            self.MU_beam[m] = self.FractionGroupSequence[0].ReferencedBeamSequence[ self.MU_order[m]].BeamMeterset
            # self.MU_beam[m] = self.FractionGroupSequence[0].ReferencedBeamSequence[m].BeamMeterset

            self.SAS2.append( np.zeros(total_segments))
            self.SAS5.append(np.zeros(total_segments))
            self.SAS10.append(np.zeros(total_segments))
            self.SAS20.append(np.zeros(total_segments))
            self.MU_seg.append(np.zeros(total_segments))

            #for lsv, aav
            self.LSV.append( np.zeros(total_segments))
            self.AAV.append( np.zeros( total_segments))

            #number of regions in a segment
            self.nRegs.append( np.zeros( total_segments))

            for k in range( total_segments):
                self.compute_segment_more_features(m , k)


            self.BA[m] = sum( self.MU_seg[m] * self.AA[m]) / self.MU_beam[m]
            self.BI[m] = sum( self.MU_seg[m] * self.AI[m]) / self.MU_beam[m]

            aperture_motion = np.any( self.apertures[m], axis = 1)
            x1_motion = np.mean( self.leftpositions[m], axis = 1)
            x2_motion = np.mean(self.rightpositions[m], axis= 1)
            (no_regs, uppers, lowers) =self.find_upper_lower_indices(aperture_motion, x1_motion, x2_motion)
            UAA_regions     = np.zeros(shape=( no_regs))

            for i in range( no_regs):
                min_left1 = np.min( self.leftpositions[m], axis = 1)
                min_left = min_left1[uppers[i]: lowers[i] + 1]
                max_right1 = np.max( self.rightpositions[m], axis = 1)
                max_right = max_right1[uppers[i]:lowers[i] + 1]
                dist = np.abs( max_right - min_left)
                dist_UAA = np.zeros( shape=( self.total_pair))
                dist_UAA[ uppers[i]: lowers[i]+1] = dist
                UAA_regions[i] = np.sum( dist_UAA * self.leaf_mat)

            self.UAA[m] = sum( UAA_regions)
            self.BM[m]  = 1 - sum( self.MU_seg[m] * self.AA[m]) /(self.MU_beam[m] * self.UAA[m])

            ###new features
            self.MFA[m]     = np.mean( self.AA[m])
            self.MSAS2[m]   = np.mean( self.SAS2[m])
            self.MSAS5[m]   = np.mean( self.SAS5[m])
            self.MSAS10[m]  = np.mean( self.SAS10[m])
            self.MLO[m]     = np.mean( self.apertures[m][self.apertures[m] > 0])
            self.MAD[m]     = self.find_MAD(m)
            self.MUCP[m]    = np.mean(self.MU_seg[m])
            self.MaxJ[m]       = self.find_MaxJ(m)
            self.AAJA[m]       = self.find_AAJA(m)

            ###new new features
            self.MSAS20[m]  = np.mean( self.SAS20[m])

            self.MaxSAS2[m]   = np.max( self.SAS2[m])
            self.MaxSAS5[m]   = np.max( self.SAS5[m])
            self.MaxSAS10[m]  = np.max( self.SAS10[m])
            self.MaxSAS20[m] = np.max( self.SAS20[m])

            #higher central moment of leaf gap MLO. MLO is 1st moment
            LO              = self.apertures[m][self.apertures[m] > 0] #leaf opening

            ##not central moment cal
            MLO          =  0 #np.mean( LO)
            self.MLO2[m] = np.mean( ( LO - MLO)**2)
            self.MLO3[m] = np.mean((LO - MLO) ** 3)
            self.MLO4[m] = np.mean((LO - MLO) ** 4)
            self.MLO5[m] = np.mean((LO - MLO) ** 5)

            #Split AP_h, AP_v, take mea
            self.minAP_h[m] = np.min( self.AP_h[m])
            self.maxAP_h[m] = np.max(self.AP_h[m])
            self.minAP_v[m] = np.min(self.AP_v[m])
            self.maxAP_v[m] = np.max(self.AP_v[m])


            # number of stand aalone leaves
            self.maxnRegs[m]   = np.max( self.nRegs[m])

            # compute MCS
            self.MCS[m] = self.compute_MCS( m)

            # compute EM
            self.EM[m] = self.compute_EM(m)

            # address the problem of AA[m][k]==0
            self.AA_0[m] = np.sum( self.AA[m]==0.0)

    def compute_segment_more_features(self, m, k):
        '''
        Add more features compared to the old one:
        SAS20
        Max SASx
        :param m:
        :param k:
        :return:
        '''
        current_beam = self.BeamSequence[m]
        current_segment = current_beam.ControlPointSequence[k]
        total_segments = current_beam.NumberOfControlPoints

        # Initialization
        leftpositions = np.empty(shape=(self.total_pair, total_segments))
        rightpositions = np.empty(shape=(self.total_pair, total_segments))
        apertures = np.empty(shape=(self.total_pair, total_segments))

        x1 = np.array(current_segment.MLCLeafSet_Convert0)  # left leaves
        x2 = np.array(current_segment.MLCLeafSet_Convert1)  # right leaves

        # check error in coordinate, x1 < x2
        if np.nonzero(x1 > x2)[0].size:
            print ("Some x1 > x2")

        dist_horz = np.abs(x2 - x1)
        dist_horz[dist_horz <= self.min_leafgap] = 0 #enforce close gaps


        #find the active leaves indices
        #i.e. x1 > jawx1, x< jawx2, y < jawy1 and y > jawy2
        active_indices = self.find_active_indices( x1, x2, m, k)

        AA = self.find_AA(dist_horz,m,k)

        ###AP computing
        AP_v = self.find_AP_v(dist_horz)
        AP_h = self.find_AP_h(dist_horz, x1, x2)
        AP = AP_v + AP_h

        # AI
        if AA == 0:
            AI = 1
        else:
            AI = AP ** 2 / (4 * np.pi * AA)

        # MU
        if k == 0:
            self.delta_prev = 0
        if k == current_beam.NumberOfControlPoints -1:
            MU_seg = .5 * self.delta_prev
        else:
            self.delta_curr = current_beam.ControlPointSequence[k+1].CumulativeMetersetWeight -\
                              current_beam.ControlPointSequence[k].CumulativeMetersetWeight
            MU_seg = (self.delta_prev + self.delta_curr )/2
            self.delta_prev = self.delta_curr

        MU_seg = MU_seg * self.MU_beam[m] * current_beam.FinalCumulativeMetersetWeight

        # BM Prep
        idx_zeros = dist_horz == 0
        x1_new = np.copy(x1)
        x1_new[idx_zeros] = 1000
        x2_new = np.copy(x2)
        x2_new[idx_zeros] = -1000
        leftpositions[:, k] = x1_new
        rightpositions[:, k] = x2_new
        apertures[:, k] = dist_horz

        #####Other NEW Features
        SAS2 = self.find_SAS(dist_horz, 2)
        SAS5 = self.find_SAS(dist_horz, 5)
        SAS10 = self.find_SAS(dist_horz, 10)
        SAS20 = self.find_SAS(dist_horz, 20)

        ###Assign back to object class
        self.AA[m][k] = AA
        self.AP[m][k] = AP
        self.AI[m][k] = AI
        self.MU_seg[m][k] = MU_seg
        self.leftpositions[m][:, k] = x1_new
        self.rightpositions[m][:, k] = x2_new
        self.apertures[m][:, k] = dist_horz

        self.SAS2[m][k] = SAS2
        self.SAS5[m][k] = SAS5
        self.SAS10[m][k] = SAS10
        self.SAS20[m][k] = SAS20

        #split Ap, Av
        self.AP_v[m][k] = AP_v
        self.AP_h[m][k] = AP_h

        #number of regions or isolated leaves crit
        dist_vert = np.array(dist_horz > 0, dtype=np.float)  # array indicates leaf open or not
        (no_regs, uppers, lowers) = self.find_upper_lower_indices(dist_vert, x1, x2)
        self.nRegs[m][k]      = no_regs


        ### for compute MSC, which needs LSV and AAV
        #need the test
        (LSV, AAV) = self.find_MCS( active_indices, x1, x2)
        self.LSV[m][k] = LSV
        self.AAV[m][k] = AAV

    def compute_EM(self, m):
        '''
        EM = sum( MU_i x AP_h/AA
        :param m:
        :return:
        '''
        res = 0
        for k in range( len( self.AA[m])):
            if self.AA[m][k] == 0.0:
                # res+= self.MU_seg[m][k]
                continue
            else:
                res += self.MU_seg[m][k] * self.AP_h[m][k]/self.AA[m][k]

        return res/self.MU_beam[m]

    def compute_MCS(self, m):
        #input: LSV, AAV m

        #output: MCS for beam
        res = 0
        for k in range(len(self.LSV[m])):
            res += self.AAV[m][k] * self.LSV[m][k] * self.MU_seg[m][k]/self.MU_beam[m]
        return res

    def find_MCS(self, active_indices, x1, x2):

        def find_pos_max():
            '''
            pos_max is the diff between max(x2_ and min(x1)
            :param x:
            :return:
            '''
            x1_ = x1[active_indices]
            min_x1 = np.min( x1_)
            x2_ = x2[active_indices]
            max_x2 = np.max( x2_)
            pos_max = max_x2 - min_x1
            return pos_max

        def find_lsv( x):
            pos_max = find_pos_max( )

            #numerator
            numer = 0
            for i in range( 0, len( x) -1):
                if active_indices[i] and active_indices[i+1]:
                    numer += (pos_max - (x[i] - x[i+1]))

            return numer / (np.sum( active_indices) * pos_max)

        def find_aav( ):
            x1_ = x1[active_indices]
            max_x1 = np.min( x1_)
            x2_ = x2[active_indices]
            max_x2 = np.max( x2_)

            numer = np.sum( x1_ - x2_)
            return numer / (np.sum( active_indices) * (max_x1 - max_x2))


        lsv1 = find_lsv( x1)
        lsv2 = find_lsv( x2)

        lsv = lsv1 * lsv2

        aav = find_aav()

        return (lsv, aav)

    def find_active_indices(self, x1, x2, m, k):
        a = 1
        '''
        for each segment m,k there are some active jaw defined by their leaft postion x and y
        '''
        jawx1, jawx2 = self.jawx[m][0], self.jawx[m][1]
        jawy1, jawy2 = self.jawy[m][0], self.jawy[m][1]

        #check y:  jawy1<y<jawy2, with a half length shift
        jawy1 += self.mlc_Y /2
        jawy2 += self.mlc_Y /2
        y_indices = (self.mlc_pos >= jawy1) & (self.mlc_pos <= jawy2)

        #check x
        x_indices = (x1 >= jawx1) & (x1 <= jawx2) & (x2 >= jawx1) & (x2 <= jawx2)


        return y_indices & x_indices

    def find_AAJA(self, m):
        AA = self.MFA[m]
        JA = self.find_JA(m) #Max Jaw is Jaw because Jaw doesn't change during beam
        return AA/JA

    def find_JA(self, m):
        jawx = self.jawx[m]
        jawy = self.jawy[m]
        JA = abs(jawx[1] - jawx[0]) * abs(jawy[1] - jawy[0])
        return JA

    def find_MaxJ(self, m):
        '''
        max coordinate
        :param m:
        :return:
        '''
        # jawx1, jawx2 = self.jawx[m]
        # jawy1, jawy2 = self.jawy[m]
        MaxJ = 0
        for i in range(2):
            for j in range(2):
                M = self.jawx[m][i]**2 + self.jawy[m][j] **2
                if M > MaxJ:
                    MaxJ = M
        #
        # return max( [ abs( jawx1), abs( jawx2), abs( jawy1), abs( jawy2)])
        return np.sqrt( MaxJ)

    def find_MAD(self, m):
        '''
        find the NOTaverage, MAX distance from center of each open leaf to the center of MLC
        using apertures, and  as open, and leaf_mat for vert
        leafpos and righpos for horz
        cross check vet and horz for equal cardinal
        :return:
        '''
        center = [0, np.cumsum( self.leaf_mat)[self.total_pair//2 -1]] # or -1

        #find x
        #open leaves
        indices = self.apertures[m] >0 #all the opening indice for entire beam m
        mid_x = (self.leftpositions[m][indices] + self.rightpositions[m][indices]) /2

        #tricky. get the vert pos of open leaf only. Make use of mat broad cast
        #mat broadcast if their inner dim match i.e, (n1,n2) match (n3, n1, n2)
        mid_y = np.cumsum( self.leaf_mat) * indices.transpose().astype(np.float)
        mid_y = mid_y[mid_y > 0] - center[1] #distance from the top leaf


        # dist = np.mean( np.sqrt( (mid_x*mid_x + mid_y*mid_y)))
        dist = np.max(np.sqrt((mid_x * mid_x + mid_y * mid_y)))
        return dist

    def find_SAS(self, dist_horz, thre):
        '''
        For each beam, each segment
        Find the percentage of opening < thre (mm
        :param thre:
        :return:
        '''
        dist_vert = np.array(dist_horz > 0, dtype=np.float) #no Open
        dist_less = np.array( np.logical_and(dist_horz > 0, dist_horz <thre), dtype=np.float)
        if np.sum( dist_vert) == 0:
            return 1
        return np.sum(dist_less) / np.sum(dist_vert)

    def find_SAS_big(self, dist_horz, thre):
        '''
        An SAS only for large leaf

        For each beam, each segment
        Find the percentage of opening < thre (mm
        :param thre:
        :return:
        '''
        # dist_horz = dist_horz[ list(self.leaftop_idx) + list(self.leafbottom_idx)]
        dist_vert = np.array(dist_horz >= 0, dtype=np.float) #no Open
        dist_less = np.array( np.logical_and(dist_horz > 0, dist_horz <=thre), dtype=np.float)
        if np.sum( dist_vert) == 0:
            return 0
        return np.sum(dist_less) / np.sum(dist_vert)

    def find_AA(self, dist_horz, m, k):
        ###AA computing
        AA = np.sum(self.leafwidth1 * dist_horz[self.leaftop_idx]) \
                + np.sum(self.leafwidth2 * dist_horz[self.leafmid_idx]) \
                + np.sum(self.leafwidth1 * dist_horz[self.leafbottom_idx])
        if AA == 0.0:
            print ("AA == 0.0: ", self.File_path, self.Beam_ID[m], k)

        return AA

    def find_AP_v(self, dist_horz):
        dist_vert = np.array(dist_horz > 0, dtype=np.float)  # array indicates leaf open or not
        AP_v = 2 * np.sum(self.leafwidth1 * dist_vert[self.leaftop_idx]) \
                  + 2 * np.sum(self.leafwidth2 * dist_vert[self.leafmid_idx]) \
                  + 2 * np.sum(self.leafwidth1 * dist_vert[self.leafbottom_idx])

        return AP_v

    def find_AP_h(self, dist_horz, x1, x2):
        dist_vert = np.array(dist_horz > 0, dtype=np.float)  # array indicates leaf open or not
        (no_regs, uppers, lowers) = self.find_upper_lower_indices(dist_vert, x1, x2)
        perim_horz = np.zeros(no_regs)
        for i in range(no_regs):
            perim_horz[i] = dist_horz[uppers[i]] + dist_horz[lowers[i]] \
                            + sum(abs(np.diff(x1[uppers[i]: lowers[i] + 1]))) \
                            + sum(abs(np.diff(x2[uppers[i]: lowers[i] + 1])))
        AP_h = sum(perim_horz)
        return AP_h

    def read_beam(self):
        '''
        Read the current beam in m order
        :return:
        '''
        total_beams         = len( self.BeamSequence)

        for m in range( total_beams):
            total_segments = self.BeamSequence[m].NumberOfControlPoints

            self.read_jaw(m)

            for k in range( total_segments):
                if len( self.BeamSequence[m].ControlPointSequence[k].BeamLimitingDevicePositionSequence) > 1:
                    try:
                        self.BeamSequence[m].ControlPointSequence[k].MLCLeafSet =  \
                        self.BeamSequence[m].ControlPointSequence[k].BeamLimitingDevicePositionSequence[2].LeafJawPositions
                    except:
                        print ("MLC read error: ", self.File_path)
                else:
                    self.BeamSequence[m].ControlPointSequence[k].MLCLeafSet  = \
                        self.BeamSequence[m].ControlPointSequence[k].BeamLimitingDevicePositionSequence[0].LeafJawPositions

                self.BeamSequence[m].ControlPointSequence[k].MLCLeafSet_Convert0 = \
                        self.BeamSequence[m].ControlPointSequence[k].MLCLeafSet[0:self.total_pair]

                self.BeamSequence[m].ControlPointSequence[k].MLCLeafSet_Convert1 = \
                        self.BeamSequence[m].ControlPointSequence[k].MLCLeafSet[self.total_pair:]

    def read_jaw(self, m):
        current_beam = self.BeamSequence[m]
        jawx = current_beam.ControlPointSequence[0].BeamLimitingDevicePositionSequence[0].LeafJawPositions
        jawy = current_beam.ControlPointSequence[0].BeamLimitingDevicePositionSequence[1].LeafJawPositions
        self.jawx.append( jawx)
        self.jawy.append( jawy)

    def find_boundary_indices(self, dist_vert):
        '''
        find the upper and lower leaves of an opening holes
        :param dist_vert: boolean
        :return: indices of the leaves on the boundary
        '''
        dist_vert_i = np.insert( dist_vert, [0, dist_vert.size], 0)
        diff_a = np.diff( dist_vert_i)
        diff_a0 = np.insert( diff_a, 0, 0)
        diff_a1 = np.insert( diff_a, diff_a.size, 0)
        res =  abs(dist_vert_i) + abs(diff_a0) + abs(diff_a1) > 1
        return res[1:-1]

    def find_upper_lower_indices_approximate(self, dist_vert):
        '''
        return region description because find_boundary_indices can cause ambigiousity
        :param dist_vert:
        :return: the only problem is if the last leaf is open, but it's rare
        '''
        no_regs = np.sum( np.diff( np.concatenate( (dist_vert, [0])) ==0) == 1)
        idx_0    = (np.concatenate( (dist_vert, [0])) ==0).astype(float)
        no_regs = np.sum( np.diff( idx_0.astype(float)) ==1)
        idx     = (dist_vert != 0).astype(float)
        dsubs   = np.diff( idx)
        upper   = np.where( np.concatenate( ([idx[0]], dsubs)) == 1)[0] #1dim
        lower   = np.where( np.concatenate((dsubs, [-idx[-1]])) ==-1)[0] #1dim

        return (no_regs, upper, lower)

    def find_upper_lower_indices(self, dist_vert, x1, x2):
        '''
        return region description because find_boundary_indices can cause ambigiousity
        :param dist_vert:
        :return: the only problem is if the last leaf is open, but it's rare
        '''
        #no_regs = np.sum( np.diff( np.concatenate( (dist_vert, [0])) ==0) == 1)
        # idx_0    = (np.concatenate( (dist_vert, [0])) ==0).astype(float)
        # no_regs = np.sum( np.diff( idx_0.astype(float)) ==1)
        # idx     = (dist_vert != 0).astype(float)
        # dsubs   = np.diff( idx)
        # upper   = np.where( np.concatenate( ([idx[0]], dsubs)) == 1)[0] #1dim
        # lower   = np.where( np.concatenate((dsubs, [-idx[-1]])) ==-1)[0] #1dim
        #no_regs, upper, lower = 0

        idx_1 = dist_vert
        no_regs = 0
        upper = []
        lower = []
        new_reg = True
        for i in range(len(x1)):
            if idx_1[i]:  # open
                if new_reg:
                    no_regs += 1
                    new_reg = False
                    upper.append(i)
                else:  # extend lower
                    if x1[i] >= x2[i - 1] or x2[i] <= x1[i - 1]:  # non overlap
                        lower.append(i - 1)  # close current rege
                        # open new reg
                        no_regs += 1
                        upper.append(i)

            else:  # close
                if not new_reg:
                    new_reg = True
                    lower.append(i-1)

        #just in case, mismatch:
        if len( lower) < len(upper):
            lower.append( upper[-1])

        return (no_regs, upper, lower)

    def mlc_pattern(self):
        '''
        Define which machine is being used and assigne hd or sd mlc
        :return:
        '''
        a = 1.0
        tr = self.metadata.BeamSequence[0].TreatmentMachineName
        if tr.lower() == 'EdgeTR8'.lower():
            self.leaftop_idx     = np.linspace( 0, 13, 14, dtype=np.int)
            self.leafmid_idx     = np.linspace( 14, 45, 32, dtype=np.int)
            self.leafbottom_idx  = np.linspace( 46, 59, 14, dtype=np.int)
            self.leafwidth1      = a* 5
            self.leafwidth2      = a*2.5
        else:
            self.leaftop_idx     = np.linspace( 0, 9, 10, dtype=np.int)
            self.leafmid_idx     = np.linspace( 10, 49, 40, dtype=np.int)
            self.leafbottom_idx  = np.linspace( 50, 59, 10, dtype=np.int)
            self.leafwidth1      = a* 10
            self.leafwidth2      = a* 5

        mlc_Y = len( self.leaftop_idx ) * self.leafwidth1 + \
                len( self.leafmid_idx) * self.leafwidth2 + \
                len( self.leafbottom_idx) * self.leafwidth1
        self.mlc_Y = mlc_Y

        #mlc array for the purpose of cum sum
        mlc_grain = np.concatenate((np.ones(len(self.leaftop_idx)) * self.leafwidth1,
                              np.ones(len(self.leafmid_idx)) * self.leafwidth2,
                              np.ones(len(self.leafbottom_idx)) * self.leafwidth1))
        mlc_pos = np.cumsum( mlc_grain)
        self.mlc_pos = mlc_pos

    def visual_test(self):
        '''
        Plot the mlc leaves and jaw in
        Input: left position, right position and jaw
        :return:
        '''


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument( '-i', '--input', help = 'path to RP file')
    parser.add_argument( '-o', '--output', help = ' csv file')

    fraction_dose = 180

    args = parser.parse_args()
    rp_path = args.input
    csv_path = args.output

    plan = Plan( rp_path, fraction_dose)

    with open(csv_path, 'w') as f:
        writer = csv.writer(f, delimiter=',', lineterminator='\n')

        row = ['PatientID', 'TreatmentSite', 'RP',
               'Beam_ID', 'BA', 'BI', 'BM', 'UAA',
               'MSAS2', 'MSAS5', 'MSAS10', 'MSAS20',
               'MaxSAS2', 'MaxSAS5', 'MaxSAS10', 'MaxSAS20',
               'MFA', 'MAD', 'MUCP',
               'MLO', 'MLO2', 'MLO3', 'MLO4', 'MLO5',
               'minAP_h', 'maxAP_h', 'minAP_v', 'maxAP_v',
               'maxRegs',
               'AAJA', 'MAXJ', 'Energy_Plan', 'MCS', 'EM'
               ]
        writer.writerow(row)

        for i in range( len(plan.Beam_ID)):
            beam = [plan.PatientID, plan.treatment_site, plan.File_path,
                    plan.Beam_ID[i], plan.BA[i], plan.BI[i], plan.BM[i], plan.UAA[i],
                    plan.MSAS2[i], plan.MSAS5[i], plan.MSAS10[i], plan.MSAS20[i],
                    plan.MaxSAS2[i], plan.MaxSAS5[i], plan.MaxSAS10[i], plan.MaxSAS20[i],

                    plan.MFA[i], plan.MAD[i], plan.MUCP[i],
                    plan.MLO[i], plan.MLO2[i], plan.MLO3[i], plan.MLO4[i], plan.MLO5[i],

                    plan.minAP_h[i], plan.maxAP_h[i], plan.minAP_v[i], plan.maxAP_v[i],

                    plan.maxnRegs[i],
                    plan.AAJA[i], plan.MaxJ[i], plan.Beam_energy[i], plan.MCS[i], plan.EM[i]
                    ]

            writer.writerow(beam)