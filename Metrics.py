
class CountMetrics:
    def __init__(self, predictedFluorCnt, predictedNonFluorCnt, actualFluorCnt, actualNonFluorCnt):
        self.predFluor = predictedFluorCnt
        self.predNonFluor = predictedNonFluorCnt
        self.actualFluor = actualFluorCnt
        self.actualNonFluor = actualNonFluorCnt

        self.checkForNone()

        self.predTotal = self.predFluor + self.predNonFluor
        #self.actualTotal = self.setActualTotal(actualTotalInclAmbig)
        self.actualTotal = self.actualFluor + self.actualNonFluor

        self.predictedTransmission = self.getTransmission(self.predFluor, self.predTotal)
        self.actualTransmission = self.getTransmission(self.actualFluor, self.actualTotal)

        self.fluorKernelDiff = self.predFluor - self.actualFluor
        self.fluorKernelABSDiff = abs(self.predFluor - self.actualFluor)

        self.fluorPerDiff = self.setPerDiff(self.fluorKernelDiff, self.actualFluor, self.predFluor)

        self.nonFluorKernelDiff = self.predNonFluor - self.actualNonFluor
        self.nonFluorKernelABSDiff = abs(self.predNonFluor - self.actualNonFluor) 

        self.nonFluorPerDiff = self.setPerDiff(self.nonFluorKernelDiff, self.actualNonFluor, self.predNonFluor)

        self.totalKernelDiff = self.predTotal - self.actualTotal
        self.totalKernelABSDiff = abs(self.predTotal - self.actualTotal)                      
        
        self.totalPerDiff = self.setPerDiff(self.totalKernelDiff, self.actualTotal, self.predTotal)

                
        self.transmissionDiff = self.predictedTransmission - self.actualTransmission
        self.transmissionABSDiff = abs(self.transmissionDiff)  

    def checkForNone(self):
        if self.predFluor == None:
            self.predFluor = 0
        if self.predNonFluor == None:
            self.predNonFluor = 0
        if self.actualFluor == None:
            self.actualFluor = 0
        if self.actualNonFluor == None:
            self.actualNonFluor = 0


    def setPerDiff(self, kernelDiff, actual, pred):
        try:          
            return kernelDiff / ((actual + pred) / 2)
        except:
            return 1

    #def setActualTotal(self, actualTotalInclAmbig):
    #    if(actualTotalInclAmbig == None):               
    ##        self.actualTotal = self.actualFluor + self.actualNonFluor
     #   else:
     #       self.actualTotal = actualTotalInclAmbig

    def getTransmission(self, numFluor, total):
        """Returns transmission percentage, or 0 if there are no kernels."""
        try:
            return (numFluor / total) * 100
        except:
            return 0





class SampleDiffDataset:
    def __init__(self):
        self.totalFluorKernelDiff = 0           #inAvgFluorKernelMiscount
        self.totalNonFluorKernelDiff = 0        #inAvgNonFluorKernelMiscount
        self.totalKernelDiff = 0                #inAvgTotalKernelMiscount
        self.totalTransmissionDiff = 0          #inAvgTransmissionDiff
        self.totalFluorKernelABSDiff = 0        #inAvgFluorABSDiff
        self.totalNonFluorKernelABSDiff = 0     #inAvgNonFluorABSDiff
        self.totalABSDiff = 0                   #inAvgTotalABSDiff
        self.totalTransmissionABSDiff = 0       #inAvgTransmissionABSDiff

    def __resetValues__(self):
        self.totalFluorKernelDiff = 0           #inAvgFluorKernelMiscount
        self.totalNonFluorKernelDiff = 0        #inAvgNonFluorKernelMiscount
        self.totalKernelDiff = 0                #inAvgTotalKernelMiscount
        self.totalTransmissionDiff = 0          #inAvgTransmissionDiff
        self.totalFluorKernelABSDiff = 0        #inAvgFluorABSDiff
        self.totalNonFluorKernelABSDiff = 0     #inAvgNonFluorABSDiff
        self.totalABSDiff = 0                   #inAvgTotalABSDiff
        self.totalTransmissionABSDiff = 0       #inAvgTransmissionABSDiff

    def __calculateAverages__(self, denominator):
        self.fluorKernelDiffAvg = self.totalFluorKernelDiff / denominator
        self.nonFluorKernelDiffAvg = self.totalNonFluorKernelDiff / denominator
        self.totalKernelDiffAvg = self.totalKernelDiff / denominator
        self.transmissionDiffAvg = self.totalTransmissionDiff / denominator

        self.fluorABSDiffAvg = self.totalFluorKernelABSDiff / denominator
        self.nonFluorKernelABSDiffAvg = self.totalNonFluorKernelABSDiff / denominator
        self.totalKernelABSDiffAvg = self.totalABSDiff / denominator
        self.transmissionABSDiffAvg = self.totalTransmissionABSDiff / denominator


    def __updateSampleDiffsWithCountMetricsObject__(self, metrics):
        self.totalFluorKernelDiff += metrics.fluorKernelDiff                #inAvgFluorKernelMiscount
        self.totalNonFluorKernelDiff += metrics.nonFluorKernelDiff          #inAvgNonFluorKernelMiscount
        self.totalKernelDiff += metrics.totalKernelDiff                     #inAvgTotalKernelMiscount
        self.totalTransmissionDiff += metrics.transmissionDiff              #inAvgTransmissionDiff

        self.totalFluorKernelABSDiff += metrics.fluorKernelABSDiff          #inAvgFluorABSDiff
        self.totalNonFluorKernelABSDiff += metrics.nonFluorKernelABSDiff    #inAvgNonFluorABSDiff
        self.totalABSDiff += metrics.totalKernelABSDiff                     #inAvgTotalABSDiff
        self.totalTransmissionABSDiff += metrics.transmissionABSDiff        #inAvgTransmissionABSDiff
        

    def __createSampleDiffOutputStr__(self):
        return f"{self.fluorKernelDiffAvg},{self.fluorABSDiffAvg},{self.nonFluorKernelDiffAvg}," + \
            f"{self.nonFluorKernelABSDiffAvg},{self.totalKernelDiffAvg},{self.totalKernelABSDiffAvg}," + \
            f"{self.transmissionDiffAvg},{self.transmissionABSDiffAvg}"

        
        