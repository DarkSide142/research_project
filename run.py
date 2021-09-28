import csv
import datetime
import importlib
import pprint
import sys
import time


import numpy as np
from htm.algorithms.anomaly_likelihood import AnomalyLikelihood
from htm.bindings.algorithms import Predictor, SpatialPooler, TemporalMemory
from htm.bindings.encoders import ScalarEncoder, ScalarEncoderParameters
from htm.bindings.sdr import SDR, Metrics
from htm.encoders.rdse import RDSE, RDSE_Parameters
from pysad.statistics import MaxMeter, MedianMeter, MinMeter, VarianceMeter
from pysad.transform.postprocessing import (RunningAveragePostprocessor,
                                            RunningMedianPostprocessor)
from sklearn.metrics import mean_squared_error

import anomaly_output

DESCRIPTION = (
    "Starts a HTM model from the model params and pushes each line of input\n"
    "from the data file into the model. Results are written to an output file\n"
    "(default) or plotted dynamically if the --plot option is specified.\n"
)
FILE_NAME = "data"
DATA_DIR = "."
MODEL_PARAMS_DIR = "./model_params"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class Model:
    def __init__(self):
        """
        :param plot: Whether to use matplotlib or not. If false, uses file output.
        :param track: Whether to track or not code execution . If false, it'll be executed in quiet mode.
        :param info: Whether to get or not information & statistics about the state of the HTM.
         If false, additional information won't be appeared.
        """
        print(DESCRIPTION)
        self.plot = False
        self.track = False
        self.info = False
        args = sys.argv[1:]
        if "--plot" in args:
            self.plot = True
        if "--track" in args:
            self.track = True
        if "--info" in args:
            self.info = True

    def createEncoder(self, modelParams):
        # Make the Encoders.  These will convert input data into binary representations.
        rdseParams = modelParams["modelParams"]["enc"]["value"]
        rdseEncoderParams = RDSE_Parameters()
        rdseEncoderParams.size = rdseParams["size"]
        rdseEncoderParams.sparsity = rdseParams["sparsity"]
        rdseEncoderParams.resolution = rdseParams["resolution"]
        rdseEncoderParams.seed = rdseParams["seed"]
        self.rdseEncoder = RDSE(rdseEncoderParams)
        rdseFields = rdseParams["fields"]

        categoryParams = modelParams["modelParams"]["enc"]["category"]
        categoryEncoderParams = ScalarEncoderParameters()
        categoryEncoderParams.activeBits = categoryParams["activeBits"]
        categoryEncoderParams.category = categoryParams["category"]
        categoryEncoderParams.minimum = categoryParams["minimum"]
        categoryEncoderParams.maximum = categoryParams["maximum"]
        self.categoryEncoder = ScalarEncoder(categoryEncoderParams)
        categoryFields = categoryParams["fields"]

        self.encodingWidth = (
            rdseFields * self.rdseEncoder.size
            + categoryFields * self.categoryEncoder.size
        )
        self.enc_info = Metrics([self.encodingWidth], 999999999)

    def createModel(self, modelParams):
        # Make the HTM. SpatialPooler & TemporalMemory & associated tools.
        spParams = modelParams["modelParams"]["sp"]
        self.sp = SpatialPooler(
            inputDimensions=(self.encodingWidth,),
            columnDimensions=(spParams["columnCount"],),
            potentialPct=spParams["potentialPct"],
            potentialRadius=self.encodingWidth,
            globalInhibition=True,
            localAreaDensity=spParams["localAreaDensity"],
            synPermInactiveDec=spParams["synPermInactiveDec"],
            synPermActiveInc=spParams["synPermActiveInc"],
            synPermConnected=spParams["synPermConnected"],
            boostStrength=spParams["boostStrength"],
            seed=spParams["seed"],
            wrapAround=True,
        )
        self.sp_info = Metrics(self.sp.getColumnDimensions(), 999999999)

        tmParams = modelParams["modelParams"]["tm"]
        self.tm = TemporalMemory(
            columnDimensions=(spParams["columnCount"],),
            cellsPerColumn=tmParams["cellsPerColumn"],
            activationThreshold=tmParams["activationThreshold"],
            initialPermanence=tmParams["initialPerm"],
            connectedPermanence=spParams["synPermConnected"],
            minThreshold=tmParams["minThreshold"],
            maxNewSynapseCount=tmParams["newSynapseCount"],
            permanenceIncrement=tmParams["permanenceInc"],
            permanenceDecrement=tmParams["permanenceDec"],
            predictedSegmentDecrement=tmParams["predictedSegmentDecrement"],
            maxSegmentsPerCell=tmParams["maxSegmentsPerCell"],
            maxSynapsesPerSegment=tmParams["maxSynapsesPerSegment"],
            seed=tmParams["seed"],
        )
        self.tm_info = Metrics([self.tm.numberOfCells()], 999999999)

    def createPredictor(self, modelParams):
        # setup likelihood, these settings are used in NAB
        anParams = modelParams["modelParams"]["anomaly"]["likelihood"]
        self.anomaly_history = AnomalyLikelihood(
            learningPeriod=anParams["learningPeriod"],
            estimationSamples=anParams["estimationSamples"],
            reestimationPeriod=anParams["reestimationPeriod"],
            historicWindowSize=anParams["historicWindowSize"],
        )
        self.anomalyThreshold = modelParams["modelParams"]["anomaly"][
            "anomalyThreshold"
        ]
        self.anomalyFrequencyThreshold = modelParams["modelParams"]["anomaly"][
            "anomalyFrequencyThreshold"
        ]
        predParams = modelParams["modelParams"]["predictor"]
        self.steps = predParams["steps"]
        self.predictor = Predictor(steps=[self.steps], alpha=predParams["sdrc_alpha"])
        self.predictor_resolution = 1

    def getModelParamsFromName(self, fileName):
        """
        Given a data name, assumes a matching model params python module exists within
        the model_params directory and attempts to import it.
        :param fileName: data name, used to guess the model params module name.
        :return: HTM Model params dictionary
        """
        importName = "model_params.%s_model_params" % (
            fileName.replace(" ", "_").replace("-", "_")
        )
        print(f"Importing model params from {importName}")
        try:
            self.modelParams = importlib.import_module(importName).MODEL_PARAMS
            print("Model parameters:")
            pprint.pprint(self.modelParams, indent=4)
        except ImportError:
            raise Exception(f"No model params exist for {fileName}!")
        return self.modelParams

    def shifter(self, steps):
        # Shift the predictions so that they are aligned with the input they predict.
        lst = []
        for x in range(steps):
            lst.insert(0, float("nan"))
        return lst

    def getValue(self, parameter, column, source):
        # Get value from the row by name and corresponding column.
        try:
            if parameter == "timestamp":
                value = datetime.datetime.strptime(source[column], DATE_FORMAT)
                return value
            elif parameter == "hookload":
                # read hookload and convert to tonne.
                value = round(float(source[column]) / 1000, 2)
                return value
            elif parameter == "pressure":
                # read pressure and convert to atmosphere.
                value = round(float(source[column]) / 100000, 2)
                return value
            return float(source[column])
        except ValueError:
            # Return zero isn't the best idea. I'm gonna change it further.
            return 0

    def movementClusterisation(self, blockSpeed):
        # Define the categorical feature of direction of drill string movement.
        if blockSpeed == 0:
            # Define the flag of drillstring w/o movement.
            return 0
        elif blockSpeed < 0:
            # Define the flag of runnig drillstring in the hole.
            return 1
        else:
            # Define the flag of pulling drillstring out of the hole.
            return 2

    def rotationClusterisation(self, rpm):
        # Define the categorical feature of drill string rotation.
        if rpm == 0:
            # Define the flag of non-rotating drillstring.
            return 0
        else:
            # Define the flag of rotating drillstring.
            return 1

    def printInfo(self, enc_info, sp_info, tm_info):
        # Print information & statistics about the state of the HTM.
        print("Encoded Input", enc_info)
        print("")
        print("Spatial Pooler Mini-Columns", sp_info)
        print(str(self.sp))
        print("")
        print("Temporal Memory Cells", tm_info)
        print(str(self.tm))
        print("")

    def runIoThroughNupic(self, inputData, fileName):
        """
        Handles looping over the input data and passing each row into the given model
        object, as well as extracting the result object and passing it into an output
        handler.
        :param inputData: file path to input data CSV
        :param fileName: data name, used for output handler naming
        """
        predicted = []

        parameters = {
            "timestamp": 0,
            "hookload": 1,
            "slipStatus": 2,
            "drillingStatus": 3,
            "blockSpeed": 4,
            "blockHeight": 5,
            "rpm": 6,
            "pressure": 7,
            "ecd": 8,
            "bitDepth": 9,
            "holeDepth": 10,
        }

        # Init running average for smoothing of hookload.
        hookloadMA = RunningAveragePostprocessor(window_size=3)
        # Init running average for smoothing of block speed.
        blockSpeedMA = RunningAveragePostprocessor(window_size=15)
        # Init running median for eliminating noise in the streaming data.
        drillStringMovementMed = RunningMedianPostprocessor(window_size=20)
        drillStringRotationMed = RunningMedianPostprocessor(window_size=20)
        drillingStatusMed = RunningMedianPostprocessor(window_size=20)

        # Init statistic trackers of minimum, median and maximum of RMSE.
        self.rmseMin = MinMeter()
        self.rmseMax = MaxMeter()
        self.rmseMedian = MedianMeter()

        # Init rolling metric.
        metric = RunningAveragePostprocessor(window_size=300)

        # Init running statistic of anomaly points frequency.
        anomalyPointsCounter = RunningAveragePostprocessor(
            window_size=1800
        )  # Assume 30 minutes.

        # Aligh predicted values with actual values.
        predicted = self.shifter(self.steps)

        with open(inputData, "r") as inputFile:
            csvReader = csv.reader(inputFile)
            # skip first header row.
            next(csvReader)

            if self.plot:
                output = anomaly_output.PlotOutput(fileName)
            else:
                output = anomaly_output.FileOutput(fileName)

            counter = 0
            start = time.time()

            for row in csvReader:
                counter += 1
                if self.track:
                    if counter % 1000 == 0:
                        end = time.time()
                        print(
                            f"Read {counter} records...",
                            int(1000 / (end - start)),
                            "points/sec",
                            end="\r",
                        )
                        start = time.time()

                # Read data from row
                for parameter, column in parameters.items():
                    globals()[parameter] = self.getValue(parameter, column, row)

                if bitDepth < holeDepth * 0.15:
                    # It is required to prevent additional computations near surface
                    # (for example during BHA or tripping).
                    anomalyLikelihood = 0
                    continue

                if slipStatus == 1 and pressure < 4:
                    # It is required to prevent additional computations during connection.
                    anomalyLikelihood = 0
                    continue

                # Smooth hookload.
                smoothedHookload = hookloadMA.fit_transform_partial(hookload)
                # Smooth block speed.
                smoothedBlockSpeed = blockSpeedMA.fit_transform_partial(blockSpeed)
                # Make categorical feature of drillstring movement.
                drillStringMovement = self.movementClusterisation(smoothedBlockSpeed)
                # Make categorical feature of drillstring rotation.
                drillStringRotation = self.rotationClusterisation(rpm)

                # Calculate median of drillStringMovement and drillStringRotation.
                drillStringMovementMedian = int(
                    drillStringMovementMed.fit_transform_partial(drillStringMovement)
                )
                drillStringRotationMedian = int(
                    drillStringRotationMed.fit_transform_partial(drillStringRotation)
                )
                drillingStatusMedian = drillingStatusMed.fit_transform_partial(
                    drillingStatus
                )

                # Call the encoders to create bit representations for each value.  These are SDR objects.
                hookloadBits = self.rdseEncoder.encode(smoothedHookload)
                feature_1Bits = self.rdseEncoder.encode(drillingStatusMedian)
                feature_2Bits = self.categoryEncoder.encode(drillStringMovementMedian)
                feature_3Bits = self.categoryEncoder.encode(drillStringRotationMedian)
                # feature_4Bits = self.scalarEncoder.encode(ecd)

                # Concatenate all these encodings into one large encoding for Spatial Pooling.
                encoding = SDR(self.encodingWidth).concatenate(
                    [hookloadBits, feature_1Bits, feature_2Bits, feature_3Bits]
                )

                # Create an SDR to represent active columns, This will be populated by the
                # compute method below. It must have the same dimensions as the Spatial Pooler.
                activeColumns = SDR(self.sp.getColumnDimensions())

                # Execute Spatial Pooling algorithm over input space.
                self.sp.compute(encoding, True, activeColumns)

                # Execute Temporal Memory algorithm over active mini-columns.
                self.tm.compute(activeColumns, learn=True)

                # Predict what will happen, and then train the predictor based on what just happened.
                pdf = self.predictor.infer(self.tm.getActiveCells())
                if pdf[self.steps]:
                    predicted.append(
                        np.argmax(pdf[self.steps]) * self.predictor_resolution
                    )
                else:
                    predicted.append(float("nan"))
                self.predictor.learn(
                    counter,
                    self.tm.getActiveCells(),
                    int(smoothedHookload / self.predictor_resolution),
                )

                # Save results.
                anomalyLikelihood = self.anomaly_history.anomalyProbability(
                    smoothedHookload, self.tm.anomaly
                )
                logLikelihood = self.anomaly_history.computeLogLikelihood(
                    anomalyLikelihood
                )

                if logLikelihood > self.anomalyThreshold:
                    anomalyFlag = 1
                else:
                    anomalyFlag = 0

                # Define the dynamic of issue during the drilling.
                # Calculate the frequency of anomaly points for 30 minutes.
                # Convert from points/1sec to points/30min.
                anomalyPointsFrequency = (
                    anomalyPointsCounter.fit_transform_partial(anomalyFlag) * 1800
                )

                # Assume that should be less than 5 overpull per 30 minutes.
                if anomalyPointsFrequency > self.anomalyFrequencyThreshold:
                    anomalyRegion = 1
                else:
                    anomalyRegion = 0

                # Calculate RMSE with current predicted and actual value and update rolling metric.
                try:
                    sklearnMetric = mean_squared_error(
                        [smoothedHookload], [predicted[-self.steps - 1]], squared=False
                    )
                except ValueError:
                    sklearnMetric = 0

                rmse = metric.fit_transform_partial(sklearnMetric)
                self.rmseMin.update(rmse)
                self.rmseMax.update(rmse)
                self.rmseMedian.update(rmse)

                if self.info:
                    self.enc_info.addData(encoding)
                    self.sp_info.addData(activeColumns)
                    self.tm_info.addData(self.tm.getActiveCells().flatten())

                if self.plot:
                    output.write(
                        timestamp,
                        smoothedHookload,
                        predicted[-self.steps - 1],
                        anomalyPointsFrequency,
                        anomalyFlag * 10,
                        pressure,
                        blockHeight,
                        drillingStatusMedian,
                        drillStringMovementMedian,
                        drillStringRotationMedian,
                        rmse,
                    )
                else:
                    output.write(
                        timestamp,
                        hookload,
                        predicted[-self.steps - 1],
                        rmse,
                        logLikelihood,
                        anomalyPointsFrequency,
                        anomalyRegion,
                    )

            output.close()

    def runModel(self, fileName):
        """
        Assumes the fileName corresponds to both a like-named model_params file in the
        model_params directory, and that the data exists in a like-named CSV file in
        the current directory.
        :param fileName: Important for finding model params and input CSV file
        :param plot: Plot in matplotlib? Don't use this unless matplotlib is
        installed.
        """
        print(f"Creating model from {fileName}")
        self.createEncoder(self.getModelParamsFromName(fileName))
        self.createModel(self.modelParams)
        self.createPredictor(self.modelParams)
        inputData = "%s/%s.csv" % (DATA_DIR, fileName.replace(" ", "_"))
        self.runIoThroughNupic(inputData, fileName)
        if self.info:
            self.printInfo(self.enc_info, self.sp_info, self.tm_info)
        print(
            f"Predictive Error (RMSE) Min/Median/Max: {self.rmseMin.get()} / {self.rmseMedian.get()} / {self.rmseMax.get()}"
        )


if __name__ == "__main__":
    model = Model()
    model.runModel(FILE_NAME)
