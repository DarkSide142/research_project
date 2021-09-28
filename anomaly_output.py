#!/usr/bin/env python

import csv
from abc import ABCMeta, abstractmethod
from collections import deque

# Try to import matplotlib, but we don't have to.
try:
    import matplotlib

    matplotlib.use("TKAgg")
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    from matplotlib.dates import DateFormatter, date2num
except ImportError:
    pass

WINDOW = 300
HIGHLIGHT_ALPHA = 0.3
ANOMALY_HIGHLIGHT_COLOR = "red"
ANOMALY_THRESHOLD = 6


class Output(object):

    __metaclass__ = ABCMeta

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def write(
        self,
        timestamp,
        hookload,
        predicted,
        rmse,
        logLikelihood,
        anomalyPointsFrequency,
        anomalyRegion,
    ):
        pass

    @abstractmethod
    def close(self):
        pass


class FileOutput(Output):
    def __init__(self, *args, **kwargs):
        super(FileOutput, self).__init__(*args, **kwargs)
        self.outputFiles = []
        self.outputWriters = []
        self.lineCount = 0
        headerRow = [
            "timestamp",
            "hookload",
            "prediction",
            "rmse",
            "logLikelihood",
            "anomalyPointsFrequency",
            "anomalyRegion",
        ]
        outputFileName = "%s_out.csv" % self.name
        print(f"Preparing to output {self.name} data to {outputFileName}")
        self.outputFile = open(outputFileName, "w")
        self.outputWriter = csv.writer(self.outputFile)
        self.outputWriter.writerow(headerRow)

    def write(
        self,
        timestamp,
        hookload,
        predicted,
        rmse,
        logLikelihood,
        anomalyPointsFrequency,
        anomalyRegion,
    ):
        if timestamp is not None:
            outputRow = [
                timestamp,
                hookload,
                predicted,
                rmse,
                logLikelihood,
                anomalyPointsFrequency,
                anomalyRegion,
            ]
            self.outputWriter.writerow(outputRow)
            self.lineCount += 1

    def close(self):
        self.outputFile.close()
        print(f"Done. Wrote {self.lineCount} data lines to {self.name}.")


def extractAnomalyIndices(anomalyProbability):
    anomaliesOut = []
    anomalyStart = None
    for i, likelihood in enumerate(anomalyProbability):
        if likelihood >= ANOMALY_THRESHOLD:
            if anomalyStart is None:
                # Mark start of anomaly
                anomalyStart = i
        else:
            if anomalyStart is not None:
                # Mark end of anomaly
                anomaliesOut.append(
                    (anomalyStart, i, ANOMALY_HIGHLIGHT_COLOR, HIGHLIGHT_ALPHA)
                )
                anomalyStart = None

    # Cap it off if we're still in the middle of an anomaly
    if anomalyStart is not None:
        anomaliesOut.append(
            (
                anomalyStart,
                len(anomalyProbability) - 1,
                ANOMALY_HIGHLIGHT_COLOR,
                HIGHLIGHT_ALPHA,
            )
        )

    return anomaliesOut


class PlotOutput(Output):
    def __init__(self, *args, **kwargs):
        super(PlotOutput, self).__init__(*args, **kwargs)
        # Turn matplotlib interactive mode on.
        plt.ion()
        self.dates = []
        self.convertedDates = []
        self.hookload = []
        self.predicted = []
        self.anomalyScore = []
        self.logLikelihood = []
        self.metric = []
        self.pressure = []
        self.blockHeight = []
        self.status = []
        self.drillStringMovement = []
        self.drillStringRotation = []
        self.actualLine = None
        self.predictedLine = None
        self.anomalyScoreLine = None
        self.logLikelihoodLine = None
        self.metricLine = None
        self.pressureLine = None
        self.blockHeightLine = None
        self.statusLine = None
        self.drillStringMovementLine = None
        self.drillStringRotationLine = None
        self.linesInitialized = False
        self._chartHighlights = []
        fig = plt.figure(figsize=(18, 10))
        gs = gridspec.GridSpec(8, 1, height_ratios=[2.5, 1, 1, 1, 1, 0.5, 0.5, 0.5])

        self._mainGraph = fig.add_subplot(gs[0, 0])
        plt.title(self.name)
        plt.ylabel("Hookload")

        self._anomalyGraph = fig.add_subplot(gs[1])
        plt.ylabel("Score")

        self._metricGraph = fig.add_subplot(gs[2])
        plt.ylabel("RMSE")

        self._pressureGraph = fig.add_subplot(gs[3])
        plt.ylabel("Standpipe pressure")

        self._blockGraph = fig.add_subplot(gs[4])
        plt.ylabel("Block height")

        self._statusGraph = fig.add_subplot(gs[5])
        plt.ylabel("Drilling status")

        self._directionGraph = fig.add_subplot(gs[6])
        plt.ylabel("Movement flag")

        self._rotationGraph = fig.add_subplot(gs[7])
        plt.ylabel("Rotation flag")
        plt.xlabel("Date")

        # Maximizes window
        #     mng = plt.get_current_fig_manager()
        #     mng.resize(*mng.window.maxsize())

        plt.tight_layout()

    def initializeLines(self, timestamp):
        print(f"initializing {self.name}...")
        anomalyRange = (0.0, 10.0)
        blockHeightRange = (0.0, 28.0)
        drillStringMovementRange = (-0.5, 2.5)
        drillStringRotationRange = (-0.5, 1.5)
        self.dates = deque([timestamp] * WINDOW, maxlen=WINDOW)
        self.convertedDates = deque(
            [date2num(date) for date in self.dates], maxlen=WINDOW
        )
        self.hookload = deque([0.0] * WINDOW, maxlen=WINDOW)
        self.predicted = deque([0.0] * WINDOW, maxlen=WINDOW)
        self.anomalyScore = deque([0.0] * WINDOW, maxlen=WINDOW)
        self.logLikelihood = deque([0.0] * WINDOW, maxlen=WINDOW)
        self.metric = deque([0.0] * WINDOW, maxlen=WINDOW)
        self.pressure = deque([0.0] * WINDOW, maxlen=WINDOW)
        self.blockHeight = deque([0.0] * WINDOW, maxlen=WINDOW)
        self.status = deque([0.0] * WINDOW, maxlen=WINDOW)
        self.drillStringMovement = deque([0.0] * WINDOW, maxlen=WINDOW)
        self.drillStringRotation = deque([0.0] * WINDOW, maxlen=WINDOW)

        (actualPlot,) = self._mainGraph.plot(self.dates, self.hookload)
        self.actualLine = actualPlot
        (predictedPlot,) = self._mainGraph.plot(self.dates, self.predicted)
        self.predictedLine = predictedPlot
        self._mainGraph.legend(tuple(["actual", "predicted"]), loc=3)

        (anomalyScorePlot,) = self._anomalyGraph.plot(
            self.dates, self.anomalyScore, "m"
        )
        anomalyScorePlot.axes.set_ylim(anomalyRange)
        self.anomalyScoreLine = anomalyScorePlot

        (logLikelihoodPlot,) = self._anomalyGraph.plot(
            self.dates, self.logLikelihood, "r"
        )
        logLikelihoodPlot.axes.set_ylim(anomalyRange)
        self.logLikelihoodLine = logLikelihoodPlot
        self._anomalyGraph.legend(
            tuple(["anomaly points frequency", "anomaly point flag"]), loc=3
        )

        (metricPlot,) = self._metricGraph.plot(self.dates, self.metric, "b")
        self.metricLine = metricPlot
        self._metricGraph.legend(tuple(["RMSE"]), loc=3)
        # metricPlot.axes.set_ylim(metricRange)

        (pressurePlot,) = self._pressureGraph.plot(self.dates, self.pressure, "r")
        self.pressureLine = pressurePlot
        self._pressureGraph.legend(tuple(["standpipe pressure"]), loc=3)

        (blockHeightPlot,) = self._blockGraph.plot(self.dates, self.blockHeight, "b")
        self.blockHeightLine = blockHeightPlot
        self._blockGraph.legend(tuple(["block height"]), loc=3)
        blockHeightPlot.axes.set_ylim(blockHeightRange)

        (statusPlot,) = self._statusGraph.plot(self.dates, self.status, "b")
        self.statusLine = statusPlot
        self._statusGraph.legend(tuple(["drilling operation status"]), loc=3)

        (drillStringMovementPlot,) = self._directionGraph.plot(
            self.dates, self.drillStringMovement, "b"
        )
        self.drillStringMovementLine = drillStringMovementPlot
        self._directionGraph.legend(tuple(["direction of block movement"]), loc=3)
        drillStringMovementPlot.axes.set_ylim(drillStringMovementRange)

        (drillStringRotationPlot,) = self._rotationGraph.plot(
            self.dates, self.drillStringRotation, "b"
        )
        self.drillStringRotationLine = drillStringRotationPlot
        self._rotationGraph.legend(tuple(["rotation of drill string"]), loc=3)
        drillStringRotationPlot.axes.set_ylim(drillStringRotationRange)

        dateFormatter = DateFormatter("%m/%d %H:%M")
        self._mainGraph.xaxis.set_major_formatter(dateFormatter)
        self._anomalyGraph.xaxis.set_major_formatter(dateFormatter)
        self._metricGraph.xaxis.set_major_formatter(dateFormatter)
        self._pressureGraph.xaxis.set_major_formatter(dateFormatter)
        self._blockGraph.xaxis.set_major_formatter(dateFormatter)
        self._statusGraph.xaxis.set_major_formatter(dateFormatter)
        self._directionGraph.xaxis.set_major_formatter(dateFormatter)
        self._rotationGraph.xaxis.set_major_formatter(dateFormatter)

        self._mainGraph.relim()
        self._mainGraph.autoscale_view(True, True, True)

        self._pressureGraph.relim()
        self._pressureGraph.autoscale_view(True, True, True)

        self._statusGraph.relim()
        self._statusGraph.autoscale_view(True, True, True)

        self.linesInitialized = True

    def highlightChart(self, highlights, chart):
        for highlight in highlights:
            # Each highlight contains [start-index, stop-index, color, alpha]
            self._chartHighlights.append(
                chart.axvspan(
                    self.convertedDates[highlight[0]],
                    self.convertedDates[highlight[1]],
                    color=highlight[2],
                    alpha=highlight[3],
                )
            )

    def write(
        self,
        timestamp,
        hookload,
        predicted,
        anomalyScore,
        logLikelihood,
        pressure,
        blockHeight,
        status,
        drillStringMovement,
        drillStringRotation,
        metric,
    ):

        # We need the first timestamp to initialize the lines at the right X value,
        # so do that check first.
        if not self.linesInitialized:
            self.initializeLines(timestamp)

        self.dates.append(timestamp)
        self.convertedDates.append(date2num(timestamp))
        self.hookload.append(hookload)
        self.predicted.append(predicted)
        self.anomalyScore.append(anomalyScore)
        self.logLikelihood.append(logLikelihood)
        self.metric.append(metric)
        self.pressure.append(pressure)
        self.blockHeight.append(blockHeight)
        self.status.append(status)
        self.drillStringMovement.append(drillStringMovement)
        self.drillStringRotation.append(drillStringRotation)

        # Update main chart data
        self.actualLine.set_xdata(self.convertedDates)
        self.actualLine.set_ydata(self.hookload)
        self.predictedLine.set_xdata(self.convertedDates)
        self.predictedLine.set_ydata(self.predicted)
        # Update anomaly chart data
        self.anomalyScoreLine.set_xdata(self.convertedDates)
        self.anomalyScoreLine.set_ydata(self.anomalyScore)
        self.logLikelihoodLine.set_xdata(self.convertedDates)
        self.logLikelihoodLine.set_ydata(self.logLikelihood)
        # Update metric chart data
        self.metricLine.set_xdata(self.convertedDates)
        self.metricLine.set_ydata(self.metric)
        # Update pressure chart data
        self.pressureLine.set_xdata(self.convertedDates)
        self.pressureLine.set_ydata(self.pressure)
        # Update blockHeight chart data
        self.blockHeightLine.set_xdata(self.convertedDates)
        self.blockHeightLine.set_ydata(self.blockHeight)
        # Update status chart data
        self.statusLine.set_xdata(self.convertedDates)
        self.statusLine.set_ydata(self.status)
        # Update drillStringMovement chart data
        self.drillStringMovementLine.set_xdata(self.convertedDates)
        self.drillStringMovementLine.set_ydata(self.drillStringMovement)
        # Update drillStringRotation chart data
        self.drillStringRotationLine.set_xdata(self.convertedDates)
        self.drillStringRotationLine.set_ydata(self.drillStringRotation)

        # Remove previous highlighted regions
        for poly in self._chartHighlights:
            poly.remove()
        self._chartHighlights = []

        anomalies = extractAnomalyIndices(self.anomalyScore)

        # Highlight anomalies in anomaly chart
        self.highlightChart(anomalies, self._anomalyGraph)

        maxHookload = max(self.hookload)
        self._mainGraph.relim()
        self._mainGraph.axes.set_ylim(0, maxHookload + (maxHookload * 0.02))

        self._mainGraph.relim()
        self._mainGraph.autoscale_view(True, True, False)

        self._anomalyGraph.relim()
        self._anomalyGraph.autoscale_view(True, True, True)

        maxMetric = max(self.metric)
        self._metricGraph.relim()
        self._metricGraph.axes.set_ylim(0, maxMetric + (maxMetric * 0.1) + 0.001)

        self._metricGraph.relim()
        self._metricGraph.autoscale_view(True, True, False)

        maxPressure = max(self.pressure)
        self._pressureGraph.relim()
        self._pressureGraph.axes.set_ylim(0, maxPressure + (maxPressure * 0.1) + 0.1)

        self._pressureGraph.relim()
        self._pressureGraph.autoscale_view(True, True, False)

        self._blockGraph.relim()
        self._blockGraph.autoscale_view(True, True, True)

        statusMaxValue = max(self.status)
        self._statusGraph.relim()
        self._statusGraph.axes.set_ylim(0, statusMaxValue + (statusMaxValue * 0.1))

        self._statusGraph.relim()
        self._statusGraph.autoscale_view(True, True, False)

        self._directionGraph.relim()
        self._directionGraph.autoscale_view(True, True, True)

        self._rotationGraph.relim()
        self._rotationGraph.autoscale_view(True, True, True)

        plt.pause(0.01)

    def close(self):
        plt.ioff()
        plt.show()
