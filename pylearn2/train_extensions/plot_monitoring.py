__author__ = 'pabramov'
"""
Training extension for allowing tweeting of monitoring values and plots while an
experiment executes.
"""
# built-in
import os, logging, datetime, socket
import pickle
from functools import wraps
from tempfile import TemporaryFile
# 3rdparty
from pylearn2.train_extensions import TrainExtension
from pylearn2.utils.model_report import Report
# local


logger = logging.getLogger()
logger.setLevel(logging.WARN)


class PlotMonitoring(TrainExtension):
    """
    Extension for tweeting monitored values and plots
    """
    def __init__(self,
                 job_name,
                 channel_names,
                 output_filename):

        """
        Initialization

        :type job_name: str
        :param job_name: job name

        :type channel_names: list/tuple
        :param channel_names: channels to plot on monitoring plot

        :type output_filename: str
        :param output_filename: output plot image filename with file type (e.g '.png')
        """
        self.job_name = job_name
        self.channel_names = channel_names
        self.output_filename = output_filename
        self.report = Report()
        self.host_name = socket.gethostname() + '_' + str(os.getpid())

    @wraps(TrainExtension.on_monitor)
    def on_monitor(self, model, dataset, algorithm):
        """
        Implementation of the on_monitor callback

        :param model:
        :param dataset:
        :param algorithm:
        """
        self.report.plot(model,
                         title=self.job_name,
                         channel_names=self.channel_names,
                         output_filename=self.output_filename)
