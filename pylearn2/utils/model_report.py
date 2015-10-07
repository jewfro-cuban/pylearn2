__author__ = 'pabramov'

# built-in
import os, logging, hashlib
# 3rdparty
import numpy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pylearn2
from pylearn2.utils import serial
# local


logger = logging.getLogger()


class Report(object):
    """

    """
    def __init__(self):
        """
        """
        self.supported_record_types = ['epoch', 'batch', 'example', 'time']
        self.line_colors = ['r', 'g', 'b', 'm', 'y', 'k', 'c']
        self.line_styles = ['-', '-.', '--', ':']

    def plot(self, model_input, channel_names, fig=None, title='',
             record_type='epoch', figsize=(6, 3), show=False, output_filename=None):

        if isinstance(model_input, str):
            model = serial.load_pickled_gpu_model(model_input)
        elif isinstance(model_input, pylearn2.models.model.Model):
            model = model_input
        else:
            ValueError("model_input should be either path to model or model instance")

        if not hasattr(model, 'monitor'):
            logger.error("Model has no monitor object")
            return

        if record_type not in self.supported_record_types:
            raise ValueError("Not supported record type: %s. Should be one of: %s", str(record_type), str(self.supported_record_types))

        if fig is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
        else:
            plt.figure(fig.number)
            ax = plt.gca()

        plt.hold = True
        for channel_name in channel_names:
            channel = model.monitor.channels[channel_name]
            x_values = getattr(channel, record_type + '_record')
            y_values = numpy.hstack(channel.val_record)
            if len(y_values) < 2:
                continue
            # Skip first value - usually initial outlier
            color, line_style = self.channel_line_spec(channel_name)
            ax.plot(x_values[1:], y_values[1:], color=color, linestyle=line_style, label=channel_name)
            # TODO: support also max value channels
            min_idx = numpy.argmin(y_values[1:]) + 1
            min_val = y_values[min_idx]
            median_val = numpy.median(y_values[1:])
            ax.scatter(min_idx, min_val, marker='o', color=color, s=60)

            xy = (min_idx, min_val)
            xytext = (min_idx + 2, min_val - 0.1*median_val)
            ax.annotate(str(min_val), xy=xy, xytext=xytext,
                        arrowprops=dict(facecolor='black',
                                        arrowstyle="->",
                                        connectionstyle="arc,angleA=0,armA=20,angleB=-90,armB=15,rad=7"))

        plt.hold = False
        ax.legend()
        plt.xlabel(record_type)
        fig.suptitle(title)

        if show:
            plt.show()

        if output_filename is not None:
            fig.savefig(output_filename, bbox_inches='tight')

        return fig

    def channel_line_spec(self, channel_name):
        m = hashlib.md5(channel_name.encode())
        digest = m.hexdigest()

        color_idx = int(digest[:16], 16) % len(self.line_colors)
        line_style_idx = int(digest[16:], 16) % len(self.line_styles)

        color = self.line_colors[color_idx]
        line_style = self.line_styles[line_style_idx]
        return color, line_style



class Compare(object):
    """

    """
    def __init__(self):
        pass


if __name__ == '__main__':

    model_filename = r'c:/temp/SampleModel/model.pkl'
    report = Report(model_filename)
    report.plot(show=True)