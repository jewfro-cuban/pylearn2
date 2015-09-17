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
import matplotlib.pyplot as plt
from TwitterAPI import TwitterAPI
from pylearn2.monitor import Monitor
from pylearn2.train_extensions import TrainExtension
# local


logger = logging.getLogger()
logger.setLevel(logging.WARN)


class TwitterMonitoring(TrainExtension):
    """
    Extension for tweeting monitored values and plots
    """

    def __init__(self, job_name,
                 credentials_filename,
                 channels_info,
                 target_user_ids=[]):
        """
        Initialization

        :type job_name: str
        :param job_name: job name

        :type credentials_filename: str
        :param credentials_filename: twitter pickled dictionary with credentials

        :type channels_info: dict
        :param channels_info: e.g. {name: 'valid_y_nll',
                              epoch_frequency: 3,
                              type: 'plot'}

        :type target_user_ids: str-list
        :param target_user_ids: list of twitter user ids that are to be the direct recipients
            of the monitoring tweets. If not supplied tweeted publicly for all followers.
        """
        self.job_name = job_name
        self.credentials_filename = credentials_filename
        self.channels_info = channels_info
        self.target_user_ids = target_user_ids
        self.twitter_api = None
        self.host_name = socket.gethostname() + '_' + str(os.getpid())
        # Open credentials and setup twitter api object
        with open(self.credentials_filename, 'rb') as f:
            creds = pickle.load(f)
            self.twitter_api = TwitterAPI(**creds)

    def tweet_channel(self, monitor, ch_info):
        """
        Tweet monitored channel values

        :type monitor: monitor object
        :param monitor: monitor at current training job state

        :type ch_info: dict
        :param ch_info: channel info
        """
        # Get channel info and channel object
        name = ch_info['name']
        epoch_freq = ch_info['epoch_frequency']
        type = ch_info['type']
        ch = monitor.channels[name]

        if ch.epoch_record[-1] % epoch_freq == 0:
            # Tweet status of channel according to tweet frequency

            if type == 'text':
                # Construct tweet status message
                status_msg = "%(host)s\n%(job_name)s\nE:%(epoch)d, T:%(time)s\n%(name)s: %(val)f" % \
                             {'host': self.host_name,
                              'job_name': self.job_name,
                              'time': datetime.datetime.now().strftime("%H:%M:%S"),
                              'epoch': ch.epoch_record[-1],
                              'name': name,
                              'val': float(ch.val_record[-1])}

                if len(status_msg) > 140:
                    logger.warn("Status message, '%s' ,longer than 140 characters: %d", status_msg, len(status_msg))

                # Tweet status message
                r = self.twitter_api.request('statuses/update', {'status': status_msg,
                                                                 'user_id': self.target_user_ids})
                if r.status_code != 200:
                    logger.error("Error while trying to tweet message: %s. Response content: %s",
                                 status_msg, r.response.content)
            elif type == 'plot':
                fig = plt.figure(figsize=(6, 3))
                ax = fig.add_subplot(111)
                ax.plot(ch.epoch_record, ch.val_record)
                fig.suptitle(name)

                # Save to temp file
                tf = TemporaryFile()
                image_filename = tf.name + '.png'
                fig.savefig(image_filename, bbox_inches='tight')

                # Post a tweet with image
                # Upload image
                image_file = open(image_filename, 'rb')
                data = image_file.read()
                r = self.twitter_api.request('media/upload', None, {'media': data,
                                                                    'user_id': '3652159996'})
                if r.status_code != 200:
                    logger.error("Error while trying to upload image. Response content: %s", r.response.content)
                else:
                    # Post tweet with reference to uploaded image
                    media_id = r.json()['media_id']
                    image_status_msg = '%(host)s\n%(job_name)s\nE:%(epoch)d, T:%(time)s\n%(name)s: %(val)f' % \
                                       {'host': self.host_name,
                                        'job_name': self.job_name,
                                        'time': datetime.datetime.now().strftime("%H:%M:%S"),
                                        'epoch': ch.epoch_record[-1],
                                        'name': name,
                                        'val': float(ch.val_record[-1])}
                    r2 = self.twitter_api.request('statuses/update', {'status': image_status_msg,
                                                                      'media_ids': media_id,
                                                                      'user_id': self.target_user_ids})
                    if r2.status_code != 200:
                        logger.error("Error while trying to post plot image. Response content: %s",
                                     r2.response.content)

                    # Remote temporary image file
                    try:
                        image_file.close()
                        os.remove(image_filename)
                    except FileNotFoundError:
                        pass

    @wraps(TrainExtension.on_monitor)
    def on_monitor(self, model, dataset, algorithm):
        """
        Implementation of the on_monitor callback

        :param model:
        :param dataset:
        :param algorithm:
        """
        monitor = Monitor.get_monitor(model)

        # Tweet all requested channels
        for ch_info in self.channels_info:
            self.tweet_channel(monitor, ch_info)
