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
import numpy
import matplotlib.pyplot as plt
from TwitterAPI import TwitterAPI
from pylearn2.monitor import Monitor
from pylearn2.train_extensions import TrainExtension
# local


logger = logging.getLogger()
logger.setLevel(logging.WARN)

STATUS_MSG_MAX_LEN = 140


# NOTE: duplicated from bp project: .utils.twitter and maintained there.
# Should be updated manually.
class TwitterManager(object):
    """
    General twitter manager for tweeting status messages
    """

    def __init__(self,
                 credentials_filename,
                 target_user_ids=[]):
        """
        Initialization

        :type credentials_filename: str
        :param credentials_filename: twitter pickled dictionary with credentials

        :type target_user_ids: str-list
        :param target_user_ids: list of twitter user ids that are to be the direct recipients
            of the monitoring tweets. If not supplied tweeted publicly for all followers.
        """
        self.credentials_filename = credentials_filename
        self.target_user_ids = target_user_ids
        self.twitter_api = None
        self.host_name = socket.gethostname() + '_' + str(os.getpid())
        # Open credentials and setup twitter api object
        with open(self.credentials_filename, 'rb') as f:
            creds = pickle.load(f)
            self.twitter_api = TwitterAPI(**creds)

    def tweet_status(self, status_msg):
        """
        Tweet status message

        :type status_msg: str
        :param status_msg: status message to tweet.
        If message length exceeds STATUS_MSG_MAX_LEN characters, message is split and send in multiple tweets.
        """

        continuation_string = ' [...] '
        # Split long message to chunks
        status_msg_chunks = list(self._chunk_string(status_msg, STATUS_MSG_MAX_LEN - 2*len(continuation_string)))

        # Add separating continuation string for better readability
        if len(status_msg_chunks) > 1:
            for k, msg in enumerate(status_msg_chunks):
                if k == 0:
                    status_msg_chunks[k] = msg + continuation_string
                elif k == len(status_msg_chunks) - 1:
                    status_msg_chunks[k] = continuation_string + msg
                else:
                    status_msg_chunks[k] = continuation_string + msg + continuation_string

        # Tweet messages in flipped order for natural reading
        results = [self._tweet_status_msg(msg) for msg in status_msg_chunks[::-1]]
        return results

    def _tweet_status_msg(self, status_msg):
        """
        Tweet message.
        Message length can't exceed STATUS_MSG_MAX_LEN (140)

        :type status_msg: str
        :param status_msg: message to tweet

        :rtype: twitter_api return value
        :return: twitter_api return value
        """
        # Tweet status message
        try:
            r = self.twitter_api.request('statuses/update',
                                         {'status': status_msg,
                                          'user_id': self.target_user_ids})
        except Exception as e:
            logger.exception(e)
            return

        if r.status_code != 200:
            logger.error("Error while trying to tweet message: %s. Response content: %s",
                         status_msg, r.response.content)
        return r

    def tweet_image(self, image_file, header_msg):
        """
        Tweet image file

        :type image_file: str
        :param image_file: image file path

        :type header_msg: str
        :param header_msg: string to be accompanied to the image.
        (length limit is less than 140 characters due to tiny url used for image)

        :type: twitter_api return value
        :return: twitter_api return value
        """
        # Upload image
        data = image_file.read()
        try:
            r = self.twitter_api.request('media/upload', None, {'media': data})
        except Exception as e:
            logger.exception(e)
            return

        if r.status_code != 200:
            logger.error("Error while trying to upload image. Response content: %s", r.response.content)
        else:
            # Post tweet with reference to uploaded image
            media_id = r.json()['media_id']
            try:
                r2 = self.twitter_api.request('statuses/update', {'status': header_msg,
                                                                  'media_ids': media_id,
                                                                  'user_id': self.target_user_ids})
            except Exception as e:
                logger.exception(e)
                return

            if r2.status_code != 200:
                logger.error("Error while trying to post plot image. Response content: %s",
                             r2.response.content)

    @staticmethod
    def _chunk_string(string, length):
        return (string[0+i:length+i] for i in range(0, len(string), length))


class TwitterMonitoring(TrainExtension):
    """
    Extension for tweeting monitored values and plots
    """
    def __init__(self, job_name,
                 channels_info,
                 credentials_filename,
                 target_user_ids=[]):
        """
        Initialization

        :type job_name: str
        :param job_name: job name

        :type channels_info: dict
        :param channels_info: e.g. {name: 'valid_y_nll',
                              epoch_frequency: 3,
                              type: 'plot'}

        :type credentials_filename: str
        :param credentials_filename: twitter pickled dictionary with credentials

        :type target_user_ids: str-list
        :param target_user_ids: list of twitter user ids that are to be the direct recipients
            of the monitoring tweets. If not supplied tweeted publicly for all followers.
        """
        self.job_name = job_name
        self.channels_info = channels_info
        self.tm = TwitterManager(credentials_filename, target_user_ids)
        self.host_name = socket.gethostname() + '_' + str(os.getpid())

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

            # Construct tweet status message
            status_msg = "%(host)s\n" \
                         "%(job_name)s\n" \
                         "E:%(epoch)d, T:%(time)s\n" \
                         "%(name)s: %(val)f\n" \
                         "min: %(min)f [%(min_first_idx)i]" % \
                         {'host': self.host_name,
                          'job_name': self.job_name,
                          'time': str(datetime.timedelta(seconds=ch.time_record[-1])),
                          'epoch': ch.epoch_record[-1],
                          'name': name,
                          'val': float(ch.val_record[-1]),
                          'min': float(min(ch.val_record)),
                          'min_first_idx': numpy.argmin(ch.val_record)}

            if type == 'text':
                self.tm.tweet_status(status_msg)
            elif type == 'plot':
                fig = plt.figure(figsize=(6, 3))
                ax = fig.add_subplot(111)
                plt.hold = True
                ax.plot(ch.epoch_record[1:], ch.val_record[1:])
                min_idx = numpy.argmin(ch.val_record[1:])
                ax.scatter(min_idx, ch.val_record[1:][min_idx], marker='o', c='red', s=60)
                fig.suptitle(name)

                # Save to temp file
                tf = TemporaryFile()
                image_filename = tf.name + '.png'
                fig.savefig(image_filename, bbox_inches='tight')

                # Tweet image
                image_file = open(image_filename, 'rb')
                self.tm.tweet_image(image_file, '')
                # Remote temporary image file
                try:
                    image_file.close()
                    os.remove(image_filename)
                except FileNotFoundError:
                    pass

    def plot_to_file(self, channels, image_filename):
        """
        TODO:

        :param image_filename:
        :return:
        """
        fig = plt.figure(figsize=(6, 3))
        ax = fig.add_subplot(111)
        ax.plot(ch.epoch_record[1:], ch.val_record[1:])
        fig.suptitle(name)

        # Save to temp file
        tf = TemporaryFile()
        image_filename = tf.name + '.png'
        fig.savefig(image_filename, bbox_inches='tight')

    @wraps(TrainExtension.on_monitor)
    def on_monitor(self, model, dataset, algorithm):
        """
        Implementation of the on_monitor callback

        :param model:
        :param dataset:
        :param algorithm:
        """
        monitor = model.monitor

        # Tweet all requested channels
        for ch_info in self.channels_info:
            self.tweet_channel(monitor, ch_info)
