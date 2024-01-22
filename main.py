import argparse
import math
import sys
import time

from loguru import logger

from entities.server import Server, TA
import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Secure aggregation protocol for federated learning")
parser.add_argument("-u", "--user", type=int, default=10, help="the number of users")
parser.add_argument("-l", "--lr", type=float, default=0.01, help="learning rate")
parser.add_argument("-r", "--round", type=int, default=10, help="training round")
parser.add_argument("-a", "--att", type=int, default=40, help="proportion of malicious clients(%), from 0 to 100")
parser.add_argument("-f", "--function", type=str, default='pafl',
                    help="aggregation method, ['pafl','krum', 'avg', 'foolsgold', 'median', 'trimmed']")

args = parser.parse_args()

logger.remove()
logger.add(sys.stderr, level="DEBUG")
logger.add('./log/{}_{}_{}.log'.format(args.function, args.user, args.att))
user_gradients_sum = {}

ml_server = Server(args.lr)
s = TA(args.user, math.ceil(math.log(args.user, 2)), args.att)
s.init_clients()
s.collect_shares()

for i in range(args.round):
    s.collect_gradients(ml_server.get_model(), function=args.function)
    start = time.time()
    new_model = s.aggregate(function=args.function)
    logger.info(f'Used Time: {int((time.time() - start) * 1000)}ms')

    s.cal_scores()

    ml_server.set_model(new_model)
    ml_server.local_val()
    s.clear_gradients()
