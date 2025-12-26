import numpy as np
import tensorflow as tf
import matplotlib

from test_along_with_train import TestManager

matplotlib.use('agg')
from env.env import Environment
# from other_agents.greedy_agent2 import GreedyAgent2
from other_agents.greedy_agent3 import GreedyAgent
# from other_agents.fifo_agent2 import FifoAgent
from other_agents.random_agent import RandomAgent
from other_agents.heft_agent import HeftAgent
from other_agents.ddos_agent import DdosAgent
from actor_agent11 import ActorAgent1
from actor_agent12 import ActorAgent2
from actor_agent13 import ActorAgent3
from param import *
from utils import *

# test.py开头添加
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# create result folder
if not os.path.exists(args.result_folder):
    os.makedirs(args.result_folder)

# tensorflow seeding
tf.set_random_seed(args.seed)

# set up agents
agents = {}

test = TestManager(f"D:")

for scheme in args.test_schemes:
    if scheme == 'learn':
        sess = tf.Session()
        # 创建 actor_agent 对象
        actor_agent1 = ActorAgent1(sess, args.node_input_dim, args.job_input_dim, args.hid_dims,
                                 args.output_dim, args.max_depth, range(0, args.num_mecs + 1))
        actor_agent2 = ActorAgent2(sess, args.node_input_dim, args.job_input_dim, args.hid_dims,
                                 args.output_dim, args.max_depth, range(0, args.num_mecs + 1))
        actor_agent3 = ActorAgent3(sess, args.node_input_dim, args.job_input_dim, args.hid_dims,
                                 args.output_dim, args.max_depth, range(0, args.num_mecs + 1))

        # 加载模型参数
        actor_agent1.load_model1("D:/")
        actor_agent2.load_model2("D:/")
        actor_agent3.load_model3("D:/")

        test.run_test(actor_agent1, actor_agent2, actor_agent3, 16000)
