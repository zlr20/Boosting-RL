# Boosting-RL
强化学习不同子模型融合提升效果的预实验

原理报告见`分治融合预实验.pdf`

algo/ 放RL算法，现在只支持ppo
docs/ 实验文字记录
log/ 训练数据日志
pretrained/ 放了训练好的agent1,agent2和融合后的策略ppo_meta.pt

env_meta.py 封装了训练meta网络需要的gym环境
train_agent.py 训练单个agent
eval_agent.py 测试单个agent
eval_agent_diff.py 测试agent之间的差异性
train_meta.py 训练meta网络，用来选择agent
eval_meta.py 测试分治融合的性能