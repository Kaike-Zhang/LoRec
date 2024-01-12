import torch
import torch.nn as nn

class NeuralTree(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NeuralTree, self).__init__()
        
        # 决策节点
        self.decision1 = nn.Linear(input_dim, 1)
        self.decision2 = nn.Linear(input_dim, 1)
        self.decision3 = nn.Linear(input_dim, 1)
        self.decision4 = nn.Linear(input_dim, 1)
        
        # 叶节点
        self.leaf1 = nn.Linear(input_dim, output_dim)
        self.leaf2 = nn.Linear(input_dim, output_dim)
        self.leaf3 = nn.Linear(input_dim, output_dim)
        self.leaf4 = nn.Linear(input_dim, output_dim)
        self.leaf5 = nn.Linear(input_dim, output_dim)
        self.leaf6 = nn.Linear(input_dim, output_dim)
        self.leaf7 = nn.Linear(input_dim, output_dim)
        self.leaf8 = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        # 第一层决策
        decision1_output = torch.sigmoid(self.decision1(x))
        
        # 第二层决策
        decision2_output = torch.sigmoid(self.decision2(x))
        decision3_output = torch.sigmoid(self.decision3(x))
        
        # 第三层决策
        decision4_output = torch.sigmoid(self.decision4(x))
        
        # 根据决策结果计算每个叶节点的权重
        leaf1_weight = decision1_output * decision2_output * decision4_output
        leaf2_weight = decision1_output * decision2_output * (1 - decision4_output)
        leaf3_weight = decision1_output * (1 - decision2_output) * decision3_output
        leaf4_weight = decision1_output * (1 - decision2_output) * (1 - decision3_output)
        leaf5_weight = (1 - decision1_output) * decision2_output * decision4_output
        leaf6_weight = (1 - decision1_output) * decision2_output * (1 - decision4_output)
        leaf7_weight = (1 - decision1_output) * (1 - decision2_output) * decision3_output
        leaf8_weight = (1 - decision1_output) * (1 - decision2_output) * (1 - decision3_output)
        
        # 计算叶节点输出
        out = (leaf1_weight * self.leaf1(x) +
               leaf2_weight * self.leaf2(x) +
               leaf3_weight * self.leaf3(x) +
               leaf4_weight * self.leaf4(x) +
               leaf5_weight * self.leaf5(x) +
               leaf6_weight * self.leaf6(x) +
               leaf7_weight * self.leaf7(x) +
               leaf8_weight * self.leaf8(x))
        
        return out


# 定义神经随机森林
class NeuralRandomForest(nn.Module):
    def __init__(self, config):
        super(NeuralRandomForest, self).__init__()
        self.trees = nn.ModuleList([NeuralTree(config["hidden_units"]+1, 1) for _ in range(config["n_trees"])])
    
    def forward(self, x, logits):
        x = torch.cat((x, logits.unsqueeze(1)), dim=-1)
        outputs = [torch.sigmoid(tree(x)) for tree in self.trees]
        return torch.mean(torch.stack(outputs), dim=0)
