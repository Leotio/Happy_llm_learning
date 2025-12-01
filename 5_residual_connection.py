'''残差连接的实现'''

# Transformer的第一个子层
# 先对输入x进行层归一化，在计算其注意力分数，把原始输入x加上计算得到的注意力分数，一起送入下一层
h = x + attention.self.attention.forward(attention.self.attention_norm(x)) # attention_norm是层归一化

# Transformer的第二个子层
# h首先进行层归一化，如何加入FFN网络中进行计算，并将结果加上原始输入h的和作为最终的输出
out = h + self.feed_forward.forward(self.fnn_norm(h))