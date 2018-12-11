思路：
    1.模型输入数据：
    
        利用四句七言古诗
        
        利用textrank提取每句诗中的权重较高的词作为本句诗的训练意图【优化点】
        
        利用不超过上三句作为上下文
        
        ^$作为分割诗句的标志
        
        gensim.model 训练的word2vec 作为预训练字向量【优化点】
        
    2.模型结构：
    
        Seq2Seq+BahdanauAttention,作为网络结构
        
        输入数据 添加每个字的拼音【优化点】作为输入数据
        
    3.训练：
    
        正常训练，每次epoch，打乱数据顺序，重新训练
        
    4.测试：
    
        从输出排序中挑选尾字押韵且possibility最高的字进行输出
        
        降低重复出现的字的权重
        
        提高押韵尾字的权重

损失： tensorboard/lr_and_loss.png
       
启动文件：

    entrance.py
    
    预训练：
    
        python entrance.py -p
        
    训练：
    
        python entrance.py -t
        
    测试：
    
        python entrance.py -i


示例：
    输入keywords，4个，通过空格区分怀归 归心 孤云 望远
    
    赋诗：
    
        山不风一春水花
        
        毛荡带修庙胜发
        
        极禅苗焚索渠节
        
        就烛危卷经离华
        
    输入keywords，4个，通过空格区分怀古 今古 江山 望远
    
    赋诗
    
        山不风一春水花
        
        毛荡带修庙胜发
        
        极禅苗焚索烂渠
        
        汝底后难莫何他
        
