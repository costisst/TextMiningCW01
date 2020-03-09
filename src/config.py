"""
Read config file
@ param string config
@ returns variables
"""

class Config:

    def __init__(self,config):
        self.path_train = config[1].split(':')[1].strip()
        self.path_dev = config[2].split(':')[1].strip()
        self.path_test = config[3].split(':')[1].strip()
        
        self.model = config[6].split(':')[1].strip()
        self.path_model = config[7].split(':')[1].strip()
        
        self.early_stopping = int(config[10].split(':')[1].strip())
        
        self.epoch = int(config[13].split(':')[1].strip())
        self.lowercase = config[14].split(':')[1].strip()
        
        self.pre_train = config[17].split(':')[1].strip()
        self.path_pre_train = config[18].split(':')[1].strip()
        
        self.word_embedding_dim = int(config[21].split(':')[1].strip())
        self.batch_size = int(config[22].split(':')[1].strip())
        
        self.lr_param = float(config[25].split(':')[1].strip())
        
        self.path_eval_result = config[28].split(':')[1].strip()
        
    def __str__(self):
        return '\n Config variables: \n------------------ \n Path_train: {0} \n Path_dev: {1} \n Path_test: {2} \n Model: {3} \n Path_model: {4} \n Early_stopping: {5} \n Epoch: {6} \n Lowercase: {7} \n Pretrain: {8} \n Path_pre_train: {9} \n Word_Embedding: {10} \n Batch_Size: {11} \n Learning_rate: {12} \n Path_to_eval_results: {13} '.format(self.path_pre_train, self.path_dev, self.path_test, self.model, self.path_model, self.early_stopping, self.epoch, self.lowercase, self.pre_train, self.path_pre_train, self.word_embedding_dim, self.batch_size, self.lr_param, self.path_eval_result)


