class gene_vocab():
    def __init__(self, first_valid_token_id=10, n_gram:int=6, predefined_tokens:list=[]):
        self.atcg_dict = {'A': 1,'G': 2,'C': 3,'T': 4,'N': 0}
        self.corpus_pad = "p"
        self.labels_pad = "p"

        self.first_valid_token_id = first_valid_token_id
        self.n_gram = n_gram
        self.predefined_tokens = predefined_tokens

        self.word_dict_alphabet = self.get_word_dict_for_n_gram_alphabet()
        self.word_dict_number = self.get_word_dict_for_n_gram_number()

        self.vocab_size = len(self.word_dict_alphabet) + self.first_valid_token_id
        self.last_valid_token_id =  self.vocab_size-1  # 损失要求标签从零开始

        self.cls_token_id = 4  # 起始token
        self.sep_token_id = 3  # 中止token
        self.mask_token_id = 2 # 掩膜token
        self.unknown_id = 1    # 除字典外的token 
        self.pad_id = -1       # input中表示用于填补的id, label中代表未被选中(未被掩膜)的位置，不参与计算损失


    def get_word_dict_for_n_gram_number(self,alphabet:list=[0, 1, 2, 3, 4]):
        
        """
        功能: 产生一个n-gram映射字典, 由前往后为n=1、2..的映射堆叠  
        返回:                 N, A, G, C, T 、         AN, GN, CN, TN, AA, GA, CA, ...     ANN, GNN, ... 
                key    =  0, 1, 2, 3, 4 、            10, 20, 30, 40, 11, 21, 31, ... 、      100, 200, ...  
                value = 10, 11, 12, 13, 14、 15, 16, 17, 18, 19, 20,... 、 
        """

        word_dict = {}

        if self.predefined_tokens is not None and len(self.predefined_tokens) > 0:
            for token in self.predefined_tokens:
                word_dict[token] = len(word_dict)

        word_set = []
        previous_layer_word_set = []
        add_word_set = set()
        self.first_valid_token_id = max(self.first_valid_token_id, len(self.predefined_tokens))
        for ii in range(self.n_gram):
            for word in alphabet:     # 0, 1, 2, 3, 4
                if ii == 0:  
                    word_set.append(word) 
                    add_word_set.add(word)
                    word_dict[word] = self.first_valid_token_id + len(word_dict)
                else:
                    for add_word in previous_layer_word_set:
                        if len(str(add_word)) == ii:
                            new_word = add_word * 10 + word  # 0, 10, 11, 12, 13, 14, 20, 21, ... 、 100, 101, ... 
                            if new_word in add_word_set: # 跳过0开头的
                                continue
                            word_set.append(new_word)  # 10, 11, 12, 13, 14, 20, 21, ... 、 100, 101, ... 
                            add_word_set.add(new_word)
                            word_dict[new_word] = self.first_valid_token_id + len(word_dict)
            previous_layer_word_set = word_set
            word_set = []
        return word_dict



    def get_word_dict_for_n_gram_alphabet(self,alphabet:list=['N', 'A', 'G', 'C', 'T']):
        
        """
        功能: 产生一个n-gram映射字典, 由前往后为n=1、2..的映射堆叠  
        返回: key    =  N, A, G, C, T 、         AN, GN, CN, TN, AA, GA, CA, ...     ANN, GNN, ...          
                value = 10, 11, 12, 13, 14、 15, 16, 17, 18, 19, 20,... 、 
        """

        word_dict = {}

        if self.predefined_tokens is not None and len(self.predefined_tokens) > 0:
            for token in self.predefined_tokens:
                word_dict[token] = len(word_dict)

        word_set = []
        previous_layer_word_set = []
        add_word_set = set()
        self.first_valid_token_id = max(self.first_valid_token_id, len(self.predefined_tokens))
        for ii in range(self.n_gram):
            for word in alphabet:
                if ii == 0:
                    word_set.append(word)
                    add_word_set.add(word)
                    word_dict[word] = self.first_valid_token_id + len(word_dict)
                else:
                    for add_word in previous_layer_word_set:
                        if len(str(add_word)) == ii:

                            if str(add_word).startswith('N'):
                                continue

                            new_word = add_word + '' + word

                            word_set.append(new_word)
                            add_word_set.add(new_word)
                            word_dict[new_word] = self.first_valid_token_id + len(word_dict)

                            # print(word, add_word, new_word, word_dict.get(new_word, 0))

            previous_layer_word_set = word_set
            word_set = []
        return word_dict

vocab = gene_vocab(n_gram = 3)
word_dict_alphabet = vocab.word_dict_alphabet

