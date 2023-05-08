import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.spaces import Discrete,Dict

# --------------------------------------------------------------------- #
''' CHILDES dataset loader '''

import math
from collections import Counter
from matplotlib.pyplot import plot

from aochildes.params import AOChildesParams
from aochildes.pipeline import Pipeline
# --------------------------------------------------------------------- #

class NLPEnv_v1(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        super().__init__()
        
        # ------------------------------------------------------------------------------------ #
        ''' CHILDES dataset loader '''
        params = AOChildesParams()
        self.pipeline = Pipeline(params)
        AOT = self.pipeline.load_age_ordered_transcripts()
        transcript_size = len(AOT)
        print("There are", transcript_size, "transcripts.")
        babyAOT = AOT[:1639] # Age group one, ages 90-1090 days
        baby_words = [w.lower() for t in babyAOT for w in t.text.split()]
        baby_vocab = set(baby_words)
        vocab_size = len(baby_vocab)
        print("Our baby model will be facing", vocab_size, "unique words.")
        # ------------------------------------------------------------------------------------ #
        
        self.data = babyAOT
        self.n_passages = len(babyAOT)
        self.vocab = baby_vocab
        self.vocab_size = vocab_size
        self.special_tokens = ['UNK', 'CLS', 'EOS', 'SEP', 'MASK']
        self.cls = self.special_tokens[1]
        self.ending_tokens = ['.', '?', '!']
        self.seperating_tokens = [',', ';']
        
        token_space = Discrete(vocab_size + len(self.special_tokens))
        self.action_space = token_space
        self.observation_space = Dict({"clear_context": Discrete(2), "token": token_space})
    
    def tokenize(self, raw):
        return raw.split(' ')
    
    def step(self, action):
        done = False
        info = {}
        # simulate for loop
        self.word_i += 1
        if self.word_i >= len(self.cur_sentence):
            self.sentence_i += 1
            if self.sentence_i >= len(self.cur_passage):
                self.passage_i += 1
                if self.passage_i >= self.n_passages:
                    # End of data
                    done = True
                    return None, 0, done, truncated, info
                else: 
                    # End of passage
                    self.cur_passage = self.data[self.passage_i].sentences
                    self.sentence_i = 0
            # End of sentence
            raw_sentence = self.cur_passage[self.sentence_i]
            print("STARTING WITH::: ", raw_sentence)
            s = self.tokenize(raw_sentence)
            self.cur_sentence = [self.cls] + s
            self.word_i = 0
        # determine new token
        token = self.cur_sentence[self.word_i]
        if token in self.ending_tokens:
            token = self.special_tokens[2] # EOS
        if token in self.seperating_tokens:
            token = self.special_tokens[3] # SEP
        self.token = token
        # determine reward
        # note that action is equivalent to the model's predicted token
        if self.word_i == 0:  # 'CLS' token, no prediction happening
            reward = 0
            '''
        TODO:
        Classification loss with smoother gradient
       
        else:
            reward = - CrossEntropyLoss(pred, truth)
            '''
        elif action == token:
            reward = 1
        else:
            reward = 0
        
        state = {'clear_cache': (self.word_i == 0), 'token': token}
        return state, reward, done, info
        # return state, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        info = {}
        
        self.passage_i = 0
        self.sentence_i = 0
        self.word_i = -1
        self.cur_passage = self.data[0].sentences
        s = self.tokenize(self.cur_passage[0])
        self.cur_sentence = [self.cls] + s
        state = {'clear_cache': 1, 'token': None}
        
        return state, 0, False, info
    
    def render(self, mode='human', close=False):
        print(self.token)
    
class NLPEnv_v0(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        super().__init__()
        pass
    def step(self, action):
        state = 1    
        reward = -1            
        terminated = True
        truncated = False
        info = {}
        return state, reward, terminated, truncated, info
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        state = 0
        info = {}
        return state,info
    def render(self, mode='human', close=False):
        pass