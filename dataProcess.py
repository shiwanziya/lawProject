from datasets import load_dataset
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing
import torch

from transformers import AutoModel, AutoTokenizer, AutoConfig

class dataPreProcess(object):
    def __init__(self, train_config):
        self.train_config = train_config
        self.rawdata = load_dataset(
            "json",
            data_dir= train_config.data_path
        )
        self.accusation_label_encoder = preprocessing.LabelEncoder()
        self.relevant_articles_label_encoder = preprocessing.LabelEncoder()
        
    def getOriData(self):
        relevant_articles = []
        accusation = []
        imprisonment = []
        fact = self.rawdata['train']['fact']
        meta = self.rawdata['train']['meta']
        

        for i in tqdm(range(len(meta))):
            accusation.append(meta[i]['accusation'])
            relevant_articles.append(meta[i]['relevant_articles'])

            if meta[i]['term_of_imprisonment']['death_penalty'] == False and meta[i]['term_of_imprisonment'][
                'life_imprisonment'] == False:
                if meta[i]['term_of_imprisonment']['imprisonment'] == 0:
                    imprisonment.append(0)
                elif 0 < meta[i]['term_of_imprisonment']['imprisonment'] <= 6:
                    imprisonment.append(1)
                elif 6 < meta[i]['term_of_imprisonment']['imprisonment'] <= 9:
                    imprisonment.append(2)
                elif 9 < meta[i]['term_of_imprisonment']['imprisonment'] <= 12:
                    imprisonment.append(3)
                elif 12 < meta[i]['term_of_imprisonment']['imprisonment'] <= 24:
                    imprisonment.append(4)
                elif 24 < meta[i]['term_of_imprisonment']['imprisonment'] <= 36:
                    imprisonment.append(5)
                elif 36 < meta[i]['term_of_imprisonment']['imprisonment'] <= 60:
                    imprisonment.append(6)
                elif 60 < meta[i]['term_of_imprisonment']['imprisonment'] <= 84:
                    imprisonment.append(7)
                elif 84 < meta[i]['term_of_imprisonment']['imprisonment'] <= 120:
                    imprisonment.append(8)
                else:
                    imprisonment.append(9)
            else:
                imprisonment.append(10)
        
        accusation_label = self.accusation_label_encoder.fit_transform(accusation)
        relevant_articles_label = self.accusation_label_encoder.fit_transform(relevant_articles)
        
        return {
            'fact':fact,
            'accusation':accusation_label,
            'relevant_articles':relevant_articles_label,
            'imprisonment':imprisonment,
                }
        
    def __len___(self):
        return len(self.fact)
    
class dataSetTorch(Dataset):
    def __init__(self, Tokenizer, data, train_config) -> None:
        self.Tokenizer = Tokenizer
        self.data = data
        self.train_config = train_config
        
    def __len__(self):
        return len(self.data['fact'])
    
    def __getitem__(self, index):
        fact = self.data['fact'][index]
        fact_token = self.Tokenizer.encode_plus(
            fact,
            max_length=self.train_config.max_len, 
            truncation=True, 
            add_special_tokens=True,
            padding='max_length'
        )
        
        input_ids = torch.tensor(fact_token['input_ids'])
        token_type_ids = torch.tensor(fact_token['token_type_ids'])
        attention_mask = torch.tensor(fact_token['attention_mask'])
        
        accusation_label = torch.tensor(self.data['accusation'][index])
        relevant_articles_label = torch.tensor(self.data['relevant_articles'][index])
        imprisonment_label = torch.tensor(self.data['imprisonment'][index], dtype=torch.float32)
        return input_ids, token_type_ids, attention_mask, accusation_label, relevant_articles_label, imprisonment_label
        
        
        
if __name__  == '__main__':
    from config import config
    train_config = config()
    
    Tokenizer = AutoTokenizer.from_pretrained(train_config.model_path)
    
    data_loader = dataPreProcess(train_config)
    data = data_loader.getOriData()
    print(data['relevant_articles'][2])
    
    dataSet = dataSetTorch(Tokenizer, data, train_config)
    
    dataloader_se = DataLoader(dataSet, batch_size=train_config.batch_size, shuffle=True)
    
    print(next(iter(dataloader_se)))
    